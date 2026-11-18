// Copyright 2026 TOPSBOT
// SPDX-License-Identifier: Apache-2.0

// CenterPoint plugin backend for the unified ai_infer_node scheduler
// (mirrors topsbot_mobilenet_cls::MobileNetBackend).
//
// Input contract: the node forwards sensor_msgs/PointCloud2 payloads as raw
// bytes with format == PixelFormat::Float32Points and step == point_step.
// Accepted point layouts (float32, little-endian, padded to point_step):
//   - x, y, z            (point_step = 12 or 16)
//   - x, y, z, intensity (point_step = 16 or 20)
// Pillars are rebuilt per worker from the raw points (single-frame variant),
// so no PillarBatch is ever cast across a message boundary.
//
// Output contract: detections become ObjectMeta entries (BEV-projected rect in
// display_rect + 3D pose/speed as tags) that the node publishes on
// /detections; when render.enabled is set, a JPEG BEV picture is attached as
// CompressedImageMeta for the node to publish on image_topic.
//
// Model paths: config.model_path uses ';' to separate stages:
//   "merged.nb"  or  "pfe.nb;rpn.nb"

#include "topsbot_ai_infer/inference_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <pluginlib/class_list_macros.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <topsbot_centerpoint/centerpoint.hpp>
#include <topsbot_centerpoint/centerpoint_preprocess.hpp>

namespace topsbot_centerpoint
{

using topsbot_ai_infer::IInferenceBackend;
using topsbot_ai_infer::InferenceFrame;
using topsbot_ai_infer::InferenceResult;
using topsbot_ai_infer::ModelConfig;
using topsbot_ai_infer::PixelFormat;

namespace
{

using SteadyClock = std::chrono::steady_clock;

float elapsed_ms(const SteadyClock::time_point & begin, const SteadyClock::time_point & end)
{
  return std::chrono::duration<float, std::milli>(end - begin).count();
}

// nuScenes class names by CenterPoint label index.
const char * class_name(int label)
{
  static constexpr const char * names[] = {
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"};
  if (label < 0 || label >= static_cast<int>(sizeof(names) / sizeof(names[0]))) {
    return "unknown";
  }
  return names[label];
}

cv::Scalar get_class_color(int label)
{
  // BGR format for OpenCV
  switch (label) {
    case 0:  // car
      return cv::Scalar(255, 100, 0);     // Blue
    case 1:  // truck
      return cv::Scalar(255, 150, 50);    // Light Blue
    case 2:  // construction_vehicle
      return cv::Scalar(200, 120, 0);     // Dark Blue
    case 3:  // bus
      return cv::Scalar(255, 200, 100);   // Cyan-ish
    case 4:  // trailer
      return cv::Scalar(180, 100, 0);     // Navy Blue
    case 5:  // barrier
      return cv::Scalar(0, 140, 255);     // Orange
    case 6:  // motorcycle
      return cv::Scalar(255, 255, 0);     // Cyan
    case 7:  // bicycle
      return cv::Scalar(200, 255, 100);   // Light Cyan
    case 8:  // pedestrian
      return cv::Scalar(0, 255, 255);     // Yellow
    case 9:  // traffic_cone
      return cv::Scalar(0, 165, 255);     // Orange-Red
    default: // unknown
      return cv::Scalar(128, 128, 128);   // Gray
  }
}

/// Convert raw PointCloud2 bytes into the nuScenes xyzin float array used by
/// the pillar builder. Returns false when the buffer cannot contain points.
bool points_from_cloud(const InferenceFrame & frame, std::vector<float> * xyzin)
{
  if (!frame.data || frame.data_size == 0U) {
    return false;
  }
  const std::size_t stride =
    (frame.step >= 12) ? static_cast<std::size_t>(frame.step) : 12U;
  const bool has_intensity = stride >= 16U;
  const std::size_t count = frame.data_size / stride;
  if (count == 0U) {
    return false;
  }
  xyzin->resize(count * 5U);
  for (std::size_t i = 0; i < count; ++i) {
    const float * src = reinterpret_cast<const float *>(frame.data + i * stride);
    float * dst = &(*xyzin)[i * 5U];
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
    dst[3] = has_intensity ? src[3] : 0.0F;
    dst[4] = 0.0F;  // time_lag unused by the single-frame pipeline
  }
  return true;
}

/// Axis-aligned bounds of the yaw-rotated box projected onto the BEV plane.
void bev_rect(const centerpoint::Detection & det, float half_range, float scale,
              int max_pixel, float * x0, float * y0, float * x1, float * y1)
{
  const float hl = det.length * 0.5F;
  const float hw = det.width * 0.5F;
  const float c = std::cos(det.yaw);
  const float s = std::sin(det.yaw);
  float min_x = std::numeric_limits<float>::max();
  float max_x = -std::numeric_limits<float>::max();
  float min_y = std::numeric_limits<float>::max();
  float max_y = -std::numeric_limits<float>::max();
  const float corners[4][2] = {
    {-hl, -hw}, {hl, -hw}, {hl, hw}, {-hl, hw}};
  for (const auto & corner : corners) {
    const float wx = det.x + corner[0] * c - corner[1] * s;
    const float wy = det.y + corner[0] * s + corner[1] * c;
    const float px = std::clamp((wx + half_range) * scale, 0.0F, static_cast<float>(max_pixel));
    const float py = std::clamp((half_range - wy) * scale, 0.0F, static_cast<float>(max_pixel));
    min_x = std::min(min_x, px);
    max_x = std::max(max_x, px);
    min_y = std::min(min_y, py);
    max_y = std::max(max_y, py);
  }
  *x0 = min_x;
  *y0 = min_y;
  *x1 = max_x;
  *y1 = max_y;
}

}  // namespace

class CenterPointBackend final : public IInferenceBackend
{
public:
  ~CenterPointBackend() override
  {
    shutdown();
  }

  bool initialize(const ModelConfig & config, std::string & error) override
  {
    shutdown();
    error.clear();
    config_ = config;

    if (config_.backend != "centerpoint") {
      error = "CenterPoint backend name mismatch";
      return false;
    }

    std::vector<std::string> model_paths;
    std::stringstream ss(config_.model_path);
    std::string token;
    while (std::getline(ss, token, ';')) {
      if (!token.empty()) {
        model_paths.push_back(token);
      }
    }

    bev_size_ = std::max(64, config_.render_size);

    if (model_paths.size() == 1U) {
      // Single-graph: one NB holding PFE + ScatterND + RPN.
      merged_model_ = model_paths[0];
      if (!std::filesystem::is_regular_file(merged_model_)) {
        error = "Merged model not found: " + merged_model_;
        return false;
      }
      if (!detector_.InitMerged(merged_model_, static_cast<uint32_t>(config_.device_id))) {
        error = "CenterPoint InitMerged failed on NPU core " + std::to_string(config_.device_id);
        return false;
      }
      detector_.set_score_threshold(config_.confidence_threshold);
      initialized_ = true;
      return true;
    }

    if (model_paths.size() != 2) {
      error = "CenterPoint model_path format: merged.nb | pfe.nb;rpn.nb";
      return false;
    }

    pfe_model_ = model_paths[0];
    rpn_model_ = model_paths[1];

    if (!std::filesystem::is_regular_file(pfe_model_)) {
      error = "PFE model not found: " + pfe_model_;
      return false;
    }
    if (!std::filesystem::is_regular_file(rpn_model_)) {
      error = "RPN model not found: " + rpn_model_;
      return false;
    }

    // CenterPoint::Init owns ta_runtime refcounting internally.
    if (!detector_.Init(pfe_model_, rpn_model_,
                        static_cast<uint32_t>(config_.device_id))) {
      error = "CenterPoint Init failed on NPU core " + std::to_string(config_.device_id);
      return false;
    }
    detector_.set_score_threshold(config_.confidence_threshold);

    initialized_ = true;
    return true;
  }

  InferenceResult infer(const InferenceFrame & frame) override
  {
    InferenceResult result;
    result.backend = name();
    result.model_name = config_.model_name;
    result.device_id = config_.device_id;
    result.frame.source_id = frame.source_id;
    result.frame.frame_number = frame.sequence;
    result.frame.timestamp_ns = frame.timestamp_ns;
    result.frame.tags["backend"] = name();
    result.frame.tags["model"] = config_.model_name;
    result.frame.tags["device_id"] = std::to_string(config_.device_id);

    if (!initialized_) {
      result.error = "CenterPoint backend is not initialized";
      return result;
    }
    if (frame.format != PixelFormat::Float32Points) {
      result.error = "CenterPoint expects PointCloud2 input (input_msg: pointcloud)";
      return result;
    }

    const auto preprocess_begin = SteadyClock::now();
    std::vector<float> xyzin;
    if (!points_from_cloud(frame, &xyzin)) {
      result.error = "empty or malformed point cloud payload";
      return result;
    }
    std::string error;
    if (!centerpoint::BuildPillarsFromPoints(
          xyzin.data(), xyzin.size() / 5U, &pillars_, &error)) {
      result.error = "pillar build failed: " + error;
      return result;
    }
    result.timing.preprocess_ms =
      elapsed_ms(preprocess_begin, SteadyClock::now());

    const auto infer_begin = SteadyClock::now();
    std::vector<centerpoint::Detection> detections;
    centerpoint::Timing timing;
    bool ok = false;
    try {
      ok = detector_.Detect(pillars_, &detections, &timing);
    } catch (const std::exception & exception) {
      result.error = std::string("CenterPoint exception: ") + exception.what();
      return result;
    }
    result.timing.infer_ms = timing.inference_ms();
    result.timing.postprocess_ms = timing.postprocess_ms;
    if (!ok) {
      result.error = "CenterPoint Detect failed";
      return result;
    }
    result.timing.tensorize_ms =
      elapsed_ms(infer_begin, SteadyClock::now()) -
      result.timing.infer_ms - result.timing.postprocess_ms;

    const float half_range = 51.2F;
    const float scale = static_cast<float>(bev_size_) / (2.0F * half_range);

    for (const auto & detection : detections) {
      if (detection.score < config_.confidence_threshold) {
        continue;
      }
      topsbot_ai_infer::ObjectMeta object;
      object.class_id = detection.label;
      object.class_name = class_name(detection.label);
      object.confidence = detection.score;
      object.type = topsbot_ai_infer::ObjectType::Vehicle;

      float x0 = 0.0F;
      float y0 = 0.0F;
      float x1 = 0.0F;
      float y1 = 0.0F;
      bev_rect(detection, half_range, scale, bev_size_, &x0, &y0, &x1, &y1);
      object.display_rect = {x0, y0, x1 - x0, y1 - y0};
      object.detector_bbox = object.display_rect;

      // Full 3D box as a landmark set (BEV-projected corners + center), the
      // node forwards ObjectMeta::user_metas landmarks into TbTarget::points
      // so webviz can draw them; confidence carries the detector score.
      auto landmark = std::make_shared<topsbot_ai_infer::LandmarkMeta>();
      landmark->confidence = detection.score;
      const float hl = detection.length * 0.5F;
      const float hw = detection.width * 0.5F;
      const float c = std::cos(detection.yaw);
      const float s = std::sin(detection.yaw);
      const float plane[4][2] = {
        {-hl, -hw}, {hl, -hw}, {hl, hw}, {-hl, hw}};
      for (const auto & corner : plane) {
        const float wx = detection.x + corner[0] * c - corner[1] * s;
        const float wy = detection.y + corner[0] * s + corner[1] * c;
        landmark->points.push_back(
          {std::clamp((wx + half_range) * scale, 0.0F, static_cast<float>(bev_size_)),
           std::clamp((half_range - wy) * scale, 0.0F, static_cast<float>(bev_size_)),
           detection.score});
      }
      object.user_metas.push_back(std::move(landmark));

      result.frame.objects.push_back(std::move(object));
    }
    result.frame.frame_attributes.push_back(
      attribute("detection_count", std::to_string(result.frame.objects.size())));

    if (config_.render_enabled) {
      auto image = std::make_shared<topsbot_ai_infer::CompressedImageMeta>();
      image->format = "jpeg";
      image->data = render_bev(detections);
      if (!image->data.empty()) {
        result.frame.user_metas.push_back(std::move(image));
      }
    }

    result.success = true;
    return result;
  }

  const char * name() const noexcept override { return "centerpoint"; }

private:
  static topsbot_ai_infer::ClassifierMeta attribute(
    const std::string & key, const std::string & value)
  {
    topsbot_ai_infer::ClassifierMeta meta;
    meta.label = key + ":" + value;
    return meta;
  }

  std::vector<uint8_t> render_bev(const std::vector<centerpoint::Detection> & detections)
  {
    cv::Mat canvas = render_cache_;
    if (canvas.cols != bev_size_ || canvas.rows != bev_size_) {
      canvas = cv::Mat(bev_size_, bev_size_, CV_8UC3);
    }
    canvas.setTo(cv::Scalar(255, 255, 255));

    const float x_min = -51.2F;
    const float y_max = 51.2F;
    const float scale = static_cast<float>(bev_size_) / 102.4F;

    for (std::size_t i = 0; i < pillars_.used; ++i) {
      const int cx = pillars_.coordinates[i * 3 + 2];
      const int cy = pillars_.coordinates[i * 3 + 1];
      const float x_world = -51.2F + (static_cast<float>(cx) + 0.5F) * 0.2F;
      const float y_world = -51.2F + (static_cast<float>(cy) + 0.5F) * 0.2F;
      const int u = static_cast<int>((x_world - x_min) * scale);
      const int v = static_cast<int>((y_max - y_world) * scale);
      if (u >= 0 && u < bev_size_ && v >= 0 && v < bev_size_) {
        cv::circle(canvas, cv::Point(u, v), 1, cv::Scalar(180, 180, 180), -1);
      }
    }

    for (const auto & det : detections) {
      if (det.score < config_.confidence_threshold) {
        continue;
      }
      const cv::Scalar box_color = get_class_color(det.label);
      const cv::Scalar arrow_color(
        static_cast<int>(box_color[0] * 0.6),
        static_cast<int>(box_color[1] * 0.6),
        static_cast<int>(box_color[2] * 0.6));

      const float c = std::cos(det.yaw);
      const float s = std::sin(det.yaw);
      const float dx = det.length * 0.5F;
      const float dy = det.width * 0.5F;
      const float local[4][2] = {{dx, dy}, {dx, -dy}, {-dx, -dy}, {-dx, dy}};
      cv::Point corners[4];
      for (int i = 0; i < 4; ++i) {
        const float wx = det.x + local[i][0] * c - local[i][1] * s;
        const float wy = det.y + local[i][0] * s + local[i][1] * c;
        corners[i] = cv::Point(
          std::clamp(static_cast<int>((wx - x_min) * scale), 0, bev_size_ - 1),
          std::clamp(static_cast<int>((y_max - wy) * scale), 0, bev_size_ - 1));
      }
      for (int i = 0; i < 4; ++i) {
        cv::line(canvas, corners[i], corners[(i + 1) % 4], box_color, 2);
      }
      const cv::Point center(
        std::clamp(static_cast<int>((det.x - x_min) * scale), 0, bev_size_ - 1),
        std::clamp(static_cast<int>((y_max - det.y) * scale), 0, bev_size_ - 1));
      const cv::Point front(
        std::clamp(static_cast<int>((det.x + c * dx - x_min) * scale), 0, bev_size_ - 1),
        std::clamp(static_cast<int>((y_max - det.y - s * dx) * scale), 0, bev_size_ - 1));
      cv::arrowedLine(canvas, center, front, arrow_color, 2, cv::LINE_8, 0, 0.35);
    }

    std::vector<uint8_t> jpeg;
    if (!cv::imencode(".jpg", canvas, jpeg)) {
      return {};
    }
    render_cache_ = canvas;
    return jpeg;
  }

  void shutdown()
  {
    initialized_ = false;
    detector_.Deinit();
  }

  ModelConfig config_;
  std::string pfe_model_;
  std::string rpn_model_;
  std::string merged_model_;
  int bev_size_{512};
  centerpoint::CenterPoint detector_;
  centerpoint::PillarBatch pillars_;
  cv::Mat render_cache_;
  bool initialized_{false};
};

}  // namespace topsbot_centerpoint

PLUGINLIB_EXPORT_CLASS(
  topsbot_centerpoint::CenterPointBackend, topsbot_ai_infer::IInferenceBackend)
