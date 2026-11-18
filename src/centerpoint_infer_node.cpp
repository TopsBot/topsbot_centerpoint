// Copyright 2026 TOPSBOT
// SPDX-License-Identifier: Apache-2.0

// CenterPoint inference/replay node (counterpart of topsbot_mobilenet_cls'
// mobilenet_file_cli). Replaces the hard-coded "input_msg: centerpoint_file"
// path that used to live inside the generic ai_infer_node.
//
// Parameters (unchanged names, config/ai_infer_centerpoint.yaml):
//   centerpoint.pfe_model / rpn_model      [.nb stages]
//   centerpoint.sweeps_manifest            [10-line manifest or one .bin]
//   centerpoint.frames_list                [list of dirs/manifests, loops]
//   centerpoint.single_frame_mode          [treat manifest path as .bin]
//   centerpoint.run_interval_ms            [0 = run once, >0 = loop playback]
//   centerpoint.score_threshold            [publish filter]
//   centerpoint.bev_size                   [square render resolution]
//   centerpoint.frame_id / image_topic / detections topic (detections_topic)
//   centerpoint.core_id                    [legacy single core]
//   centerpoint.device_ids                 [NPU cores; >1 enables worker pool]
//   centerpoint.load_balancer.queue_size   [per-worker queue depth]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <geometry_msgs/msg/point32.hpp>
#include <tb_det_msgs/msg/tb_perception_targets.hpp>
#include <tb_det_msgs/msg/tb_perf.hpp>
#include <tb_det_msgs/msg/tb_point.hpp>
#include <tb_det_msgs/msg/tb_roi.hpp>
#include <tb_det_msgs/msg/tb_target.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <topsbot_centerpoint/centerpoint.hpp>
#include <topsbot_centerpoint/centerpoint_preprocess.hpp>
#include <topsbot_centerpoint/centerpoint_worker.hpp>

namespace
{

template<typename T>
T read_parameter(rclcpp::Node & node, const std::string & name, const T & fallback)
{
  if (!node.has_parameter(name)) {
    node.declare_parameter(name, fallback);
  }
  return node.get_parameter(name).get_value<T>();
}

std::string expand_user_path(const std::string & path)
{
  if (path.empty() || path.front() != '~') {
    return path;
  }
  const char * user_home = std::getenv("HOME");
  if (user_home == nullptr || user_home[0] == '\0') {
    return path;
  }
  if (path.size() == 1U || path[1] == '/') {
    return std::string(user_home) + path.substr(1);
  }
  return path;
}

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

tb_det_msgs::msg::TbPerf make_perf(const std::string & type, const float milliseconds)
{
  tb_det_msgs::msg::TbPerf perf;
  perf.type = type;
  perf.time_ms_duration = static_cast<double>(milliseconds);
  return perf;
}

uint32_t clamp_bev_pixel(const float value, const int max_pixel)
{
  if (!std::isfinite(value) || value <= 0.0F) {
    return 0U;
  }
  if (value >= static_cast<float>(max_pixel)) {
    return static_cast<uint32_t>(max_pixel);
  }
  return static_cast<uint32_t>(value);
}

/// Render snapshot handed to the worker pool via DetectionRequest::context so
/// async results can still draw the point cloud that produced them. It owns
/// ONLY the small render metadata: the heavy decorated tensor is moved into
/// the request and must never be touched here afterwards (use-after-move of
/// PillarBatch left `used` set while the vectors were emptied -> SIGSEGV).
struct FrameSnapshot {
  std::size_t pillar_count{0};
  std::vector<int32_t> coordinates;  // [pillar_count * 3], z/y/x per pillar
  double preprocess_ms{0.0};
};

using FrameSnapshotPtr = std::shared_ptr<const FrameSnapshot>;

}  // namespace

class CenterPointInferNode : public rclcpp::Node
{
public:
  CenterPointInferNode()
  : Node(
      "centerpoint_infer",
      rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true))
  {
    pfe_model_ = expand_user_path(read_parameter<std::string>(*this, "centerpoint.pfe_model", ""));
    rpn_model_ = expand_user_path(read_parameter<std::string>(*this, "centerpoint.rpn_model", ""));
    merged_model_ = expand_user_path(
      read_parameter<std::string>(*this, "centerpoint.merged_model", ""));
    sweeps_manifest_ = expand_user_path(
      read_parameter<std::string>(*this, "centerpoint.sweeps_manifest", ""));
    frames_list_ = expand_user_path(
      read_parameter<std::string>(*this, "centerpoint.frames_list", ""));
    run_interval_ms_ = std::max(0, read_parameter<int>(*this, "centerpoint.run_interval_ms", 0));
    score_threshold_ = static_cast<float>(
      read_parameter<double>(*this, "centerpoint.score_threshold", 0.2));
    bev_size_ = std::max(64, read_parameter<int>(*this, "centerpoint.bev_size", 512));
    frame_id_ = read_parameter<std::string>(*this, "centerpoint.frame_id", "centerpoint");
    image_topic_ = read_parameter<std::string>(*this, "centerpoint.image_topic", "/image_jpeg");
    detections_topic_ = read_parameter<std::string>(*this, "detections_topic", "/detections");
    single_frame_mode_ = read_parameter<bool>(*this, "centerpoint.single_frame_mode", false);

    std::vector<int64_t> device_ids;
    if (has_parameter("centerpoint.device_ids")) {
      device_ids = get_parameter("centerpoint.device_ids").get_value<std::vector<int64_t>>();
    }
    if (device_ids.empty()) {
      device_ids.push_back(read_parameter<int>(*this, "centerpoint.core_id", 0));
    }
    for (const auto id : device_ids) {
      device_ids_.push_back(static_cast<int>(id));
    }
    queue_size_ = static_cast<std::size_t>(
      std::max(1, read_parameter<int>(*this, "centerpoint.load_balancer.queue_size", 2)));

    if (merged_model_.empty() && (pfe_model_.empty() || rpn_model_.empty())) {
      throw std::invalid_argument(
        "centerpoint.merged_model or (centerpoint.pfe_model and centerpoint.rpn_model) are required");
    }

    load_frame_list();

    detections_pub_ =
      create_publisher<tb_det_msgs::msg::TbPerceptionTargets>(detections_topic_, 10);
    image_pub_ =
      create_publisher<sensor_msgs::msg::CompressedImage>(image_topic_, 10);

    centerpoint::CenterPointWorkerPool::Config pool_config;
    pool_config.device_ids = device_ids_;
    pool_config.pfe_model = pfe_model_;
    pool_config.rpn_model = rpn_model_;
    pool_config.merged_model = merged_model_;
    pool_config.max_queue_size = queue_size_;
    pool_config.score_threshold = score_threshold_;
    pool_ = std::make_unique<centerpoint::CenterPointWorkerPool>(pool_config);
    std::string error;
    if (!pool_->initialize(&error)) {
      throw std::runtime_error("centerpoint pool init failed: " + error);
    }
    pool_->set_result_callback(
      std::bind(&CenterPointInferNode::on_detection_result, this, std::placeholders::_1));

    RCLCPP_INFO(
      get_logger(),
      "centerpoint infer ready: frames=%zu single_frame=%d cores=%zu queue=%zu "
      "interval_ms=%d detections=%s image=%s",
      frame_paths_.size(), single_frame_mode_ ? 1 : 0, device_ids_.size(), queue_size_,
      run_interval_ms_, detections_topic_.c_str(), image_topic_.c_str());

    if (run_interval_ms_ == 0) {
      timer_ = create_wall_timer(
        std::chrono::milliseconds(1),
        [this]() {
          timer_->cancel();
          infer_once();
        });
    } else {
      timer_ = create_wall_timer(
        std::chrono::milliseconds(run_interval_ms_),
        std::bind(&CenterPointInferNode::infer_once, this));
    }
  }

private:
  void load_frame_list()
  {
    if (!frames_list_.empty()) {
      std::ifstream ifs(frames_list_);
      if (!ifs) {
        throw std::runtime_error("cannot open frames list: " + frames_list_);
      }
      const std::filesystem::path base_dir =
        std::filesystem::path(frames_list_).parent_path();
      std::string line;
      while (std::getline(ifs, line)) {
        const auto first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos || line[first] == '#') {
          continue;
        }
        std::string frame_path = line.substr(first);
        const auto last = frame_path.find_last_not_of(" \t\r");
        if (last != std::string::npos) {
          frame_path = frame_path.substr(0, last + 1);
        }
        std::filesystem::path sweeps_path;
        if (single_frame_mode_ || frame_path.find(".txt") != std::string::npos ||
            frame_path.find(".bin") != std::string::npos) {
          sweeps_path = frame_path;
        } else if (frame_path.front() == '/') {
          sweeps_path = std::filesystem::path(frame_path) / "sweeps.txt";
        } else {
          sweeps_path = base_dir / frame_path / "sweeps.txt";
        }
        frame_paths_.push_back(sweeps_path.string());
      }
      if (frame_paths_.empty()) {
        throw std::runtime_error("frames list is empty: " + frames_list_);
      }
      RCLCPP_INFO(
        get_logger(), "centerpoint multi-frame mode: %zu frames from %s",
        frame_paths_.size(), frames_list_.c_str());
    } else if (!sweeps_manifest_.empty()) {
      frame_paths_.push_back(sweeps_manifest_);
    } else {
      throw std::invalid_argument(
        "either centerpoint.sweeps_manifest or centerpoint.frames_list must be specified");
    }
  }

  void infer_once()
  {
    const std::size_t index = current_frame_ % frame_paths_.size();
    centerpoint::PillarBatch pillars;
    std::string error;
    const auto preprocess_started = std::chrono::steady_clock::now();
    if (single_frame_mode_) {
      if (!centerpoint::BuildPillarsFromSingleFrame(
            frame_paths_[index], &pillars, &error)) {
        RCLCPP_ERROR(get_logger(), "frame %zu preprocessing failed: %s", index, error.c_str());
        return;
      }
    } else {
      std::vector<centerpoint::SweepSpec> sweeps;
      if (!centerpoint::LoadSweepManifest(frame_paths_[index], &sweeps, &error)) {
        RCLCPP_ERROR(get_logger(), "frame %zu manifest failed: %s", index, error.c_str());
        return;
      }
      if (!centerpoint::BuildPillarsFromNuScenes(sweeps, &pillars, &error)) {
        RCLCPP_ERROR(get_logger(), "frame %zu preprocessing failed: %s", index, error.c_str());
        return;
      }
    }
    auto snapshot = std::make_shared<FrameSnapshot>();
    snapshot->pillar_count = pillars.used;
    snapshot->coordinates.assign(
      pillars.coordinates.begin(),
      pillars.coordinates.begin() + static_cast<std::ptrdiff_t>(pillars.used * 3U));
    snapshot->preprocess_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - preprocess_started).count();

    centerpoint::DetectionRequest request;
    request.pillars = std::move(pillars);
    request.context = snapshot;
    if (!pool_->submit(std::move(request))) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "centerpoint infer: no worker accepted frame %zu (queue full?)", index);
    }
    ++current_frame_;
  }

  void on_detection_result(const centerpoint::DetectionResult & result)
  {
    const auto snapshot = std::static_pointer_cast<const FrameSnapshot>(result.context);
    if (!result.success) {
      RCLCPP_ERROR(get_logger(), "centerpoint inference failed: %s", result.error.c_str());
      return;
    }

    const double preprocess_ms = snapshot ? snapshot->preprocess_ms : 0.0;
    centerpoint::Timing timing = result.timing;
    timing.preprocess_ms = preprocess_ms;

    if (timing.single_graph) {
      const double total_ms =
        timing.preprocess_ms + timing.inference_ms() + timing.postprocess_ms;
      RCLCPP_INFO(
        get_logger(),
        "centerpoint frame %llu: detections=%zu inference=%.1fms postprocess=%.1fms "
        "preprocess=%.1fms total=%.1fms FPS=%.1f core=%d (merged)",
        static_cast<unsigned long long>(result.frame_id), result.detections.size(),
        timing.inference_ms(), timing.postprocess_ms, timing.preprocess_ms, total_ms,
        total_ms > 0.0 ? 1000.0 / total_ms : 0.0, result.device_id);
    } else {
      RCLCPP_INFO(
        get_logger(),
        "centerpoint frame %llu: detections=%zu pfe_ms=%.1f scatter_ms=%.1f rpn_ms=%.1f "
        "postprocess_ms=%.1f preprocess_ms=%.1f core=%d",
        static_cast<unsigned long long>(result.frame_id), result.detections.size(),
        timing.pfe_ms, timing.scatter_ms, timing.rpn_ms, timing.postprocess_ms,
        timing.preprocess_ms, result.device_id);
    }

    publish_bev(result.detections, timing);
    if (snapshot) {
      publish_image(result.detections, *snapshot);
    }
  }

  void publish_bev(
    const std::vector<centerpoint::Detection> & detections, const centerpoint::Timing & timing)
  {
    tb_det_msgs::msg::TbPerceptionTargets message;
    message.header.stamp = get_clock()->now();
    message.header.frame_id = frame_id_;
    message.fps = 0;
    if (timing.single_graph) {
      message.perfs.push_back(make_perf("centerpoint_inference", timing.inference_ms()));
    } else {
      message.perfs.push_back(make_perf("centerpoint_pfe", timing.pfe_ms));
      message.perfs.push_back(make_perf("centerpoint_scatter", timing.scatter_ms));
      message.perfs.push_back(make_perf("centerpoint_rpn", timing.rpn_ms));
    }
    message.perfs.push_back(make_perf("centerpoint_preprocess", timing.preprocess_ms));
    message.perfs.push_back(make_perf("centerpoint_postprocess", timing.postprocess_ms));

    const float half_range = 51.2F;
    const float scale = static_cast<float>(bev_size_) / (2.0F * half_range);
    const int max_pixel = bev_size_;

    for (const auto & detection : detections) {
      if (detection.score < score_threshold_) {
        continue;
      }
      // Axis-aligned bounds of the yaw-rotated box projected onto the BEV.
      const float hl = detection.length * 0.5F;
      const float hw = detection.width * 0.5F;
      const float c = std::cos(detection.yaw);
      const float s = std::sin(detection.yaw);
      float min_x = std::numeric_limits<float>::max();
      float max_x = -std::numeric_limits<float>::max();
      float min_y = std::numeric_limits<float>::max();
      float max_y = -std::numeric_limits<float>::max();
      const float corners[4][2] = {{-hl, -hw}, {hl, -hw}, {hl, hw}, {-hl, hw}};
      for (const auto & corner : corners) {
        const float wx = detection.x + corner[0] * c - corner[1] * s;
        const float wy = detection.y + corner[0] * s + corner[1] * c;
        const float px = std::clamp(
          (wx + half_range) * scale, 0.0F, static_cast<float>(max_pixel));
        const float py = std::clamp(
          (half_range - wy) * scale, 0.0F, static_cast<float>(max_pixel));
        min_x = std::min(min_x, px);
        max_x = std::max(max_x, px);
        min_y = std::min(min_y, py);
        max_y = std::max(max_y, py);
      }

      tb_det_msgs::msg::TbTarget target;
      target.type = class_name(detection.label);
      tb_det_msgs::msg::TbRoi roi;
      roi.type = target.type;
      roi.rect.x_offset = clamp_bev_pixel(min_x, max_pixel);
      roi.rect.y_offset = clamp_bev_pixel(min_y, max_pixel);
      roi.rect.width = clamp_bev_pixel(max_x, max_pixel) - clamp_bev_pixel(min_x, max_pixel);
      roi.rect.height = clamp_bev_pixel(max_y, max_pixel) - clamp_bev_pixel(min_y, max_pixel);
      roi.confidence = detection.score;
      target.rois.push_back(std::move(roi));

      // 3D box corners as a landmark set (type "bev_box"): four BEV corners in
      // pixels plus the center; z carries the object height reference.
      tb_det_msgs::msg::TbPoint points;
      points.type = "bev_box";
      for (const auto & corner : corners) {
        const float wx = detection.x + corner[0] * c - corner[1] * s;
        const float wy = detection.y + corner[0] * s + corner[1] * c;
        geometry_msgs::msg::Point32 pixel;
        pixel.x = (wx + half_range) * scale;
        pixel.y = (half_range - wy) * scale;
        pixel.z = detection.z;
        points.point.push_back(pixel);
        points.confidence.push_back(detection.score);
      }
      geometry_msgs::msg::Point32 center;
      center.x = (detection.x + half_range) * scale;
      center.y = (half_range - detection.y) * scale;
      center.z = detection.z;
      points.point.push_back(center);
      points.confidence.push_back(detection.score);
      target.points.push_back(std::move(points));

      message.targets.push_back(std::move(target));
    }

    detections_pub_->publish(message);
  }

  void publish_image(
    const std::vector<centerpoint::Detection> & detections, const FrameSnapshot & snapshot)
  {
    const int size = bev_size_;
    const float x_min = -51.2F;
    const float y_max = 51.2F;
    const float scale = static_cast<float>(size) / 102.4F;

    cv::Mat canvas(size, size, CV_8UC3, cv::Scalar(255, 255, 255));

    // Step 1: gray pillar occupancy from voxel coordinates (snapshot owns a
    // copy of the coordinates; the heavy tensor lives in the worker only).
    for (std::size_t i = 0; i < snapshot.pillar_count; ++i) {
      const int cx = snapshot.coordinates[i * 3 + 2];
      const int cy = snapshot.coordinates[i * 3 + 1];
      const float x_world = -51.2F + (static_cast<float>(cx) + 0.5F) * 0.2F;
      const float y_world = -51.2F + (static_cast<float>(cy) + 0.5F) * 0.2F;
      const int u = static_cast<int>((x_world - x_min) * scale);
      const int v = static_cast<int>((y_max - y_world) * scale);
      if (u >= 0 && u < size && v >= 0 && v < size) {
        cv::circle(canvas, cv::Point(u, v), 1, cv::Scalar(180, 180, 180), -1);
      }
    }

    // Step 2: class-colored rotated boxes + darker heading arrows.
    std::size_t drawn = 0;
    for (const auto & det : detections) {
      if (det.score < score_threshold_) {
        continue;
      }
      const cv::Scalar box_color = get_class_color(det.label);
      const cv::Scalar arrow_color(
        static_cast<int>(box_color[0] * 0.6),
        static_cast<int>(box_color[1] * 0.6),
        static_cast<int>(box_color[2] * 0.6));

      const float yaw_adjusted = det.yaw + M_PI / 2.0F;  // 旋转 90 度
      const float c = std::cos(yaw_adjusted);
      const float s = std::sin(yaw_adjusted);
      const float dx = det.width * 0.5F;   // 交换：使用宽度作为前方方向
      const float dy = det.length * 0.5F;  // 交换：使用长度作为侧向方向
      const float local[4][2] = {{dx, dy}, {dx, -dy}, {-dx, -dy}, {-dx, dy}};
      cv::Point corners_px[4];
      for (int i = 0; i < 4; ++i) {
        const float wx = det.x + local[i][0] * c - local[i][1] * s;
        const float wy = det.y + local[i][0] * s + local[i][1] * c;
        corners_px[i] = cv::Point(
          std::clamp(static_cast<int>((wx - x_min) * scale), 0, size - 1),
          std::clamp(static_cast<int>((y_max - wy) * scale), 0, size - 1));
      }
      for (int i = 0; i < 4; ++i) {
        cv::line(canvas, corners_px[i], corners_px[(i + 1) % 4], box_color, 2);
      }
      const cv::Point center_px(
        std::clamp(static_cast<int>((det.x - x_min) * scale), 0, size - 1),
        std::clamp(static_cast<int>((y_max - det.y) * scale), 0, size - 1));
      const cv::Point front_px(
        std::clamp(static_cast<int>((det.x + c * dx - x_min) * scale), 0, size - 1),
        std::clamp(static_cast<int>((y_max - det.y - s * dx) * scale), 0, size - 1));
      cv::arrowedLine(canvas, center_px, front_px, arrow_color, 2, cv::LINE_8, 0, 0.35);
      ++drawn;
    }

    std::vector<uint8_t> jpeg;
    if (!cv::imencode(".jpg", canvas, jpeg)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000, "centerpoint: BEV render encode failed");
      return;
    }

    sensor_msgs::msg::CompressedImage message;
    message.header.stamp = get_clock()->now();
    message.header.frame_id = frame_id_;
    message.format = "jpeg";
    message.data.assign(jpeg.begin(), jpeg.end());
    image_pub_->publish(message);
    (void)drawn;
  }

  std::string pfe_model_;
  std::string rpn_model_;
  std::string merged_model_;
  std::string sweeps_manifest_;
  std::string frames_list_;
  std::string frame_id_;
  std::string image_topic_;
  std::string detections_topic_;
  int run_interval_ms_{0};
  int bev_size_{512};
  float score_threshold_{0.2F};
  bool single_frame_mode_{false};
  std::vector<int> device_ids_;
  std::size_t queue_size_{2};
  std::vector<std::string> frame_paths_;
  std::size_t current_frame_{0};

  std::unique_ptr<centerpoint::CenterPointWorkerPool> pool_;
  rclcpp::Publisher<tb_det_msgs::msg::TbPerceptionTargets>::SharedPtr detections_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr image_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CenterPointInferNode>());
  rclcpp::shutdown();
  return 0;
}
