// Copyright 2026 TOPSBOT
// SPDX-License-Identifier: Apache-2.0

// Universal single-frame streaming inference node
// Supports single-core or multi-core NPU execution based on device_ids config

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
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
#include <topsbot_centerpoint/centerpoint_worker.hpp>
#include <topsbot_centerpoint/centerpoint_preprocess.hpp>

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
  switch (label) {
    case 0:  return cv::Scalar(255, 100, 0);     // car - Blue
    case 1:  return cv::Scalar(255, 150, 50);    // truck - Light Blue
    case 2:  return cv::Scalar(200, 120, 0);     // construction_vehicle - Dark Blue
    case 3:  return cv::Scalar(255, 200, 100);   // bus - Cyan-ish
    case 4:  return cv::Scalar(180, 100, 0);     // trailer - Navy Blue
    case 5:  return cv::Scalar(0, 140, 255);     // barrier - Orange
    case 6:  return cv::Scalar(255, 255, 0);     // motorcycle - Cyan
    case 7:  return cv::Scalar(200, 255, 100);   // bicycle - Light Cyan
    case 8:  return cv::Scalar(0, 255, 255);     // pedestrian - Yellow
    case 9:  return cv::Scalar(0, 165, 255);     // traffic_cone - Orange-Red
    default: return cv::Scalar(128, 128, 128);   // unknown - Gray
  }
}

tb_det_msgs::msg::TbPerf make_perf(const std::string & type, const float milliseconds)
{
  tb_det_msgs::msg::TbPerf perf;
  perf.type = type;
  perf.time_ms_duration = static_cast<double>(milliseconds);
  return perf;
}

bool load_bin_file(const std::string & path, std::vector<float> * points)
{
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    return false;
  }
  const std::streamoff bytes = stream.tellg();
  if (bytes <= 0 || bytes % static_cast<std::streamoff>(5 * sizeof(float)) != 0) {
    return false;
  }
  points->resize(static_cast<std::size_t>(bytes) / sizeof(float));
  stream.seekg(0);
  return stream.read(reinterpret_cast<char *>(points->data()), bytes).good();
}

// Context for multi-core mode
struct FrameContext {
  std::size_t frame_index;
  std::string bin_file;
  float load_ms;
  float preprocess_ms;
  centerpoint::PillarBatch pillars;
};

}  // namespace

class CenterPointSingleFrameStream : public rclcpp::Node
{
public:
  CenterPointSingleFrameStream()
  : Node("centerpoint_single_frame_stream",
         rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true))
  {
    // Read parameters
    pfe_model_ = expand_user_path(
      read_parameter<std::string>(*this, "pfe_model", ""));
    rpn_model_ = expand_user_path(
      read_parameter<std::string>(*this, "rpn_model", ""));
    merged_model_ = expand_user_path(
      read_parameter<std::string>(*this, "merged_model", ""));

    bin_files_dir_ = expand_user_path(
      read_parameter<std::string>(*this, "bin_files_dir", ""));
    bin_list_file_ = expand_user_path(
      read_parameter<std::string>(*this, "bin_list_file", ""));

    frame_interval_ms_ = std::max(1,
      read_parameter<int>(*this, "frame_interval_ms", 100));
    loop_ = read_parameter<bool>(*this, "loop", true);
    score_threshold_ = static_cast<float>(
      read_parameter<double>(*this, "score_threshold", 0.2));
    bev_size_ = std::max(64, read_parameter<int>(*this, "bev_size", 1024));
    max_queue_size_ = std::max(1,
      read_parameter<int>(*this, "max_queue_size", 10));

    // Read device_ids (supports both single and multiple cores)
    auto device_ids_param = read_parameter<std::vector<int64_t>>(*this, "device_ids",
      std::vector<int64_t>{0});
    device_ids_.clear();
    for (auto id : device_ids_param) {
      device_ids_.push_back(static_cast<int>(id));
    }
    if (device_ids_.empty()) {
      device_ids_.push_back(0);  // Fallback to core 0
    }

    frame_id_ = read_parameter<std::string>(*this, "frame_id", "lidar");
    image_topic_ = read_parameter<std::string>(*this, "image_topic", "/image_jpeg");
    detections_topic_ = read_parameter<std::string>(*this, "detections_topic", "/detections");

    if (merged_model_.empty() && (pfe_model_.empty() || rpn_model_.empty())) {
      throw std::invalid_argument(
        "merged_model or (pfe_model and rpn_model) are required");
    }

    // Load bin file list
    load_bin_files();

    // Choose single-core or multi-core mode
    multi_core_ = (device_ids_.size() > 1);

    if (multi_core_) {
      initialize_multi_core();
    } else {
      initialize_single_core();
    }

    // Create publishers
    detections_pub_ = create_publisher<tb_det_msgs::msg::TbPerceptionTargets>(
      detections_topic_, 10);
    image_pub_ = create_publisher<sensor_msgs::msg::CompressedImage>(
      image_topic_, 10);

    // Start timer
    timer_ = create_wall_timer(
      std::chrono::milliseconds(frame_interval_ms_),
      std::bind(&CenterPointSingleFrameStream::process_next_frame, this));

    RCLCPP_INFO(get_logger(),
      "CenterPoint single-frame stream ready: %zu frames, interval=%dms, loop=%d, FPS=%.1f",
      bin_files_.size(), frame_interval_ms_, loop_ ? 1 : 0,
      1000.0f / frame_interval_ms_);
  }

  ~CenterPointSingleFrameStream()
  {
    timer_->cancel();
    if (multi_core_) {
      worker_pool_.reset();
    } else {
      detector_.Deinit();
    }
  }

private:
  void initialize_single_core()
  {
    RCLCPP_INFO(get_logger(), "Initializing single-core mode on NPU core %d...", device_ids_[0]);
    bool ok;
    if (!merged_model_.empty()) {
      ok = detector_.InitMerged(merged_model_, static_cast<uint32_t>(device_ids_[0]));
      if (!ok) throw std::runtime_error("CenterPoint InitMerged failed");
    } else {
      ok = detector_.Init(pfe_model_, rpn_model_,
                          static_cast<uint32_t>(device_ids_[0]));
      if (!ok) throw std::runtime_error("CenterPoint Init failed");
    }
    detector_.set_score_threshold(score_threshold_);
  }

  void initialize_multi_core()
  {
    centerpoint::CenterPointWorkerPool::Config config;
    config.device_ids = device_ids_;
    config.pfe_model = pfe_model_;
    config.rpn_model = rpn_model_;
    config.merged_model = merged_model_;
    config.max_queue_size = static_cast<std::size_t>(max_queue_size_);
    config.strategy = centerpoint::BalanceStrategy::LeastLoaded;
    config.score_threshold = score_threshold_;

    RCLCPP_INFO(get_logger(), "Initializing multi-core mode on %zu NPU cores (queue_size=%d)...",
                device_ids_.size(), max_queue_size_);

    worker_pool_ = std::make_unique<centerpoint::CenterPointWorkerPool>(config);

    std::string error;
    if (!worker_pool_->initialize(&error)) {
      throw std::runtime_error("Worker pool initialization failed: " + error);
    }

    worker_pool_->set_result_callback(
      std::bind(&CenterPointSingleFrameStream::on_detection_result, this,
                std::placeholders::_1));
  }

  void load_bin_files()
  {
    if (!bin_list_file_.empty()) {
      std::ifstream ifs(bin_list_file_);
      if (!ifs) {
        throw std::runtime_error("Cannot open bin_list_file: " + bin_list_file_);
      }
      std::string line;
      while (std::getline(ifs, line)) {
        const auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#') {
          continue;
        }
        std::string path = line.substr(first);
        const auto last = path.find_last_not_of(" \t\r\n");
        if (last != std::string::npos) {
          path = path.substr(0, last + 1);
        }
        if (std::filesystem::exists(path)) {
          bin_files_.push_back(path);
        }
      }
    } else if (!bin_files_dir_.empty()) {
      for (const auto & entry : std::filesystem::directory_iterator(bin_files_dir_)) {
        if (entry.is_regular_file() && entry.path().extension() == ".bin") {
          bin_files_.push_back(entry.path().string());
        }
      }
      std::sort(bin_files_.begin(), bin_files_.end());
    } else {
      throw std::invalid_argument(
        "Either bin_files_dir or bin_list_file must be specified");
    }

    if (bin_files_.empty()) {
      throw std::runtime_error("No .bin files found");
    }
  }

  void process_next_frame()
  {
    if (current_index_ >= bin_files_.size()) {
      if (loop_) {
        current_index_ = 0;
        RCLCPP_INFO(get_logger(), "=== Looping back to first frame ===");
      } else {
        RCLCPP_INFO(get_logger(), "=== All frames processed. Stopping. ===");
        timer_->cancel();
        return;
      }
    }

    if (multi_core_) {
      process_next_frame_multi_core();
    } else {
      process_next_frame_single_core();
    }
  }

  void process_next_frame_single_core()
  {
    const std::string & bin_file = bin_files_[current_index_];
    const auto load_begin = std::chrono::steady_clock::now();

    // Load .bin file
    std::vector<float> points;
    if (!load_bin_file(bin_file, &points)) {
      RCLCPP_ERROR(get_logger(), "Failed to load: %s", bin_file.c_str());
      ++current_index_;
      return;
    }
    const auto load_ms = std::chrono::duration<float, std::milli>(
      std::chrono::steady_clock::now() - load_begin).count();

    // Build pillars
    const auto preprocess_begin = std::chrono::steady_clock::now();
    centerpoint::PillarBatch pillars;
    std::string error;
    if (!centerpoint::BuildPillarsFromPoints(
          points.data(), points.size() / 5, &pillars, &error)) {
      RCLCPP_ERROR(get_logger(), "Pillar build failed: %s", error.c_str());
      ++current_index_;
      return;
    }
    const auto preprocess_ms = std::chrono::duration<float, std::milli>(
      std::chrono::steady_clock::now() - preprocess_begin).count();

    // Inference
    std::vector<centerpoint::Detection> detections;
    centerpoint::Timing timing;
    if (!detector_.Detect(pillars, &detections, &timing)) {
      RCLCPP_ERROR(get_logger(), "Detection failed");
      ++current_index_;
      return;
    }

    // Publish
    publish_detections(detections, timing, preprocess_ms, load_ms, device_ids_[0]);
    publish_bev_image(detections, pillars);

    if (timing.single_graph) {
      const double total_ms = load_ms + preprocess_ms + timing.inference_ms() +
                              timing.postprocess_ms;
      RCLCPP_INFO(get_logger(),
        "Frame %zu/%zu: %zu detections, load=%.1fms preproc=%.1fms inference=%.1fms "
        "post=%.1fms total=%.1fms FPS=%.1f core=%d (merged)",
        current_index_ + 1, bin_files_.size(), detections.size(),
        load_ms, preprocess_ms, timing.inference_ms(), timing.postprocess_ms, total_ms,
        total_ms > 0.0 ? 1000.0 / total_ms : 0.0, device_ids_[0]);
    } else {
      RCLCPP_INFO(get_logger(),
        "Frame %zu/%zu: %zu detections, load=%.1fms preproc=%.1fms pfe=%.1fms rpn=%.1fms post=%.1fms core=%d",
        current_index_ + 1, bin_files_.size(), detections.size(),
        load_ms, preprocess_ms, timing.pfe_ms, timing.rpn_ms, timing.postprocess_ms,
        device_ids_[0]);
    }

    ++current_index_;
  }

  void process_next_frame_multi_core()
  {
    const std::string & bin_file = bin_files_[current_index_];
    const auto load_begin = std::chrono::steady_clock::now();

    // Load .bin file
    std::vector<float> points;
    if (!load_bin_file(bin_file, &points)) {
      RCLCPP_ERROR(get_logger(), "Failed to load: %s", bin_file.c_str());
      ++current_index_;
      return;
    }
    const auto load_ms = std::chrono::duration<float, std::milli>(
      std::chrono::steady_clock::now() - load_begin).count();

    // Build pillars
    const auto preprocess_begin = std::chrono::steady_clock::now();
    centerpoint::PillarBatch pillars;
    std::string error;
    if (!centerpoint::BuildPillarsFromPoints(
          points.data(), points.size() / 5, &pillars, &error)) {
      RCLCPP_ERROR(get_logger(), "Pillar build failed: %s", error.c_str());
      ++current_index_;
      return;
    }
    const auto preprocess_ms = std::chrono::duration<float, std::milli>(
      std::chrono::steady_clock::now() - preprocess_begin).count();

    // Create context
    auto context = std::make_shared<FrameContext>();
    context->frame_index = current_index_;
    context->bin_file = bin_file;
    context->load_ms = load_ms;
    context->preprocess_ms = preprocess_ms;
    context->pillars = pillars;

    // Submit to worker pool
    centerpoint::DetectionRequest request;
    request.pillars = std::move(pillars);
    request.frame_id = current_index_;
    request.enqueued_at = std::chrono::steady_clock::now();
    request.context = context;

    if (!worker_pool_->submit(std::move(request))) {
      RCLCPP_WARN(get_logger(), "Frame %zu: worker pool queue full, dropping frame",
                  current_index_);
    }

    ++current_index_;
  }

  void on_detection_result(const centerpoint::DetectionResult & result)
  {
    if (!result.success) {
      RCLCPP_ERROR(get_logger(), "Frame %lu detection failed: %s",
                   result.frame_id, result.error.c_str());
      return;
    }

    auto context = std::static_pointer_cast<FrameContext>(result.context);
    if (!context) {
      RCLCPP_ERROR(get_logger(), "Frame %lu: missing context", result.frame_id);
      return;
    }

    publish_detections(result.detections, result.timing,
                      context->preprocess_ms, context->load_ms, result.device_id);
    publish_bev_image(result.detections, context->pillars);

    if (result.timing.single_graph) {
      const double total_ms = context->load_ms + context->preprocess_ms +
                              result.timing.inference_ms() + result.timing.postprocess_ms;
      RCLCPP_INFO(get_logger(),
        "Frame %zu/%zu: %zu detections, load=%.1fms preproc=%.1fms inference=%.1fms "
        "post=%.1fms total=%.1fms FPS=%.1f core=%d (merged)",
        context->frame_index + 1, bin_files_.size(), result.detections.size(),
        context->load_ms, context->preprocess_ms, result.timing.inference_ms(),
        result.timing.postprocess_ms, total_ms,
        total_ms > 0.0 ? 1000.0 / total_ms : 0.0, result.device_id);
    } else {
      RCLCPP_INFO(get_logger(),
        "Frame %zu/%zu: %zu detections, load=%.1fms preproc=%.1fms pfe=%.1fms rpn=%.1fms post=%.1fms core=%d",
        context->frame_index + 1, bin_files_.size(), result.detections.size(),
        context->load_ms, context->preprocess_ms,
        result.timing.pfe_ms, result.timing.rpn_ms, result.timing.postprocess_ms,
        result.device_id);
    }
  }

  void publish_detections(
    const std::vector<centerpoint::Detection> & detections,
    const centerpoint::Timing & timing,
    float preprocess_ms,
    float load_ms,
    int core_id)
  {
    tb_det_msgs::msg::TbPerceptionTargets message;
    message.header.stamp = get_clock()->now();
    message.header.frame_id = frame_id_;
    message.fps = 1000.0f / frame_interval_ms_;

    message.perfs.push_back(make_perf("load", load_ms));
    message.perfs.push_back(make_perf("preprocess", preprocess_ms));
    if (timing.single_graph) {
      message.perfs.push_back(make_perf("inference", timing.inference_ms()));
    } else {
      message.perfs.push_back(make_perf("pfe", timing.pfe_ms));
      message.perfs.push_back(make_perf("scatter", timing.scatter_ms));
      message.perfs.push_back(make_perf("rpn", timing.rpn_ms));
    }
    message.perfs.push_back(make_perf("postprocess", timing.postprocess_ms));
    message.perfs.push_back(make_perf("core_id", static_cast<float>(core_id)));

    const float half_range = 51.2F;
    const float scale = static_cast<float>(bev_size_) / (2.0F * half_range);

    for (const auto & det : detections) {
      if (det.score < score_threshold_) {
        continue;
      }

      tb_det_msgs::msg::TbTarget target;
      target.type = class_name(det.label);

      const float hl = det.length * 0.5F;
      const float hw = det.width * 0.5F;
      const float c = std::cos(det.yaw);
      const float s = std::sin(det.yaw);

      float min_x = std::numeric_limits<float>::max();
      float max_x = -std::numeric_limits<float>::max();
      float min_y = std::numeric_limits<float>::max();
      float max_y = -std::numeric_limits<float>::max();

      const float corners[4][2] = {{-hl, -hw}, {hl, -hw}, {hl, hw}, {-hl, hw}};
      for (const auto & corner : corners) {
        const float wx = det.x + corner[0] * c - corner[1] * s;
        const float wy = det.y + corner[0] * s + corner[1] * c;
        const float px = (wx + half_range) * scale;
        const float py = (half_range - wy) * scale;
        min_x = std::min(min_x, px);
        max_x = std::max(max_x, px);
        min_y = std::min(min_y, py);
        max_y = std::max(max_y, py);
      }

      tb_det_msgs::msg::TbRoi roi;
      roi.type = target.type;
      roi.rect.x_offset = static_cast<uint32_t>(std::max(0.0f, min_x));
      roi.rect.y_offset = static_cast<uint32_t>(std::max(0.0f, min_y));
      roi.rect.width = static_cast<uint32_t>(max_x - min_x);
      roi.rect.height = static_cast<uint32_t>(max_y - min_y);
      roi.confidence = det.score;
      target.rois.push_back(std::move(roi));

      // Full 3D box as a landmark set (type "bev_box"): four BEV corners in
      // pixels plus the center; z carries the object height reference. Matches
      // the online node (centerpoint_infer_node) so webviz can draw the box.
      tb_det_msgs::msg::TbPoint points;
      points.type = "bev_box";
      for (const auto & corner : corners) {
        const float wx = det.x + corner[0] * c - corner[1] * s;
        const float wy = det.y + corner[0] * s + corner[1] * c;
        geometry_msgs::msg::Point32 pixel;
        pixel.x = (wx + half_range) * scale;
        pixel.y = (half_range - wy) * scale;
        pixel.z = det.z;
        points.point.push_back(pixel);
        points.confidence.push_back(det.score);
      }
      geometry_msgs::msg::Point32 center;
      center.x = (det.x + half_range) * scale;
      center.y = (half_range - det.y) * scale;
      center.z = det.z;
      points.point.push_back(center);
      points.confidence.push_back(det.score);
      target.points.push_back(std::move(points));

      message.targets.push_back(std::move(target));
    }

    detections_pub_->publish(message);
  }

  void publish_bev_image(
    const std::vector<centerpoint::Detection> & detections,
    const centerpoint::PillarBatch & pillars)
  {
    const int size = bev_size_;
    const float x_min = -51.2F;
    const float y_max = 51.2F;
    const float scale = static_cast<float>(size) / 102.4F;

    cv::Mat canvas(size, size, CV_8UC3, cv::Scalar(255, 255, 255));

    // Draw pillar occupancy
    for (std::size_t i = 0; i < pillars.used; ++i) {
      const int cx = pillars.coordinates[i * 3 + 2];
      const int cy = pillars.coordinates[i * 3 + 1];
      const float x_world = -51.2F + (static_cast<float>(cx) + 0.5F) * 0.2F;
      const float y_world = -51.2F + (static_cast<float>(cy) + 0.5F) * 0.2F;
      const int u = static_cast<int>((x_world - x_min) * scale);
      const int v = static_cast<int>((y_max - y_world) * scale);
      if (u >= 0 && u < size && v >= 0 && v < size) {
        cv::circle(canvas, cv::Point(u, v), 1, cv::Scalar(180, 180, 180), -1);
      }
    }

    // Draw detection boxes
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

      cv::Point corners[4];
      for (int i = 0; i < 4; ++i) {
        const float wx = det.x + local[i][0] * c - local[i][1] * s;
        const float wy = det.y + local[i][0] * s + local[i][1] * c;
        corners[i] = cv::Point(
          std::clamp(static_cast<int>((wx - x_min) * scale), 0, size - 1),
          std::clamp(static_cast<int>((y_max - wy) * scale), 0, size - 1));
      }

      for (int i = 0; i < 4; ++i) {
        cv::line(canvas, corners[i], corners[(i + 1) % 4], box_color, 2);
      }

      const cv::Point center(
        std::clamp(static_cast<int>((det.x - x_min) * scale), 0, size - 1),
        std::clamp(static_cast<int>((y_max - det.y) * scale), 0, size - 1));
      const cv::Point front(
        std::clamp(static_cast<int>((det.x + c * dx - x_min) * scale), 0, size - 1),
        std::clamp(static_cast<int>((y_max - det.y - s * dx) * scale), 0, size - 1));

      cv::arrowedLine(canvas, center, front, arrow_color, 2, cv::LINE_8, 0, 0.35);
    }

    std::vector<uint8_t> jpeg;
    if (!cv::imencode(".jpg", canvas, jpeg)) {
      return;
    }

    sensor_msgs::msg::CompressedImage message;
    message.header.stamp = get_clock()->now();
    message.header.frame_id = frame_id_;
    message.format = "jpeg";
    message.data.assign(jpeg.begin(), jpeg.end());
    image_pub_->publish(message);
  }

  std::string pfe_model_;
  std::string rpn_model_;
  std::string merged_model_;
  std::string bin_files_dir_;
  std::string bin_list_file_;
  int frame_interval_ms_;
  bool loop_;
  float score_threshold_;
  int bev_size_;
  int max_queue_size_;
  std::vector<int> device_ids_;
  std::string frame_id_;
  std::string image_topic_;
  std::string detections_topic_;

  std::vector<std::string> bin_files_;
  std::size_t current_index_{0};

  bool multi_core_;
  centerpoint::CenterPoint detector_;  // For single-core mode
  std::unique_ptr<centerpoint::CenterPointWorkerPool> worker_pool_;  // For multi-core mode

  rclcpp::Publisher<tb_det_msgs::msg::TbPerceptionTargets>::SharedPtr detections_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr image_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CenterPointSingleFrameStream>());
  rclcpp::shutdown();
  return 0;
}
