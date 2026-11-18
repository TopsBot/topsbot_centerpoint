// Copyright 2026 TOPSBOT
// SPDX-License-Identifier: Apache-2.0

// CenterPoint offline single-image CLI (counterpart of mobilenet/yolo/gesture/
// segformer *_file_cli): one point-cloud input -> one BEV detection image on
// disk. No ROS, no topics, no webviz.
//
// Input forms:
//   <dir>            -> <dir>/sweeps.txt (10-sweep nuScenes manifest)
//   <file.bin>       -> single-frame mode (N x 5 float32 xyzin)
//   <manifest.txt>   -> 10-sweep mode (first line keyframe, 9 sweeps)
//
// Exit code 0 on success; prints one detection per line for scripting.

#include <topsbot_centerpoint/centerpoint.hpp>
#include <topsbot_centerpoint/centerpoint_preprocess.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace
{

namespace fs = std::filesystem;

struct Options
{
  std::string input;
  std::string output;
  std::string pfe;
  std::string rpn;
  std::string merged;
  bool single_frame{false};
  uint32_t core{0};
  float score_threshold{0.2F};
  int bev_size{1024};
};

void print_usage(const char * executable)
{
  std::cerr
    << "Usage: " << executable << " <input> <output.(jpg|png)> --pfe PATH --rpn PATH [options]\n"
    << "  input        frame dir (sweeps.txt inside), *.bin (single frame) or manifest\n"
    << "  --merged PATH          single-graph NB (PFE+ScatterND+RPN); overrides --pfe/--rpn\n"
    << "  --pfe PATH             required PFE .nb (multi-stage)\n"
    << "  --rpn PATH             required RPN .nb (multi-stage)\n"
    << "  --single-frame         force single-frame mode (input is one .bin)\n"
    << "  --core N               NPU core index (default: 0)\n"
    << "  --score-threshold F    publish/draw filter (default: 0.2)\n"
    << "  --bev-size N           square render pixels (default: 1024)\n"
    << "  -h, --help             this text\n";
}

bool parse_args(int argc, char ** argv, Options * opt)
{
  std::vector<std::string> positional;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&](std::string * dest) -> bool {
      if (i + 1 >= argc) {
        std::cerr << "[ERROR] missing value for " << arg << "\n";
        return false;
      }
      *dest = argv[++i];
      return true;
    };
    if (arg == "--pfe") { if (!next(&opt->pfe)) return false; }
    else if (arg == "--rpn") { if (!next(&opt->rpn)) return false; }
    else if (arg == "--merged") { if (!next(&opt->merged)) return false; }
    else if (arg == "--core") {
      std::string v; if (!next(&v)) return false;
      opt->core = static_cast<uint32_t>(std::stoul(v));
    }
    else if (arg == "--score-threshold") {
      std::string v; if (!next(&v)) return false;
      opt->score_threshold = std::stof(v);
    }
    else if (arg == "--bev-size") {
      std::string v; if (!next(&v)) return false;
      opt->bev_size = std::max(64, std::stoi(v));
    }
    else if (arg == "--single-frame") { opt->single_frame = true; }
    else if (arg == "-h" || arg == "--help") { print_usage(argv[0]); return false; }
    else if (!arg.empty() && arg[0] == '-') {
      std::cerr << "[ERROR] unknown option: " << arg << "\n";
      print_usage(argv[0]);
      return false;
    }
    else {
      positional.push_back(arg);
    }
  }
  if (positional.size() != 2) {
    print_usage(argv[0]);
    return false;
  }
  opt->input = positional[0];
  opt->output = positional[1];
  if (opt->merged.empty() && (opt->pfe.empty() || opt->rpn.empty())) {
    std::cerr << "[ERROR] --merged or (--pfe and --rpn) are required\n";
    return false;
  }
  return true;
}

/// dir -> dir/sweeps.txt; *.bin -> single frame; else manifest as-is.
bool resolve_input(const Options & opt, std::string * path, bool * single_frame)
{
  *single_frame = opt.single_frame;
  if (fs::is_directory(opt.input)) {
    *path = (fs::path(opt.input) / "sweeps.txt").string();
    if (!fs::is_regular_file(*path)) {
      std::cerr << "[ERROR] no sweeps.txt in directory: " << opt.input << "\n";
      return false;
    }
    return true;
  }
  if (!fs::is_regular_file(opt.input)) {
    std::cerr << "[ERROR] input not found: " << opt.input << "\n";
    return false;
  }
  if (opt.input.size() >= 4 && opt.input.substr(opt.input.size() - 4) == ".bin") {
    *single_frame = true;
  }
  *path = opt.input;
  return true;
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

void render_bev(
  cv::Mat * canvas, const centerpoint::PillarBatch & pillars,
  const std::vector<centerpoint::Detection> & detections, const Options & opt)
{
  const int size = opt.bev_size;
  const float x_min = -51.2F;
  const float y_max = 51.2F;
  const float scale = static_cast<float>(size) / 102.4F;

  canvas->create(size, size, CV_8UC3);
  canvas->setTo(cv::Scalar(255, 255, 255));

  // Point-cloud occupancy in gray.
  for (std::size_t i = 0; i < pillars.used; ++i) {
    const int cx = pillars.coordinates[i * 3 + 2];
    const int cy = pillars.coordinates[i * 3 + 1];
    const float x_world = -51.2F + (static_cast<float>(cx) + 0.5F) * 0.2F;
    const float y_world = -51.2F + (static_cast<float>(cy) + 0.5F) * 0.2F;
    const int u = static_cast<int>((x_world - x_min) * scale);
    const int v = static_cast<int>((y_max - y_world) * scale);
    if (u >= 0 && u < size && v >= 0 && v < size) {
      cv::circle(*canvas, cv::Point(u, v), 1, cv::Scalar(180, 180, 180), -1);
    }
  }

  // Rotated boxes (class-colored) + heading arrows (darker shade).
  std::size_t drawn = 0;
  for (const auto & det : detections) {
    if (det.score < opt.score_threshold) {
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
      cv::line(*canvas, corners[i], corners[(i + 1) % 4], box_color, 2);
    }
    const cv::Point center_px(
      std::clamp(static_cast<int>((det.x - x_min) * scale), 0, size - 1),
      std::clamp(static_cast<int>((y_max - det.y) * scale), 0, size - 1));
    const cv::Point front_px(
      std::clamp(static_cast<int>((det.x + c * dx - x_min) * scale), 0, size - 1),
      std::clamp(static_cast<int>((y_max - det.y - s * dx) * scale), 0, size - 1));
    cv::arrowedLine(*canvas, center_px, front_px, arrow_color, 2, cv::LINE_8, 0, 0.35);
    ++drawn;
  }
  std::cerr << "[INFO] drew " << drawn << " of " << detections.size() << " detections\n";
}

}  // namespace

int main(int argc, char ** argv)
{
  Options opt;
  if (!parse_args(argc, argv, &opt)) {
    return 1;
  }

  std::string manifest_path;
  bool single_frame = false;
  if (!resolve_input(opt, &manifest_path, &single_frame)) {
    return 1;
  }
  if (!opt.merged.empty()) {
    if (!fs::is_regular_file(opt.merged)) { std::cerr << "[ERROR] merged model not found: " << opt.merged << "\n"; return 1; }
  } else {
    if (!fs::is_regular_file(opt.pfe)) { std::cerr << "[ERROR] PFE not found: " << opt.pfe << "\n"; return 1; }
    if (!fs::is_regular_file(opt.rpn)) { std::cerr << "[ERROR] RPN not found: " << opt.rpn << "\n"; return 1; }
  }

  // 1) build pillars
  centerpoint::PillarBatch pillars;
  std::string error;
  const auto preprocess_begin = std::chrono::steady_clock::now();
  if (single_frame) {
    if (!centerpoint::BuildPillarsFromSingleFrame(manifest_path, &pillars, &error)) {
      std::cerr << "[ERROR] single-frame preprocessing: " << error << "\n";
      return 1;
    }
  } else {
    std::vector<centerpoint::SweepSpec> sweeps;
    if (!centerpoint::LoadSweepManifest(manifest_path, &sweeps, &error)) {
      std::cerr << "[ERROR] manifest: " << error << "\n";
      return 1;
    }
    if (!centerpoint::BuildPillarsFromNuScenes(sweeps, &pillars, &error)) {
      std::cerr << "[ERROR] preprocessing: " << error << "\n";
      return 1;
    }
  }
  const auto preprocess_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - preprocess_begin).count();
  std::cerr << "[INFO] pillars=" << pillars.used
            << " points_loaded=" << pillars.points_loaded
            << " points_in_range=" << pillars.points_in_range
            << " preprocess_ms=" << preprocess_ms << "\n";

  // 2) init + infer
  centerpoint::CenterPoint detector;
  bool init_ok;
  if (!opt.merged.empty()) {
    init_ok = detector.InitMerged(opt.merged, opt.core);
  } else {
    init_ok = detector.Init(opt.pfe, opt.rpn, opt.core);
  }
  if (!init_ok) {
    std::cerr << "[ERROR] CenterPoint Init failed (core " << opt.core << ")\n";
    return 1;
  }
  detector.set_score_threshold(opt.score_threshold);

  std::vector<centerpoint::Detection> detections;
  centerpoint::Timing timing;
  if (!detector.Detect(pillars, &detections, &timing)) {
    std::cerr << "[ERROR] Detect failed\n";
    detector.Deinit();
    return 1;
  }
  timing.preprocess_ms = preprocess_ms;
  detector.Deinit();

  if (timing.single_graph) {
    // Mirrors axera's single-graph summary: one inference number + total + FPS.
    const double total_ms = timing.preprocess_ms + timing.inference_ms() + timing.postprocess_ms;
    std::cout << "[Performance] detections=" << detections.size()
              << ", preprocess=" << timing.preprocess_ms << "ms"
              << ", inference=" << timing.inference_ms() << "ms"
              << ", postprocess=" << timing.postprocess_ms << "ms"
              << ", total=" << total_ms << "ms"
              << ", FPS=" << (total_ms > 0.0 ? 1000.0 / total_ms : 0.0)
              << "\n";
  } else {
    std::cout << "[INFO] detections=" << detections.size()
              << " pfe_ms=" << timing.pfe_ms << " scatter_ms=" << timing.scatter_ms
              << " rpn_ms=" << timing.rpn_ms << " postprocess_ms=" << timing.postprocess_ms
              << "\n";
  }
  // One machine-readable line per kept detection: type x y z l w h yaw score
  for (const auto & det : detections) {
    if (det.score < opt.score_threshold) continue;
    std::printf("%s %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.4f\n",
                class_name(det.label), det.x, det.y, det.z,
                det.length, det.width, det.height, det.yaw, det.score);
  }

  // 3) render + write
  cv::Mat canvas;
  render_bev(&canvas, pillars, detections, opt);
  if (!cv::imwrite(opt.output, canvas)) {
    std::cerr << "[ERROR] failed to write image: " << opt.output << "\n";
    return 1;
  }
  std::cerr << "[INFO] wrote " << opt.output << " (" << opt.bev_size << "x"
            << opt.bev_size << ")\n";
  return 0;
}
