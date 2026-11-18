#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace centerpoint {

struct SweepSpec {
  std::string point_path;
  float time_lag = 0.0f;
  std::array<double, 16> transform{};  // row-major sweep-to-keyframe transform
};

struct PillarBatch {
  std::vector<float> decorated;       // [30000,20,10], zero padded
  std::vector<int32_t> coordinates;   // [30000,3], z/y/x, zero padded
  std::vector<int32_t> counts;        // [30000]
  std::size_t used = 0;
  std::size_t points_loaded = 0;
  std::size_t points_in_range = 0;
};

// Manifest line: point_file time_lag [m00 m01 ... m33].
// The optional matrix defaults to identity. The first line is the keyframe;
// later lines are sweeps and receive the official 1 m ego-point removal.
bool LoadSweepManifest(const std::string& manifest_path,
                       std::vector<SweepSpec>* sweeps,
                       std::string* error);

bool BuildPillarsFromNuScenes(const std::vector<SweepSpec>& sweeps,
                              PillarBatch* batch,
                              std::string* error);

// Single-frame mode: build pillars from one .bin file, fill only the first
// temporal position [0,:,:,:], leave the rest 9 positions as zeros.
// time_lag is set to 0.0 for all points.
bool BuildPillarsFromSingleFrame(const std::string& point_path,
                                 PillarBatch* batch,
                                 std::string* error);

// In-memory variant of single-frame mode: raw points as N x 5 float32
// (x, y, z, intensity, time-unused). Used by online PointCloud2 input.
bool BuildPillarsFromPoints(const float* xyzin, std::size_t num_points,
                            PillarBatch* batch,
                            std::string* error);

}  // namespace centerpoint