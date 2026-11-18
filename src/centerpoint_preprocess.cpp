#include <topsbot_centerpoint/centerpoint_preprocess.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_map>

namespace centerpoint {
namespace {
constexpr std::size_t kMaxPillars = 30000;
constexpr std::size_t kMaxPoints = 20;
constexpr std::size_t kRawFeatures = 5;
constexpr std::size_t kDecoratedFeatures = 10;
constexpr int kGridX = 512, kGridY = 512, kGridZ = 1;
constexpr float kMinX = -51.2f, kMinY = -51.2f, kMinZ = -5.0f;
constexpr float kMaxX = 51.2f, kMaxY = 51.2f, kMaxZ = 3.0f;
constexpr float kVoxelX = 0.2f, kVoxelY = 0.2f, kVoxelZ = 8.0f;

std::array<double, 16> Identity() {
  return {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
}

bool ReadPoints(const std::string& path, std::vector<float>* points, std::string* error) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) { *error = "cannot open point file: " + path; return false; }
  const std::streamoff bytes = stream.tellg();
  if (bytes <= 0 || bytes % static_cast<std::streamoff>(5 * sizeof(float)) != 0) {
    *error = "nuScenes point file must contain N x 5 float32 values: " + path; return false;
  }
  points->resize(static_cast<std::size_t>(bytes) / sizeof(float));
  stream.seekg(0);
  if (!stream.read(reinterpret_cast<char*>(points->data()), bytes)) {
    *error = "failed reading point file: " + path; return false;
  }
  return true;
}
}  // namespace

bool LoadSweepManifest(const std::string& path, std::vector<SweepSpec>* sweeps,
                       std::string* error) {
  std::ifstream stream(path);
  if (!stream) { *error = "cannot open sweep manifest: " + path; return false; }
  sweeps->clear();
  const std::filesystem::path base = std::filesystem::absolute(path).parent_path();
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(stream, line)) {
    ++line_number;
    const auto first = line.find_first_not_of(" \t\r");
    if (first == std::string::npos || line[first] == '#') continue;
    std::istringstream values(line);
    SweepSpec sweep;
    sweep.transform = Identity();
    if (!(values >> sweep.point_path >> sweep.time_lag)) {
      *error = "invalid manifest line " + std::to_string(line_number); return false;
    }
    double value = 0.0;
    if (values >> value) {
      sweep.transform[0] = value;
      for (std::size_t i = 1; i < sweep.transform.size(); ++i) {
        if (!(values >> sweep.transform[i])) {
          *error = "transform must contain 16 floats at line " + std::to_string(line_number);
          return false;
        }
      }
    }
    std::string extra;
    if (values >> extra) { *error = "extra token at manifest line " + std::to_string(line_number); return false; }
    std::filesystem::path point_path(sweep.point_path);
    if (point_path.is_relative()) sweep.point_path = (base / point_path).lexically_normal().string();
    sweeps->push_back(std::move(sweep));
  }
  if (sweeps->empty()) { *error = "sweep manifest is empty"; return false; }
  if (sweeps->size() != 10) { *error = "official model requires exactly 10 sweeps"; return false; }
  return true;
}

bool BuildPillarsFromNuScenes(const std::vector<SweepSpec>& sweeps,
                              PillarBatch* batch, std::string* error) {
  if (sweeps.size() != 10) { *error = "expected exactly 10 sweeps"; return false; }
  batch->decorated.assign(kMaxPillars * kMaxPoints * kDecoratedFeatures, 0.0f);
  batch->coordinates.assign(kMaxPillars * 3, 0);
  batch->counts.assign(kMaxPillars, 0);
  batch->used = batch->points_loaded = batch->points_in_range = 0;
  std::vector<float> voxels(kMaxPillars * kMaxPoints * kRawFeatures, 0.0f);
  std::unordered_map<int, std::size_t> lookup;
  lookup.reserve(kMaxPillars);

  for (std::size_t sweep_index = 0; sweep_index < sweeps.size(); ++sweep_index) {
    std::vector<float> raw;
    if (!ReadPoints(sweeps[sweep_index].point_path, &raw, error)) return false;
    const auto& m = sweeps[sweep_index].transform;
    const std::size_t point_count = raw.size() / 5;
    batch->points_loaded += point_count;
    for (std::size_t i = 0; i < point_count; ++i) {
      const float x0 = raw[i * 5], y0 = raw[i * 5 + 1], z0 = raw[i * 5 + 2];
      if (sweep_index != 0 && std::abs(x0) < 1.0f && std::abs(y0) < 1.0f) continue;
      const float x = static_cast<float>(m[0] * x0 + m[1] * y0 + m[2] * z0 + m[3]);
      const float y = static_cast<float>(m[4] * x0 + m[5] * y0 + m[6] * z0 + m[7]);
      const float z = static_cast<float>(m[8] * x0 + m[9] * y0 + m[10] * z0 + m[11]);
      if (!(x >= kMinX && x < kMaxX && y >= kMinY && y < kMaxY && z >= kMinZ && z < kMaxZ)) continue;
      const int cx = static_cast<int>(std::floor((x - kMinX) / kVoxelX));
      const int cy = static_cast<int>(std::floor((y - kMinY) / kVoxelY));
      const int cz = static_cast<int>(std::floor((z - kMinZ) / kVoxelZ));
      if (cx < 0 || cx >= kGridX || cy < 0 || cy >= kGridY || cz < 0 || cz >= kGridZ) continue;
      ++batch->points_in_range;
      const int key = (cz * kGridY + cy) * kGridX + cx;
      auto found = lookup.find(key);
      std::size_t pillar = 0;
      if (found == lookup.end()) {
        if (batch->used == kMaxPillars) continue;
        pillar = batch->used++;
        lookup.emplace(key, pillar);
        batch->coordinates[pillar * 3] = cz;
        batch->coordinates[pillar * 3 + 1] = cy;
        batch->coordinates[pillar * 3 + 2] = cx;
      } else {
        pillar = found->second;
      }
      const int slot = batch->counts[pillar];
      if (slot >= static_cast<int>(kMaxPoints)) continue;
      float* point = &voxels[(pillar * kMaxPoints + static_cast<std::size_t>(slot)) * kRawFeatures];
      point[0] = x; point[1] = y; point[2] = z; point[3] = raw[i * 5 + 3];
      point[4] = sweeps[sweep_index].time_lag;
      ++batch->counts[pillar];
    }
  }

  for (std::size_t pillar = 0; pillar < batch->used; ++pillar) {
    const int count = batch->counts[pillar];
    float mean_x = 0, mean_y = 0, mean_z = 0;
    for (int p = 0; p < count; ++p) {
      const float* point = &voxels[(pillar * kMaxPoints + p) * kRawFeatures];
      mean_x += point[0]; mean_y += point[1]; mean_z += point[2];
    }
    mean_x /= count; mean_y /= count; mean_z /= count;
    const float center_x = kMinX + (batch->coordinates[pillar * 3 + 2] + 0.5f) * kVoxelX;
    const float center_y = kMinY + (batch->coordinates[pillar * 3 + 1] + 0.5f) * kVoxelY;
    for (int p = 0; p < count; ++p) {
      const float* point = &voxels[(pillar * kMaxPoints + p) * kRawFeatures];
      float* out = &batch->decorated[(pillar * kMaxPoints + p) * kDecoratedFeatures];
      std::copy_n(point, kRawFeatures, out);
      out[5] = point[0] - mean_x; out[6] = point[1] - mean_y; out[7] = point[2] - mean_z;
      out[8] = point[0] - center_x; out[9] = point[1] - center_y;
    }
  }
  return true;
}

bool BuildPillarsFromPoints(const float* xyzin, std::size_t num_points,
                            PillarBatch* batch, std::string* error) {
  (void)error;
  // Single-frame layout: all zeros; only temporal position 0 is filled.
  batch->decorated.assign(kMaxPillars * kMaxPoints * kDecoratedFeatures, 0.0f);
  batch->coordinates.assign(kMaxPillars * 3, 0);
  batch->counts.assign(kMaxPillars, 0);
  batch->used = batch->points_loaded = batch->points_in_range = 0;

  std::vector<float> voxels(kMaxPillars * kMaxPoints * kRawFeatures, 0.0f);
  std::unordered_map<int, std::size_t> lookup;
  lookup.reserve(kMaxPillars);

  batch->points_loaded = num_points;

  // Voxelize: identity transform, no ego-point removal (matches file mode).
  for (std::size_t i = 0; i < num_points; ++i) {
    const float x = xyzin[i * 5];
    const float y = xyzin[i * 5 + 1];
    const float z = xyzin[i * 5 + 2];

    if (!(x >= kMinX && x < kMaxX && y >= kMinY && y < kMaxY && z >= kMinZ && z < kMaxZ)) continue;

    const int cx = static_cast<int>(std::floor((x - kMinX) / kVoxelX));
    const int cy = static_cast<int>(std::floor((y - kMinY) / kVoxelY));
    const int cz = static_cast<int>(std::floor((z - kMinZ) / kVoxelZ));
    if (cx < 0 || cx >= kGridX || cy < 0 || cy >= kGridY || cz < 0 || cz >= kGridZ) continue;
    ++batch->points_in_range;
    const int key = (cz * kGridY + cy) * kGridX + cx;

    auto found = lookup.find(key);
    std::size_t pillar = 0;
    if (found == lookup.end()) {
      if (batch->used == kMaxPillars) continue;
      pillar = batch->used++;
      lookup.emplace(key, pillar);
      batch->coordinates[pillar * 3] = cz;
      batch->coordinates[pillar * 3 + 1] = cy;
      batch->coordinates[pillar * 3 + 2] = cx;
    } else {
      pillar = found->second;
    }

    const int slot = batch->counts[pillar];
    if (slot >= static_cast<int>(kMaxPoints)) continue;

    float* point = &voxels[(pillar * kMaxPoints + static_cast<std::size_t>(slot)) * kRawFeatures];
    point[0] = x;
    point[1] = y;
    point[2] = z;
    point[3] = xyzin[i * 5 + 3];  // intensity
    point[4] = 0.0f;              // time_lag = 0 (single frame)
    ++batch->counts[pillar];
  }

  for (std::size_t pillar = 0; pillar < batch->used; ++pillar) {
    const int count = batch->counts[pillar];
    float mean_x = 0, mean_y = 0, mean_z = 0;
    for (int p = 0; p < count; ++p) {
      const float* point = &voxels[(pillar * kMaxPoints + p) * kRawFeatures];
      mean_x += point[0]; mean_y += point[1]; mean_z += point[2];
    }
    mean_x /= count; mean_y /= count; mean_z /= count;

    const float center_x = kMinX + (batch->coordinates[pillar * 3 + 2] + 0.5f) * kVoxelX;
    const float center_y = kMinY + (batch->coordinates[pillar * 3 + 1] + 0.5f) * kVoxelY;

    for (int p = 0; p < count; ++p) {
      const float* point = &voxels[(pillar * kMaxPoints + p) * kRawFeatures];
      float* out = &batch->decorated[(pillar * kMaxPoints + p) * kDecoratedFeatures];
      std::copy_n(point, kRawFeatures, out);
      out[5] = point[0] - mean_x;
      out[6] = point[1] - mean_y;
      out[7] = point[2] - mean_z;
      out[8] = point[0] - center_x;
      out[9] = point[1] - center_y;
    }
  }
  return true;
}

bool BuildPillarsFromSingleFrame(const std::string& point_path,
                                 PillarBatch* batch, std::string* error) {
  // File wrapper: read N x 5 float32 then reuse the in-memory pipeline.
  std::vector<float> raw;
  if (!ReadPoints(point_path, &raw, error)) return false;
  return BuildPillarsFromPoints(raw.data(), raw.size() / 5, batch, error);
}

}  // namespace centerpoint