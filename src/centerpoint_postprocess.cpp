#include <topsbot_centerpoint/centerpoint_postprocess.hpp>

#include <algorithm>
#include <cmath>

namespace centerpoint {
namespace {
constexpr int kGrid = 128;
constexpr float kVoxel = 0.2f, kPcMin = -51.2f, kOutFactor = 4.0f;
constexpr int kTaskOffset[] = {0, 1, 3, 5, 6, 8};
}

std::vector<Detection> DecodeCenterHead(const float* const* output,
                                        const std::size_t* channels,
                                        float threshold,
                                        std::size_t pre_max) {
  std::vector<Detection> result;
  constexpr std::size_t plane = kGrid * kGrid;
  for (int task = 0; task < 6; ++task) {
    std::vector<Detection> candidates;
    const float* reg = output[task * 6], *height = output[task * 6 + 1];
    const float* dim = output[task * 6 + 2], *rot = output[task * 6 + 3];
    const float* vel = output[task * 6 + 4], *hm = output[task * 6 + 5];
    const std::size_t classes = channels[task * 6 + 5];
    // One box per cell: pick the argmax class channel, then score that winner.
    // sigmoid is monotonic, so the argmax over raw logits equals the argmax
    // over probabilities — no need to re-export the graph with a fused
    // argmax. Single-class tasks (classes == 1) keep their prior behavior.
    for (int y = 0; y < kGrid; ++y) for (int x = 0; x < kGrid; ++x) {
      const std::size_t cell = static_cast<std::size_t>(y * kGrid + x);
      std::size_t best_cls = 0;
      float best_logit = hm[cell];
      for (std::size_t cls = 1; cls < classes; ++cls) {
        const float logit = hm[cls * plane + cell];
        if (logit > best_logit) { best_logit = logit; best_cls = cls; }
      }
      const float score = 1.0f / (1.0f + std::exp(-best_logit));
      if (score < threshold) continue;
      Detection d{};
      d.x = (x + reg[cell]) * kVoxel * kOutFactor + kPcMin;
      d.y = (y + reg[plane + cell]) * kVoxel * kOutFactor + kPcMin;
      d.z = height[cell];
      if (d.x < -61.2f || d.x > 61.2f || d.y < -61.2f || d.y > 61.2f ||
          d.z < -10.0f || d.z > 10.0f) continue;
      d.length = std::exp(dim[cell]);
      d.width = std::exp(dim[plane + cell]);
      d.height = std::exp(dim[2 * plane + cell]);
      d.yaw = std::atan2(rot[cell], rot[plane + cell]);
      d.vx = vel[cell]; d.vy = vel[plane + cell]; d.score = score;
      d.label = kTaskOffset[task] + static_cast<int>(best_cls); d.task = task;
      candidates.push_back(d);
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
      return a.score > b.score;
    });
    if (candidates.size() > pre_max) candidates.resize(pre_max);
    result.insert(result.end(), candidates.begin(), candidates.end());
  }
  return result;
}

}  // namespace centerpoint