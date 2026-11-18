#pragma once

#include <cstddef>
#include <vector>

namespace centerpoint {

struct Detection {
  float x, y, z, length, width, height, yaw, vx, vy, score;
  int label;
  int task;
};

// Produces at most 1000 score-sorted candidates per CenterHead task.
std::vector<Detection> DecodeCenterHead(const float* const* task_outputs,
                                        const std::size_t* task_channels,
                                        float score_threshold = 0.2f,
                                        std::size_t pre_max_per_task = 1000);

std::vector<std::size_t> RotatedNms(const std::vector<Detection>& boxes,
                                    float iou_threshold = 0.2f,
                                    std::size_t max_output = 83);

// Axis-aligned BEV NMS matching axera's nms_bev: ignore yaw and treat each box
// as length(x) x width(y), suppress by AABB IoU. AABB IoU >= rotated IoU, so
// this keeps strictly fewer detections than RotatedNms and is much cheaper.
std::vector<std::size_t> AabbNms(const std::vector<Detection>& boxes,
                                 float iou_threshold = 0.2f,
                                 std::size_t max_output = 500);

}  // namespace centerpoint