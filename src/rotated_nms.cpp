#include <topsbot_centerpoint/centerpoint_postprocess.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <vector>

namespace centerpoint {
namespace {
struct Point { double x, y; };
using Polygon = std::vector<Point>;

std::array<Point, 4> Corners(const Detection& b) {
  const double c = std::cos(b.yaw), s = std::sin(b.yaw);
  const double hx = b.length * 0.5, hy = b.width * 0.5;
  const std::array<Point, 4> local{{{-hx, -hy}, {hx, -hy}, {hx, hy}, {-hx, hy}}};
  std::array<Point, 4> result{};
  for (std::size_t i = 0; i < 4; ++i)
    result[i] = {b.x + local[i].x * c - local[i].y * s,
                 b.y + local[i].x * s + local[i].y * c};
  return result;
}

double Cross(Point a, Point b, Point c) {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

Point Intersection(Point p, Point q, Point a, Point b) {
  const double rx = q.x - p.x, ry = q.y - p.y;
  const double sx = b.x - a.x, sy = b.y - a.y;
  const double denominator = rx * sy - ry * sx;
  if (std::abs(denominator) < 1e-12) return q;
  const double t = ((a.x - p.x) * sy - (a.y - p.y) * sx) / denominator;
  return {p.x + t * rx, p.y + t * ry};
}

Polygon Clip(Polygon subject, const std::array<Point, 4>& clip) {
  for (std::size_t edge = 0; edge < 4 && !subject.empty(); ++edge) {
    const Point a = clip[edge], b = clip[(edge + 1) % 4];
    Polygon input = std::move(subject); subject.clear();
    Point previous = input.back();
    bool previous_inside = Cross(a, b, previous) >= -1e-9;
    for (const Point current : input) {
      const bool current_inside = Cross(a, b, current) >= -1e-9;
      if (current_inside != previous_inside) subject.push_back(Intersection(previous, current, a, b));
      if (current_inside) subject.push_back(current);
      previous = current; previous_inside = current_inside;
    }
  }
  return subject;
}

double Area(const Polygon& polygon) {
  double twice = 0.0;
  for (std::size_t i = 0; i < polygon.size(); ++i) {
    const Point& a = polygon[i]; const Point& b = polygon[(i + 1) % polygon.size()];
    twice += a.x * b.y - a.y * b.x;
  }
  return std::abs(twice) * 0.5;
}

double RotatedIou(const Detection& a, const Detection& b) {
  if (a.length <= 0 || a.width <= 0 || b.length <= 0 || b.width <= 0) return 0.0;
  const auto ca = Corners(a), cb = Corners(b);
  const double intersection = Area(Clip(Polygon(ca.begin(), ca.end()), cb));
  const double union_area = a.length * a.width + b.length * b.width - intersection;
  return union_area > 0.0 ? intersection / union_area : 0.0;
}
}  // namespace

std::vector<std::size_t> RotatedNms(const std::vector<Detection>& boxes,
                                    float iou_threshold, std::size_t max_output) {
  std::vector<std::size_t> order(boxes.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return boxes[a].score > boxes[b].score;
  });
  std::vector<std::size_t> keep;
  for (std::size_t index : order) {
    bool suppressed = false;
    for (std::size_t prior : keep) {
      if (RotatedIou(boxes[index], boxes[prior]) > iou_threshold) { suppressed = true; break; }
    }
    if (!suppressed) {
      keep.push_back(index);
      if (keep.size() == max_output) break;
    }
  }
  return keep;
}

std::vector<std::size_t> AabbNms(const std::vector<Detection>& boxes,
                                 float iou_threshold, std::size_t max_output) {
  const std::size_t n = boxes.size();
  std::vector<std::size_t> keep;
  if (n == 0) return keep;

  // Score-descending visit order, exactly axera's nms_bev.
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return boxes[a].score > boxes[b].score;
  });

  // Precompute AABB bounds in world coordinates: length along x, width along y.
  std::vector<float> x1(n), y1(n), x2(n), y2(n), area(n);
  for (std::size_t i = 0; i < n; ++i) {
    const auto& det = boxes[i];
    x1[i] = det.x - det.length * 0.5f;
    y1[i] = det.y - det.width * 0.5f;
    x2[i] = det.x + det.length * 0.5f;
    y2[i] = det.y + det.width * 0.5f;
    area[i] = det.width * det.length;
  }

  std::vector<bool> suppressed(n, false);
  for (std::size_t rank = 0; rank < n; ++rank) {
    const std::size_t i = order[rank];
    if (suppressed[i]) continue;
    keep.push_back(i);
    if (keep.size() == max_output) break;
    for (std::size_t next = rank + 1; next < n; ++next) {
      const std::size_t j = order[next];
      if (suppressed[j]) continue;
      const float ix1 = std::max(x1[i], x1[j]);
      const float iy1 = std::max(y1[i], y1[j]);
      const float ix2 = std::min(x2[i], x2[j]);
      const float iy2 = std::min(y2[i], y2[j]);
      const float iw = std::max(0.0f, ix2 - ix1);
      const float ih = std::max(0.0f, iy2 - iy1);
      const float inter = iw * ih;
      const float union_area = area[i] + area[j] - inter;
      const float iou = inter / std::max(union_area, 1e-6f);
      if (iou > iou_threshold) suppressed[j] = true;
    }
  }
  return keep;
}

}  // namespace centerpoint