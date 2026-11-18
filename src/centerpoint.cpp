#include <topsbot_centerpoint/centerpoint.hpp>
#include <topsbot_centerpoint/fp16_utils.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>

namespace centerpoint {
namespace {
using Clock = std::chrono::steady_clock;
constexpr std::size_t kPfeBatch = 5268, kMaxPillars = 30000;
constexpr std::size_t kPoints = 20, kFeatures = 10, kChannels = 64;
constexpr std::size_t kHeight = 512, kWidth = 512, kHeadOutputs = 36;

double Milliseconds(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

// Process-wide ta-runtime reference counting. Multiple CenterPoint instances
// (one per NPU worker) may live in the same process; the runtime must be
// initialized exactly once and deinitialized only when the last one is gone.
std::mutex runtime_mutex;
std::size_t runtime_users = 0U;

bool AcquireRuntime() {
  std::lock_guard<std::mutex> lock(runtime_mutex);
  if (runtime_users == 0U && ta_runtime_init() != NNRT_SUCCESS) {
    std::cerr << "ta_runtime_init failed\n";
    return false;
  }
  ++runtime_users;
  return true;
}

void ReleaseRuntime() {
  std::lock_guard<std::mutex> lock(runtime_mutex);
  if (runtime_users == 0U) {
    return;
  }
  --runtime_users;
  if (runtime_users == 0U) {
    (void)ta_runtime_deinit();
  }
}
}  // namespace

bool CenterPoint::Init(const std::string& pfe_model, const std::string& rpn_model, uint32_t core) {
  Deinit();
  if (!AcquireRuntime()) { return false; }

  if (!pfe_.Load(pfe_model, core) || !rpn_.Load(rpn_model, core)) {
    pfe_.Reset(); rpn_.Reset(); ReleaseRuntime(); return false;
  }
  if (rpn_.inputs().size() != 1 || rpn_.outputs().size() != kHeadOutputs) {
    std::cerr << "unexpected RPN input-output count\n";
    pfe_.Reset(); rpn_.Reset(); ReleaseRuntime(); return false;
  }
  if (pfe_.inputs().size() == 2 && pfe_.outputs().size() == 1) {
    fused_scatter_ = true;
  } else if (pfe_.inputs().size() == 1 && pfe_.outputs().size() == 1) {
    fused_scatter_ = false;
  } else {
    std::cerr << "unexpected PFE input-output count\n";
    pfe_.Reset(); rpn_.Reset(); ReleaseRuntime(); return false;
  }

  if (!fused_scatter_ && rpn_.input_is_int8(0)) {
    pseudo_int8_.assign(kChannels * kHeight * kWidth, 0);
    std::cout << "centerpoint: RPN input is "
              << (rpn_.inputs()[0].data_format == TACONN_DATA_FORMAT_UINT8 ? "uint8" : "int8")
              << "; CPU scatter emits the quantized pseudo-BEV directly\n";
  }

  if (!fused_scatter_ && rpn_.input_is_fp16(0)) {
    pseudo_fp16_.assign(kChannels * kHeight * kWidth, 0);
    fp16_direct_ = true;
    std::cout << "centerpoint: RPN input is fp16; CPU scatter emits the half-precision "
                 "pseudo-BEV directly\n";
  }

  initialized_ = true;
  return true;
}

bool CenterPoint::InitMerged(const std::string& merged_model, uint32_t core) {
  Deinit();
  if (!AcquireRuntime()) { return false; }

  if (!merged_.Load(merged_model, core)) {
    merged_.Reset(); ReleaseRuntime(); return false;
  }
  // Single graph: decorated_pillars + indices_input -> 36 heads.
  if (merged_.inputs().size() != 2 || merged_.outputs().size() != kHeadOutputs) {
    std::cerr << "unexpected merged NPU input-output count\n";
    merged_.Reset(); ReleaseRuntime(); return false;
  }

  merged_mode_ = true;
  initialized_ = true;
  return true;
}

bool CenterPoint::Detect(const PillarBatch& pillars, std::vector<Detection>* detections,
                         Timing* timing) {
  if (!initialized_ || pillars.used == 0 || pillars.used > kMaxPillars ||
      pillars.decorated.size() < pillars.used * kPoints * kFeatures ||
      pillars.coordinates.size() < pillars.used * 3) return false;
  Timing local{};
  // Skip the fp32 staging tensor on the direct paths (fp16/int8 RPN input);
  // every remaining path sizes it as before (0 = empty vector).
  std::vector<float> pseudo(
      (pseudo_int8_.empty() && !fp16_direct_) ? kChannels * kHeight * kWidth : 0, 0.0f);

  if (merged_mode_) {
    // Single-graph path: PFE + ScatterND + RPN in one NB. Build the [P, 2]
    // indices inline (col 0 = batch 0; col 1 = flat BEV idx y*512 + x). The
    // fused graph carries a HW+1-row canvas, so unused pillar rows point at
    // row HW == 512*512 and get sliced away without touching the 512x512 map.
    constexpr int64_t kPadRow = kHeight * kWidth;
    std::vector<int64_t> indices(kMaxPillars * 2, kPadRow);
    for (std::size_t r = 0; r < kMaxPillars; ++r) indices[r * 2 + 0] = 0;
    for (std::size_t i = 0; i < pillars.used; ++i) {
      const int32_t y = pillars.coordinates[i * 3 + 1];
      const int32_t x = pillars.coordinates[i * 3 + 2];
      indices[i * 2 + 1] = static_cast<int64_t>(y) * kWidth + static_cast<int64_t>(x);
    }
    InputView views[2];
    views[0].data = pillars.decorated.data();
    views[0].bytes = kMaxPillars * kPoints * kFeatures * sizeof(float);
    views[0].is_float32 = true;
    views[1].data = indices.data();
    views[1].bytes = kMaxPillars * 2 * sizeof(int64_t);
    views[1].is_float32 = false;
    auto start = Clock::now();
    if (!merged_.Run(std::vector<InputView>{views[0], views[1]})) return false;
    local.infer_ms = Milliseconds(start, Clock::now());  // one fused run
    start = Clock::now();
    if (!DecodeHeads(&merged_, detections)) return false;
    local.postprocess_ms = Milliseconds(start, Clock::now());
    if (timing) {
      *timing = local;
      timing->single_graph = true;
    }
    return true;
  }

  if (fused_scatter_) {
    // Fused scatter path: 2-input PFE with indices
    std::vector<int64_t> indices(kMaxPillars * 2, 0);
    for (std::size_t i = 0; i < pillars.used; ++i) {
      const int32_t y = pillars.coordinates[i * 3 + 1];
      const int32_t x = pillars.coordinates[i * 3 + 2];
      indices[i * 2 + 0] = static_cast<int64_t>(y);
      indices[i * 2 + 1] = static_cast<int64_t>(x);
    }
    InputView views[2];
    views[0].data = pillars.decorated.data();
    views[0].bytes = kMaxPillars * kPoints * kFeatures * sizeof(float);
    views[0].is_float32 = true;
    views[1].data = indices.data();
    views[1].bytes = kMaxPillars * 2 * sizeof(int64_t);
    views[1].is_float32 = false;

    auto start = Clock::now();
    if (!pfe_.Run(std::vector<InputView>{views[0], views[1]}) ||
        pfe_.output_elements(0) != kChannels * kHeight * kWidth) return false;
    local.pfe_ms = Milliseconds(start, Clock::now());

    // output() already returns host fp32 for fp16/int8/uint8 NB boundaries;
    // the previous static_cast<float>(uint16_t) here reinterpreted raw fp16
    // bit patterns as integers and corrupted the pseudo-BEV tensor.
    std::copy_n(pfe_.output(0), kChannels * kHeight * kWidth, pseudo.data());

  } else {
    // CPU scatter path: chunked PFE + CPU loop
    std::vector<float> chunk(kPfeBatch * kPoints * kFeatures, 0.0f);
    // Background is 91% exact zeros, which encode to constant bytes (fp16
    // +0.0 == 0x0000; int8 -> the zero-point byte): one fill instead of
    // 16.7M conversions.
    if (fp16_direct_)
      std::fill(pseudo_fp16_.begin(), pseudo_fp16_.end(), uint16_t{0});
    else if (!pseudo_int8_.empty())
      std::fill(pseudo_int8_.begin(), pseudo_int8_.end(), rpn_.zero_point_byte(0));

    for (std::size_t begin = 0; begin < pillars.used; begin += kPfeBatch) {
      const std::size_t valid = std::min(kPfeBatch, pillars.used - begin);
      std::fill(chunk.begin(), chunk.end(), 0.0f);
      std::copy_n(pillars.decorated.data() + begin * kPoints * kFeatures,
                  valid * kPoints * kFeatures, chunk.data());
      auto start = Clock::now();
      if (!pfe_.Run(chunk.data(), chunk.size()) || pfe_.output_elements(0) != kPfeBatch * kChannels)
        return false;
      local.pfe_ms += Milliseconds(start, Clock::now());
      start = Clock::now();
      const float* features = pfe_.output(0);
      for (std::size_t row = 0; row < valid; ++row) {
        const int32_t y = pillars.coordinates[(begin + row) * 3 + 1];
        const int32_t x = pillars.coordinates[(begin + row) * 3 + 2];
        if (x < 0 || x >= static_cast<int32_t>(kWidth) || y < 0 || y >= static_cast<int32_t>(kHeight)) continue;
        const std::size_t base = static_cast<std::size_t>(y) * kWidth + static_cast<std::size_t>(x);
        for (std::size_t c = 0; c < kChannels; ++c) {
          const float v = features[row * kChannels + c];
          if (fp16_direct_)
            pseudo_fp16_[c * kHeight * kWidth + base] = F32ToF16(v);
          else if (!pseudo_int8_.empty())
            pseudo_int8_[c * kHeight * kWidth + base] =
                static_cast<uint8_t>(static_cast<signed char>(rpn_.quantize_scalar(0, v)));
          else
            pseudo[c * kHeight * kWidth + base] = v;
        }
      }
      local.scatter_ms += Milliseconds(start, Clock::now());
    }
  }

  auto start = Clock::now();
  if (fp16_direct_) {
    InputView view;
    view.data = pseudo_fp16_.data();
    view.bytes = pseudo_fp16_.size() * sizeof(uint16_t);
    view.is_float16 = true;
    if (!rpn_.Run(std::vector<InputView>{view})) return false;
  } else if (!pseudo_int8_.empty()) {
    InputView view;
    view.data = pseudo_int8_.data();
    view.bytes = pseudo_int8_.size();
    view.is_float32 = false;
    view.prequantized = true;
    if (!rpn_.Run(std::vector<InputView>{view})) return false;
  } else {
    if (!rpn_.Run(pseudo.data(), pseudo.size())) return false;
  }
  local.rpn_ms = Milliseconds(start, Clock::now());
  start = Clock::now();

  if (!DecodeHeads(&rpn_, detections)) return false;
  local.postprocess_ms = Milliseconds(start, Clock::now());
  if (timing) {
    timing->pfe_ms = local.pfe_ms; timing->scatter_ms = local.scatter_ms;
    timing->rpn_ms = local.rpn_ms; timing->postprocess_ms = local.postprocess_ms;
  }
  return true;
}

bool CenterPoint::DecodeHeads(RuntimeModel * head, std::vector<Detection> * detections) {
  const char* keys[] = {"reg", "height", "dim", "rot", "vel", "hm"};
  const float* ordered[kHeadOutputs]{};
  std::size_t channels[kHeadOutputs]{};
  std::vector<bool> used(head->outputs().size(), false);
  for (std::size_t task = 0; task < 6; ++task) for (std::size_t key = 0; key < 6; ++key) {
    const std::size_t target = task * 6 + key;
    const std::string expected = "task" + std::to_string(task) + "_" + keys[key];
    std::size_t source = head->outputs().size();
    for (std::size_t i = 0; i < head->outputs().size(); ++i) {
      if (used[i]) continue;
      const std::string name = head->outputs()[i].name;
      // NB export renames outputs to "attach_<name>/out0"; match both forms.
      if (name == expected || name.find(expected) != std::string::npos) { source = i; break; }
    }
    if (source >= head->outputs().size()) {
      for (std::size_t i = 0; i < head->outputs().size(); ++i)
        if (!used[i]) { source = i; break; }
    }
    if (source >= head->outputs().size() || used[source]) return false;
    used[source] = true; ordered[target] = head->output(source);
    const std::size_t elements = head->output_elements(source);
    if (elements % (128 * 128) != 0) return false;
    channels[target] = elements / (128 * 128);
  }

  const float candidate_threshold = (score_threshold_ > 0.0f) ? score_threshold_ : 0.2f;
  const auto candidates = DecodeCenterHead(ordered, channels, candidate_threshold, 1000);
  // axis-aligned global NMS across all tasks, matching axera's nms_bev + the
  // 500 max_per_sample cap. Multi-class boxes now suppress each other.
  std::vector<Detection> final_boxes;
  for (std::size_t index : AabbNms(candidates, 0.2f, 500)) {
    final_boxes.push_back(candidates[index]);
  }
  *detections = std::move(final_boxes);
  return true;
}

void CenterPoint::Deinit() {
  if (!initialized_) return;
  pfe_.Reset(); rpn_.Reset(); merged_.Reset(); ReleaseRuntime();
  fused_scatter_ = false; fp16_direct_ = false;
  merged_mode_ = false; initialized_ = false;
}

}  // namespace centerpoint