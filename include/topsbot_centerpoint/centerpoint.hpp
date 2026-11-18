#pragma once

#include <topsbot_centerpoint/centerpoint_postprocess.hpp>
#include <topsbot_centerpoint/centerpoint_preprocess.hpp>
#include <topsbot_centerpoint/runtime_model.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace centerpoint {

struct Timing {
  double load_ms = 0.0;
  double preprocess_ms = 0.0;
  double pfe_ms = 0.0;      // multi-stage: PFE chunking
  double scatter_ms = 0.0;  // multi-stage: CPU scatter
  double rpn_ms = 0.0;      // multi-stage: RPN head
  double infer_ms = 0.0;    // single-graph: one fused PFE+ScatterND+RPN run
  double postprocess_ms = 0.0;
  bool single_graph = false;  // set when timing came from the merged path

  // Model runtime as a single number, mirroring axera's "Inference" stage:
  // one fused run for the merged graph, otherwise the sum of the three stages.
  double inference_ms() const {
    return single_graph ? infer_ms : (pfe_ms + scatter_ms + rpn_ms);
  }
};

class CenterPoint {
 public:
  // Multi-stage init: pfe.nb -> rpn.nb. The scatter stage can be fused inside
  // the PFE (2-input PFE) or done on the CPU; Detect dispatches on which one
  // Load reports.
  bool Init(const std::string& pfe_model, const std::string& rpn_model, uint32_t core = 0);

  // Single-graph init: one nb that already contains PFE + ScatterND + RPN
  // (built by scripts/build_fused_graph.py + scripts/merge_fused_rpn.py).
  // Inputs: decorated_pillars [30000,20,10] fp32 + indices_input [1,30000,2] int64;
  // outputs: 36 heads (6 tasks x {reg,height,dim,rot,vel,hm}).
  bool InitMerged(const std::string& merged_model, uint32_t core = 0);

  bool Detect(const PillarBatch& pillars, std::vector<Detection>* detections,
              Timing* timing = nullptr);
  void Deinit();

  // Score gate for CenterHead candidate decoding. int8 NBs shift the heatmap
  // distribution, so re-calibrate this per model with real frames.
  void set_score_threshold(float value) { score_threshold_ = value; }

 private:
  // Shared CenterHead decode + rotated NMS over whichever model holds the 36
  // head outputs (rpn_ for multi-stage, merged_ for single-graph).
  bool DecodeHeads(RuntimeModel * head, std::vector<Detection> * detections);

  RuntimeModel pfe_;
  RuntimeModel rpn_;
  RuntimeModel merged_;
  std::vector<uint16_t> pseudo_fp16_;
  std::vector<uint8_t> pseudo_int8_;  // non-empty when the RPN wants an int8/uint8 input
  bool fused_scatter_ = false;
  bool fp16_direct_ = false;  // CPU scatter writes the fp16 pseudo-BEV directly
  bool merged_mode_ = false;  // single-graph (merged PFE+ScatterND+RPN) mode
  bool initialized_ = false;
  float score_threshold_ = 0.2F;
};

}  // namespace centerpoint