#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <ta-runtime-api.h>

namespace centerpoint {

struct InputView {
  const void* data = nullptr;
  std::size_t bytes = 0;
  bool is_float32 = true;
  bool is_float16 = false;
  // Bytes are already encoded for this input's data_format/quantization
  // (produced via quantize_scalar): Run() memcpy's them without conversion.
  bool prequantized = false;
};

class RuntimeModel {
 public:
  RuntimeModel() = default;
  ~RuntimeModel();
  RuntimeModel(const RuntimeModel&) = delete;
  RuntimeModel& operator=(const RuntimeModel&) = delete;

  bool Load(const std::string& path, uint32_t core = 0);
  bool Run(const float* input, std::size_t elements);
  bool Run(const std::vector<InputView>& inputs);
  void Reset();
  const std::vector<taconn_inout_attr_t>& inputs() const { return input_attrs_; }
  const std::vector<taconn_inout_attr_t>& outputs() const { return output_attrs_; }

  // int8/uint8 boundary helpers so producers can emit already-quantized
  // bytes (CPU scatter fast path) instead of a full fp32 tensor.
  bool input_is_int8(std::size_t index) const;
  bool input_is_fp16(std::size_t index) const;
  bool output_is_int8(std::size_t index) const;
  bool output_is_fp16(std::size_t index) const;
  uint8_t zero_point_byte(std::size_t index) const;
  int quantize_scalar(std::size_t index, float value) const;
  const float* output(std::size_t index) const;
  const void* output_raw(std::size_t index) const;
  std::size_t output_elements(std::size_t index) const;

 private:
  void FreeInputBuffers();

  ta_runtime_context context_{};
  bool loaded_ = false;
  std::vector<taconn_input_t> inputs_;
  std::vector<void*> input_buffers_;
  std::vector<taconn_inout_attr_t> input_attrs_;
  std::vector<taconn_inout_attr_t> output_attrs_;
  std::vector<taconn_buffer_t> output_buffers_;
  std::vector<std::vector<float>> output_fp32_;
  std::string model_path_;
};

}  // namespace centerpoint