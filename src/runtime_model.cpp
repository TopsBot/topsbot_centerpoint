#include <topsbot_centerpoint/runtime_model.hpp>
#include <topsbot_centerpoint/fp16_utils.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace centerpoint {
namespace {
std::size_t Elements(const taconn_inout_attr_t& attr) {
  std::size_t result = 1;
  for (unsigned i = 0; i < attr.dim_count; ++i) result *= attr.dim_size[i];
  return result;
}

std::size_t ElementSize(const taconn_inout_attr_t& attr) {
  switch (attr.data_format) {
    case TACONN_DATA_FORMAT_FP32:
    case TACONN_DATA_FORMAT_INT32:
    case TACONN_DATA_FORMAT_UINT32: return 4;
    case TACONN_DATA_FORMAT_FP64:
    case TACONN_DATA_FORMAT_INT64:
    case TACONN_DATA_FORMAT_UINT64: return 8;
    case TACONN_DATA_FORMAT_FP16:
    case TACONN_DATA_FORMAT_INT16:
    case TACONN_DATA_FORMAT_UINT16:
    case TACONN_DATA_FORMAT_BFP16: return 2;
    case TACONN_DATA_FORMAT_INT8:
    case TACONN_DATA_FORMAT_UINT8:
    case TACONN_DATA_FORMAT_CHAR: return 1;
    default: return 4;
  }
}

// int8/uint8 boundary support: quantize fp32 inputs and dequantize integer
// outputs with the per-tensor metadata reported by the loaded NB
// (ASYMMETRIC: scale + zero point; DFP: fixed-point position).
double QuantizeScale(float value, const taconn_inout_attr_t& attr) {
  if (attr.quant_format == TACONN_QNT_TYPE_ASYMMETRIC) {
    const float scale = attr.quant_data.affine.tf_scale;
    const double raw = (scale > 0.0f) ? static_cast<double>(value) / scale
                                      : static_cast<double>(value);
    return raw + attr.quant_data.affine.tf_zero_point;
  }
  if (attr.quant_format == TACONN_QNT_TYPE_DFP) {
    return std::ldexp(static_cast<double>(value), attr.quant_data.dfp.fixed_point_pos);
  }
  return static_cast<double>(value);
}

[[maybe_unused]] float DequantizeValue(int64_t q, const taconn_inout_attr_t& attr) {
  if (attr.quant_format == TACONN_QNT_TYPE_ASYMMETRIC) {
    const double raw = static_cast<double>(q - attr.quant_data.affine.tf_zero_point);
    return static_cast<float>(raw * attr.quant_data.affine.tf_scale);
  }
  if (attr.quant_format == TACONN_QNT_TYPE_DFP) {
    return static_cast<float>(std::ldexp(static_cast<double>(q),
                                         -attr.quant_data.dfp.fixed_point_pos));
  }
  return static_cast<float>(q);
}

// Hot path note: this runs over the full pseudo-BEV tensor (64*512*512 =
// 16.7M elements) every frame. The original double + nearbyint(libm) loop
// cost ~850 ms on the board and made int8 slower than fp16; keep the loop
// float-only and exploit that ~91% of the tensor is exact zero, which
// always quantizes to the constant zero-point byte.
void QuantizeFloats(const float* source, std::size_t elements,
                    const taconn_inout_attr_t& attr, void* target) {
  const bool uint8 = (attr.data_format == TACONN_DATA_FORMAT_UINT8);
  const int hi = uint8 ? 255 : 127;
  const int lo = uint8 ? 0 : -128;
  unsigned char* out = static_cast<unsigned char*>(target);
  // signed-char round trip keeps the byte pattern for uint8 values >= 128.
  const auto pack = [](int v) { return static_cast<unsigned char>(static_cast<signed char>(v)); };

  if (attr.quant_format == TACONN_QNT_TYPE_ASYMMETRIC) {
    const float scale = attr.quant_data.affine.tf_scale;
    const int zp = attr.quant_data.affine.tf_zero_point;
    const float inv = (scale > 0.0f) ? 1.0f / scale : 1.0f;
    const float raw_lo = static_cast<float>(lo - zp);
    const float raw_hi = static_cast<float>(hi - zp);
    const unsigned char zp_byte = pack(zp);
    for (std::size_t k = 0; k < elements; ++k) {
      const float x = source[k];
      if (x == 0.0f) { out[k] = zp_byte; continue; }
      float raw = x * inv;
      if (!(raw == raw)) { out[k] = zp_byte; continue; }  // NaN -> zero point
      raw = raw < raw_lo ? raw_lo : (raw > raw_hi ? raw_hi : raw);
      out[k] = pack(static_cast<int>(std::lrintf(raw)) + zp);
    }
    return;
  }

  if (attr.quant_format == TACONN_QNT_TYPE_DFP) {
    const float mult = std::ldexp(1.0f, attr.quant_data.dfp.fixed_point_pos);
    const float raw_lo = static_cast<float>(lo);
    const float raw_hi = static_cast<float>(hi);
    for (std::size_t k = 0; k < elements; ++k) {
      const float x = source[k];
      if (x == 0.0f) { out[k] = 0; continue; }
      float raw = x * mult;
      if (!(raw == raw)) { out[k] = 0; continue; }  // NaN -> 0 (dfp zero point is 0)
      raw = raw < raw_lo ? raw_lo : (raw > raw_hi ? raw_hi : raw);
      out[k] = pack(static_cast<int>(std::lrintf(raw)));
    }
    return;
  }

  // Fallback for NONE or unexpected quant formats: generic per-element path.
  for (std::size_t k = 0; k < elements; ++k) {
    double q = QuantizeScale(source[k], attr);
    if (!(q == q)) q = 0.0;
    const long r = std::lround(q);
    const int clamped = static_cast<int>(r < lo ? lo : (r > hi ? hi : r));
    out[k] = pack(clamped);
  }
}

void DequantizeInt8(const void* source, std::size_t elements,
                    const taconn_inout_attr_t& attr, float* target) {
  const bool uint8 = (attr.data_format == TACONN_DATA_FORMAT_UINT8);
  const unsigned char* in = static_cast<const unsigned char*>(source);
  const auto unpack = [uint8, in](std::size_t k) {
    return uint8 ? static_cast<int>(in[k])
                 : static_cast<int>(static_cast<signed char>(in[k]));
  };
  if (attr.quant_format == TACONN_QNT_TYPE_ASYMMETRIC) {
    const float scale = attr.quant_data.affine.tf_scale;
    const int zp = attr.quant_data.affine.tf_zero_point;
    for (std::size_t k = 0; k < elements; ++k)
      target[k] = static_cast<float>(unpack(k) - zp) * scale;
    return;
  }
  if (attr.quant_format == TACONN_QNT_TYPE_DFP) {
    const float mult = std::ldexp(1.0f, -attr.quant_data.dfp.fixed_point_pos);
    for (std::size_t k = 0; k < elements; ++k)
      target[k] = static_cast<float>(unpack(k)) * mult;
    return;
  }
  for (std::size_t k = 0; k < elements; ++k) target[k] = static_cast<float>(unpack(k));
}

const char* DataFormatName(unsigned fmt) {
  switch (fmt) {
    case TACONN_DATA_FORMAT_FP32: return "fp32";
    case TACONN_DATA_FORMAT_FP16: return "fp16";
    case TACONN_DATA_FORMAT_BFP16: return "bfp16";
    case TACONN_DATA_FORMAT_INT8: return "int8";
    case TACONN_DATA_FORMAT_UINT8: return "uint8";
    case TACONN_DATA_FORMAT_INT32: return "int32";
    case TACONN_DATA_FORMAT_INT64: return "int64";
    default: return "other";
  }
}

const char* QntName(unsigned q) {
  switch (q) {
    case TACONN_QNT_TYPE_NONE: return "none";
    case TACONN_QNT_TYPE_DFP: return "dfp";
    case TACONN_QNT_TYPE_ASYMMETRIC: return "asym";
    default: return "unknown";
  }
}

bool ConvertIntegers(const void* source, std::size_t elements, unsigned data_format, void* target) {
  for (std::size_t i = 0; i < elements; ++i) {
    const int64_t value = static_cast<const int64_t*>(source)[i];
    switch (data_format) {
      case TACONN_DATA_FORMAT_INT8:   static_cast<int8_t*>(target)[i]   = static_cast<int8_t>(value);   break;
      case TACONN_DATA_FORMAT_UINT8:  static_cast<uint8_t*>(target)[i]  = static_cast<uint8_t>(value);  break;
      case TACONN_DATA_FORMAT_INT16:  static_cast<int16_t*>(target)[i]  = static_cast<int16_t>(value);  break;
      case TACONN_DATA_FORMAT_UINT16: static_cast<uint16_t*>(target)[i] = static_cast<uint16_t>(value); break;
      case TACONN_DATA_FORMAT_INT32:  static_cast<int32_t*>(target)[i]  = static_cast<int32_t>(value);  break;
      case TACONN_DATA_FORMAT_UINT32: static_cast<uint32_t*>(target)[i] = static_cast<uint32_t>(value); break;
      case TACONN_DATA_FORMAT_INT64:
      case TACONN_DATA_FORMAT_UINT64:
      case TACONN_DATA_FORMAT_FP64:   static_cast<int64_t*>(target)[i]  = value; break;
      case TACONN_DATA_FORMAT_FP32:   static_cast<float*>(target)[i]    = static_cast<float>(value);    break;
      case TACONN_DATA_FORMAT_FP16:
      case TACONN_DATA_FORMAT_BFP16:
        static_cast<uint16_t*>(target)[i] = F32ToF16(static_cast<float>(value)); break;
      default: return false;
    }
  }
  return true;
}
}  // namespace

bool RuntimeModel::input_is_int8(std::size_t index) const {
  if (index >= input_attrs_.size()) return false;
  const unsigned f = input_attrs_[index].data_format;
  return f == TACONN_DATA_FORMAT_INT8 || f == TACONN_DATA_FORMAT_UINT8;
}

bool RuntimeModel::input_is_fp16(std::size_t index) const {
  if (index >= input_attrs_.size()) return false;
  const unsigned f = input_attrs_[index].data_format;
  return f == TACONN_DATA_FORMAT_FP16 || f == TACONN_DATA_FORMAT_BFP16;
}

bool RuntimeModel::output_is_int8(std::size_t index) const {
  if (index >= output_attrs_.size()) return false;
  const unsigned f = output_attrs_[index].data_format;
  return f == TACONN_DATA_FORMAT_INT8 || f == TACONN_DATA_FORMAT_UINT8;
}

bool RuntimeModel::output_is_fp16(std::size_t index) const {
  if (index >= output_attrs_.size()) return false;
  const unsigned f = output_attrs_[index].data_format;
  return f == TACONN_DATA_FORMAT_FP16 || f == TACONN_DATA_FORMAT_BFP16;
}

uint8_t RuntimeModel::zero_point_byte(std::size_t index) const {
  const taconn_inout_attr_t& a = input_attrs_[index];
  const bool u8 = (a.data_format == TACONN_DATA_FORMAT_UINT8);
  const int lo = u8 ? 0 : -128;
  const int hi = u8 ? 255 : 127;
  int q = 0;
  if (a.quant_format == TACONN_QNT_TYPE_ASYMMETRIC) q = a.quant_data.affine.tf_zero_point;
  q = q < lo ? lo : (q > hi ? hi : q);
  return static_cast<uint8_t>(static_cast<signed char>(q));
}

int RuntimeModel::quantize_scalar(std::size_t index, float value) const {
  const taconn_inout_attr_t& a = input_attrs_[index];
  const bool u8 = (a.data_format == TACONN_DATA_FORMAT_UINT8);
  const int lo = u8 ? 0 : -128;
  const int hi = u8 ? 255 : 127;
  int q;
  if (a.quant_format == TACONN_QNT_TYPE_ASYMMETRIC) {
    const float s = a.quant_data.affine.tf_scale;
    const int zp = a.quant_data.affine.tf_zero_point;
    if (value == 0.0f) q = zp;
    else {
      const float inv = (s > 0.0f) ? 1.0f / s : 1.0f;
      float raw = value * inv;
      if (!(raw == raw)) raw = 0.0f;  // NaN -> zero point
      const float rlo = static_cast<float>(lo - zp);
      const float rhi = static_cast<float>(hi - zp);
      raw = raw < rlo ? rlo : (raw > rhi ? rhi : raw);
      q = static_cast<int>(std::lrintf(raw)) + zp;
    }
  } else if (a.quant_format == TACONN_QNT_TYPE_DFP) {
    const float m = std::ldexp(1.0f, a.quant_data.dfp.fixed_point_pos);
    float raw = value * m;
    if (!(raw == raw)) raw = 0.0f;
    const float rlo = static_cast<float>(lo);
    const float rhi = static_cast<float>(hi);
    raw = raw < rlo ? rlo : (raw > rhi ? rhi : raw);
    q = static_cast<int>(std::lrintf(raw));
  } else {
    q = static_cast<int>(std::lrintf(value));
  }
  return q < lo ? lo : (q > hi ? hi : q);
}

RuntimeModel::~RuntimeModel() { Reset(); }

void RuntimeModel::FreeInputBuffers() {
  for (void* buffer : input_buffers_) if (buffer != nullptr) std::free(buffer);
  input_buffers_.clear();
}

bool RuntimeModel::Load(const std::string& path, uint32_t core) {
  Reset();
  if (ta_runtime_load_model_from_file(&context_, path.c_str(), core) != NNRT_SUCCESS) {
    std::cerr << "ta_runtime_load_model_from_file failed: " << path << "\n"; return false;
  }
  loaded_ = true;
  auto fail = [this](const char* operation) { std::cerr << operation << " failed\n"; Reset(); return false; };
  taconn_input_output_num_t count{};
  if (ta_runtime_query(&context_, TACONN_QUERY_IN_OUT_NUM, &count) != NNRT_SUCCESS ||
      count.input_num == 0 || count.output_num == 0)
    return fail("query input/output count");
  input_attrs_.resize(count.input_num);
  output_attrs_.resize(count.output_num);
  output_buffers_.resize(count.output_num);
  for (uint32_t i = 0; i < count.input_num; ++i) {
    input_attrs_[i] = {}; input_attrs_[i].index = i;
    if (ta_runtime_query(&context_, TACONN_QUERY_INPUT_ATTR, &input_attrs_[i]) != NNRT_SUCCESS)
      return fail("query input attribute");
  }
  for (uint32_t i = 0; i < count.output_num; ++i) {
    output_attrs_[i] = {}; output_attrs_[i].index = i;
    if (ta_runtime_query(&context_, TACONN_QUERY_OUTPUT_ATTR, &output_attrs_[i]) != NNRT_SUCCESS)
      return fail("query output attribute");
    const std::size_t bytes = Elements(output_attrs_[i]) * ElementSize(output_attrs_[i]);
    if (ta_runtime_create_buffer(&context_, static_cast<uint32_t>(bytes), &output_buffers_[i]) != NNRT_SUCCESS)
      return fail("create output buffer");
  }
  if (ta_runtime_set_output(&context_, count.output_num, output_buffers_.data()) != NNRT_SUCCESS)
    return fail("ta_runtime_set_output");
  model_path_ = path;
  {
    const char* env = std::getenv("CENTERPOINT_IO_DEBUG");
    const bool verbose = env && env[0] != '\0' && env[0] != '0';
    const auto report = [](const char* kind, const taconn_inout_attr_t& a) {
      std::cout << "  [centerpoint-io] " << kind << " " << a.index << " " << a.name
                << " fmt=" << DataFormatName(a.data_format) << " qnt=" << QntName(a.quant_format);
      if (a.quant_format == TACONN_QNT_TYPE_ASYMMETRIC)
        std::cout << " scale=" << a.quant_data.affine.tf_scale
                  << " zp=" << a.quant_data.affine.tf_zero_point;
      else if (a.quant_format == TACONN_QNT_TYPE_DFP)
        std::cout << " fp=" << a.quant_data.dfp.fixed_point_pos;
      std::cout << "\n";
    };
    std::cout << "[centerpoint-io] model " << path << "\n";
    for (const auto& a : input_attrs_) report("in", a);
    for (std::size_t i = 0; i < output_attrs_.size(); ++i) {
      const bool same = output_attrs_[i].data_format == output_attrs_[0].data_format &&
                        output_attrs_[i].quant_format == output_attrs_[0].quant_format;
      if (verbose || i == 0 || !same) report("out", output_attrs_[i]);
    }
  }
  return true;
}

namespace {
uint64_t HashBytes(const void* data, std::size_t bytes) {
  const unsigned char* p = static_cast<const unsigned char*>(data);
  uint64_t h = 1469598103934665603ull;
  for (std::size_t k = 0; k < bytes; k += 4001) { h ^= p[k]; h *= 1099511628211ull; }
  return h;
}
bool ProbeEnabled() {
  static const bool on = [] { const char* e = std::getenv("CENTERPOINT_PROBE"); return e && e[0] == '1'; }();
  return on;
}
}  // namespace

bool RuntimeModel::Run(const std::vector<InputView>& views) {
  // Scratch buffers persist across frames on purpose: freeing and
  // re-allocating 32MiB every frame makes glibc munmap/remap fresh physical
  // pages at the same VA, which is the suspect for the NPU re-reading stale
  // input bytes (frozen fp16 output). Allocation happens once per model.
  if (!loaded_ || views.size() != input_attrs_.size()) return false;
  if (input_buffers_.size() != views.size()) {
    FreeInputBuffers();
    input_buffers_.resize(views.size(), nullptr);
  }

  inputs_.assign(views.size(), {});
  bool ok = true;
  for (std::size_t i = 0; i < views.size() && ok; ++i) {
    const taconn_inout_attr_t& attr = input_attrs_[i];
    const std::size_t elements = Elements(attr);
    std::size_t source_bytes;
    if (views[i].prequantized)     source_bytes = elements * ElementSize(attr);
    else if (views[i].is_float16)  source_bytes = elements * sizeof(uint16_t);
    else if (views[i].is_float32)  source_bytes = elements * sizeof(float);
    else                           source_bytes = elements * sizeof(int64_t);
    if (views[i].data == nullptr || views[i].bytes < source_bytes) return false;

    const std::size_t bytes = elements * ElementSize(attr);
    void* aligned = input_buffers_[i];
    if (aligned == nullptr) {
      if (posix_memalign(&aligned, 256, bytes) != 0) return false;
      input_buffers_[i] = aligned;
    }
    if (views[i].prequantized) {
      if (views[i].bytes != bytes) { return false; }
      std::memcpy(aligned, views[i].data, bytes);
    } else if (views[i].is_float16) {
      const uint16_t* half = static_cast<const uint16_t*>(views[i].data);
      if (attr.data_format == TACONN_DATA_FORMAT_FP16 ||
          attr.data_format == TACONN_DATA_FORMAT_BFP16) {
        std::memcpy(aligned, half, elements * sizeof(uint16_t));
      } else if (attr.data_format == TACONN_DATA_FORMAT_FP32) {
        float* target = static_cast<float*>(aligned);
        for (std::size_t k = 0; k < elements; ++k) target[k] = F16ToF32(half[k]);
      } else if (attr.data_format == TACONN_DATA_FORMAT_INT8 ||
                 attr.data_format == TACONN_DATA_FORMAT_UINT8) {
        std::vector<float> tmp(elements);
        for (std::size_t k = 0; k < elements; ++k) tmp[k] = F16ToF32(half[k]);
        QuantizeFloats(tmp.data(), elements, attr, aligned);
      } else {
        return false;
      }
    } else if (views[i].is_float32) {
      if (attr.data_format == TACONN_DATA_FORMAT_FP16 ||
          attr.data_format == TACONN_DATA_FORMAT_BFP16) {
        uint16_t* half = static_cast<uint16_t*>(aligned);
        const float* source = static_cast<const float*>(views[i].data);
        for (std::size_t k = 0; k < elements; ++k) half[k] = F32ToF16(source[k]);
      } else if (attr.data_format == TACONN_DATA_FORMAT_FP32) {
        std::memcpy(aligned, views[i].data, bytes);
      } else if (attr.data_format == TACONN_DATA_FORMAT_INT8 ||
                 attr.data_format == TACONN_DATA_FORMAT_UINT8) {
        QuantizeFloats(static_cast<const float*>(views[i].data), elements, attr, aligned);
      } else {
        return false;
      }
    } else if (!ConvertIntegers(views[i].data, elements, attr.data_format, aligned)) {
      return false;
    }
    taconn_input_t input{};
    input.index = static_cast<uint32_t>(i);
    input.data = aligned;
    input.size = static_cast<uint32_t>(bytes);
    inputs_[i] = input;
  }
  if (ProbeEnabled()) {
    for (std::size_t i = 0; i < input_buffers_.size(); ++i)
      std::cout << "[cp-probe] " << model_path_ << " submitted in" << i
                << " bytes=" << inputs_[i].size
                << " hash=" << std::hex << HashBytes(input_buffers_[i], inputs_[i].size) << std::dec << "\n";
  }

  ok = ta_runtime_set_input_cva(&context_, static_cast<uint32_t>(inputs_.size()), inputs_.data()) ==
           NNRT_SUCCESS &&
       ta_runtime_run_network(&context_) == NNRT_SUCCESS;
  if (ok) {
    for (auto& buffer : output_buffers_)
      if (ta_runtime_invalidate_buffer(&context_, &buffer) != NNRT_SUCCESS) ok = false;
  }
  if (ProbeEnabled() && ok) {
    for (std::size_t i = 0; i < output_buffers_.size(); ++i)
      std::cout << "[cp-probe] " << model_path_ << " raw out" << i
                << " hash=" << std::hex << HashBytes(output_buffers_[i].data, Elements(output_attrs_[i]) * ElementSize(output_attrs_[i]))
                << std::dec << "\n";
  }

  if (ok) {
    output_fp32_.assign(output_buffers_.size(), {});
    for (std::size_t i = 0; i < output_buffers_.size(); ++i) {
      const taconn_inout_attr_t& attr = output_attrs_[i];
      const std::size_t elements = Elements(attr);
      switch (attr.data_format) {
        case TACONN_DATA_FORMAT_FP16:
        case TACONN_DATA_FORMAT_BFP16: {
          output_fp32_[i].resize(elements);
          const uint16_t* half = static_cast<const uint16_t*>(output_buffers_[i].data);
          for (std::size_t k = 0; k < elements; ++k) output_fp32_[i][k] = F16ToF32(half[k]);
          break;
        }
        case TACONN_DATA_FORMAT_INT8:
        case TACONN_DATA_FORMAT_UINT8: {
          output_fp32_[i].resize(elements);
          DequantizeInt8(output_buffers_[i].data, elements, attr, output_fp32_[i].data());
          break;
        }
        default: break;  // fp32/int32 pass through via output()
      }
    }
  }
  return ok;
}

bool RuntimeModel::Run(const float* source, std::size_t elements) {
  if (!loaded_ || input_attrs_.size() != 1) return false;
  if (Elements(input_attrs_[0]) != elements) return false;
  InputView view;
  view.data = source;
  view.bytes = elements * sizeof(float);
  view.is_float32 = true;
  return Run(std::vector<InputView>{view});
}

const void* RuntimeModel::output_raw(std::size_t index) const {
  if (index >= output_buffers_.size()) return nullptr;
  return output_buffers_[index].data;
}

const float* RuntimeModel::output(std::size_t index) const {
  if (index < output_fp32_.size() && !output_fp32_[index].empty()) return output_fp32_[index].data();
  return static_cast<const float*>(output_buffers_.at(index).data);
}
std::size_t RuntimeModel::output_elements(std::size_t index) const {
  return Elements(output_attrs_.at(index));
}

void RuntimeModel::Reset() {
  if (!loaded_) return;
  for (auto& buffer : output_buffers_) if (buffer.data != nullptr) ta_runtime_destroy_buffer(&context_, &buffer);
  ta_runtime_destroy_context(&context_);
  loaded_ = false; context_ = {}; FreeInputBuffers(); inputs_.clear();
  output_buffers_.clear(); output_attrs_.clear(); input_attrs_.clear(); output_fp32_.clear();
}

}  // namespace centerpoint