// Copyright 2026 TOPSBOT
// SPDX-License-Identifier: Apache-2.0

// Shared fp16 codec, extracted verbatim from runtime_model.cpp so the CPU
// scatter can emit already-encoded half-precision pseudo-BEV buffers (the
// fp16 pass-through path) with bit-identical results to the old in-Run()
// conversion.

#ifndef TOPSBOT_CENTERPOINT_FP16_UTILS_HPP_
#define TOPSBOT_CENTERPOINT_FP16_UTILS_HPP_

#include <cstdint>
#include <cstring>

namespace centerpoint {

inline uint16_t F32ToF16(float value) {
  // NOTE: the previous bit-trick ((x & 0x7f800000) - 0x38000000) >> 13
  // underflows for zero and subnormal inputs: 0.0f was converted to 0x4000
  // (fp16 2.0), which corrupts the sparse pseudo-BEV tensor fed to the RPN.
  // Handle zero/subnormal/overflow explicitly with round-to-nearest-even.
  uint32_t x = 0;
  std::memcpy(&x, &value, sizeof(x));
  const uint16_t sign = static_cast<uint16_t>((x >> 16) & 0x8000u);
  const uint32_t raw_exp = (x >> 23) & 0xFFu;
  const uint32_t mant = x & 0x7FFFFFu;
  if (raw_exp == 0xFFu) return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0u));
  const int32_t exp = static_cast<int32_t>(raw_exp) - 127 + 15;
  if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);  // overflow -> inf
  if (exp <= 0) {
    if (exp < -10) return sign;  // underflow -> signed zero
    const uint32_t shifted = mant | 0x800000u;
    const uint32_t shift = static_cast<uint32_t>(14 - exp);
    uint16_t half = static_cast<uint16_t>(shifted >> shift);
    const uint32_t remainder = shifted & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1u);
    if (remainder > halfway || (remainder == halfway && (half & 1u))) ++half;
    return static_cast<uint16_t>(sign | half);
  }
  uint16_t half = static_cast<uint16_t>((static_cast<uint32_t>(exp) << 10) | (mant >> 13));
  const uint32_t remainder = mant & 0x1FFFu;
  if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u))) ++half;  // carry into exponent is correct
  return static_cast<uint16_t>(sign | half);
}

inline float F16ToF32(uint16_t h) {
  uint32_t sign = (h >> 15) & 0x1;
  uint32_t exponent = (h >> 10) & 0x1F;
  uint32_t mantissa = h & 0x3FF;
  uint32_t f;
  if (exponent == 0) {
    if (mantissa == 0) {
      f = sign << 31;
    } else {
      uint32_t e = 127 - 15 + 1;
      while ((mantissa & 0x400) == 0) {
        mantissa <<= 1;
        --e;
      }
      mantissa &= 0x3FF;
      f = (sign << 31) | (e << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    f = (sign << 31) | 0x7F800000 | (mantissa << 13);
  } else {
    f = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  }
  float out;
  std::memcpy(&out, &f, sizeof(out));
  return out;
}

}  // namespace centerpoint

#endif  // TOPSBOT_CENTERPOINT_FP16_UTILS_HPP_
