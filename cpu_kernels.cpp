
static inline uint8_t rqt_fp8_stochastic_code(float value, uint32_t& rng) {
  if (!std::isfinite(value)) return value < 0.0f ? 0xFE : 0x7E;
  const bool negative = value < 0.0f;
  const float magnitude = std::abs(value);
  if (magnitude >= 448.0f) return negative ? 0xFE : 0x7E;
  if (magnitude == 0.0f) return 0;

  uint8_t lower, upper;
  float lo, hi;
  if (magnitude < 0.015625f) {
    const float scaled = magnitude * 512.0f;
    const int base = (int)std::floor(scaled);
    lower = (uint8_t)std::min(base, 7);
    upper = lower < 7 ? (uint8_t)(lower + 1) : 8;
    lo = rqt_fp8_level(lower);
    hi = rqt_fp8_level(upper);
  } else {
    const uint32_t bits = std::bit_cast<uint32_t>(magnitude);
    const int exponent = (int)((bits >> 23) & 0xFF) - 127;
    const int exponent_field = exponent + 7;
    const float step = std::ldexp(1.0f, exponent - 3);
    const float base_value = std::ldexp(1.0f, exponent);
    const int mantissa = (int)std::floor((magnitude - base_value) / step);
    lower = (uint8_t)((exponent_field << 3) | std::min(mantissa, 7));
    lo = rqt_fp8_level(lower);
    if (mantissa < 7 || exponent_field == 15) {
      upper = lower;
      hi = lo;
      if (lo < magnitude && lower < 0x7E) {
        upper = (uint8_t)(lower + 1);
        hi = rqt_fp8_level(upper);
      }
    } else {
      upper = (uint8_t)((exponent_field + 1) << 3);
      hi = rqt_fp8_level(upper);
    }
  }

  if (lo == hi || lo == magnitude) return negative ? (uint8_t)(lower | 0x80) : lower;
  const float p = (magnitude - lo) / (hi - lo);
  const float u = (float)(rqt_xorshift(rng) & 0x00FFFFFFu) / 16777216.0f;
  const uint8_t code = u < p ? upper : lower;
  return negative ? (uint8_t)(code | 0x80) : code;
}

static inline uint8_t rqt_code(const uint8_t* packed, int index, int bits) {
  if (bits == 4) { const uint8_t p = packed[index >> 1]; return (index & 1) ? p & 15 : p >> 4; }
  if (bits == 8) return packed[index];
  const uint8_t* p = packed + (index >> 2) * 3;
  switch (index & 3) { case 0: return p[0] >> 2; case 1: return ((p[0] & 3) << 4) | (p[1] >> 4); case 2: return ((p[1] & 15) << 2) | (p[2] >> 6); default: return p[2] & 63; }
}

static inline void rqt_set_code(uint8_t* packed, int index, int bits, uint8_t code) {