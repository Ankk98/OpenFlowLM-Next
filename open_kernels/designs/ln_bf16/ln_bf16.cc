// LayerNorm, bf16 in / bf16 out, THREE buffers, no fused residual add.
//
//   xn = x * rsqrt(mean(x^2) + eps) * w        x, w, xn all bfloat16
//
// WHY THIS EXISTS, because it is not designs/ln:
//   src/open_npue/npue_encoder.hpp's NPU LayerNorm path (layer_norm(), the
//   !host_ln branch) does exactly this and nothing more --
//     bf16_fill((uint16_t *)layernorm.host_ptr(0) + lo, x.data() + lo, ...)
//     layernorm.dispatch_only()
//     bf16_read(x.data() + lo, (const uint16_t *)layernorm.host_ptr(2) + lo, ...)
//   so the engine wants arg 0 = activation in (bf16), arg 1 = the LayerNorm
//   weight staged from the container and selected by bind(1, slot), and
//   arg 2 = activation out (bf16). The residual add stays on the host as
//   add_plain.
//
//   designs/ln is a DIFFERENT interface: x fp32, add fp32, w bf16 -> y fp32,
//   xn bf16, five buffers, with the add fused in. Dropping that into
//   art + "/layernorm" binds the wrong buffers.
//
// The math is not new. ln_nr32.cc in designs/ln is already the no-add form;
// this is the same reduction and the same scalar-multiply trick, reading and
// writing bf16 instead of fp32 halves.
//
// SPDX-License-Identifier: MIT
#include "vecmath.h"

#ifndef LN_N
#define LN_N 2048
#endif
#ifndef LN_EPS
#define LN_EPS 1e-6f
#endif

static constexpr unsigned kN = LN_N;
static constexpr unsigned kV = 32;

extern "C" {
void ln_bf16(const uint8_t *xb, const uint8_t *wb, uint8_t *xnb) {
  const bfloat16 *x = reinterpret_cast<const bfloat16 *>(xb);
  const bfloat16 *w = reinterpret_cast<const bfloat16 *>(wb);
  bfloat16 *xn = reinterpret_cast<bfloat16 *>(xnb);

  // sum(x^2) in the fp32 accumulator. aie::mac on two bf16 vectors is the
  // native bf16 MMAC, so this is the same instruction designs/ln uses -- the
  // difference there was that it had to split fp32 into halves to square it.
  accf32 ss = aie::zeros<accfloat, kV>();
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kN; j += kV) {
    const v32b xj = aie::load_v<kV>(x + j);
    ss = aie::mac(ss, xj, xj);
  }
  const float inv =
      srsqrt(aie::reduce_add(ss.template to_vector<float>()) * (1.0f / kN) + LN_EPS);

  // inv as a bf16 pair so the multiply is one vector op rather than a
  // broadcast. Same trick as designs/ln/ln.cc's ih/il.
  const bfloat16 ih = static_cast<bfloat16>(inv);
  const bfloat16 il = static_cast<bfloat16>(inv - static_cast<float>(ih));

#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kN; j += kV) {
    accf32 t = aie::zeros<accfloat, kV>();
    t = aie::mac(t, aie::load_v<kV>(x + j), aie::load_v<kV>(w + j));  // x * w
    accf32 o = aie::zeros<accfloat, kV>();
    o = mac_vs(o, t.template to_vector<float>(), ih, il);              // * inv
    aie::store_v(xn + j, o.template to_vector<bfloat16>());
  }
}
}