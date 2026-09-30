//===- npue_encoder.cpp --------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- definitions for the encoder's process-wide model geometry.
// SPDX-License-Identifier: MIT
//
// This file is small on purpose. Everything else the encoder needs is defined
// in npue_encoder.hpp, for the inlining reason recorded there; what has to
// live in exactly one translation unit is the mutable state.
//
// Everything here is a scalar with a CONSTANT initialiser, so it is
// initialised before any code runs and there is no static-initialisation
// order left in this file to get wrong. That is why the container-typed
// state (prompts, rope_inv_freq, the two names) is NOT here -- those are
// function-local statics behind accessors in the header, constructed on
// first use. tasks/0156 step 2.
//
// detail::apply_model_shape() in the header is the only writer, it runs
// once under a ShapeLease, and every value starts at 0 so that a missed
// initialisation divides by zero or allocates nothing rather than quietly
// using a stale number from a previously loaded model.

#include "npue_encoder.hpp"

namespace npue {
namespace enc {

int64_t g_seq = 0, g_hidden = 0, g_heads = 0, g_head_dim = 0;
int64_t g_ffn = 0, g_layers = 0, g_max_positions = 0;
bool g_cls_pool = false, g_l2_normalize = true;
bool g_allow_truncation = false;
std::atomic<bool> g_truncation_warned{false};
bool g_wide_lock = false;
bool g_rope = false, g_gated_ffn = false;
double g_rope_theta = 0.0;
GatedAct g_gated_act = GatedAct::Silu;
// UpFirst: the packed order for every container up to and including arch=3, and
// therefore the value that must be in force for arch=0/1/2/3. apply_model_shape()
// writes this for EVERY container and this initialiser is only what a caller
// that never went through a container sees.
//
// A default that is wrong for the new architecture would be worse than no
// default, because apply_model_shape() REFUSES a container that omits the key --
// so an arch=4 container can never reach this line. What the initialiser has to
// get right is the other five.
GateOrder g_gate_order = GateOrder::UpFirst;

// Empty for every arch that has no locality term, which is all six
// shipping encoders. band_for_layer() reads 0 out of an empty
// vector and that is the whole point: an unconditional band is not
// "a band for the new model", it is a band on bge-base, and it
// returns a correctly-sized, correctly-normed, plausible vector.
std::vector<char> g_sliding_layer;

// arch=4's three, all reset for every container that is not arch=4. Their
// zero values are the CORRECT values for the other five arches: post-LN, a real
// layer-0 norm, and no band. That is deliberate -- a default that happened to
// be harmless-but-wrong for arch=4 would be the dangerous one, and this build
// refuses any container that omits them.
bool g_preln = false;
bool g_identity_ln1_layer0 = false;
int64_t g_band_half = 0;

}  // namespace enc
}  // namespace npue
