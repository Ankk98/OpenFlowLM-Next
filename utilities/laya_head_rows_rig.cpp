// Scores a DUMPED gathered-rows buffer with the engine's Head, and nothing else.
//
// The point is to take the encoder out of the comparison. Every engine-vs-oracle
// logit difference so far has been confounded by the NPU encoder: the gathered
// rows agree to 0.997-0.999 cosine, which is not enough to decide whether the
// HEAD is exact, because LayerNorm turns that cosine into a 0.55 logit shift.
// Feeding the head the SAME rows on both sides is the only measurement that
// isolates it -- and it needs no NPU, no model load, and no dispatch.
//
//   laya_head_rows_rig <container> <rows.bin> <k> [threads]
//
// <rows.bin> is fp32 [k, 768] as written by
// `laya_preln_reference.py --dump-rows` or `LAYA_DUMP_GATHERED`. Prints the raw
// logits, one per option, NOT masked and NOT temperature-scaled, because both of
// those are the caller's -- the same contract as `Head::score`.
//
// build:
//   g++ -std=c++17 -O2 -o /tmp/hrr laya_head_rows_rig.cpp decision_engine.cpp \
//       npue.cpp npu_device.cpp npue_pack.cpp json_min.cpp \
//       xlmr_tokenizer_gen.cpp gemma_tokenizer_gen.cpp bbpe_tokenizer_gen.cpp \
//       tokenizer_bbpe.cpp tokenizer_xlmr.cpp tokenizer_gemma.cpp \
//       tokenizer.cpp gemma_kernels.cpp gemma_encode.cpp npue_encoder.cpp \
//       -Isrc/open_npue -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib64 \
//       -lxrt_coreutil -Wl,-rpath,/opt/xilinx/xrt/lib64
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "npue_encoder.hpp"
#include "decision_engine.hpp"

int main(int argc, char **argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: laya_head_rows_rig <container> <rows.bin> <k> [threads]\n");
    return 2;
  }
  try {
    const int64_t k = std::stoll(argv[3]);
    npue::File m(argv[1]);
    // Two pieces of process state a Head cannot supply for itself, both of which
    // Decider gets for free from its Encoder and neither of which is obvious:
    //
    //  * `apply_model_shape` publishes hidden/heads/head_dim into the encoder's
    //    GLOBALS, and `Head`'s constructor reads them from there. Build a Head
    //    without it and hidden_ is 0, so the very first LayerNorm check fails as
    //    "head.layers.0.norm1.weight is not [0]" -- a shape complaint about a
    //    tensor that is perfectly fine, because the EXPECTED shape is the number
    //    that is wrong. That is a bad enough error to be worth a rig that
    //    cannot make it.
    //  * the arch refusal, so this rig fails on a non-arch=4 container instead
    //    of reading tensors that do not mean what the head thinks they mean.
    npue::enc::detail::apply_model_shape(m);
    if (m.config_string("arch") != "modernbert_rope_geglu")
      throw std::runtime_error("not an arch=4 container: " + m.config_string("arch"));
    npue::dec::Head H(m);
    const int64_t d = H.hidden();

    const std::string path(argv[2]);
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::vector<float> rows(static_cast<size_t>(k) * static_cast<size_t>(d));
    f.read(reinterpret_cast<char *>(rows.data()),
           static_cast<std::streamsize>(rows.size() * sizeof(float)));
    if (static_cast<size_t>(f.gcount()) != rows.size() * sizeof(float))
      throw std::runtime_error("rows.bin is short: " + path);

    const int64_t workers =
        argc > 4 ? std::stoll(argv[4])
                 : std::max(1u, std::thread::hardware_concurrency() / 2u);

    // "fwd" runs the FULL head -- the two attention layers -- over the rows and
    // prints the post-head state. It exists because `score` alone CANNOT validate
    // the head: score is the readout (LayerNorm -> Linear -> GELU -> Linear) and
    // skips the attention entirely, so a Q-scaling bug in the attention passed
    // every score-based comparison at 7 significant figures while making the
    // answers wrong. The rows are [k, d]; they are treated as [1, k, d] with all
    // k positions real, which is the head's own contract for a gathered row.
    if (std::getenv("LAYA_HEAD_FWD")) {
      std::vector<float> h = rows;                 // [1, k, d]
      std::vector<float> pad(static_cast<size_t>(k), 1.0f);
      H.forward(h, pad, 1, k, workers);
      std::printf("head state");
      for (float v : h) std::printf(" %.9g", v);
      std::printf("\n");
      return 0;
    }

    std::vector<float> logits;
    H.score(rows, 1, k, d, workers, logits);

    std::printf("head logits");
    for (float v : logits) std::printf(" %.7g", v);
    std::printf("\n");
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "laya_head_rows_rig: %s\n", e.what());
    return 1;
  }
}
