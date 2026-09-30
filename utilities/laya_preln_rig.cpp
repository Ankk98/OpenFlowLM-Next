// The Phase 5 MEASUREMENT RIG: the ENGINE side of the encoder gate.
//
// Reads one text per line on stdin, frames it the way the decision engine does
// ([CLS] ... [SEP] by hand, which is why AnyTokenizer::encode() refuses this
// arch), runs the real Encoder over the real NPU designs, echoes the ids it
// used, and writes the final-norm hidden states to a .bin. The oracle in
// utilities/laya_preln_reference.py reads the ids and the checkpoint and
// produces the same array; the gate is the cosine between the two.
//
// Built out of tree, because it is a measurement and not a product:
//
//   g++ -std=c++17 -O2 -mavx2 -mfma -o rig utilities/laya_preln_rig.cpp \
//       src/open_npue/{npue_encoder,npue,npu_device,npue_pack,json_min,\
//       xlmr_tokenizer_gen,gemma_tokenizer_gen,bbpe_tokenizer_gen,\
//       tokenizer_bbpe,tokenizer_xlmr,tokenizer_gemma,tokenizer,\
//       gemma_kernels,gemma_encode}.cpp \
//       -Isrc/open_npue -I/opt/xilinx/xrt/include \
//       -L/opt/xilinx/xrt/lib64 -lxrt_coreutil
//   PATH=/opt/xilinx/xrt/bin:$PATH ./rig <container> <designs> out.bin
//       [nofuse] < prompts.txt
//
// The fourth argument, `nofuse`, turns OFF ffn epilogue fusion so the two
// gated-activation copies can be compared byte for byte -- which is how the
// three-copies-must-agree property in Phase 5 is actually checked rather than
// asserted.
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "npue_encoder.hpp"

int main(int argc, char **argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: rig <container> <artifacts> <ids-file>\n");
    return 2;
  }
  try {
    npue::File m(argv[1]);
    npue::enc::ShapeLease lease(m);
    auto tok = npue::enc::load_tokenizer(m, argv[1]);
    const float *w_word = m.raw("embeddings.word").as<float>();
    const float *w_pos = m.raw("embeddings.position").as<float>();
    const float *w_typ = m.raw("embeddings.token_type").as<float>();

    npue::npu::Device dev;
    npue::enc::StackOptions so;
    so.threads = 8;
    so.lanes = 1;
    so.fuse_ffn_epilogue = !(argc > 4 && std::string(argv[4]) == "nofuse");
    npue::enc::Stack stack(dev, m, argv[2], so);
    npue::enc::Encoder &e = *stack.lead;
    // The design's streams are sized for its BATCH TIERS (4/16/32 rows), and
    // `rows` has to be one of them: a rows=1 encode makes the GEMM write 32
    // rows into a 1-row buffer, which is a segfault rather than an error.
    // Rows 1..3 are entirely pad, and the additive mask zeroes their
    // contribution, so comparing row 0 alone is sound.
    const int64_t BT = e.use_tier(1);
    std::printf("# datapath %s\n", stack.p_qkv->info().emulate_bfp16
                                     ? "bfp16-emulated" : "bf16");
    std::printf("# hidden %lld seq %lld layers %lld heads %lld inter %lld\n",
                (long long)npue::enc::g_hidden, (long long)npue::enc::g_seq,
                (long long)npue::enc::g_layers, (long long)npue::enc::g_heads,
                (long long)npue::enc::g_ffn);

    // One TEXT per line on stdin, encoded with add_special=false and framed
    // [CLS] ... [SEP] by hand -- which is what the decision engine does, and
    // the reason AnyTokenizer::encode() refuses this arch.
    std::vector<int32_t> ids;
    const int32_t cls = tok.bbpe->cls_id, sep = tok.bbpe->sep_id;
    ids.push_back(cls);
    std::string line;
    while (std::getline(std::cin, line)) {
      if (line.empty()) continue;
      auto en = tok.bbpe->encode(line, static_cast<int>(npue::enc::g_seq) - 2);
      // The encode pads to max_len, so the tail is pad ids; take en.n_tokens
      // and let the framing below supply the real length.
      for (int64_t i = 0; i < en.n_tokens; ++i) ids.push_back(en.input_ids[i]);
      ids.push_back(sep);
    }
    while (ids.size() > 2 && ids.back() == tok.bbpe->pad_id) ids.pop_back();
    if (ids.size() > static_cast<size_t>(npue::enc::g_seq)) {
      std::fprintf(stderr, "REFUSED: %zu ids exceed seq %lld\n", ids.size(),
                   (long long)npue::enc::g_seq);
      return 1;
    }
    const int64_t S = npue::enc::g_seq, H = npue::enc::g_hidden;
    const int32_t pad_id = tok.bbpe ? tok.bbpe->pad_id : 0;
    std::vector<float> buf(static_cast<size_t>(BT) * S * H, 0.f);
    std::vector<float> cmask(static_cast<size_t>(BT) * S, -1.0e30f);
    for (int64_t b = 0; b < BT; ++b) {
      const bool real = (b == 0);
      for (int64_t s = 0; s < S; ++s) {
        const bool keep = real && s < static_cast<int64_t>(ids.size());
        const int32_t id = keep ? ids[static_cast<size_t>(s)] : pad_id;
        cmask[static_cast<size_t>(b) * S + s] = keep ? 0.f : -1.0e30f;
        float *dst = buf.data() + (b * S + s) * H;
        const float *wv = w_word + static_cast<size_t>(id) * H;
        const float *pv = w_pos + static_cast<size_t>(s) * H;
        for (int64_t c = 0; c < H; ++c) dst[c] = wv[c] + pv[c] + w_typ[c];
      }
    }
    // Echo the ids so the oracle is given the SAME input rather than
    // re-tokenizing: the tokenizer was gated in Phase 2 against 418 exact
    // HuggingFace strings, and re-deriving the ids here would test it a
    // second time in a place that cannot report which of the two disagreed.
    for (size_t i = 0; i < ids.size(); ++i)
      std::printf("%d\n", ids[i]);
    std::printf("# n_ids %zu\n", ids.size());
    e.add_mask = cmask;
    auto h = e.run_dispatch(buf);
    std::fprintf(stderr, "ran %lld rows x %lld (tier %lld)\n", (long long)S,
                 (long long)H, (long long)BT);
    std::ofstream o(argv[3], std::ios::binary);
    o.write(reinterpret_cast<const char *>(h.data()),
            static_cast<std::streamsize>(h.size() * sizeof(float)));
    return o ? 0 : 1;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "REFUSED: %s\n", e.what());
    return 1;
  }
}
