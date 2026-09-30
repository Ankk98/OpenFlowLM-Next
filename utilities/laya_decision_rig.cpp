// The Phase 7 MEASUREMENT RIG: the ENGINE side of the decision-head gate.
//
// Reads a request JSON, runs Decider, and prints per row the ids it built, the
// marker positions, the raw logits and the probabilities. The ids matter as much
// as the logits: utilities/laya_preln_reference.py's `head()` takes them with
// --ids-file, so the two sides are provably running the SAME prompt and a
// disagreement is the head rather than the prompt builder. Comparing the two
// with independent --ids would blame the head for a prompt bug.
//
// Built out of tree, because it is a measurement and not a product:
//
//   g++ -std=c++17 -O2 -mavx512f -mavx2 -mfma -o drigr \
//       utilities/laya_decision_rig.cpp \
//       src/open_npue/{decision_engine,npue_encoder,npue,npu_device,npue_pack,
//       json_min,xlmr_tokenizer_gen,gemma_tokenizer_gen,bbpe_tokenizer_gen,
//       tokenizer_bbpe,tokenizer_xlmr,tokenizer_gemma,tokenizer,gemma_kernels,
//       gemma_encode}.cpp -Isrc/open_npue -I/opt/xilinx/xrt/include \
//       -L/opt/xilinx/xrt/lib64 -lxrt_coreutil
//   PATH=/opt/xilinx/xrt/bin:$PATH ./drigr <container> <designs> req.json
//
// The request is {state, temperature?, questions:[{t, ins, options, criteria,
// noul_labels?}]} -- labels and criteria SEPARATE, because render_options is
// what turns them into option text and moving that format into the rig would
// hide the part that decides whether the model reads the slot correctly.
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include "decision_engine.hpp"
#include "json_min.hpp"

int main(int argc, char **argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: drigr <container> <designs> <request.json>\n");
    return 2;
  }
  try {
    std::ifstream f(argv[3]);
    std::stringstream ss;
    ss << f.rdbuf();
    const auto body = npue::json::parse(ss.str());

    npue::dec::Decider::Options o;
    o.container = argv[1];
    o.artifacts_dir = argv[2];
    o.threads = std::max(1u, std::thread::hardware_concurrency() / 2u);
    o.lanes = 1;
    npue::dec::Decider D(o);

    const auto *st = body.find("state");
    const std::string state = st ? st->as_string() : std::string();
    std::vector<float> temperature = {1.f, 1.f, 1.f};
    if (const auto *tp = body.find("temperature")) {
      if (!tp->is_array() || tp->as_array().size() != 3)
        throw std::runtime_error(
            "temperature must be a 3-vector in the QTYPES order "
            "[choice, score, noul]. Upstream refuses a different length for the "
            "same reason: _decode_answers indexes it by question type, so a "
            "shorter list is an IndexError on the first score question.");
      for (size_t i = 0; i < 3; ++i)
        temperature[i] = static_cast<float>(tp->as_array()[i].as_number());
    }

    std::vector<npue::dec::PromptRow> rows;
    const auto &qs = body.at("questions");
    for (const auto &q : qs.as_array()) {
      npue::dec::PromptRow r;
      const auto *tp = q.find("t");
      const std::string type = tp ? tp->as_string() : std::string("choice");
      r.qtype = type == "score" ? 1 : (type == "noul" ? 2 : 0);
      // labels + criteria, kept SEPARATE and rendered by the engine's
      // render_options. A single `options` array of finished strings would
      // move the label format out of the engine and into the rig, and the label
      // is the part that decides whether the model reads the slot correctly.
      std::vector<std::string> labels, criteria, noul_labels;
      if (const auto *v = q.find("options"))
        for (const auto &x : v->as_array()) labels.push_back(x.as_string());
      if (const auto *v = q.find("criteria"))
        for (const auto &x : v->as_array()) criteria.push_back(x.as_string());
      if (const auto *v = q.find("noul_labels"))
        for (const auto &x : v->as_array()) noul_labels.push_back(x.as_string());
      if (labels.empty())
        throw std::runtime_error(
            "question has no `options` labels. An empty option list makes the "
            "row's k zero, which the readout refuses rather than answering.");
      auto opts = npue::dec::Decider::render_options(r.qtype, labels, criteria,
                                                    noul_labels);
      const auto *ip = q.find("ins");
      auto p = D.build_prompt(type, ip ? ip->as_string() : std::string(), state,
                              opts, r.qtype);
      r.ids = p.ids;
      r.markers = p.markers;
      rows.push_back(r);
    }
    auto answers = D.decide(rows, temperature);
    for (size_t i = 0; i < answers.size(); ++i) {
      std::printf("ROW %zu k %lld argmax %lld conf %.6f score %.6f\n", i,
                  (long long)answers[i].k, (long long)answers[i].argmax,
                  answers[i].confidence, answers[i].score);
      std::printf("  ids");
      for (int32_t v : rows[i].ids) std::printf(" %d", v);
      std::printf("\n  markers");
      for (int64_t v : rows[i].markers) std::printf(" %lld", (long long)v);
      std::printf("\n  logits");
      for (float v : answers[i].logits) std::printf(" %.7g", v);
      std::printf("\n  probs");
      for (float v : answers[i].probabilities) std::printf(" %.7g", v);
      std::printf("\n");
    }
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "REFUSED: %s\n", e.what());
    return 1;
  }
}
