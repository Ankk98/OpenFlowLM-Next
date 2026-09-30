// The adapter between a MODEL DIRECTORY and a `npue::dec::Decider`.
//
// Four things have to be resolved before the Decider can be constructed, and
// each of them is a place where a wrong value produces a model that answers:
//
//   the .npue container, which may be beside the checkpoint or already built;
//   the design set, which is a GEOMETRY and is found by the same two-tier rule
//   the embedding adapter uses (a model-local copy wins);
//   the checkpoint's SUBDIRECTORIES, because laya nests them and a packer
//     pointed at the served root would refuse;
//   the readout, which is refused by name when this build does not implement it.
#include "AutoDecisionModel/npue_decision.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "decision_engine.hpp"
#include "npue.hpp"
#include "npue_encoder.hpp"
#include "npue_pack.hpp"
#include "utils/utils.hpp"

namespace {

namespace fs = std::filesystem;

std::string read_text(const fs::path &p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) return {};
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

/// The design set. IDENTICAL resolution to the embedding adapter's
/// `find_artifacts`, and duplicated rather than shared because that one is in
/// an anonymous namespace inside npue_embedding.cpp.
///
/// The duplication is the lesser evil against the alternative, which is
/// exporting a function from a .cpp to make two callers share a dozen lines.
/// If a THIRD caller appears, the shared function is the right answer and this
/// comment is the marker for it.
std::string find_artifacts(const fs::path &dir,
                           const nlohmann::ordered_json &info) {
  auto has_design = [](const fs::path &d) {
    std::error_code ec;
    return fs::is_regular_file(d / "gemm_rtp" / "design.json", ec);
  };
  const fs::path local = dir / "npue_designs";
  if (has_design(local)) return local.string();
  std::string prefix;
  try {
    prefix = utils::find_xclbin_path();
  } catch (const std::exception &) {
    prefix.clear();
  }
  if (!prefix.empty() && info.contains("npue_design_family") &&
      info["npue_design_family"].is_string()) {
    const fs::path cand =
        fs::path(prefix) / "xclbins" /
        info["npue_design_family"].get<std::string>();
    if (has_design(cand)) return cand.string();
  }
  throw std::runtime_error(
      "NpueDecision: no design set for this model. Looked for " +
      local.string() +
      "/gemm_rtp/design.json and, if the model entry names "
      "\"npue_design_family\", for that family under the installed xclbin "
      "tree. This model's is BERT-h768-gated-i1152-bf16; see "
      "npu_offload/gemm_rtp/README.md for the command that builds it.");
}

/// The container: one beside the checkpoint, or pack one.
///
/// Packing here rather than leaving it to the caller is what makes a first run
/// work at all, and it is the same policy the embedding adapter follows: tens
/// of seconds once, then an mmap.
std::string prepare_container(const fs::path &dir,
                              const nlohmann::ordered_json &info) {
  std::vector<fs::path> found;
  for (const auto &e : fs::directory_iterator(dir)) {
    if (e.path().extension() == ".npue") found.push_back(e.path());
  }
  // TWO containers are two different datapaths (bf16 vs bfp16) or two different
  // sequence lengths, and they are not interchangeable. Picking by sort order
  // is how upstream's tasks/0104 nearly shipped the wrong datapath.
  if (found.size() == 1) return found.front().string();
  if (found.size() > 1) {
    std::string names;
    for (const auto &f : found) names += (names.empty() ? "" : ", ") + f.filename().string();
    throw std::runtime_error(
        "NpueDecision: " + dir.string() + " holds several .npue containers (" +
        names +
        ") and they are not interchangeable -- they differ in datapath or "
        "sequence length. Remove the one this model does not use.");
  }

  // THE SUBDIRECTORIES. A repository that NESTS its files rather than laying
  // them out flat needs all three named, and none of them can be guessed: the
  // config is at a different depth from the weights, and the tokenizer at a
  // third. Pointed at the served root with no keys, the packer refuses with a
  // message about a missing config -- which is correct and unhelpful, because
  // the config IS there, two levels down.
  // The packer's `config_subdir` is RELATIVE TO THE CHECKPOINT ROOT, while the
  // entry's `config_subdir` -- the one LM_Config and the downloader use -- is
  // relative to the SERVED directory. The checkpoint subdir is therefore
  // prepended here, and getting that wrong is a refusal from the packer rather
  // than a wrong model, which is the good kind of wrong.
  auto sub = [&](const char *key) -> std::string {
    if (info.contains(key) && info[key].is_string())
      return info[key].get<std::string>();
    return "";
  };
  const std::string ckpt = sub("npue_checkpoint_subdir");
  std::string cfg = sub("npue_config_subdir");
  if (cfg.empty()) cfg = sub("config_subdir");
  const std::string tok = sub("npue_tokenizer_subdir");
  if (!fs::is_regular_file(dir / "model.safetensors") && ckpt.empty())
    throw std::runtime_error(
        "NpueDecision: " + dir.string() +
        " has neither a .npue container nor a model.safetensors to pack one "
        "from. If this repository NESTS its files, the model entry must name "
        "npue_checkpoint_subdir, npue_config_subdir and npue_tokenizer_subdir "
        "-- the config, the weights and the tokenizer are at three different "
        "depths in laya's tree and none of them can be guessed.");

  npue::PrepareOptions po;
  po.checkpoint_dir = dir.string();
  po.out_path = (dir / (dir.filename().string() + ".npue")).string();
  if (info.contains("npue_source_repo") && info["npue_source_repo"].is_string())
    po.source_repo = info["npue_source_repo"].get<std::string>();
  else if (info.contains("url") && info["url"].is_string()) {
    const std::string url = info["url"].get<std::string>();
    const std::string host = "huggingface.co/";
    const size_t i = url.find(host);
    if (i != std::string::npos) po.source_repo = url.substr(i + host.size());
  }
  po.checkpoint_subdir = ckpt;
  po.config_subdir = cfg;
  po.tokenizer_subdir = tok;
  if (info.contains("npue_tile_n") && info["npue_tile_n"].is_number_integer())
    po.tile_n = info["npue_tile_n"].get<int>();
  po.log = [](const std::string &s) { header_print("NPUE", s); };
  return npue::prepare_model_auto(po);
}

}  // namespace

struct NpueDecision::Impl {
  std::unique_ptr<npue::dec::Decider> dec;
  std::string container;
  std::string design_dir;
  // The packed per-type temperatures, in the QTYPES order. Read from the
  // CONTAINER rather than taken from the caller, because they are the
  // calibration that was fitted with the checkpoint: a caller-supplied
  // temperature is an override of these, never a replacement for them.
  std::vector<float> temperature = {1.f, 1.f, 1.f};
  std::string readout = "scored_slot";
};

NpueDecision::NpueDecision(std::string tag) : impl_(new Impl), tag_(std::move(tag)) {}
NpueDecision::~NpueDecision() = default;

void NpueDecision::load_model(const std::string& model_path,
                              const nlohmann::ordered_json& model_info,
                              int threads) {
  const fs::path dir(model_path);
  if (!fs::is_directory(dir))
    throw std::runtime_error("NpueDecision: " + model_path + " is not a directory");
  impl_->design_dir = find_artifacts(dir, model_info);
  impl_->container = prepare_container(dir, model_info);

  // The readout is read BEFORE the Decider is built, so a container this build
  // cannot serve is refused with the container's own words in the message --
  // rather than after a 220 MB weight load and an NPU dispatch.
  {
    npue::File probe(impl_->container);
    const std::string r = probe.config_string("readout");
    if (r != "scored_slot")
      throw std::runtime_error(
          "this build implements the 'scored_slot' readout and this container "
          "declares '" + r +
          "'. Refusing rather than answering with a neighbouring recipe: the "
          "answer would be well-formed and for a different question.");
    impl_->readout = r;
  }

  npue::dec::Decider::Options o;
  o.container = impl_->container;
  o.artifacts_dir = impl_->design_dir;
  o.threads = threads > 0 ? threads
                          : std::max(1u, std::thread::hardware_concurrency() / 2u);
  o.lanes = 1;
  impl_->dec = std::make_unique<npue::dec::Decider>(o);
  header_print("NPUE", "decision engine ready: " + impl_->container);
}

std::vector<decision::decision_answer> NpueDecision::decide(
    decision::decision_request& request) {
  namespace dec = npue::dec;
  using decision::decision_answer;
  std::vector<decision_answer> out;
  if (!impl_->dec)
    throw std::runtime_error("NpueDecision: decide() before load_model()");
  if (request.questions.empty()) return out;

  std::vector<dec::PromptRow> rows;
  rows.reserve(request.questions.size());
  int64_t tokens = 0;
  for (const auto &q : request.questions) {
    const int64_t qt = q.kind == decision::DECISION_KIND_SCORE   ? dec::QTYPE_SCORE
                       : q.kind == decision::DECISION_KIND_NOUL  ? dec::QTYPE_NOUL
                                                                 : dec::QTYPE_CHOICE;
    // The type TEXT upstream renders into the prompt is the question TYPE as a
    // word, not the enum's number: `"%s question: %s" % (q["t"], ins)`. A
    // number there tokenizes to a digit sequence the model has never seen in
    // that position.
    const char *type_text = qt == dec::QTYPE_SCORE  ? "score"
                            : qt == dec::QTYPE_NOUL ? "noul"
                                                    : "choice";
    std::vector<std::string> labels, criteria, noul_labels;
    for (const auto &o : q.options) {
      labels.push_back(o.label);
      criteria.push_back(o.description);
    }
    if (q.kind == decision::DECISION_KIND_NOUL) {
      noul_labels = q.noul_labels;   // empty means the default ["false","true"]
    }
    auto opts = dec::Decider::render_options(qt, labels, criteria, noul_labels);
    // Slot s shows option `option_order[s]`, upstream's convention
    // (`common.py:171`, `for i in order`). This permutes the PROMPT, not just
    // the answer, and that distinction is the whole feature: a bidirectional
    // encoder's option markers attend to each other, so where an option sits
    // changes what the model computes for it. Permuting only the readout would
    // look like it worked and measure nothing.
    opts = decision::apply_option_order(opts, q.option_order);
    auto prompt = impl_->dec->build_prompt(type_text, q.instructions,
                                           request.state, opts, qt);
    if (prompt.markers.empty())
      throw std::runtime_error(
          "question '" + q.key +
          "' produced no markers: the head budget left no room for its options. "
          "The option texts are too long for head_max_len, and upstream "
          "collapses a question to a single marker rather than answering it "
          "wrongly -- so this is refused by name rather than answered with one "
          "uninformative probability.");
    dec::PromptRow r;
    r.ids = std::move(prompt.ids);
    r.markers = std::move(prompt.markers);
    r.qtype = qt;
    r.option_order = q.option_order;
    tokens += static_cast<int64_t>(r.ids.size());
    rows.push_back(std::move(r));
  }
  request.input_tokens = tokens;

  // The temperature. The container's PER-TYPE values are the default, because
  // they are the calibration fitted with this checkpoint; a caller-supplied one
  // REPLACES them for all three types at once. An override that applied to some
  // types and not others would leave one batch internally inconsistent, and
  // there is no schema field for a per-type one to be inconsistent about.
  //
  // The flag is `temperature_overridden` and not the value, because 1.0 is both
  // "the caller asked for a neutral scale" and "the caller asked for nothing".
  //
  // ORDER IS THE WHOLE FIX. The container is read FIRST and the caller's
  // override applied SECOND. It was the other way round, and the override was
  // therefore dead: `t.assign(3, ...)` wrote the caller's value and the
  // container block below overwrote all three of them unconditionally. With
  // this checkpoint's `[1.0, 1.0, 1.0]` a `--decisiontemperature 3.0` run
  // returned BYTE-IDENTICAL probabilities to the default, and nothing about the
  // output looked wrong -- 1.0 is the neutral scale, so a discarded override
  // produces exactly the answer a working neutral scale would.
  std::vector<float> t = impl_->temperature;
  {
    npue::File probe(impl_->container);
    const std::string raw = probe.config_string("temperature");
    // "temperature" is packed as a JSON array of three floats, read through
    // config_string as raw JSON text.
    const npue::json::Value v = npue::json::parse(raw);
    size_t i = 0;
    for (const auto& e : v.as_array()) {
      if (i >= 3) break;
      t[i] = static_cast<float>(e.as_number());
      ++i;
    }
    if (i != 3)
      throw std::runtime_error(
          "the container's temperature has " + std::to_string(i) +
          " entries, expected 3 in the QTYPES order [choice, score, noul]. "
          "Upstream refuses a different length for the same reason: it indexes "
          "the list by question type, so a short one is an IndexError on the "
          "first score question.");
    // The container's own values get the same [0.5, 5.0] bound the caller's do.
    // Upstream applies that bound when it USES a temperature, and a checkpoint
    // that ships 0.1006 for choice:11+ is exactly the case the bound exists for.
    for (int k = 0; k < 3; ++k)
      if (!(t[static_cast<size_t>(k)] >= 0.5f && t[static_cast<size_t>(k)] <= 5.0f))
        throw std::runtime_error(
            "the container's temperature[" + std::to_string(k) + "]=" +
            std::to_string(t[static_cast<size_t>(k)]) +
            " is outside [0.5, 5.0]. A fitted temperature below 1 sharpens the "
            "logits rather than softening them, and upstream's own shipped "
            "choice:11+ value of 0.1006 multiplies them ~10x -- a 0.24 top "
            "probability published as 0.99, so a caller gating on confidence is "
            "told a coin flip is a certainty. Refused by name rather than "
            "applied.");
  }
  if (request.temperature_overridden) {
    if (!(request.temperature >= 0.5 && request.temperature <= 5.0))
      throw std::runtime_error(
          "the request's temperature override is " +
          std::to_string(request.temperature) +
          ", outside the pinned schema's [0.5, 5.0]. parse_request already "
          "refuses that range; this is the check that survives a caller that "
          "set the flag by hand without going through the parser.");
    t.assign(3, static_cast<float>(request.temperature));
  }

  auto raw = impl_->dec->decide(rows, t, request.temperature_overridden);
  if (raw.size() != request.questions.size())
    throw std::runtime_error(
        "the engine returned " + std::to_string(raw.size()) + " rows for " +
        std::to_string(request.questions.size()) +
        " questions. A shorter vector would be zipped away and lose an answer, "
        "so it is caught here where the cause is still visible.");

  out.reserve(raw.size());
  for (size_t i = 0; i < raw.size(); ++i) {
    const auto &q = request.questions[i];
    decision_answer a;
    a.kind = q.kind;
    const auto &probs = raw[i].probabilities;
    if (q.kind == decision::DECISION_KIND_NOUL) {
      // [false, true] -- the semantic order, which is why the second slot is
      // the boolean. A one-option noul cannot happen: render_options refuses
      // two labels and the schema fixes it at two.
      a.noul = probs.size() >= 2 ? probs[1] : 0.f;
    } else if (q.kind == decision::DECISION_KIND_CHOICE) {
      const int64_t j = raw[i].argmax;
      if (j < 0 || j >= static_cast<int64_t>(q.options.size()))
        throw std::runtime_error(
            "question '" + q.key + "' produced an argmax of " +
            std::to_string(j) + " for " +
            std::to_string(q.options.size()) +
            " options. The option list and the marker count are the same "
            "number by construction, so this is the readout and the prompt "
            "disagreeing.");
      a.choice = q.options[static_cast<size_t>(j)].label;
      a.confidence = raw[i].confidence;
    } else {
      a.score = raw[i].score;
      a.confidence = raw[i].confidence;
    }
    a.probabilities = probs;
    a.logits = raw[i].logits;
    out.push_back(std::move(a));
  }
  return out;
}

std::string NpueDecision::name() const { return tag_; }
std::string NpueDecision::readout() const { return impl_->readout; }

std::vector<std::string> NpueDecision::supported_kinds() const {
  return {"noul", "choice", "score"};
}

