// `oflm decide <tag> --input-file <f.json>` -- the CLI surface for a decision
// model, and the FIRST of the two Phase 8 surfaces. It is here, and the server
// route is written second, so that the route's only untested part is HTTP: a
// route that wraps an unverified engine is one more layer between a bug and
// the person who has to find it.
//
// The request file is a `SystemOneRequest` body -- the same JSON the route
// accepts -- so the CLI and the server cannot drift in what they parse, and a
// request that works on one works on the other. That is the whole reason the
// plan asks for the CLI to be gated against the reference BEFORE the route
// exists.
#ifndef OPENFLOWLM_DECISION_CLI_H
#define OPENFLOWLM_DECISION_CLI_H

#include <unistd.h>

#include <memory>

#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "AutoDecisionModel/decision_types.hpp"
#include "AutoDecisionModel/all_decision_model.hpp"
#include "program_args.hpp"
#include "utils/utils.hpp"

namespace decision_cli {

/// The model directory for `tag`, and the refusal when there is none.
///
/// The models root is `utils::get_models_directory()` -- the same call
/// `oflm pull` writes into -- so a tag resolves to the one directory the
/// downloader filled. Resolution does NOT go through the supported-models
/// registry: a decision model's entry names a design family and a set of
/// subdirectories, and the CLI needs the DIRECTORY, not the entry. The
/// registry is consulted inside the adapter, which is where those keys are read.
inline std::filesystem::path model_dir(const std::string &tag) {
  // The DECISION registry's own completion, not the embedding registry's. The
  // two are separate on purpose: borrowing `complete_simple_embedding_tag`
  // here would let an embedding alias name a decision model, which is the same
  // class of substitution this file's header is about.
  const std::string full = complete_simple_decision_tag(tag);
  const std::filesystem::path root = utils::get_models_directory();
  for (const std::string &name : {tag, full}) {
    if (name.empty()) continue;
    const std::filesystem::path p = root / name;
    if (std::filesystem::is_directory(p)) return p;
  }
  throw std::runtime_error(
      "no such decision model: '" + tag + "'. Looked for " +
      (root / tag).string() + " and " + (root / full).string() +
      ". A tag that names a chat or embedding model is a different kind of "
      "model and is refused rather than loaded and asked to decide -- it would "
      "answer with fluent text, which is the worst version of a wrong answer.");
}

/// Run the CLI. Returns a process exit code, and prints the response to stdout.
///
/// `--input-file` is `-` for stdin. `--json` prints the response body and
/// nothing else, which is what makes this scriptable; the human form is a
/// per-question summary because a five-question request's JSON is unreadable
/// and the JSON is still one flag away.
/// `entry` is the model_list.json entry for `tag`, read by the caller through
/// the SAME registry every other command uses. It is passed in rather than
/// re-read here because it is the registry's parse of model_list.json, and a
/// second parse is a second answer to "what does this tag mean".
inline int run(const program_args_t &args,
               const nlohmann::ordered_json& entry) {
  try {
    // --decisionmodel is the ENGINE to load; the positional tag is the MODEL
    // to ask. They are different axes and conflating them is how a request ends
    // up answered by the wrong model -- so supplying both differently is a
    // refusal, not a preference.
    const std::string tag =
        args.model_tag.empty() ? args.decision_model : args.model_tag;
    if (!args.decision_model.empty() && !args.model_tag.empty() &&
        args.decision_model != args.model_tag)
      throw std::runtime_error(
          "--decisionmodel '" + args.decision_model + "' and the model tag '" +
          args.model_tag +
          "' name different models. One of them would be ignored, and the "
          "request would be answered by whichever the code happened to read "
          "first.");
    if (tag.empty())
      throw std::runtime_error(
          "no model named. `oflm decide <tag> --input-file <f.json>`, or "
          "--decisionmodel <tag>.");

    std::string body;
    if (args.input_file_name == "-") {
      std::stringstream ss;
      ss << std::cin.rdbuf();
      body = ss.str();
    } else {
      std::ifstream f(args.input_file_name, std::ios::binary);
      if (!f)
        throw std::runtime_error("cannot open " + args.input_file_name);
      std::stringstream ss;
      ss << f.rdbuf();
      body = ss.str();
    }
    if (body.find_first_not_of(" \t\r\n") == std::string::npos)
      throw std::runtime_error(
          args.input_file_name + " is empty. A request needs at least `state` "
          "and one question, and an empty body would answer nothing while "
          "looking like a successful run.");

    // Parsed with the NON-throwing overload, so a malformed body produces a
    // message naming the FILE instead of a byte offset into a stream nobody
    // is looking at.
    const auto parsed = nlohmann::ordered_json::parse(body, nullptr, false);
    if (parsed.is_discarded())
      throw std::runtime_error(
          args.input_file_name + " is not JSON");

    // The entry is REQUIRED, not optional, and the reason is concrete: a model
    // that NESTS its files names the three subdirectories in its entry, and a
    // decision model with an empty entry cannot pack itself -- the packer is
    // pointed at a served root, looks for config.json, and refuses. Which is a
    // correct refusal of an incomplete request.
    if (entry.is_null() || entry.empty())
      throw std::runtime_error(
          "no model_list.json entry for '" + tag +
          "'. A decision model is registered there, and the entry is where the "
          "design family and the checkpoint's subdirectory keys live.");
    // `--json` MUST EMIT THE RESPONSE AND NOTHING ELSE, and the loader's status
    // lines go to STDOUT: the Stack constructor prints the design, the datapath,
    // the toolchain and the weight staging on the way in, and a first run packs
    // a container while printing more. Piping `oflm decide --json` into a JSON
    // parser therefore failed on line 1.
    //
    // The status lines stay on stdout in the rest of the tree on purpose --
    // npue_encoder.hpp's own comment records that verify_tail.py and
    // verify_embed_e2e.py SCRAPE them -- so this does not move them. Instead fd 1
    // is pointed at fd 2 for the DURATION OF THE LOAD and restored after, which
    // keeps stdout a clean document without changing what anything else sees.
    struct RestoreStdout {
      int fd;
      ~RestoreStdout() {
        if (fd < 0) return;
        std::cout.flush();
        ::dup2(fd, STDOUT_FILENO);
        ::close(fd);
      }
    };
    std::unique_ptr<AutoDecisionModel> model;
    {
      // The redirect is scoped to the LOAD and no further. A guard living until
      // the end of the function would restore stdout AFTER the response was
      // written, and the response is the one thing that must reach the real
      // stdout -- which is the failure this was added to fix.
      const int saved = args.json_output ? ::dup(STDOUT_FILENO) : -1;
      if (saved >= 0) {
        std::cout.flush();
        ::dup2(STDERR_FILENO, STDOUT_FILENO);
      }
      RestoreStdout restore{saved};
      model = get_auto_decision_model(tag, model_dir(tag), &entry,
                                      args.decision_threads);
    }
    if (!model)
      throw std::runtime_error("the decision model did not load");
    // A temperature OVERRIDE for the whole request, and 1.0 is what "unset"
    // resolves to -- NOT 0.0. parse_request refuses anything outside the pinned
    // schema's [0.5, 5.0], so a 0.0 sentinel for "use the model's own" is
    // refused by the layer below, which is a refusal that points at the caller
    // for something the caller never typed.
    //
    // 1.0 is the right default because it is a NEUTRAL scale: the engine then
    // divides the logits by 1.0 and uses the container's own per-type
    // temperatures, which are the ones fitted with the checkpoint. A caller who
    // wants the model's calibration asks for nothing.
    const double temp = args.decision_temperature < 0.f
                            ? 1.0
                            : static_cast<double>(args.decision_temperature);
    auto req = decision::parse_request(parsed, temp);
    // Set HERE and not in parse_request, because parse_request cannot know
    // whether its `temperature` argument came from the caller or from this
    // surface's own default. Inferring the override from the value would make
    // an explicit `--decisiontemperature 1.0` silently ignored, and would make
    // the default silently replace the checkpoint's fitted calibration.
    req.temperature_overridden = args.decision_temperature >= 0.f;
    for (const auto &d : req.dropped)
      std::fprintf(stderr,
                   "OFLM: dropped question %s: its `type` is not implemented. "
                   "The rest of the batch is served.\n", d.c_str());

    auto answers = model->decide(req);
    const auto response = decision::render_response(
        model->name(), req.questions, answers, req.input_tokens);

    if (args.json_output) {
      // stdout is the real one again: the guard above was scoped to the load.
      // The response is the only thing this block writes to it.
      std::cout << response.dump(2) << std::endl;
    } else {
      // A human form that is still checkable: the answer AND its confidence,
      // because a bare answer cannot be told from a coin flip.
      std::cout << "model  " << model->name() << std::endl;
      for (size_t i = 0; i < answers.size(); ++i) {
        const auto &a = answers[i];
        const std::string key = i < req.questions.size() ? req.questions[i].key
                                                         : std::to_string(i);
        if (a.kind == decision::DECISION_KIND_NOUL) {
          // NO confidence, and that is the schema's rule rather than a missing
          // field: `confidence` is `1 - H(p)/ln 2`, which is 1.0 for EVERY
          // two-option question and therefore carries no information. The
          // calibrated number for a noul is max(p), which is p(true) -- and it
          // is printed as p(true) so it cannot be mistaken for the degenerate
          // one. render_answer refuses to serialise a confidence on a noul at
          // all; printing it here would have been the CLI inventing the field
          // the renderer refuses.
          std::cout << key << "  noul=" << (a.noul > 0.5f ? "true" : "false")
                    << "  p(true)=" << a.probabilities.at(1) << std::endl;
        }
        else if (a.kind == decision::DECISION_KIND_CHOICE)
          std::cout << key << "  choice=" << a.choice
                    << "  confidence=" << a.confidence << std::endl;
        else
          std::cout << key << "  score=" << a.score
                    << "  confidence=" << a.confidence << std::endl;
      }
    }
    return 0;
  } catch (const decision::DecisionRequestInvalid &e) {
    std::fprintf(stderr, "OFLM: bad request: %s\n", e.what());
    return 2;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "OFLM: %s\n", e.what());
    return 1;
  }
}

}  // namespace decision_cli

#endif
