/// \file decision_types.hpp
/// \brief The SystemOne decision layer: questions in, typed answers out.
///
/// \date 2026-09-30
///
/// SPDX-License-Identifier: MIT
///
/// THIS IS NOT AN EMBEDDING MODEL AND NOT A CHAT MODEL. It is a
/// non-autoregressive readout: a state plus typed questions go in, one
/// probability distribution per question comes out, in one forward pass. There
/// is no generation to parse and therefore no generation to get wrong.
///
/// THE WIRE FORMAT IS NOT OURS. `POST /v1/systemone` is byte-compatible with
/// TypeSafe's `SystemOneRequest` / `SystemOneResponse`, pinned at
/// typesafe-ai/typesafe-sdk-python @ 0ffd094c and vendored at
/// specs/open-engine/plans/typesafe-systemone-0ffd094c.py. Being a drop-in for a
/// shipped ecosystem is worth more than an OFLM-native route name: llmgateway,
/// jev-rs, Mesh-LLM and OpenRouter all serve it, and LiteLLM is a pass-through
/// to it, so there is no second implementation free to drift from the schema --
/// only pipes.
///
/// FOUR READOUTS, ONE IMPLEMENTED. The wire format has exactly three question
/// kinds, all of them scored-slot, but a decision model as a family does not:
/// `letter_slot` (causal, distribution over label tokens) and `rank_head` (a
/// classifier on a pooled vector) are real shapes other checkpoints use. They are
/// DECLARED so that the next model of this class drops in without an API change,
/// and each REFUSES BY NAME when this build is asked to serve it. It does not
/// fall back to `scored_slot`. Routing a model to the nearest recipe "would emit
/// kernels that drop a whole stage of the layer and then report parity against a
/// replica making the same mistake" -- and the resulting answer is
/// correctly-shaped, correctly-normed and wrong, which is the one failure mode
/// nothing downstream can detect.
///
/// ORDER IS SEMANTIC, TWICE OVER, and both times it is a std::map that loses it.
/// A `choice`'s option order IS its answer space, and a `score`'s order IS its
/// scale. `nlohmann::json` is a `std::map`, so it sorts keys: a three-level score
/// emitted through it comes back alphabetised and its levels renumbered. Every
/// json type on this path is `nlohmann::ordered_json`, option order travels in an
/// array and never in an object, and the only object that reaches the wire with
/// caller order inside it is built in that order.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace decision {

/// The four readouts, declared. See the file header for why the three that are
/// not implemented refuse rather than substitute.
enum decision_readout {
    DECISION_READOUT_LETTER_SLOT = 0,   ///< [not implemented]
    DECISION_READOUT_MASKED_SLOT = 1,   ///< [not implemented]
    DECISION_READOUT_SCORED_SLOT = 2,   ///< the one this build serves
    DECISION_READOUT_RANK_HEAD = 3,     ///< [not implemented]
};

/// The three question kinds the pinned schema defines. There is no fourth; a
/// question naming one is DROPPED WITH A WARNING, not refused, because forward
/// compatibility in this ecosystem is defined as "ignore what you do not know,
/// do not fail the batch".
enum decision_kind {
    DECISION_KIND_NOUL = 0,
    DECISION_KIND_CHOICE = 1,
    DECISION_KIND_SCORE = 2,
};

/// The schema's own limits, restated here as constants so the refusals can name
/// them. `choice` 1-255 and `score` 2-10 are llmgateway's bounds, not the pinned
/// schema's -- `ScoreQuestion.criteria` there is `min_length=1` with no upper
/// bound -- so this build imposes them and says so at the refusal rather than
/// pretending a schema said something it did not.
constexpr size_t kChoiceMinOptions = 1;
constexpr size_t kChoiceMaxOptions = 255;
constexpr size_t kScoreMinLevels = 2;
constexpr size_t kScoreMaxLevels = 10;

/// A request that cannot be answered as asked. Typed so the HTTP layer can map
/// it to a 4xx WITHOUT parsing the message: re-deriving a status from prose is
/// how a "cannot" becomes a 500 and a caller retries a request that can never
/// succeed.
class DecisionRequestInvalid : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/// One rendered option. `label` is what a `choice` answer returns and a `score`
/// legend is keyed by position instead; `description` is the caller's criterion
/// text, already rendered (see render_criterion).
struct decision_option {
    std::string label;
    std::string description;
};

struct decision_question {
    decision_kind kind = DECISION_KIND_NOUL;
    /// Caller-chosen name. Never sent to the model and never used in inference --
    /// it exists so the response can be keyed. Carried here because the response
    /// is an object keyed by it.
    std::string key;
    std::string instructions;   ///< empty is legal
    /// choice: (label, description) in the CALLER's order, description may be
    ///        empty -- upstream's `crit` dict for a choice.
    /// score:  the ordered levels, index 0 == 0. The order IS the scale.
    /// noul:   exactly two, always in the semantic order [false, true], filled
    ///         by the parser from `criteria.true` / `criteria.false` because the
    ///         pinned schema makes noul criteria an OBJECT and so there is no
    ///         caller order to preserve.
    std::vector<decision_option> options;
    /// noul ONLY, and it is a documented EXTENSION: the pinned schema's
    /// `NoulQuestion` has no `labels` field at all. It is what upstream's
    /// `q["labels"]` is, and upstream raises ValueError on it for any other
    /// question type -- so supplying it on a choice or score is refused here for
    /// the same reason upstream refuses, and a conforming client that never
    /// sends it is unaffected.
    std::vector<std::string> noul_labels;   ///< 0, or 2, or a refusal
};

struct decision_answer {
    decision_kind kind = DECISION_KIND_NOUL;
    float noul = 0.f;                       ///< noul only
    std::string choice;                     ///< choice only: the argmax label
    float score = 0.f;                      ///< score only: sum(i * p_i)
    /// Length is k -- THIS ROW's marker count, not the batch's kmax, and not the
    /// number of options if the head budget truncated one. Serialised as
    /// {label: p} for choice (keyed off `options`) and {"i": p} for score, both
    /// in option order.
    std::vector<float> probabilities;
    /// choice, score only. NEVER set for noul, and serialising one is a refusal
    /// rather than a lie: `1 - H(p)/ln 2` degenerates to 1.0 for a two-option
    /// question, so the field would be a constant.
    float confidence = 0.f;
    std::vector<float> logits;              ///< raw, length k
};

/// A parsed request, plus what was dropped from it. `dropped` is not an error
/// path: the SDK's own decoder deletes an answer whose `type` is outside
/// {noul, choice, score} and logs a warning, BEFORE Pydantic validation, so a
/// client and a server can disagree about unknown kinds without either being
/// wrong. Carrying them here is how the same batch still gets served.
struct decision_request {
    std::string state;
    std::string model;                       ///< as asked for; the answer is
                                             ///< keyed by the tag that ANSWERED
    std::vector<decision_question> questions; ///< in wire order
    std::vector<std::string> dropped;         ///< keys of questions dropped
};

// ---------------------------------------------------------------------------
// the plan layer: request in, response body out. Both pure, both on the host.

///
/// Parse a `SystemOneRequest` body.
///
/// Throws DecisionRequestInvalid with a message that NAMES the offending field.
/// An unknown `type` is not thrown: it lands in `dropped`.
///
decision_request parse_request(const nlohmann::ordered_json& body,
                               double temperature);

/// The response envelope, exactly `{model, answers, usage}`.
///
/// `model` is the tag that ANSWERED, not the tag that was asked for. That is the
/// pinned schema's own wording ("may differ from the alias supplied in the
/// request") and it is the second reason the model-identity guard in the route
/// is not optional: a response echoing the asked-for name over another model's
/// answers is the worst version of a wrong answer.
///
/// `output_tokens` is always 0. That is definitional for a non-autoregressive
/// readout, not a stub: there is nothing generated to count.
nlohmann::ordered_json render_response(const std::string& answered_model,
                                       const std::vector<decision_question>& questions,
                                       const std::vector<decision_answer>& answers,
                                       int64_t input_tokens);

///
/// One answer's object, without the envelope. Exposed because a test asserting
/// "a noul answer has no confidence key" wants the answer, not the whole body,
///
/// Throws DecisionRequestInvalid if a noul answer carries a confidence, or if a
/// choice/score answer is missing its confidence -- the same shape of refusal in
/// both directions, because both are "this field is not true of this kind".
///
nlohmann::ordered_json render_answer(const decision_question& q,
                                     const decision_answer& a);

// ---------------------------------------------------------------------------
// upstream's prompt rendering, which is pure string work and belongs with the
// wire rather than with a model that needs a tokenizer to consume it.

/// One criterion value as text. A string passes through; anything structured
/// becomes compact JSON, so a rubric reads as JSON rather than as a Python repr.
///
/// The `not in (None, "")` in upstream is `not or`, and the difference is the
/// whole reason this function exists in C++ rather than being inlined at the call
/// site: `0`, `False`, `0.0`, `[]` and `{}` are falsy but MEANINGFUL criterion
/// values, and `c or default` would silently replace a criterion of `0` with an
/// English sentence. Only absent and empty-string mean "no description".
std::string render_criterion(const nlohmann::ordered_json& v);

/// The default wording when a noul side has no criterion. Two different
/// sentences for two different sides, because the semantic order is [false,
/// true] and saying "no, the statement does not hold" on the true side would
/// invert the question.
extern const char* const kNoulFalseDefault;
extern const char* const kNoulTrueDefault;

}  // namespace decision
