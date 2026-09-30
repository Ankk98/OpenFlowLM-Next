/// \file auto_decision_model.hpp
/// \brief The decision-model interface
/// \note A SIBLING of AutoEmbeddingModel, not a subclass. A decision model
///       answers typed questions with a distribution and generates no tokens,
///       so it has no context length, no sampler and no token stream. Sharing
///       that base would mean every one of those is virtual-and-ignored, which
///       is the shape a caller stops checking.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "AutoDecisionModel/decision_types.hpp"

/// A model that answers typed questions in one forward pass.
class AutoDecisionModel {
 public:
  virtual ~AutoDecisionModel() = default;

  /// Load from a model directory. Throws with a message naming what is missing
  /// -- never loads "something" and answers with it.
  virtual void load_model(const std::string& model_path,
                          const nlohmann::ordered_json& model_info,
                          int threads) = 0;

  /// One answer per question IN THE REQUEST'S ORDER, including for a question
  /// this build does not implement: an unimplemented kind is a dropped
  /// question, and it is DROPPED from the request before it reaches here, so
  /// the two vectors are the same length by construction. A returned vector
  /// that is shorter than the request's questions is a bug in the model, and
  /// the caller checks it rather than zipping the two and losing a row.
  virtual std::vector<decision::decision_answer> decide(
      decision::decision_request& request) = 0;

  /// The tag that ANSWERED. Not necessarily the tag that was asked for: the
  /// pinned schema says so explicitly, and the model-identity guard in the
  /// route exists because of it.
  virtual std::string name() const = 0;

  /// The readout kind this model implements, and the one it was asked for.
  /// A container whose readout is not the one this build implements is refused
  /// BY NAME, naming the readout -- a neighbouring recipe would produce
  /// well-formed answers for a different question.
  virtual std::string readout() const = 0;

  /// The question kinds this build implements. The union over the registry, for
  /// `oflm list` and for a client that wants to know before it builds a batch.
  virtual std::vector<std::string> supported_kinds() const = 0;

  /// Task prompt names, for the same reason AutoEmbeddingModel has them: a
  /// decision model has none, and returning an empty list that a caller
  /// iterates is better than pretending it has some.
  virtual std::vector<std::string> prompt_names() const { return {}; }
};
