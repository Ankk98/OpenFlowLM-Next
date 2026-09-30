/// \file npue_decision.hpp
/// \brief The NPU decision model: arch=4's encoder plus its host head
#pragma once

#include <memory>
#include <string>

#include "AutoDecisionModel/auto_decision_model.hpp"

/// Loads an Laya-format decision container and answers typed questions on it.
///
/// This owns a `npue::dec::Decider`, which owns the NPU `Stack` and therefore
/// one `hw_context`. A server that also serves chat models runs those in a
/// different process or behind the NPU lock: see `requires_npu_access()`, which
/// this route is enrolled in for exactly that reason.
class NpueDecision : public AutoDecisionModel {
 public:
  explicit NpueDecision(std::string tag);
  ~NpueDecision() override;

  void load_model(const std::string& model_path,
                  const nlohmann::ordered_json& model_info,
                  int threads) override;
  std::vector<decision::decision_answer> decide(
      decision::decision_request& request) override;
  std::string name() const override;
  std::string readout() const override;
  std::vector<std::string> supported_kinds() const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string tag_;
};
