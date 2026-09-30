/// \file all_decision_model.hpp
/// \brief The decision-model registry
/// \note A REGISTRY, not a funnel. Every tag here resolves to the backend that
///       implements it, and an unknown tag is an explicit error. The embedding
///       registry's own header records that this used to be a funnel in that
///       tree -- every tag rewritten to one model -- and the reason it mattered
///       is the reason it matters here: a substituted model returns well-formed
///       answers, and nothing downstream can tell they are for the wrong model.
///
/// Adding a model is one line here plus an entry in model_list.json.
#pragma once

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "AutoDecisionModel/npue_decision.hpp"

/// Which backend serves which tag.
enum class DecisionBackend { Npue };

inline const std::unordered_map<std::string, DecisionBackend>&
decision_backend_registry() {
    static const std::unordered_map<std::string, DecisionBackend> reg = {
        {"laya-decision:multilingual", DecisionBackend::Npue},
    };
    return reg;
}

inline std::string complete_simple_decision_tag(std::string model_tag) {
    return model_tag;
}

/// The decision model kinds, in the QTYPES order the container's `type_emb` is
/// indexed by. A model that supports fewer refuses the others BY NAME rather
/// than coercing: a `score` answered as a `choice` returns an option label
/// where a number belongs.
inline const std::vector<std::string>& decision_kind_names() {
    static const std::vector<std::string> k = {"noul", "choice", "score"};
    return k;
}

/// Build the model for `tag` from `dir`. Throws on an unknown tag, naming it.
inline std::unique_ptr<AutoDecisionModel> get_auto_decision_model(
    const std::string& model_tag, const std::filesystem::path& dir,
    const nlohmann::ordered_json* model_info = nullptr,
    int threads = 0) {
    const std::string tag = complete_simple_decision_tag(model_tag);
    const auto& reg = decision_backend_registry();
    const auto it = reg.find(tag);
    if (it == reg.end()) {
        std::string known;
        for (const auto& kv : reg) {
            if (!known.empty()) known += ", ";
            known += kv.first;
        }
        throw std::runtime_error(
            "no decision engine for '" + model_tag + "'. Known: [" + known +
            "]. A tag that names a chat or embedding model is a different kind "
            "of model, and loading it and asking it to decide would return "
            "fluent text rather than an error.");
    }
    std::unique_ptr<AutoDecisionModel> m;
    switch (it->second) {
        case DecisionBackend::Npue:
            m = std::make_unique<NpueDecision>(tag);
            break;
    }
    // An absent model_info means "load from the directory alone", which is what
    // the CLI does. The adapter then reads the keys it needs off the container
    // and off the directory, and says which is missing.
    nlohmann::ordered_json empty = nlohmann::ordered_json::object();
    m->load_model(dir.string(), model_info ? *model_info : empty, threads);
    return m;
}
