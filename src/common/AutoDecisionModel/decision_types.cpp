/// \file decision_types.cpp
/// \brief The SystemOne plan/answer layer. See decision_types.hpp.
///
/// \date 2026-09-30
///
/// SPDX-License-Identifier: MIT
///
/// EVERYTHING HERE IS PURE AND ON THE HOST: parse a request, render a response,
/// render an option string. No model, no tokenizer, no NPU. That is deliberate
/// and it is the cheapest thing that can be wrong in this whole port, so it is
/// where the port starts.
#include "AutoDecisionModel/decision_types.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

namespace decision {

using json = nlohmann::ordered_json;

const char* const kNoulFalseDefault = "no, the statement does not hold";
const char* const kNoulTrueDefault = "yes, the statement holds";

namespace {

/// Python's `repr()` for a finite double: the SHORTEST string that round-trips.
///
/// Not `%.17g`, and not `%.17g` with a couple of digits trimmed. llama.cpp's
/// laya port formatted criterion floats with `%.17g` and it flipped 17 of 208
/// decisions, because the prompt text changed and the prompt text is the input.
/// `%.17g` on 1.0 also prints `1`, where Python prints `1.0`, and JSON's number
/// grammar agrees with Python here only by luck.
///
/// The search is the standard shortest-round-trip ladder: print at increasing
/// precision until the text parses back to the same bits. `try_from_chars` is
/// exact rather than correctly-rounded, so the comparison is a bit comparison
/// and not a tolerance.
std::string py_repr_double(double v) {
  if (std::isnan(v)) return "NaN";
  if (std::isinf(v)) return v > 0 ? "Infinity" : "-Infinity";
  char buf[64];
  for (int prec = 1; prec <= 17; ++prec) {
    std::snprintf(buf, sizeof(buf), "%.*g", prec, v);
    double back = 0.0;
    auto res = std::from_chars(buf, buf + std::strlen(buf), back);
    if (res.ec == std::errc() && back == v) break;
  }
  std::string s(buf);
  // Python's float repr always carries a decimal point or an exponent; %g drops
  // it for an integral value. `json.dumps(1.0)` is `1.0`, not `1`.
  if (s.find('.') == std::string::npos && s.find('e') == std::string::npos &&
      s.find('E') == std::string::npos && s.find("inf") == std::string::npos &&
      s.find("nan") == std::string::npos) {
    s += ".0";
  }
  return s;
}

void escape_json_string(const std::string& in, std::string& out) {
  out += '"';
  for (unsigned char c : in) {
    switch (c) {
      case '"':  out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        // ensure_ascii=False: non-ASCII goes out RAW, as UTF-8, not as \uXXXX.
        // Escaping it would be a valid JSON string that is a DIFFERENT string,
        // and the difference is the prompt text.
        if (c < 0x20) {
          char esc[8];
          std::snprintf(esc, sizeof(esc), "\\u%04x", c);
          out += esc;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  out += '"';
}

/// `json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)`.
///
/// Hand-built rather than delegated to nlohmann, because the separators are
/// part of the contract: nlohmann emits `{"a":1}` and Python emits
/// `{"a": 1}`, and a rubric string that differs by two spaces is a different
/// prompt and a different answer.
void dump_python(const json& v, std::string& out);

void dump_scalar(const json& v, std::string& out) {
  if (v.is_string()) {
    escape_json_string(v.get_ref<const std::string&>(), out);
  } else if (v.is_boolean()) {
    out += v.get<bool>() ? "true" : "false";
  } else if (v.is_number_integer()) {
    out += std::to_string(v.get<int64_t>());
  } else if (v.is_number_unsigned()) {
    out += std::to_string(v.get<uint64_t>());
  } else if (v.is_number_float()) {
    out += py_repr_double(v.get<double>());
  } else if (v.is_null()) {
    out += "null";
  } else {
    out += "null";
  }
}

void dump_python(const json& v, std::string& out) {
  if (v.is_array()) {
    out += '[';
    bool first = true;
    for (const auto& e : v) {
      if (!first) out += ", ";
      first = false;
      dump_python(e, out);
    }
    out += ']';
  } else if (v.is_object()) {
    out += '{';
    bool first = true;
    // Iteration order IS insertion order: this is ordered_json, and that is the
    // whole reason the plan's "option order is carried by the array" rule and
    // this hand-rolled serialiser are the same decision.
    for (auto it = v.begin(); it != v.end(); ++it) {
      if (!first) out += ", ";
      first = false;
      escape_json_string(it.key(), out);
      out += ": ";
      dump_python(it.value(), out);
    }
    out += '}';
  } else {
    dump_scalar(v, out);
  }
}

[[noreturn]] void bad(const std::string& what) {
  throw DecisionRequestInvalid(what);
}

/// The schema types `instructions` as `str | dict | list | None`. The prompt
/// builder concatenates it into a string ("%s question: %s"), so a structured
/// value has no honest rendering here and says so rather than dumping JSON into
/// the middle of an English sentence and calling it a prompt.
std::string read_instructions(const json& q, const std::string& key) {
  if (!q.contains("instructions")) return std::string();
  const json& f = q.at("instructions");
  if (f.is_null()) return std::string();
  if (f.is_string()) return f.get_ref<const std::string&>();
  bad("question '" + key + "': 'instructions' is a " +
      (f.is_object() ? "JSON object" : f.is_array() ? "JSON array" : "number") +
      ". This build renders instructions as text; structured instructions are "
      "not implemented. Serialise it to a string yourself, or expect the "
      "prompt to be a different string than you intended.");
}

/// The three types the pinned schema defines. `kind_from` returning false is
/// what makes an unknown type DROPPABLE rather than fatal -- see parse_request.
bool kind_from(const std::string& t, decision_kind& out) {
  if (t == "noul") { out = DECISION_KIND_NOUL; return true; }
  if (t == "choice") { out = DECISION_KIND_CHOICE; return true; }
  if (t == "score") { out = DECISION_KIND_SCORE; return true; }
  return false;
}

const char* kind_name(decision_kind k) {
  switch (k) {
    case DECISION_KIND_NOUL: return "noul";
    case DECISION_KIND_CHOICE: return "choice";
    case DECISION_KIND_SCORE: return "score";
  }
  return "?";
}

}  // namespace

// ---------------------------------------------------------------------------

std::string render_criterion(const json& v) {
  if (v.is_string()) return v.get_ref<const std::string&>();
  std::string out;
  dump_python(v, out);
  return out;
}

/// The noul side's criterion, or the default sentence for that side.
///
/// `is absent or is the empty string` -- and NOT "is falsy". An empty list and
/// an empty dict are meaningful criteria that upstream renders as `[]` and `{}`;
/// treating them as absent silently replaces the caller's rubric with English.
std::string noul_criterion(const json& crit, const char* side,
                           const char* fallback) {
  if (!crit.contains(side)) return fallback;
  const json& f = crit.at(side);
  if (f.is_null()) return fallback;
  if (f.is_string() && f.get_ref<const std::string&>().empty()) return fallback;
  return render_criterion(f);
}

std::vector<std::string> apply_option_order(const std::vector<std::string> &opts,
                                            const std::vector<int> &order) {
  if (order.empty()) return opts;
  if (order.size() != opts.size())
    throw DecisionRequestInvalid(
        "option_order has " + std::to_string(order.size()) + " entries for " +
        std::to_string(opts.size()) + " options. It is one slot per option.");
  std::vector<std::string> out;
  out.reserve(opts.size());
  for (const int i : order) {
    if (i < 0 || static_cast<size_t>(i) >= opts.size())
      throw DecisionRequestInvalid("option_order entry " + std::to_string(i) +
                                   " is outside [0, " + std::to_string(opts.size()) + ")");
    out.push_back(opts[static_cast<size_t>(i)]);
  }
  return out;
}

namespace {

/// Resolve `q.labels` into exactly two distinct non-empty strings, or refuse.
///
/// Upstream's `option_order` validation, `agent.py:815-828`, including the
/// reason it exists: anything other than a permutation of `range(n)` would show
/// an option twice or not at all, and that has to be refused BEFORE the encoder
/// rather than turned into a plausible answer.
///
/// Upstream accepts any sequence of `int`, and so does this, except that a JSON
/// `true`/`false` is an `int` in Python and upstream's `isinstance(i, bool)`
/// guard is the only thing stopping it from becoming an option index. That
/// guard is reproduced here because without it `option_order: [true, false]`
/// permutes 1 and 0 -- a silent swap.
void parse_option_order(const json &q, const std::string &key,
                        decision_question &dq) {
  const auto it = q.find("option_order");
  if (it == q.end()) return;   // absent is the identity, and the only identity
  // An explicit `null` and an explicit `[]` are REFUSED, not read as "absent".
  // Upstream's check is `if "option_order" in qdef:` followed by a permutation
  // test, so anything present must BE a permutation -- and `null` is present.
  // Reading them as absent is a silent widening of the contract: a client that
  // sends `"option_order": []` because its serialiser emits empty lists for
  // unset fields would get the identity without being told the permutation was
  // dropped. It is a small thing to be strict about, because the whole failure
  // mode of this feature is a caller believing in a permutation that did not
  // happen.
  if (!it->is_array())
    bad("question '" + key + "': 'option_order' must be an array of option "
        "indices, one per option.");
  std::vector<int> order;
  for (const auto &e : *it) {
    // `is_number_integer()` is false for a JSON boolean in nlohmann, so
    // `option_order: [true, false]` is refused by this line alone. Upstream needs
    // an explicit `isinstance(i, bool)` guard for the same reason in Python,
    // where bool IS an int -- there, `[true, false]` would otherwise permute by
    // 1 and 0 and pass the sorted() check. Worth being explicit that the refusal
    // is load-bearing rather than incidental.
    if (!e.is_number_integer())
      bad("question '" + key + "': 'option_order' holds a non-integer. A JSON "
          "true/false is not an option index -- upstream refuses it for exactly "
          "this reason, and accepting it would silently permute by 1 and 0.");
    order.push_back(static_cast<int>(e.get<int64_t>()));
  }
  const int64_t n = static_cast<int64_t>(dq.options.size());
  if (static_cast<int64_t>(order.size()) != n)
    bad("question '" + key + "': 'option_order' has " +
        std::to_string(order.size()) + " entries for " + std::to_string(n) +
        " options. It is one slot per option, each option exactly once.");
  std::vector<char> seen(static_cast<size_t>(n), 0);
  for (const int v : order) {
    if (v < 0 || v >= n || seen[static_cast<size_t>(v)])
      bad("question '" + key + "': 'option_order' must be a permutation of "
          "range(" + std::to_string(n) + "), got an out-of-range or repeated "
          "index. Slot s shows option option_order[s], so a repeat would show "
          "one option twice and drop another.");
    seen[static_cast<size_t>(v)] = 1;
  }
  dq.option_order = std::move(order);
}

/// Mirrors upstream's `_resolve_noul_labels`, including its `set(labels) !=`
/// shape: a caller that sends one key, or three, has not sent noul labels and
/// does not know it has not.
void resolve_noul_labels(const json& q, const std::string& key,
                         std::vector<std::string>& out) {
  if (!q.contains("labels")) return;          // 0: upstream defaults both
  const json& lf = q.at("labels");
  if (lf.is_null()) return;
  if (!lf.is_object())
    bad("question '" + key + "': 'labels' must be an object with exactly the "
        "keys \"false\" and \"true\"");
  std::string f, t;
  bool have_f = false, have_t = false;
  for (auto it = lf.begin(); it != lf.end(); ++it) {
    if (it.key() == "false") { have_f = true; f = it.value().is_string() ? it.value().get_ref<const std::string&>() : std::string(); }
    else if (it.key() == "true") { have_t = true; t = it.value().is_string() ? it.value().get_ref<const std::string&>() : std::string(); }
    else bad("question '" + key + "': 'labels' has the key '" + it.key() +
             "'; noul labels must map exactly \"false\" and \"true\"");
  }
  if (!have_f || !have_t)
    bad("question '" + key + "': 'labels' must map exactly \"false\" and "
        "\"true\"");
  // Upstream strips and then refuses empty or equal sides.
  auto trim = [](std::string s) {
    const char* ws = " \t\n\r\f\v";
    const size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos) return std::string();
    const size_t e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
  };
  f = trim(f);
  t = trim(t);
  if (f.empty() || t.empty() || f == t)
    bad("question '" + key + "': 'labels' must name two DISTINCT non-empty "
        "wording strings for the false and true sides");
  out = {f, t};
}

}  // namespace

decision_request parse_request(const json& body, double temperature) {
  decision_request r;
  decision_kind probe_kind = DECISION_KIND_NOUL;

  if (!body.contains("state"))
    bad("request has no 'state'. It is required: a decision model answers about "
        "a state, and there is no default one.");
  const json& sf = body.at("state");
  if (sf.is_string()) {
    r.state = sf.get_ref<const std::string&>();
  } else if (sf.is_array()) {
    bad("'state' as a list of content parts is not supported by this build. "
        "Serialise the parts into one string -- the model reads one state.");
  } else if (sf.is_object()) {
    bad("'state' as a structured JSON object is not implemented. Upstream's "
        "serialize_state() json.dumps()es it with ensure_ascii=False; that "
        "serialisation, down to Python's shortest round-trip float formatting, "
        "is a follow-on and not a formatting shortcut. Pass the string.");
  } else {
    bad("'state' must be a string");
  }

  if (!body.contains("model") || !body.at("model").is_string())
    bad("request has no string 'model'. Required: the answer says which model "
        "gave it, and a request that does not say which model to ask is not a "
        "request.");
  r.model = body.at("model").get_ref<const std::string&>();

  if (!body.contains("questions") || !body.at("questions").is_object())
    bad("request has no object 'questions'");
  const json& qf = body.at("questions");
  if (qf.empty())
    bad("request carries zero questions; the schema requires at least one "
        "('minProperties: 1'). There is nothing to answer.");

  for (auto it = qf.begin(); it != qf.end(); ++it) {
    const std::string& key = it.key();
    const json& q = it.value();
    // DROP, DO NOT REFUSE, on a type this build does not implement -- including
    // a question with no `type` at all. Forward compatibility in this ecosystem
    // is "ignore what you do not know, do not fail the batch": the SDK's own
    // decoder runs a pre-pass that deletes such an answer and logs a warning
    // BEFORE pydantic validation, so a client and a server can disagree about
    // unknown kinds without either being wrong. Refusing the batch would refuse
    // a request the hosted API serves.
    if (!q.is_object() || !q.contains("type") || !q.at("type").is_string() ||
        !kind_from(q.at("type").get_ref<const std::string&>(), probe_kind)) {
      r.dropped.push_back(key);
      continue;
    }
    decision_question dq;
    try {
      dq.key = key;
      dq.kind = probe_kind;
      dq.instructions = read_instructions(q, key);
      static const json empty = json::object();
      const json& c = (q.is_object() && q.contains("criteria")) ? q.at("criteria") : empty;

      switch (dq.kind) {
        case DECISION_KIND_CHOICE: {
          // Upstream raises on `labels` for anything but noul, and a wire type
          // that accepts `labels` on a choice is accepting something upstream
          // refuses -- the request would be answered by a model that never
          // agreed to it.
          if (q.contains("labels"))
            bad("question '" + key + "': 'labels' is only supported for noul "
                "questions");
          if (!c.is_object())
            bad("question '" + key + "': a choice needs 'criteria' as an object "
                "keyed by choice name. Its KEY ORDER is the answer space.");
          for (auto ci = c.begin(); ci != c.end(); ++ci) {
            decision_option o;
            o.label = ci.key();
            // Upstream: `str(k) if v is None or v == "" else "%s: %s" % (k, ...)`,
            // with `v == ""` and NOT `not v` -- the comment at render_options
            // calls out that 0 and False are legitimate criterion values.
            const bool absent = ci.value().is_null();
            const bool empty_str =
                ci.value().is_string() &&
                ci.value().get_ref<const std::string&>().empty();
            o.description = absent || empty_str ? std::string()
                                                : render_criterion(ci.value());
            dq.options.push_back(std::move(o));
          }
          if (dq.options.size() < kChoiceMinOptions ||
              dq.options.size() > kChoiceMaxOptions)
            bad("question '" + key + "': a choice takes " +
                std::to_string(kChoiceMinOptions) + "-" +
                std::to_string(kChoiceMaxOptions) + " options, this one has " +
                std::to_string(dq.options.size()) +
                ". The 255 is upstream's, not ours: the act head's feature "
                "vector carries k/255.0, so k is calibrated against 255 and a "
                "256th option pushes that feature outside the range it was "
                "fitted on.");
          break;
        }
        case DECISION_KIND_SCORE: {
          if (q.contains("labels"))
            bad("question '" + key + "': 'labels' is only supported for noul "
                "questions");
          if (!c.is_array())
            bad("question '" + key + "': a score's 'criteria' is an ORDERED "
                "LIST of levels; index 0 is level 0. It arrived as a " +
                (c.is_object() ? "JSON object" : std::string("non-list")) +
                ", which has no order to recover.");
          for (const auto& lv : c) {
            decision_option o;
            o.label = std::to_string(dq.options.size());
            o.description = lv.is_string() ? lv.get_ref<const std::string&>()
                                           : render_criterion(lv);
            dq.options.push_back(std::move(o));
          }
          if (dq.options.size() < kScoreMinLevels ||
              dq.options.size() > kScoreMaxLevels)
            bad("question '" + key + "': a score takes " +
                std::to_string(kScoreMinLevels) + "-" +
                std::to_string(kScoreMaxLevels) + " ordered levels, this one "
                "has " + std::to_string(dq.options.size()) +
                ". (The pinned schema says only min_length=1; the 2-10 band is "
                "this ecosystem's and this build's, and it is imposed here "
                "rather than claimed of the schema.)");
          break;
        }
        case DECISION_KIND_NOUL: {
          resolve_noul_labels(q, key, dq.noul_labels);
          const std::string fl =
              dq.noul_labels.size() == 2 ? dq.noul_labels[0] : "false";
          const std::string tl =
              dq.noul_labels.size() == 2 ? dq.noul_labels[1] : "true";
          const json& cc = c.is_object() ? c : empty;
          // Semantic order [false, true], ALWAYS. The pinned schema makes noul
          // criteria an object with `true` and `false` keys, so there is no
          // caller order to preserve -- and an order that could vary would put
          // the answer's 0 under whichever label the object happened to sort to.
          dq.options.push_back({fl, noul_criterion(cc, "false", kNoulFalseDefault)});
          dq.options.push_back({tl, noul_criterion(cc, "true", kNoulTrueDefault)});
          break;
        }
      }
      parse_option_order(q, key, dq);
    } catch (const DecisionRequestInvalid&) {
      throw;
    } catch (const std::exception& e) {
      bad(std::string("question '") + key + "': " + e.what());
    }
    r.questions.push_back(std::move(dq));
  }

  if (r.questions.empty() && !r.dropped.empty())
    bad("every question in the request named a type this build does not "
        "implement, so there is nothing to answer. Implementations: noul, "
        "choice, score.");

  // The temperature rule is here, not at the call site, because the reason it
  // exists is a wire fact: the TypeSafe request schema is {state, model,
  // questions}. There is no temperature field, so a per-question temperature has
  // nowhere to live and cannot be supported without extending the contract.
  if (!(temperature >= 0.5 && temperature <= 5.0))
    bad("temperature " + py_repr_double(temperature) +
        " is outside [0.5, 5.0]. One value for the whole request, not per "
        "question -- and the pinned schema has no per-question temperature "
        "field at all, so this is the only place it can come from.");
  r.temperature = temperature;
  return r;
}

// ---------------------------------------------------------------------------

json render_answer(const decision_question& q, const decision_answer& a) {
  if (q.kind != a.kind)
    bad("answer for question '" + q.key + "' is a " + kind_name(a.kind) +
        " and the question is a " + kind_name(q.kind) +
        ". Refusing rather than reshaping one into the other.");
  // k, from the QUESTION's own option count -- not from the batch's marker
  // count and not from probabilities.size(). Upstream masks to -1e4 and
  // softmaxes the batch's padded kmax axis, then reads only p[:k]; a "the
  // probabilities sum to 1" test passes on the wrong length, so the length is
  // asserted here where it can be.
  const size_t k = q.options.size();
  if (a.probabilities.size() != k)
    bad("answer for question '" + q.key + "' carries " +
        std::to_string(a.probabilities.size()) + " probabilities for a question "
        "with " + std::to_string(k) + " options. Length k is THIS row's marker "
        "count; a batch's padded kmax here is a silently wrong answer.");
  if (!a.logits.empty() && a.logits.size() != k)
    bad("answer for question '" + q.key + "' carries " +
        std::to_string(a.logits.size()) + " logits for a question with " +
        std::to_string(k) + " options");

  json out = json::object();
  switch (q.kind) {
    case DECISION_KIND_NOUL: {
      // No `confidence` key AT ALL. Not absent-because-null: absent. The pinned
      // schema's NoulAnswer has no such field, and the entropy form degenerates
      // to 1.0 for a two-option question, so emitting it would be a constant
      // that reads as certainty.
      if (a.confidence != 0.f)
        bad("a noul answer carries a confidence of " + py_repr_double(a.confidence) +
            ". The pinned schema's NoulAnswer has no 'confidence' field and the "
            "entropy form is identically 1.0 for a two-option question, so the "
            "field would be a constant that reads as certainty. Refusing rather "
            "than emitting it.");
      out["type"] = "noul";
      out["noul"] = a.noul;
      break;
    }
    case DECISION_KIND_CHOICE: {
      out["type"] = "choice";
      out["choice"] = a.choice;
      out["confidence"] = a.confidence;
      json probs = json::object();
      for (size_t i = 0; i < k; ++i) probs[q.options[i].label] = a.probabilities[i];
      out["probabilities"] = std::move(probs);
      break;
    }
    case DECISION_KIND_SCORE: {
      out["type"] = "score";
      out["score"] = a.score;
      out["confidence"] = a.confidence;
      // Index-keyed maps serialise with STRING keys. The pinned schema types
      // both as dict[str, ...], and a JSON object key is always a string, so a
      // 3-level score emits {"0": ..., "1": ..., "2": ...} -- not [{...}] and
      // not {0: ...}.
      json legend = json::object(), probs = json::object();
      for (size_t i = 0; i < k; ++i) {
        legend[std::to_string(i)] = q.options[i].description;
        probs[std::to_string(i)] = a.probabilities[i];
      }
      out["legend"] = std::move(legend);
      out["probabilities"] = std::move(probs);
      break;
    }
  }
  return out;
}

json render_response(const std::string& answered_model,
                     const std::vector<decision_question>& questions,
                     const std::vector<decision_answer>& answers,
                     int64_t input_tokens) {
  if (questions.size() != answers.size())
    bad("internal: " + std::to_string(questions.size()) + " questions and " +
        std::to_string(answers.size()) + " answers");
  json out = json::object();
  // `model` is the tag that ANSWERED, not the one that was asked for.
  out["model"] = answered_model;
  json ans = json::object();
  for (size_t i = 0; i < questions.size(); ++i)
    ans[questions[i].key] = render_answer(questions[i], answers[i]);
  out["answers"] = std::move(ans);
  // output_tokens is ALWAYS 0. Definitional, not a stub: a non-autoregressive
  // readout generates nothing, so there is nothing to bill for and nothing to
  // count.
  out["usage"] = json{{"input_tokens", input_tokens}, {"output_tokens", 0}};
  return out;
}

}  // namespace decision
