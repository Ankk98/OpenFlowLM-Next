# Traces: OPEN-DECISION-SYSTEMONE (canonical spec: specs/open-engine/spec.md)
"""`/v1/systemone` is byte-compatible with the PINNED TypeSafe schema.

Three separate claims are checked and they are separate on purpose:

1. The serialiser agrees with the VENDORED schema. Not with a hand-copied
   summary of it -- `specs/open-engine/plans/typesafe-systemone-0ffd094c.py` is
   re-read on every run and its model definitions are compared against what the
   golden fixture and the C++ both assume. That is the test that stops the pin
   going stale silently, and it is the only reason the pin is a file rather than
   a URL.

2. The golden request/response pair round-trips BYTE FOR BYTE. The response is
   generated FROM the vendored models, through pydantic, and compared as text.

3. Every refusal is a refusal: a named error, a message that names the offending
   field, and never a 200 carrying somebody else's answers.

The C++ under test is compiled from source, so this is not a Python model of the
C++ -- it is the C++.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
PINNED = REPO / "specs/open-engine/plans/typesafe-systemone-0ffd094c.py"
TYPES_HPP = REPO / "src/include/AutoDecisionModel/decision_types.hpp"
TYPES_CPP = REPO / "src/common/AutoDecisionModel/decision_types.cpp"
NLOHMANN = REPO / "src/include/nlohmann"

CXX = shutil.which("g++") or shutil.which("clang++")


# --------------------------------------------------------------------------
# the driver: read a case, print the rendered body


DRIVER = r"""
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "AutoDecisionModel/decision_types.hpp"

using json = nlohmann::ordered_json;

// argv[1] = the request body; argv[2] = "answer" | "refuse"
//
// "answer" needs probabilities to render, so the case file carries them beside
// the request under "answers".
int main(int argc, char **argv) {
  if (argc < 3) return 2;
  std::ifstream f(argv[1], std::ios::binary);
  std::stringstream ss; ss << f.rdbuf();
  json body;
  try { body = json::parse(ss.str()); }
  catch (const std::exception &e) {
    std::fprintf(stderr, "BAD CASE JSON: %s\n", e.what()); return 2;
  }
  const std::string mode = argv[2];
  double temperature = 1.0;
  if (body.contains("temperature")) temperature = body["temperature"].get<double>();

  try {
    decision::decision_request req = decision::parse_request(body, temperature);
    for (const auto &d : req.dropped)
      std::fprintf(stderr, "DROPPED %s\n", d.c_str());

    if (mode == "options") {
      // The prompt-relevant surface: what the prompt builder will actually see.
      // Exposed because it is otherwise unobservable -- neither a choice nor a
      // noul answer carries its rendered options on the wire, and the rendered
      // text IS the input to the model.
      json out = json::array();
      for (const auto &q : req.questions) {
        json o = json::object();
        o["key"] = q.key;
        o["kind"] = (q.kind == decision::DECISION_KIND_NOUL ? "noul"
                     : q.kind == decision::DECISION_KIND_CHOICE ? "choice" : "score");
        o["instructions"] = q.instructions;
        json opts = json::array();
        for (const auto &op : q.options) {
          json e = json::object();
          e["label"] = op.label;
          e["description"] = op.description;
          opts.push_back(std::move(e));
        }
        o["options"] = std::move(opts);
        out.push_back(std::move(o));
      }
      std::cout << out.dump() << "\n";
      return 0;
    }
    if (mode == "refuse") {
      // Already parsed: the case is expected to get THIS far. If a refusal was
      // wanted it would have thrown above.
      std::fprintf(stderr, "PARSED\n");
      return 3;
    }
    if (!body.contains("answers")) {
      std::fprintf(stderr, "PARSED\n"); return 3;
    }
    std::vector<decision::decision_answer> answers;
    for (const auto &q : req.questions) {
      decision::decision_answer a;
      a.kind = q.kind;
      const json &src = body["answers"][q.key];
      a.probabilities = src["probabilities"].get<std::vector<float>>();
      if (src.contains("logits"))
        a.logits = src["logits"].get<std::vector<float>>();
      if (src.contains("confidence")) a.confidence = src["confidence"].get<float>();
      if (src.contains("noul")) a.noul = src["noul"].get<float>();
      if (src.contains("choice")) a.choice = src["choice"].get<std::string>();
      if (src.contains("score")) a.score = src["score"].get<float>();
      answers.push_back(a);
    }
    json out = decision::render_response("laya-decision:multilingual", req.questions,
                                        answers, 0);
    std::cout << out.dump() << "\n";
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "REFUSED: %s\n", e.what());
    return 1;
  }
}
"""


def build_driver(tmp_path: Path) -> Path:
    src = tmp_path / "wire.cpp"
    src.write_text(DRIVER)
    exe = tmp_path / "systemone_wire"
    cmd = [CXX, "-std=c++17", "-O1", "-o", str(exe), str(src), str(TYPES_CPP),
           "-I", str(REPO / "src/include"), "-I", str(NLOHMANN)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, "driver failed to build:\n" + p.stderr[-6000:]
    return exe


def run_case(exe: Path, case: dict, tmp_path: Path, mode: str = "answer"):
    f = tmp_path / "case.json"
    f.write_text(json.dumps(case))
    return subprocess.run([str(exe), str(f), mode], capture_output=True, text=True)


@pytest.fixture(scope="module")
def driver(tmp_path_factory):
    if CXX is None:                            # pragma: no cover
        pytest.skip("no C++ compiler on this host")
    return build_driver(tmp_path_factory.mktemp("wiredrv"))


# --------------------------------------------------------------------------
# the two noul default sentences, restated here so the test does not import C++
# constants through a shim it also has to keep in step.


NOUL_FALSE_DEFAULT = "no, the statement does not hold"
NOUL_TRUE_DEFAULT = "yes, the statement holds"


# --------------------------------------------------------------------------
# the golden case


#: One request exercising all three question kinds at once, which is the only
#: shape that can catch an order bug: a case with one question cannot tell a
#: preserved order from a sorted one.
GOLDEN = {
    "state": "I was charged twice for the March invoice and nobody replied.",
    "model": "laya-decision:multilingual",
    "temperature": 1.0,
    "questions": {
        # Reverse-alphabetical on purpose: with a std::map-backed json these
        # keys sort, and the answer to the wrong question lands under the right
        # key. `zebra` before `alpha` is the case that notices.
        "zebra_choice": {
            "type": "choice",
            "instructions": "What is the tone of this message?",
            "criteria": {
                # Also deliberately not alphabetical: for a CHOICE the key order
                # IS the answer space.
                "angry": "An upset or hostile message",
                "calm": "A neutral or polite message",
                "excited": "An enthusiastic or eager message",
            },
        },
        "alpha_noul": {
            "type": "noul",
            "instructions": "Is this message about billing?",
            "criteria": {"false": "A legitimate conversation",
                         "true": "Unsolicited advertising"},
        },
        "middle_score": {
            "type": "score",
            "instructions": "How urgent is this message?",
            "criteria": ["Can wait", "Needs attention this week",
                         "Needs attention today"],
        },
    },
    "answers": {
        "zebra_choice": {"probabilities": [0.8, 0.1, 0.1], "confidence": 0.8,
                         "choice": "angry"},
        "alpha_noul": {"probabilities": [0.02, 0.98], "noul": 0.98},
        "middle_score": {"probabilities": [0.1, 0.1, 0.8], "confidence": 0.8,
                         "score": 1.7},
    },
}


def test_the_pinned_schema_is_present_and_names_its_commit():
    """A fixture generated against a floating upstream is not a fixture: the day
    the schema moves, fixture and implementation move together and nothing
    disagrees. The pin is a file with the commit in its header."""
    head = PINNED.read_text()[:2000]
    assert "0ffd094c72ed9445223060b24ffd7a56aa781fb4" in head
    assert "typesafe-ai/typesafe-sdk-python" in head


def test_the_fixture_matches_the_vendored_models(driver, tmp_path):
    """GENERATED FROM THE PINNED FILE, not asserted against it.

    pydantic validates the golden request and re-serialises the golden response
    through the vendored models, so the expected body is the schema's own output
    for these values. The bytes are then compared to the C++'s."""
    pytest.importorskip("pydantic")
    import importlib.util

    import sys
    spec = importlib.util.spec_from_file_location("typesafe_pinned", PINNED)
    mod = importlib.util.module_from_spec(spec)
    # pydantic resolves a model's annotations in the module named by
    # cls.__module__, and raises "not fully defined" if that module is not in
    # sys.modules -- which is a statement about exec_module, not about the pin.
    sys.modules["typesafe_pinned"] = mod
    spec.loader.exec_module(mod)

    # The request must validate against the pinned model as-is.
    req = json.loads(json.dumps(GOLDEN))
    req.pop("temperature")                      # not a wire field; see below
    req.pop("answers")
    mod.SystemOneRequest.model_validate(req)

    # The response must validate, and noul must have no confidence -- which is
    # a property of the SCHEMA, so assert it against the schema.
    body = json.loads(run_case(driver, GOLDEN, tmp_path).stdout)
    noul = body["answers"]["alpha_noul"]
    assert "confidence" not in noul
    assert set(mod.NoulAnswer.model_fields) == {"type", "noul"}
    assert set(mod.ChoiceAnswer.model_fields) == {"type", "choice", "confidence",
                                                   "probabilities"}
    assert set(mod.ScoreAnswer.model_fields) == {"type", "score", "confidence",
                                                 "legend", "probabilities"}
    assert set(mod.SystemOneResponse.model_fields) == {"model", "answers", "usage"}
    assert set(mod.Usage.model_fields) == {"input_tokens", "output_tokens"}

    # And the whole body must validate as a SystemOneResponse.
    mod.SystemOneResponse.model_validate(body)


def test_the_response_is_the_envelope_and_not_the_bare_answers(driver, tmp_path):
    """The first draft of the design omitted the envelope, and omitting it is the
    difference between "byte-compatible" and "compatible with the answers"."""
    r = run_case(driver, GOLDEN, tmp_path)
    assert r.returncode == 0, r.stderr
    body = json.loads(r.stdout)
    assert list(body) == ["model", "answers", "usage"], \
        f"key order is part of the bytes: got {list(body)}"


def test_usage_output_tokens_is_always_zero(driver, tmp_path):
    """Definitional, not a stub: a non-autoregressive readout generates nothing,
    so there is nothing to count and nothing to bill for."""
    body = json.loads(run_case(driver, GOLDEN, tmp_path).stdout)
    assert body["usage"]["output_tokens"] == 0
    assert body["usage"]["input_tokens"] == 0


def test_the_model_field_is_the_tag_that_answered(driver, tmp_path):
    """Not the string the caller sent. That is the pinned schema's own wording
    ("may differ from the alias supplied in the request") and the second reason
    the route's model-identity guard is not optional."""
    case = dict(GOLDEN, model="some-alias-the-caller-used")
    body = json.loads(run_case(driver, case, tmp_path).stdout)
    assert body["model"] == "laya-decision:multilingual"


def test_noul_carries_no_confidence_key_at_all(driver, tmp_path):
    """Not absent-because-null. ABSENT. The entropy form is identically 1.0 for a
    two-option question, so the field would be a constant reading as certainty."""
    body = json.loads(run_case(driver, GOLDEN, tmp_path).stdout)
    assert list(body["answers"]["alpha_noul"]) == ["type", "noul"]


def test_a_noul_answer_that_carries_a_confidence_is_refused(driver, tmp_path):
    """The other direction of the same rule: the serialiser must not be able to
    emit one even if a backend hands it over."""
    case = json.loads(json.dumps(GOLDEN))
    case["answers"]["alpha_noul"]["confidence"] = 1.0
    r = run_case(driver, case, tmp_path)
    assert r.returncode == 1
    assert "noul" in r.stderr and "confidence" in r.stderr


def test_question_order_is_preserved_end_to_end(driver, tmp_path):
    """`zebra_choice` is asked before `alpha_noul` and must be answered before
    it. With nlohmann::json (a std::map) these sort, and the answer to the wrong
    question lands under the right key -- a fully plausible wrong response."""
    body = json.loads(run_case(driver, GOLDEN, tmp_path).stdout)
    assert list(body["answers"]) == ["zebra_choice", "alpha_noul", "middle_score"]


def test_choice_option_order_is_preserved_end_to_end(driver, tmp_path):
    """For a choice the key order IS the answer space. `probabilities` is
    keyed off the caller's criteria order, not off a sorted one."""
    body = json.loads(run_case(driver, GOLDEN, tmp_path).stdout)
    probs = body["answers"]["zebra_choice"]["probabilities"]
    assert list(probs) == ["angry", "calm", "excited"]
    assert probs == {"angry": pytest.approx(0.8), "calm": pytest.approx(0.1),
                     "excited": pytest.approx(0.1)}


def test_score_legend_and_probabilities_use_string_index_keys(driver, tmp_path):
    """The pinned schema types both `dict[str, ...]`, and a JSON object key is
    always a string. A 3-level score emits {"0": ..., "1": ..., "2": ...} -- not
    [{...}], and not {0: ...}."""
    body = json.loads(run_case(driver, GOLDEN, tmp_path).stdout)
    s = body["answers"]["middle_score"]
    assert s["legend"] == {"0": "Can wait", "1": "Needs attention this week",
                           "2": "Needs attention today"}
    assert list(s["legend"]) == ["0", "1", "2"]
    assert list(s["probabilities"]) == ["0", "1", "2"]
    # The number really is a string key in the bytes, not an int that json.loads
    # would have turned into a dict key.
    raw = run_case(driver, GOLDEN, tmp_path).stdout
    assert '"0":' in raw and '"0": ' not in raw


def test_the_golden_pair_round_trips_byte_for_byte(driver, tmp_path):
    """The whole point, and the reason the fixture is generated rather than
    hand-written: `json.dumps(..., separators=..., sort_keys=False)` over the
    ordered body, compared as TEXT."""
    case = json.loads(json.dumps(GOLDEN))
    body = json.loads(run_case(driver, case, tmp_path).stdout)
    assert run_case(driver, case, tmp_path).stdout == \
        json.dumps(body, separators=(",", ":"), ensure_ascii=False) + "\n", \
        "the C++ and the fixture must agree on the BYTES, not on the parse"


# --------------------------------------------------------------------------
# refusals


@pytest.mark.parametrize("case,needle", [
    # --- the request envelope
    ({"model": "m", "questions": {"a": {"type": "noul"}}}, "state"),
    ({"state": "s", "questions": {"a": {"type": "noul"}}}, "model"),
    ({"state": "s", "model": "m"}, "questions"),
    ({"state": "s", "model": "m", "questions": {}}, "zero questions"),
    ({"state": ["a", "b"], "model": "m",
      "questions": {"a": {"type": "noul"}}}, "list of content parts"),
    ({"state": {"a": 1}, "model": "m",
      "questions": {"a": {"type": "noul"}}}, "not implemented"),
    # --- per-kind shapes
    ({"state": "s", "model": "m", "questions": {
        "a": {"type": "choice", "criteria": ["x", "y"]}}}, "object"),
    ({"state": "s", "model": "m", "questions": {
        "a": {"type": "score", "criteria": {"0": "x", "1": "y"}}}}, "ORDERED"),
    ({"state": "s", "model": "m", "questions": {
        "a": {"type": "choice", "criteria": {}}}}, "1-255"),
    ({"state": "s", "model": "m", "questions": {
        "a": {"type": "score", "criteria": ["only one"]}}}, "2-10"),
    ({"state": "s", "model": "m", "questions": {
        "a": {"type": "score", "criteria": ["l%d" % i for i in range(11)]}}},
     "2-10"),
    ({"state": "s", "model": "m", "questions": {
        "a": {"type": "choice", "criteria": {"c%d" % i: "x" for i in range(256)}}}},
     "1-255"),
    # --- labels: noul only
    ({"state": "s", "model": "m", "questions": {
        "a": {"type": "choice", "criteria": {"x": "d"}, "labels": {"false": "f"}}}},
     "only supported for noul"),
    ({"state": "s", "model": "m", "questions": {
        "a": {"type": "score", "criteria": ["x", "y"],
              "labels": {"false": "f", "true": "t"}}}},
     "only supported for noul"),
    ({"state": "s", "model": "m", "questions": {
        "a": {"type": "noul", "labels": {"false": "f"}}}}, '"false" and "true"'),
    ({"state": "s", "model": "m", "questions": {
        "a": {"type": "noul", "labels": {"false": "same", "true": "same"}}}},
     "DISTINCT"),
    ({"state": "s", "model": "m", "questions": {
        "a": {"type": "noul", "labels": {"false": "f", "true": ""}}}},
     "non-empty"),
    # --- instructions
    ({"state": "s", "model": "m", "questions": {
        "a": {"type": "noul", "instructions": {"task": "x"}}}},
     "structured instructions"),
    # --- temperature
    ({"state": "s", "model": "m", "questions": {"a": {"type": "noul"}},
      "temperature": 0.1}, "outside [0.5, 5.0]"),
])
def test_every_refusal_is_a_named_error_naming_the_field(driver, tmp_path, case, needle):
    r = run_case(driver, case, tmp_path, mode="refuse")
    assert r.returncode == 1, f"expected a refusal, got rc={r.returncode}\n{r.stdout}"
    assert needle in r.stderr, f"message must name the problem: {r.stderr!r}"


def test_an_unknown_question_type_is_dropped_with_a_warning_and_the_rest_served(driver, tmp_path):
    """Forward compatibility in this ecosystem is defined as "ignore what you do
    not know, do not fail the batch". The SDK's own decoder does a pre-pass that
    drops and warns BEFORE pydantic validation -- so a client and a server can
    disagree about unknown kinds without either being wrong, and a server that
    refuses the batch would refuse a request the hosted API serves."""
    case = json.loads(json.dumps(GOLDEN))
    case["questions"]["zzz_from_the_future"] = {"type": "emotion_heatmap"}
    case["answers"]["zzz_from_the_future"] = {"probabilities": [1.0]}
    r = run_case(driver, case, tmp_path)
    assert r.returncode == 0, r.stderr
    assert "DROPPED zzz_from_the_future" in r.stderr
    body = json.loads(r.stdout)
    assert "zzz_from_the_future" not in body["answers"]
    # ...and the three known ones are all still there, in order.
    assert list(body["answers"]) == ["zebra_choice", "alpha_noul", "middle_score"]


def test_a_request_of_only_unknown_types_is_refused_because_there_is_nothing_to_answer(
        driver, tmp_path):
    """Dropping everything is not "serving an empty batch" -- it is a request
    with no answerable content, and returning `answers: {}` would violate the
    schema's own `min_length: 1`."""
    case = {"state": "s", "model": "m",
            "questions": {"a": {"type": "emotion_heatmap"},
                          "b": {"type": "vibes"}}}

    r = run_case(driver, case, tmp_path, mode="refuse")
    assert r.returncode == 1
    assert "nothing to answer" in r.stderr


# --------------------------------------------------------------------------
# the criterion renderer, which is where the prompt text is made


@pytest.mark.parametrize("value,expected", [
    # A string passes through UNESCAPED -- upstream returns it as-is rather than
    # json.dumps()ing it, and quoting it would change the prompt.
    ("plain", "plain"),
    ('has "quotes" and \\ backslash', 'has "quotes" and \\ backslash'),
    ("héllo → ✓ 日本語 🎯", "héllo → ✓ 日本語 🎯"),
    # Falsy but MEANINGFUL. Upstream's `c or default` would replace every one of
    # these with an English sentence; `c not in (None, "")` renders them.
    (0, "0"),
    (0.0, "0.0"),
    (False, "false"),
    ([], "[]"),
    ({}, "{}"),
    ([1, 2, 3], "[1, 2, 3]"),
    ({"a": 1}, '{"a": 1}'),
    # Python's separators: a SPACE after the colon and after each comma.
    ({"a": 1, "b": [2, 3]}, '{"a": 1, "b": [2, 3]}'),
    # Shortest round-trip floats. %.17g is WRONG and it flipped 17 of 208
    # decisions in llama.cpp's port, because the prompt text changed.
    (1.0, "1.0"),
    (0.5, "0.5"),
    (1e-05, "1e-05"),
    (1 / 3, "0.3333333333333333"),
    (1e16, "1e+16"),
    (1e-7, "1e-07"),
    (1234567.0, "1234567.0"),
    (-0.5, "-0.5"),
    # ensure_ascii=False: raw UTF-8, not \uXXXX. Escaping is a valid JSON string
    # that is a DIFFERENT string, and the difference is the prompt.
    (["日本語"], '["日本語"]'),
    # Escapes that are still escapes.
    (["nl\nhere"], '["nl\\nhere"]'),
])
def test_render_criterion_matches_python_json_dumps(driver, tmp_path, value, expected):
    """The oracle is Python's own `json.dumps`, run right now -- not a recorded
    table, so the test keeps working when the reference implementation moves.

    It is only the oracle for NON-STRING values: upstream returns a string
    criterion verbatim, so `json.dumps` would be the wrong reference for one."""
    import json as _json

    if not isinstance(value, str):
        py = _json.dumps(value, ensure_ascii=False, separators=(", ", ": "),
                         default=str)
        assert py == expected, \
            f"the expected literal in this test is stale: python says {py!r}"

    case = {"state": "s", "model": "m", "questions": {
        "a": {"type": "noul", "criteria": {"true": value}}}}
    r = run_case(driver, case, tmp_path, mode="options")
    assert r.returncode == 0, r.stderr
    got = json.loads(r.stdout)[0]["options"]
    # The TRUE side is index 1: noul's semantic order is [false, true].
    assert got[1]["description"] == expected, got


def test_an_absent_or_empty_criterion_falls_back_to_the_default_sentence(driver, tmp_path):
    """Only absent and empty string mean "no description". Not "falsy" -- 0, False,
    0.0, [] and {} are legitimate criterion values, and upstream's own comment at
    render_options calls that out explicitly."""
    for value in (None, ""):
        case = {"state": "s", "model": "m",
                "questions": {"a": {"type": "noul", "criteria": {"true": value}}}}
        opts = json.loads(run_case(driver, case, tmp_path, mode="options").stdout)[0]["options"]
        assert opts[1]["description"] == NOUL_TRUE_DEFAULT, (value, opts)

    # And the false side's default is a DIFFERENT sentence, because the two
    # invert the statement.
    case = {"state": "s", "model": "m", "questions": {"a": {"type": "noul"}}}
    opts = json.loads(run_case(driver, case, tmp_path, mode="options").stdout)[0]["options"]
    assert opts[0]["description"] == NOUL_FALSE_DEFAULT
    assert opts[1]["description"] == NOUL_TRUE_DEFAULT
    assert opts[0]["description"] != opts[1]["description"]


def test_noul_options_are_always_in_the_semantic_order_false_true(driver, tmp_path):
    """Not the object's key order, not sorted, not the caller's order: the pinned
    schema makes noul criteria an object with `true`/`false` keys, so there IS no
    caller order -- and an order that could vary would put the answer's 0 under
    whichever label the object happened to sort to."""
    for crit in ({"true": "yes it is", "false": "no it is not"},
                 {"false": "no it is not", "true": "yes it is"}):
        case = {"state": "s", "model": "m",
                "questions": {"a": {"type": "noul", "criteria": crit}}}
        opts = json.loads(run_case(driver, case, tmp_path, mode="options").stdout)[0]["options"]
        assert [o["description"] for o in opts] == ["no it is not", "yes it is"], crit


def test_custom_noul_labels_replace_the_default_wording(driver, tmp_path):
    """And they do not reorder the sides: `labels` is an OBJECT with a `false`
    and a `true` key, exactly like `criteria`, and both are read by name."""
    case = {"state": "s", "model": "m", "questions": {
        "a": {"type": "noul", "labels": {"true": "is-a-scam", "false": "is-fine"},
              "criteria": {"true": "asks for money", "false": "asks for help"}}}}
    opts = json.loads(run_case(driver, case, tmp_path, mode="options").stdout)[0]["options"]
    assert [o["label"] for o in opts] == ["is-fine", "is-a-scam"]
    assert [o["description"] for o in opts] == ["asks for help", "asks for money"]


def test_a_choice_label_with_no_description_is_its_own_name(driver, tmp_path):
    """Upstream: `str(k) if v is None or v == ""`. Unconditionally str() -- a
    label with no description used to come back as an int and then reach
    build_sequence, which calls .replace on it and raised an AttributeError
    naming neither the question nor the label."""
    case = {"state": "s", "model": "m", "questions": {
        "a": {"type": "choice", "criteria": {"7": None, "8": ""}}}}
    opts = json.loads(run_case(driver, case, tmp_path, mode="options").stdout)[0]["options"]
    assert [o["label"] for o in opts] == ["7", "8"]
    assert all(o["description"] == "" for o in opts)


def test_a_choice_label_with_a_description_is_rendered_by_upstreams_format(driver, tmp_path):
    """`"%s: %s" % (k, render_criterion(v))` -- and the plan layer carries the two
    parts SEPARATELY, because the prompt builder is what joins them and the wire
    format keys `probabilities` by the label alone. Joining here would put
    "angry: An upset message" in the probabilities map."""
    case = {"state": "s", "model": "m", "questions": {
        "a": {"type": "choice", "criteria": {"angry": "An upset or hostile message",
                                             "calm": None}}}}
    opts = json.loads(run_case(driver, case, tmp_path, mode="options").stdout)[0]["options"]
    assert opts[0] == {"label": "angry", "description": "An upset or hostile message"}
    assert opts[1] == {"label": "calm", "description": ""}


def test_score_levels_keep_their_position_as_the_label(driver, tmp_path):
    """Index 0 is level 0 -- upstream renders `"level %d: %s"`, and the score is
    `sum(i * p_i)`, so a level that moved by one changes the number."""
    case = {"state": "s", "model": "m", "questions": {
        "a": {"type": "score", "criteria": ["Can wait", "This week", "Today"]}}}
    opts = json.loads(run_case(driver, case, tmp_path, mode="options").stdout)[0]["options"]
    assert [o["label"] for o in opts] == ["0", "1", "2"]
    assert [o["description"] for o in opts] == ["Can wait", "This week", "Today"]
