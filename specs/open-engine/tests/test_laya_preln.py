# Traces: OPEN-ENC-MODERNBERT (specs/open-engine/spec.md), Phase 5 of
# specs/open-engine/plans/laya-decision-encoder.md
"""Phase 5: the pre-LN loop, and the refusals that keep it from running the
post-LN model.

Phase 5's arithmetic gate is a measurement against the real checkpoint; it lives
in `utilities/laya_preln_reference.py` (the fp64/bf16 oracle) and the numbers
are recorded in the Phase 5 commit and in the plan. What is here is the half of
Phase 5 that CAN be a test, and it is the half that regresses silently.

The shape of the risk: every one of the six facts this phase reads out of the
container -- pre-LN, the final norm, the identity layer-0 norm, the gate half
order, the band width, the per-layer attention types -- has a plausible wrong
value. A pre-LN container read by a post-LN loop returns vectors of exactly the
right shape, exactly the right norm, and the wrong values. So each fact is
tested in the direction that matters, which is that a container MISSING it, or
carrying the OTHER value, is REFUSED BY NAME -- and that the arch-4 state does
not survive into a second, non-arch-4 container loaded in the same process.

That last one is the failure a per-container test cannot see, and it is the one
that turns a correct arch=4 load into a wrong arch=2 load: the geometry globals
are process-wide, so a BERT container loaded after a ModernBERT one must not
inherit its band.
"""
from __future__ import annotations

import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_modernbert_pack as mp  # noqa: E402  the fixtures and the drivers

REPO = mp.REPO
ORACLE = REPO / "utilities/laya_preln_reference.py"

# The load driver with a SECOND load in the same process, so the global-reset
# test has something to observe. The first shape is mp.LOAD_DRIVER's; this one
# takes two containers and prints the arch-4 globals after each.
TWO_LEASES = r"""
#include <cstdio>
#include <exception>
#include <string>
#include "npue_encoder.hpp"

static void report(const char *tag) {
  std::printf("%s arch_gate=%s preln=%d identity_ln1=%d band_half=%lld "
              "sliding=%zu rope_theta=%g\n",
              tag,
              npue::enc::g_gate_order == npue::enc::GateOrder::GateFirst
                  ? "GateFirst" : "UpFirst",
              (int)npue::enc::g_preln, (int)npue::enc::g_identity_ln1_layer0,
              (long long)npue::enc::g_band_half,
              npue::enc::g_sliding_layer.size(), npue::enc::g_rope_theta);
}

int main(int argc, char **argv) {
  if (argc < 2) return 2;
  for (int i = 1; i < argc; ++i) {
    try {
      npue::File m(argv[i]);
      auto tok = npue::enc::load_tokenizer(m, argv[i]);
      npue::enc::ShapeLease lease(m);
      report(argv[i]);
      if (i + 1 < argc) continue;
    } catch (const std::exception &e) {
      std::fprintf(stderr, "REFUSED[%d]: %s\n", i, e.what());
      return 1;
    }
  }
  return 0;
}
"""


# pytest does not resolve another MODULE's fixtures -- only a conftest's -- so
# the parent's two drivers are rebuilt here out of the parent's own source
# text and link list. A fixture that wraps `mp.load_driver` instead would work
# and would also silently inherit the parent's tmp_path_factory scope, which is
# a module-scoped compiler build this module would then depend on invisibly.
LOADER_LINK = ("npue_encoder.cpp", "npue.cpp", "npue_pack.cpp", "json_min.cpp",
               "xlmr_tokenizer_gen.cpp", "gemma_tokenizer_gen.cpp",
               "bbpe_tokenizer_gen.cpp", "tokenizer_bbpe.cpp",
               "tokenizer_xlmr.cpp", "tokenizer_gemma.cpp", "tokenizer.cpp",
               "gemma_kernels.cpp", "gemma_encode.cpp")


@pytest.fixture(scope="module")
def driver(tmp_path_factory):
    if mp.CXX is None:                                   # pragma: no cover
        pytest.skip("no C++ compiler on this host")
    return mp.build_driver(tmp_path_factory.mktemp("p5_packdriver"))


@pytest.fixture(scope="module")
def packed(driver, tmp_path_factory):
    """The parent's `packed`, rebuilt here for the same reason as the drivers:
    a fixture in another module is not a fixture here."""
    d = tmp_path_factory.mktemp("p5_packed")
    ck = mp.make_checkpoint(d / "mb")
    r = mp.run_driver(driver, "pack", ck)
    assert r.returncode == 0, r.stdout + r.stderr
    npue = mp._container_of(r)
    js = json.loads(mp._container_config(npue))
    return npue, js["config"], {t["name"]: t for t in js["tensors"]}


@pytest.fixture(scope="module")
def load_driver(tmp_path_factory):
    if mp.CXX is None:                                   # pragma: no cover
        pytest.skip("no C++ compiler on this host")
    d = tmp_path_factory.mktemp("p5_loaddrv")
    src = d / "load.cpp"
    src.write_text(mp.LOAD_DRIVER)
    exe = d / "npue_load"
    cmd = ([mp.CXX, "-std=c++17", "-O1", "-mavx2", "-mfma", "-o", str(exe),
            str(src)]
           + [str(REPO / "src/open_npue" / s) for s in LOADER_LINK]
           + ["-I", str(REPO / "src/open_npue")])
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-6000:]
    return exe


@pytest.fixture(scope="module")
def two_lease_driver(tmp_path_factory):
    if mp.CXX is None:                                   # pragma: no cover
        pytest.skip("no C++ compiler on this host")
    d = tmp_path_factory.mktemp("twoloase")
    src = d / "two.cpp"
    src.write_text(TWO_LEASES)
    exe = d / "npue_two"
    cmd = ([mp.CXX, "-std=c++17", "-O1", "-mavx2", "-mfma", "-o", str(exe),
            str(src)]
           + [str(REPO / "src/open_npue" / s) for s in LOADER_LINK]
           + ["-I", str(REPO / "src/open_npue")])
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-4000:]
    return exe


def _rewrite_container(npue: Path, mutate) -> Path:
    """A copy of `npue` whose JSON block has been edited, still a container.

    NOT a repack, because the packer emits the arch-4 keys unconditionally --
    it hard-codes them, correctly, having just derived them from the
    checkpoint. So a container missing one cannot be produced by this packer at
    all, which is exactly why the runtime has to refuse it rather than default
    it: these are defence in depth against a hand-built container, a third-party
    packer, or a checkpoint whose facts change and whose packer has not.

    The layout is what makes this cheap: a fixed 64-byte header with
    json_offset / json_len / data_offset / data_len at 16 / 24 / 32 / 40, and
    the tensors' own offsets are relative to the DATA section, so only the two
    header words move. The new JSON is padded with trailing spaces -- which JSON
    permits -- to put `data_offset` back on a 4096 boundary, because the
    reader REFUSES a container whose data section is not, and a test that
    failed on the alignment would be testing the wrong refusal.
    """
    raw = bytearray(npue.read_bytes())
    assert bytes(raw[:4]) == b"NPUE", "not a .npue container"
    json_off, json_len = struct.unpack_from("<QQ", raw, 16)
    data_off, = struct.unpack_from("<Q", raw, 32)
    block = json.loads(bytes(raw[json_off:json_off + json_len]).decode())
    mutate(block)
    new = json.dumps(block, separators=(",", ":")).encode()
    new += b" " * (-(64 + len(new)) % 4096)
    out = bytearray(raw[:json_off]) + bytearray(new) + raw[data_off:]
    struct.pack_into("<Q", out, 24, len(new))
    struct.pack_into("<Q", out, 32, json_off + len(new))
    dest = npue.with_name(npue.stem + ".edited.npue")
    dest.write_bytes(bytes(out))
    # and it must still READ, or the refusal under test is a broken-container
    # failure wearing a disguise
    assert mp._container_json(dest)["config"]["arch"] == "modernbert_rope_geglu"
    assert len(mp._container_tensors(dest)) == len(mp._container_tensors(npue))
    return dest


# The container's top level is {"config": {...}, "tensors": [...]}, so a
# mutation has to reach INTO the config object -- popping the key off the top
# level is a no-op that silently passes every assertion below it, which is the
# worst shape a fixture bug can have.
def _drop(key):
    def go(block):
        assert key in block["config"], f"the container has no config key {key!r}"
        block["config"].pop(key)
    return go


def _set(key, value):
    def go(block):
        assert key in block["config"], f"the container has no config key {key!r}"
        block["config"][key] = value
    return go


def _packed(driver, modernbert: bool, tmp_path: Path) -> Path:
    """A container. `modernbert=False` packs the arch=0 BERT fixture, which is
    what the global-reset test needs as its SECOND load -- a container that
    carries none of arch=4's state and must therefore not inherit any."""
    if modernbert:
        ck = mp.make_checkpoint(tmp_path)
    else:
        ck = _bert_checkpoint(tmp_path)
    r = mp.run_driver(driver, "pack", ck)
    assert r.returncode == 0, r.stdout + r.stderr
    npue = mp._container_of(r)
    if not modernbert:
        # The arch-0 container falls back to a vocab.txt BESIDE itself when it
        # carries no embedded vocabulary, and the driver writes next to the
        # checkpoint's parent. Copying it is the fixture's job, not the test's
        # subject, and its absence would fail the TOKENIZER rather than the
        # thing under test.
        # In a DIRECTORY named after the container's stem -- that is where the
        # loader looks, and a fallback that missed would fail on the tokenizer
        # rather than on the thing under test.
        beside = npue.parent / npue.stem
        beside.mkdir(parents=True, exist_ok=True)
        shutil.copy(ck / "vocab.txt", beside / "vocab.txt")
    return npue


def _bert_checkpoint(dir_: Path) -> Path:
    """A flat arch=0 BERT-shaped checkpoint, packed by the same driver.

    Deliberately not the ModernBERT fixture with its config edited: the arch
    dispatch is `model_type`, and a container whose arch string was rewritten
    would still be a ModernBERT container underneath.
    """
    r = np.random.default_rng(7)
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / "1_Pooling").mkdir(parents=True, exist_ok=True)
    (dir_ / "1_Pooling/config.json").write_text(json.dumps(
        {"pooling_mode_mean_tokens": True, "pooling_mode_cls_token": False}))
    # A real WordPiece vocabulary, specials included -- the loader refuses one
    # without [CLS] and [SEP], and a refusal here would be a fixture failure
    # masquerading as the one under test.
    (dir_ / "vocab.txt").write_text("\n".join(
        ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        + [f"tok{i}" for i in range(mp.VOCAB - 5)]))
    cfg = {
        "model_type": "bert", "hidden_size": mp.HIDDEN,
        "num_hidden_layers": 2, "num_attention_heads": mp.HEADS,
        "intermediate_size": mp.INTER, "max_position_embeddings": mp.MAX_SEQ,
        "vocab_size": mp.VOCAB, "type_vocab_size": 2, "layer_norm_eps": mp.EPS,
        "hidden_act": "gelu",
    }
    (dir_ / "config.json").write_text(json.dumps(cfg))
    (dir_ / "CHECKPOINT.json").write_text(json.dumps({"repo_id": "fixture/bert"}))
    n = lambda *sh, scale=0.05: (r.standard_normal(sh) * scale).astype(np.float32)
    t = {
        "embeddings.word_embeddings.weight": n(mp.VOCAB, mp.HIDDEN),
        "embeddings.position_embeddings.weight": n(mp.MAX_SEQ, mp.HIDDEN),
        "embeddings.token_type_embeddings.weight": n(2, mp.HIDDEN),
        "embeddings.LayerNorm.weight": n(mp.HIDDEN, scale=1.0),
        "embeddings.LayerNorm.bias": n(mp.HIDDEN, scale=0.01),
        "pooler.dense.weight": n(mp.HIDDEN, mp.HIDDEN),
        "pooler.dense.bias": n(mp.HIDDEN, scale=0.01),
    }
    for i in range(2):
        p = f"encoder.layer.{i}."
        for proj in ("query", "key", "value"):
            t[p + f"attention.self.{proj}.weight"] = n(mp.HIDDEN, mp.HIDDEN)
            t[p + f"attention.self.{proj}.bias"] = n(mp.HIDDEN, scale=0.01)
        t[p + "attention.output.dense.weight"] = n(mp.HIDDEN, mp.HIDDEN)
        t[p + "attention.output.dense.bias"] = n(mp.HIDDEN, scale=0.01)
        t[p + "attention.output.LayerNorm.weight"] = n(mp.HIDDEN, scale=1.0)
        t[p + "attention.output.LayerNorm.bias"] = n(mp.HIDDEN, scale=0.01)
        t[p + "intermediate.dense.weight"] = n(mp.INTER, mp.HIDDEN)
        t[p + "intermediate.dense.bias"] = n(mp.INTER, scale=0.01)
        t[p + "output.dense.weight"] = n(mp.HIDDEN, mp.INTER)
        t[p + "output.dense.bias"] = n(mp.HIDDEN, scale=0.01)
        t[p + "output.LayerNorm.weight"] = n(mp.HIDDEN, scale=1.0)
        t[p + "output.LayerNorm.bias"] = n(mp.HIDDEN, scale=0.01)
    mp._write_safetensors(dir_ / "model.safetensors",
                         {k: (v, "F32") for k, v in t.items()})
    return dir_


# --------------------------------------------------------------------------
# the refusals
#
# All of these edit the CONTAINER, not the checkpoint: the packer emits the
# arch-4 keys unconditionally because it has just derived them, so a container
# missing or contradicting one is not something this packer can produce -- which
# is the whole reason the runtime has to refuse it rather than default it.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("key", [
    "pre_layernorm", "final_norm", "identity_attn_norm_layer0",
])
def test_a_MISSING_arch4_fact_is_refused(packed, load_driver, key):
    """The packer always writes these three, so a container missing one is not
    a checkpoint this model could have -- which is exactly why a MISSING key has
    to be a refusal and not a default. The defaults are the post-LN world: run
    over a pre-LN model they return a correctly-shaped, correctly-normed, wrong
    vector, and the residual stream is re-centred twice per layer."""
    npue, _, _ = packed
    edited = _rewrite_container(npue, _drop(key))
    p = subprocess.run([str(load_driver), str(edited)], capture_output=True,
                       text=True)
    assert p.returncode == 1, p.stdout
    assert key in p.stderr, p.stderr


def test_a_MISSING_sliding_window_is_refused(packed, load_driver):
    """The band width is read from the container, and a default would be a
    guess. Zero is the one value that must not be invented: zero means "full
    attention", i.e. silently the wrong model on 14 of 22 layers."""
    npue, _, _ = packed
    edited = _rewrite_container(npue, _drop("sliding_window"))
    p = subprocess.run([str(load_driver), str(edited)], capture_output=True,
                       text=True)
    assert p.returncode == 1, p.stdout
    assert "sliding_window" in p.stderr, p.stderr


@pytest.mark.parametrize("width", [0, -1])
def test_a_NON_POSITIVE_band_is_refused(packed, load_driver, width):
    npue, _, _ = packed
    edited = _rewrite_container(npue, _set("sliding_window", width))
    p = subprocess.run([str(load_driver), str(edited)], capture_output=True,
                       text=True)
    assert p.returncode == 1, p.stdout
    assert "sliding_window" in p.stderr, p.stderr


def test_the_WRONG_gate_half_order_is_refused(packed, load_driver):
    """`glu_halves` is a DATA field, and it is the one field in Phase 5 whose
    misreading is invisible: up-first and gate-first both produce a model that
    runs, emits fluent text and is wrong. So the container's own key is honoured
    and anything else is refused BY NAME, and the message says which order the
    packer moved -- a reader who trips over this later needs to know that."""
    npue, _, _ = packed
    edited = _rewrite_container(
        npue, _set("glu_halves", "up_first|gate_second -- the other order"))
    p = subprocess.run([str(load_driver), str(edited)], capture_output=True,
                       text=True)
    assert p.returncode == 1, p.stdout
    assert "glu_halves" in p.stderr, p.stderr


def test_an_absolute_position_key_is_refused(packed, load_driver):
    """`position_embedding_type` must be rope. Some releases carry the key as a
    DEAD value. Reading an absolute table would add a SECOND source of position
    on top of RoPE, which is a model that has never existed and one whose error
    is a slowly-varying rotation of the right answer."""
    npue, _, _ = packed
    edited = _rewrite_container(npue, _set("position_embedding_type", "absolute"))
    p = subprocess.run([str(load_driver), str(edited)], capture_output=True,
                       text=True)
    assert p.returncode == 1, p.stdout
    assert "position_embedding_type" in p.stderr, p.stderr


def test_a_LAYER_TYPES_list_of_the_wrong_length_is_refused(packed, load_driver):
    """length != num_layers. The band mask indexes by layer, so a short list
    leaves the tail of the stack UNBANDED -- full attention on layers whose names
    say local, which is the exact failure the flag-off-by-default design exists
    to prevent."""
    npue, _, _ = packed
    edited = _rewrite_container(
        npue, _set("layer_types", mp.LAYER_TYPES[:-1]))
    p = subprocess.run([str(load_driver), str(edited)], capture_output=True,
                       text=True)
    assert p.returncode == 1, p.stdout
    assert "layer_types" in p.stderr, p.stderr


def test_a_LAYER_TYPES_entry_that_is_neither_full_nor_sliding_is_refused(
        packed, load_driver):
    """An unknown attention type is a refusal rather than "not sliding". The
    alternative silently bands a layer whose locality is unknown, and the band
    is applied to a layer that may be global."""
    npue, _, _ = packed
    bad = list(mp.LAYER_TYPES)
    bad[4] = "local_attention"
    edited = _rewrite_container(npue, _set("layer_types", bad))
    p = subprocess.run([str(load_driver), str(edited)], capture_output=True,
                       text=True)
    assert p.returncode == 1, p.stdout
    assert "layer_types[4]" in p.stderr, p.stderr


# --------------------------------------------------------------------------
# the state that must not survive a second load
# --------------------------------------------------------------------------

def test_arch4_state_does_NOT_survive_a_later_non_arch4_container(
        driver, two_lease_driver, tmp_path):
    """The one thing a per-container test cannot see.

    `g_gate_order`, `g_preln`, `g_identity_ln1_layer0`, `g_band_half` and
    `g_sliding_layer` are PROCESS-WIDE, because the encoder's geometry globals
    are (a second source of truth for one value is how two lanes end up running
    two different models). The cost of that choice is that a container loaded
    after an arch=4 one inherits its state unless apply_model_shape clears it --
    and an arch=2 bge container then gets a ModernBERT band, a ModernBERT gate
    order and the pre-LN loop, and returns embeddings of the right shape for
    neither model.

    So the second load in this process is an arch=0 container and every one of
    those five must read as its non-arch-4 value.
    """
    mb = _packed(driver, True, tmp_path / "mb")
    bert = _packed(driver, False, tmp_path / "bert")
    p = subprocess.run([str(two_lease_driver), str(mb), str(bert)],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    lines = [l for l in p.stdout.splitlines() if "arch_gate=" in l]
    assert len(lines) == 2, p.stdout + p.stderr
    assert "arch_gate=GateFirst preln=1 identity_ln1=1 band_half=64 sliding=22" in lines[0], lines[0]
    # The arch=0 one: up-first, post-LN, a real layer-0 norm, and NO band.
    assert "arch_gate=UpFirst preln=0 identity_ln1=0 band_half=0 sliding=0" in lines[1], lines[1]


# --------------------------------------------------------------------------
# the oracle's own arithmetic
#
# The oracle is what the Phase 5 numbers are measured against, so its two
# subtleties are unit-tested here rather than trusted: the bf16 rounding (which
# was wrong twice while this was being written) and the gate order (which is
# the very thing the engine's packed order exists to reproduce).
# --------------------------------------------------------------------------

def _oracle():
    sys.path.insert(0, str(REPO / "utilities"))
    import laya_preln_reference as R
    return R


def test_bf16_rounding_is_round_half_to_even_on_SEVEN_mantissa_bits():
    R = _oracle()
    ulp = 2.0 ** -7                      # bfloat16 keeps 7 mantissa bits
    x = np.array([1.0, 1.0 + ulp, 1.0 + ulp / 2, 1.0 + 1.5 * ulp,
                  1.0 + 2 * ulp, -3.14159265, 0.0, 2.0])
    got = R.bf16(x)
    # Exact representable values are fixed points.
    assert got[0] == 1.0 and got[1] == 1.0 + ulp and got[4] == 1.0 + 2 * ulp
    # Ties go to the EVEN neighbour: 1+ulp/2 is halfway between 1.0 (even) and
    # 1+ulp (odd), so it lands on 1.0. 1+1.5*ulp is halfway between 1+ulp (odd)
    # and 1+2*ulp (even), so it lands on 1+2*ulp. A round-half-up rounding gets
    # BOTH of these wrong, and in opposite directions, which is why the tie
    # rule is the thing under test rather than the ulp.
    assert got[2] == 1.0, got[2]
    assert got[3] == 1.0 + 2 * ulp, got[3]
    # Not tf32: 16 discarded mantissa bits would leave 8 and give
    # 1 + ulp/2 -> 1 + ulp/4. This is the check that the exponent split is
    # bfloat16's and not tf32's.
    assert got[7] == 2.0
    # A carry out of the mantissa into the exponent must survive, which is the
    # reason the rounding is done in the integer domain.
    assert got[5] == -3.140625, got[5]
    assert got[6] == 0.0


def test_the_oracles_gate_is_the_SECOND_half_which_is_what_the_packer_moved():
    """ModernBERT binds `x, gate = Wi(h).chunk(2, -1)`, so the gate is the
    second half. The packer moves it first so the engine's `up * act(gate)`
    arithmetic applies unchanged, and the oracle has to read the CHECKPOINT's
    layout (second half) or the two would agree on the wrong model."""
    R = _oracle()
    src = (REPO / "src/open_npue/npue_pack.cpp").read_text()
    assert "packed FIRST" in src and "glu_halves" in src, (
        "the packer no longer documents moving the gate half first, so the "
        "engine's half order has lost its justification")
    cfg = mp._config("modernbert")
    C = R.Config(cfg)
    assert C.band == cfg["local_attention"] // 2 == 64
    assert C.theta == 160000.0
    assert C.layer_types[:4] == ["full_attention", "sliding_attention",
                                  "sliding_attention", "full_attention"]
    # The oracle's own line, read back: it splits on the OUTPUT axis and takes
    # the SECOND half as the gate.
    body = (REPO / "utilities/laya_preln_reference.py").read_text()
    assert "up, gate = wi[:I].T, wi[I:].T" in body


def test_the_oracle_REFUSES_two_rope_thetas():
    """The packer refuses it too, and the oracle refusing it is what keeps the
    two from disagreeing about a model neither supports."""
    R = _oracle()
    cfg = mp._config("modernbert")
    cfg["rope_parameters"]["sliding_attention"]["rope_theta"] = 10000
    with pytest.raises(SystemExit) as e:
        R.Config(cfg)
    assert "two RoPE thetas" in str(e.value)


def test_the_oracle_REFUSES_a_layer_types_that_contradicts_the_rule():
    """ModernBERT DERIVES layer_types from global_attn_every_n_layers. A
    checkpoint that stores a contradicting list has one of the two wrong, and
    the band mask is decided by this list -- so it is refused, not preferred."""
    R = _oracle()
    cfg = mp._config("modernbert")
    cfg["layer_types"] = ["full_attention"] * cfg["num_hidden_layers"]
    with pytest.raises(SystemExit) as e:
        R.Config(cfg)
    assert "disagrees with the derived" in str(e.value)


# --------------------------------------------------------------------------
# the source-level contracts, read from the file
#
# These are assertions about TEXT, which is normally the wrong thing to do. They
# are here for the two invariants that no runtime test can reach without a
# 1.4 GB checkpoint, and each names the file:line it is about.
# --------------------------------------------------------------------------

def test_the_three_gated_activation_copies_all_take_their_halves_from_the_helpers():
    """The gated activation exists in THREE copies -- swiglu_cpu, the bf16 fused
    epilogue and the int8 fused epilogue -- and the plan's own note is that two
    of them say they were copied verbatim from the third. `fuse_ffn_epilogue`
    defaults to true, so the bf16 copy is the one that runs in production: a
    half-order key honoured by swiglu_cpu alone would leave --no-fuse-ffn and
    the default computing different models, and the difference would be a gate
    swap rather than a rounding difference.

    So: no copy may divide the row by position any more. Each must name the
    helpers.
    """
    src = mp.ENC.read_text()
    assert src.count("up_half(v, inter)") == 4, (
        "expected the two fused epilogues' two arms to take their halves from "
        f"up_half; found {src.count('up_half(v, inter)')}")
    assert src.count("gate_half(v, inter)") == 4, (
        f"found {src.count('gate_half(v, inter)')} gate_half uses, expected 4")
    code = [l for l in src.splitlines()
            if not l.lstrip().startswith("//")]
    joined = "\n".join(code)
    assert "gate_first_half" not in joined and "gate_second_half" not in joined, (
        "a positional helper is back in CODE. up_half/gate_half are named for "
        "the ROLE because a helper called `gate_first_half` is one assignment "
        "away from being used as the destination, which computes "
        "gate * act(up).")
    # and the up/gate roles are the ones the formula says
    assert "up[k] = gated_pair(up, gate, k, gelu_erf_exact)" in src
    assert "up[j] = gated_pair(up, gate, j, gelu_erf_exact)" in src
    assert "dst[j] = gated_pair(up, gate, j, gelu_erf_exact)" in src


def test_the_preLN_loop_NEVER_reaches_add_norm():
    """`add_norm_quant` / `add_norm_bf16` end with a store of the NORMALISED
    value into the residual. That is correct under post-LN and is a correctness
    bug under pre-LN, and the two places it could creep back in are the two
    `x +=` sites."""
    src = mp.ENC.read_text()
    body = src[src.index("std::vector<float> run_preln("):
               src.index("std::vector<float> run_dispatch(")]
    assert "add_norm" not in body, (
        "run_preln calls add_norm_*, which stores the normalised value as the "
        "residual -- the residual stream would be re-centred twice per layer")
    assert body.count("add_plain(x, proj)") == 1
    assert body.count("add_plain(x, down)") == 1
    # and the layer-0 copy is unconditional, with only the NORM conditional
    assert "hbuf = x;\n      if (!(g_identity_ln1_layer0 && L == 0))" in body, (
        "the layer-0 skip must not take hbuf = x with it: the qkv GEMM would "
        "read resize()'s zero fill, which is a constant and runs clean")


def test_the_final_norm_is_staged_in_BOTH_arms():
    """The two arms number their sites differently -- `--host-ln` pushes
    `s_ln.size() + 1` and `layer_norm` then does `layer_norm_cpu(x, slot - 1)`,
    while the staged arm pushes the device slot -- and `layer_norm_cpu` indexes
    h_gamma[site]/h_beta[site] with no bounds check. A site pushed into one arm
    only is an out-of-range READ in the other: a silent wrong number, not a
    crash."""
    src = mp.ENC.read_text()
    assert 'if (g_preln) ln_host("final_norm.weight", "final_norm.bias");' in src
    assert 'if (g_preln) ln_one("final_norm.weight", "final_norm.bias");' in src
    # and the site is computed, not written down
    assert ("size_t final_norm_site() const { return 1 + 2 * static_cast<size_t>(g_layers); }"
            in src)


def test_the_single_encode_entry_point_is_run_dispatch():
    """One place picks the loop. A caller that reached for run() directly would
    run the post-LN model over a pre-LN container, and nothing downstream could
    tell: the output is the right shape and the right norm."""
    src = mp.ENC.read_text()
    assert "e.run_dispatch(buf)" in src
    body = src[src.index("void chunk(Encoder &e,"):]
    assert ".run(buf)" not in body, "a second encode entry point bypasses the dispatch"


def _fn(src: str, header: str) -> str:
    """One function's body, from its header to the next member declaration.

    Slicing to a NAMED second header is what these tests did first and it is
    wrong in a way that passes: `av_impl`'s doc comment sits ABOVE it, so
    "everything up to the ctx[b,i,h] comment" is the empty string, and an
    assertion over an empty slice is an assertion over nothing.
    """
    i = src.index(header)
    j = src.find("\n  void ", i + len(header))
    k = src.find("\n  static ", i + len(header))
    end = min(x for x in (j, k, len(src)) if x > 0)
    return src[i:end]


# --------------------------------------------------------------------------
# Phase 6: the band mask
#
# The band's correctness is a MEASUREMENT (utilities/laya_preln_reference.py and
# the rig, and the numbers are in the Phase 6 commit). What is here is the
# structure, and the structure is where a band goes wrong: a band applied to the
# wrong layer, through the wrong path, or with the wrong width is a model that
# runs and answers.
# --------------------------------------------------------------------------

def test_the_band_term_is_ONE_definition_used_by_both_mask_paths():
    """Two mask paths and both must band.

    `run` picks between `softmax_cpu` (host) and `add_additive_mask` (the array
    softmax design) on a flag. Banding one leaves a model that is SILENTLY
    full-attention on all 22 layers the moment the design is loaded -- and the
    only symptom is that the numbers change, which is what a tuning change looks
    like too.
    """
    src = mp.ENC.read_text()
    assert "kBandFill" in src
    # exactly two mask loops assign the fill, and they are the two paths
    n = src.count("(j < jlo || j >= jhi) ? kBandFill")
    assert n == 2, f"expected 2 banded mask loops, found {n}"
    soft = src[src.index("void softmax_cpu("):]
    assert "kBandFill" in _fn(src, "void softmax_cpu("), "softmax_cpu does not band"
    assert "kBandFill" in _fn(src, "void add_additive_mask("), \
        "add_additive_mask does not band"


def test_the_band_cannot_have_gone_into_add_mask():
    """`add_mask` is [batch, g_seq] and both paths index it with a
    per-SEQUENCE offset -- it has no head axis and no layer axis. A band depends
    on (layer, i, j), so putting it there would be both impossible and silently
    wrong if it were possible. The `i` decode has to be in the loop."""
    src = mp.ENC.read_text()
    body = _fn(src, "void add_additive_mask(")
    assert "const int64_t i = r % g_seq;" in body, (
        "the band needs the query position, and r % g_seq is the whole decode: "
        "the band depends on (layer, i, j) and not on b or h")
    # and add_mask itself is untouched
    assert "add_mask.resize" not in body and "add_mask.push_back" not in body


def test_qk_and_av_BOTH_clamp_the_j_loop():
    """The mask alone computes the same answer and NONE of the saving, because
    it adds a term to all seq^2 scores. Skipping the MACs is the point, so both
    reductions clamp.

    And the clamp has to be in BOTH: a clamp in qk() without one in av() leaves
    out-of-band softmax weights multiplied in, and in floating point 0 * inf is
    not 0.
    """
    src = mp.ENC.read_text()
    assert src.count("j = jlo; j < jhi") == 4, (
        "expected qk's arm plus av's three arms (AVX512, AVX2, scalar) to clamp; "
        f"found {src.count('j = jlo; j < jhi')}")
    for fn in ("void qk_impl(", "void av_impl("):
        body = _fn(src, fn)
        assert "band_lo(i)" in body and "band_hi(i)" in body, fn


def test_the_clamp_is_computed_ONCE_per_i_and_not_per_arm():
    """The AVX512, AVX2 and scalar arms are three copies of the same reduction.
    A clamp written into each is a clamp that can be correct in two -- and the
    head (Phase 7) needs FULL-band attention over these same functions, which it
    gets by calling with band_now == 0 rather than by instantiating a second
    copy of either function."""
    src = mp.ENC.read_text()
    decl = "const int64_t jlo = band_lo(i), jhi = band_hi(i);"
    # Four in the file: the two mask paths and the two reductions. What matters
    # is that each FUNCTION has exactly one, and that it sits above the #if that
    # selects the arm rather than inside an arm.
    for fn in ("void qk_impl(", "void av_impl("):
        body = _fn(src, fn)
        assert body.count(decl) == 1, (
            f"{fn} computes the clamp {body.count(decl)} times; it belongs "
            "once, above the arm selection")
    assert _fn(src, "void add_additive_mask(").count(decl) == 1
    assert _fn(src, "void softmax_cpu(").count(decl) == 1


def test_band_now_is_SET_on_every_layer_including_the_global_ones():
    """The failure this guards is a band left set across layers: `band_half` is a
    model property, `qk_impl` is a method, and a loop that forgets to clear it
    bands the GLOBAL layers. That returns a correctly shaped, correctly normed
    vector."""
    src = mp.ENC.read_text()
    body = src[src.index("std::vector<float> run_preln("):
               src.index("std::vector<float> run_dispatch(")]
    assert "band_now = band_for_layer(L);" in body
    # inside the loop, not before it
    loop = body.index("for (int64_t L = 0; L < g_layers; ++L) {")
    assert body.index("band_now = band_for_layer(L);") > loop
    # and band_for_layer returns 0 for a non-sliding layer, which is what makes
    # "set it on every layer" equivalent to "clear it on every global layer"
    assert "return g_sliding_layer[static_cast<size_t>(L)] ? g_band_half : 0;" in src


def test_the_out_of_band_term_is_ASSIGNED_and_the_in_band_one_is_exact_zero():
    """Two properties, both about what the mask pass writes.

    Out of band: ASSIGNED, not added to. qk_impl() clamps, so an out-of-band
    score is never written and still holds the previous layer's value; adding to
    that is correct only while the stale value stays finite, and the fills
    compound across layers.

    In band: the term is exactly 0.0f, folded into the padding add, so an
    in-band score is the unmasked path's score plus an exact zero -- which is
    what makes "the band cannot change an in-band value" a property of reading
    the code rather than of a tolerance."""
    src = mp.ENC.read_text()
    assert src.count("row[j] = (j < jlo || j >= jhi) ? kBandFill : row[j] + mk[j];") == 2
    assert "static constexpr float kBandFill = -1.0e30f;" in src, (
        "the band must use the SAME fill as the padding mask. -inf would sit in "
        "one buffer next to -1.0e30f as two conventions for one job.")


def test_an_UNBANDED_container_still_runs_the_unbanded_loop():
    """Six shipping encoders have no locality term. `band_now` is 0 for every
    one of them, and the loops take their original arm -- so arch=0 through 3
    are not merely close to bit-identical, they execute the same instructions on
    the same data."""
    src = mp.ENC.read_text()
    add = _fn(src, "void add_additive_mask(")
    soft = _fn(src, "void softmax_cpu(")
    assert "if (!band_now) {" in add
    assert "for (int64_t j = 0; j < g_seq; ++j) row[j] += mk[j];\n          continue;" in add
    assert "if (band_now) {" in soft
    qk = src[src.index("void qk_impl("):src.index("void av_impl(")]
    assert "const int64_t jlo = band_lo(i), jhi = band_hi(i);" in qk, (
        "band_lo/band_hi return the whole range when band_now is 0, so the "
        "unbanded path takes the same loop with the same bounds")
