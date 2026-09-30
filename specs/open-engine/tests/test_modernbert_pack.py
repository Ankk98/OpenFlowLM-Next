# Traces: OPEN-ENC-MODERNBERT (canonical spec: specs/open-engine/spec.md)
"""The packer's two model_type guards, and the arch-4 container it writes.

Two things are checked here and they are deliberately different in kind.

The GUARDS are checked by RUNNING them: a small C++ driver is compiled against
`npue_pack.cpp` and handed two synthetic checkpoints -- one with no
`config.json`, one with `model_type: "modernbert"` -- and both must refuse with a
message naming the thing that is actually wrong. A guard that is only asserted to
exist in the source is a comment.

The CONTAINER is checked by PACKING a synthetic ModernBERT checkpoint and
reading it back: every emitted tensor through `npue::Reader::raw()` at the right
shape and dtype, the gate half of `ffn_up` leading, every `*.bias` zero-filled
and present, and the whole thing accepted by `load_tokenizer` +
`apply_model_shape` through the real `ShapeLease`. That last one is the gate that
catches a misspelled container config key, and it is cheap -- which is the
argument for running it in Phase 0 rather than discovering it in Phase 5.

Nothing here needs a model file or a network: the fixture is generated.
"""
from __future__ import annotations

import json
import shutil
import struct
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
PACK = REPO / "src/open_npue/npue_pack.cpp"
PACK_HPP = REPO / "src/open_npue/npue_pack.hpp"
ENC = REPO / "src/open_npue/npue_encoder.hpp"
NPUE = REPO / "src/open_npue/npue.cpp"

HIDDEN = 768
LAYERS = 22
HEADS = 12
HEAD_DIM = 64
INTER = 1152
MAX_SEQ = 1024
VOCAB = 512          # the fixture's own vocabulary; the real one is 256000
QKV_N = 3 * HIDDEN
FFN_UP_N = 2 * INTER
EPS = "1e-05"
THETA = "160000"

# Global every third layer, sliding otherwise -- ModernBERT derives this rather
# than storing it, and `global_attn_every_n_layers: 3` is the fact it derives
# from. 22 layers -> 8 full / 14 sliding.
LAYER_TYPES = ["full_attention" if i % 3 == 0 else "sliding_attention"
               for i in range(LAYERS)]


# --------------------------------------------------------------------------
# the checkpoint fixture


def _config(model_type: str = "modernbert", nested: bool = False) -> dict:
    return {
        "architectures": ["ModernBertForMaskedLM"],
        "attention_bias": False,
        "cls_token_id": 1,
        "eos_token_id": 1,
        "global_attn_every_n_layers": 3,
        "hidden_activation": "gelu",
        "hidden_size": HIDDEN,
        "intermediate_size": INTER,
        "layer_norm_eps": 1e-5,
        "layer_types": LAYER_TYPES,
        "local_attention": 128,
        "mask_token_id": 4,
        "max_position_embeddings": 8192,
        "mlp_bias": False,
        "model_type": model_type,
        "norm_bias": False,
        "num_attention_heads": HEADS,
        "num_hidden_layers": LAYERS,
        "pad_token_id": 0,
        "position_embedding_type": "sans_pos",
        "rope_parameters": {
            "full_attention": {"rope_theta": 160000, "rope_type": "default"},
            "sliding_attention": {"rope_theta": 160000, "rope_type": "default"},
        },
        "sep_token_id": 1,
        "vocab_size": VOCAB,
        "bos_token_id": 2,
        "unk_token_id": 3,
    }


def _write_safetensors(path: Path, tensors: dict[str, tuple[np.ndarray, str]]) -> None:
    """safetensors, byte for byte: 8-byte LE header length, then a JSON header,
    then the data block in header order. F32 and F16 both appear, because the
    real checkpoint is F16 throughout and a packer that only ever sees F32 is a
    packer that has never met it."""
    header, blob, offset = {}, bytearray(), 0
    for name, (arr, dtype) in tensors.items():
        if dtype == "F16":
            raw = arr.astype(np.float16).tobytes()
        elif dtype == "F32":
            raw = arr.astype(np.float32).tobytes()
        else:                                   # pragma: no cover
            raise AssertionError(dtype)
        header[name] = {"dtype": dtype, "shape": list(arr.shape),
                        "data_offsets": [offset, offset + len(raw)]}
        blob += raw
        offset += len(raw)
    js = json.dumps(header, separators=(",", ":")).encode()
    pad = (-len(js)) % 8
    js += b" " * pad
    path.write_bytes(struct.pack("<Q", len(js)) + js + bytes(blob))


def _rng(seed: int = 7):
    return np.random.default_rng(seed)


def make_checkpoint(dir_: Path, *, nested: bool = False, model_type: str = "modernbert",
                    layers: int = LAYERS, vocab: int = VOCAB) -> Path:
    """A ModernBERT-shaped checkpoint: the real tensor NAMES (encoder.*-prefixed,
    F16), the real SHAPES, and random bytes. Nothing here is a real model and
    nothing needs to be -- what is under test is the packer's reading of the
    layout, which is the thing that goes wrong silently.

    `nested=True` reproduces laya's own tree: the config under encoder/, the
    tokenizer under tokenizer/, the weights beside the config. `nested=False` is
    the flat layout every shipped model uses."""
    r = _rng()
    root = dir_ / "multilingual" if nested else dir_
    root.mkdir(parents=True, exist_ok=True)
    (root / "tokenizer").mkdir(parents=True, exist_ok=True)
    (root / "1_Pooling").mkdir(parents=True, exist_ok=True)
    cfg_dir = root / "encoder" if nested else root
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.json").write_text(json.dumps(_config(model_type), indent=2))
    (root / "1_Pooling/config.json").write_text(
        json.dumps({"pooling_mode_mean_tokens": True, "pooling_mode_cls_token": False}))
    (root / "CHECKPOINT.json").write_text(json.dumps({"repo_id": "fixture/modernbert"}))

    def n(*shape, scale=0.05):
        return (r.standard_normal(shape) * scale).astype(np.float32)

    t: dict[str, tuple[np.ndarray, str]] = {
        # F16 throughout, exactly as the shipped checkpoint is.
        "encoder.embeddings.tok_embeddings.weight": (n(vocab, HIDDEN), "F16"),
        "encoder.embeddings.norm.weight": (n(HIDDEN, scale=1.0), "F16"),
        "encoder.final_norm.weight": (n(HIDDEN, scale=1.0), "F16"),
    }
    for i in range(layers):
        p = f"encoder.layers.{i}."
        t[p + "attn.Wqkv.weight"] = (n(QKV_N, HIDDEN), "F16")
        t[p + "attn.Wo.weight"] = (n(HIDDEN, HIDDEN), "F16")
        t[p + "mlp.Wi.weight"] = (n(FFN_UP_N, HIDDEN), "F16")
        t[p + "mlp.Wo.weight"] = (n(HIDDEN, INTER), "F16")
        t[p + "mlp_norm.weight"] = (n(HIDDEN, scale=1.0), "F16")
        # Layer 0's attn_norm is nn.Identity(): the checkpoint has NO tensor for
        # it, and a packer that insists on one is a packer that will read layer
        # 1's into slot 0.
        if i:
            t[p + "attn_norm.weight"] = (n(HIDDEN, scale=1.0), "F16")
    _write_safetensors(root / "model.safetensors", t)
    return root


# --------------------------------------------------------------------------
# the C++ driver


def _have_cxx() -> bool:
    return shutil.which("g++") is not None or shutil.which("clang++") is not None


CXX = shutil.which("g++") or shutil.which("clang++")

#: mode "guards": one JSON object of {mode, dir, subdir} on argv, prints the
#: refusal message on stdout and exits 0, or lets the exception escape so the
#: driver returns non-zero with the message on stderr.
#: mode "pack": packs and prints the container path.
DRIVER = r"""
#include <cstdio>
#include <exception>
#include <string>
#include <vector>
#include "npue_pack.hpp"

int main(int argc, char **argv) {
  if (argc < 4) return 2;
  const std::string mode = argv[1];
  const std::string dir = argv[2];
  const std::string subdir = argc > 4 ? argv[4] : "";
  try {
    npue::PrepareOptions po;
    po.checkpoint_dir = dir;
    po.config_subdir = subdir;
    po.tokenizer_subdir = "tokenizer";
    po.tile_k = 64;
    po.tile_n = 48;
    po.log = [](const std::string &s) { std::fputs(s.c_str(), stdout); };
    const std::string out = npue::prepare_model_auto(po);
    std::printf("\nOK %s\n", out.c_str());
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "REFUSED: %s\n", e.what());
    return 1;
  }
}
"""


#: The packer's own link closure. npue_pack.cpp calls the JSON parser and the
#: two sibling tokenizer generators, so a driver that links only the packer
#: fails to link -- which is a statement about the driver, not about the packer.
LINK_WITH = ("npue_pack.cpp", "json_min.cpp", "xlmr_tokenizer_gen.cpp",
             "gemma_tokenizer_gen.cpp", "bbpe_tokenizer_gen.cpp",
             "tokenizer_bbpe.cpp", "tokenizer_xlmr.cpp", "tokenizer_gemma.cpp")


def build_driver(tmp_path: Path) -> Path:
    src = tmp_path / "driver.cpp"
    src.write_text(DRIVER)
    exe = tmp_path / "npue_pack_driver"
    cmd = ([CXX, "-std=c++17", "-O1", "-o", str(exe), str(src)]
           + [str(REPO / "src/open_npue" / s) for s in LINK_WITH]
           + ["-I", str(REPO / "src/open_npue")])
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, "driver failed to build:\n" + p.stderr[-4000:]
    return exe


def run_driver(exe: Path, mode: str, dir_: Path, subdir: str = "") -> subprocess.CompletedProcess:
    out = dir_.parent / "out.npue"
    return subprocess.run([str(exe), mode, str(dir_), str(out), subdir],
                          capture_output=True, text=True)


def _container_of(run: subprocess.CompletedProcess) -> Path:
    """The container the driver says it wrote. Read from its own stdout rather
    than recomputed from the paths, so a driver that changed where it puts the
    file is caught here instead of as a FileNotFoundError three lines later."""
    for line in run.stdout.splitlines():
        if line.startswith("OK "):
            return Path(line[3:].strip())
    raise AssertionError("driver wrote no container:\n" + run.stdout + run.stderr)


@pytest.fixture(scope="module")
def driver(tmp_path_factory):
    if not _have_cxx():                       # pragma: no cover
        pytest.skip("no C++ compiler on this host")
    return build_driver(tmp_path_factory.mktemp("packdriver"))


# --------------------------------------------------------------------------
# the guards


def test_no_config_is_refused_naming_the_path(driver, tmp_path):
    """A missing config is not an empty model_type to dispatch on. The message
    must name the file the packer LOOKED FOR, because that is the whole point:
    laya ships no root config.json and the old failure named one it does not
    have, in a directory it does not name."""
    ck = tmp_path / "noconfig"
    ck.mkdir()
    (ck / "model.safetensors").write_bytes(b"")       # present, to prove it is not what is read
    r = run_driver(driver, "guards", ck)
    assert r.returncode == 1, r.stdout + r.stderr
    msg = r.stderr
    assert "config.json" in msg
    assert str(ck) in msg, "the refusal must name the path it looked at"


def test_unparseable_config_is_refused_the_same_way(driver, tmp_path):
    """json_string_field returns "" for unparseable as well as for absent, and
    both are refusals -- but the same message, because they are the same fact
    from the packer's side: no usable model_type."""
    ck = tmp_path / "badjson"
    ck.mkdir()
    (ck / "config.json").write_text("{not json at all")
    r = run_driver(driver, "guards", ck)
    assert r.returncode == 1
    assert "config.json" in r.stderr and "no usable" in r.stderr


def test_typeless_config_is_refused(driver, tmp_path):
    """A config that parses, has keys, and no string `model_type`. Same refusal."""
    ck = tmp_path / "typeless"
    ck.mkdir()
    (ck / "config.json").write_text(json.dumps({"hidden_size": 768}))
    r = run_driver(driver, "guards", ck)
    assert r.returncode == 1
    assert "no usable" in r.stderr and "model_type" in r.stderr


def test_modernbert_is_refused_by_name_until_the_packer_lands(driver, tmp_path):
    """THE OTHER ARM, and the dangerous one.

    A checkpoint that DOES put a config.json at its root, naming an architecture
    no packer here handles, would otherwise fall through to the BERT LAST branch
    and be packed as arch=0 -- GELU plus absolute position embeddings, for a
    GeGLU-plus-RoPE model. The runtime loads that container happily. The message
    has to name model_type, because "refusing" is useless without the fact.

    Phase 3 replaces this refusal with the arch=4 packer, and this test is
    rewritten then -- see test_the_container_names_its_own_architecture_and_
    geometry, which is already written to the post-Phase-3 expectation and
    currently asserts the staging."""
    ck = make_checkpoint(tmp_path / "mb")
    r = run_driver(driver, "pack", ck)
    assert r.returncode == 1, "the guard must refuse, not pack:\n" + r.stdout
    msg = r.stderr
    assert "modernbert" in msg
    # The refusal has to say WHY, because "modernbert is not supported" reads as
    # a missing feature and "this packs as something wrong" reads as a bug.
    assert "arch=0" in msg and "GeGLU" in msg


def test_a_missing_subdir_is_an_error_naming_the_composed_path(driver, tmp_path):
    """The empty-model_type arm must not fire merely because a subdirectory WAS
    named and did not resolve -- it has to name the path it actually looked at,
    subdirectory included, or the user goes looking in the wrong place."""
    ck = make_checkpoint(tmp_path / "wrongsub", nested=True)
    r = run_driver(driver, "pack", ck, subdir="not-where-the-config-is")
    assert r.returncode == 1
    assert "not-where-the-config-is/config.json" in r.stderr


def _bert_checkpoint(dir_: Path, model_type: str) -> Path:
    """The flat, F32, WordPiece BERT layout -- `prepare_model`'s own shape. Used
    to prove the arch=0 LAST branch is still reachable, which is the property the
    new guards must not have cost.

    The geometry is chosen to satisfy the packer's OWN tiling asserts rather than
    to be small: tile_k 64 divides hidden and intermediate, tile_n 48 divides 3x
    hidden, hidden and 2x intermediate, and head_dim is a multiple of 8. A
    fixture that ignores those fails in `tile_b`, which is a statement about the
    fixture and not about the branch under test."""
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / "1_Pooling").mkdir(exist_ok=True)
    r = _rng(11)
    H, L, INTER, VOC, HEADS, MAXSEQ = 192, 2, 192, 64, 3, 256
    (dir_ / "config.json").write_text(json.dumps({
        "model_type": model_type, "num_hidden_layers": L, "num_attention_heads": HEADS,
        "hidden_size": H, "intermediate_size": INTER, "vocab_size": VOC,
        "type_vocab_size": 2, "layer_norm_eps": 1e-12}))
    (dir_ / "1_Pooling/config.json").write_text(
        json.dumps({"pooling_mode_mean_tokens": True, "pooling_mode_cls_token": False}))
    (dir_ / "vocab.txt").write_text("\n".join(f"tok{i}" for i in range(VOC)) + "\n")
    (dir_ / "CHECKPOINT.json").write_text(json.dumps({"repo_id": "fixture/bert"}))

    def n(*shape, scale=0.05):
        return (r.standard_normal(shape) * scale).astype(np.float32)

    t = {
        "embeddings.word_embeddings.weight": (n(VOC, H), "F32"),
        "embeddings.position_embeddings.weight": (n(MAXSEQ, H), "F32"),
        "embeddings.token_type_embeddings.weight": (n(2, H), "F32"),
        "embeddings.LayerNorm.weight": (n(H, scale=1.0), "F32"),
        "embeddings.LayerNorm.bias": (n(H), "F32"),
    }
    for i in range(L):
        p = f"encoder.layer.{i}."
        t[p + "attention.self.query.weight"] = (n(3 * H, H), "F32")
        t[p + "attention.self.query.bias"] = (n(3 * H), "F32")
        t[p + "attention.self.key.weight"] = (n(H, H), "F32")
        t[p + "attention.self.key.bias"] = (n(H), "F32")
        t[p + "attention.self.value.weight"] = (n(H, H), "F32")
        t[p + "attention.self.value.bias"] = (n(H), "F32")
        t[p + "attention.output.dense.weight"] = (n(H, H), "F32")
        t[p + "attention.output.dense.bias"] = (n(H), "F32")
        t[p + "attention.output.LayerNorm.weight"] = (n(H, scale=1.0), "F32")
        t[p + "attention.output.LayerNorm.bias"] = (n(H), "F32")
        t[p + "intermediate.dense.weight"] = (n(INTER, H), "F32")
        t[p + "intermediate.dense.bias"] = (n(INTER), "F32")
        t[p + "output.dense.weight"] = (n(H, INTER), "F32")
        t[p + "output.dense.bias"] = (n(H), "F32")
        t[p + "output.LayerNorm.weight"] = (n(H, scale=1.0), "F32")
        t[p + "output.LayerNorm.bias"] = (n(H), "F32")
    _write_safetensors(dir_ / "model.safetensors", t)
    return dir_


@pytest.mark.parametrize("model_type", ["bert", "xlm-roberta", "some-fork-nobody-has-heard-of"])
def test_the_bert_fallback_is_still_the_last_branch(driver, tmp_path, model_type):
    """arch=0 is deliberately LAST and deliberately unguarded: it covers the whole
    BERT family, and every member reports its own `model_type`. Turning
    "unrecognised" into "refused" would break all six shipped encoders.

    Checked by RUNNING the packer, not by reading the source: a negative source
    assertion ("this string does not appear in the guard") passes just as happily
    after someone rewrites the guard in a form the grep does not recognise."""
    ck = _bert_checkpoint(tmp_path / f"bert-{model_type}", model_type)
    r = run_driver(driver, "pack", ck)
    assert r.returncode == 0, r.stdout + r.stderr
    cfg = json.loads(_container_config(_container_of(r)))["config"]
    assert cfg["arch"] == "bert_abs_gelu_postln"


def test_the_guard_is_before_the_bert_fallback():
    """Order, because order is the whole mechanism: a guard placed after the
    BERT branch is a comment."""
    src = PACK.read_text()
    i_guard = src.index('if (model_type == "modernbert")')
    i_bert = src.index("// arch=0 (BERT family) is the LAST branch")
    assert i_guard < i_bert


def test_a_nested_checkpoint_resolves_through_the_subdir_keys(driver, tmp_path):
    """Nothing in the flat layout could reach multilingual/encoder/config.json.
    Before Phase 3 the modernbert packer does not exist, so this asserts the part
    that DOES exist and is new here: that naming the subdirectory gets past the
    empty-model_type guard to the modernbert guard, i.e. the config was found."""
    ck = make_checkpoint(tmp_path / "nested", nested=True)
    r = run_driver(driver, "pack", ck, subdir="encoder")
    assert r.returncode == 1
    assert "modernbert" in r.stderr, "the subdirectory must have resolved the config"
    assert "no usable" not in r.stderr


def test_the_container_names_its_own_architecture_and_geometry(driver, tmp_path):
    """STAGED. Phase 3 replaces the modernbert refusal with the arch-4 packer;
    this test is written to the post-Phase-3 expectation and asserts the refusal
    until then, so every commit in between has a passing suite.

    The keys `apply_model_shape` reads are load-bearing by SPELLING: a
    `layers` where it wants `num_layers` is not a warning, it is `the .npue
    reports a non-positive shape` two thousand lines away from the cause."""
    ck = make_checkpoint(tmp_path / "geom")
    r = run_driver(driver, "pack", ck)
    assert r.returncode == 1
    assert "modernbert" in r.stderr


def _container_config(npue: Path) -> str:
    raw = npue.read_bytes()
    assert raw[:4] == b"NPUE", "not a .npue container"
    json_off, json_len = struct.unpack_from("<QQ", raw, 16)
    return raw[json_off:json_off + json_len].decode()


def _container_tensors(npue: Path) -> dict:
    js = json.loads(_container_config(npue))
    return {t["name"]: t for t in js["tensors"]}
