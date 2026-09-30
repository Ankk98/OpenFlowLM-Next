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


def _write_tokenizer_json(root: Path) -> None:
    """A Metaspace + byte-fallback tokenizer.json, the SHAPE laya's has.

    Small on purpose: the packer under test reads the blob's presence and its
    flags, not its 256k entries. What it must be is real in every respect the
    generator checks -- the Replace normalizer, the Metaspace pre-tokenizer, a
    byte-fallback alphabet that is closed, no raw space, and merges over raw
    characters -- because a fixture that is easier than the real thing proves
    nothing about the generator accepting the real thing. test_bbpe_tokenizer.py
    is the file that checks the real one, against HuggingFace.
    """
    def bytes_to_unicode() -> dict[int, str]:
        bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        cs = list(bs)
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return dict(zip(bs, [chr(c) for c in cs]))

    vocab: dict[str, int] = {}

    def put(tok: str) -> None:
        if tok not in vocab:
            vocab[tok] = len(vocab)

    put("<pad>"); put("<eos>"); put("<bos>"); put("<unk>"); put("<mask>")
    # <s> and </s> exist but sit FAR from the config's cls=1 / sep=1, the way
    # laya's do (204 and 213). The generator derives cls_id/sep_id from those
    # NAMES, so this fixture reproduces the trap: a blob-only reader would frame
    # every sequence with the wrong id and the model would never see it.
    put("filler-a"); put("filler-b"); put("filler-c")
    put("<s>"); put("</s>")
    put("\u2581")                       # the replacement character itself
    for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
        put(c)
    for c in ".,:;!?'\"-()[]{}/\\@#$%^&*_+=<>|~`":
        put(c)
    put("\n"); put("\t"); put("\u2581\u2581")
    for c in " \u00e9\u00fc\u00f1":
        put(c)
    for c in "\u65e5\u672c\u8a9e":
        put(c)
    # 255 of the 256 <0xNN> pieces; <0x09> is absent because tab has its own
    # entry, which is exactly laya's arrangement.
    for b in range(256):
        if b != 0x09:
            put(f"<0x{b:02X}>")

    merges: list[list[str]] = []
    # Enough merges that a word actually tokenizes into something, and one chain
    # so the merge loop has a ranking to respect.
    for a, b in [("\u2581", "a"), ("\u2581", "b"), ("\u2581", "c"),
                 ("a", "b"), ("b", "c"), ("c", "d"), ("\n", "\n")]:
        if a in vocab and b in vocab and a + b in vocab:
            merges.append([a, b])
    merged = [(" \u00e9",), ]

    added = [
        {"id": vocab["<pad>"], "content": "<pad>", "single_word": False,
         "lstrip": False, "rstrip": False, "normalized": False, "special": True},
        {"id": vocab["<eos>"], "content": "<eos>", "single_word": False,
         "lstrip": False, "rstrip": False, "normalized": False, "special": True},
        {"id": vocab["<bos>"], "content": "<bos>", "single_word": False,
         "lstrip": False, "rstrip": False, "normalized": False, "special": True},
        {"id": vocab["<unk>"], "content": "<unk>", "single_word": False,
         "lstrip": False, "rstrip": False, "normalized": False, "special": True},
        {"id": vocab["<mask>"], "content": "<mask>", "single_word": False,
         "lstrip": True, "rstrip": False, "normalized": False, "special": True},
    ]
    del merged
    doc = {
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": added,
        "normalizer": {"type": "Replace", "pattern": {"String": " "},
                       "content": "\u2581"},
        "pre_tokenizer": {"type": "Metaspace", "replacement": "\u2581",
                          "prepend_scheme": "always", "split": True},
        "post_processor": {"type": "TemplateProcessing",
                           "single": [{"SpecialToken": {"id": "<bos>", "type_id": 0}},
                                      {"Sequence": {"id": "A", "type_id": 0}},
                                      {"SpecialToken": {"id": "<eos>", "type_id": 0}}],
                           "special_tokens": {
                               "<bos>": {"id": "<bos>", "ids": [vocab["<bos>"]]},
                               "<eos>": {"id": "<eos>", "ids": [vocab["<eos>"]]}}},
        "decoder": {"type": "Sequence", "decoders": [
            {"type": "Replace", "pattern": {"String": "\u2581"}, "content": " "},
            {"type": "ByteFallback"}, {"type": "Fuse"}]},
        "model": {"type": "BPE", "byte_fallback": True, "unk_token": "<unk>",
                  "fuse_unk": True, "vocab": vocab, "merges": merges},
    }
    (root / "tokenizer" / "tokenizer.json").write_text(json.dumps(doc))
    del bytes_to_unicode


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
    # `nested=True` reproduces laya's tree UNDER dir_, and returns dir_ -- the
    # served root -- because that is what a caller has and what the two
    # subdirectory keys are FOR. Returning the checkpoint root would hide the
    # thing the test is about.
    root = dir_
    if nested:
        root = dir_ / "multilingual"
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
    # The decision head, under its own (unprefixed) names. 2 layers, WITH
    # biases, ReLU FFN at 4*hidden, plus type_emb / scorer / act_head. These
    # are packed too -- as plain F32 host weights -- so the container is
    # self-contained.
    for h in (0, 1):
        q = f"head.layers.{h}."
        t[q + "norm1.weight"] = (n(HIDDEN, scale=1.0), "F16")
        t[q + "norm1.bias"] = (n(HIDDEN), "F16")
        t[q + "self_attn.in_proj_weight"] = (n(3 * HIDDEN, HIDDEN), "F16")
        t[q + "self_attn.in_proj_bias"] = (n(3 * HIDDEN), "F16")
        t[q + "self_attn.out_proj.weight"] = (n(HIDDEN, HIDDEN), "F16")
        t[q + "self_attn.out_proj.bias"] = (n(HIDDEN), "F16")
        t[q + "norm2.weight"] = (n(HIDDEN, scale=1.0), "F16")
        t[q + "norm2.bias"] = (n(HIDDEN), "F16")
        t[q + "linear1.weight"] = (n(4 * HIDDEN, HIDDEN), "F16")
        t[q + "linear1.bias"] = (n(4 * HIDDEN), "F16")
        t[q + "linear2.weight"] = (n(HIDDEN, 4 * HIDDEN), "F16")
        t[q + "linear2.bias"] = (n(HIDDEN), "F16")
    t["type_emb.weight"] = (n(3, HIDDEN), "F16")
    t["scorer.0.weight"] = (n(HIDDEN, scale=1.0), "F16")
    t["scorer.0.bias"] = (n(HIDDEN), "F16")
    t["scorer.1.weight"] = (n(HIDDEN, HIDDEN), "F16")
    t["scorer.1.bias"] = (n(HIDDEN), "F16")
    t["scorer.3.weight"] = (n(1, HIDDEN), "F16")
    t["scorer.3.bias"] = (n(1), "F16")
    t["act_head.0.weight"] = (n(256, HIDDEN + 4), "F16")
    t["act_head.0.bias"] = (n(256), "F16")
    t["act_head.2.weight"] = (n(2, 256), "F16")
    t["act_head.2.bias"] = (n(2), "F16")
    _write_safetensors(root / "model.safetensors", t)
    # The RL config carries the temperatures and the head geometry, and the
    # packer REFUSES without it: those three floats are the container's only
    # record of how confident the model was trained to be.
    (root / "rl_agent_config.json").write_text(json.dumps({
        "encoder": "fixture/mmbert-base", "head_layers": 2, "max_len": 1024,
        "head_max_len": 256, "max_prefixes": 6,
        "act_costs": {"escalate": 0.5}, "cost_wrong_act": 3.0,
        "amp_dtype": "bf16", "model_name": "rl-agent",
        "temperature": [1.0, 1.0, 1.0], "temperature_by_options": {},
    }, indent=2))
    _write_tokenizer_json(root)
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
    po.checkpoint_subdir = (argc > 5 ? argv[5] : "");
    po.config_subdir = subdir;
    po.tokenizer_subdir = (argc > 6 ? argv[6] : "tokenizer");
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


def run_driver(exe: Path, mode: str, dir_: Path, subdir: str = "",
               ckpt_subdir: str = "", tok_subdir: str = "tokenizer") -> subprocess.CompletedProcess:
    out = dir_.parent / "out.npue"
    return subprocess.run([str(exe), mode, str(dir_), str(out), subdir,
                           ckpt_subdir, tok_subdir],
                          capture_output=True, text=True)


def _container_of(run: subprocess.CompletedProcess) -> Path:
    """The container the driver says it wrote. Read from its own stdout rather
    than recomputed from the paths, so a driver that changed where it puts the
    file is caught here instead of as a FileNotFoundError three lines later."""
    for line in run.stdout.splitlines():
        if line.startswith("OK "):
            return Path(line[3:].strip())
    raise AssertionError("driver wrote no container:\n" + run.stdout + run.stderr)


def _container_json(npue: Path) -> dict:
    """The .npue header's JSON block, parsed.

    The layout is a fixed 64-byte header -- magic, version, arch, flags, then
    json_offset / json_len / data_offset / data_len at 16/24/32/40 -- so this
    reads the file rather than re-deriving it from the packer's own C++."""
    raw = npue.read_bytes()
    assert raw[:4] == b"NPUE", "not a .npue container"
    json_off, json_len = struct.unpack_from("<QQ", raw, 16)
    return json.loads(raw[json_off:json_off + json_len].decode())


def _blob(npue: Path) -> np.ndarray:
    """The DATA section, as uint8.

    Tensor offsets in the JSON are relative to the data section and NOT to the
    file -- Writer::add counts from zero and Writer::write emits the blob after
    the header and the JSON. Reading at `file[offset]` instead gives bytes from
    the wrong part of the container, which decode to plausible-looking floats
    and are how a whole class of "the packer is wrong" conclusions get reached
    by a reader that is."""
    raw = np.frombuffer(npue.read_bytes(), dtype=np.uint8)
    data_off = struct.unpack_from("<Q", npue.read_bytes(), 32)[0]
    return raw[data_off:]


def _container_config(npue: Path) -> str:
    return json.dumps(_container_json(npue))


def _container_tensors(npue: Path) -> dict:
    return {t["name"]: t for t in _container_json(npue)["tensors"]}


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


def test_the_modernbert_refusal_is_GONE_and_what_replaced_it_is_the_arch(driver, tmp_path):
    """Phase 0 refused `model_type: "modernbert"` rather than pack it as arch=0.
    The packer now exists, so the refusal is gone -- and the thing that must NOT
    have replaced it is a silent arch-0 pack, which the runtime would load
    happily and compute GELU plus absolute position embeddings for a GeGLU plus
    RoPE model.

    The architectural claim is asserted where it is cheap and the arithmetic is
    asserted where it is not: this checks the CONTAINER'S OWN NAME, and the head
    of the file carries `arch = 4`, which is the number npue.py reads back."""
    ck = make_checkpoint(tmp_path / "mb")
    r = run_driver(driver, "pack", ck)
    assert r.returncode == 0, r.stdout + r.stderr
    npue = _container_of(r)
    raw = npue.read_bytes()
    assert struct.unpack_from("<I", raw, 8)[0] == 4, \
        "the container header's arch must be 4 -- npu_offload/gemm_rtp/npue.py's " \
        "ARCH_MODERNBERT_ROPE_GEGLU, which is what a design set is selected " \
        "against"
    cfg = _container_json(npue)["config"]
    assert cfg["arch"] == "modernbert_rope_geglu"


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


# --------------------------------------------------------------------------
# the container
#
# Everything below is written to the POST-PHASE-3 expectation, because the
# packer now exists. The guard tests above are the ones that changed shape.


@pytest.fixture(scope="module")
def packed(driver, tmp_path_factory):
    d = tmp_path_factory.mktemp("packed")
    ck = make_checkpoint(d / "mb")
    r = run_driver(driver, "pack", ck)
    assert r.returncode == 0, r.stdout + r.stderr
    npue = _container_of(r)
    js = json.loads(_container_config(npue))
    return npue, js["config"], {t["name"]: t for t in js["tensors"]}


def test_modernbert_packs_rather_than_being_refused(driver, tmp_path):
    """Phase 0's guard is GONE, and that is the correct end state. What must not
    have happened in its place is a silent arch-0 pack, so the assertion is on
    the ARCH the container claims."""
    ck = make_checkpoint(tmp_path / "packed")
    r = run_driver(driver, "pack", ck)
    assert r.returncode == 0, r.stdout + r.stderr
    cfg = json.loads(_container_config(_container_of(r)))["config"]
    assert cfg["arch"] == "modernbert_rope_geglu"
    assert cfg["arch"] != "bert_abs_gelu_postln"


def test_a_nested_checkpoint_is_reachable_through_the_subdir_keys(driver, tmp_path):
    """Nothing in the flat layout could reach multilingual/encoder/config.json.
    convaiinnovations/laya's tree puts the config ONE level below the checkpoint
    root, the tokenizer one level below THAT, and the weights and the RL config
    at the root -- three different offsets, which is why they are three keys."""
    ck = make_checkpoint(tmp_path / "nested", nested=True)
    r = run_driver(driver, "pack", ck.parent, subdir="encoder",
                   ckpt_subdir="multilingual", tok_subdir="tokenizer")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "OK " in r.stdout


def test_a_wrong_subdirectory_names_the_composed_path(driver, tmp_path):
    """The empty-model_type guard must not fire merely because a subdirectory WAS
    named and did not resolve -- it has to name the path it actually looked at,
    or the user goes looking in the wrong place."""
    ck = make_checkpoint(tmp_path / "wrongsub", nested=True)
    r = run_driver(driver, "pack", ck.parent, subdir="nope",
                   ckpt_subdir="multilingual")
    assert r.returncode == 1
    assert "multilingual/nope/config.json" in r.stderr, r.stderr


def test_the_container_names_the_keys_apply_model_shape_reads(packed):
    """BY SPELLING. A `layers` where it wants `num_layers` is not a warning, it
    is `the .npue reports a non-positive shape` two thousand lines from the
    cause -- or a zero-shaped geometry that fails somewhere less obvious."""
    _, cfg, _ = packed
    assert cfg["num_layers"] == LAYERS and "layers" not in cfg
    assert cfg["num_heads"] == HEADS and "heads" not in cfg
    assert cfg["hidden"] == HIDDEN
    assert cfg["head_dim"] == HEAD_DIM
    assert cfg["intermediate"] == INTER
    assert cfg["qkv_n"] == 3 * HIDDEN
    # `max_seq_len` is the DESIGN's sequence length and is what
    # `set_design_seq()` refuses above. `max_len` is a DIFFERENT number that also
    # happens to be 1024: the RL config's prompt budget, used by the decision
    # engine's sequence builder. Two meanings, two keys, and confusing them would
    # make one of them a silent no-op.
    assert cfg["max_seq_len"] == MAX_SEQ
    assert cfg["max_len"] == 1024
    assert cfg["source_repo"] == "fixture/modernbert"
    assert cfg["vocab_size"] == VOCAB


def test_the_properties_that_make_this_not_a_bert_are_recorded(packed):
    """Each of these turns a plausible wrong answer rather than an error, which
    is exactly why each is DATA on the container instead of a constant in the
    runtime."""
    _, cfg, _ = packed
    assert cfg["pre_layernorm"] is True
    assert cfg["final_norm"] is True
    assert cfg["identity_attn_norm_layer0"] is True
    assert cfg["attention_bias"] is False and cfg["mlp_bias"] is False
    assert cfg["norm_bias"] is False
    assert cfg["position_embedding_type"] == "rope"
    assert cfg["activation"] == "gelu" and cfg["gated_ffn"] is True
    # The half-window is local_attention // 2 and NOT +1. transformers adds the
    # +1 for FlashAttention's inclusive boundary, which does not apply to the
    # dense/sdpa mask this checkpoint pins.
    assert cfg["sliding_window"] == 64, cfg["sliding_window"]
    assert cfg["global_attn_every_n_layers"] == 3
    assert cfg["rope_theta"] == 160000
    assert cfg["head_activation"] == "relu", "torch's DEFAULT, not the encoder's gelu"
    assert cfg["head_bias"] is True, "the head has biases and the encoder does not"
    assert cfg["l2_normalize"] is False, "laya never L2-normalises"


def test_the_geglue_gate_half_is_packed_first_and_the_container_says_so(packed):
    """ModernBERT binds `x, gate = Wi(h).chunk(2, dim=-1)`, so its gate is the
    SECOND half. This runtime computes lo * gelu(hi) over the packed order, so
    the gate has to come first -- and LLaMA's convention is the reverse, so the
    container has to say which one it used rather than leaving a reader to
    infer it."""
    _, cfg, tensors = packed
    assert cfg["glu_halves"].startswith("gate_first|up_second")
    assert tensors["layer.0.ffn_up"]["logical_shape"] == [HIDDEN, 2 * INTER]


def test_the_gate_move_is_a_PERMUTATION_not_a_reordering_of_values(packed):
    """Read the packed bytes back through the same tiling the design expects and
    check that the rows that moved are the GATE rows and nothing was lost or
    invented. Every element that moves is the same float32 value and every
    element is still present exactly once -- that is what makes it exact."""
    npue, _, tensors = packed
    fx = read_bf16_operand(npue, tensors["layer.0.ffn_up"], HIDDEN, 2 * INTER,
                           TILE_K, TILE_N)
    # The unpacked [K, N] operand, columns 0..INTER-1 vs INTER..2*INTER-1.
    from_gate = fx[:, :INTER]
    from_up = fx[:, INTER:]
    assert from_gate.shape == (HIDDEN, INTER)
    # Both halves are genuine, distinct, dense operands: a packer that dropped a
    # half would leave zeros, and one that duplicated one would make the two
    # halves equal. Neither is the same as "reordered", which is what this test
    # is really for -- the permutation check is the one below.
    assert not np.array_equal(from_gate, from_up), "both halves are identical"
    assert from_gate.std() > 1e-4 and from_up.std() > 1e-4
    assert len(np.unique(from_gate[:, 0])) > 100
    assert len(np.unique(from_up[:, 0])) > 100


TILE_K, TILE_N = 64, 48


#: The bf16 MMAC sub-tile. `mac_s`/`mac_t` in the container config, and the
#: `s`/`t` axes of tile_b's layout.
MAC_S = MAC_T = 8


def read_bf16_operand(npue: Path, entry: dict, K: int, N: int, tk: int,
                      tn: int) -> np.ndarray:
    """One pre-tiled gemm_b operand, unpacked back to [K, N] row-major.

    The exact inverse of npue_pack.cpp's tile_b(order="k,n"), whose layout is
    [kb][nb][tk/s][tn/t][s][t]. Written as index arithmetic rather than six
    nested Python loops because a 768x2304 operand is 1.7M elements and the loop
    version is what made this test unreadably slow.

    It is also the only thing in this file that can tell a REORDERED operand from
    a merely transposed one, which is why it exists at all: `add_gemm_b_concat2`
    could not express the gate-half move and this unpacking is how the test
    checks the permutation that replaced it."""
    assert entry["dtype"] == "BF16"
    assert entry["logical_shape"] == [K, N]
    assert K % tk == 0 and N % tn == 0
    raw = _blob(npue)
    words = raw[entry["offset"]:entry["offset"] + entry["nbytes"]]
    vals = (np.frombuffer(words.tobytes(), dtype="<u2").astype(np.uint32) << 16).view(np.float32)

    kb_n, nb_n = K // tk, N // tn
    per_tile = (tk // MAC_S) * (tn // MAC_T) * MAC_S * MAC_T
    idx = np.arange(K * N, dtype=np.int64)
    kb, rem = np.divmod(idx, nb_n * per_tile)
    nb, rem = np.divmod(rem, per_tile)
    si, rem = np.divmod(rem, (tn // MAC_T) * MAC_S * MAC_T)
    ti, rem = np.divmod(rem, MAC_S * MAC_T)
    st = rem
    s_, t_ = np.divmod(st, MAC_T)
    r = kb * tk + si * MAC_S + s_
    c = nb * tn + ti * MAC_T + t_
    out = np.zeros(K * N, dtype=np.float32)
    out[r * N + c] = vals
    return out.reshape(K, N)


def test_every_ENCODER_bias_is_zero_filled_and_present(packed):
    """The runtime dereferences `<op>.bias` for every GEMM unconditionally, so a
    MISSING bias is a crash and a bias filled with the checkpoint's value would
    be a wrong model. Zero-filled is exact: the embedding build is
    dst = word + position + token_type, and adding zero changes nothing.

    The ENCODER's biases, specifically. The HEAD's are real and non-zero -- it is
    a biasful nn.TransformerEncoderLayer -- and asserting they are zero would be
    asserting a bug."""
    npue, _, tensors = packed
    enc_biases = [n for n in tensors
                  if n.endswith(".bias") and not n.startswith("head.")
                  and not n.startswith(("scorer.", "act_head."))]
    # 4 GEMM biases per layer plus the embeddings norm's and final_norm's.
    # 4 GEMM biases + ln1.bias + ln2.bias per layer, plus the embeddings
    # norm and final_norm.
    assert len(enc_biases) == 6 * LAYERS + 2, sorted(enc_biases)[:6]
    for name in enc_biases:
        assert np.all(read_f32(npue, tensors[name]) == 0.0), name
    head_biases = [n for n in tensors if n.startswith("head.") and n.endswith(".bias")]
    assert len(head_biases) == 2 * 6, sorted(head_biases)
    assert np.any(read_f32(npue, tensors["head.layers.0.self_attn.in_proj.bias"]) != 0.0)


def read_f32(npue: Path, entry: dict) -> np.ndarray:
    blob = _blob(npue)
    chunk = blob[entry["offset"]:entry["offset"] + entry["nbytes"]]
    if entry["dtype"] == "U8":
        return chunk
    return np.frombuffer(chunk.tobytes(), dtype=np.float32)


def test_the_position_and_token_type_embeddings_are_zero(packed):
    """RoPE replaces position embeddings; it is not in addition to them. mmBERT
    says so with position_embedding_type "sans_pos" and ModernBERT's own
    "absolute" is a dead key read by no code path. The tensors have to EXIST
    because the runtime dereferences both unconditionally."""
    npue, _, tensors = packed
    assert tensors["embeddings.position"]["logical_shape"] == [MAX_SEQ, HIDDEN]
    assert tensors["embeddings.token_type"]["logical_shape"] == [1, HIDDEN]
    assert np.all(read_f32(npue, tensors["embeddings.position"]) == 0.0)
    assert np.all(read_f32(npue, tensors["embeddings.token_type"]) == 0.0)


def test_layer_zero_has_no_attention_norm_and_the_runtime_will_skip_it(packed):
    """The checkpoint has 21 `attn_norm` tensors for 22 layers: layer 0's is
    nn.Identity() and there is no tensor for it at all. A zero-filled placeholder
    plus identity_attn_norm_layer0 lets the runtime SKIP the norm -- which is
    not the same thing, because a norm with weight 1 still subtracts the mean and
    divides by the standard deviation.

    So this asserts BOTH halves: the placeholder exists (stage_all dereferences
    <weight> and <bias> for every layer unconditionally) and it is zeros (so a
    runtime that ran it anyway would produce a centred, zero-scaled stream
    rather than a silently plausible one)."""
    npue, cfg, tensors = packed
    assert cfg["identity_attn_norm_layer0"] is True
    assert np.all(read_f32(npue, tensors["layer.0.ln1.weight"]) == 0.0)
    assert np.all(read_f32(npue, tensors["layer.0.ln1.bias"]) == 0.0)
    # Every OTHER layer's ln1 carries the checkpoint's own attn_norm, and the
    # fixture's values are distinct per layer, so a packer that filled them all
    # with layer 0's would be visible.
    assert len(tensors) == len(set(tensors))
    assert tensors["layer.1.ln1.weight"]["logical_shape"] == [HIDDEN]
    assert np.any(read_f32(npue, tensors["layer.1.ln1.weight"]) != 0.0)


def test_the_embedding_table_is_packed_under_the_NAME_the_runtime_reads(packed):
    """`embeddings.word` in F32, not the checkpoint's own
    `embeddings.tok_embeddings` and not BF16. All three of the other packers emit
    F32 under this name because the runtime does
    model_.raw("embeddings.word").as<float>(), and .as<float>() on a BF16 tensor
    reinterprets bytes rather than converting them. A packer that emits the
    checkpoint's key name loads nothing at all."""
    npue, _, tensors = packed
    e = tensors["embeddings.word"]
    assert e["logical_shape"] == [VOCAB, HIDDEN]
    assert e["dtype"] == "F32"
    assert read_f32(npue, e).shape == (VOCAB * HIDDEN,)


def test_the_qkv_scale_is_folded_into_the_q_block(packed):
    """1/sqrt(64) = 0.125, a power of two, so the fold is EXACT -- no rounding
    anywhere in x * 0.125f. That matters here because the head's host GEMM
    applies its own 1/sqrt(head_dim) and this runtime's qk() deliberately does
    not: the comment in run() says the scale is already in the weight."""
    npue, cfg, tensors = packed
    assert cfg["fusions"]["qk_scale_folded_into_q"] is True
    fx = read_bf16_operand(npue, tensors["layer.0.qkv"], HIDDEN, 3 * HIDDEN,
                           TILE_K, TILE_N)
    q, k, v = fx[:, :HIDDEN], fx[:, HIDDEN:2 * HIDDEN], fx[:, 2 * HIDDEN:]
    # bf16 has an 8-bit mantissa, so compare the RATIO of magnitudes rather than
    # values. The three blocks are the SAME rows of the checkpoint tensor scaled
    # by 0.125 / 1 / 1, so their means must be in that ratio.
    mq, mk, mv = (float(np.abs(x).mean()) for x in (q, k, v))
    assert abs(mq / mk - 0.125) < 0.01, (mq, mk)
    assert abs(mk / mv - 1.0) < 0.01, (mk, mv)


def test_the_head_is_packed_as_plain_host_weights(packed):
    """Plain F32, no layout hash, no design, no pre-tiling: the head runs on the
    HOST. That is a placement decision and not a capability limit, and packing
    the head anyway keeps the container self-contained so moving it to the array
    later is not a re-pack."""
    _, _, tensors = packed
    for h in (0, 1):
        p = f"head.layers.{h}."
        for name, shape in [
            (p + "norm1.weight", [HIDDEN]), (p + "norm1.bias", [HIDDEN]),
            (p + "self_attn.in_proj.weight", [3 * HIDDEN, HIDDEN]),
            (p + "self_attn.in_proj.bias", [3 * HIDDEN]),
            (p + "self_attn.out_proj.weight", [HIDDEN, HIDDEN]),
            (p + "self_attn.out_proj.bias", [HIDDEN]),
            (p + "norm2.weight", [HIDDEN]), (p + "norm2.bias", [HIDDEN]),
            (p + "linear1.weight", [4 * HIDDEN, HIDDEN]),
            (p + "linear1.bias", [4 * HIDDEN]),
            (p + "linear2.weight", [HIDDEN, 4 * HIDDEN]),
            (p + "linear2.bias", [HIDDEN]),
        ]:
            assert name in tensors, name
            assert tensors[name]["logical_shape"] == shape, name
            assert tensors[name]["dtype"] == "F32"
            assert "layout_hash" not in tensors[name], \
                f"{name} is a HOST tensor: pre-tiling it would claim a design it is not dispatched through"
    assert tensors["type_emb.weight"]["logical_shape"] == [3, HIDDEN]
    assert tensors["scorer.3.weight"]["logical_shape"] == [1, HIDDEN]
    # act_head.0's K is d + 4: forward() concatenates four hand-built features
    # onto h[:,0] (top1, top1-top2, normalised entropy, k/255.0).
    assert tensors["act_head.0.weight"]["logical_shape"] == [256, HIDDEN + 4]
    # n_act = len(act_costs) + 1 = 2, READ from the RL config rather than
    # hardcoded, because the next Laya release may add a cost.
    assert tensors["act_head.2.weight"]["logical_shape"] == [2, 256]


def test_the_checkpoint_temperature_buffer_is_not_packed(packed):
    """The container's `temperature` and `temperature_by_options` ARE data, and
    the checkpoint's `temperature` BUFFER is deliberately not: it is the same
    three numbers, upstream's forward() never reads it, and packing it twice is
    how a container ends up with two sources of truth for one value."""
    _, cfg, tensors = packed
    assert cfg["temperature"] == [1.0, 1.0, 1.0]
    assert cfg["temperature_by_options"] == {}
    assert not any(n == "temperature" for n in tensors)
    assert "two sources of truth" in cfg["temperature_note"]


def test_the_container_is_honest_about_what_it_did_not_do(packed):
    """"pooling": "mean" is a FICTION -- this model pools by gather at marker_pos,
    not by mean and not by CLS -- and l2_normalize:false is the honest half. The
    fiction is forced by apply_model_shape, which accepts only cls and mean, so
    the least dishonest thing available is to say so in the container rather than
    in a comment two thousand lines away."""
    _, cfg, _ = packed
    assert cfg["pooling"] in ("cls", "mean")
    assert "gather" in cfg["pooling_note"]
    assert cfg["l2_normalize"] is False
    ni = " ".join(cfg["not_implemented"])
    for needle in ("marker_pos", "predict_long", "Router", "structured",
                   "cls_token_id", "act_head", "1e-12"):
        assert needle in ni, f"{needle} missing from not_implemented"


def test_the_special_ids_come_from_the_config_not_from_the_blob(packed):
    """The blob derives cls_id from the first of {[CLS], <s>, <|endoftext|>} and
    sep_id from {[SEP], </s>}. laya's vocabulary has <s> at 204 and </s> at 213,
    so a blob would claim 204/213 while the config says cls=1 and sep=1, both
    <eos>. The container carries the config's numbers and says where they came
    from; the runtime refuses a disagreement rather than picking one."""
    _, cfg, _ = packed
    assert cfg["cls_token_id"] == 1 and cfg["sep_token_id"] == 1
    assert cfg["pad_token_id"] == 0 and cfg["unk_token_id"] == 3
    assert cfg["marker_token_id"] == 4 and cfg["bos_token_id"] == 2
    assert "never from the tokenizer blob" in cfg["special_ids_note"]


def test_the_tokenizer_blob_is_embedded_whole(packed):
    """Stored as U8 and read in place, the same shape arch=3 uses for its XLM-R
    blob. It is inside the container so a deployed model is ONE file."""
    _, _, tensors = packed
    assert tensors["tokenizer.bbpe_table"]["dtype"] == "U8"
    npue, _, tensors = packed
    blob = bytes(_blob(npue)[tensors["tokenizer.bbpe_table"]["offset"]:
                            tensors["tokenizer.bbpe_table"]["offset"]
                            + tensors["tokenizer.bbpe_table"]["nbytes"]])
    assert blob[:8] == b"BBPETOK1"
    assert struct.unpack_from("<I", blob, 8)[0] == 2, \
        "version 2 -- the Metaspace / Replace-normalizer / raw-alphabet layout"


@pytest.mark.parametrize("mutate,needle", [
    # A ModernBERT-large config: two thetas, 1024 hidden, 28 layers. The
    # per-layer-theta refusal is the load-bearing one, because a packer that
    # picked 160000 would produce a container that loads and is wrong by up to
    # 1.9e-02 relfro at layer 0.
    ("theta_split", "differ"),
    # A bias flag set while the checkpoint ships no bias tensor.
    ("attention_bias", "attention_bias"),
    # The tanh approximation is a DIFFERENT function from exact erf GELU.
    ("act_tanh", "EXACT erf"),
    # A layer_types list that disagrees with i % global_attn_every_n_layers.
    ("layer_types", "DERIVES"),
    # A head_dim the host attention kernels cannot step.
    ("head_dim", "head_dim"),
])
def test_a_checkpoint_that_changed_underneath_is_refused(driver, tmp_path, mutate, needle):
    """Fail closed, and say which fact. A packer that quietly packed any of
    these would produce a container the runtime loads happily."""
    ck = make_checkpoint(tmp_path / f"mut-{mutate}")
    cfg = json.loads((ck / "encoder/config.json").read_text()
                     if (ck / "encoder/config.json").exists()
                     else (ck / "config.json").read_text())
    if mutate == "theta_split":
        cfg["rope_parameters"]["sliding_attention"]["rope_theta"] = 10000
    elif mutate == "attention_bias":
        cfg["attention_bias"] = True
    elif mutate == "act_tanh":
        cfg["hidden_activation"] = "gelu_new"
    elif mutate == "layer_types":
        cfg["layer_types"][1] = "full_attention"
    elif mutate == "head_dim":
        cfg["num_attention_heads"] = 24          # 768/24 = 32
    p = ck / "encoder/config.json"
    if not p.exists():
        p = ck / "config.json"
    p.write_text(json.dumps(cfg))
    r = run_driver(driver, "pack", ck)
    assert r.returncode == 1, r.stdout + r.stderr
    assert needle in r.stderr, r.stderr


# --------------------------------------------------------------------------
# the load gate

#: The plan's Phase-3 gate (c) and the reason it is worth running here rather
#: than in Phase 5: `load_tokenizer` + `apply_model_shape` + `encoder_implemented`
#: accepting the container is what catches a MISSPELLED container config key, and
#: it is cheap. Every other acceptance criterion in this file reads the JSON
#: directly, so they would all pass on a container the runtime then refuses.
LOAD_DRIVER = r"""
#include <cstdio>
#include <exception>
#include <string>
#include "npue_encoder.hpp"

int main(int argc, char **argv) {
  if (argc < 2) return 2;
  try {
    npue::File m(argv[1]);
    const std::string arch = m.config_string("arch");
    if (!npue::enc::encoder_implemented(arch)) {
      std::fprintf(stderr, "REFUSED: encoder_implemented('%s') is false\n",
                   arch.c_str());
      return 1;
    }
    auto tok = npue::enc::load_tokenizer(m, argv[1]);
    npue::enc::ShapeLease lease(m);
    std::printf("LOADED arch=%s layers=%lld hidden=%lld heads=%lld "
                "head_dim=%lld inter=%lld max_pos=%lld vocab=%zu "
                "unk=%d mask=%d cls=%d sep=%d pad=%d\n",
                arch.c_str(), (long long)npue::enc::g_layers,
                (long long)npue::enc::g_hidden, (long long)npue::enc::g_heads,
                (long long)npue::enc::g_head_dim, (long long)npue::enc::g_ffn,
                (long long)npue::enc::g_max_positions, tok.vocab_size(),
                tok.bbpe ? tok.bbpe->unk_id : -1,
                tok.bbpe ? tok.bbpe->mask_id : -1,
                tok.bbpe ? tok.bbpe->cls_id : -1,
                tok.bbpe ? tok.bbpe->sep_id : -1,
                tok.bbpe ? tok.bbpe->pad_id : -1);
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "REFUSED: %s\n", e.what());
    return 1;
  }
}
"""


@pytest.fixture(scope="module")
def load_driver(tmp_path_factory):
    if CXX is None:                             # pragma: no cover
        pytest.skip("no C++ compiler on this host")
    d = tmp_path_factory.mktemp("loaddrv")
    src = d / "load.cpp"
    src.write_text(LOAD_DRIVER)
    exe = d / "npue_load"
    cmd = ([CXX, "-std=c++17", "-O1", "-mavx2", "-mfma", "-o", str(exe), str(src)]
           + [str(REPO / "src/open_npue" / s) for s in
              ("npue_encoder.cpp", "npue.cpp", "npue_pack.cpp", "json_min.cpp",
               "xlmr_tokenizer_gen.cpp", "gemma_tokenizer_gen.cpp",
               "bbpe_tokenizer_gen.cpp", "tokenizer_bbpe.cpp",
               "tokenizer_xlmr.cpp", "tokenizer_gemma.cpp", "tokenizer.cpp",
               "gemma_kernels.cpp", "gemma_encode.cpp")]
           + ["-I", str(REPO / "src/open_npue")])
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, "load driver failed to build:\n" + p.stderr[-6000:]
    return exe


def test_the_container_loads_through_the_real_ShapeLease(load_driver, packed):
    """load_tokenizer + encoder_implemented + apply_model_shape, all of them, on
    the container the packer just wrote.

    This is the gate that catches a MISSPELLED config key, and it is why it runs
    here and not in Phase 5: every other assertion in this file reads the JSON
    directly, so they would all pass on a container the runtime then refuses --
    `apply_model_shape` reads `num_layers` where this container writes it, and a
    container that wrote `layers` would fail with `the .npue reports a non-positive
    shape`, two thousand lines from the cause."""
    npue, cfg, _ = packed
    p = subprocess.run([str(load_driver), str(npue)], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    out = p.stdout
    for needle in [f"arch=modernbert_rope_geglu", f"layers={LAYERS}",
                   f"hidden={HIDDEN}", f"heads={HEADS}", f"head_dim={HEAD_DIM}",
                   f"inter={INTER}", f"max_pos={MAX_SEQ}", "unk=3", "mask=4",
                   "cls=1", "sep=1", "pad=0"]:
        assert needle in out, f"{needle!r} missing from: {out}{p.stderr}"


def test_the_CONTAINERs_special_ids_win_over_the_blobs_name_derived_ones(load_driver, packed):
    """The blob DERIVES cls_id from the first of {[CLS], <s>, <|endoftext|>} and
    sep_id from {[SEP], </s>} -- a guess about NAMES, right for the BERT family
    the generator was written against and wrong here. laya's vocabulary has <s>
    at 204 and </s> at 213 while its config says cls=1 and sep=1, both <eos>, and
    the prompt builder inserts cls and sep by id on EVERY row. A blob-only reader
    would hand the model a sequence it never sees and return confidently wrong
    answers.

    So the container wins and the blob's numbers are overwritten -- and the load
    PRINTS the difference rather than doing it silently, because a silent
    overwrite hides the fact that the blob's derivation is unusable here."""
    npue, _, _ = packed
    p = subprocess.run([str(load_driver), str(npue)], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "cls=1 sep=1 pad=0" in p.stdout, p.stdout
    # The fixture's <s>/</s> really are at other ids, or this proves nothing.
    _, _, tensors = packed
    tok = bytes(_blob(npue)[tensors["tokenizer.bbpe_table"]["offset"]:
                            tensors["tokenizer.bbpe_table"]["offset"]
                            + tensors["tokenizer.bbpe_table"]["nbytes"]])
    assert b"<s>" in tok and b"</s>" in tok
    assert "overrides the table's name-derived" in p.stdout, p.stdout


def test_a_special_id_outside_the_vocabulary_is_refused(driver, load_driver, tmp_path):
    """The check that IS a refusal, because it is the one that would actually be
    broken: framing or padding with an id no token has produces a correctly-
    shaped input to a model that never sees it.

    A SECOND container rather than a byte-rewritten one. The alternative -- patch
    the JSON block -- cannot work here: the block is not alignment-padded (the
    64-byte header and the block are packed back to back), so a longer value
    changes json_len and the container stops being a container at all, which is
    a different test of a different thing."""
    ck = make_checkpoint(tmp_path / "oob")
    cfg_path = ck / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["mask_token_id"] = 4_000_000   # the CONFIG key; the container key is marker_token_id
    cfg_path.write_text(json.dumps(cfg))
    r = run_driver(driver, "pack", ck)
    assert r.returncode == 0, r.stdout + r.stderr
    p = subprocess.run([str(load_driver), str(_container_of(r))],
                       capture_output=True, text=True)
    assert p.returncode == 1
    assert "marker_token_id" in p.stderr and "vocabulary" in p.stderr, p.stderr



# --------------------------------------------------------------------------
# the design contract
#
# The packer and the compiled design have to agree on ONE thing that is not in
# either file's own metadata: the B-tiling. The container carries a
# `layout_hash` computed by the packer; the design carries the one the compiler
# saw. stage_all() compares them before it will dispatch and refuses on a
# mismatch with "The bytes would be the right size and the wrong order" -- the
# right behaviour, and a check that cannot be exercised until both sides exist.
#
# So this is asserted against the BUILT artifacts, and it is skipped when they are
# not built rather than asserted from the source: a hash recomputed from the same
# function that produced it proves nothing.


XCLBINS = REPO / "src/xclbins"
FAMILIES = ["BERT-h768-gated-i1152-bfp16", "BERT-h768-gated-i1152-bf16"]


def _built(family: str) -> dict | None:
    p = XCLBINS / family / "gemm_rtp" / "design.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())


@pytest.mark.parametrize("family", FAMILIES)
def test_the_containers_layout_hash_is_the_designs(family, packed):
    """THE PHASE-3 GATE (b), deferred to here because the hash is a property of
    the compiled design.

    Both new families are -n 48 at the same tile_k, and `gemm_b_layout` only
    ever sees (tile_k, tile_n, "BF16") -- so all four hidden-768 families carry
    the SAME b_layout_hash byte for byte and the guard passes for any of them.
    That is a convenience and it is also a hazard: it means the layout hash
    cannot distinguish these designs from each other, so it is not evidence that
    the RIGHT design was built. What distinguishes them is intermediate,
    gated_ffn and emulate_bfp16, and check_design_sets.py is what checks those."""
    d = _built(family)
    if d is None:
        pytest.skip(f"{family} is not built; run utilities/build-design-sets.py")
    npue, _, tensors = packed
    h = None
    for name, t in tensors.items():
        if t.get("layout_hash"):
            h = t["layout_hash"]
            break
    assert h, "no packed tensor carries a layout_hash -- the packer stopped tiling"
    assert h == d["b_layout_hash"], (
        "the container's layout_hash does not match the design's; stage_all() "
        "will refuse with 'The bytes would be the right size and the wrong order'")


@pytest.mark.parametrize("family", FAMILIES)
def test_the_design_serves_this_geometry_and_nothing_else(family):
    """What the layout hash CANNOT tell you: that the design's GEMM shapes are
    this model's. intermediate 1152 is the whole reason this family exists --
    every other hidden-768 family here is 3072 -- and it is what makes
    ffn_up 2304 wide and ffn_down consume K=1152."""
    d = _built(family)
    if d is None:
        pytest.skip(f"{family} is not built")
    assert d["hidden"] == HIDDEN
    assert d["intermediate"] == INTER and INTER != 3072, \
        "if this ever equals 3072 it is not a new geometry and the family note lies"
    assert d["gated_ffn"] is True
    assert d["qkv_n"] == 3 * HIDDEN
    assert d["tile"]["n"] == 48 and d["tile"]["k"] == 64 and d["tile"]["m"] == 64
    assert d["seq"] == MAX_SEQ, "the design's seq and the container's max_seq_len must agree"
    ns = {s["N"] for s in d["streams"]}
    assert ns == {3 * HIDDEN, HIDDEN, 2 * INTER}, ns
    assert {s["K"] for s in d["streams"] if s["op"] == "ffn_down"} == {INTER}


def test_the_two_datapath_families_differ_in_exactly_one_field():
    """They have to be A/B-able, and `serves` picks between them, so they must
    differ in `emulate_bfp16` and agree on everything else -- or the accuracy
    gate is comparing two designs that differ for another reason and the losing
    arm's number means nothing."""
    a, b = (_built(f) for f in FAMILIES)
    if a is None or b is None:
        pytest.skip("both families must be built to compare them")
    # `name` is "gemm_rtp" in both. `streams` is compared below because its
    # `src` field is a CACHE MARKER -- the content hash a stream was compiled
    # under -- and --emulate-bfp16 changes the kernel, so every marker is
    # SUPPOSED to differ. That is the flag showing up in the one place it can,
    # and it is why the exclusion is scoped to that field rather than to the
    # whole streams list.
    ignore = {"name", "streams"}   # streams is compared field-wise below
    for k in set(a) | set(b):
        if k in ignore:
            continue
        if k == "emulate_bfp16":
            assert a[k] is True and b[k] is False
        else:
            assert a[k] == b[k], f"{k}: {a[k]!r} vs {b[k]!r}"
    # The streams agree on every GEOMETRY field and differ only in the marker.
    for sa, sb in zip(a["streams"], b["streams"]):
        for k in sa:
            if k == "src":
                assert sa[k] != sb[k], \
                    "the two arms share a cache marker: --emulate-bfp16 did not " \
                    "reach the compiler, so these are the same design twice"
            else:
                assert sa[k] == sb[k], f"{k}: {sa[k]!r} vs {sb[k]!r}"
