#!/usr/bin/env python3
"""A numpy oracle for convaiinnovations/laya's encoder, for the Phase 5 gate.

It reads the CHECKPOINT's own safetensors -- not the .npue container -- and
computes the pre-LN ModernBERT forward pass in float64, so it is an independent
reading of the model rather than a second run of the code under test. Every
choice the plan's twelve gotchas name is written out here, and the ones that
are not exercised are refused rather than skipped.

Why float64 and not float32: the question is whether the NPU's bf16 operands
reproduce the arithmetic, so the reference has to be exact enough that its own
error is not the thing being measured. float64 keeps it ~1e-12.

The band mask is a flag, not a constant, because Phase 5 runs the loop with
full attention and Phase 6 turns the band on. The same oracle covers both, and
the caller is expected to say which it is asking for.

Usage, as the gate uses it:

    python3 utilities/laya_preln_reference.py encode \\
        --checkpoint /home/ankk98/models/laya-multilingual \\
        --ids 1,2,714,7679 \\
        --band 0 \\
        --out /tmp/ref.bin
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

# --------------------------------------------------------------------------
# safetensors, read directly.
#
# The format is an 8-byte little-endian header length, that many bytes of JSON
# describing {"name": {"dtype", "shape", "data_offsets"}}, then the payload --
# and the `safetensors` wheel has no CPython 3.14 build, so it is 30 lines
# here rather than a dependency that cannot be installed.
_DTYPES = {"F16": np.float16, "F32": np.float32, "BF16": None, "F64": np.float64,
           "I64": np.int64, "I32": np.int32, "I8": np.int8, "U8": np.uint8,
           "BOOL": np.bool_}


def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name, meta in head.items():
            if name == "__metadata__":
                continue
            dt = meta["dtype"]
            if dt == "BF16":
                # numpy has no bfloat16. Read as u16 and convert by hand; the
                # checkpoint is F16 throughout, so this is a guard, not a path.
                raise SystemExit(f"BF16 tensor {name}: no numpy bfloat16 here")
            arr = np.frombuffer(f.read(meta["data_offsets"][1] - meta["data_offsets"][0]),
                                dtype=_DTYPES[dt])
            out[name] = arr.reshape(meta["shape"]).copy()
    return out


# bfloat16 is 1 sign + 8 exponent + 7 MANTISSA bits, so 16 of a float32's 23
# mantissa bits are discarded. (Discarding 23 would keep one mantissa bit --
# tf32, a lossier model than the one under test, which would set the gate's
# threshold below the engine's real noise and pass a broken encoder.)
# The two roots, set from the command line. laya's tree is NESTED and the two
# subdirectories are not one prefix: the config and the weights are at different
# depths under the served root.
ROOT = {"ckpt": "multilingual", "config": "encoder"}

BF16_DISCARD = 16
BF16_KEEP = np.uint64(0xFFFF0000)
BF16_HALF = np.uint64(0x8000)


def bf16(x):
    """Round to bfloat16, round-half-to-even, in float64.

    The NPU takes its GEMM operands in bf16 and accumulates in fp32, so the
    REPLICA has to round exactly where the design rounds and nowhere else. That
    is the whole point of it: a replica that also rounded the attention and the
    softmax would be a different model, and a gate calibrated against it would
    forgive exactly the errors the gate exists to catch.

    Through float32 rather than a uint64 view, because numpy has no bfloat16
    and a uint64 view of a float64 array reinterprets 8-byte elements as
    4-byte ones -- which returns numbers of the wrong shape, not an error.

    The rounding is done in the integer domain because the carry out of the
    mantissa into the exponent is the entire mechanism, and a float add would
    lose it: 0x7FFF biases everything below the halfway point down and the tie
    up, then the retained LSB decides the tie, which is round-half-to-even.
    """
    y32 = np.ascontiguousarray(x, dtype=np.float32)
    bits = y32.view(np.uint32).astype(np.uint64)
    low = bits & np.uint64(0xFFFF)
    lsb = (bits >> BF16_DISCARD) & 1
    # Carry into the retained field: above the halfway point always, and at
    # exactly the halfway point only when the retained value is ODD (so that a
    # tie lands on the even neighbour).
    carry = (low > BF16_HALF) | ((low == BF16_HALF) & (lsb == 1))
    r = np.uint64(0x7FFF) + carry.astype(np.uint64)
    out = ((bits + r) & np.uint64(0xFFFFFFFF) & BF16_KEEP).astype(np.uint32)
    return out.view(np.float32).astype(np.float64)


def gelu_exact(x):
    """0.5*x*(1+erf(x/sqrt(2))), the erF not the tanh approximation.

    scipy is not available, and math.erf is scalar. numpy has no erf, so this
    uses the identity erf(x) = 2*Phi(x*sqrt(2)) - 1 via math.erf through
    np.vectorize -- slow, and it does not matter: this is a 768-wide gate on
    four rows, not a training loop.
    """
    import math
    return 0.5 * x * (1.0 + np.vectorize(math.erf, otypes=[np.float64])(x * 0.70710678118654752440))


def layer_norm(x, w, b, eps):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def rope_tables(seq, head_dim, theta):
    # inv_freq_i = theta^(-i/head_dim) for i in 0..head_dim/2-1, NeoX
    # half-split: the first head_dim/2 dims pair with the last head_dim/2.
    i = np.arange(head_dim // 2, dtype=np.float64)
    inv = theta ** (-i / head_dim)
    pos = np.arange(seq, dtype=np.float64)
    ang = pos[:, None] * inv[None, :]
    cos = np.cos(ang)
    sin = np.sin(ang)
    return cos, sin


def apply_rope(t, cos, sin):
    """t is [S, Nh, Dh]; the half-split rotation, on Q and K only, never V."""
    S, Nh, Dh = t.shape
    a = t[:, :, : Dh // 2]
    b = t[:, :, Dh // 2:]
    # A head AXIS for the tables. The rotation depends on the position and the
    # channel pair and on nothing else, so the head axis is broadcast -- and
    # getting that wrong is a shape error rather than a silent one, which is
    # the only kind of broadcast mistake that announces itself.
    c = cos[:, None, :]
    s = sin[:, None, :]
    return np.concatenate([a * c - b * s, b * c + a * s], axis=-1)


class Config:
    def __init__(self, cfg):
        self.d = cfg
        h = cfg["hidden_size"]
        self.hidden = h
        self.layers = cfg["num_hidden_layers"]
        self.heads = cfg["num_attention_heads"]
        self.head_dim = cfg.get("head_dim", h // cfg["num_attention_heads"])
        self.inter = cfg["intermediate_size"]
        self.eps = cfg.get("layer_norm_eps", 1e-5)
        rope = cfg.get("rope_parameters", {})
        full = rope.get("full_attention", {})
        slide = rope.get("sliding_attention", {})
        tf = float(full.get("rope_theta", cfg.get("rope_theta")))
        ts = float(slide.get("rope_theta", cfg.get("rope_theta")))
        if abs(tf - ts) > 0:
            raise SystemExit(
                f"two RoPE thetas ({tf} global, {ts} local). This oracle "
                "implements the single-table path only, because the packer "
                "refuses this case; a model that reaches here has already "
                "diverged from what was validated.")
        self.theta = tf
        local = int(cfg.get("local_attention", 0))
        self.band = local // 2
        every = int(cfg["global_attn_every_n_layers"])
        self.layer_types = ["full_attention" if l % every == 0 else "sliding_attention"
                            for l in range(self.layers)]
        if cfg.get("layer_types"):
            got = cfg["layer_types"]
            if got != self.layer_types:
                raise SystemExit(
                    f"layer_types {got} disagrees with the derived "
                    f"{self.layer_types}; ModernBERT DERIVES them, and a "
                    "disagreement means one of the two is wrong.")


def encode(root, ids, band_on, seq=None, bf16_replica=False, want_layers=False):
    cfg = json.load(open(os.path.join(root, ROOT["config"], "config.json")))
    C = Config(cfg)
    t = load_safetensors(os.path.join(root, ROOT["ckpt"], "model.safetensors"))
    t["_root"] = root
    t["_cfg"] = os.path.join(root, ROOT["config"])
    S = len(ids) if seq is None else seq
    H, Nh, Dh, I, L = C.hidden, C.heads, C.head_dim, C.inter, C.layers

    def g(*names):
        for n in names:
            if n in t:
                return t[n].astype(np.float64)
        raise SystemExit(f"no tensor among {names}; the checkpoint layout moved")

    def gn(w, b, default_zero=True):
        """A norm's gamma and beta. This checkpoint's norms are BIAS-FREE --
        `norm_bias: false`, and the safetensors header carries no *.bias under
        encoder/ at all -- so the missing beta is a zero of the right shape
        rather than a missing tensor. Reading it as absent is the point: a
        reader that demanded both would refuse a model that is complete."""
        if w not in t:
            raise SystemExit(f"no {w}")
        if b in t:
            return t[w].astype(np.float64), t[b].astype(np.float64)
        return t[w].astype(np.float64), np.zeros(t[w].shape, dtype=np.float64)

    # Embeddings. RoPE REPLACES the absolute table (research gotcha 12), and
    # there is no token_type table either: the row is zero, which is why the
    # packer emits a zero-filled `embeddings.token_type` and the engine adds it.
    x = g("encoder.embeddings.tok_embeddings.weight")[list(ids)]
    x = layer_norm(x, *gn("encoder.embeddings.norm.weight",
                          "encoder.embeddings.norm.bias"), C.eps)

    cos, sin = rope_tables(S, Dh, C.theta)
    pos = np.arange(S)
    band = C.band if band_on else 0
    # RIGHT padding, position_ids = arange(S) (gotcha 8). The caller's ids are
    # the real prefix; the rest is pad and the padding mask removes it.
    keep = np.zeros(S, dtype=bool)
    keep[: len(ids)] = True
    neg = np.where(keep, 0.0, -1.0e30)

    # `r` is the rounding function: identity in fp64, bfloat16 in the replica.
    # It is applied to BOTH the operands and the OUTPUT of every GEMM, because
    # the design rounds both: the weights are tiled in bf16, the activations are
    # converted to bf16 on the way in, and the accumulator is WRITTEN BACK as
    # bf16 -- which the runtime says out loud on its own status line
    # ("datapath bf16 MMAC, C as bf16"). Rounding only the operands leaves the
    # replica an order of magnitude closer to the exact answer than the engine
    # is, and a gate calibrated against that would forgive the very noise it
    # exists to bound.
    #
    # What stays fp32, because the runtime keeps it there: the residual stream,
    # the softmax, and the attention scores. `scores` is not a GEMM output --
    # it is a host dot product -- and the AV accumulate is fp32 too.
    r = bf16 if bf16_replica else (lambda z: z)

    def gemm(w, h):
        """[K, N] @ [N, S] -> [K, S], operands and accumulator in bf16."""
        return r(r(w) @ r(h).T)

    per_layer = [] if want_layers else None
    for l in range(L):
        p = f"encoder.layers.{l}."
        # --- attention, PRE-LN. Layer 0's attn_norm is nn.Identity().
        if l > 0:
            if p + "attn_norm.weight" not in t:
                raise SystemExit(
                    f"layer {l} has no attn_norm, so it is not merely layer 0's "
                    "identity -- the checkpoint's norm count changed and every "
                    "layer index after this one is suspect.")
            h = layer_norm(x, *gn(p + "attn_norm.weight",
                                  p + "attn_norm.bias"), C.eps)
        else:
            h = x
        # (3, Nh, Dh) is the 3 as the OUTER stride (gotcha 5).
        qkv = gemm(g(p + "attn.Wqkv.weight"), h).T.reshape(S, 3, Nh, Dh)
        q = apply_rope(qkv[:, 0], cos, sin)
        k = apply_rope(qkv[:, 1], cos, sin)
        v = qkv[:, 2]
        scores = np.einsum("ihd,jhd->hij", q, k) / np.sqrt(Dh)
        if band and C.layer_types[l] == "sliding_attention":
            far = np.abs(pos[:, None] - pos[None, :]) > band
            scores = np.where(far[None, :, :], -1.0e30, scores)
        scores = scores + neg[None, None, :]
        # Softmax in fp32 (gotcha 7) -- here in float64 throughout, which is
        # strictly more exact.
        scores = scores - scores.max(-1, keepdims=True)
        e = np.exp(scores)
        p_attn = e / e.sum(-1, keepdims=True)
        ctx = np.einsum("hij,jhd->ihd", p_attn, v).reshape(S, H)
        x = x + gemm(g(p + "attn.Wo.weight"), ctx).T

        # --- FFN, PRE-LN. x_, gate_ = Wi(h).chunk(2, -1): the GATE is the
        # SECOND half (gotcha 1), and this is the line that the engine's
        # packed gate_first order exists to reproduce.
        h = layer_norm(x, *gn(p + "mlp_norm.weight", p + "mlp_norm.bias"), C.eps)
        wi = gemm(g(p + "mlp.Wi.weight"), h)
        # Split on the OUTPUT axis, which is the chunk(2, -1) axis. Slicing by
        # S here would silently take the first S rows of a [2I, S] matrix --
        # the shape error only shows up because I is not S.
        up, gate = wi[:I].T, wi[I:].T
        act = gelu_exact(gate) * up
        x = x + gemm(g(p + "mlp.Wo.weight"), act).T
        if per_layer is not None:
            per_layer.append(x.copy())

    # final_norm, and its OUTPUT is the encoder's output (upstream reads
    # last_hidden_state, not the residual stream).
    out = layer_norm(x, *gn("encoder.final_norm.weight",
                            "encoder.final_norm.bias"), C.eps)
    return (out, per_layer) if want_layers else out


# --------------------------------------------------------------------------
# the decision head
#
# Separated from the encoder above because the Phase 7 gate has to be able to
# blame one or the other: fed the SAME ids the engine used, this answers "is the
# engine's head arithmetic right", and the prompt builder is then the only
# remaining suspect. Run together they answer neither.
# --------------------------------------------------------------------------

def _ln(x, w, b=None, eps=1e-5):
    """LayerNorm with an OPTIONAL beta. This checkpoint's norms are bias-free --
    `norm_bias: false`, and its safetensors header carries no encoder `*.bias`
    at all -- so a required beta is a refusal against a complete model."""
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    y = (x - mu) / np.sqrt(var + eps) * w
    return y if b is None else y + b


def _gb(t, wkey, bkey):
    """A norm's gamma/beta pair, with the beta OPTIONAL. See `_ln`."""
    if wkey not in t:
        raise SystemExit(f"no {wkey}")
    return (t[wkey].astype(np.float64),
            t[bkey].astype(np.float64) if bkey in t else None)


def head(t, ids, markers, qtype, keep=None):
    """The 2 head layers + the scorer, in float64, on the checkpoint's weights.

    `t` is the loaded safetensors dict, `ids` a single row, `markers` that row's
    marker positions, `qtype` its index into type_emb.

    Mirrors the engine exactly, including the three places the two could
    disagree: ReLU (not GELU) in the head FFN, the explicit 1/sqrt(head_dim) on
    Q, and LayerNorm eps 1e-5 (torch's default) rather than the encoder's 1e-12.
    """
    cfg = json.load(open(os.path.join(t["_root"], t["_cfg"], "config.json")))
    C = Config(cfg)
    d = C.hidden
    S = len(ids)
    x = t["encoder.embeddings.tok_embeddings.weight"][list(ids)].astype(np.float64)
    x = _ln(x, *_gb(t, "encoder.embeddings.norm.weight", "encoder.embeddings.norm.bias"), eps=C.eps)
    cos, sin = rope_tables(S, C.head_dim, C.theta)
    pos = np.arange(S)
    keepm = np.ones(S, dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    neg = np.where(keepm, 0.0, -1.0e30)
    Nh, Dh, L, I = C.heads, C.head_dim, C.layers, C.inter
    band = C.band

    for l in range(L):
        p = f"encoder.layers.{l}."
        h = x if l == 0 else _ln(x, *_gb(t, p + "attn_norm.weight", p + "attn_norm.bias"), eps=C.eps)
        qkv = (t[p + "attn.Wqkv.weight"].astype(np.float64) @ h.T).T.reshape(S, 3, Nh, Dh)
        q = apply_rope(qkv[:, 0], cos, sin)
        k = apply_rope(qkv[:, 1], cos, sin)
        v = qkv[:, 2]
        sc = np.einsum("ihd,jhd->hij", q, k) / np.sqrt(Dh)
        if band and C.layer_types[l] == "sliding_attention":
            sc = np.where((np.abs(pos[:, None] - pos[None, :]) > band)[None, :, :],
                          -1.0e30, sc)
        sc = sc + neg[None, None, :]
        sc = sc - sc.max(-1, keepdims=True)
        e = np.exp(sc)
        ctx = np.einsum("hij,jhd->ihd", e / e.sum(-1, keepdims=True), v).reshape(S, d)
        x = x + (t[p + "attn.Wo.weight"].astype(np.float64) @ ctx.T).T
        h = _ln(x, *_gb(t, p + "mlp_norm.weight", p + "mlp_norm.bias"), eps=C.eps)
        wi = t[p + "mlp.Wi.weight"].astype(np.float64) @ h.T
        up, gate = wi[:I].T, wi[I:].T
        x = x + (t[p + "mlp.Wo.weight"].astype(np.float64) @ (gelu_exact(gate) * up).T).T
    h = _ln(x, *_gb(t, "encoder.final_norm.weight", "encoder.final_norm.bias"), eps=C.eps)

    # --- the head. type_emb first, broadcast over the sequence.
    h = h + t["type_emb.weight"].astype(np.float64)[qtype]
    for l in range(cfg.get("head_layers", 2)):
        p = f"head.layers.{l}."
        n = _ln(h, *_gb(t, p + "norm1.weight", p + "norm1.bias"))
        # PyTorch's names. The CONTAINER renames in_proj -> in_proj.weight and
        # out_proj -> out_proj.weight (it collides with the encoder's
        # attn_out otherwise), so a reader that used the container's names
        # against the checkpoint gets a KeyError -- which is the cheap failure,
        # unlike the reverse.
        ip = t[p + "self_attn.in_proj_weight"].astype(np.float64)
        ib = t[p + "self_attn.in_proj_bias"].astype(np.float64)
        proj = (n @ ip.T + ib).reshape(S, 3, Nh, Dh)     # [Q|K|V], plain MHA
        q, k, v = proj[:, 0], proj[:, 1], proj[:, 2]
        q = q / np.sqrt(Dh)                              # EXPLICIT, on Q only
        sc = np.einsum("ihd,jhd->hij", q, k)              # no band, no RoPE
        sc = sc + neg[None, None, :]
        sc = sc - sc.max(-1, keepdims=True)
        e = np.exp(sc)
        ctx = np.einsum("hij,jhd->ihd", e / e.sum(-1, keepdims=True), v).reshape(S, d)
        h = h + (ctx @ t[p + "self_attn.out_proj.weight"].astype(np.float64).T
                 + t[p + "self_attn.out_proj.bias"].astype(np.float64))
        n = _ln(h, *_gb(t, p + "norm2.weight", p + "norm2.bias"))
        f = np.maximum(0.0, n @ t[p + "linear1.weight"].astype(np.float64).T
                       + t[p + "linear1.bias"].astype(np.float64))   # ReLU
        h = h + (f @ t[p + "linear2.weight"].astype(np.float64).T
                 + t[p + "linear2.bias"].astype(np.float64))

    m = h[np.asarray(markers, dtype=int)]
    m = _ln(m, *_gb(t, "scorer.0.weight", "scorer.0.bias"))
    m = gelu_exact(m @ t["scorer.1.weight"].astype(np.float64).T
                   + t["scorer.1.bias"].astype(np.float64))
    return (m @ t["scorer.3.weight"].astype(np.float64).T
            + t["scorer.3.bias"].astype(np.float64)).reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ids", default="", help="comma-separated token ids")
    ap.add_argument("--seq", type=int, default=0, help="0 = one row per id")
    ap.add_argument("--band", type=int, default=0, choices=[0, 1])
    ap.add_argument("--bf16", type=int, default=0, choices=[0, 1],
                    help="round the GEMM operands to bfloat16: the replica the "
                         "plan's threshold is calibrated from")
    ap.add_argument("--out", default="")
    ap.add_argument("--head", default="",
                    help="markers,qtype for the head, e.g. '12,20;2'")
    ap.add_argument("--checkpoint-subdir", default="multilingual",
                    help="where the weights are, relative to --checkpoint")
    ap.add_argument("--config-subdir", default="encoder",
                    help="where config.json is, relative to the CHECKPOINT")
    ap.add_argument("--ids-file", default="",
                    help="read the ids from the engine's own output, so the two "
                         "sides are provably running the same prompt")
    ap.add_argument("--dump", default="", help="a .bin the engine wrote")
    a = ap.parse_args()
    if a.ids_file:
        ids = [int(x) for x in open(a.ids_file).read().split()]
    else:
        ids = [int(x) for x in a.ids.split(",") if x.strip()]
    if not ids:
        raise SystemExit(
            "no ids: pass --ids or --ids-file. Reading the ENGINE's ids with "
            "--ids-file is what makes the head comparison a test of the head "
            "rather than of the prompt builder; with --ids the two sides could "
            "disagree about the prompt and the comparison would blame the head "
            "for a prompt bug.")
    ROOT["ckpt"] = a.checkpoint_subdir
    ROOT["config"] = os.path.join(a.checkpoint_subdir, a.config_subdir)
    out = encode(a.checkpoint, ids, a.band, a.seq or None, bool(a.bf16))
    print(f"# reference shape {out.shape} band={a.band} bf16_replica={a.bf16}",
          file=sys.stderr)
    if a.dump:
        eng = np.fromfile(a.dump, dtype=np.float32).reshape(-1, out.shape[1])
        eng = eng[: out.shape[0]].astype(np.float64)
        ref = out[: eng.shape[0]]
        num = float((eng * ref).sum())
        den = float(np.sqrt((eng ** 2).sum()) * np.sqrt((ref ** 2).sum()))
        print(f"cosine        {num / den:.9f}")
        d = np.abs(eng - ref)
        print(f"max_abs_diff  {d.max():.6g}   mean_abs_diff {d.mean():.6g}")
        rel = d.max() / max(1e-12, np.abs(ref).max())
        print(f"max_rel_diff  {rel:.6g}")
    if a.head:
        mk, qt = a.head.split(";")
        tt = load_safetensors(
            os.path.join(a.checkpoint, a.checkpoint_subdir, "model.safetensors"))
        tt["_root"] = a.checkpoint
        tt["_cfg"] = os.path.join(a.checkpoint_subdir, a.config_subdir)
        # The oracle runs the REAL PREFIX only, so every position it has is
        # real. (An earlier version indexed this by token id, which is a
        # different thing entirely -- ids reach 235337, so it built a quarter of
        # a million mask entries and then failed to broadcast.)
        logits = head(tt, ids, [int(x) for x in mk.split(",")], int(qt),
                      keep=[True] * len(ids))
        print("# head logits " + " ".join("%.7g" % v for v in logits))
    if a.out:
        out.astype(np.float32).tofile(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
