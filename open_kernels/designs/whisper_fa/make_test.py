"""Test vectors + fp64 reference for the fused-attention design in this directory.

    python make_test.py [--valid-len N] [--seed S]

Layout follows src/open_whisper/fa_attention.cpp exactly, because that is the
production host contract and getting it wrong would produce a meaningless
comparison:

  - four device buffers, all bf16: Q, K, V in and O out.
  - each is head-first [heads][seq_pad][head_dim], i.e.
        offset(h, t) = h * seq_pad * head_dim + t * head_dim
    (repack_qkv()'s `per_head` and `off`; scatter_output() reads the same way).
  - seq_pad is the stride constant kSeqPad, which fa_guards requires to equal
    m_padded; for every geometry this ships lq == lk == seq_pad.
  - valid_len (`t`) real rows; rows at or beyond it are zero-filled by
    repack_qkv's memset, so the buffers are always full seq_pad rows.

The reference is computed in float64 *from the bf16 values that were written*,
not from the pre-rounding floats. That is deliberate: the kernel is given bf16
inputs, so comparing it against a higher-precision input would measure the input
rounding rather than the kernel. Any tolerance stated in compare_fa.py is a
kernel tolerance, not an input-rounding allowance.

Q is generated unscaled. The kernel folds 1/sqrt(head_dim) into its exp2
argument (attn_npu2.cc's `log2e / constexpr_sqrt_dk`), and the reference applies
the same factor to the scores. Scaling Q here as well double-scales by 8.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent

# Same env contract as attn_fa.py's SPECIALIZE, so the harness always describes
# the artifacts that were actually built rather than a shape in the abstract.
L = lambda k, d: int(os.environ.get(k, d))
LQ      = L("FA_LQ", 1536)
LK      = L("FA_LK", 1536)
LQP     = L("FA_LQP", 256)
LKP     = L("FA_LKP", 64)
DK      = L("FA_DK", 64)
DV      = L("FA_DV", 64)
HEADS   = L("FA_NUM_HEADS", 20)
UNROLL  = L("FA_HEADS_PER_UNROLL", 2)
STAGES  = L("FA_CASCADE_STAGES", 4)
BUILD   = os.environ.get("FA_BUILD", "build")


def rne_bf16(a: np.ndarray) -> np.ndarray:
    """Round float32 to bfloat16, round-to-nearest-even.

    Matches bf16_fill() in npue_encoder.hpp: (u + 0x7FFF + ((u >> 16) & 1)) >> 16.
    numpy/ml_dtypes also rounds to nearest even, so the two agree bit-for-bit.
    """
    return np.asarray(bfloat16(a.astype(np.float32)), dtype=np.float32)


def reference(q, k, v, heads, seq_pad, hd, t):
    """float64 fused non-causal attention, reading only rows [0, t).

    q/k/v are [heads, seq_pad, hd] float64. Returns [heads, seq_pad, hd] float64
    with rows >= t left at zero (the kernel is told valid_len, and the host
    contract says those rows of `out` are zeroed).

    The 1/sqrt(head_dim) is applied HERE, to the scores, and NOT to q before
    rounding. The kernel already does it internally: attn_npu2.cc defines
    `log2e (1.44269504089 / constexpr_sqrt_dk)` with constexpr_sqrt_dk = 8.0 at
    dk=64, folding the attention scale into the exp2 argument. Pre-scaling q
    here too divides the scores by 8 twice -- and since 1/8 is a power of two it
    survives bf16 rounding exactly, so the error is invisible in the inputs and
    only shows up as a far flatter softmax and a shrunken output. That mistake
    cost one diagnostic round: kernel rms came back 0.604x the reference with
    cos 0.667, and the head mapping was still identity, which pointed at a
    temperature mismatch rather than a layout error.
    """
    out = np.zeros((heads, seq_pad, hd), dtype=np.float64)
    for h in range(heads):
        qh = q[h, :t]                       # [t, hd]
        kh = k[h, :t]                       # [t, hd]
        vh = v[h, :t]                       # [t, hd]
        s = (qh @ kh.T) / np.sqrt(hd)       # [t, t], non-causal
        s -= s.max(axis=-1, keepdims=True)  # softmax stability
        p = np.exp(s)
        p /= p.sum(axis=-1, keepdims=True)
        out[h, :t] = p @ vh
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid-len", type=int, default=None,
                    help="real rows; defaults to FA_VALID_LEN or seq_pad")
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--build", default=BUILD)
    a = ap.parse_args()

    if LQ != LK:
        raise SystemExit(
            f"this harness assumes lq == lk (seq_pad is a single stride constant "
            f"in the host contract); got lq={LQ} lk={LK}")

    seq_pad = LQ
    t = a.valid_len if a.valid_len is not None else L("FA_VALID_LEN", seq_pad)
    if not (0 < t <= seq_pad):
        raise SystemExit(f"valid_len must be in (0, {seq_pad}], got {t}")

    rng = np.random.default_rng(a.seed)
    per_head = seq_pad * DK

    # fp32 randoms -> bf16 -> back to fp32. Everything downstream uses the
    # rounded values, so the reference sees exactly what the device will see.
    q = np.zeros((HEADS, seq_pad, DK), dtype=np.float64)
    k = np.zeros((HEADS, seq_pad, DK), dtype=np.float64)
    v = np.zeros((HEADS, seq_pad, DK), dtype=np.float64)
    # Unscaled: the kernel applies 1/sqrt(dk) to the scores itself.
    q[:, :t] = rne_bf16(rng.standard_normal((HEADS, t, DK)))
    k[:, :t] = rne_bf16(rng.standard_normal((HEADS, t, DK)))
    v[:, :t] = rne_bf16(rng.standard_normal((HEADS, t, DK)))

    # bf16 little-endian, head-first, exactly repack_qkv's addressing.
    for name, arr in (("q", q), ("k", k), ("v", v)):
        (HERE / f"{name}.bin").write_bytes(
            bfloat16(arr.astype(np.float32)).astype(bfloat16).tobytes())

    ref = reference(q, k, v, HEADS, seq_pad, DK, t)
    np.save(HERE / "ref_o_f32.npy", ref.astype(np.float32))

    nbytes = HEADS * per_head * 2
    cfg = [
        "device",
        f"xclbin G {HERE / a.build / 'final.xclbin'}",
        f"kernelx fa G {HERE / a.build / 'insts.bin'}",
        f"buf q {nbytes} {HERE / 'q.bin'}",
        f"buf k {nbytes} {HERE / 'k.bin'}",
        f"buf v {nbytes} {HERE / 'v.bin'}",
        f"buf o {nbytes}",
        "run fa q k v o",
        f"dump o {HERE / 'o_out.bin'} {nbytes}",
        "",
    ]
    (HERE / "fa.cfg").write_text("\n".join(cfg), newline="\n")

    meta = dict(heads=HEADS, seq_pad=seq_pad, valid_len=t, dk=DK, dv=DV,
                lq=LQ, lk=LK, lqp=LQP, lkp=LKP,
                heads_per_unroll=UNROLL, cascade_stages=STAGES,
                build=a.build, bytes_per_buf=nbytes, seed=a.seed)
    (HERE / "fa_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"heads={HEADS} seq_pad={seq_pad} valid_len={t} dk={DK} "
          f"buf={nbytes} B  build={a.build}")
    print(f"ref_o[0,0,:4]={ref[0,0,:4]}  rms={np.sqrt((ref**2).mean()):.6f}")
    print(f"padded rows zeroed: {seq_pad - t} of {seq_pad}")
    print(f"run:   (cd {HERE} && run_kernel fa.cfg && python compare_fa.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())