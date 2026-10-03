"""Test vectors + float64 reference for the bf16 LayerNorm (designs/ln_bf16).

    python make_test.py [--seed S]

Interface under test -- the one the engine's layer_norm() actually drives:

    arg 0  x   bfloat16[N]   activation in
    arg 1  w   bfloat16[N]   LayerNorm weight
    arg 2  xn  bfloat16[N]   activation out

    xn = x * rsqrt(mean(x^2) + eps) * w

Two deliberate choices:

  * The reference is computed from the bf16 values that were WRITTEN, not from
    the pre-rounding floats. The kernel is handed bf16, so comparing against a
    higher-precision input would measure input rounding rather than the kernel.

  * The inputs are scaled to exercise the reduction. Unit-normal rows give
    mean(x^2) ~ 1 and a near-flat softmax-equivalent regime where a wrong
    reciprocal is easy to miss; rows are drawn with a spread of magnitudes so
    the rsqrt is tested across scales, and a few rows are scaled hard toward
    zero to hit the epsilon.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent

N = int(os.environ.get("LN_N", 2048))
EPS = float(os.environ.get("LN_EPS", "1e-5"))
BUILD = os.environ.get("LN_BUILD", "build")


def rne_bf16(a: np.ndarray) -> np.ndarray:
    """Round to bfloat16, nearest-even. Matches bf16_fill()'s
    (u + 0x7FFF + ((u >> 16) & 1)) >> 16, which numpy/ml_dtypes also does."""
    return np.asarray(bfloat16(a.astype(np.float32)), dtype=np.float32)


def reference(x, w, eps):
    """float64 LayerNorm over the LAST axis, from bf16-valued inputs."""
    xf = x.astype(np.float64)
    inv = 1.0 / np.sqrt((xf * xf).mean(axis=-1, keepdims=True) + eps)
    return xf * inv * w.astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--build", default=BUILD)
    ap.add_argument("--rows", type=int, default=768)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    rows = a.rows

    # A spread of row magnitudes so the reciprocal is exercised across scales,
    # plus two deliberately tiny rows to make eps matter.
    scale = np.exp(rng.uniform(-3.0, 3.0, size=(rows, 1))).astype(np.float32)
    x = rne_bf16(rng.standard_normal((rows, N)).astype(np.float32) * scale)
    w = rne_bf16(rng.standard_normal((rows, N)).astype(np.float32) * 0.5 + 1.0)
    x[0] *= 1e-3          # mean(x^2) ~ 1e-6, so eps=1e-5 dominates
    x[1] *= 1e-4          # mean(x^2) ~ 1e-8, eps dominates by 1000x

    nbytes = rows * N * 2
    for name, arr in (("x", x), ("w", w)):
        (HERE / f"{name}.bin").write_bytes(bfloat16(arr).astype(bfloat16).tobytes())

    ref = reference(x, w, EPS)
    np.save(HERE / "ref_xn_f32.npy", ref.astype(np.float32))

    cfg = [
        "device",
        f"xclbin G {HERE / a.build / 'final.xclbin'}",
        f"kernelx ln G {HERE / a.build / 'insts.bin'}",
        f"buf x {nbytes} {HERE / 'x.bin'}",
        f"buf w {nbytes} {HERE / 'w.bin'}",
        f"buf xn {nbytes}",
        # One dispatch per row. The design's stream element is ELEM = N*2 bytes,
        # i.e. exactly one bf16 row of N values, and the fifo is depth 2 with a
        # single acquire -- so one `run` processes ONE row. A cfg with a single
        # run computes row 0 and leaves the rest of the output buffer zeroed,
        # which the gate reports as maxrel=1.000 rather than as an error.
        *["run ln x w xn"] * rows,
        f"dump xn {HERE / 'xn_out.bin'} {nbytes}",
        "",
    ]
    (HERE / "ln.cfg").write_text("\n".join(cfg), newline="\n")
    (HERE / "ln_meta.json").write_text(json.dumps(
        dict(n=N, eps=EPS, rows=rows, bytes_total=nbytes, build=a.build, seed=a.seed),
        indent=2) + "\n")

    print(f"N={N} rows={rows} eps={EPS} build={a.build} buf={nbytes} B")
    print(f"ref row0 rms={np.sqrt((ref[0]**2).mean()):.6g}  "
          f"row1 rms={np.sqrt((ref[1]**2).mean()):.6g}  (eps-dominated rows)")
    print(f"run:   (cd {HERE} && run_kernel ln.cfg && python compare.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())