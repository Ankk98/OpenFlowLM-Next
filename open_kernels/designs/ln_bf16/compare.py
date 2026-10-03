"""Gate for the bf16 LayerNorm kernel against the float64 reference.

    python compare.py

Reads xn_out.bin (dumped by run_kernel) and ref_xn_f32.npy (make_test.py).

Tolerance reasoning:

  The output is bf16, whose 8-bit mantissa gives ~2^-9 = 2e-3 relative
  quantisation on its own. The reduction runs in an fp32 accumulator but the
  INPUTS are bf16, so x^2 carries bf16-level relative error which the rsqrt
  halves and the final multiply passes through.

  So: a max-relative gate around 1e-2 leaves room for one bf16 output ulp plus
  the input rounding, and is still far below anything a wrong reciprocal, a
  wrong denominator (kN vs a row count) or a missed epsilon would produce --
  those are factor-of-N or order-of-magnitude errors, not 1e-2 ones.

  The mean relative error is reported too, because it is the number that
  distinguishes "bf16 noise" from "systematically slightly wrong": noise has
  mean << max, a wrong scale has mean ~ max.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent

MAX_REL = 1e-2
MAX_MEAN_REL = 2e-3


def main() -> int:
    meta = json.loads((HERE / "ln_meta.json").read_text())
    n, rows, nbytes = meta["n"], meta["rows"], meta["bytes_total"]

    raw = (HERE / "xn_out.bin").read_bytes()
    if len(raw) != nbytes:
        print(f"FAIL size: xn_out.bin is {len(raw)} B, expected {nbytes} B")
        return 1

    got = np.frombuffer(raw, dtype=bfloat16).astype(np.float64).reshape(rows, n)
    ref = np.load(HERE / "ref_xn_f32.npy").astype(np.float64)

    # The two eps-dominated rows are checked separately: they are the only ones
    # where the +eps in the denominator matters, and a kernel that dropped it
    # would pass on every other row.
    scale = np.abs(ref).max() + 1e-30
    diff = np.abs(got - ref)
    rel = float(diff.max() / scale)
    mean_rel = float(diff.mean() / (np.abs(ref).mean() + 1e-30))

    eps_rows = float(diff[:2].max() / (np.abs(ref[:2]).max() + 1e-30))
    ok_eps = eps_rows < MAX_REL

    ok_rel, ok_mean = rel < MAX_REL, mean_rel < MAX_MEAN_REL
    print(f"{'PASS' if ok_rel else 'FAIL'} rel   maxrel={rel:.3e}   (gate < {MAX_REL:g})")
    print(f"{'PASS' if ok_mean else 'FAIL'} mean  meanrel={mean_rel:.3e}  "
          f"(gate < {MAX_MEAN_REL:g})")
    print(f"{'PASS' if ok_eps else 'FAIL'} eps   rows0-1 maxrel={eps_rows:.3e}  "
          f"(gate < {MAX_REL:g}; mean(x^2) ~ 1e-6 and 1e-8, so eps dominates)")
    print(f"     N={n} rows={rows} eps={meta['eps']} build={meta['build']}")
    return 0 if (ok_rel and ok_mean and ok_eps) else 1


if __name__ == "__main__":
    sys.exit(main())