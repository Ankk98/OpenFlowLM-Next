"""Gate for the fused-attention kernel against the float64 reference.

    python compare_fa.py

Reads the kernel's bf16 output written by `run_kernel ... dump o o_out.bin`, and
ref_o_f32.npy from make_test.py.

Tolerances, and why they are what they are:

  The kernel consumes bf16 and produces bf16. It runs Q@K^T in native bf16 MMAC
  and accumulates in a bf16 accumulator, so the scores carry bf16-level error
  *before* the softmax exponentiates them. Attention is a softmax, so a small
  score error becomes a multiplicative probability error -- this is not a case
  where a bit-exact or near-exact comparison is available, and pretending
  otherwise would just hide real defects.

  Two gates, chosen to catch different failures:
    - max relative error on O, over the real rows only.
    - cosine similarity, which is scale-free and so catches a kernel that is
      merely too small or too large, which max-rel alone can miss.

  The output rows at or beyond valid_len are NOT gated, and that is deliberate.
  The kernel writes real values there -- measured max 1.3e-1 at valid_len=700 --
  but the host contract never asks it to: FaAttention::dispatch_and_scatter
  calls scatter_output() for rows [0, t) and then zero_pad_rows(out, t,
  m_padded, d) for the tail. The kernel's own O buffer tail is dead data that
  nothing reads. An earlier version of this file gated on it and reported FAIL
  for a kernel that was in fact correct; a gate that rejects correct work is
  worse than no gate. The tail is reported below as an observation only.

Exit 0 only if the three numeric gates pass.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent

# Tolerances are set from a measured error budget, not fitted to one run.
#
# bf16 output quantisation alone -- rounding the float64 reference to bf16 and
# nothing else -- gives, at Laya's shape:
#     max 2.315e-03   p99 6.027e-04   mean 1.424e-04
# The kernel measures against the float64 reference:
#     max 2.914e-02   p99 9.714e-03   mean 2.821e-03
# The excess is the documented bf16 score accumulation plus the bf16
# online-softmax rescale factor (attn_npu2.cc: "all in float except the bf16
# rescale factor r"). So the gates below sit ~2x above the measured maximum
# rather than just above it: a gate fitted to the observed number catches
# nothing and only makes the next run look like a failure.
#
# max_rel is noise-sensitive -- one outlier element sets it -- so the p99 gate
# carries most of the signal and the max is there to catch gross corruption.
MAX_REL = 6e-2
P99_REL = 2e-2
MIN_COS = 0.999


def main() -> int:
    meta = json.loads((HERE / "fa_meta.json").read_text())
    heads, seq_pad, t = meta["heads"], meta["seq_pad"], meta["valid_len"]
    hd, nbytes = meta["dk"], meta["bytes_per_buf"]

    raw = (HERE / "o_out.bin").read_bytes()
    if len(raw) != nbytes:
        print(f"FAIL size: o_out.bin is {len(raw)} B, expected {nbytes} B")
        return 1

    got = np.frombuffer(raw, dtype=bfloat16).astype(np.float64).reshape(heads, seq_pad, hd)
    ref = np.load(HERE / "ref_o_f32.npy").astype(np.float64)

    g = got[:heads, :t, :]
    r = ref[:heads, :t, :]
    scale = np.abs(r).max() + 1e-30
    diff = np.abs(g - r) / scale
    rel = float(diff.max())
    p99 = float(np.percentile(diff, 99))
    cos = float((g.ravel() @ r.ravel()) /
                (np.linalg.norm(g) * np.linalg.norm(r) + 1e-30))

    ok_rel = rel < MAX_REL
    ok_p99 = p99 < P99_REL
    ok_cos = cos > MIN_COS

    # Observation only -- see the module docstring. The host zeroes this.
    tail = got[:, t:, :] if t < seq_pad else np.zeros((1, 1, 1))
    tail_max = float(np.abs(tail).max()) if tail.size else 0.0

    print(f"{'PASS' if ok_rel else 'FAIL'} rel   maxrel={rel:.3e}  (gate < {MAX_REL:g})")
    print(f"{'PASS' if ok_p99 else 'FAIL'} p99   p99rel={p99:.3e}   (gate < {P99_REL:g})")
    print(f"{'PASS' if ok_cos else 'FAIL'} cos   cos={cos:.8f}      (gate > {MIN_COS:g})")
    print(f"  note  tail  max|o[t:]|={tail_max:.3e}  ({seq_pad - t} padded rows, "
          f"not gated -- zero_pad_rows() is the host's job)")
    print(f"     shape heads={heads} seq_pad={seq_pad} valid_len={t} dk={hd} "
          f"unroll={meta['heads_per_unroll']} stages={meta['cascade_stages']}")

    return 0 if (ok_rel and ok_p99 and ok_cos) else 1


if __name__ == "__main__":
    sys.exit(main())