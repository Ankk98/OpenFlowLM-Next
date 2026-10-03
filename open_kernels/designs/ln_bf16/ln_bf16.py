"""LayerNorm as the open NPU encoder's LN path actually wants it: bf16 in,
bf16 out, three buffers, no fused residual add.

    LN_N=768 LN_EPS=1e-5 python3 open_kernels/build_design.py \
        open_kernels/designs/ln_bf16/ln_bf16.py <out_dir>

See ln_bf16.cc for why this is not designs/ln. The short version: the engine's
layer_norm() writes bf16 into host_ptr(0), dispatches, and reads bf16 out of
host_ptr(2), with the weight staged from the container and selected by
bind(1, slot). Three buffers, all bfloat16. designs/ln is fp32 in and out with
the add fused in, so it does not fit that path.

Buffers, in the order the kernel takes them:

    0  x   bfloat16[N]   activation in
    1  w   bfloat16[N]   LayerNorm weight (container slot; not filled here)
    2  xn  bfloat16[N]   activation out

STATUS 2026-10-02 -- the kernel is right, the BUFFERING MODEL IS NOT.

  Done and verified:
    * builds at LN_N=768 (insts.bin 508 B, 127 instruction words)
    * dispatches on the AIE: device state 4, 0.086 ms
    * two IRON traps fixed and written down below

  NOT done: this design is NOT yet usable by the engine, and the reason is the
  buffering model rather than the kernel.

      npue_encoder.hpp's layer_norm() fills the WHOLE activation buffer with
      par(), syncs, and issues ONE dispatch_only(). A design therefore has to
      consume the entire buffer per call. This one moves one row (ELEM = N*2
      bytes = one bf16 row of N values) per acquire, so:

        - a 768-row buffer needs 768 dispatches where the engine issues 1, and
        - the second dispatch HANGS (device state 8 after 4073 ms) because the
          depth-2 fifo is drained by the first call and nothing refills it. The
          init `sequence` fills once at program launch, not per dispatch.

  Observed exactly that: run 1 -> state 4 in 0.086 ms, run 2 -> state 8 in
  4073 ms, then RUN FAILED.

  What it needs: the (rows, N) blocking designs/ln uses. Tile the activation
  through 4 KB stream elements with a TensorAccessPattern over (rows, N), and
  make the kernel take a BLOCK of rows per acquire, computing the per-row
  reduction independently inside the block. designs/ln's Pipeline(3) plus its
  (1, N) TAP is the pattern; the arithmetic here is per-row already, so this is
  a loop and a TAP change, not new mathematics.

  Until that is done, keep host_ln=true. designs/ln (fp32, add fused) is
  validated and is the only LN design the engine can drive.

  This is also a structural finding, bigger than one kernel: no design in
  open_kernels/designs/ ships the buffer convention the engine's eltwise paths
  (LN, softmax, gelu) assume -- whole-buffer in arg0, weights chosen per-call
  from the container via bind(arg1, slot), whole-buffer out in arg2, one
  dispatch. Every existing design there is row-oriented with the weight passed
  as a fixed buffer. Which is why the non-unified 7-design branch in
  npue_encoder.hpp is dead: it names art/ + "/layernorm", "/softmax", "/gelu",
  but none of those directories exists as a valid pair for its runtime model.
  Wiring LayerNorm onto the NPU therefore means writing a new whole-buffer LN
  kernel/design, plus the equivalent for softmax and gelu, not " dropping
  in" designs/ln.

Two IRON traps hit while building this, both of which surface as a func.call
operand type mismatch rather than an assert:

  1. ObjectFifo depth is the number of BUFFERS carried per acquire, not the
     number of calls in flight. depth=1 with two inputs acquires one buffer.
  2. At depth 1 the output acquire() returns the memref ITSELF instead of a
     one-element list, so o[0] indexes into the memref and yields a ui8 where a
     memref<1536xui8> is expected. Both fifos are depth 2 here so acquire()
     returns the same shape on both sides.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402

N = int(os.environ.get("LN_N", 2048))
EPS = float(os.environ.get("LN_EPS", "1e-5"))
ELEM = N * 2  # bf16 -> two bytes per element, one 4 KB-element per buffer


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def ln_bf16(x: In, w: In, xn: Out, *,
            n: CompileTime[int] = 2048,
            eps: CompileTime[int] = 0,
            srchash: CompileTime[int] = 0):
    u8 = np.ndarray[(ELEM,), np.dtype[np.uint8]]
    b_ty = np.ndarray[(N,), np.dtype[bfloat16]]

    fn = ExternalFunction(
        "ln_bf16", source_file=str(HERE / "ln_bf16.cc"),
        arg_types=[u8, u8, u8],
        include_dirs=include_dirs(),
        compile_flags=[f"-DLN_N={N}", f"-DLN_EPS={EPS:g}f"],
    )

    # depth = how many BUFFERS the fifo carries per acquire, not how many calls
    # are in flight: designs/ln's input fifo is depth 5 because it holds
    # [x0 x1 a0 a1 w]. This kernel takes two inputs (x, w) and one output, so
    # 2 and 1. Getting this wrong shows up as a func.call operand type mismatch
    # (a scalar where a memref<1536xui8> is expected), not as an assert.
    # Both fifos are depth 2 even though only one output buffer is used. At
    # depth 1 the output acquire() hands back the memref ITSELF rather than a
    # one-element list, so o[0] indexes into the memref and the func.call gets
    # a ui8 where a memref<1536xui8> is expected. Matching the depths keeps
    # acquire()'s return shape the same on both sides; the spare buffer is
    # 1.5 KB against a 63 KB L1 budget.
    of_in = ObjectFifo(u8, name="in", depth=2)
    of_out = ObjectFifo(u8, name="out", depth=2)

    def core_body(ain, aout, f):
        e = ain.acquire(2)
        o = aout.acquire(2)
        f(e[0], e[1], o[0])
        aout.release(2)
        ain.release(2)

    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod(), fn],
                    tile=Tile(0, 2), stack_size=0x1800)

    # Row pattern is (1, N) like designs/ln: one LayerNorm per token, and the
    # reduction inside the kernel covers exactly those N values.
    tap = TensorAccessPattern((1, N), 0, [1, 1, 1, N], [0, 0, 0, 1])

    def sequence(a_x, a_w, c_xn, inp, outc):
        pipe = Pipeline(3)
        pipe.drain(outc, c_xn, tap)
        pipe.fill(inp, a_x, tap)
        pipe.fill(inp, a_w, tap)
        pipe.finish()

    rt = Runtime(sequence, [b_ty, b_ty, b_ty, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


DESIGN = ln_bf16
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) +
                 [(HERE.parent.parent / "include" / "vecmath.h").read_bytes()])
SPECIALIZE = {
    "n": N,
    "eps": int(round(-1e6 * __import__("math").log10(EPS))) if EPS > 0 else 0,
    "srchash": int(hashlib.sha256(_src).hexdigest()[:8], 16),
}