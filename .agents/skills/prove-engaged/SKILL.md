---
name: prove-engaged
description: Prove the change actually ran before any A/B comparison or performance claim. Name the observable that must differ, in the predicted direction, and refuse the delta if the arms are indistinguishable. Use before reading any speedup, before benchmarking a build or a flag, and when a change may be silently inactive.
---

# Prove Engaged

A change that never ran and a change that ran perfectly produce the same
comfortable feeling. The difference is only visible if you look for it before the
numbers.

Four from this repo, each of which survived review and a full test suite:

| what | how it looked engaged | how it was actually engaged |
|---|---|---|
| `--decisiontemperature 3.0` | parsed, range-checked, set `temperature_overridden`, returned a well-formed answer | **dead**. The caller's value was written and then the container's `[1.0, 1.0, 1.0]` overwrote all three slots. Byte-identical output to the default. |
| `option_order` | parsed, validated as a permutation, documented on the struct | **dead**. One writer, zero readers, for the whole branch. |
| the phase timer `t_npu` | a named field, documented, printed in the timer line | **reads 0.0000, reproducibly**, while every other counter moved. The field existed; nothing filled it in this configuration. |
| `xrt-capture` | exit 0, 130 KB of artifacts, a replay manifest | **copied the input xclbin byte for byte** (same md5) and wrote all-null fields. |

The rule that kills all four: **no engagement proof, no delta.** It runs first,
then correctness, then performance.

## Trigger

- Before any A/B comparison of a change: a kernel, a flag, a fusion, a build, a
  routing fix, an env-gated path, a datapath or family choice.
- Before reading any speedup, latency or throughput number.
- When a change is behind a guard, a build flag, a config key, or a
  preprocessor condition.
- When a number moved but you cannot say why.

## Preconditions + refusal conditions

- **Both arms identified**: binary, hash, build flags, model file, design family.
  An unknown arm is not an arm.
- **The predicted observable, written before the run**: which log line, counter,
  symbol, dispatch count or behaviour must differ, and in which direction.
- **Refuse to compare** if the two arms produce identical observables. That is not
  "no measurable difference" -- it is "the change did not run", which is a
  different finding with a different next step.

## Requires + Never

Requires: shell `md5sum`, `nm`, `grep`, `strings`, `git diff`; read both arms'
provenance and both arms' logs; the pinned file-hash check from
`upstream-prior-art` when the change touches a synced file.

Never: compare arms whose observables are identical; accept "it should be
running" as evidence; bench a tree whose engagement proof failed; edit a pinned
synced file to test an idea; leave instrumentation in shipped code.

## Engagement inventory for this repo

Where a change can be silently inactive, and what proves it is live:

| surface | proof it engaged |
|---|---|
| a new kernel or family | the design's `insts_*.bin` and `design.json` exist; `aiebu-dump -p -m aie2ps <insts>` shows non-zero opcodes; `check_design_sets.py` passes |
| an env-gated path | the env var appears in the process environment of the child; the branch prints something new |
| a CLI flag | two runs differing **only** in that flag produce different observables |
| a wire field (`option_order`, `labels`, ...) | the value reaches the consumer: grep for a **reader**, not a writer |
| a counter or timer | it moves between two samples during known work, and its magnitude is plausible |
| a callback / functor on the host | `nm` shows the symbol in the binary that ran, and `gprof` attributes samples to it |
| `utilities/*_rig.cpp` | the rig's own output changes; a rig that prints a constant is a failed proof |

## Workflow

0. **Placement audit, first.** Brace-match the new code's enclosing `if`/`#if`/
   switch and confirm the taken branch executes on the target device. A placement
   miss reads *exactly* like a disabled flag, and no observable can save it --
   you will be hunting the wrong thing. In this repo the two live examples are a
   `OFLM_USE_HRX` guard that changes which sources are compiled at all, and a
   fused-epilogue path that changes which counters are written.
1. **Write the prediction down before running anything.** "Counter X rises by
   N", "the log prints path name P instead of Q", "dispatch count per call drops
   from 5.7 to 2", "the symbol S is present in the binary that runs". If the log
   prints op names rather than path names, predict a **quantitative** delta
   instead: per-kernel time, or dispatch count, moving in the expected direction
   under identical flags on both arms.
2. **Cheap probes before instrumentation.**
   - Env kill-switch A/B on **one** binary: on vs off must differ.
   - A temporary print behind an env var, removed before review. **No
     instrumentation ships.**
   - Prefer both over a rebuild.
3. **Collect from both arms** with identical flags except the change.
4. **Compare against the prediction, not just arm-vs-arm.** A difference in a
   channel you did not predict is a **confound**, not proof. Say so.
5. **Refuse the delta** if the observables are identical. Stop and find why:
   guard, wiring, stale binary, wrong path, wrong arm, file never written. Do not
   read the delta.
6. **If the counter cannot move, it is not a counter.** A documented field that
   always reads zero (`t_npu`) is absent, not fast. A tool that emits artifacts
   with no data (`xrt-capture`) is absent, not slow. Do not reason from either.
7. **Then** correctness, then performance. Engagement without identity is a
   faster wrong answer.

## Verification

- The prediction exists in writing **before** the run; the matching observation
  is quoted **after**.
- Both arm hashes recorded, and the differing observable named.
- A reader can tell from the note alone that the arms differed as intended.
- No instrumentation left in the tree: `git diff` shows none.

## Non-goals

- Output correctness. That is the next gate; engagement without identity is a
  faster wrong answer.
- Whether the change is *good*. `npu-profiling` owns where time goes.
- Whether the numbers are statistically sound across repeats -- that belongs with
  the bench discipline in `host-discipline` S7.5, and this skill defers to it.
- Blaming the code. Step 5's first hypothesis is "this never ran", not "this is
  wrong".

## Concurrency

- Read-only analysis over both arms' logs; parallelizes freely.
- Needs both arms' artifacts **present and hash-pinned**. Never re-run another
  agent's arm to "check" it -- that is a new session with different cache and
  page-cache state, so it is a different measurement.
- Mutating: if proving engagement requires a rebuild, that rebuild takes the same
  lock discipline as any other mutating work (see `host-discipline`).