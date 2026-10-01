---
name: prove-correctness
description: Prove output identity and work conservation before believing any speedup, and prove your gate can see the bug you might have introduced. Greedy identical, counts conserved, precision bounds met, gate shown to have teeth: speed without identity is skipped work until proven otherwise. Use after engagement proof, before reading any delta, or when a change is engaged but its output may still be wrong.
---

# Prove Correctness

A timer rewards skipped work, and a green gate rewards having chosen the wrong
metric. Both are the same failure: **the measurement ran, and it measured
something other than what broke.**

This is the second gate in engaged, then correct, then fast. It runs after
`prove-engaged` and before any delta is read.

The worked example is from this repo, and it is the reason the gate-teeth step
below exists. A change was engaged -- the code ran, the pipeline moved, the
numbers changed in the predicted direction. It was also wrong. A cosine gate
against a float64 reference still measured **0.999**, because the defect
perturbed a stream only slightly; a decision-level gate measured **0.60
agreement over 60 pairs**, because the same defect moved the argmax. The failing
gate was dismissed as "datapath error versus model margin." The gate was right.
The explanation was wrong.

## Trigger

- Any change claiming less time, fewer dispatches, or more throughput for the
  same work: tiling, fusion, batching, quantized or reduced-precision paths,
  routing, caching, early exit, verify paths.
- Any gate that just went green on a diff that touched layout, strides, dtypes,
  reduction axes, or accumulation order.
- Any time an aggregate metric is being used to accept or reject a change that
  alters discrete decisions (argmax, top-k, a chosen route, a chosen option).

## Preconditions + refusal conditions

- **Engagement proven** (`prove-engaged`). Correctness of a path that never
  executed is meaningless, and an unengaged change has no output to compare.
- **A specimen of truth named** before the first comparison: a reference
  implementation, an N==1 serial path, a pinned-output corpus, or a stock build
  of the same source. "Whatever it printed" is not a specimen.

Refuse to proceed, and say which precondition is missing, when:

- The comparison baseline was produced by a build that is not otherwise
  identical to the arm under test. A difference in flags, precision, thread
  count, or input ordering is a variable, not a baseline.
- The only available reference shares code with the thing under test. Then a
  shared bug is invisible, and the honest statement is "not independently
  verified", not a pass.
- The change is a deliberate behavior change. Re-scope it as a spec change and
  gate it against the new contract (`prove-precise` step 4), never against the
  old output.

## Requires + Never

Requires:

- Read: the diff under test, and `prove-engaged`'s named observable.
- Read: the reference implementation, at the level where the two share no code.
- Shell: a deterministic run of each arm, its output written to a file, and
  `diff` between them. Example pattern, paths per project:
  `python3 <runner>.py --arm stock --out /tmp/stock.txt` then the same with
  `--arm patched`, then `diff /tmp/stock.txt /tmp/patched.txt`.
- Shell: a counter read on both arms for any quantity that should conserve.
- Shell: a deliberate break, for step 2 below.

Never:

- Read a speedup, or any delta, before identity holds. A bench number taken
  before steps 3-5 is decoration.
- Accept an aggregate similarity metric as a value check on its own. Cosine,
  correlation, PSNR, mean-squared error and perplexity all have resolutions, and
  a defect smaller than the resolution is invisible to them.
- Explain away a gate that failed. A red gate is information. An account of why
  it failed is a hypothesis until it has been tested, and the default reading of
  "the gate is measuring X and the bug is in Y" is that the gate is right.
- Treat "the gate would not have caught that" as a reason to keep a change that
  the gate green-lit. That sentence is the finding.
- Average away a disagreement. Two runs that differ have produced a finding; a
  third identical run does not explain the first two.
- Push, open or edit a PR or issue, write a review reply, or use sudo. Humans
  write every GitHub word.

## Workflow

1. **Name the specimen of truth, and write down what it is independent of.**
   State in one sentence which code, flags and inputs the reference does *not*
   share with the arm under test. If that sentence is empty, stop here.

2. **Prove the gate has teeth, before trusting it green.** Introduce, on the
   reference side only, a defect of the same *class* the diff could plausibly
   have introduced -- a wrong stride, a transposed axis, a dropped tail, a
   changed reduction order, an off-by-one on the last element. Confirm the gate
   goes red. Record the number it reported.
   - If the gate stays green, the gate is insufficient. That is a result, and
     the correct next step is to add a gate at the resolution where the defect
     is visible, not to accept the green.
   - This step is cheap and it is the only way to learn what a passing number
     means. A gate that has never been seen to fail has never been measured.

3. **Output identity.** Reference and arm, same input, same flags, same
   determinism settings, outputs to files. Compare them directly rather than by
   eye. Any difference stops the line: it is either a real behavior change
   (re-scope as a spec change) or a defect. Quantify it and report the
   *direction*: a difference is not "within tolerance" unless the tolerance was
   chosen before the run and justified by the consumer of the number.

4. **Work conservation.** Count what should be constant and check it is. Kernel
   launches, dispatches, tiles, blocks, covered layers, bytes moved, streams
   synchronized. A run that is faster *and* counted less work has skipped work,
   and no timing on it is meaningful until the invariant clears it. If a count
   legitimately drops, say which invariant permits it and by construction.

5. **Precision contract.** Where precisions mix, bound the difference against
   the consumer's decision, not against a tolerance that looks reasonable. The
   right question is rarely "is this close" and almost always "does this change
   any decision." A residual that leaves every argmax, top-k and threshold
   untouched is a diagnostic; one that moves any of them is a defect, however
   small the residual. When a path is defined to change precision, N==1 serial
   is the spec and the fast path must reproduce it.

6. **Only now read the delta**, and read it under bench conditions that cannot
   flatter it (`host-discipline` for the machine, `locked-clock-bench` for
   timing bands and interleaving). Report the delta with the identity evidence
   quoted alongside it, not in a separate section.

## Verification

- Identity evidence is **quoted, not asserted**: the diff output, or the
  matching hashes, in the report.
- The deliberate break from step 2 is quoted with the number the gate produced.
  If it is missing, the identity claim is untested and must be labelled so.
- Counts are shown conserved, or the reduction is attributed to a named
  invariant.
- Gaps are explicit. Write down what the gate cannot catch, and which cover
  would catch it, as part of the result rather than as a caveat.
- The consumer of the number is named, and the bound is stated against it.

## Non-goals

- Engagement proof. `prove-engaged` runs first and owns "did it run".
- Suite selection. This skill owns the verdict order and the refusal, not which
  suites to run; pick the suites from the change class and the project's own
  test layout.
- Timing validity. `host-discipline` and `locked-clock-bench` own whether a
  number is comparable; this skill owns whether it is about the right work.
- Precision policy. `prove-precise` owns what precision a path must hold.

## Concurrency

Read-only. This skill only runs commands and compares their output, so parallel
loads are safe and two agents may run independent arms at once.

One shared obligation, inherited from the others: gate runs that need exclusive
use of an accelerator take the same lock the bench arms take, and log files are
namespaced per agent and session. Identity comparisons are expected to be
deterministic -- if re-running the same pair twice disagrees, stop and report
it as a gate defect. Do not average it, and do not proceed to the delta.
