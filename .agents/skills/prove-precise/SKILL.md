---
name: prove-precise
description: Precision is contract, never a tuning knob. The single-threaded host path is the spec; every fast or reduced-precision path must reproduce it bit-for-bit or within a bound stated before the run. Diverging inputs are already suspect. Use wherever precisions mix, wherever a batch or tier selects a path, and before widening any accuracy gate.
---

# Prove Precise

Precision is part of the answer's contract. Batch size, tier selection, thread
count and "which side of the host boundary" must never choose it.

Two failure families, and the second is the expensive one. The first is a
reduced-precision or vectorized path that loses enough to flip near-tied
decisions while aggregate similarity metrics stay green. The second is subtler
and it is a *process* failure: the gate goes red, someone widens the gate to
accommodate the number, and the defect ships wearing the new threshold.

This branch has an instance of the second, and it is the reason the widening
rule below is written as a refusal rather than a caution. A fixture's accuracy
gate was restated from "every disagreement sits above 0.05" to "every
disagreement sits below 0.05" after it failed. The stated reason was that the
stratum was empty. The actual reason, found later, was a stride bug in the
reference. The gate had been telling the truth and the threshold was moved.

## Trigger

- Any change where precisions mix: a host path against an accelerator path,
  float32 against bfloat16 or a reduced-precision emulation, a widened epsilon,
  a fused kernel against unfused reference math.
- Any change where a shape, batch size, tier, sequence length or thread count
  selects a different code path than the default case.
- Any change to a kernel that computes a reduction: matmul, attention scores,
  normalization, softmax, a scoring head.
- **Before widening, loosening or re-baselining any accuracy or tolerance
  gate.** This is the trigger people forget.
- Any accuracy number that moved and does not yet have a cause.

## Preconditions + refusal conditions

- **The spec named.** The single-threaded, highest-precision, simplest path is
  the spec -- not merely a baseline. A faster path that reproduces the spec is
  an optimization; one that diverges is a bug until proven otherwise. In a
  repository with a reference implementation, that is the reference, and it must
  share no code with the thing under test.
- **Precision pairs listed**: which path is at which precision, and where they
  meet. Most precision bugs live exactly at that seam, not inside either side.

Refuse to proceed, and say which precondition is missing, when:

- The proposed gate change is a relaxation. A tolerance may be *set* before a
  run, with a reason. It may not be *moved* after one, to explain a result. If
  the gate must change, first find out why the number moved, and re-derive the
  new bound from the consumer's decision rather than from the observed spread.
- The only proposed reference shares code with the arm under test. Then the
  comparison cannot see a shared defect and the result is not evidence.
- The number has moved and the cause is unknown. "Datapath error", "tolerance",
  "model margin" and "expected numerical drift" are placeholders for "not
  explained". An unexplained movement stops the line; it does not license a
  wider gate, because the same symptom is what a real defect looks like.
- The change is precision-changing *by design* and the proposed gate is a diff
  against the old path. See step 4 -- identity is the wrong gate there.

## Requires + Never

Requires:

- Read: the reference implementation, and `ground-truth-verify`'s source for
  every precision constant cited -- an epsilon, a dtype, a cast -- read at the
  code that defines it, not where it is used.
- Shell: a deterministic single-threaded run of each arm, outputs written to
  files and compared directly. Paths per project; record the exact commands.
- Shell: a pairwise comparison harness that reports the **first diverging
  element or decision** and the size of the gap there, not only a summary
  statistic. A summary over a whole tensor hides the one element that matters.
- For accuracy gates: a fixed input set, held constant, with the disagreements
  enumerated and each one's margin recorded.

Never:

- Ship a shape-dependent, batch-dependent or tier-dependent precision. If two
  shapes take different precision silently, that is a defect even when both are
  individually defensible.
- Accept a similarity metric, a perplexity, or a loss as a value check on its
  own. All three have resolutions and blind spots.
- Call two outputs identical because they look alike, because they agree on
  most examples, or because they are both plausible. The decision decides:
  token ids, argmax indices, selected routes, chosen options.
- Widen a gate, raise a tolerance, or drop the worst-matching cases without
  re-deriving the bound and re-proving it.
- Reuse a diagnostic score as a gate. A metric computed on data the model scores
  near-perfectly on carries no information, and a diagnostic quantity becomes a
  gate the moment a test asserts on it.
- Average away a disagreement between two runs of the same pair.
- Push, open or edit a PR or issue, write a review reply, or use sudo.

## Workflow

1. **Record the spec, and make it immutable.** Run the single-threaded
   highest-precision path once, on a fixed input set, and store the output plus
   a hash. This is the contract. Once recorded it does not change -- if a later
   run no longer reproduces it, the *environment* moved, and that is a finding
   to report, not a golden to refresh.

2. **Par-vs-spec, elementwise.** Every optimized path reproduces the spec
   exactly where it claims to, and within a stated bound where it cannot.
   Report the **first** diverging index and the gap there. Divergence at
   near-tied decisions is the expected symptom of precision loss and must not be
   dismissed as noise: a residual of 0.04 is nothing until it moves an argmax
   whose margin was 0.03, and nothing at all once it does.

3. **Run the matrix, both directions, and let one bad cell stop the line.** A
   path that agrees at one shape and diverges at another is worse than one that
   always diverges, because the passing case hides the defect. Include the
   awkward shapes: the smallest, the largest, the one with no convenient
   multiple, the one whose accumulation order differs. If a fix changes the
   result, show the counts **before and after** (`8/16 -> 16/16`), not only
   after.

4. **When the change is precision-altering by design, identity is the wrong
   gate.** Two arms that deliberately differ in precision cannot be gated by
   diffing each other -- there is nothing to diff against. Gate them by
   distribution instead: compare each against a higher-precision base, on
   in-distribution inputs, and require it to be no worse than an already-accepted
   path of the same precision class. Record which class each arm belongs to.
   Two arms in different classes are being compared on their *outputs*, never
   on their speed against each other without this.

5. **State precision constants as contracts, and check they were read from the
   defining code.** A normalization epsilon that differs between two layers of
   the same network is almost always correct, and almost always looks like a
   bug. The rule that prevents a well-meaning "cleanup" from changing results:
   for every epsilon, dtype and cast, record *why* it has the value it has,
   traceable to the reference implementation. Where a value is justified by
   the precision of the operands it acts on, write that justification down --
   "the accumulator is float32 here, so the difference is not hidden" is a
   contract; "it was inherited from the adjacent layer" is not.

6. **Only then read speedups**, under bands (`host-discipline` for the machine,
   `locked-clock-bench` for timing). A faster path that diverges is a wrong
   answer, not an optimization. Conversely, a slow path that is bit-identical
   and has a stated reason to exist is a correct result, not a failure to
   report a speedup for.

## Verification

- Golden hashes quoted, and the environment confirmed stable across the run.
- First diverging index and the gap at that index quoted, for every arm that
  is not bit-identical.
- Matrix counts shown before and after any fix.
- For distribution gates: the base, the inputs, and the accepted path each arm
  is compared against, all named.
- Every precision constant traced to the code that defines it, with a stated
  reason.
- Any widened gate appears in the result with its original value, its new
  value, the cause of the movement, and the re-derivation. A widened gate
  without all four is a regression wearing a threshold.

## Non-goals

- Proving the change ran (`prove-engaged`), or that its plan was grounded
  (`ground-truth-verify`).
- Gate resolution and identity as such. `prove-correctness` owns whether a gate
  can see the defect; this skill owns what bound the output must meet.
- Timing validity. `host-discipline` and `locked-clock-bench` own it.
- Choosing which suites to run.

## Concurrency

Read-only for the analysis; the runs themselves are mutating on the accelerator
and take the same exclusive lock the bench arms take. Log and golden files are
namespaced per agent and session.

Goldens are immutable once recorded. A second agent reproducing the matrix must
reproduce the golden; if it does not, report that the environment moved. Do not
re-record the golden to match, and do not average the two.
