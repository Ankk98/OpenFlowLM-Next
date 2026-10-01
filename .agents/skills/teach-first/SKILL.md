---
name: teach-first
description: Explain technical work so the reader can check it - reasoning and assumptions stated before acting, every term defined at first use, every number carrying units and direction, one idea per message, and a worked example from real numbers. Use for any explanation, result summary, investigation write-up or handoff addressed to the human. Fires by default, not by request.
---

# Teach First

The value of an explanation is not that it is clear. It is that the reader can
**disagree with it**, early, before the work is done.

That is not abstract. In this repo the reader caught two real errors that way:
that a test was running on the CPU when it should have been on the NPU, and that
the machine had in fact frozen rather than merely gone quiet. Both were caught
because the work was described in terms that could be checked, not because a
number looked wrong in isolation.

So the rule is: **state the reasoning and the assumption before the action, and
make every claim falsifiable.** An explanation the reader cannot check is a
summary.

## Trigger

- Any explanation, result, investigation write-up, plan or handoff addressed to
  the human. Fires by default, not on request.
- Before an action whose correctness depends on an assumption -- especially a
  measurement, a rebuild, or anything that consumes time or disk.
- When reporting a null or surprising result. "It did not work" needs the
  reasoning that produced it.

## Preconditions + refusal conditions

- **The audience is derived, not assumed.** Read what the reader has actually
  shown they know -- from their questions, their corrections, and the code they
  have changed -- and write down three lines before explaining anything:
  1. what they already know (do not re-teach it),
  2. what they have not touched,
  3. the one idea this message exists to land.
  Deriving this takes a minute and is the difference between teaching and
  talking past someone.
- **One message, one idea.** If there are three findings, that is three
  messages or three clearly separated sections that can be read out of order.
- **Refuse to present a number you cannot source.** Say which run, which arm,
  which commit, or say "not measured".

## Requires + Never

Requires: every term defined where it first appears; every number with **units
and a direction** ("885 mW, higher is busier"); at least one worked example using
real numbers from this repo; short sentences; the assumption stated before the
action it licenses.

Never: unexplained jargon at first use; a number without units or without saying
whether higher or better is better; a ratio or a percentage whose denominator is
unstated; an analogy left standing after it stops being true; external image
links (ASCII renders everywhere and images rot); a table without a polarity note.

## Workflow

1. **Derive the audience** (above). Then anchor to the nearest thing they have
   already shown they understand, and **say where that analogy breaks** -- an
   analogy that quietly stops being true is worse than none, because it is
   trusted.
2. **Name the assumption before acting on it.** One line, and make it checkable:
   "the NPU is idle, so a counter that does not move is not measuring me". If the
   reader can falsify it in one step, they will; that is the point.
3. **Revise any stale belief in one line before using it.** Not a lecture on it:
   "RAPL is the usual power source; on AMD it is present and disabled, so this
   box reports zero from it".
4. **Number first, meaning second, why it matters third.** Concrete before
   abstract, and the direction always attached.
5. **Draw it in ASCII** for anything non-trivial -- a dataflow, a timeline, a
   memory map, a loop nest. Under 15 lines. For a stride bug, the picture *is* the
   explanation: "`[rows, 3*d]`, so a flat prefix of `rows*d` elements is the first
   third of the rows, all three thirds each" needs no prose to be checkable.
6. **Worked example, real numbers.** From this repo, not invented. The four
   engagement failures in `prove-engaged` are the worked set: each is a case where
   the output looked right and was not.
7. **Close with one next step**, not a reading list: one file and line, or one
   question to answer.

## What this repo's reader already knows

Derived from the questions actually asked here, and worth re-deriving rather than
assuming:

| known | not assumed |
|---|---|
| C++ and Python fluently; threads, SIMD, memory layout | AIE2P specifics, MLIR-AIE dialects, what a stream or a tile is |
| what a model, a checkpoint and a kernel are | this repo's vocabulary: arch numbers, `gemm_rtp`, families, tiers, the synced-copy arrangement |
| that a benchmark can lie | the host/NPU boundary: what XRT does, what the driver counts, what an fdinfo field means |
| how to read a diff and spot a bad loop bound | why a number is plausible: DRAM bandwidth, socket power, clock behaviour under load |

The second column is where the teaching goes. The first column is where being
explanatory becomes condescending.

## Verification

Before sending:

- Could the reader restate the one idea in a sentence? If not, the message
  failed -- not the reader.
- Every number has units and a direction. Every acronym is expanded on first use.
- Every claim is falsifiable by something the reader could actually run.
- The assumption is stated, and it is one they could check in a single step.
- No paragraph requires a term defined later in the message.

## Non-goals

- It does not change technical content or numbers. Pair it with the correctness
  skills, which own whether a number is right.
- It is **not** for reviewer-facing text. Humans write their own PR descriptions
  and review replies; this repo's policy forbids drafting them.
- Analogies never enter code comments. Comments stay terse.
- It is not a licence to pad. Concise and complete beats thorough and re-read.

## Concurrency (read-only skill)

Parallelizes without limit: no locks, no files, no state.

The one thing it cannot do concurrently is overwrite the reader's attention --
several agents each sending a full explanation is how the one idea per message
rule gets lost. If more than one thread has something to say, rank them and lead
with the one that changes what the reader does next.