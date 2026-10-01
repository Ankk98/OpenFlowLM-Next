---
name: project-scaffold
description: Start any multi-session task with a structured .local/<slug>/ folder so results, experiments, logs, scripts and status have an honest home, and keep a baseline row before the first candidate arm. Use when beginning work that will span sessions or produce numbers worth keeping, and when resuming one.
---

# Project Scaffold

Unstructured scratch loses the things that were expensive to learn. A week-old
binary that cannot be rebuilt, a pre-fix run sitting next to a post-fix run with
nothing marking which is which, an arm nobody can attribute because its hash was
never written down.

This skill creates the housing once so every later skill has somewhere honest to
write. It does not decide the hypothesis.

This repo already has `.local/laya-implement/` built organically over a long
session, so the layout below is **codified from what worked**, with the gaps
closed rather than a new structure imposed. Where it diverges from the
llama.cpp-skills original, the divergence is marked and justified.

## Trigger

- Starting any task expected to span sessions or produce kept numbers: a port, a
  bug hunt, a benchmark campaign, a survey, an investigation.
- Resuming one. Read `STATUS.md` first; it is the entry point.

## Preconditions + refusal conditions

- **Slug decided**: short, hyphenated, no dates, no versions. One slug per line
  of work. A rejected tree gets a **new slug**, or an explicit `supersedes:` line
  in the old folder's `STATUS.md` -- never a silent reuse, because "which attempt
  was this" is the first question a reader asks.
- **`.local/` stays untracked.** Verify with `git check-ignore -v .local/`.
- **Refuse to start** if `.local/` is staged, or if a previous run's artifacts
  would be mixed in without hashes.

## Requires + Never

Requires: shell `mkdir`, `md5sum`, `git rev-parse`; read the existing `.local/`
layout for precedent before creating anything.

Never: keep anything durable in `/tmp` -- it is RAM-backed here and disappears on
reboot (`host-discipline` S2); mix pre-fix and post-fix runs in one directory
unlabelled; stage `.local/` or `.agents/` content into a code commit; reuse a
slug for a new hypothesis; write a "temporary" note directly into `results/`.

## Layout

```
.local/<slug>/
  README.md        what / why / the commit list / the rule this project learned
  STATUS.md        CURRENT seat + next step. Pointer, not content.
  experiments/     NN-<slug>.md, one per investigation, numbered, newest last.
                   Wrong turns kept -- they are the useful part.
  results/         durable findings a later reader acts on. One file per
                   question, with the numbers and what they mean.
  handoff/         per-topic docs for someone picking this up cold: accuracy,
                   performance, bugs, setup, skills, upstream-PR.
  logs/            raw output and binaries. Every binary md5 recorded in the
                   experiment note that used it.
  patches/         one durable .patch per tree, re-appliable with `git apply`
  scripts/         helpers that were actually used. Not /tmp copies.
  assets/          probes and dumps too large to inline
```

Divergences from the original, and why:

* **`experiments/NN-*.md` instead of one `experiments.md`, and `results/`
  separate from `experiments/`.** One append-only ledger does not survive two
  agents, and it mixes "I tried X" with "X is true". `experiments/` is the ledger
  of attempts; `results/` is what a later reader acts on.
* **`HANDOFF.md` becomes `handoff/`.** One file cannot hold "accuracy", "bugs",
  "setup" and "skills" without becoming unreadable, and only some of them are
  written at a pause.
* **`LEARNINGS.md` points at `.agents/lessons/`, it does not duplicate it.**
  Lessons get promoted into tracked, reviewed skills; a local duplicate drifts.

## Workflow

1. **Create the layout with one-line stubs.** Stubs, not empty directories: git
   and most tools ignore empty dirs, and a stub with a sentence in it tells the
   next reader what belongs there.
2. **Write the baseline row BEFORE any candidate arm.** In this repo it is:
   ```
   HEAD            <sha>            git rev-parse --short HEAD
   upstream        <sha> <date>     per upstream-prior-art
   engine binary   <md5> <bytes>    md5sum build/src/oflm
   model container <md5> <bytes>    the .npue -- 1.08 GB here
   design family   BERT-h768-gated-i1152-bf16
   checkpoint      <revision sha>   from the fixture, not "latest"
   toolchain       mlir-aie / Peano / XRT versions
   host            power mode, governor, boost     only if numbers will be kept
   ```
   The container and checkpoint hashes matter as much as the binary: a model file
   swapped underneath a fixed binary looks exactly like a code regression.
3. **Hash every artifact you intend to keep** -- binaries, dumps, captured
   artifacts -- and write the hash **in the experiment note that used it**, the
   same day. `xrt-capture` produced 130 KB of plausible-looking output that
   turned out to be a byte-identical copy of its input; the md5 is what caught it.
4. **Use a fixed verdict vocabulary.** Per experiment: `PROMOTE` / `REJECTED` /
   `PARKED` / `CLOSED`. Per end-to-end case: `OK` / `BUG` / `TRUNC_BUG`. No
   synonyms, because a verdict nobody can grep for is not a verdict.
5. **Keep wrong turns.** The three highest-value documents here are a
   nondeterminism bisection, a disk/freeze incident, and an 8x Q-scaling bug that
   every gate passed. None of them would exist as documents if only the
   successful path had been written down.
6. **On pause, update `STATUS.md` and the relevant `handoff/` file**, with exact
   resume commands.

   **The test is readability, not a line count.** "One screenful" is a symptom of
   a good STATUS, not the target -- and a line budget is a false precision that
   gets met by deleting the next step. This repo's grew to 154 lines because
   review-triage evidence and per-phase detail had accumulated in it. Pushing
   those down to `handoff/review-triage.md` and `results/phases-0-9a.md` brought
   it to ~100 lines of tables and an ordered list, which a fresh agent reads in
   under a minute. The real failure looks like a STATUS that opens with history
   instead of a verdict.
7. **Keep `.local/` untracked and say so in the commit.** The durable record is
   the commit message and the tests that pin a finding -- not a scratch file.

## Verification

- Layout exists with stubs; the baseline row is present **before** any candidate
  arm was run.
- `git check-ignore -v .local/` shows it ignored, and `git ls-files .local` is
  empty.
- Every kept binary has an md5 in the note that used it; every tree has a patch
  or a commit hash, recorded the same day it was measured.
- `STATUS.md` is readable in under a minute by someone with no context. Test it
  by asking whether it opens with a **verdict** rather than with history, and
  whether the next step is findable without scrolling.
- A fresh agent can answer "what is true right now, and what is next" from
  `STATUS.md` alone.

## Non-goals

- Deciding the hypothesis. This skill only guarantees the folder exists.
- Whether a number is right. `prove-engaged`, then correctness, then
  `npu-profiling`.
- Replacing the tests. A finding that only exists in a scratch file has not been
  captured; promote it to a test or to a skill.

## Concurrency

- One slug per line of work. Two agents on one slug coordinate through the seat
  line in `STATUS.md`; otherwise split slugs.
- Namespaced filenames per agent or session (`04-profiling-survey-agent.md`).
  Never a shared `latest.log` -- "latest" is meaningless the moment two runs
  overlap, and that is exactly when you need to know which is which.
- Mutating skills that run on the NPU serialise on the single-device rule in
  `host-discipline` S3: one run at a time, and the others do read-only analysis
  until handoff.