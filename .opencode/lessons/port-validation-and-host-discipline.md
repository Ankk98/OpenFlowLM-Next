---
name: port-validation-and-host-discipline
description: >-
  SOURCE DOCUMENT, not yet a skill. How to validate a numerical port against an
  independent reference, how to catch the silent-acceptance bug class, and how to
  run heavy builds on this workstation without filling its disk or freezing its
  desktop. Use when porting a model to the NPU engine, when a gate passes but the
  output is suspected wrong, when adding a regression test, or before any build
  that writes gigabytes. Promote to a skill once a second port needs it.
---

# Port validation and host discipline

Source material from the `convaiinnovations/laya` → SystemOne NPU port
(`add-laya-support-implement`). Everything here cost a real defect or a real
outage. The first section is the one that matters most: **two of the bugs below
passed every gate that existed at the time.**

Evidence for each item is in the commit named beside it, and in
`.local/laya-implement/` (scratch, gitignored, not durable — the commit messages
are the durable record).

---

## 1. Proving a port is correct

### 1.1 A cosine gate and an argmax gate cannot see a structural bug

`Head::forward` zeroed its attention accumulator with `t < d` (768) where it must
be `t < head_dim_` (64). Every head overwrote the other eleven. With one worker
the last head always won: **wrong but stable**. With twelve it was wrong *and*
unstable, and two identical `oflm decide` calls disagreed by 1.5e-02.

It passed, before the fix:

* a cosine gate against a float64 oracle (every value was a plausible float),
* an argmax gate on three easy probes (losing 11 of 12 heads still picked the
  same option),
* the whole existing test suite.

**Rule: a gate's resolution must be finer than the failure you are looking for.**
If the defect changes which answer comes out, gate on the answer. If it changes
values, gate on values. If it changes *stability*, gate on equality across
repeats and across worker counts — nothing else will see it.

### 1.2 "The reference agrees" is only evidence under two conditions

The same head bug passed because the oracle was compared on a metric too coarse
to see it. Worse, I then wrote in a commit message that the oracle was "wrong the
same way", which was **false**: the oracle reduces the head's attention with
`np.einsum("hij,jhd->ihd", …)`, has no head-slice indexing to get wrong, and its
logits were byte-identical before and after the fix.

Agreement between an implementation and a reference is evidence only when:

1. the reference is **independent** — a second implementation that could fail
   differently, not a re-run or a shared helper; and
2. the gate is **finer than the failure**.

**Rule: before citing a reference as agreement, check the reference's own code
for the defect's signature. If it cannot have that bug, say so — and if the
reference moves when the fix lands, the reference was the oracle, not the code.**

### 1.3 Localise by dumping state on BOTH sides of a stage boundary

End-to-end logits differing by 0.06–0.84 localises nothing: the gather, the head
layers, the scorer, or the encoder could each be responsible. Dumping the
**gathered marker rows** from the engine and from the oracle, and comparing
either side of the scorer, turned a guess into one measurement:

```
row 0  per-marker cosine 0.99900 0.99926   max|d| 54.2 at row RMS 199
logits row 0  engine 0.5497 0.3293   oracle 0.3317 0.2689
```

A 0.999-cosine input became a 0.55 logit difference, and the mechanism fell out
of the numbers: `LayerNorm(768)` divides by a standard deviation dominated by the
checkpoint's outlier channels, so it converts 0.1 % of cosine into a logit-scale
shift.

**Rule: when two implementations disagree end to end, find the earliest point
where their state differs. Dump both sides of that point.** Instrument with an
*overlay* (a patched copy of one header/source on the include path) when the dump
is diagnostic-only, so the product tree stays clean.

### 1.4 Remove a confound by feeding the reference's output in as input

Still ambiguous after 1.3: "the head is exact and amplifies the encoder's error"
versus "the head has its own error". Every comparison so far fed the head a
*different* encoder output, so both stories fit.

Feeding the head the reference's own rows settled it — the head is exact to 7
significant figures, and the whole gap is the datapath's already-accepted error:

```
row 0  engine head 0.3317095 0.2689044   oracle 0.3317 0.2689
row 2  engine head -0.1867242 … -0.1113582  oracle -0.1867 … -0.1114
```

**Rule: when a comparison confounds two components, construct the input that
removes the confound.** If the downstream stage can be called directly, call it
directly. This needed no NPU, no model load and no dispatch.

### 1.5 Measure the residual against the quantity it must resolve

The end-to-end result: the reference's entire top-2 decision margin over 60
clinical pairs is **0.043** (median 0.006), while the datapath's residual logit
error reaches **0.84**.

The error is ~20× the decision it has to get right, so argmax agreement is 36/60
and **no confidence gate above chance is supportable**. That is not a port bug;
it is the arithmetic of the hardware against this checkpoint, and it is the most
important thing the port learned.

**Rule: before choosing a gate, measure both the error and the thing being
resolved. If error ≫ quantity, the metric is meaningless — say so and gate on
something that still discriminates** (here: "every disagreement is a pair where
the reference had no opinion", plus "beat chance by a margin computed from the
fixture").

### 1.6 Check your own fixture before blaming the model

The first accuracy fixture measured a max reference gap of 0.043 and the first
instinct was "the model is flat". The fixture was at fault: every choice option
was the instruction text with a different level word in front of it —

```
"routine: what is the clinical urgency of the presentation"
"expedited: what is the clinical urgency of the presentation"
```

— so the markers differed by one token. Option **content**, not the question
repeated. The conclusion survived, but it was nearly published as a model
property.

**Rule: when a result is surprising, suspect the test data first, and say which
you suspect.** Vary option counts, question types and phrasings deliberately;
a fixture with one option count cannot exercise a per-option-count code path.

### 1.7 Every regression test gets a teeth check

A test that cannot fail is not a test. For the dead-override test (below) the
fix was reverted in a scratch build, rebuilt, and the test confirmed to fail with
its intended message before the fix was restored.

**Rule: for each new regression test, revert the fix, observe the failure, restore.
Budget the rebuild.** A test verified only against the fixed tree may be asserting
something incidental.

### 1.8 Correct claims in the record, do not quietly edit them

Two claims I made were wrong and are now corrected in follow-up commits, because
a reader meets a claim in a commit message:

* that the oracle shared the head bug (it did not, §1.2);
* that the model "picks whichever option is in slot 0" — from a single pair of
  runs that were not the same prompt; a controlled pair showed it answers
  **consistently** in both orders, which is the clinically correct answer.

**Rule: when a published claim turns out wrong, write the correction as its own
commit and say what the correct statement is.** The `experiments/` files in
`.local/` keep wrong turns deliberately; they are more useful than a clean file.

---

## 2. The silent-acceptance bug class

Three separate instances. All three passed every existing test. All three produce
output that looks entirely reasonable.

### 2.1 A field parsed and validated, then never applied

`PromptRow::option_order` existed, carried the comment "for the unpermute", and
nothing read it. A caller permuting options to probe position bias got the
unpermuted model and never found out.

**Rule: for every field a caller can set, there must be a test that fails when
the field is ignored.** Grep for the field name and count readers; one writer and
zero readers is a silent no-op.

### 2.2 A flag that parses, validates, sets an internal flag, and does nothing

`--decisiontemperature 3.0` returned **byte-identical** probabilities to the
default. The intended order was written down correctly eleven lines above the bug
("a caller-supplied one REPLACES them for all three types at once"), the
override was written, and then a later block read the container's
`temperature: [1.0, 1.0, 1.0]` and overwrote all three slots.

**Why it was invisible: 1.0 is the neutral scale, so a discarded override returns
exactly what a working neutral override returns.** A flag that does nothing is
indistinguishable from a flag that asks for nothing.

**Rule: test a knob with at least two non-default values, and assert the
DIRECTION of the effect, not merely that something changed.** Below 1 sharpens,
above 1 softens — asserting the spread ordering catches a knob wired to the wrong
variable, which a "did the number change" test would not.

### 2.3 A note packed into a shipped artifact that stopped being true

The container config said `"temperature_note": "packed, not read at run time"`.
It became false when the per-bucket lookup landed. The note is **inside the
artifact users read**.

**Rule: grep for prose in generated/shipped files. A comment that documents
behaviour is a test that fails to fail.**

### 2.4 Permutations: a scatter, not a gather

Upstream's convention: slot `s` shows option `order[s]`, and the model's
slot-ordered row is inverted by `canonical[order[s]] = p[s]`.

A gather instead of a scatter attaches every probability to the wrong option
while still summing to 1 and still carrying a plausible confidence. Nothing about
the output looks broken.

**Rule: for any index permutation, write the inverse by hand from the
specification and test it with a NON-symmetric permutation** — a 3-cycle, not a
swap. Identity and a single swap are the two cases a confused implementation is
most likely to pass.

### 2.5 Leniency where strictness is the point

`option_order: []` and `null` were "absent" in a first draft. Upstream's check is
`if "option_order" in qdef:` then a permutation test, so anything **present** must
be one. Reading them as absent silently widens the contract: a client whose
serialiser emits `[]` for unset fields would believe in a permutation that never
happened — which is this feature's whole failure mode.

**Rule: match the reference's strictness exactly, and say out loud which
readings you deliberately reject.**

### 2.6 A bool is an int

Upstream is Python, where `isinstance(True, int)` is true, so it needs an explicit
guard or `option_order: [true, false]` passes the sorted check and quietly
permutes by 1 and 0. nlohmann separates the types, so `is_number_integer()`
refuses it for free — recorded as load-bearing so a future "simplification" does
not remove it.

**Rule: when porting a validation, ask what the source language's type system
lets through that yours does not, and close it explicitly.**

---

## 3. Running heavy builds on this workstation

Read this before any build that writes gigabytes. Both incidents below are
mine.

### 3.1 The host

| | |
|---|---|
| RAM | 26 GB, **no disk swap**; 24 GB **zram** (compressed RAM) |
| `/tmp` | **14 GB tmpfs — that is RAM, not disk** |
| `/home` | 390 GB, was 71 % full; `/var/log/journal` alone is 1.2 GB |
| display | **Wayland: `gnome-shell` owns every output** |
| watchdog | hardware, `SP5100 TCO timer`, **10 min** — a wedged machine dies by itself |
| pressure | `systemd-oomd` **enabled**, `ManagedOOMSwap=auto` |

### 3.2 The disk incident

`~/.cache/` reached **126 GB**. A single full-suite `--basetemp` is **5.6 GB**,
because the open-engine tests compile C++ drivers linking **15 translation
units** of `src/open_npue` (16 sources counting the driver itself). The cause was not the filesystem: every run used a
*new* basetemp name (`oo`, `oo2` … `full2`) and nothing removed the previous one.

**Rule: one reused scratch directory, deleted after every run. Never a fresh name
per run.**

### 3.3 The freeze, caused by the "fix"

Having read that `/tmp` was small, I moved the scratch there because tmpfs is
"self-cleaning". It is RAM-backed: **a 5.6 GB basetemp on `/tmp` is 5.6 GB of the
26 GB of RAM.** I traded disk pressure for memory pressure, then added 24-wide
`g++` on 15 TUs at `-O2 -mavx512f`, detached `setsid` runs that could overlap,
and a 1.08 GB container mapped per process.

Symptom: display blank on **both** the internal panel and the external monitor
while the machine kept running, then a forced power-off.

What the journal showed afterwards — and the shape to expect next time:

* **no OOM kill, no hung task, no panic, no GPU reset**;
* boot ends with a **clean systemd-driven reboot** reaching `final.target` (a
  power-button press produces exactly this, so "forced" ≠ panic);
* a Wayland compositor stall blanks every output at once and leaves **no kernel
  log entry** — which is why the journal looks innocent;
* the freeze window itself has **no records**; the boot was 7085 lines.

**Rule: a blank screen on every output with the machine still running is a
compositor stall until proven otherwise, and the kernel log will not tell you.**
Check `journalctl -b -1 | grep -iE "oom|hung|panic|gpu reset"` to rule the
kernel out, then look at memory and CPU pressure rather than the display driver.

### 3.4 `setsid` is not "run it in the background safely"

`setsid cmd ;` returns **immediately**. Twice I then ran `rm -rf`, `tail` or
`grep` against output that did not exist yet — once deleting the basetemp out
from under a run that was still starting, producing a `FileNotFoundError` that
looked like a test bug.

**Rule: run long work in the foreground and let the tool block. If it must be
detached, write to a log and poll for a completion marker — never chain
operations after `setsid` that depend on its result.**

### 3.5 Do not use `nproc` for build concurrency here

24-wide `g++` competes with zram compression (which is CPU-bound and triggered by
the very memory pressure) and with GNOME, on the same 24 cores.

**Rule: cap build concurrency at ~8.** The tests do not get meaningfully faster.

### 3.6 The runner

`.local/laya-implement/scripts/pytest-safe.sh` (scratch) encodes all of the
above: disk scratch, one reused name, deleted on exit, memory reported before and
after. Reproduce it if the scratch is gone.

---

## 4. Porting another model to this engine

Reusable, from the arch-4 (ModernBERT) port.

* **Read the checkpoint's own config; refuse what you cannot honour.** The packer
  is not model-specific — it refuses, by name, rather than defaulting. Every
  refusal in the arch-4 path exists because the alternative is a plausible wrong
  model: two RoPE thetas, a banded head, a post-LN loop over a pre-LN checkpoint,
  an unbanded sliding layer. **Never relax a refusal to make a model load.**
* **The runtime builds ONE RoPE table.** A checkpoint with different global and
  sliding thetas (ModernBERT-large: 160000/10000) is out of reach until per-layer
  theta selection lands. This is the whole reason the port targets mmBERT.
* **Gate half order.** ModernBERT binds `x, gate = Wi(h).chunk(2, -1)`, so its
  gate is the **second** half — the reverse of this runtime's arch 2/3, which pack
  `[up, gate]`. The packer permutes. Getting it backwards does not crash and
  looks plausible: final-hidden-state cosine **0.199** when swapped, **0.9977**
  when right. The *intermediate* states stay plausible for several layers, so gate
  on the final state.
* **Name helpers for role, not position.** A first version used
  `gate_first_half`/`gate_second_half` and assigned the *gate* half as the
  accumulator's destination, computing `gate * act(up)`. The names made the bug
  look reasonable. They are now `up_half`/`gate_half`, so
  `up_half(v,i)[j] * act(gate_half(v,i)[j])` reads as the formula it is.
* **A duplicated computation is two models.** The gated activation exists in
  `swiglu_cpu` and in `fuse_ffn_epilogue`, and the fused path defaults ON. Verify
  `--nofuse` and the default produce **byte-identical** output; a half-order key
  honoured by only one of them ships a model that changes with a flag.
* **The band is correctness, and it costs.** 14 of 22 layers are wrong without it.
  Host attention is ~1 % of the encode at S=1024, so there is nothing there to win.
  Do not quote "129/1024 saves 87 % of attention" as a throughput argument.
* **Datapath choice is by measurement, and record the loser's number.** Two
  families are built per model shape on purpose. Put the ablation's losing arm
  next to the winner in `families.json` so the choice stays legible.
* **Operand rounding is usually not the error.** The oracle's bf16 operand replica
  moved logits by **0.0000**; the datapath's residual is MMAC accumulation and
  scheduling. So a bfp16-vs-bf16 ablation will not move argmax agreement — run it
  for the record, do not spend a rebuild expecting it to.
* **Layout hash cache is per tile shape.** A new hidden size reusing tiles **must
  not** build concurrently with shapes it shares markers with.

---

## 5. Merge and review hygiene

* **Verify a reviewer's claims against HEAD before acting.** A review of
  `add-laya-support-implement` reported the temperature-override bug as present
  and unfixed; it had been fixed and committed two commits earlier. Acting on the
  reviewer's tree instead of `git show HEAD:<file>` would have "fixed" fixed code
  and produced a confusing revert.
* **Severity labels are not evidence.** A finding described as "can print `ok`
  after a MISMATCH" was real but inverted and cosmetic (the tally and exit status
  were always right). Check what the gate *returns* separately from what it
  *prints*.
* **A doc note in a generated artifact is code.** See §2.3.
* **Prefer reverting a fix in a scratch build over reasoning about whether a test
  would catch it.** See §1.7.

---

## Promote-to-skill checklist

When a second port needs this material:

1. Move this file to `.opencode/skill/<name>/SKILL.md`, keep the frontmatter
   `name`/`description`, and write a description that triggers on the *task*
   ("validating a numerical port", "gate passes but output looks wrong", "build
   filled the disk"), not on the topic.
2. Split §3 into the existing `npu-offload-pipeline` skill's "Known blockers on
   this host" section, or into a `host-discipline` skill — it is independent of
   NPU work and applies to any heavy build here.
3. Keep §1 and §2 as one skill: they are the same lesson (a gate that cannot see
   the defect) seen from two directions.
4. Replace each "Rule:" with an imperative step an agent can execute, and keep the
   measured numbers — they are what make the rules credible.
5. Cross-link from `npu-offload-pipeline/SKILL.md`.
