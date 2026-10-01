---
name: locked-clock-bench
description: Enforce honest benchmarking when the accelerator cannot be clock-locked: verify instruments are alive and the machine is quiet before every arm, discard warmup, interleave and bracket A/B arms with cooldowns, label every number with its operating point, and refuse cross-arm comparisons that are not like-for-like. Use for any throughput or latency delta claim, and for any timing number that will be quoted.
---

# Locked Clock Bench

A number is only as honest as the conditions it was taken under, and most
machines do not let you lock those conditions. So this skill is not "lock the
clocks" -- it is **establish what could have moved between the arms, and refuse
the comparison if it did.**

**Read the hardware premise first, do not assume it.** On this box:

- The NPU hwmon node exposes `power1_input` and **no frequency or clock
  attributes at all**. NPU clocks cannot be read, pinned, or locked by anyone,
  including with root. The host CPU governor *is* exposed and is currently
  `performance`.
- **`power1_input` is in MICROWATTS, not milliwatts.** The driver's own source
  settles it -- `drivers/accel/amdxdna/amdxdna_sensors.c` computes
  `*val = npu_metrics.npu_power * MICROWATT_PER_MILLIWATT`, where `npu_power`
  is a `u16` in milliwatts. A reading of `1041000` is **1.04 W**, not 1041 W.
  Check the unit in the driver before quoting a power number anywhere; a factor
  of 1000 in the wrong direction reads as a hardware fault.
- **The underlying field saturates.** `npu_power` is a `u16` in mW, so it tops
  out at 65535 mW = **65.5 W**. A reading of exactly `1048575` is a saturated
  field, not a measurement, and must not be reported as a power.
- `amdxdna` reads **0 at idle**. So a zero reading is the correct idle
  value and a *non*-zero reading is what proves work happened. A zero reading
  *during* a timed run is a dead instrument, not a fast kernel. Note that a
  plausible-looking 1 W under load is itself worth a second look: it may mean
  the accelerator was barely used and the work was on the host. Confirm the
  work is where you think it is before quoting the number.
- The NPU (`amdxdna`), the iGPU (`amdgpu`) and the battery each have their own
  hwmon node with a `power1_input`. Indices move when hardware changes;
  **identify a node by its `name` field, never by its number.**

A protocol that assumes clock control exists will be followed by a human who
cannot comply, and then the protocol quietly stops meaning anything. The
substitutes that *do* work here: a governor check, instrument liveness, and
bracketing.

## Trigger

- Any throughput, latency, per-token or per-layer timing claim.
- Any A/B comparison where one arm is a code change rather than a config flag.
- Any timing number that will be quoted, written down, or shipped.
- Any comparison whose arms were produced in different sessions, or on
  different builds, or by different agents.
- Before reporting "no measurable difference" -- that is a claim, and it is
  easier to get wrong than a speedup.

## Preconditions + refusal conditions

**Run the pre-flight before EVERY arm, not once per session.** A machine that
was quiet at the start of a session is not evidence it was quiet for arm four.

```sh
# host CPU frequency policy (this one IS settable, and usually already is)
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor     # expect: performance

# identify the instrument by name, not by hwmon index
for h in /sys/class/hwmon/hwmon*; do printf '%s %s\n' "$h" "$(cat $h/name 2>/dev/null)"; done

# NPU power, before and after each arm. Idle reads 0; loaded does not.
cat /sys/class/hwmon/hwmon*/power1_input

# machine quietness -- the parts that make a number incomparable
free -g ; df -h ; uptime
```

Refuse to proceed, and hand back to the human, when:

- The CPU governor is not `performance` and re-locking it would need root.
  **Never re-lock it yourself.** Reads are unprivileged; the write is not, and
  an agent that escalates to change a machine's power policy has crossed the
  line between measuring and modifying the experiment.
- Another heavy job is running, or the machine is thermally throttled, or
  memory is short enough to have begun swapping.
- The timing instrument reads a constant -- most often zero -- across a run
  that demonstrably did work. **A counter that cannot move is not a counter.**
  Report the instrument as broken and go find a live one; do not report the
  zero as a result. This has happened here: one timing source returns exactly
  `0.0000`, reproducibly, while the run it measures is visibly long.
- The two arms were not built from a known source state. See the binary-freshness
  rule in Verification.

## Requires + Never

Requires:

- Read: `host-discipline` for scratch placement, build concurrency and the
  one-run-at-a-time rule. This skill does not set `-j`, choose a scratch
  directory, or manage disk; it sequences *measurements* and assumes the machine
  has already been made fit to measure on.
- Shell: the pre-flight block above, before every arm, with its output logged.
- Shell: `md5sum` on each arm's binary and on any shared library it loads.
- Shell: an exclusive lock on the accelerator for the duration of a bench
  session, so two agents cannot interleave their loads invisibly.
- Shell: the workload itself, run in the foreground.

Never:

- Escalate to change clocks, governor, power policy or any other machine
  setting. Read them, report them, hand the fix to the human.
- Run a bench and a build concurrently, or a bench and a test suite. On a
  memory-tight machine both are correct and their sum is an invalid number.
- Background a long run to escape a tool timeout. It converts a visible failure
  into an invisible one and loses the exit status.
- Compare arms from different sessions for a close call, and never compare
  medians across different machine states.
- Report a negative result from a single screening pass. See below.
- Push, open or edit a PR or issue, write a review reply, or use sudo.

## Workflow

1. **Pre-flight before every arm.** Governor, hwmon nodes by name, NPU power,
   memory, disk, load average. Log the machine state **per arm** in the results
   file, not once in a preamble. If a value cannot be captured, the arm is
   labelled unverified rather than assumed.

2. **Warmup, discarded.** Run the workload once and throw the result away. The
   first run after a load or a relink carries cold page and device caches, and
   comparing it against a warm arm is how a cold artifact becomes a finding.
   **Run one is never compared.**

3. **Screen, for direction only.** A single pass with matched settings, purely
   to learn which way the number moved. Directional signal, never a verdict --
   and a screen that comes out flat is "unverified", not "no change".

4. **Verdict: interleaved, in one session, repeated.** Alternate arms
   A/B/A/B rather than running all of A then all of B, so that any drift over
   the session is shared instead of landing entirely on one arm. Take at least
   three repetitions per arm. Put cooldowns between arms, and **scale the
   cooldown with the work**: a longer prompt or a larger model needs a longer
   settle, and a cooldown that is too short turns heat into a measured
   regression. A monotonic decline across successive arms is thermal or thermal-
   adjacent sag -- refuse the session and report sag, do not average it away.

5. **Bracket close calls.** When the claim is single-digit percent, the middle
   arm must be bracketed, because ordinary session drift can span several
   percent and a two-point comparison cannot distinguish the effect from the
   drift. Warm the load path first with a discarded run so the first *counted*
   arm is not also the cold one.

6. **Builds go in the cooldown slots, sequentially.** Keep the previous binary
   until the comparison is complete, and record its hash. Never swap the binary
   under a session that is still measuring.

7. **State the operating point with every number.** A throughput figure without
   its shape is not a measurement, it is a vibe. Always alongside the rate:
   sequence length, batch or tier, layer count, head count, and whether the
   number is host time, accelerator time, or end-to-end. This is not pedantry:
   a token's cost on this stack grows steeply with context position, so a rate
   quoted without context length is not comparable to a rate quoted with it.

8. **Assert the direction in the table header.** "Higher is better" or "lower is
   better" belongs in the header of every timing table, and milliseconds and
   rates go in separate columns. A reader who has to infer which is which will
   occasionally infer backwards.

9. **Negative results get the same rigor as positive ones.** "No measurable
   difference" is a claim and needs the interleaving, repetition and bracketing
   that a speedup needs. A single screening pass that comes out flat is labelled
   *screen, unverified*, and it never closes a line of work -- a null result is
   a reason to look harder, and the honest report says which it is.

## Verification

All of these hold, or the claim is refused:

- **Every arm is labelled** with machine state and operating point, not with a
  single session-level preamble.
- **Instruments were proven alive during the work.** A power reading that rose
  from zero, or a duration that is non-zero, is the evidence that the run
  happened and that the measurement observed it. This is the same requirement
  as `prove-engaged`, applied to the bench rather than to the code: an
  instrument reading a plausible value under load. **State the unit when you
  state the reading**, because the plausible-but-wrong-by-1000 number is
  indistinguishable from a fault until you divide it.
- **Binary freshness.** Hash each arm's binary and any library, and confirm the
  artifact is newer than the last source edit. A stale binary is the most
  expensive trap in benchmarking because it produces *identical magnitudes
  across a real code change*, which reads exactly like "the change is neutral".
  Identical numbers after a code edit means the old binary ran.
- **Interleaved, same session, same lock, same machine state.**
- **Run one discarded. No cross-state medians. No close call from two points.**
- **Direction asserted; units in separate columns; operating point stated.**
- **Accuracy claims** are not timing claims and do not inherit this rig, but
  they do inherit the labeling: the fixture identity, the arm, and the binary
  hash go in the same table, so a number cannot be quoted without knowing what
  produced it.

## Non-goals

- Deciding what a number means -- which path ran, whether it is engaged, whether
  the outputs are correct. `prove-engaged`, `prove-correctness` and
  `prove-precise` own those, and they run first.
- Whether the plan was grounded (`ground-truth-verify`).
- Machine, disk and memory policy. `host-discipline` owns that; this skill owns
  the ordering of measurements and assumes the machine is fit.

## Concurrency

**Mutating skill.** An accelerator session is exclusive: take the shared lock
before the first arm and hold it until the session ends, whether the verdict is
positive, negative or abandoned. Arms compared across two agents are
cross-session arms and are invalid unless they were taken under the same lock,
in the same machine state, interleaved.

One build directory and one pinned binary per agent; never swap a shared
library while another agent holds the lock. Namespace log files per agent and
session, and label every claim with the agent and session that produced it
alongside the machine state -- a number with no owner is a number nobody can
re-run.
