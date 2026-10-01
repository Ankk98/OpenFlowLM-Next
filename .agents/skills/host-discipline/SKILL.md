---
name: host-discipline
description: Run heavy builds, model loads, test suites and benchmarks on a shared workstation without filling its disk, exhausting its RAM, or producing numbers that cannot be compared. Use before any job that writes gigabytes or runs a model, when choosing a scratch directory, setting -j, backgrounding a long job, or setting up or interpreting a benchmark -- also when diagnosing "the disk is full", "the machine froze", "the screen went blank but the machine is still running", or inconsistent benchmark results. Covers scratch placement, one-run-at-a-time, build concurrency, and reading CPU/NPU/memory-bandwidth/power/frequency/throttling.
---

# Host discipline

A workstation is not a build server. It runs your desktop -- and on a modern AMD
APU, often a compositor that owns the display -- on the same RAM and the same cores
you want for a build. Most "this machine is unusable" incidents are not the job
failing; they are the job being well-behaved on a machine whose limits were never
measured. And most "these numbers moved" incidents are two jobs sharing DRAM
bandwidth.

**Nothing here is a number to memorise.** Limits are per-machine and per-workload.
1 is how to find yours; the rules after it are conditional on what you find.

---

For profiling *inside* the stack -- host C++, the host/NPU boundary, AIE kernels,
AIE graphs -- see `../npu-profiling/SKILL.md`. It records what works here and, more
importantly, what is present but inert.

## 1. Measure first -- the preflight

```bash
# 1. What KIND of filesystem is your scratch candidate? tmpfs is RAM.
findmnt -no FSTYPE,SIZE,AVAIL,TARGET --target /tmp
df -hT /tmp "$(mktemp -d)" 2>/dev/null

# 2. RAM, and what is swap? zram/zswap means the swap is RAM.
free -h; swapon --show

# 3. Repo disk, and how full.
df -hT "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

# 4. Is a desktop sharing the box, and does something reclaim under pressure?
nproc; systemctl is-active systemd-oomd 2>/dev/null
loginctl list-sessions --no-pager 2>/dev/null | head

# 5. Heterogeneous cores? SMT siblings? This decides whether host threads are
#    even comparable between runs.
lscpu | grep -iE "^CPU\(s\)|Thread|Core|Socket|Model name|MHz"
grep . /sys/devices/system/cpu/cpu*/topology/core_cpus_list 2>/dev/null | head
```

| finding | consequence |
|---|---|
| scratch candidate is `tmpfs` | it is **RAM**. Never build, load or benchmark there -- see 2. |
| swap is `zram`/`zswap` | swap is compressed RAM: no OOM kill, but CPU contention instead. `-j` now competes with compression. |
| no swap at all | memory pressure reaches the user session immediately. |
| a desktop session is active | cores and RAM are shared. `-j $(nproc)` is wrong (4). |
| SMT and/or mixed P/E cores | a thread's speed depends on which core it lands on. Pin deliberately (7). |

---

## 2. `/tmp` and every other `tmpfs` are prohibited

Not "prefer a real filesystem" -- **prohibited for scratch, build output, model
files, and anything else large.** `tmpfs` is RAM with a disk-shaped interface:
pages cannot be swapped, so filling it is memory exhaustion wearing a path.

```bash
# PROHIBITED, on any machine where /tmp is tmpfs:
SCRATCH=/tmp/pytest-scratch          # /tmp/build  /tmp/model.npue  /tmp/oflm
```

Use a real filesystem, and **verify the class before relying on it** -- not the
size, the class:

```bash
df -T "$(dirname "$SCRATCH")" | awk 'NR==2{print $2}'     # must NOT be tmpfs
```

The guard in 8 exists because of this. Note it resolves the nearest **existing**
ancestor: `df -T` on a not-yet-created path fails silently, and a naive guard
reads the empty result as "not tmpfs" and accepts. That fail-open is the exact
shape of mistake the check exists to catch.

**Also prohibited:** large model files and containers under `tmpfs`, even
"temporarily". A 1 GB container there is 1 GB of RAM that cannot be reclaimed
under pressure.

**The one that bites hardest, because it is a default rather than a choice:
pytest's own `tmp_path` basetemp is `/tmp/pytest-of-$USER`.** Omit
`--basetemp` and every test that writes a build artifact puts it in RAM, with no
command of yours in the transcript. Measured on this host: **5.5 GB** accumulated
in `/tmp/pytest-of-ankk98` from runs that "did not ask for tmpfs".

So `--basetemp` is not optional here:

```bash
--basetemp=/home/ankk98/.cache/pytest-scratch/run   # or any real filesystem
```

and after a session, check `du -sh /tmp/pytest-of-$USER` -- it is the single most
likely place a runaway test suite goes, and `df /tmp` is the fastest way to notice.
The runner in 8 sets it; use the runner rather than the flag.

---

## 3. One model load and one run at a time

Two rules that are really one: **the device has one set of hardware contexts, and
the memory system has one set of bandwidth.**

* **AIE contexts are exclusive.** `xrt-smi examine -r aie-partitions` reports
  "No hardware contexts running on device" when idle. A second concurrent process
  either fails to get a context or shares one. A single-decision engine that
  serialises on a mutex makes this safe *within* a process and unsafe *between*
  processes.
* **Two loads is twice the residency.** A ~1 GB container plus staged device
  weights, twice, before any compute.
* **Two runs share DRAM bandwidth and socket power**, so neither number means
  anything. This is the single largest source of irreproducible benchmarks on
  this class of machine, and it is invisible in the results: you get a plausible
  number that is simply wrong.

Before a run, and especially before a benchmark:

```bash
pgrep -af 'oflm|drigr|layaacc|xrt-' || echo "no model process running"
timeout 20 xrt-smi examine -r aie-partitions | tail -3
```

`pgrep -f` matching your own shell is a known trap: use `pgrep -af -x` on the
binary name, or filter out `$$`.

**Serialise your own work too.** Do not start a second test suite while one is
running, and do not "just quickly" kick off a rebuild during a measurement. The
one-run rule covers you, not only your scripts.

---

## 4. Build concurrency: do not use `nproc`

```bash
JOBS=8          # a floor, not a ceiling
```

`nproc` is right on a build server and wrong here: the cores are also serving a
compositor, a desktop, and -- under `zram` -- the compression work **triggered by
the memory pressure you are about to create**. On a many-core APU, N-wide `g++`
is the fastest way to make the machine unresponsive.

---

## 5. Never detach a long job and then act on its output

```bash
setsid some-long-job > out.log 2>&1 ; tail out.log     # WRONG
```

`setsid`, `&` and `nohup` return **immediately**; anything chained after them
races the work it is supposed to follow. Observed twice in one session: a cleanup
`rm -rf` deleted a scratch directory out from under a job that had not created it
yet, and the resulting `FileNotFoundError` looked like a product bug.

Run long work in the **foreground**. If it must be detached, use a completion
marker and poll:

```bash
setsid sh -c 'long-job; echo done > "$0.flag"' out.log
while [ ! -e out.log.flag ]; do sleep 5; done
```

---

## 6. A blank screen with the machine still running is a compositor stall

The kernel log will be innocent, so rule it out with one command before looking
anywhere else.

```bash
journalctl -b -1 --no-pager | grep -iE "oom-kill|out of memory|hung task|panic|gpu reset"
```

* **No hits, and the boot ends in a clean `systemd` shutdown reaching
  `final.target`** -- the kernel was alive. A power-button press produces exactly
  that sequence, so "I had to force it off" does **not** mean a panic.
* **On a Wayland session the compositor owns every output.** A stall blanks the
  panel *and* every external monitor together, while the machine keeps running,
  and writes nothing to the kernel log.
* **A hardware watchdog is often armed** (commonly a 10-minute TCO timer), so a
  wedged machine eventually powers itself off. `journalctl -b -1 | grep -i watchdog`.
* A flood of compositor warnings in the **final second** of a boot is teardown,
  not cause. Check timestamps before reading them as a cause.

Then look at memory and CPU pressure, not the display driver.

---

## 7. Benchmarking: pin the conditions or the numbers are decoration

A result is only comparable to another result **measured under the same
conditions**. Record the conditions with the result; a number without them is not
reproducible, it is just a number.

### 7.1 What to check before every measurement

```bash
# Power mode. "performance", and not degraded.
powerprofilesctl get
powerprofilesctl list | grep -A1 '^\*'

# Frequency policy. Benchmark with boost ON, performance governor, performance EPP.
cat /sys/devices/system/cpu/cpufreq/boost
for f in /sys/devices/system/cpu/cpufreq/policy0/scaling_governor \
         /sys/devices/system/cpu/cpufreq/policy0/energy_performance_preference \
         /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq; do
  printf '%-56s %s\n' "$f" "$(cat "$f" 2>/dev/null)"
done

# One run at a time (3).
pgrep -af -x oflm || echo "clear"
```

Fix these for the session rather than per run. If you cannot change them (no root),
**record them** -- a comparison across two different power modes is not a result.

### 7.2 What to sample during a measurement

Availability varies, so probe rather than assume. On this class of machine
(AMD APU with an AIE NPU, `amd_pmf` loaded) the working set is:

| quantity | source | notes |
|---|---|---|
| **NPU power** | `sensors` -> `NPU_power`, or raw `hwmon*/power1_input` | works, in **mW**; see 7.2.1. `amd-smi`'s `SOCKET_POWER` is the whole socket, not the NPU rail |
| NPU temperature | `sensors` -> `NPU_temperature` | `amdxdna` hwmon; **`N/A` on this platform** (PMF does not report it) |
| **NPU occupancy** | `xrt-smi examine -r aie-partitions` | context-running state; there is **no AIE activity-% counter** (see 7.2.2) |
| **per-process NPU memory** | `/proc/<pid>/fdinfo/<fd>` on the `/dev/accel/accelN` fd | the only per-client NPU attribution available -- see 7.2.2 |
| **Memory bandwidth** | `amd-smi metric` -> `APU_AVERAGE_DRAM_READS` / `WRITES` | MB/s; the real answer to "is it bandwidth-bound" |
| **Package power** | `amd-smi metric` -> `SOCKET_POWER`; `sensors` -> `PPT` | two independent paths; if they disagree, distrust both |
| Per-core residency | `amd-smi metric` -> `APU_AVERAGE_CORE_C0_ACTIVITY` | better than `%CPU` for "was this core idle or busy-waiting" |
| Per-core clock | `amd-smi metric` -> `APU_CURRENT_CORECLK` | the throttling signal -- see 7.3 |
| CPU utilisation | `/proc/stat` deltas, or `top -b` | this box has **no `sysstat`**, so no `mpstat`/`pidstat` |
| Fans / temperature | `sensors` -> `cpu_fan`, `gpu_fan`, `edge` | fan RPM is a coarse throttle indicator |

**Tools that are commonly assumed and are NOT present here.** Check before writing
a recipe that needs them: `perf` (so no uncore/IMC hardware counters, and no
precise CPU profiling), `sysstat` (`mpstat`/`pidstat`/`iostat`), `turbostat`,
`numactl`, `cpupower`, and `ryzen_smu`.

**`intel-rapl` is present but disabled and returns nothing on AMD.** Check
`/sys/class/powercap/intel-rapl:0/enabled`; if it is `0`, use `amd-smi`/`sensors`.
A power script that reads RAPL and gets 0 is worse than one that does not exist.

### 7.2.1 NPU power: use hwmon, and READ THE UNIT IT PRINTS

Measured on this host during a 12-question dispatch, raw sysfs and the `sensors`
rendering side by side:

```
/sys/class/hwmon/hwmon12/power1_input   0 -> 421000 -> 885000 -> 25000   (uW)
sensors -> NPU_power                    0.00 mW -> 885.00 mW -> 25.00 mW
```

So the NPU power channel **works**, peaks around 885 mW for that workload, and
tracks the run: ramp, plateau, decay. The driver is right -- `amdxdna_sensors.c`
does `npu_power * MICROWATT_PER_MILLIWATT` into hwmon's uW convention, and the
field it reads is a `u16` of milliwatts.

**Two traps, both of which I hit:**

1. **The unit is not always W.** This channel prints **mW**. An earlier version of
   this file claimed a 1000x units bug because a shell line of mine read the
   number and then appended a hardcoded `W`:
   `printf "%s W", "$(sensors | grep NPU_power | grep -oE '[0-9.]+')"` printed
   "755.00 W" for what is really 755 **mW**. **The bug was in my measurement, not
   in the driver.** Never hardcode a unit in the reader -- print the one `sensors`
   gives you, and cross-check against the raw sysfs integer when a number looks
   impossible.
2. **`NPU_temperature` is `N/A` on this platform.** Not a failure: the driver
   gates on `AMDXDNA_INVALID_TEMPERATURE` and PMF does not report it here. So the
   channel exists and is empty -- which is the "documented field that always reads
   zero" case from 6, in the other direction.

Cross-check power against `amd-smi`'s `SOCKET_POWER` when a figure looks
implausible, but expect them to disagree in *scope*: this is the NPU rail only,
that is the whole socket.

### 7.2.2 `fdinfo` for the XDNA device: yes for memory, no for utilisation

`/dev/accel/accel0` is the AIE node. Any process holding it has an fd whose
`fdinfo` the driver populates:

```bash
for p in $(ls /proc | grep -E '^[0-9]+$'); do
  for f in /proc/$p/fd/*; do
    [ "$(readlink "$f" 2>/dev/null)" = /dev/accel/accel0 ] || continue
    echo "pid $p"; grep -E 'drm-' /proc/$p/fdinfo/"$(basename "$f")"
  done
done
```

Measured during a real dispatch (`amdxdna` 2.25.0):

```
drm-driver:                              amdxdna_accel_driver
drm-client-id:                           106
drm-pdev:                                0000:c5:00.1
drm-engine-amdxdna_accel_driver:         339706748 ns
drm-total-memory:                        507756 KiB
drm-shared-memory:                       0
```

**What it is good for -- per-client NPU memory.** `drm-total-memory` and
`drm-shared-memory` are real, per-process figures, and nothing else on this class
of machine gives per-process NPU attribution (`amd-smi` only reports device-total
VRAM). Two uses:

* as a **precondition check for the one-run rule** (3) -- a stale process still
  holding ~500 MB of NPU memory means you have more than one client, whatever
  `pgrep` says;
* to attribute memory when a run fails for want of it.

**What it is NOT good for -- utilisation.** The busy-time counter did **not advance
at all** across a 4 s window in which the NPU was demonstrably working: frozen at
`339706748 ns` while power moved. Note also that the engine name is the generic
accel class, `amdxdna_accel_driver`, not `aie` -- which is itself a hint that it is
not a per-engine accumulator. **Do not derive NPU busy% from it.** Use
`xrt-smi examine -r aie-partitions` for presence, and `amd-smi` for DRAM and
package power.

**General lesson, and it applies to every row in this table:** a counter existing
is not the same as a counter meaning something. Sample it twice during known work
and confirm it moves before you report it, and sanity-check its magnitude against
an independent source.

### 7.3 Throttling, when there are no throttle counters

Some platforms expose no `thermal_throttle` counters at all. Then infer it:

* **Sustained `APU_CURRENT_CORECLK` below `APU_CURRENT_CORE_MAXFREQ` under load.**
  This is the primary signal and it needs no new tooling. A core that never
  reaches max while busy is being held down by PL/thermal policy.
* **Package power well below the expected plateau** for a sustained load.
* **Fan RPM ramping** with flat clocks.
* `journalctl -k | grep -iE 'mce|thermal|throttl|hhvm'` for hard events.

### 7.4 Pin the cores, or do not compare them

With SMT and/or mixed P/E cores, a host thread's speed depends on where it landed,
so host-side timings vary run to run for reasons unrelated to your code. For
host-bound measurements (this runtime keeps GELU, softmax and LayerNorm on the
host):

```bash
taskset -c 2-7 <command>       # a fixed set of physical cores
```

Pick physical cores, not SMT siblings -- read `core_cpus_list` first. And keep the
thread count identical between arms, or you are measuring the thread pool.

### 7.5 The minimum a bench record must carry

Power mode and whether it was degraded - governor, EPP, boost, max freq - core
set and thread count, if pinned - NPU context present - sample interval and
duration - whether any other process was running - the loss/gap counts, not just
the mean.

**Interleave arms and repeat.** A single A-then-B pair cannot distinguish the
change from drift. Run A,B,A,B at least three times and report the spread; if the
arms overlap within noise, the honest answer is "no measurable difference".

---

## 8. A runner that enforces the mechanical rules

```bash
#!/usr/bin/env bash
# Run the open-engine suite without filling the disk, exhausting RAM, or racing.
#   * scratch on a REAL filesystem; tmpfs is RAM and is REFUSED
#   * ONE reused name, deleted on exit -- name proliferation is what fills disks
#   * foreground: never chain anything after this that depends on its output
#   * one job at a time: refuse if a model process is already running
#   * memory reported before and after, because invisible growth is the symptom
set -euo pipefail
R="$(cd "$(dirname "$0")/../.." && pwd)"
SCRATCH="${PYTEST_SCRATCH:-${XDG_CACHE_HOME:-$HOME/.cache}/pytest-scratch}"

# Resolve the filesystem of a path that may NOT EXIST YET, by walking up to the
# nearest existing ancestor. `df -T` on a nonexistent path fails and prints
# nothing, so a naive check reads that as "not tmpfs" and ACCEPTS -- a fail-open,
# and the wrong default for the check whose absence caused an outage.
scratch_fs() {
  local p="$1"
  while [ -n "$p" ] && [ ! -d "$p" ]; do
    p="$(dirname "$p")"; [ "$p" = "/" ] && break
  done
  df -T "$p" 2>/dev/null | awk 'NR==2{print $2}'
}
case "$(scratch_fs "$SCRATCH")" in
  tmpfs) echo "REFUSING: scratch resolves to tmpfs ($SCRATCH), which is RAM." >&2
         echo "Set PYTEST_SCRATCH to a real filesystem." >&2; exit 2 ;;
  "")    echo "REFUSING: cannot determine the filesystem for $SCRATCH." >&2
         exit 2 ;;
esac

if pgrep -x oflm >/dev/null 2>&1; then
  echo "REFUSING: an oflm process is already running (one run at a time, 3)." >&2
  exit 2
fi

mkdir -p "$SCRATCH"; rm -rf "$SCRATCH/run"
trap 'rm -rf "$SCRATCH/run"' EXIT

echo "== scratch: $SCRATCH ($(scratch_fs "$SCRATCH")) =="
free -h | sed -n 2p
cd "$R"; source ironvenv/bin/activate
set +e
python3 -m pytest specs/open-engine/tests/ -q -p no:randomly --basetemp="$SCRATCH/run" "$@"
rc=$?
set -e
free -h | sed -n 2p
echo "scratch removed on exit; rc=$rc"
exit $rc
```

**Test the guards before trusting them.** Point the scratch at a real path, at
`/tmp`, and at a nested path that does not exist yet; and start a second copy
while the first runs. All must behave.

---

## 9. Rules

- **Measure before planning** (1), on any new machine or workload.
- **Never `tmpfs`** for scratch, build output, or model files -- it is RAM (2).
- **One model load, one run, at a time** (3); verify with `pgrep` and
  `xrt-smi examine -r aie-partitions`.
- **One scratch name, reused, deleted after every run** (8). Never accumulate.
- **Learn your workload's real cost** with `du -sh`; inherit nobody's number.
- **`-j 8`, not `-j $(nproc)`**, on anything sharing the box with a desktop (4).
- **Foreground long jobs**; detach only with a completion marker and a poll (5).
- **Pin power mode, governor, EPP, boost and cores before a benchmark**, and
  record them with the result (7).
- **Sample DRAM bandwidth and package power during the run**, not just wall time.
- **Interleave arms and repeat**; report the spread, and say "no measurable
  difference" when they overlap (7.5).
- **Infer throttling from clocks** when no throttle counters exist (7.3).
- **A blank screen with the machine alive is a compositor stall** until one grep
  rules the kernel out (6).
- **Cap the build cache** (`ccache -M 2G`); it is free until it is not.

---

## Example: one host, probed 2026-10-01 -- what 1 and 7 came out as

Illustrative only. **Re-run the probes; do not reuse these numbers.** A different
firmware, a different `amd_pmf`, or a different SMI build changes what is
available -- that is why every recipe above probes first.

| probe | value here | consequence |
|---|---|---|
| `/tmp` | `tmpfs` | RAM; 2 applies |
| RAM / swap | 26 GB, no disk swap, 24 GB `zram` | no cushion; swap contends for CPU |
| cores | 24 logical, SMT siblings present | pin for host-bound work (7.4) |
| observed core clocks | 5050 / 4190 / 3175 / 3156 MHz at idle | heterogeneous in practice |
| power mode | `performance`, `Degraded: no` | already bench-correct; still verify |
| governor / EPP / boost | `performance` / `performance` / `1` | same |
| `intel-rapl` | present, `enabled=0`, reads nothing | **do not use**; use `amd-smi`/`sensors` |
| `perf`, `sysstat`, `turbostat` | absent | no `mpstat`/`pidstat`; no uncore IMC counters |
| NPU power | `sensors` `NPU_power` / `power1_input` | **works**, in **mW**, peaks ~885 mW on a 12-question run; see 7.2.1 |
| NPU memory, per process | `/proc/<pid>/fdinfo` on `/dev/accel/accel0` | real (`drm-total-memory`, `drm-client-id`); see 7.2.2 |
| NPU busy time | `drm-engine-amdxdna_accel_driver` **frozen while busy** | **do not use for utilisation**; see 7.2.2 |
| NPU activity % | **not exposed at all** | use `xrt-smi examine -r aie-partitions` for presence |
| DRAM bandwidth | `amd-smi metric` -> `APU_AVERAGE_DRAM_READS/WRITES` | MB/s, the bandwidth signal |
| package power | `amd-smi metric` -> `SOCKET_POWER`; `sensors` -> `PPT` | two paths, cross-check them |
| throttle counters | no `thermal_throttle` sysfs | infer from clocks and power (7.3) |
| one suite run | 5.6 GB scratch, 15 translation units per driver | x12 names = 126 GB |
| container | ~1 GB resident when loaded | two concurrent loads is 2 GB (3) |

Sequence that produced the two incidents: twelve runs, twelve scratch names,
nothing cleaned up -> 126 GB. Then the "fix" was to move scratch to `/tmp` because
tmpfs is self-cleaning -- trading disk pressure for memory pressure, adding
24-wide `g++` and detached jobs, which froze the desktop. **The lesson is not
"use disk, not tmpfs". It is "`/tmp` is RAM, one name cleaned up, one run at a
time, and measure the machine before you rely on it".**
