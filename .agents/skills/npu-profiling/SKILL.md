---
name: npu-profiling
description: Profile this repo's NPU stack at every level that has a working tool - host C++, host/NPU boundary, AIE kernel, AIE graph, and whole-system. Use when asking where time goes, why an encode is slow, which host phase dominates, whether a kernel is efficient, what a dispatch actually does, or when a profiling attempt produced artifacts but no data. Records what was measured working and what is present-but-inert, so a dead end is not re-explored.
---

# NPU profiling, by level

**The thesis, and it is the whole skill:** on this stack you can profile the host
thoroughly, inspect the NPU statically, and read exactly **one** NPU-side runtime
signal -- the power rail. There is **no AIE activity percentage, no per-kernel
timer, and no working trace**.

That is narrower than a first pass suggests, and getting it narrower is the point.
A source-level survey of `~/repos/xdna-driver` (which matches the installed module
exactly -- see `references/dead-ends-and-provenance.md` 5.4) found that the driver *holds* a per-column busy counter,
`u16 npu_busy[AMDXDNA_NPU_MAX_PMF_COLUMNS]`, and exposes it as
`AMDXDNA_SENSOR_TYPE_COLUMN_UTILIZATION`. So the data exists. What is missing is
anything that reads it: hwmon wires up only power and temperature, `amd-smi` does
not report the AIE at all, and the ioctl path is not wired to a tool here. 4.3
has the ABI so the next person can finish it.

That is a property of the build, not of the tooling in general -- and it was
established by measurement, not by absence of evidence. The evidence is in 5.

Read `host-discipline` for the whole-system level (power, DRAM bandwidth, clocks,
one-run-at-a-time). This skill covers everything below that.

---

## 1. The map

| level | tool | status | gives you |
|---|---|---|---|
| whole system | `amd-smi metric`, `sensors` | works | DRAM MB/s, package W, per-core clocks/residency |
| application | the `utilities/*_rig.cpp` rigs | works | phase timers, and the ids/markers the oracle needs |
| **host C++** | **gprof** (`-pg` build) | **works -- richest** | function-level host ranking with call counts |
| host C++ | `gdb` sampling | available | poor-man's profile, no rebuild |
| host C++ | `strace -c` | works | syscall/ioctl counts per dispatch |
| **host->NPU boundary** | in-repo phase timers | **partly broken** | `t_npu` reads **0.0000**; host phases are fine |
| host->NPU boundary | `/proc/<pid>/fdinfo` | works | per-client NPU **memory** only |
| **AIE kernel** | **`aiebu-dump -p/-d -m aie2ps`** | **works** | opcode histogram, full disassembly, group binding |
| **AIE graph** | **`xclbinutil --dump-section :JSON:`** | **works** | `ip_layout`, `connectivity`, `group_topology` |
| AIE graph | `mlir_aie` Python API | **broken here** | refs 5.2 |
| **NPU device counters** | fdinfo `drm-engine-*` + `QUERY_SENSORS` columns | **works** | derive per-dispatch ms; columns via `npu-sensors`; power is uW (5.0) |
| NPU trace | `xrt-capture` / `xrt-replay` | **inert** | copies the xclbin, records nothing (refs 5.1) |

---

## 2. Host C++ -- gprof is the best tool here

The in-repo timers give phases; **gprof gives functions**, and it contradicts
them. Build the rig with `-pg`:

```bash
S="${XDG_CACHE_HOME:-$HOME/.cache}/scratch-prof"; mkdir -p "$S"   # never tmpfs
R=/home/ankk98/repos/OpenFlowLM-Next; cd "$R"
U="npue_encoder npue npu_device npue_pack json_min xlmr_tokenizer_gen \
gemma_tokenizer_gen bbpe_tokenizer_gen tokenizer_bbpe tokenizer_xlmr \
tokenizer_gemma tokenizer gemma_kernels gemma_encode"
SRC=""; for u in $U; do SRC="$SRC src/open_npue/$u.cpp"; done
g++ -std=c++17 -O2 -pg -g -mavx512f -mavx2 -mfma -o "$S/rigpg" \
    utilities/laya_preln_rig.cpp $SRC -Isrc/open_npue -I/opt/xilinx/xrt/include \
    -L/opt/xilinx/xrt/lib64 -lxrt_coreutil -Wl,-rpath,/opt/xilinx/xrt/lib64
(cd "$S" && ./rigpg <container> $R/src/xclbins/<family> ids.txt >/dev/null)
gprof -b -p "$S/rigpg" "$S/gmon.out" | head -20
```

Measured on the Laya bf16 family, 1024 tokens, 88 dispatches:

```
 %  self s   calls  name
39.41  0.80    161  Encoder::softmax_cpu(...)::{lambda(int,int)#1}
22.66  0.46    160  Encoder::av_impl<8>(...)::{lambda(int,int)#1}
12.32  0.25    158  Encoder::qk_impl<8>(...)::{lambda(int,int)#1}
 7.88  0.16    169  Encoder::swiglu_cpu(...)::{lambda(int,int)#1}
 6.40  0.13     88  Encoder::gemm(...)            1.48 self / 2.40 total ms/call
 2.96  0.06    346  Encoder::layer_norm_cpu(...)
```

**Softmax is the top host cost, not GELU** -- the in-repo timer ranks
`hostgelu` (0.36 s) above `hostsm` (0.145 s), gprof ranks softmax at 0.80 s. Both
can be right: the timers are **wall-clock per section**, gprof is **CPU-time
sampled process-wide**, and the encoder is threaded. **Never compare the two
directly**, and say which one a number came from.

`gemm`'s 88 calls x 2.40 ms ~ **211 ms of NPU-path cost** -- which is also the
answer to "why does `t_npu` say zero" (3).

Notes: gprof needs the `-pg` build, so it is a *separate binary*; `-O2` is fine and
the ranking is what matters, not the absolute numbers. Because the hot code is
`std::function` lambdas in headers, symbol names are long -- `c++filt` helps.
`gprof` with no `-pg` binary produces an empty profile, silently.

---

## 3. The in-repo phase timers -- and the `t_npu` trap

`npue_encoder.hpp` accumulates a genuinely good decomposition. **The rigs are the
only place it is reported**; `src/` accumulates and never prints.

```bash
$S/rig <container> <designs> ids.txt 2>&1 >/dev/null
```

```
timers  npu 0.0000  attn 0.1180 (qk 0.0480  av 0.0700)  hostln 0.0090
        hostsm 0.1453  hostgelu 0.3602  dispatches 88
```

Fields: `t_npu` (memcpy+sync+dispatch), `t_attn` split `t_qk`/`t_av`, host
`t_hostln`/`t_hostsm`/`t_hostgelu`, and further splits `t_conv` (fp32<->bf16),
`t_in` (`sync_to_device`), `t_disp` (`kernel()`+wait), `t_out`
(`sync_from_device`), `t_bias`, plus `n_dispatch` (88 = 22 layers x 4 ops).

### `t_npu` reads 0.0000, reproducibly

Two independent runs: `npu 0.0000` both times, while everything else moved
within noise (`attn` 0.1180/0.1200, `hostgelu` 0.3602/0.3527). So in the
fused-epilogue configuration **the host<->device time is not being measured at
all**, and the single most important number for an NPU-offload profile is the one
that is absent.

Consequences, and they are the practical reason to care:

* You **cannot** split the encode into memcpy / conversion / sync / dispatch with
  these counters. The fields exist and are zero.
* Wall minus the host timers is where the device time hides, but it is a
  *remainder*: it contains the DMA, the sync, the dispatch and anything
  unaccounted, and nothing in the tree separates them.
* **gprof is the workaround**: `gemm`'s per-call total (2.40 ms x 88) puts a
  number on the NPU path that `t_npu` refuses to report.

Do not "fix" a profile by trusting `t_npu`, and do not conclude the NPU is free.

---

## 4. Static views -- the NPU kernel and the AIE graph

These are the only NPU-level views that work, and they are **static**: a profile
of the compiled code, not of a run.

### 4.1 AIE kernel: `aiebu-dump`

The per-tier, per-op instruction binaries under a family's `gemm_rtp/` are the
kernel-level artifacts (`insts_qkv_b4.bin`, `insts_attn_out_b16.bin`, ...).

```bash
A=/opt/xilinx/xrt/bin            # NOT on PATH
I=src/xclbins/BERT-h768-gated-i1152-bf16/gemm_rtp/insts_qkv_b4.bin

$A/aiebu-dump -p -m aie2ps $I    # opcode frequency
$A/aiebu-dump -d -m aie2ps $I    # full disassembly
```

```
v0.1, gen4
6x8 M1
35856B, 992ops
XAIE_IO_WRITE                352      XAIE_IO_MASKPOLL        0
XAIE_IO_BLOCKWRITE           256      XAIE_IO_PREEMPT         0
XAIE_IO_MASKWRITE             64      XAIE_IO_NOOP            0
XAIE_IO_CUSTOM_OP_TCT         64      XAIE_IO_CUSTOM_OP_RECORD_TIMER  0
XAIE_IO_CUSTOM_OP_DDR_PATCH  256      XAIE_IO_CUSTOM_OP_MERGE_SYNC   0
```

Read it as: the kernel is a **6x8** AIE2P design in 992 ops; it streams in with
`BLOCKWRITE` and issues `TCT` (token control) 64 times; and it contains
**no `NOOP`, no `PREEMPT`, no polling** -- a tight kernel with no idle spin.

**`XAIE_IO_CUSTOM_OP_RECORD_TIMER` is 0.** That is the important line: the compiled
kernels carry **no hardware timer instrumentation**, so per-kernel cycle counts
would require rebuilding with timer ops. There is no post-hoc way to get them.

`-m aie2ps` is required (the device is AIE2P / "NPU Strix Halo"). **With no flags
`aiebu-dump` exits 0 and prints nothing** -- silent, not an error.

### 4.2 AIE graph: `xclbinutil` on the shipped xclbin

This works on the **built artifact**, so it answers "what did we actually ship".

```bash
X=src/xclbins/BERT-h768-gated-i1152-bf16/gemm_rtp/final.xclbin
/opt/xilinx/xrt/bin/unwrapped/xclbinutil --dump-section ":JSON:$S/meta.json" --input $X
# -> ip_layout, connectivity, group_connectivity, group_topology
```

`group_topology` is the graph-level counterpart of the `.attach_to_group 0..7`
lines in the disassembly -- the same eight groups seen from both sides. The wrapper
at `/opt/xilinx/xrt/bin/xclbinutil` does **not** support `--dump-partinfo`; use
the `unwrapped/` binary and the empty-section `:JSON:` form.

---

### 4.3 NPU column utilisation: the ABI, and a half-finished probe

`utilities/npu-sensors` is the working probe (fixed in `0d176f0`; expect **9**
records -- 1 power + 8 columns). Both of its original defects are documented in
5.0: a zero-size query that can never return records, and a temperature enum the
installed header does not define. Read 5.0 before modifying it.

What is known, from the driver source that matches the running module:

```c
/* drivers/accel/amdxdna/amdxdna_sensors.h */
struct amdxdna_sensors {
        u16 npuclk_freq;
        u16 npu_busy[AMDXDNA_NPU_MAX_PMF_COLUMNS];   /* <-- per-column busy */
        u16 npu_power;                                 /* milliwatts */
        u16 mpnpuclkfreq, npu_temp;
};
```

`amdxdna_query_sensors()` (same file, `amdxdna_sensors.c:49`) fills a
caller-supplied buffer with `struct amdxdna_drm_query_sensor` records:

| record | `type` | `units` | `unitm` |
|---|---|---|---|
| Total Power | `AMDXDNA_SENSOR_TYPE_POWER` | `mW` | -3 |
| Temperature | `AMDXDNA_SENSOR_TYPE_TEMPERATURE` | `C` | 0 |
| Column %d Utilization (one per column) | `AMDXDNA_SENSOR_TYPE_COLUMN_UTILIZATION` | `%` | 0 |

Reached with the standard DRM info ioctl:

```c
struct amdxdna_drm_get_info { __u32 param; __u32 buffer_size; __u64 buffer; };
info.param = DRM_AMDXDNA_QUERY_SENSORS;      /* include/uapi/drm/amdxdna_accel.h */
/* pass buffer_size = 0 first: the kernel writes the size it needs */
ioctl(fd, DRM_IOCTL_AMDXDNA_GET_INFO, &info);
```

Header for the ABI: `~/repos/xdna-driver/include/uapi/drm/amdxdna_accel.h`.

**Two open questions, in order:**

1. **Why 0 records?** `amdxdna_get_sensors()` demonstrably *works* -- hwmon reads
   power from it -- so a 0-byte reply is about the ioctl path, not the data. Likely
   candidates: the `param` value, a required hwctx, or `total_col` being 0 for the
   node the probe opened. `aie2_pci.c:855` passes `ndev->total_col`, so a probe on
   the wrong node would report no columns.
2. **`amdxdna_drm_query_sensor.label` is not NUL-terminated** (the driver
   `scnprintf`s into a fixed 64-byte field). `%s` on it runs into the rest of the
   record. The probe copies it into a bounded buffer; keep that.

If this is finished, it replaces the biggest gap in this skill: a real NPU
utilisation number, per column, while a run is in flight.

**Until then, the honest NPU-side runtime picture is: power, and nothing else.**
Presence is still available via `xrt-smi examine -r aie-partitions`.

## 5. What is present but inert -- do not re-explore these

### 5.0 Per-dispatch NPU time IS obtainable — derive it, do not look for a counter

Re-verified 2026-10-02 against a real dispatch. An earlier revision of this skill
claimed every device-side busy-time instrument reads zero. **That was wrong**, in
two different ways, and both errors are instructive:

- `drm-engine-amdxdna_accel_driver` in `/proc/<pid>/fdinfo` works and reports
  ~0.8% of wall-clock across runs. It read 0 previously because the sampler used
  `pgrep -f`, which matched the **bash wrapper** rather than the process actually
  holding `/dev/accel`. Always resolve the pid via `/proc/*/fd` -> `accel`, never
  by name pattern.
- The 8 column-utilization sensors **do** return values. `amdxdna_hwmon_read`
  handles only `hwmon_temp` and `hwmon_power`, so they are absent from **hwmon
  sysfs** — but `DRM_AMDXDNA_QUERY_SENSORS` is a *separate* path and does return
  them. Conflating the two paths is what made them look dead.

There is still **no absolute AIE execution or cycle counter**; that part was right.

**How to get per-dispatch NPU time.** Take two fdinfo samples around a dispatch
and divide:

```
npu_ms = (engine_active_ns_after - engine_active_ns_before) / command_submissions_delta
```

where `command_submissions` comes from
`DRM_AMDXDNA_QUERY_HW_CONTEXT` with `HW_CONTEXT_ALL`. Cross-checked at 40.56 ms
against a 40.6 ms host wait. Note `HW_CONTEXT_ALL` is **root-only**; unprivileged,
read the submissions count from the same fdinfo block instead.

Column utilization (`DRM_AMDXDNA_QUERY_SENSORS`, 8 records, reaches 100% under
load with an ~8 s ramp) is the second usable instrument. Two traps:

- **There is no size-query pass.** `amdxdna_query_sensors()` guards each record
  with `if (args->buffer_size < sizeof(sensor)) goto out;` and then writes
  `args->buffer_size = sensors_count * sizeof(sensor)` at `out:`. A zero-size
  probe therefore skips every record, counts zero, and returns **rc 0** — it looks
  like success with no data. Pass a full-size buffer in one call.
- The installed uapi header defines only `AMDXDNA_SENSOR_TYPE_POWER` and
  `..._COLUMN_UTILIZATION`. There is no `_TEMPERATURE`; the driver skips that
  record without `HAVE_7_2_AMD_PMF_NPU_METRICS_NPU_TEMP`, matching
  `hwmon temp1_input` = `EOPNOTSUPP`. Referencing it makes the probe fail to
  compile, which is how `utilities/npu-sensors` stayed broken for so long.

`utilities/npu-sensors.cpp` is now correct on both counts (fixed in `0d176f0`);
verify it returns **9** records (1 power + 8 columns) before trusting it.

What else is usable:

- `/sys/class/hwmon/hwmon12/power1_input` — `amdxdna`, in **MICROWATTS**
  (`amdxdna_sensors.c:144` multiplies a `u16` mW field by
  `MICROWATT_PER_MILLIWATT`). 0 at rest. The `u16` saturates at 65,535 mW =
  65.5 W, so a reading of exactly 1048575 uW is a saturated field, not a
  measurement.
- `/proc/<pid>/fdinfo` on `/dev/accel/accel0` — `drm-driver`, `drm-client-id`,
  `drm-pdev`, and heap/alloc sizes. Proves a context exists and how much device
  memory is resident.
- **NPU clocks are observable but not settable unprivileged.** Read frequency via
  `DRM_AMDXDNA_QUERY_CLOCK_METADATA` (MP-NPU ~1267 MHz). `SET_STATE` is root-only,
  so treat every benchmark as unlocked and label its operating point.
- `xclbinutil --input <f> --info` and `--dump-section AIE_PARTITION:json:<out>` —
  `column_width`, `start_columns`, `operations_per_cycle`, PDI/DPU CDO inventory.
  Use `--info`; **`--dump` alone is ambiguous** with `--dump-section` and errors.
- `xrt-smi` is at `/opt/xilinx/xrt/bin/xrt-smi` — hyphenated, and not on PATH.
- `xrt::profile::user_event` works and emits a VTF trace.

**Do NOT use a power dose-response as proof.** It was tried here and it is
confounded: scaling the batch scales the *host* attention too, so a rising reading
may be entirely host work. It did rise (8x the row-passes -> 2.05x power, 0.972 W
-> 1.993 W, from 0 at rest), and it would have passed just as well with the
encoder doing nothing on the NPU. Flagging it here so it is not repeated.

### 5.0.1 Prove a kernel executes by corrupting it

The direct test, and the one to reach for first. Flip a few bytes in the middle
of an `insts_*.bin` AIE instruction stream, leave the size unchanged, and run:

```
pristine   cd0ad90a2f20d7724e6b6c23e30eaffb  -> completes, logits printed
corrupted  0adf1b4e3005ecd595780752eee28d56  -> REFUSED: gemm_rtp: kernel state 5
```

The device **executed the corrupted instructions and reported a bad kernel
state.** No host fallback can produce a device kernel state, so this is direct
proof of dispatch. It is strictly stronger than a power or timing signal, it
costs one extra run, and it cannot be confounded by host work.

**Confirm the subject of any `/proc` probe.** `pgrep -f layaacc` matched the *bash
wrapper*, because the shell's own command line contains the rig's path. 35
samples were taken of a shell whose only device fd was `/dev/null`, and the
plausible conclusion was "the engine never opens the device". Match on
`/proc/PID/exe`, not on a command line that merely mentions the name.

**Prove the refusal as well as the success.** A run that works does not
distinguish "dispatched" from "quietly computed on the host". Pointed at an
empty design directory, the rig exits 1 in 0 s naming the `design.json` it could
not find -- so there is no host fallback, and the only route to an answer is
through the AIE.

### 5.0.2 Ask the tool where the work is before measuring it

`laya_decision_rig` prints its own dispatch accounting, and reading it is faster
than any profile:

```
designs    ONE xclbin, 12 streams (3 batch tiers), one hw_context
gelu       on the HOST (fp32) -- 22 fewer NPU dispatches
softmax    on the HOST (fp32) -- 22 fewer NPU dispatches
layernorm  on the HOST (fp32) -- 45 fewer NPU dispatches
weights    220.59 MB staged on the device once, not per call
```

Here it says the AIE runs **only the four projections per layer (88 dispatches)**
while GELU, softmax and LayerNorm are host fp32 -- 89 host-side ops per 22
layers. That single line explains both symptoms that otherwise look like a
broken NPU path: **high CPU, and ~1 W of device power.** "Idle NPU" and "a design
that only offloads the GEMMs" are indistinguishable from power alone. It also
matches the earlier profile that put `softmax_cpu` first at 39.4% of host CPU,
LayerNorm third.

Memory follows from the same fact. Host attention materialises
`batch x heads x seq^2 x 4 B` -- measured 0.09 / 0.38 / **1.50 GB** at batch
2 / 8 / 32, on a ~1.47 GB floor (1.08 GB container + 220 MB device-staged
weights), for peak RSS of 1.47 / 1.48 / **6.04 GB**. A reading *below* the floor
is a different process or an early sample. This is why tiers cap at 32 with no
batch-64 tier, and it is **not** a tuning knob: no allocation flag shrinks
`batch x heads x seq^2`. Moving attention onto the device is a kernel-design
question, not a configuration one.

Full chain, the confounded power test, and the limits are in
`.local/laya-implement/results/npu-dispatch-proof.md`.

### 5.1 - 5.4 Dead ends, and where the source lives

Moved to [`references/dead-ends-and-provenance.md`](references/dead-ends-and-provenance.md):

- **5.1** `xrt-capture` / `xrt-replay` copy the xclbin and record nothing.
- **5.2** the `mlir_aie` Python API does not import in this venv.
- **5.3** two claims this skill previously made, both of which were *my own*
  measurement errors -- read this before repeating either.
- **5.4** where the driver and XRT source live, and how to confirm it matches
  the running modules.

## 6. The rule that keeps this from wasting a day

> **A tool that produced artifacts has not measured anything. Check the artifact.**

Every false positive here looked fine:

* `xrt-capture` -> rc 0, 130 KB of artifacts, all nulls, and a byte-identical copy
  of the input;
* `aiebu-dump` -> rc 0, no output, because it needed flags;
* `sensors` NPU power -> a confident two-digit number, 1000x wrong;
* fdinfo busy time -> a plausible nanosecond counter that never moved;
* the in-repo `t_npu` -> present, named, documented, and zero.

The checks that catch all five: `md5sum` the artifact against the input; sample a
counter **twice during known work** and confirm it moves; cross-check a magnitude
against an independent source; and treat a documented field that always reads zero
as absent, not as a fast path.

---

## 7. A working recipe, end to end

For "where does the encode go", on this stack:

1. **One run at a time** (`host-discipline` 3), scratch on a real filesystem.
2. **gprof** for the host function ranking (2). This is the primary tool.
3. **The rig's timers** for the phase view, **knowing `t_npu` is 0** (3), and take
   `gemm`'s per-call total from gprof as the NPU-path number.
4. **`strace -c -e trace=ioctl,mmap,openat`** for the dispatch shape. Measured:
   265 `mmap`, **504 `ioctl`** for 88 dispatches (~5.7 per dispatch), 69 `openat`.
   A sudden change in the ioctl-per-dispatch ratio means the command queue changed
   shape, which matters more than the absolute count.
5. **`aiebu-dump -p -m aie2ps`** per kernel to confirm the shipped code is the code
   you think (4.1).
6. **`xclbinutil` metadata** to confirm the shipped graph/tile layout (4.2).
7. **`amd-smi metric`** for DRAM MB/s and package W during the run, so "the host is
   the bottleneck" and "the device is starved of bandwidth" can be told apart
   (`host-discipline` 7.2).
8. **Repeat interleaved, >=3 reps, and report the spread.** If arms overlap within
   noise, the answer is "no measurable difference".

## 8. Rules

- **One run at a time; scratch never on tmpfs** (`host-discipline`).
- **gprof needs its own `-pg` binary**; never run it against a non-instrumented one.
- **Never compare gprof's CPU-time to the timers' wall-clock.** Label which.
- **Treat `t_npu == 0` as "not measured"**, and take the NPU-path number from
  `gemm`'s per-call total instead.
- **AIE kernel and graph inspection are static.** They describe the artifact, never
  the run.
- **There is no absolute AIE execution counter.** Per-dispatch NPU time *is*
  derivable from fdinfo `drm-engine-*` over `command_submissions`, and column
  utilisation works -- use those (5.0). Do not synthesise a substitute from a
  counter that does not move, and do not repeat the `pgrep -f` pid mistake.
- **Check the artifact, not the exit code** (6).
- **Keep a "dead ends" list** -- 5 exists so the next person does not spend a day
  re-proving that `xrt-capture` is inert.
