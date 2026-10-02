# XDNA2 / AIE2P reference: hardware, API, dependencies

Everything needed to write and debug a kernel on the AMD client NPU (XDNA2,
Strix Halo, AIE2P) without re-deriving it. Written 2026-10-02 against a
Strix Halo host.

**Every number here was read from a source or measured on this box.** The
provenance of each block is stated. Where a figure is weakly sourced, it says
so — see §11 and `sources-and-standing.md`. Where AMD publishes nothing, this
document says that rather than filling the gap.

---

## 1. Identify the part before anything else

| part | target triple | columns | notes |
|---|---|---|---|
| Phoenix, Hawk Point (XDNA1) | `aie2-none-unknown-elf` | 5 | older client NPU |
| **Strix Halo (XDNA2)** | **`aie2p-none-unknown-elf`** | **8** | this class of hardware |

Get this wrong and every number below is wrong. `llvm-aie` also registers
`aie`, `aie2ps` — `aie2ps` is a **Versal** part (36 columns), not a client NPU.

Verify rather than assume:

    ./ironvenv/lib/python3.*/site-packages/llvm-aie/bin/llc --version | grep aie

Confirm the device:

    build/src/oflm validate
    # [Linux] NPU: /dev/accel/accel0 with 8 columns
    # [Linux] NPU FW Version: 1.1.2.65
    # [Linux] amdxdna version: 0.15
    # [Linux] Memlock Limit: infinity

---

## 2. Array geometry — authoritative

From `third_party/mlir-aie/include/aie/Dialect/AIE/IR/AIETargetModel.h`,
classes `BaseNPU2TargetModel` and `NPU2TargetModel`:

```
rows()   = 6      // "1 Shim row, 1 memtile row, and 4 Core rows"
columns() = 8     // NPU2TargetModel
getNumMemTileRows() = 1
getTileType(col,row): row 0 -> ShimNOC, row 1 -> MemTile, rows 2..5 -> CoreTile
```

So **32 compute tiles, 8 memtiles, 8 ShimNOC**, in 8 columns of 6 rows. The
row numbering is not cosmetic: cascade designs place compute work at
`row = 2 + stage`, which is why `designs/whisper_fa` uses `row = 2 + ty`.

The kernel driver documentation (`docs.kernel.org/accel/amdxdna/amdnpu.html`)
independently states *"AMD Strix Point client APU have 4x8 topology"* and
*"AMD Strix Point NPU has a total of 4096 KB of L2 memory"*. Three sources agree
(`xrt-smi examine` also reports 6x8). **This block is high confidence.**

---

## 3. Memory hierarchy

| resource | value | source |
|---|---|---|
| L1 data memory / core | **64 KB** (`getLocalMemorySize() = 0x10000`) | AIE2, inherited |
| program memory / core | **16 KB** (`0x4000`) | AIE2, inherited |
| memtile | **512 KB** (`getMemTileSize() = 0x80000`) | AIE2, inherited |
| total L2 | **4 MB** = 8 × 512 KB | driver doc + arithmetic |
| memtile banks | 8 (core tiles: 4) | `getNumBanks()` |
| core data address space | 1 MB (`0x100000`) | `getCoreDataAddressSpaceSize()` |
| accumulator cascade | 512 bits | `getAccumulatorCascadeSize()` |
| address-generation granularity | 32 bits | `getAddressGenGranularity()` |

**Per-column L2 is 512 KB, not more.** There is exactly one memtile row. An
earlier note in this repo claimed "3 MB addressable per column" and was wrong.

**`getComputeTileMaxVectorAlignBits() = 512` on NPU2**, while
`getComputeTileLoadStoreBusWidth() = 256` on AIE2. A full-width vector access
therefore needs **512-bit alignment even though the bus is 256-bit**. A buffer
that is merely 256-bit aligned will split or fault. This is new relative to
AIE2 and is the single easiest constraint to trip.

Address generation is 32-bit granular, so **a DMA can never perform a layout
transform at sub-32-bit element granularity**: an int8 or bf16 swizzle must be
expressed as in-core `vshuffle`, not as a BD wrap/stride.

---

## 4. Locks, buffers, channels

| resource | memtile | core / shim |
|---|---|---|
| locks | **64** | **16** |
| buffer descriptors | **48** | **16** |
| BD dimensions | **4** | **3** |
| BD max length | `(1<<17)-1` = **131,071** | core `(1<<14)-1` = 16,383; shim 32-bit field |
| max lock value | `0x3F` = **63** (6-bit) | same |

`getMaxChannelNumForAdjacentMemTile() = 4`.

Practical guidance, not from the header: **≤8 objects in L1**; deeper buffering
goes to a memtile ObjectFIFO with depth ≥4. A 2-input kernel already consumes
both Shim MM2S channels of a column, so a 3-input mapping **cannot** be fed from
a compute tile.

---

## 5. Vector datatypes and throughput

Per core, from Peano disassembly and `aie_api` config — **weak provenance, see
§11**:

| datatype | MAC/cycle | status |
|---|---|---|
| int8 | 512 | native 8×8×8 |
| bfp16 (block float) | 512 | native 8×8×8 |
| **bf16** | **256** | native 4×8×8 — the default path |
| int16 | 256 | native 4×4×8 |
| **fp32** | **0** | **emulated** via bf16 |

`ironvenv/.../aie_api/detail/aie2p/config.hpp` states
`#define __AIE_API_FP32_SUPPORT__ 0`, and
`__AIE_API_FP32_EMULATION__` is enabled for model ≥ 10600. AMD's marketing page
independently calls FP32 *"SW emulation"*.

**Consequence:** moving an fp32 host op to the AIE costs an emulation, not a
native unit. bf16 8×8×8 needs two native 4×8×8 ops and two accumulators.

### Native vs scalar vs absent — measured from disassembly

| instruction | status |
|---|---|
| `vexp2` | native, exp2, 512-bit, 16 fp32 lanes |
| `vtanh` | native, tanh, 512-bit — **so tanh-approx GELU is ~2 vector ops** |
| `vmac.f` | 1/cycle, latency 6 |
| `vld`/`vst` | vector load/store, 7c / 1c |
| `vconv` | fp32 → bf16 |
| `inv`, `invsqrt` | **scalar only**, 83 ops per 32 lanes |
| `sqrt` | **scalar only**, 13 ops per 32 lanes |
| `sin`, `cos` | **not implemented** (TODO CRVO-7950) |
| `sigmoid`, `log2` | **do not exist** in `aie_api` |
| horizontal reduce | **no instruction** — O(log lanes) shuffle tree |

`vtanh` being native is good news for GELU. `rsqrt` being scalar-only is the
LayerNorm trap: a naive design emits 83 instructions per 32-lane vector, so the
reciprocal dominates. A native LayerNorm needs rsqrt decomposed (power-of-two
seed plus Newton iterations on the `vexp2` path) — measurable, unclaimed work.

---

## 6. Cascade and movement

- Cascade is **512 bit/cycle**, one accumulator per cycle, direction N↔S.
- **One in, one out; endpoints adjacent; no NoC access, no memory-tile or shim
  involvement.** A cascade chain can never reach host memory.
- The last core must land the result in L1 for a normal ObjectFifo.
  4 cores/column → `FA_CASCADE_STAGES = 4` is exactly the AIE2P structure.
- Shim and compute tile each have **2 MM2S + 2 S2MM** channels. Memtile has 6.
- NPU DDR ≈ **50 GB/s effective**, measured; shared with the CPU and contended,
  so **not a dedicated NPU figure**. The only NPU path to DDR is shim
  `ShimNOC → ShimSIC`.

---

## 7. Pre-build constraint checklist

**Check every one of these before running a design's build script.** Each was
found by reading source after a collision; all are silent failures otherwise.

| # | constraint | where it lives | failure mode |
|---|---|---|---|
| 1 | `LN_N % 64 == 0` | `designs/ln/ln.h` — `static_assert(kHalf % kV == 0)` with `kV = 32`, so `(LN_N/2) % 32 == 0` | assert fires; note the *message* says 64 while the code says 32 |
| 2 | FA column budget = `FA_HEADS_PER_UNROLL × NQ` (NQ = 4), **not** `FA_NUM_HEADS` | `designs/whisper_fa/attn_fa.py:26-29` | silently wrong shape. `2 × 4 = 8` fits AIE2P exactly |
| 3 | design seq: positive, multiple of 8, ≤ `max_positions` | `npue_encoder.hpp` `set_design_seq` | throws at load |
| 4 | full-width vector access needs **512-bit** alignment | `getComputeTileMaxVectorAlignBits()` | split or fault |
| 5 | ≤8 objects in L1; memtile ObjectFIFO depth ≥4 for deeper | design guidance | port/BD exhaustion |
| 6 | a 3rd input cannot be fed from a compute tile | channel budget | switchbox cannot be fed |
| 7 | request columns explicitly; 1–7 are legal virtualized sizes | `VirtualizedNPU2TargetModel`, `TK_AIE2_NPU2_1Col.._7Col` | silently given fewer than 8 |

Note on 1: a shape that satisfies the *assert* may still be wrong. Read the
expression, not the message.

---

## 8. Runtime, from the kernel driver documentation

`amdnpu.rst`, authoritative:

- **16 concurrent workload contexts** on Strix Point (6 on Phoenix/Hawk Point).
- **64 MB host-resident instruction buffer per context**, PASID-protected,
  mapped into both the firmware context and user space. Paid per context
  whether or not the design is large.
- Partitioning is at **column granularity**; the microcontroller programs
  column-isolation registers and each partition gets its own PASID.
- A **Resource Solver** allocates columns from workload-declared hints plus its
  own heuristics, and firmware enforces the binding. *A design cannot assume it
  receives the columns it asked for* — see constraint 7.
- Submission: host compiles an **overlay** (stream-switch config + per-tile ELF)
  and a **`ctrlcode`** opcode stream. Context open → Resource Solver provisions
  columns → MERT creates an ERT and maps the 64 MB buffer → userspace writes
  `ctrlcode`, submits a command buffer over the mailbox, sleeps. ERT *executes*
  `ctrlcode`, which kicks off DMAs while the array runs, then raises MSI-X.
- **DMA is encoded as `XAIE_IO_BLOCKWRITE` inside `ctrlcode`.** A host dispatch
  call is *not* the same event as a data movement being issued.
- Firmware telemetry exists (L1 interrupt, DMA, deep-sleep counters) with no
  userspace path on this box.
- Its worked `drm-total-memory` example is a **usage sum** (heap + internal +
  external). That figure is *not* a capacity or a carve-out limit.

---

## 9. Userspace and telemetry — what actually works

| instrument | verdict | how |
|---|---|---|
| `drm-engine-amdxdna_accel_driver` fdinfo | **works** | `/proc/<pid>/fdinfo` on the accel fd; ~0.8% of wall clock |
| per-dispatch NPU time | **works** | fdinfo `engine_active` delta ÷ `HW_CONTEXT_ALL.command_submissions` delta. Cross-checked 40.56 ms vs 40.6 ms |
| column utilisation | **works** | `DRM_AMDXDNA_QUERY_SENSORS`; 8 records; **98% peak after ~4 s warmup**, ~5 s decay |
| NPU power | works, **µW** | `hwmon/hwmon12/power1_input`; underlying `u16` mW saturates at **65.5 W** |
| NPU clocks | observable, **not settable** | `DRM_AMDXDNA_QUERY_CLOCK_METADATA`; MP-NPU ~1267 MHz |
| `xrt::profile::user_event` | **works** | writes a VTF trace |
| `xrt::aie::profiling` | throws "not supported" | do not retry |
| `Debug.aie_profile=true` | **segfaults** | in `AieProfilePlugin` at exit |
| `xrt-capture` | **inert** | output byte-identical to input; manifest fields null |
| tracepoints, debugfs | root, and effectively empty | |
| `SET_STATE` (pin clocks) | **root only** | so every benchmark is unlocked — label its operating point |

Two traps that make working instruments look dead:

1. **`pgrep -f` finds the wrong process.** The shell's own command line
   contains the binary's path, so `pgrep -f layaacc` matches the *wrapper*.
   Resolve the pid via `/proc/*/fd → accel`, never by name pattern.
2. **Column utilisation has no size-query pass.**
   `amdxdna_query_sensors()` guards each record with
   `if (buffer_size < sizeof(sensor)) goto out;` then writes
   `buffer_size = sensors_count * sizeof(sensor)`. A zero-size probe skips every
   record and returns **rc 0** — indistinguishable from success. Pass a full
   16-record buffer. (`utilities/npu-sensors` does this and returns **9**
   records: 1 power + 8 columns.) The installed uapi header defines only
   `AMDXDNA_SENSOR_TYPE_POWER` and `..._COLUMN_UTILIZATION`; referencing
   `..._TEMPERATURE` does not compile, and the driver skips temperature without
   `HAVE_7_2_AMD_PMF_NPU_METRICS_NPU_TEMP` anyway.

### Reading the ERT command state

`run_kernel` and the engine both report a state number. The enum is in
`src/include/hrx_cpp/hrx_cpp.hpp`, and two values carry most of the signal:

| state | name | meaning |
|---|---|---|
| **4** | `ERT_CMD_STATE_COMPLETED` | the run finished |
| **8** | `ERT_CMD_STATE_TIMEOUT` | it never signalled completion |

This makes the corruption test self-interpreting. The LayerNorm 768 kernel
returns **state 4** in 0.295 ms with intact instructions and **state 8** after a
4087 ms timeout with 36 bytes flipped — the kernel started and hung, rather
than faulting cleanly. So a state-8 result is not automatically "the driver is
broken"; it is what a kernel that spins without completing looks like. Always
compare against the known-good state for the same design before concluding
anything from the number.

The full enum: NEW 1, QUEUED 2, RUNNING 3, COMPLETED 4, ERROR 5, ABORT 6,
SUBMITTED 7, TIMEOUT 8, NORESPONSE 9, SKERROR 10, SKCRASHED 11.

**Never use a power dose-response as engagement evidence.** Scaling the batch
scales host work too; a rising reading may be entirely host. To prove a kernel
ran, corrupt its instruction stream and require a device error — no host
fallback can produce a device kernel state.

---

## 10. Toolchain and dependencies

Exact versions on this host, which is also what built the shipped xclbins
(`src/open_npue/SYNCED.md`):

| component | version | location |
|---|---|---|
| Peano | **clang 21.0.0 @ `c9c5ecb`** | `./ironvenv/lib/python3.*/site-packages/llvm-aie/bin` |
| mlir-aie | `1.4.2.dev16+g7e00b57` | `third_party/mlir-aie` |
| XRT | **2.25** | `/opt/xilinx/xrt` |
| amdxdna driver | `2.25.0_20260628` @ `e2d8f83` | `~/repos/xdna-driver` |
| NPU firmware | `1.1.2.65` | on device |

Dependency notes:

- **The venv's Python minor version tracks the system Python and moves.** It was
  `3.12`, it is now `3.14`. Glob `python3.*`; do not hardcode.
- **Peano is deliberately not on PATH** (it would shadow system clang). Invoke it
  by full path.
- **XRT can be installed twice.** This host has 2.20 in `/usr/include/xrt` and
  2.25 in `/opt/xilinx/xrt`. The build now gates on version
  (`OFLM_XRT_EXPECTED_VERSION`, default 2.25) and warns about a split. Note the
  two layouts expose *different* macros — 2.20 defines `XRT_VERSION_MAJOR`/
  `_MINOR`/`_PATCH` in `version.h`; 2.25 defines `XRT_VERSION_CODE` and
  `xrt_build_version[] = "2.25.0"` in `version-slim.h` and **no
  `XRT_VERSION_MAJOR` at all**, so a naive check passes vacuously.
- The running driver module reports its own version; **check that, not just your
  checkout**. Here they match exactly.
- Upstream `mlir-aie` has `iron/operators/` (mha, rope, rms_norm, softmax, flm)
  at `3fc9a60`, but the **installed 1.4.2 does not ship it**. Using it means
  changing the environment — get approval first.

### Model install: three things that bite

1. **`model_info.json` `oid` is a git `blob_id`, not a content hash.** For
   LFS-tracked files (every `.qclbin`, and the multi-hundred-MB `model.q4nx`)
   it is the hash of the LFS *pointer*, so it cannot be reproduced from the
   bytes you downloaded. Verify against the hub's **`lfs.sha256`**. (Checked:
   sha256 of the downloaded bytes equals live `lfs.sha256` for all four
   kernels.)
2. **`oflm-add` requires `model.q4nx`.** Registry entries pointing at *unconverted*
   upstream repos (`convaiinnovations/laya`, `sentence-transformers/*`, `BAAI/*`)
   cannot be installed; only the converted `Atomic-Germ/*-NPU2` repos can.
3. **`oflm-add` does not download loose `.xclbin` files**, and this host has no
   system install for it to symlink from. The tool's own comment says custom
   models "never ship xclbins (they are closed source)", so it expects them at
   `<prefix>/xclbins/<model-dir>/` in an *installed* OpenFlowLM and skips the
   step otherwise. The converted HF repos *do* ship loose kernels; fetch them to
   `$OFLM_XCLBIN_PATH/<model-dir>/` yourself, or a run dies with
   `No such file '<xclbin_root>/<model-dir>/layer.xclbin'`. Kernel counts differ
   per model — Qwen3-0.6B ships 4 (`attn`, `dequant`, `layer`, `mm`), LFM2-1.2B
   ships 5 (adds `conv`).
   `model.q4nx` is a **weights** container, not a kernel container: a `u32`
   length then a JSON tensor map (`data_offsets`, `dtype`, `shape`) followed by
   the blobs. So there is no second copy of the kernels to fall back on.
   A build-tree binary also needs `OFLM_MODELINFO_PATH=src/model_info.json`.

### Open: freshly installed converted models time out on first dispatch

Both `qwen3:0.6b` and `lfm2:1.2b`, installed and sha256-verified, with the
loose kernels placed in the layout the tool expects, fail identically:

```
[OFLM] Prefill chunk 1/1 with 20 tokens
[ERROR] Insertion error: runlist failed execution (ERT_CMD_STATE_TIMEOUT)
Kernel Instance: MLIR_AIE
txn_op_idx = 0xFFFFFFFF
ctx_pc = 0x28B060AD
```

Systematic, not model-specific: two different architectures, two different
kernel sets (4 and 5 xclbins), same failure at the same point. And it is *not* an
idle device — column utilisation ramps 2 -> 98% during the attempt, so the array
genuinely executes and then never signals completion. `txn_op_idx = 0xFFFFFFFF`
is a sentinel rather than a real opcode index.

Unresolved. Ruled out so far: the xclbins are correct (sha256 matches the hub's
`lfs.sha256`); the layout matches what `link_xclbins` would create; there is no
second kernel copy to mismatch against. Not yet ruled out: stale kernels in the
repos relative to their `model.q4nx` runlists, or a closed-kernel/firmware
incompatibility against FW 1.1.2.65. Worth knowing before spending time on it:
the repo's own open-kernel path (`designs/`) is unaffected and is what the Laya
migration uses.

---

## 10a. Building a design: the recipe, and three undocumented traps

```bash
source ironvenv/bin/activate
export PATH="$PATH:/opt/xilinx/xrt/bin"        # aiebu-asm + xclbinutil
export LN_N=768 LN_EPS=1e-5                     # whatever the design reads
python3 open_kernels/build_design.py <design>.py <out_dir>
```

1. **`/opt/xilinx/xrt/bin` must be on `PATH`.** `aiebu-asm` and `xclbinutil`
   are not on it by default and `aiecc` dies with a bare "tool not found" deep
   in the pipeline (step 27/38 and 38/38 respectively), which reads like a
   kernel bug. That directory holds `aiebu-asm`, `aiebu-dump`,
   `aiebu-transform`, `xclbinutil` and `xrt-smi`; prefer it to the copies in
   the driver's XRT build tree. Note `AGENTS.md` points at
   `utilities/mlir-aie/utils/env_setup.sh`, which does not exist.
2. **The output directory is `argv[2]`**, defaulting to `designs/<name>/build`.
   Omit it and the build silently overwrites whatever was in `build/`. Outputs
   are gitignored, so nothing reaches git either way.
3. **The on-hardware harness is behind an option that defaults OFF:**
   `-DOFLM_BUILD_OPEN_KERNELS_HARNESS=ON`. Also pass `-DOFLM_BUILD_KERNELS=OFF`
   to skip the long kernel compile when you only want the harness, and expect to
   supply `-DOFLM_VERSION` and `-DNPU_VERSION`, which are required with no
   defaults.

Validation, once built, needs no packed model:

```bash
cd open_kernels/designs/<design>
LN_N=<N> LN_BUILD=<out_dir> python3 make_test.py    # fp64 reference + run.cfg
run_kernel run.cfg                                  # device state per run
python3 compare.py                                  # the gate; exit 0 = pass
```

`make_test.py` honours `LN_N` and `LN_BUILD`, and `run.cfg` pins the buffer
widths — a good check that the width really reached the design. Always run the
corruption arm: flip 36 bytes of `insts.bin`, and require the device state to
change and the run to fail. For the LayerNorm 768 kernel, good = state 4 in
0.295 ms, corrupted = state 8 after a 4087 ms timeout. Without that arm a
numerical pass could come from a host fallback.

### The run_kernel cfg grammar

The harness fails in ways that read like missing files. Three things to get
right, each of which cost a round trip:

```bash
device
xclbin G <abs path to final.xclbin>     # G is the xclbin's NAME
kernelx fa G <abs path to insts.bin>    # 3 tokens: kernel name, xclbin name, path
buf <name> <bytes> <path>
run <kernel> <bufs...>
```

- `kernelx` takes **three** tokens. Give it two and it reports
  `missing kernelx insts.bin` while echoing your real path back at you.
- The middle token of `kernelx` is the **name** given to `xclbin`, not a kind.
  Mismatched names give `no xclbin G`.
- Relative paths resolve against the cfg file's directory, so a cfg written to
  `/tmp` will not find the artifacts. Use absolute paths, or put the cfg beside
  the design.

Successful loads print `kernelx <name> (<path>, <N> words)`; `DONE runs=0` means
everything loaded and nothing was executed, which is the expected result for a
load-only check.

---

## 11. The honest gap

**AMD publishes no XDNA2 architecture manual.** Consequences, stated plainly:

- Array geometry (§2), memory sizes (§3), BD/lock limits (§4) — **high
  confidence**: `AIETargetModel.h` plus the kernel driver documentation agree.
- Datatype throughput and the native/scalar/absent table (§5) — **weaker**:
  from generated Peano disassembly and `aie_api` config. Directionally
  corroborated by AMD marketing, which is a different chip family.
- Per-column MAC figures must not be quoted from AIE1 course notes (4 columns,
  Versal) or AMD's Versal AIE/AIE-ML page — the latter also contradicts itself,
  claiming "128 INT8 MACs per tile" in one bullet and INT8 256 in the table
  above it.

Corroboration of one number does not transfer to its neighbour. See
`sources-and-standing.md` for the standing of every source and the method.

---

## 12. One worked sizing example

Why a "few hundred million parameter" model can need more memory than it
occupies. Laya multilingual: 22 layers, hidden 768, 12 heads, seq 1024.

```
scores = batch x heads x seq^2 x 4 bytes        // std::vector<float>
       = 32 x 12 x 1024^2 x 4
       = 402,653,184 floats = 1536 MiB = 1.61 GB
```

- It is **quadratic in sequence**, not related to parameter count. At seq 128
  the same work is 64× smaller.
- `seq` is the **design** length (`set_design_seq`), not the prompt length — a
  40-token prompt costs the same 1.61 GB at batch 32.
- The encoder's GEMM+LayerNorm parameters total ~90.9M → **173 MiB in bf16**.
  The score tensor has **4.4× more elements than the encoder has parameters**
  and 8.9× more bytes.
- The "only a few layers are global" argument does not help *today*: the band
  mask is applied on the host **after** softmax, so all 22 layers materialize
  and walk the full matrix. A banded layer with `half_window=64` needs
  129×129 = 16,641 cells = **1.59%**. Averaged over the stack only 37.4% of the
  work is real, so **62.6% of that 1.61 GB is computed and then masked away.**

**Rule: size an NPU offload by peak activation, not by parameter count.** That
is the number that decides whether offload is worth doing at all.