# Laya encoder: layer-by-layer NPU migration plan

Target: `laya-decision:multilingual` on Strix Halo (AIE2P, NPU2), moving encoder
operations from the host onto the NPU one verified layer at a time.

Every shape, parameter name, and constraint below was read from the working tree
at the commit this document landed on. Where a number comes from the driver,
XRT, or upstream mlir-aie, the source is named. Nothing here is inferred from a
sibling model's recipe.

Companion artifact: [`laya-npu-dataflow.html`](./laya-npu-dataflow.html) (offline,
no external assets) — per-layer op inventory, buffer sizes, and dispatch sites.

---

## 1. Where the work is today

Laya multilingual is mmBERT-base: 22 layers, hidden 768, 12 heads x 64,
intermediate 1152, seq 1024, gated GeGLU, `position_embedding_type: sans_pos`
(no positional add, so no RoPE to offload).

Of the eleven operations per layer, four projection GEMMs already dispatch to
the NPU. The seven that do not are the migration backlog:

| Op | Where | Buffer at batch 32, seq 1024 | Notes |
|---|---|---|---|
| `qkv` | NPU | — | `insts_qkv_b{0,4}.bin` |
| `attn_out` | NPU | — | |
| `ffn_up` | NPU | — | |
| `ffn_down` | NPU | — | |
| `qk` | host | 1.61 GB fp32 scores | **needs a real dispatch branch** |
| `softmax` | host | 1.61 GB in/out | branch exists, disabled |
| `av` | host | 1.61 GB in | **needs a real dispatch branch** |
| GELU | host | | branch exists, disabled |
| LayerNorm x2 | host | | branch exists, disabled |

"NPU branches already exist" is precise and load-bearing: `npue_encoder.hpp`
contains dispatch code for LayerNorm, GELU, and softmax. They are switched off
in unified mode because the shipped xclbin has no eltwise designs. Enabling them
is a build-and-attach job. `qk()` and `av()` have **no** dispatch branch at all
and are real code work.

### Why attention is first despite being 24% of the math

Per layer, projection GEMMs are 5,013,504 MACs and attention is 1,572,864 MACs —
attention is 23.9% of the arithmetic but **100% of it runs on the host**. It also
dominates memory: the `scores` tensor is 1.61 GB fp32 at batch 32 and peak RSS
reaches 6.04 GB. Cost is not the only axis; peak memory and host O(seq^2) work
are what actually gate this model.

### The band mask is a second, larger win

Every third layer is global; the rest are local with `half_window=64`. Local
layers only need `|i-j| <= 64` — 129 cells per row, 16,641 of 1,048,576 score
cells, **1.59%**. But the mask is applied on the host *after* softmax, so a
banded layer still materialises and walks the entire matrix. Mean fraction
genuinely needed across the 22 layers is 37.4%; a band-aware kernel would skip
~87% of the score work on average. This is a design decision to make
deliberately, not an accident to inherit.

---

## 2. Step 1 — LayerNorm at hidden 768

The smallest unit of work with a reusable oracle.

`open_kernels/designs/ln/` is a fused **add + LayerNorm**: it reads `x` and
`add` as fp32, writes `y = x + add` as fp32 and `xn = bf16(y * rsqrt(mean(y^2)+eps) * w)`.
The add is fused because LayerNorm immediately follows the residual add, so this
removes a full extra round trip of a 768-wide fp32 tensor per site, 45 sites per
request (1 embedding + 2 per layer).

Shape contract, from `designs/ln/ln.h`:

```c
static constexpr unsigned kN     = LN_N;
static constexpr unsigned kHalf  = LN_N / 2;
static constexpr unsigned kV     = 32;
static_assert(kHalf % kV == 0, "LN_N must be a multiple of 64");
```

`(LN_N/2) % 32 == 0` is equivalent to `LN_N % 64 == 0`. **Laya's `LN_N=768`
passes** (kHalf=384, 384 % 32 = 0). Existing builds are 2048, 2560, 3072, 3840,
4096 — all satisfy it too, so this is a normal parameterisation, not a new shape
class.

Build:

```bash
cd open_kernels/designs/ln
LN_N=768 LN_EPS=1e-5 python3 ln.py
```

Then, in order, with no step skipped:

1. **Prove engagement.** Flip 36 bytes of the emitted `insts.bin` and confirm the
   run refuses (`kernel state 5`). A kernel that silently produces the same
   output is worse than one that crashes. This is the corruption test already
   used for the projection GEMMs.
2. **Prove correctness.** `utilities/laya_preln_reference.py` is the float64
   oracle; compare against it on adversarial inputs (large mean, near-zero
   variance, mixed sign) before trusting any timing.
3. **Only then** set `host_ln=false` for the unified path and re-measure.

Hardware notes for the build, all from `AIETargetModel.h`
`BaseNPU2TargetModel` and the kernel driver's own documentation:

- `rsqrt` is scalar-only on AIE2P (this is `srsqrt`), so the reduction dominates
  the kernel. Expect the 768 build to be latency-bound on the horizontal
  reduction, which AIE2P has no single instruction for.
- **`getComputeTileMaxVectorAlignBits() = 512`.** A full-width vector access needs
  512-bit alignment even though the load/store bus is 256-bit. A buffer that is
  only 256-bit aligned will split or fault. This is a new constraint relative to
  AIE2 and is easy to trip.
- The L2 budget is **512 KB per column**, not the 3 MB an earlier note claimed:
  there is exactly one memtile row (`getNumMemTileRows() = 1`) and
  `getMemTileSize() = 0x80000`. The kernel docs independently state 4096 KB total
  for Strix Point, i.e. 8 × 512 KB. At `LN_N=768` the whole layer plus its
  weights is about 4.6 KB, so capacity is not the constraint here — bandwidth and
  the reduction are.
- AIE2P exposes **1–7 virtualized column counts**, not just the full 8
  (`VirtualizedNPU2TargetModel`, `TK_AIE2_NPU2_1Col`..`_7Col`). A design that
  wants all 8 must ask for them explicitly and cannot assume it.

---

## 3. Step 2 — fused encoder flash attention

This is the largest single win, and a verified design already exists.

`open_kernels/designs/whisper_fa/` implements non-causal fused encoder attention
and is hardware-verified. Its C++ sources (`attn_npu2.cc`,
`attn_cascade_wrap.cc`) are used unmodified by Whisper's encoder; only the IRON
topology in `attn_fa.py` is project-specific.

**Corrected parameter mapping.** The shape is driven by `FA_*` environment
variables (`attn_fa.py:928-937`), and the column budget is
`FA_HEADS_PER_UNROLL * NQ` with `NQ=4` — *not* `FA_NUM_HEADS`. The design runs
`HEADS_PER_UNROLL` head-groups concurrently across `HEADS_PER_UNROLL * NQ`
physical columns, then loops. AIE2P has 8 columns, so `2 * 4 = 8` fits exactly.

| Variable | Whisper (verified) | Laya |
|---|---|---|
| `FA_LQ` / `FA_LK` | 1536 | **1024** |
| `FA_VALID_LEN` | 1500 | **1024** |
| `FA_LQP` / `FA_LKP` | 256 / 64 | 256 / 64 |
| `FA_DK` / `FA_DV` | 64 / 64 | 64 / 64 |
| `FA_NUM_HEADS` | 20 | **12** |
| `FA_HEADS_PER_UNROLL` | 2 | 2 |
| `FA_CASCADE_STAGES` | 4 | 4 |

Laya is strictly *smaller* than the verified Whisper point on every axis that
presses memtile budget (seq 1024 < 1536, heads 12 < 20), which is the good
direction. Divisibility holds: `12 * 64 = 768`, `768 % (4 * 64) == 0`.

Two cautions the driver documentation adds. First, column allocation is the
driver's **Resource Solver** decision, made from workload-declared hints plus its
own heuristics and enforced by firmware — requesting 8 columns is not a guarantee
of receiving them. Second, each workload context costs a **64 MB host-resident
instruction buffer** regardless of design size, so a multi-design family pays that
per context, not per dispatch. Strix Point supports 16 concurrent contexts.

```bash
cd open_kernels/designs/whisper_fa
FA_LQ=1024 FA_LK=1024 FA_VALID_LEN=1024 FA_LQP=256 FA_LKP=64 \
FA_DK=64 FA_DV=64 FA_NUM_HEADS=12 FA_HEADS_PER_UNROLL=2 FA_CASCADE_STAGES=4 \
python3 attn_fa.py
```

**Open question that must be settled before integration:** Whisper is non-causal
with no padding, so `FA_VALID_LEN` is a pure length. Laya pads to `seq=1024`
with a prefix mask. Whether `FA_VALID_LEN=1024` (full) or the true unpadded
length is correct — and how the pad mask reaches the kernel at all — is
**unverified**. Get this wrong and attention silently attends to pad tokens.
Settle it by feeding a deliberately pad-heavy batch and comparing against the
host reference, not by reading the code.

Fusing `qk + softmax + av` in one kernel also collapses the 1.61 GB score tensor
into per-stage tiles, which is what actually removes the 6.04 GB peak.

---

## 4. Step 3 — GELU, then a banded-softmax design

Gated GeGLU, so the host applies GELU to the gate half before the elementwise
product. The dispatch branch exists; the missing piece is an eltwise design in
the family xclbin. Hardware: AIE2P has native `vexp2` and `vtanh`, so erf-GELU
is expressible, but **not** a fused erf — expect two transcendentals plus a
multiply.

Softmax is the harder one. It is not merely "move an eltwise op": it needs the
row max and row sum, which on AIE2P means a horizontal reduction with no single
instruction. The right answer is almost certainly to fuse softmax *into* the
attention kernel from step 2 rather than build a standalone softmax — which is
another reason step 2 comes before step 3.

For the local layers, prefer a banded kernel over a full-matrix one: it is 1.59%
of the work, and it is the only version that scales when batch or sequence grows.

---

## 5. Build order, and why

1. **LayerNorm 768** — smallest, oracle already exists, reuses an existing
   design, exercises the full build -> attach -> verify -> enable loop that
   every later step needs. Get the loop right while the stakes are low.
2. **Flash attention at Laya shape** — biggest win (host O(seq^2) and 6.04 GB
   peak both disappear), design already hardware-verified, and it subsumes the
   standalone softmax question.
3. **GELU** — eltwise, low risk, no reduction.
4. **Banded softmax** — only as a component of a banded attention kernel.

Steps 1 and 3 are genuinely independent and can proceed in parallel; step 2 is
the long pole.

---

## 6. Instrumentation available for every step

Verified on this host, so each step can be measured rather than asserted:

- **Per-dispatch NPU time**: fdinfo `engine_active` delta divided by
  `DRM_AMDXDNA_QUERY_HW_CONTEXT` `HW_CONTEXT_ALL.command_submissions` delta.
  Cross-checked at 40.56 ms against a 40.6 ms host wait.
- **Column utilisation**: `DRM_AMDXDNA_QUERY_SENSORS`, 8 records.
  `utilities/npu-sensors` had a protocol bug that made this read as dead; fixed
  in `0d176f0`. Previously it also failed to compile, because it referenced
  `AMDXDNA_SENSOR_TYPE_TEMPERATURE`, which the installed uapi header does not
  define.
- **Power**: `power1_input` is **microwatts**; the underlying `u16 npu_power` is
  mW, so it saturates at 65.5 W. Under Laya it reads ~0.93–1.99 W. Treat power as
  a sanity check only — host work dominates the socket, so power delta is
  confounded and is **not** engagement evidence.
- **Clocks**: observable (MP-NPU ~1267 MHz) but not settable unprivileged.
- **Dead ends, do not re-explore**: `xrt::aie::profiling` throws;
  `Debug.aie_profile=true` segfaults; `xrt-capture` produces no data;
  `xrt-capture`/AIE PC trace unavailable.

---

## 7. Gates each step must pass

1. **Engaged.** Name the observable that must differ, in the predicted
   direction. If the arms are indistinguishable, the kernel did not run — stop.
2. **Identity.** Output matches the float64 oracle within a bound stated *before*
   the run. For a banded path, the count of score cells touched must equal
   16,641 x rows, not 1,048,576 — a banded kernel that still walks the full
   matrix passes an output check while doing 63x the work.
3. **Conserved.** No dropped rows, heads, or layers.
4. **Peak RSS.** Must not regress. Attention should *drop* it substantially.

## 8. Known-blocked, do not attempt as a workaround

- No PyTorch/Transformers reference on this host, so the irreducible
  fp32-vs-bf16 noise floor is unknown. State a bound; do not invent one.
- Upstream mlir-aie `iron/operators/mha` exists at `3fc9a60` and would be the
  natural way to write this, but the installed build is 1.4.2 and does not
  provide `operators/`. Using it requires approval to change the environment.
- MSVC and full HRX builds are unavailable here.
- **AMD publishes no AIE2P architecture manual.** Per-core MAC counts and cascade
  rules therefore come from generated Peano disassembly and `aie_api` config,
  which is weaker evidence than the array geometry — the geometry is confirmed by
  `AIETargetModel.h`, `amdnpu.rst` and `xrt-smi examine` agreeing, but the
  throughput table has no such triple. Treat it as a working figure.
- Two commonly-reached sources are **wrong for this chip** and must not be merged:
  `pp4fpgas.readthedocs.io` describes AIE1 (4 columns, Versal AI Core), and
  `amd.com/.../ai-engine.html` is Versal AIE/AIE-ML marketing that contradicts
  itself on INT8 throughput. See *How to read the sources* in the visualization.
