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

### DONE 2026-10-02 — built and validated

`LN_N=768` is built, numerically correct on real hardware, and proven to have
actually executed on the NPU. Results:

| gate | result | threshold |
|---|---|---|
| `y` (fp32 add) max rel err | **5.38e-08** | < 1e-6 |
| `xn` (bf16) cosine | **0.99999999** | > 0.999999 |
| `xn` max rel err | **1.263e-03** | < 8e-3 |
| bf16 element mismatches | **1 / 768** | < 5% |
| device state, good kernel | **4**, 0.295 ms then 0.087 ms | — |
| device state, 36 bytes corrupted | **8**, 4087 ms, `run k FAILED` | must differ |

The corruption arm is what makes the rest mean something: the same harness on
the same buffers returns state 4 in under a millisecond when the instructions
are intact and state 8 after a timeout when they are not. No host fallback can
produce a device kernel state, so the numbers above came from the AIE.

Build recipe (note the PATH entry — see below):

```bash
source ironvenv/bin/activate
export PATH="$PATH:/opt/xilinx/xrt/bin"     # aiebu-asm + xclbinutil, see below
export LN_N=768 LN_EPS=1e-5
python3 open_kernels/build_design.py open_kernels/designs/ln/ln.py \
       open_kernels/designs/ln/build_768_1e-05
```

Three build facts that cost time and are not documented anywhere:

- **`/opt/xilinx/xrt/bin` must be on `PATH`.** `aiebu-asm` and `xclbinutil` are
  not on it by default, and `aiecc` fails with a bare "tool not found" at steps
  27/38 and 38/38. `AGENTS.md` points at `utilities/mlir-aie/utils/env_setup.sh`,
  which **does not exist**. `/opt/xilinx/xrt/bin` has all four tools
  (`aiebu-asm`, `aiebu-dump`, `aiebu-transform`, `xclbinutil`, `xrt-smi`), so
  prefer it over the copy in the driver's XRT build tree.
- **`build_design.py` takes the output directory as `argv[2]`** and otherwise
  writes to `designs/ln/build`, silently overwriting whatever was there. Build
  outputs are gitignored, so nothing lands in git either way.
- **The harness is behind a CMake option that defaults OFF:**
  `-DOFLM_BUILD_OPEN_KERNELS_HARNESS=ON` (plus `-DOFLM_BUILD_KERNELS=OFF` to
  skip the long kernel compile, and `-DOFLM_VERSION` / `-DNPU_VERSION`, which
  are both required and have no defaults).

### The width assert has teeth — verified

```bash
LN_N=100 LN_EPS=1e-5 python3 open_kernels/build_design.py \
    open_kernels/designs/ln/ln.py /tmp/ln-teeth
# ln.h:24:15: error: static assertion failed due to requirement 'kHalf % kV == 0'
```

So the constraint is live rather than compiled out, and the build fails loudly
instead of silently producing a wrong kernel. `build_768_1e-05/insts.bin` is
836 bytes, sha256 `3c45bcd9...`, and differs from the pre-existing
`build_2048_1e-05/insts.bin` (same size, sha256 `2e9e5937...`) — the width
reached the kernel rather than defaulting.

### Still to do

Wire the built xclbin into the family bundle and set `host_ln=false` for the
unified path, then re-measure. The numerical and engagement gates above are
satisfied; what remains is integration, not kernel work.

---

## 3. Step 2 — fused encoder flash attention: VALIDATED at Laya's shape

Done 2026-10-02. Builds, runs, and is numerically correct on the AIE, with the
padding semantics settled empirically.

```bash
export FA_LQ=1024 FA_LK=1024 FA_VALID_LEN=1024 FA_LQP=256 FA_LKP=64 \
       FA_DK=64 FA_DV=64 FA_NUM_HEADS=12 FA_HEADS_PER_UNROLL=2 FA_CASCADE_STAGES=4
python3 open_kernels/build_design.py open_kernels/designs/whisper_fa/attn_fa.py \
       open_kernels/designs/whisper_fa/build_laya
# BUILD_OK -> final.xclbin 500110 B, insts.bin 100624 B
```

| gate | valid_len=1024 | valid_len=700 (324 pad rows) |
|---|---|---|
| device state | **4**, 3.93 ms | **4**, 3.95 ms |
| max rel err vs fp64 | 2.925e-02 | 4.134e-02 |
| p99 rel err | 9.885e-03 | 1.159e-02 |
| cosine | 0.99962830 | 0.99963657 |
| gates | max<6e-2, p99<2e-2, cos>0.999 | same |
| output tail rows | 0 padded rows | 1.31e-1, **not gated** — see below |

Builds: `final.xclbin` 500,110 B, `insts.bin` 100,624 B, 25,156 instruction
words, all accepted by the device. Runtime ~3.9 ms for 12 heads x seq 1024 fused
attention, i.e. qk + softmax + av in one dispatch with no host round trip.

A new harness lives beside the design: `designs/whisper_fa/make_test.py`
(vectors + float64 reference + `run.cfg`) and `designs/whisper_fa/compare_fa.py`
(the gate), mirroring the `designs/ln/` pair. The reference is computed from the
**bf16 values that were written**, not from the pre-rounding floats, so the
tolerance measures the kernel rather than input rounding.

### The error budget, and why the gate is where it is

bf16 output quantisation alone — round the float64 reference to bf16 and change
nothing else — gives, at this shape:

| | max | p99 | mean |
|---|---|---|---|
| bf16 output floor | 2.315e-03 | 6.027e-04 | 1.424e-04 |
| kernel, valid_len=1024 | 2.925e-02 | 9.885e-03 | 2.821e-03 |

The ~12x excess is the documented bf16 score accumulation plus the **bf16
online-softmax rescale factor** (`attn_npu2.cc`: *"all in float except the bf16
rescale factor r"*). So the gates sit roughly 2x above the measured maximum
rather than just above it. A gate fitted to the observed number catches nothing
and makes the next run look like a failure. `max_rel` is noise-sensitive — one
outlier element sets it — so the p99 gate carries most of the signal.

### Padding: `valid_len` masking works, and the output tail is the host's job

**The kernel masks correctly.** At a compiled-in `valid_len=700` with 324 padded
rows, the real rows match a reference that reads only rows `[0, 700)` within the
same error budget as the unpadded case (rel 4.1e-2 vs 2.9e-2, cos 0.99964 vs
0.99963). The design states `apply_length_mask (valid_len)` is *"the only mask,
unconditional"* (`attn_fa.py:44`), and that is confirmed.

**The output tail is not the kernel's contract.** The kernel writes real values
into rows `[t, seq_pad)` (1.31e-1 at valid_len=700). That is harmless: the host
contract never reads them — `FaAttention::dispatch_and_scatter` calls
`scatter_output()` for rows `[0, t)` and then `zero_pad_rows(out, t, m_padded, d)`
for the tail. My harness gated on the tail at first and reported FAIL for a
kernel that was in fact correct; the gate was removed and the value is now an
observation. **A gate that rejects correct work is worse than no gate.**

### Two traps that would have produced plausible-looking wrong answers

1. **`valid_len` is compile-time.** `attn_fa.py:255` declares
   `valid_len: CompileTime[int]` and bakes it via `-Dvalid_len=` at line 311. I
   first built at 1024 and then ran the harness with `valid_len=700`. The result
   was `cos = 0.99943` — apparently fine — with `max rel = 0.208`. Cosine is
   scale-free, so a uniformly shrunken output sails through it while being badly
   wrong. **Changing `valid_len` requires a rebuild, and cosine alone will not
   tell you.** This is the single most dangerous property of this design.
2. **The kernel applies `1/sqrt(dk)` itself.** `attn_npu2.cc:357` defines
   `log2e (1.44269504089 / constexpr_sqrt_dk)` with `constexpr_sqrt_dk = 8.0` at
   dk=64, folding the attention scale into the exp2 argument. Pre-scaling Q in
   the harness divides the scores by 8 **twice** — and because 1/8 is a power of
   two it survives bf16 rounding exactly, so the inputs look perfectly reasonable
   and only the output is wrong: rms 0.604x the reference, cos 0.667. I found
   this by checking whether the head mapping was still identity (it was), which
   pointed at a temperature mismatch rather than a layout error.

`export_whisper_kernels.py` notes that `FA_*` env overrides exist precisely as a
"debugging/smoke-shape convenience", so this build is that: a shape smoke test,
not a production artifact. Production shapes are owned by the exporter.

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

Build (from the repo root; see the reference for why `/opt/xilinx/xrt/bin` must be
on `PATH`, and note the output dir is `argv[2]`):

```bash
export FA_LQ=1024 FA_LK=1024 FA_VALID_LEN=1024 FA_LQP=256 FA_LKP=64 \
       FA_DK=64 FA_DV=64 FA_NUM_HEADS=12 FA_HEADS_PER_UNROLL=2 FA_CASCADE_STAGES=4
python3 open_kernels/build_design.py open_kernels/designs/whisper_fa/attn_fa.py \
       open_kernels/designs/whisper_fa/build_laya
```

Running `attn_fa.py` directly also works and takes the same values as
`--lq/--lk/...` flags, but it writes `fa_iron.xclbin` into the current directory
and does not use `SPECIALIZE`; `build_design.py` is the supported path.

### Integration: an end-to-end Laya decision now runs on the NPU

Done 2026-10-02, which unblocks everything below.

The model is **not** installed with `oflm-add` (that needs `model.q4nx`, the Q4NX
container for decoder LLMs; Laya uses `.npue`) and **not** with `q4nx-build`
(GGUF->Q4NX, decoder-only configs, no encoder architecture). The working path
needs no new code:

1. Download the five files the registry entry lists into
   `~/.config/oflm/models/laya/`, keeping the nested layout
   (`multilingual/{encoder/config.json, model.safetensors, tokenizer/...,
   rl_agent_config.json}`).
2. `NpueDecision` packs the container on first load via
   `npue::prepare_model_auto` -- it finds `model.safetensors` and writes
   `laya.npue`. The `npue_checkpoint_subdir` / `npue_config_subdir` /
   `npue_tokenizer_subdir` keys exist *because of Laya*: config, weights and
   tokenizer are at three different depths and none can be guessed.
3. Link the family design set the entry names:
   `ln -s <repo>/src/xclbins/BERT-h768-gated-i1152-bf16/gemm_rtp \
        ~/.config/oflm/models/laya/npue_designs/gemm_rtp`

Packed and ran:

```
[NPUE]  packing .../multilingual/ -> .../laya.npue (arch=modernbert_rope_geglu)
  hidden=768 heads=12 head_dim=64 layers=22 inter=1152 ffn_up=2304
  window=64 (local_attention/2, NOT +1)  8 full / 14 sliding
  tensors 307, data 1080.61 MB
  designs ONE xclbin, 12 streams (3 batch tiers), one hw_context
  tiers 4, 16, 32   (requests are right-sized, not padded)
  gelu / softmax on the HOST (fp32) -- 22 fewer NPU dispatches
  layernorm on the HOST (fp32) -- 45 fewer NPU dispatches
  weights 220.59 MB staged on the device once

$ oflm decide laya-decision:multilingual -i req.json
  supplier_choice  choice=contoso  confidence=0.517991
  is_urgent        noul=false      p(true)=0.489608
```

**Engagement: column utilisation peaked at 21%** during the request, 18 of 40
samples non-zero. The encoder really executed on the AIE.

21% rather than the 98% a saturated kernel reaches is the *expected* number and
it is the point of this whole plan: the NPU is only carrying the 88 projection
GEMMs while qk, softmax, av, GELU and 45 LayerNorms run on the host. Moving
LayerNorm and fused attention onto the NPU is what moves this number, and it is
now measurable rather than inferred.

Corrections worth recording: Laya's `model_type` **is `modernbert`**, so it
packs as `arch=modernbert_rope_geglu`. I had assumed arch 4 was a different
model because its comment quotes `ffn_up 3072`; at `intermediate=1152` the
packed geometry is `ffn_up=2304` and this is Laya's own path. Also `oflm decide`
takes `-i <file>`, not `--input-file` as its own header comment claimed, and
`questions` is an object keyed by question key, not an array -- for a choice the
`criteria` key order *is* the answer space.

---

## 3b. BLOCKED: how LayerNorm actually reaches the NPU

Investigated 2026-10-02. The answer is not "set `host_ln=false`", and the reason
matters, because the plan above assumed it was.

**`unified` is decided by one file, and it forces the eltwise ops to the host:**

```cpp
// npue_encoder.hpp Stack::Stack
const bool unified = std::ifstream(art + "/gemm_rtp/design.json").good();
...
if (unified) { host_ln = host_sm = host_gelu = true; }   // forced
```

The non-unified branch loads **seven** design directories:

```cpp
ld_qkv = ... art + "/qkv";        ld_ao = ... art + "/attn_out";
ld_fu  = ... art + "/ffn_up";     ld_fd = ... art + "/ffn_down";
ld_gelu= ... art + "/gelu";       ld_ln = ... art + "/layernorm";
ld_sm  = ... art + "/softmax";
printf("  designs    7 resident xclbins\n");
```

So neither existing path gets LayerNorm onto the NPU as things stand:

- **Unified** (what ships): one xclbin, 12 streams, and the eltwise ops are
  *forced* onto the host regardless of `StackOptions::host_ln`.
- **Non-unified**: needs four *separate* GEMM design directories, and the repo
  ships none — `open_kernels/designs/` has no `qkv`, `attn_out`, `ffn_up` or
  `ffn_down`. Those kernels exist only inside the unified `gemm_rtp` set, and
  `build-design-sets.py` calls `export_gemm_rtp.py` alone with no mechanism for
  extra designs.

Two ways forward, and the choice is architectural rather than mechanical:

### The catch that changes (a)'s size: the LN kernel interface does not match

Before doing (a), the *existing* NPU LayerNorm dispatch path was read, and it
does not want the kernel that was built and validated in step 1.

```cpp
// layer_norm(), the !host_ln branch
bf16_fill((uint16_t *)layernorm.host_ptr(0) + lo, x.data() + lo, ...);  // bf16 in
layernorm.sync_to_device(0);
layernorm.dispatch_only();
layernorm.sync_from_device(2);
bf16_read(x.data() + lo, (const uint16_t *)layernorm.host_ptr(2) + lo, ...); // bf16 out
```

So the engine expects a **3-buffer bf16-in / bf16-out** LayerNorm — activation in
`host_ptr(0)`, weights staged from the container, activation out of
`host_ptr(2)`. It does the residual add separately (`add_plain`).

`designs/ln`, which step 1 built and validated, is a different interface:

```
in  = [x fp32[N], add fp32[N], w bf16[N]]
out = [y fp32[N], xn bf16[N]]          # designs/ln/ln.py, docstring
```

Five buffers, fp32 in and fp32 out, with the add **fused in**. Dropping it into
`art + "/layernorm"` unchanged would bind the wrong buffers and produce
correct-looking nonsense — which is the trap 7c failure mode this project has
already paid for five times.

So (a) splits again, and this is the part that actually costs:

- **(a1) Give `designs/ln` a no-add, bf16-in/bf16-out mode.** The fused `add` is
  an optimisation over the engine's current `add_plain` + LN pair; dropping it
  gives a 3-buffer kernel that matches `layer_norm()` as written. The kernel
  already has the reduction and the weight multiply, so this is a port of the
  existing data path, not new mathematics — and it reuses the fp64 oracle and
  the corruption test that step 1 established.
- **(a2) Write a fresh bf16 LayerNorm kernel.** More control, more risk, and it
  discards a validated design.

(a1) is the smaller change and keeps the oracle. Either way the *engine* half of
(a) is still the two lines described below.

Note the same question will have to be answered for fused attention before that
step is real work: the engine has **no `fa` Design at all**, so unlike LayerNorm
there is nothing to bind.

**(a) Let unified mode bind an eltwise design when one is present.** Keep
`gemm_rtp/` as-is and additionally emit `layernorm/` beside it; change the
forcing so it becomes `if (unified && !eltwise_design_present)`. This is
**fail-safe**: a model without the directory keeps today's behaviour exactly,
because the condition is the absence of a file the build controls. The family
build gains one extra design, and the engine gains a presence check. It also
composes with fused attention later, since FA would ride the same mechanism.

**(b) Build the full seven-design set.** Emit four standalone GEMM designs
alongside `layernorm`, `gelu` and `softmax`, and switch this model to
non-unified. No engine change at all, but four new GEMM design builds per family
and a second code path to keep correct — and the 7-xclbin switch cost is a real
regression against unified's "zero switches".

(a) is the smaller change and keeps the single-xclbin benefit. It is proposed,
not done, because it alters dispatch semantics for every model that uses unified
mode, which is not a call to make silently.

Note also that the non-unified list has **no `fa` entry** either. Fused attention
needs a new `Design` member plus a dispatch branch in `qk()`/`av()` either way —
the placement work above does not come with it.

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
