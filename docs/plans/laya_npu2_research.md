# Deep Research: Running the Laya model on AMD NPU2 via OpenFlowLM-Next

**Status:** research complete 2026-09-29. Nothing implemented.
**Scope:** what it would take to run `convaiinnovations/laya` on XDNA2/NPU2
through this repo. The companion implementation plan is
`specs/open-engine/plans/laya-decision-encoder.md`.

---

## TL;DR

- **Laya is not an LLM.** It is a *non-autoregressive, bidirectional encoder* —
  a ModernBERT backbone plus a 2-layer decision head that scores `[MASK]`
  marker positions into `choice`/`score`/`noul` answers in one forward pass,
  ~33 ms on a T4.[^1][^2] That makes it structurally a **sibling of the seven
  embedding models already running in `src/open_npue/`**, not of the causal-LM
  path — and this repo already has a *fully-specified, zero-implemented*
  ModernBERT architecture hook (`arch=4`).
- **The repo is closer than any prior effort suggests.**
  `npu_offload/gemm_rtp/npue.py:106-141` documents arch=4 in complete
  architectural detail (pre-LN, final norm, bias-free, no position table, two
  RoPE thetas, ±64 sliding window, `gate|up` GeGLU, byte-level BPE). The
  **byte-level BPE tokenizer is already written, generated, compiled, and linked
  into `oflm`** — and is called from nowhere.[^9]
- **The GEMM geometry is solved for the *multilingual* checkpoint and
  *near*-solved for the English one.** Laya-multilingual (mmBERT-base,
  h768 / inter 1152 / gated) is an ordinary new design family at the shipping
  `tile_n=48`. Laya-large (h1024 / inter 2624 / gated) is forced to
  `tile_n=16` — uniquely — because `ffn_up` N=5248 = 2⁷·41, which nothing
  else divides.
- **The decision head is literally a BERT layer** (`in_proj`=qkv,
  `out_proj`=attn_out, `linear1`=ffn_up, `linear2`=ffn_down,
  `dim_feedforward=4d`). It needs **no new kernel family at all** —
  `BERT-h1024-bfp16` and `BERT-h768-bfp16` already exist and already fit it.
- **The real blockers are not the NPU.** They are (a) a missing pre-LN +
  final-norm + per-layer-RoPE + banded-mask code path in the host runtime,
  (b) `open_npue` has **no API for anything but a pooled vector** — Laya needs
  `[rows, options]` logits, so a new class + endpoint is required, and
  (c) sequence length: the design family is fixed at `seq=64` and the repo
  *explicitly refuses to predict* throughput above it.[^10]
- **The llama.cpp PR is not a shortcut.** ggml-org/llama.cpp#29363 is **still
  open**, CPU-only, and is architecturally dead-ended for this repo: it
  registers `LLM_ARCH_LAYA` *solely* so `llama-quantize` can load a GGUF, and
  puts the actual graph in a standalone `tools/laya` with its own CPU
  backend.[^6]

---

## Scope and framing

- **Topic:** Adding Laya model support to run on AMD NPU through OpenFlowLM-Next
- **Scope:** narrow drill-down
- **Audience:** OpenFlowLM-Next maintainers
- **Recency cutoff:** all-time (PR state verified 2026-09-29)
- **Sub-goals investigated:**
  - Laya's exact PyTorch architecture, op set, and tensor shapes
  - llama.cpp PR 29363 scope/status/backends
  - OpenFlowLM-Next's existing encoder (open_npue) path and its add-a-model
    procedure
  - AMD NPU2 / XRT / AIE2P hardware + toolchain constraints
  - ModernBERT architecture and mmBERT variant specifics
  - NPU2 GEMM tile/geometry legality for Laya's shapes
  - Attention feasibility (host vs NPU, bidirectional, sliding-window)
  - The required new API/server surface for a logit-returning model
  - q4nx-build / packaging / oflm-add integration path

---

## Key findings

1. **Laya = ModernBERT encoder + typed decision head, and the head is a plain
   2-layer BERT.** The `DecisionModel` wraps `AutoModel` (which resolves to
   `ModernBertModel` for all three checkpoints) and adds a 2-layer
   `nn.TransformerEncoderLayer` head with `dim_feedforward=4d`, then gathers
   `[MASK]` marker positions and scores them with a
   `LayerNorm→Linear(d,d)→GELU→Linear(d,1)` MLP.[^1] The FFN activation is
   torch-default **ReLU**, while the encoder uses exact-erf **GELU** inside a
   **GeGLU** (gate is the *second* half of `Wi`).

2. **The single most valuable artifact in the repo for this task is a
   docstring.** `npu_offload/gemm_rtp/npue.py:106-141` specifies arch=4
   completely, written by someone who *ablated* each claim ("a WRONG reading
   for each of thirteen architectural claims, 3.5e+04x to 2.7e+07x worse"). It
   also names the two traps most likely to be got wrong: layer 0's `attn_norm`
   is `nn.Identity()` (11 norms for 12 layers — a weight-1 norm is *not* the
   same thing), and the 128 window is halved by the mask to ±64.[^9][^10]

3. **`design_fits()` structurally forbids sharing one design set between
   encoder and head.** Its loop returns false on *any* mismatching occurrence
   of an op name, and the `Want` table is derived from a single
   `(hidden, intermediate, gated_ffn)` triple.[^11] But that constraint is
   moot: the head's geometry (hidden=d, intermediate=4d, ungated) already
   matches two shipping families exactly, so the head needs **zero new
   xclbins**.

4. **Attention is not the near-term bottleneck; sequence length is.** For
   ModernBERT-large the corrected GEMM:attention FLOP ratio is 12:1 at S=512 and
   6:1 at S=1024, crossing at S≈6144 — and with the 10-global/18-banded split,
   crossing at S≈3400.[^12] At S=1024, host attention with a banded mask is
   ~1.8× the NPU GEMM time. The repo's own comment says the Laya-shaped path
   has **never been measured above seq 64** and prints a warning on
   `--seq != 64`.[^10]

5. **A hardware-verified bidirectional, non-causal, online-softmax
   FlashAttention kernel already exists in the tree** —
   `open_kernels/designs/whisper_fa/`, built for Whisper's 20 heads / 1536
   seq, using 8 KB of score tile per core and never materializing S×S. It is
   compile-time-shaped (no runtime seq) and its `apply_window_mask` is present
   but **causal and unwired** (zero call sites repo-wide).[^13]

---

## Evidence

### Sub-goal 1: What exactly is Laya?

Three checkpoints, all `model_type: modernbert`:

| checkpoint | backbone | params | max_len | head_max_len |
|---|---|---|---|---|
| `laya` | ModernBERT-large (h1024, 28L, 16H, inter 2624, vocab 50368) | 421 M | 512 | 192 |
| `laya-multilingual` | mmBERT-base (h768, 22L, 12H, inter 1152, vocab 256000) | 322 M | 1024 | 256 |
| `laya-typed-decisions` | ModernBERT-large | 421 M | 1024 | 256 |

Forward pass, end to end[^1][^3]:

```
input_ids [N,L]  + attention_mask + marker_pos [N,K] + marker_mask + qtype [N]
  → ModernBertModel (bidirectional, no KV cache)            h [N,L,d]
  → h += type_emb[qtype]                                     (broadcast over L)
  → 2 × TransformerEncoderLayer(d, nhead=d//64, ff=4d,
                                norm_first=True, activation=ReLU, WITH biases)
  → gather at marker_pos                                     m [N,K,d]
  → scorer: LN(d) → Linear(d,d) → gelu → Linear(d,1)        logits [N,K]
  → masked_fill(~marker_mask, -1e4)
  → softmax(logits) / entropy / top2 / k → feats [N,4]
  → act_head: Linear(d+4,256) → GELU → Linear(256,2)        act_logits [N,2]
```

Three findings that matter for a port:

- **Pooling is a `gather` at `[MASK]` positions, not CLS or mean.** The `act`
  head pools CLS (`h[:,0]`). Temperature calibration is applied *outside* the
  graph, in numpy, from `rl_agent_config.json` (a per-qtype 3-vector, plus
  optional `temperature_by_options` buckets) — so it costs nothing on the NPU
  and belongs in the host wrapper.
- **`temperature` is a registered buffer that `forward` never reads** — the
  calibration table lives entirely in `rl_agent_config.json`.
- **The stock path pins `attn_implementation="sdpa"`** and never uses flash-attn
  or flex-attention. There is no vendor-specific primitive anywhere in the op
  set. That is the good news: there is nothing exotic to re-target.

The only GPU-specific code is `laya/fast.py` + `laya/tl_kernels.py` (TileLang,
CUDA-only): five fused kernels — `gemm`, `gemm_geglu`, `add_ln`, `rope`, `attn`
(flash-attention with an online `exp2` softmax and an optional
`|i-j| <= window` band).[^1] It is an *optimization*, not a requirement, and it
is NVIDIA-targeted. Notably it keeps an **fp32 residual stream** because
"ModernBERT-large's residual activations reach ~3e4, where bf16's 8-bit
mantissa would lose ~100 units per add and drift layer by layer."

**INT8 path exists but is CPU-only ONNX:**
`scripts/export_onnx.py --quantize` does
`quantize_dynamic(op_types_to_quantize=["MatMul"], weight_type=QInt8,
per_channel=True)` — weights only, activations stay fp32, **embedding table
untouched** (it is a `Gather`, not a `MatMul`). The comment records why
per-channel matters: per-tensor flipped 3/20 decisions, per-channel flipped
none.[^1]

### Sub-goal 2: The llama.cpp PR — status and scope

PR ggml-org/llama.cpp#29363, *"laya: Add support for the laya multilingual
decision model"*, author `Clauszy`: **state `open`, `merged: false`**, 4
commits, +4,768/−0 across 32 files, reviewers `ggerganov` and `CISC` both still
"Awaiting requested review", 0 inline review comments.[^6]

The critical structural fact:

> `src/llama-model.cpp` — the entire change is 5 lines:
> `case LLM_ARCH_LAYA: return new llama_model_modern_bert(params);` with the
> comment *"so that llama-quantize can instantiate the arch for M2 quantization
> (**full inference graph lands in M3**)"*.

So the arch registration exists **only** to make quantization work. There is no
`src/models/laya.cpp`, no graph builder, no `build_*`. Real inference lives in a
standalone `tools/laya/` (1,181-line `laya.cpp`, 742-line CLI) that hard-codes
`GGML_BACKEND_DEVICE_TYPE_CPU` with a single-backend
`ggml_backend_sched`.[^6]

**Backend support: CPU only.** Zero backend files in the diff. **The conversion
path is sound** — `conversion/laya.py` registers a hparams loader keyed on
`rl_agent_config.json` existing (Laya ships no root `config.json`) and
subclasses `ModernBertModel`; tensor mapping is arch-scoped for the head
tensors.

Ops used are all stock ggml and are a good cross-check on the architecture:
`ggml_mul_mat` ×14, `ggml_rope_ext` ×3 (NEOX), `ggml_geglu_erf` ×2,
`ggml_gelu_erf` ×2, `ggml_relu` ×1 (the head), `ggml_norm` ×1 (used for both
LayerNorm and RMSNorm with an optional bias add), `ggml_argsort_top_k`,
`ggml_soft_max_ext`.[^6]

Two documented bugs worth inheriting as guardrails:

- **RoPE `Wo` was loaded but never applied** in the encoder graph — "which
  produced O(1) wrong scorer logits and wrong choices."[^6]
- **The encoder is attention-saturated**: pre-softmax scores reach ~55, so
  "tiny summation-order differences are amplified."[^6]

And a quantization finding nobody has resolved: a reviewer measured that with
the PR's protection recipe, **Q8_0 still flips ~90/2000 decisions**, traced to an
*attention-sink gate* (massive activations ≈ 2.7e3 / 4.3e3 in two residual dims,
switched by a single FFN neuron in layers 11 and 18) amplifying a uniform 0.5 %
weight rounding.[^6]

**Verdict: not a shortcut.** Even merged, it is CPU-only and out-of-core. It
is, however, an excellent **independent reference implementation** for the
tokenizer, the marker-gather protocol, and the head graph — and it documents
exactly which bugs a from-scratch port will hit.

**Downstream consumer:** `Mesh-LLM/mesh-llm#2083` (merged) ports it as
`model_support/0006` and states the reason it exists: *"it is not a causal
model, and llama.cpp support exists only as the open draft."* It adds Skippy
feature bit 40 `laya_decisions` and a `decision` workload class = proto value 6
— a capability-dispatch concept **this repo has no counterpart for**.[^6]

### Sub-goal 3: ModernBERT architecture — the traps

Verified against the safetensors header directly (both checkpoints are
single-file; **there is no `model.safetensors.index.json` on the Hub**), with a
parameter-count proof that independently confirms the absence of qk-norm and
position embeddings.[^4][^5]

Per-layer tensors (large, 28 layers):

| tensor | shape | bias? |
|---|---|---|
| `attn.Wqkv` | [3072, 1024] | no |
| `attn.Wo` | [1024, 1024] | no |
| `mlp.Wi` | [5248, 1024] | no |
| `mlp.Wo` | [1024, 2624] | no |
| `attn_norm` | [1024] | **layers 1..27 only** |
| `mlp_norm` | [1024] | all 28 |

Plus `embeddings.tok_embeddings` [50368,1024], `embeddings.norm`,
`final_norm`.

**Twelve ranked gotchas** (corrected against primary source):

1. **GeGLU gate is the SECOND half** of `Wi` — `chunk(2,-1) → input, gate`.
   LLaMA convention is the reverse. Getting it backwards still produces
   plausible garbage.
2. **The window is ±64 (a 129-token band), not 128.** `sliding_window` is a
   *property* = `local_attention // 2`. The `+1` in
   `self.sliding_window = config.sliding_window + 1` is a FlashAttention
   inclusive-boundary convention only; dense masks use 64.
3. **Layer 0's `attn_norm` is `nn.Identity()`** — 27 norm tensors, not 28.
4. **Exact erf GELU**, never tanh.
5. **QKV packed layout is `(3, Nh, Dh)`** — the 3 is the *outer* stride.
6. **RoPE is NeoX half-split**, full 64-dim, on Q and K only, **never V**.
7. **Softmax in fp32** regardless of activation dtype.
8. **Right-pad only** — `position_ids = arange(S)`, not a mask-cumsum.
   Left-padding silently breaks positions.
9. `layer_types` is **derived, not stored**: `"full_attention" if i % 3 == 0 else
   "sliding_attention"`. large → 10 global / 18 sliding; base/mmBERT → 8 / 14.
10. **Two RoPE bases for ModernBERT-large** (global 160000, local 10000) — but
    **mmBERT-base uses 160000 for BOTH**.[^14] Do not hardcode the split.
11. **Unpadding is an optimization, not a correctness requirement** — the padded
    path produces identical logits under `eager`/`sdpa`.
12. **RoPE replaces absolute position embeddings; it is not in addition to
    them.** `position_embedding_type: "absolute"` in ModernBERT's config is a
    **dead key**, read by no code path. The parameter count proves an
    8192×1024 table is absent.

The paper is **arXiv 2412.13663**, not 2406.00264.[^4] mmBERT is JHU-CLSP
(arXiv 2509.06888), 256k Gemma-2 vocab, `position_embedding_type: "sans_pos"` —
which is why its embedding table is ~1.7× its entire 22-layer stack.[^14]

### Sub-goal 4: The open_npue encoder path — what exists

`src/open_npue/` is a synced copy of NpuEmbeddings, MIT, toolchain mlir-aie
1.4.2 / Peano 21.0.0. It is a **complete, gated, production** bidirectional
BERT-family encoder: 4 fused NPU GEMMs per layer over one resident xclbin, host
AVX2/AVX512 for LayerNorm/GELU/attention, CLS-or-mean pooling, per-row int8
activation quant with SmoothQuant, per-channel weight scales, batch tiers, and
a layout-hash guard that throws *"The bytes would be the right size and the
wrong order."*[^9][^10]

Four architectures are implemented: `bert_abs_gelu_postln` (0),
`gemma3_mqa_rope_geglu` (1), `nomic_bert_rope_swiglu` (2),
`gte_new_rope_geglu` (3). **arch=4 is documented in Python and absent from C++
entirely** — `grep -rni modernbert` over the tree returns one docstring, one
constant, and one tokenizer comment.[^9]

Feature matrix against ModernBERT's needs:

| feature | status | gap |
|---|---|---|
| bidirectional attention | ✅ host, today | — |
| fused QKV / O / FFN GEMMs | ✅ four per layer | — |
| GeGLU, both half-orders | ✅ `GatedAct{GeluErf}` | third order `gate\|up` needs a third key |
| LayerNorm (biased) | ✅ | — |
| **Pre-LN** | ❌ | `add_norm_*` fuses residual-then-norm and stores the *normalised* value as the residual — **wrong, not merely slow**, under pre-LN |
| **Final norm** | ❌ | no `Encoder::run()` equivalent; `GemmaNpuEncoder:3887` is the one-liner to copy |
| **Per-layer RoPE theta** | ❌ | one table; arch=1 Gemma already does two-table selection — copy that |
| **Sliding-window band mask** | ❌ | `add_mask` is padding-only; no locality term exists anywhere |
| learned position embeddings | ✅ read and summed | but ModernBERT needs **zeros** — `embeddings.position` and `token_type` must be emitted as zeros (the runtime dereferences both unconditionally) |
| byte-level BPE tokenizer | ✅ **written + compiled** | **zero call sites**; sibling Gemma/XLM-R tokenizers *are* wired, proving this is accidental |
| int8 per-channel | ✅ fully | no family uses it |

### Sub-goal 5: GEMM geometry — the decisive computation

The design asserts `N % (tile_n * n_aie_cols) == 0` (`gemm_pretiled.py:157`,
`n_aie_cols` defaults to 8 from `--cols`) and `M % (m * 4) == 0` (line 155,
`m=64` → M multiple of 256). L1 is `2mk + b_l1_depth·kn + 2mn` bytes, rejected at
`>= 65536` — but **that check lives only in the test harness `run_one()`, not
the export path**.[^15]

Independently recomputed and confirmed:

**Laya-large / ModernBERT-large encoder** (qkv 3072, attn_out 1024, ffn_up
**5248**, ffn_down 1024; K ∈ {1024, 2624}):

| tile_n | divisor | N-div | m×n | `narrow_f32_bf16` | l1 | legal |
|---:|---:|---|---:|---|---:|---|
| 8 | 64 | ✅ | 512 | ❌ | 22,528 | no |
| **16** | 128 | ✅ | 1024 | ✅ | 28,672 | **YES — unique** |
| 32 | 256 | ❌ (5248) | 2048 | ✅ | 40,960 | no |
| 48 | 384 | ❌ (1024, 5248) | 3072 | ✅ | 53,248 | no |
| 64 | 512 | ❌ | 4096 | ❌ | 65,536 | no |

`5248 = 2⁷ · 41`. The prime factor 41 is what forces `tile_n=16`. (Caveat: this
uniqueness is a property of `--cols 8`; at `--cols 4` the divisors halve and 32
becomes legal. Also: `tile_n=48` could be recovered by padding `ffn_up`
5248→5376 with zero columns — exact, not approximate, and the repo already uses
that trick for Gemma MQA — but that needs a pad-aware epilogue and a second
source of truth.)

**Laya-multilingual / mmBERT-base encoder** (all N ∈ {2304, 768}): tile_n ∈
{16, 32, 48} all legal, `l1(48)=53,248` identical to every shipping family.
**This is an ordinary new design family at the shipping width 48.**

**The decision head needs no new xclbin.**
`nn.TransformerEncoderLayer(d, d//64, 4d)` is arithmetically *a BERT layer*:

| | head GEMMs (K,N) | `design_fits(h, 4d, gated=false, qkv_n=3d)` |
|---|---|---|
| large head (d=1024, ff=4096) | (1024,3072) (1024,1024) (1024,4096) (4096,1024) | **`BERT-h1024-bfp16` → true** |
| multilingual head (d=768, ff=3072) | (768,2304) (768,768) (768,3072) (3072,768) | **`BERT-h768-bfp16` → true** |

The head's *tail* (`scorer` N=1, `act_head` N=2, and `act_head.0` with **K =
d+4**, which fails `K % 64 == 0`) cannot go on the array under any tile.
Host-side is the only option — the same shape as Gemma's post-pool
`2_Dense`/`3_Dense` heads, which are already host GEMMs.

**Net design inventory: two new families, two existing ones reused.**

| what | family | flags | status |
|---|---|---|---|
| laya-large **encoder** | `BERT-h1024-gated-i2624-bfp16` (new) | `--hidden 1024 --intermediate 2624 --qkv-n 3072 --gated-ffn --emulate-bfp16 --c-bf16 -n 16` | NEW |
| laya-multilingual **encoder** | `BERT-h768-gated-i1152-bfp16` (new) | `--hidden 768 --intermediate 1152 --qkv-n 2304 --gated-ffn --emulate-bfp16 --c-bf16 -n 48` | NEW |
| large **head** | `BERT-h1024-bfp16` | — | EXISTS |
| multilingual **head** | `BERT-h768-bfp16` | — | EXISTS |

A build of one family is ~3–4 min (`4 shapes × 4 tiers = 16 JIT designs`); all
batch tiers share one xclbin (verified by `xclbin_identical_mod_uuid`, which now
bounds differing runs structurally rather than masking UUIDs).[^15]

### Sub-goal 6: Attention feasibility

**The "too many small ops" worry is quantitatively wrong.** Measured fixed XRT
dispatch overhead is **~86 µs** (linear fit, cross-checked against the repo's
own ~150 µs figure). ModernBERT-large's 28 layers × 4 GEMMs = 112 dispatches ≈
**13 ms** against ~1.6 s of compute — about 1%.[^16]

**Score-matrix memory is the real limit, and the kernel doesn't need it.** At
S=1024/H=16 the S×S fp32 matrix is 64 MiB against **4 MB of total on-array L2**
(8 mem tiles × 512 KB) — 16× over. At S=8192 it is 4 GiB. The Whisper split
design was rejected in writing for exactly this ("9.94 GB per window"), and the
fix — the chained flash-attention form — is what `whisper_fa` then
built.[^13][^17]

`whisper_fa` is a genuine online-softmax FlashAttention: per core it holds a
**constant 8 KB score tile** regardless of sequence length, loops K/V in 64-wide
chunks, and cascades 4 stages. Per-core L1 is 57,344 of 64,512 bytes — 89 % — and
none of it scales with S. It is **bidirectional** (Whisper's own config is
non-causal, so all the causal machinery was dropped as dead code) and its only
mask is a per-column *length* mask, which is exactly ModernBERT's padding
requirement.[^13]

Transferability to ModernBERT:

| requirement | status |
|---|---|
| bidirectional | exact |
| head_dim 64, fp32 softmax state | exact |
| padding mask | exact (`apply_length_mask` ≡ per-row valid-length) |
| 16 heads | parameter (`H*NQ ≤ 8` forces H=2, so 8 head-groups vs Whisper's 10 — a loop, not spatial) |
| 18 local layers, ±64 band | `apply_window_mask` **exists but is causal and unwired (0 call sites)**; making it two-sided is the one new kernel-side edit |
| S = 512/1024/8192 | **compile-time only** — `lq`/`lk` are `CompileTime` and the L3 tensor *shapes* derive from them. One xclbin per S. |

**Host attention, corrected arithmetic.** The existing `open_npue` host path
materializes `scores[batch·heads·seq·seq]` and is O(S²) in time *and* memory.
For ModernBERT-large per layer (Σ K·N = 12,582,912):

| S | attention (all-full) | GEMMs | ratio |
|---:|---:|---:|---:|
| 512 | 1.07 GF | 12.9 GF | 12.0 : 1 |
| 1024 | 4.30 GF | 25.8 GF | 6.0 : 1 |
| 8192 | 274.9 GF | 206.2 GF | **0.75 : 1** |

With the 10/18 band split (129/1024 = 12.6 % of the work on local layers), the
S=8192 ratio improves to **2.04 : 1**. The band's saving is *real work
reduction*, not just mask arithmetic, provided `j` is range-clamped in
`qk_impl`/`av_impl` so the MACs are skipped too.

**Ranked path: (a) host attention to S=1024 → (d) add the band → (b) generalize
`whisper_fa` at one S.** Option (b) is genuinely close — one mask edit, one
routing edit, everything else is a parameter — but it is *one S per xclbin*, and
Laya is a sequence-length-variable workload by construction. Whisper gets away
with one S because a 30-second window is 1500 frames by definition; Laya's whole
reason for existing is 512 → 8192.

### Sub-goal 7: The missing API surface

This is the largest single gap and it is **not** an NPU problem.

`AutoEmbeddingModel`'s entire contract is `load_model`, **`embed(text,
task_type) -> vector<float>`** (pure virtual), `embed_batch`, `embedding_dim`,
`prompt_names`, `supports_task_prompts`.[^18] There is no `classify`, no
`score`, no `num_labels`, no output-mode. `EmbedService::embed` pre-sizes its
output to `texts.size() * g_hidden`; `chunk()` builds the full
`[take, seq, hidden]` tensor, hands it straight to `pool_rows`, and the
intermediate dies there. `pool_rows` accepts only `"cls"` or `"mean"` and
**throws** on anything else.

Repo-wide, there is **no classification, token-classification, or
logit-returning surface**. The only real logits are internal:
`open_gemma3::prefill` returns last-position LM logits and is used by exactly
one test; `rest_handler.cpp` stubs `logprobs` to `null` in every OpenAI
response. `task_classification` in the enum is a *text prefix selector*, not an
output.

Two consequences:

- **`/v1/embeddings` is structurally incapable of carrying scores.** `data[i]`
  has exactly three keys and `embedding` is sliced at `dim` floats by
  `embedding_batch_dim`, which *throws* if the flat size doesn't equal
  `n_inputs × expected_dim`. A `[n_inputs, n_options]` return would fail that
  check — and passing it by faking `embedding_dim() == n_options` makes the
  result indistinguishable from an embedding everywhere downstream.
- **The `ShapeLease` enforces one geometry per process.** `g_hidden`/`g_seq`/
  `g_layers`/… are file-scope globals; a second model in the process would
  "silently reinterpret the first one's weights and return plausible embeddings
  for neither." Laya cannot be co-resident with a chat model **unless** its
  geometry happens to match, or unless it gets a dedicated process.

`oflm run` is hard-wired to `AutoModel` and already explicitly refuses
embeddings. But **`oflm bench-embed` is the right precedent** for a decision
surface: a one-shot, non-interactive, non-server command over the same
interface, needing only a new `.hpp` + unit test + one `main.cpp` branch + one
help line. That is dramatically cheaper than the server path, and it should be
first.

### Sub-goal 8: q4nx-build / packaging — which route

There are **two non-interoperating embedding pipelines**, and the one people
assume is wrong for this model.

| | `open_embedding` (`--open-embedding`) | `open_npue` (`.npue`) |
|---|---|---|
| container | safetensors + `weights_manifest.json` | pre-tiled `.npue` |
| architectures | **Gemma3 only**, hardcoded | arch 0/1/2/3 |
| kernels | any shape, `npu_matmul_f32/m{M}_{K}x{N}` | `BERT-*` design sets, geometry-locked |
| install | `oflm pull` | `oflm pull` |

`q4nx-build --open-embedding` **hard-fails** on any model without
`2_Dense`/`3_Dense` heads, and even if it didn't,
`open_embedding::engine.cpp` reads only Gemma3 config keys and hard-fails when
`layer_types.size() != num_hidden_layers`. Laya has neither shape, so that route
is out.

The `.npue` route has a **silent fail-open** worth flagging loudly:
`prepare_model_auto` dispatches on `model_type`, with the BERT packer as the
*deliberate* last branch. `model_type: "modernbert"` therefore falls into it and
produces a **valid arch-0 container** — GELU + absolute position embeddings, for
a GeGLU + RoPE model. The runtime will load it and return wrong vectors. The
code's own comment says this is intentional ("this is where a genuinely new
architecture will first show up as a wrong answer rather than an error") and the
arch field is the guard — but the guard only fires if a *packer* writes the right
arch, which today none does. **A `model_type == "modernbert"` branch must be
added before anything else, and it should throw until the packer exists.**

Also blocking both routes: Laya ships **no `config.json` at repo root** (it is
under `encoder/`), **no `1_Pooling/config.json`** (which `resolve_pooling`
requires by name), **no `vocab.txt`**, and its tokenizer is either GPT-2-style
byte-level BPE (English) or **Metaspace `prepend_scheme: "always"`**
byte-fallback BPE (multilingual) — neither is WordPiece, and the
`tokenizer.vocab` U8 blob route does not apply without conversion. The
multilingual tokenizer's `prepend_scheme: "always"` is a specific trap: it
inserts a space prefix a naive port will miss.

`oflm-add` **cannot install an embedding or npue model at all** —
`REQUIRED_FILES` includes `model.q4nx` and the tool is Q4NX-only. Embedding
models ship via `oflm pull` keyed on `model_list.json` + `model_info.json`, with
kernels resolved by the `npue_design_family` **name** (a GEMM geometry), and the
entry point is `all_embedding_model.hpp`'s flat tag→enum map, where an unknown
tag **throws** by design ("Refusing to substitute one: an embedding for the
wrong model is correctly shaped and correctly normed, so nothing downstream can
tell that the answer is wrong").[^19]

---

## Counter-evidence and dissent

- **The `apply_window_mask` reuse case is weaker than it looks.** The function
  is inherited from MLIR-AIR and is not in the vendored-modification list at
  `attn_npu2.cc:12-25`; the FA port explicitly dropped all causal/window
  machinery as dead code. It has **never been hardware-tested on this
  hardware**. Making it two-sided means a *new* mask, not a re-enable — and the
  design already runs at 89 % of L1.
- **The `BERT-h1024-gated` route at `tile_n=16` is slower than planned.** An
  8-column, 16-wide N tile is a materially worse MMA shape than 48; the repo's
  own measurement at `attn_block` was 2.2 TFLOPS at K=256 vs 6.3 at K=1024. The
  alternative (pad 5248→5376 to recover 48) is exact but needs a pad-aware
  epilogue and breaks `design_fits` into a second source of truth. This is a
  real trade-off, not a settled answer.
- **The `open_embedding` route's existence is counter-evidence to "one embedding
  path."** Two pipelines, neither reusable for Laya as-is, is a maintenance
  liability the repo already carries. Adding a third (`decisions`) compounds it.
- **The bf16/bfp16 datapath is a live accuracy risk specific to this model.**
  Five of six shipping encoders use `--emulate-bfp16`; `bge-small` **failed the
  MTEB gate at −0.5010 bit-reproducibly** and had to be rebuilt on plain bf16.
  Given the llama.cpp reviewer's attention-sink finding (a single FFN neuron in
  layers 11 and 18 gating massive activations), the `--emulate-bfp16` choice
  for Laya is **not** a free default and must be decided by ablation against the
  bf16 replica, with a real accuracy gate — not inherited.
- **Upstream Laya's own position is that this port is unusual.** Their hardware
  story is entirely PyTorch + ONNX; a word-boundary grep of their README for
  `gguf|ggml|llama|cann|npu|metal|vulkan|rocm` returns zero matches for the
  first five. The one NPU reference anywhere in the project is a third-party
  Huawei Ascend port, claimed 34–71× at batch 1.[^2] Nobody upstream is
  producing a NPU2 or GGUF build; third-party GGUF repos on the Hub predate the
  PR and cannot have come from it.[^6]
- **The whisper spec is stale in a way that will mislead a reader.**
  `specs/open-whisper/spec.md:242-301` still argues *against* a chained
  attention design because the FA kernel did not exist when it was written.
  Anyone re-reading that section today reaches the wrong conclusion. Likewise
  `export_gemm_rtp.py:88-93`'s "no measurement above seq 64" is accurate for
  *that* path but the whisper path *has* been measured at 1536, and the
  technique is now in the tree.

---

## Gaps and unknowns

- **No measurement of this path above seq 64 exists**, by the repo's own
  admission. Everything in sub-goal 6 about S=1024 is extrapolation from
  whisper's ~90 GFLOP/s at S=1500 and the repo's 2.2 TFLOPS NPU GEMM figure. The
  *first* deliverable should be a measurement, not a kernel.
- **`n_cols` is a build parameter, not 8.** All the divisibility tables are
  computed at `--cols 8`. The uniqueness of `tile_n=16` for the large encoder is
  a property of that configuration; `--cols 4` changes the answer. The best
  choice is undetermined.
- **The cost of two resident `hw_context` objects is explicitly unmeasured** —
  the repo notes a real `serve <llm> --embed 1` has never been run on a full
  xclbin tree. Whether a decision model can coexist with anything is unknown.
- **Whether the llama.cpp PR will merge, and in what form** is unknown.
  `mergeable_state: "unstable"`; no maintainer has commented beyond bot flags.
  Whether M3's graph lands in `src/models/laya.cpp` or stays in `tools/laya` is
  unspecified.
- **The packing decision is a measurement, not a derivation.** The
  `rl_agent_config.json` `temperature_by_options` buckets are read by no llama.cpp
  converter and would need a home in the host wrapper.
- **No end-to-end accuracy gate exists for a decision model.** `oflm-test
  --embedding`'s E1–E11 are all shape/determinism/identity checks on vectors;
  the *decision-accuracy* analogue (agreement with the PyTorch reference on the
  choice) is entirely unbuilt, and given the attention-saturation finding that is
  the gate that actually matters.
- **The `temperature` buffer is dead in the PyTorch forward** but still occupies
  3 floats in the checkpoint; whether to carry it in the container at all is a
  small open decision.

---

## Recommended path

Ordered, with the honest cost of each step. The sequencing matters: two of these
are cheap and unblock everything after them.

1. **Add the `model_type == "modernbert"` dispatch branch to
   `prepare_model_auto` that throws `not implemented`.** ~10 lines. Removes the
   silent arch-0 fail-open *today*, before any other work. Per the repo's own
   rule: *"If closed behavior is not reproduced, return an explicit not
   implemented error rather than silently depending on the closed component."*
2. **Wire the byte-level BPE tokenizer** for the English (ModernBERT-large)
   tokenizer, and generate a Metaspace blob for the multilingual one. The
   generator and tokenizer are already compiled and linked; the sibling
   Gemma/XLM-R wiring is the template. This is the single largest
   completed-but-unreachable piece of ModernBERT work already in the tree.
3. **Write `prepare_model_modernbert()`** modeled on `prepare_model_gte` (the
   only packer carrying a *computed* RoPE set, a prose half-order key, a
   zero-filled bias, and an embedded tokenizer blob). Emit zeros for
   `embeddings.position` / `token_type` / all `*.bias` (the runtime dereferences
   them unconditionally). Splice `Wi`'s gate half into `add_gemm_b_concat2` —
   that helper already exists and is exactly the needed shape, but ModernBERT
   stores `w1` fused, so it needs a row-slice variant.
4. **Build the two encoder families** and gate the `--emulate-bfp16` decision by
   ablation, not by inheritance.
5. **Add the pre-LN runtime path.** `run_preln()` alongside `run()`, with a
   separate `hbuf`, a `final_norm` at the new `s_ln[1 + 2*g_layers]` site, an
   `identity_ln1` flag for layer 0, and a two-table RoPE selection copied from
   `GemmaNpuEncoder`. **Bypass `add_norm_*` entirely** — they store the
   normalised value as the residual, which is wrong under pre-LN, not merely
   slow.
6. **Add the ±64 band mask**, clamped in `qk_impl`/`av_impl` so the MACs are
   skipped. One-line index change in `softmax_cpu` (it currently loops `(b,h)`
   uniformly) plus a `j`-range clamp.
7. **Measure at S=1024** with the existing phase timers, and replace the
   `export_gemm_rtp.py:400-406` "unknown, not assumed" warning with a number. Do
   this *before* deciding on NPU attention.
8. **Build `AutoDecisionModel` + a `decisions` endpoint**, patterned on
   `AutoEmbeddingModel` and on `bench-embed` for the CLI. Enrol the route in
   `requires_npu_access()` or it will run outside the NPU lock. A dedicated
   process is the honest first shape given `ShapeLease`.
9. **Generalize `whisper_fa`** (two-sided `apply_window_mask` + `q_block_idx`
   routing, H=16, one S) only if step 7 says attention is the remaining
   bottleneck at your target length.
10. **Write the skill file**, per the repo's standing instruction, so the next
    ModernBERT-shaped model doesn't repeat this research.

**Explicitly not recommended:** waiting on the llama.cpp PR, or routing Laya
through `q4nx-build --open-embedding`. The first is CPU-only and out-of-core; the
second is a Gemma3-only engine that hard-fails on both missing dense heads and
non-Gemma3 config.

---

## Sources

[^1]: [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) — `laya/common.py`, `laya/agent.py`, `laya/fast.py`, `laya/tl_kernels.py`, `scripts/export_onnx.py`, README — Tier 1
[^2]: [laya.convaiinnovations.com](https://laya.convaiinnovations.com/) · [Laya README, "Installation details" + integrations](https://github.com/NandhaKishorM/laya) — Tier 1
[^3]: [convaiinnovations/laya on HuggingFace](https://huggingface.co/convaiinnovations/laya) — `multilingual/rl_agent_config.json`, `multilingual/encoder/config.json`, repo file tree — Tier 1
[^4]: [Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder (arXiv 2412.13663)](https://arxiv.org/abs/2412.13663) · [answerdotai/ModernBERT-large config.json](https://huggingface.co/answerdotai/ModernBERT-large/raw/main/config.json) — Tier 1/2
[^5]: [transformers `modeling_modernbert.py` (main)](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/modernbert/modeling_modernbert.py) · [`configuration_modernbert.py`](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/modernbert/configuration_modernbert.py) · safetensors headers read by HTTP range request — Tier 1
[^6]: [ggml-org/llama.cpp PR #29363](https://github.com/ggml-org/llama.cpp/pull/29363) · [GitHub API: PR files](https://api.github.com/repos/ggml-org/llama.cpp/pulls/29363/files) · [Mesh-LLM/mesh-llm#2083](https://github.com/Mesh-LLM/mesh-llm/pull/2083) — Tier 1
[^7]: [jhu-clsp/mmBERT-base config.json](https://huggingface.co/jhu-clsp/mmBERT-base/raw/main/config.json) · [JHU-CLSP/mmBERT](https://github.com/JHU-CLSP/mmBERT) (arXiv 2509.06888) — Tier 1/2
[^8]: [ModernBERT HF Blog / collection](https://huggingface.co/collections/jhu-clsp/mmbert-a-modern-multilingual-encoder) — Tier 1
[^9]: `npu_offload/gemm_rtp/npue.py:106-141` (arch=4 spec) · `src/open_npue/SYNCED.md` · `src/open_npue/tokenizer_bbpe.hpp` — local repo, Tier 1
[^10]: `npu_offload/gemm_rtp/export_gemm_rtp.py:61-126, 375-406` · `src/open_npue/npue_encoder.hpp:392-826, 2809-2995, 3938-4018` — local repo, Tier 1
[^11]: `src/open_npue/npue_encoder.hpp:4000-4018` (`design_fits` Want table + occurrence loop) — local repo, Tier 1
[^12]: `specs/open-whisper/spec.md:45-49, 158-159, 242-301` · `src/open_whisper/README.md:155-168` — local repo, Tier 1
[^13]: `open_kernels/designs/whisper_fa/attn_fa.py:41-44, 240-332` · `attn_npu2.cc:1102-1347` · `src/open_whisper/fa_attention.cpp:213-223` — local repo, Tier 1
[^14]: `transformers/masking_utils.py` (`sliding_window_bidirectional_overlay`) · `npu_offload/gemm_rtp/families.json:30-66` — Tier 1
[^15]: `npu_offload/gemm_rtp/gemm_pretiled.py:100-160, 360-392, 875-882` · `export_gemm_rtp.py:235-282, 286-374, 460-483` · `npu_offload/m5-eltwise/kernels/narrow_f32_bf16.cc` — local repo, Tier 1
[^16]: XRT 2.25.0 dispatch-overhead measurements on Ryzen AI 300 (AIE2P, 8 cols × 6 rows) — Tier 1 measurement, local
[^17]: `open_kernels/designs/attn_block/README.md:24-33` (2.2 TFLOPS; 62× host-vs-NPU) — local repo, Tier 1
[^18]: `src/include/AutoEmbeddingModel/auto_embedding_model.hpp:54-132` · `all_embedding_model.hpp:30-85` · `src/server/rest_handler.cpp:1039-1359` · `openai_compat.hpp:200-224` — local repo, Tier 1
[^19]: `utilities/q4nx-build/q4nx/open_embedding.py:98-193` · `src/open_npue/npue_pack.cpp:1971-2049` · `utilities/oflm-add/oflm_add/__init__.py:49-51` — local repo, Tier 1
[^20]: [zzhdbw/laya-Ascend](https://github.com/zzhdbw/laya-Ascend) — third-party Huawei Ascend port — Tier 6
