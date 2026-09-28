# Plan: Laya on NPU2 — a ModernBERT decision encoder, and the logit surface it needs

**Status:** designed 2026-09-29. Nothing built, nothing run.
**Spec impact:** three new requirements — `OPEN-FAM-NPUE-MODERNBERT` (the
arch=4 encoder), `OPEN-DECISION-SURFACE` (a model that returns scores, not a
vector), `OPEN-DECISION-ACCURACY` (the gate that actually matters). No existing
requirement changes.
**Research:** `docs/plans/laya_npu2_research.md` — read it before touching
code. It carries the architecture, the tile arithmetic, and the twelve
traps.
**Why it is worth doing:** Laya is a *sibling of the seven embedding models
already in `src/open_npue/`*, not a new engine. `arch=4` is already specified in
`npu_offload/gemm_rtp/npue.py:106-141` and implemented nowhere. The byte-level
BPE tokenizer is already compiled into `oflm` and called from nowhere. The
decision head is arithmetically a BERT layer and reuses two shipping design
families untouched.

## What Laya is, in one paragraph

A non-autoregressive, bidirectional **encoder**, not a decoder. A
ModernBERT backbone runs once over
`[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 … [SEP]
<state> [SEP]`, then 2 `nn.TransformerEncoderLayer`s run over the result, then
the hidden states at the `[MASK]` **marker positions** are gathered and scored
by an MLP into one logit per option. A softmax over those logits, divided by a
per-question-type temperature, is the answer distribution. No KV cache, no
autoregressive loop, 33 ms on a T4.

Three shipped checkpoints:

| tag | backbone | geometry | ctx |
|---|---|---|---|
| `laya` | ModernBERT-large | h1024, 28L, 16H, inter 2624, vocab 50368 | 512 |
| `laya-multilingual` | mmBERT-base | h768, 22L, 12H, inter 1152, vocab 256000 | 1024 (to 8192) |
| `laya-typed-decisions` | ModernBERT-large | as `laya` | 1024 |

## The four decisions that shape everything

| # | Decision | Rationale |
|---|---|---|
| 1 | **Target `laya-multilingual` first.** Its geometry is an ordinary new family at the shipping `tile_n=48`, its 1024-token default is the length we must support anyway, and mmBERT uses **one** RoPE base (160000 for both layer types) so there is no per-layer-theta logic to get wrong. | The English checkpoint forces `tile_n=16` and two RoPE thetas. Do the hard geometry second, on a model that needs less of the API. |
| 2 | **Serve a dedicated process.** One decision model per server, no chat model alongside. | `ShapeLease` makes the engine geometry process-wide; a second model "silently reinterpret[s] the first one's weights and return[s] plausible embeddings for neither". Co-residency of two `hw_context` objects is explicitly unmeasured in this repo. |
| 3 | **Attention stays on the host; banded.** | The FLOP ratio is 12:1 (GEMM:attn) at S=512 and 6:1 at S=1024. The crossover with the 10/18 band split is S≈3400. `whisper_fa` exists and is bidirectional, but it is compile-time-shaped — one xclbin per sequence length — and Laya's whole point is that the length is a request parameter. Defer until a measurement says it is the bottleneck. |
| 4 | **`--emulate-bfp16` is decided by ablation, not inherited.** Five of six shipping encoders use it; `bge-small` failed MTEB at −0.5010 bit-reproducibly and had to be rebuilt. | The llama.cpp reviewer found this model's residual stream carries an attention-sink gate (massive activations ≈2.7e3/4.3e3 in two dims, switched by one FFN neuron in layers 11 and 18) that amplifies a uniform 0.5 % weight rounding — Q8_0 still flipped ~90/2000 decisions. This model is *more* sensitive than bge, not less. |

## What we already have, and what each piece is missing

| piece | state | what is missing |
|---|---|---|
| `arch=4` spec | written in full at `npu_offload/gemm_rtp/npue.py:106-141` | the C++ packer and runtime that read it |
| byte-level BPE tokenizer | written, generated, compiled, linked (`tokenizer_bbpe.cpp`, `bbpe_tokenizer_gen.cpp`, in `OPEN_NPUE_SOURCES`) | **any call site** — `BbpeTokenizer` and `generate_bbpe_tokenizer_table` have zero callers. The sibling Gemma/XLM-R generators *are* called (`npue_pack.cpp:1006`, `:1724`), which proves the omission is accidental |
| `encoder_implemented()` whitelist | lists arch 0/1/2/3 (`npue_encoder.hpp:301-314`) | the arch=4 entry |
| `Encoder::run()` | post-LN, single RoPE table, padding-only mask | pre-LN, per-layer theta, ±64 band, final norm |
| `add_norm_quant` / `add_norm_bf16` | fuse residual-then-norm, storing the **normalised** value as the residual | wrong under pre-LN, not merely slow. Must be bypassed, not adapted |
| two shipping design families | `BERT-h768-bfp16`, `BERT-h1024-bfp16` | **nothing** — the decision head's GEMMs already fit both |
| `AutoEmbeddingModel` | `embed() -> vector<float>`, pure virtual | a class that can return `[rows, options]` |
| `oflm bench-embed` | one-shot, non-server command over the same interface | the template for `oflm decide` |

## The geometry, decided

`design_fits()` (`npue_encoder.hpp:4000-4018`) requires all four ops present and
**every occurrence** to match, against a single
`(hidden, intermediate, gated_ffn)` triple — so one design cannot serve two FFN
widths. It does not need to.

**Encoder** — one new family per checkpoint:

```
laya-multilingual  h=768  inter=1152  qkv_n=2304  gated  ffn_up N=2304
  → -n 48   (legal: {16,32,48}; 48 matches every shipping family, l1 53,248)
laya (large)       h=1024 inter=2624  qkv_n=3072  gated  ffn_up N=5248
  → -n 16   (forced: 5248 = 2^7 * 41, and narrow_f32_bf16 has no m*n=512 entry)
```

**Head** — a `TransformerEncoderLayer(d, d//64, 4d)` *is* a BERT layer:
`in_proj`→qkv, `out_proj`→attn_out, `linear1`→ffn_up, `linear2`→ffn_down.

```
laya-multilingual head  hidden 768  inter 3072  ungated  → BERT-h768-bfp16  ✓ exists
laya            head   hidden 1024 inter 4096  ungated  → BERT-h1024-bfp16 ✓ exists
```

The head's *tail* cannot go on the array under any tile: `scorer` N=1,
`act_head` N=2, and `act_head.0` has **K = d+4**, which fails `K % 64 == 0`.
Host-side only — the same shape as Gemma's post-pool `2_Dense`/`3_Dense` heads.

**`tile_n=16` caveat, recorded so the next agent does not rediscover it:** the
uniqueness above is a property of `--cols 8`. At `--cols 4` the divisors halve
and 32 becomes legal. The alternative worth measuring is padding `ffn_up`
5248→5376 to recover `tile_n=48` — zero columns of B give exactly-zero columns
of C, so it is exact, and the repo already uses this trick for Gemma MQA. It
needs a pad-aware epilogue and stops `design_fits` from being the only source of
truth. **Measure both; do not assume.**

## Phases

Each phase ends with a gate that can fail.

### Phase 0 — Stop the silent wrong answer *(do this first, alone)*

`prepare_model_auto` (`npue_pack.cpp:1971-2049`) dispatches on `model_type`
with the BERT packer as the *deliberate* last branch. `model_type:
"modernbert"` therefore produces a **valid arch-0 container** — GELU + absolute
position embeddings for a GeGLU + RoPE model — and the runtime loads it and
returns wrong vectors.

Add one branch before the fallback that throws. ~10 lines. The repo's own rule:
*"If closed behavior is not reproduced, return an explicit not implemented error
rather than silently depending on the closed component."*

**Gate:** packing a `modernbert` checkpoint raises an error naming
`model_type`, and a BERT checkpoint is unaffected.

### Phase 1 — Tokenizer

Wire `BbpeTokenizer` for the English checkpoint; generate a Metaspace blob for
the multilingual one. Copy the Gemma wiring (`npue_pack.cpp:1006`,
`gemma_encode.cpp:107`) — the shape is already proven.

**The multilingual trap:** its tokenizer is Metaspace with
`prepend_scheme: "always"`, so a leading `▁` is always inserted. A port that
does not reproduce this is silently wrong on every input, and the model's output
will still look plausible.

**Gate:** round-trip every tokenizer fixture against HF's own tokenizer; the
byte-BPE unicode tables are already measured against it, so this is a wiring
check, not a re-derivation.

### Phase 2 — The packer

`prepare_model_modernbert()`, modelled on `prepare_model_gte` (the only packer
carrying a *computed* RoPE set, a prose half-order key, a zero-filled bias, and
an embedded tokenizer blob).

- Emit **zeros** for `embeddings.position`, `embeddings.token_type` and every
  `*.bias`. The runtime dereferences all of them unconditionally; a zero tensor
  of the right shape is exact and costs far less than nullable branches.
- `Wi` is stored fused `[2·inter, hidden]`. The runtime wants
  `concat([gate_half, up], N=0)`. `add_gemm_b_concat2` already does the
  concat but takes whole tensors, so it needs a **row-slice variant**. This is
  the one genuinely new helper.
- Record the half order as `swiglu_halves: "gate|up"` and mean it.
- Record `layer_types` as data, with two thetas where the checkpoint has two.
  mmBERT has one; do not hardcode the ModernBERT split.
- Write a `not_implemented` list. Minimum honest entries: the band mask (if
  Phase 4 is not done), Laya's decision head (if Phase 6 is not done), and the
  8192-token context against a `seq=64` design.

**Gate:** byte-parity against `npu_offload/gemm_rtp/npue.py`'s `Writer` for a
synthetic fixture. Note `tools/verify_pack_parity.py` lives upstream and is
**not vendored here**, so parity must be re-established rather than assumed.

### Phase 3 — The design families

Two builds via `families.json` → `build.ps1` → `check_design_sets.py`. Serial
only — `purge()` matches on content markers and the two hidden-768 families
share 8 of their 16 markers, so parallel builds delete each other's output.

**Gate:** `check_design_sets.py` passes; the two new sets coexist with
`BERT-h768-bfp16` and `BERT-h1024-bfp16` (the head families) without purge
collisions.

### Phase 4 — The pre-LN runtime

The shape is `GemmaNpuEncoder::encode_batch` with norms 2 and 4 deleted and
norm 1 made unconditional (except layer 0).

- `run_preln()` alongside `run()`; a separate `hbuf`; **bypass `add_norm_*`** —
  they store the normalised value as the residual, which is wrong here.
- `final_norm` at the new `s_ln[1 + 2*g_layers]` site.
- `identity_ln1` flag for layer 0. A weight-1 norm is *not* the same as
  `nn.Identity()` — it still centres and scales — which is why this is a config
  field and not a unit tensor.
- Two-table RoPE selection copied from `GemmaNpuEncoder` (`gemma_is_full_attention_layer`).
- The `gate|up` GeGLU order as a third key; an unrecognised string must throw,
  as the other two arches do.

**Gate:** per-layer hidden states within the existing `oflm-test` cosine
discipline against a fp64 reference. Calibrate the threshold **from a bf16
replica, never in advance** — the whisper precedent is that 0.999 on `enc.out`
is not reachable at 32-layer depth in bf16.

### Phase 5 — The band mask, and the measurement

±64 band on the local layers only. Two changes, both small:

- `softmax_cpu` currently loops `(b,h)` uniformly; index the mask by layer
  instead.
- Clamp `j` in `qk_impl`/`av_impl` to the band so the MACs are **skipped** —
  otherwise the 129/1024 saving is mask arithmetic, not work.

**Safety property:** outside the band must be exactly −inf and inside it
untouched, so in-band values are bit-identical to the unmasked path restricted
to the band. `open_whisper/host_ops.cpp:531-536` records that a 1-ulp softmax
change cost a golden token path.

Then **measure at S=1024** with the existing phase timers and replace
`export_gemm_rtp.py:400-406`'s "unknown, not assumed" warning with a number.

**Gate:** the measurement exists and is written down. Do not start Phase 7
without it.

### Phase 6 — The decision surface

This is the largest gap and it is not an NPU problem. `AutoEmbeddingModel`'s
whole contract is `embed() -> vector<float>`; `pool_rows` accepts only `"cls"`
or `"mean"` and throws otherwise; `/v1/embeddings` slices at `dim` floats and
`embedding_batch_dim` *throws* if the flat size does not equal
`n_inputs × expected_dim`. A `[n_inputs, n_options]` return fails that check,
and faking `embedding_dim() == n_options` makes the result indistinguishable
from an embedding everywhere downstream.

- `AutoDecisionModel`, sibling to `AutoEmbeddingModel`, with a
  `decide(state, questions) -> {answers, routing}` entry.
- `oflm decide <tag> --input-file …` first, modelled on `bench-embed`: a
  new `.hpp` + unit test + one `main.cpp` branch + one help line. No server
  work at all.
- Then `POST /v1/decisions`. **Enrol the route in `requires_npu_access()`** or
  it runs outside the NPU lock and can be served concurrently with
  `/v1/chat/completions`.
- Temperature calibration lives here, on the host, from
  `rl_agent_config.json` — including `temperature_by_options` buckets, which no
  llama.cpp converter reads.

**Gate:** the CLI path works end to end against a CPU reference before the
server route is written.

### Phase 7 — NPU attention, only if measured

`whisper_fa` is bidirectional, non-causal, online-softmax, and holds a constant
8 KB score tile per core regardless of sequence length. Its
`apply_window_mask` exists but is **causal and unwired** — zero call sites
repo-wide, and not in the vendored-modification list, so it has never run on
this hardware.

To use it: two-sided bound instead of one-sided, route `q_block_idx`, rebuild at
H=16. The blocker is that `lq`/`lk` are `CompileTime` and the L3 tensor shapes
derive from them — **one xclbin per sequence length.** Whisper gets away with
one because a 30-second window is 1500 frames by definition.

**Gate:** only enter this phase if Phase 5 measured attention as the bottleneck
at the target length.

## What we are explicitly not doing

- **Not** waiting on llama.cpp PR 29363. It is open, CPU-only, and registers
  `LLM_ARCH_LAYA` solely so `llama-quantize` can load a GGUF; the real graph is
  in a standalone `tools/laya` with its own CPU backend. It is a good
  *reference* for the tokenizer, the marker-gather protocol and the head graph —
  and it documents the `Wo`-never-applied bug and the attention saturation that
  a from-scratch port will hit. Read it; do not depend on it.
- **Not** routing through `q4nx-build --open-embedding`. It hard-fails on any
  model without `2_Dense`/`3_Dense` heads, and `open_embedding::engine.cpp`
  implements Gemma3 only.
- **Not** co-residency in phase one. See decision 2.

## Open questions for whoever picks this up

1. `tile_n=16` vs padded-to-48 for the large encoder — measure, do not assume.
   And consider whether `--cols 4` is worth revisiting at all.
2. Is `seq=1024` the right single design, or does Laya need a tier axis? The
   `--batches` tiers are batch tiers whose M is `batch × seq`, so they are not
   a sequence-length mechanism as written.
3. Does the multilingual tokenizer's Metaspace path warrant a second blob
   format, or does the existing BBPE generator grow a Metaspace mode?
4. Should the 256k-vocab embedding table (196 M params, ~1.7× the 22-layer
   stack) be packed, quantised, or refused? It is the dominant memory consumer
   and the ONNX quantiser explicitly leaves it in fp32.
5. Is a `decision` workload class wanted, matching what mesh-llm did? This repo
   has no workload enum and `capabilities` in `rest_handler.cpp:752` is a
   hardcoded literal that already omits `embeddings`.

## When it is done

Write a skill file, per `AGENTS.md`, so the next ModernBERT-shaped model — and
there will be others, since the arch=4 packer is reusable — does not repeat this
research. `open-phi3-nanbeige-kernels` and `open-qwen36-kernels` are the
templates.
