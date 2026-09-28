# Plan: Laya on NPU2 — the `scored_slot` readout

**Status:** designed 2026-09-29. Nothing built, nothing run.
**Spec impact:** five new requirements —
`OPEN-NPUE-MODERNBERT` (arch=4 encoder), `OPEN-DECISION-READOUT` (the plan/answer
layer), `OPEN-DECISION-SYSTEMONE` (the wire schema), `OPEN-DECISION-HEAD`
(the NPU-resident decision head), `OPEN-DECISION-ACCURACY` (the gate).
No existing requirement changes.
**Research:** `docs/plans/laya_npu2_research.md`. Read it first — it carries the
architecture, the tile arithmetic, the twelve traps, and the counter-evidence.

## What we are building

A non-autoregressive **System 1 decision model** served over a TypeSafe-compatible
`/v1/systemone`. Not a chat model, not an embedding model: it takes a *state*
plus typed *questions* and returns a probability distribution per question in
one forward pass. No generation, no parsing, no hallucination.

We are building **one of four readouts** and naming the other three so a future
model of this class drops in without an API change:

```c
enum decision_readout {
  DECISION_READOUT_LETTER_SLOT,  // causal; distribution over label tokens   [not implemented]
  DECISION_READOUT_MASKED_SLOT,  // bidirectional; distribution at a mask    [not implemented]
  DECISION_READOUT_SCORED_SLOT,  // bidirectional; one NUMBER per option    [PHASE 5-7]  <- Laya
  DECISION_READOUT_RANK_HEAD,    // classification head on a pooled vector  [not implemented]
};
```

An unimplemented readout **refuses by name**. It does not silently fall back to
`scored_slot` — the repo's rule, and llama.cpp's: routing a model to the nearest
recipe "would emit kernels that drop a whole stage of the layer and then report
parity against a replica making the same mistake."

## Fixed decisions

These are settled. An implementer does not re-open them.

| # | Decision | Rationale |
|---|---|---|
| 1 | **Wire format is TypeSafe `/v1/systemone`, byte-compatible.** | Already implemented by LiteLLM, llmgateway, Mesh-LLM and `yijunyu/jev-rs`. Being a drop-in for the ecosystem is worth more than an OFLM-native route name. |
| 2 | **All four readouts are declared; only `scored_slot` is implemented.** | Matches llama.cpp #29321's enum exactly. Unimplemented ones refuse by name. |
| 3 | **`laya-multilingual` only.** | One RoPE base (160000 for both layer types), `tile_n=48` matches shipping families, default 1024 ctx. English is deferred — it forces `tile_n=16` and two thetas for no proof-of-surface. |
| 4 | **Head on the NPU, reusing `BERT-h768-bfp16`.** | The head is arithmetically a BERT layer; zero new xclbins. |
| 5 | **Dedicated process, no co-residency.** | `ShapeLease` makes the engine geometry process-wide. Two resident `hw_context` objects is explicitly unmeasured in this repo. |
| 6 | **Extend `open_npue`; add `AutoDecisionModel` as a sibling seam.** | `SYNCED.md` says edit upstream, not the synced copy. The abstraction that needs a new interface is the *app* seam, not the engine. |
| 7 | **`oflm pull` from a `model_list.json` entry.** | Consistent with the six existing encoders. |
| 8 | **S=1024, batch tiers 1,4,16,32,64.** | Laya's own default `max_len`. Tiers capped at 64 because host attention materialises `batch x heads x S^2 x 4` = 3.2 GB at 64. |
| 9 | **Performance is a separate project.** | Build for correctness. `tile_n` and the pad-vs-16 trade-off are re-opened only after this ships. |

---

# Ground truth

Everything here is fixed by the checkpoint. Do not re-derive it; do not
"improve" it.

## Geometry — `laya-multilingual` (mmBERT-base)

| field | value | source |
|---|---|---|
| `model_type` | `modernbert` | `multilingual/encoder/config.json` |
| hidden | 768 |  |
| layers | 22 |  |
| heads | 12 |  |
| head_dim | 64 |  |
| intermediate | **1152** | 1.5x hidden, *not* 4x |
| vocab | 256000 |  |
| `max_position_embeddings` | 8192 | RoPE cache length only; **not** a table |
| `local_attention` | 128 → half-window **64** |  |
| `global_attn_every_n_layers` | 3 | → 8 global (0,3,6,9,12,15,18,21), 14 sliding |
| `norm_eps` / `layer_norm_eps` | 1e-5 |  |
| biases | **none anywhere in the encoder** | `attention_bias`/`mlp_bias`/`norm_bias` all false |
| `position_embedding_type` | `sans_pos` | honest value; ModernBERT's `"absolute"` is a dead key |
| RoPE theta | **160000 for BOTH layer types** | differs from ModernBERT-large; do not hardcode the split |
| `head_layers` | 2 | `rl_agent_config.json` |
| `max_len` / `head_max_len` | 1024 / 256 |  |
| `mask_token_id` | 4 |  |
| `cls`/`sep`/`eos` | 1 (all three) | `bos`=2, `pad`=0, `unk`=3 |
| `n_qtype` | 3 | choice=0, score=1, noul=2 |

## Encoder GEMMs — what the design must serve

| op | K | N | notes |
|---|---|---|---|
| `qkv` | 768 | 2304 | fused, bias-free |
| `attn_out` | 768 | 768 | bias-free |
| `ffn_up` | 768 | **2304** | gated: `Wi` is [2304,768], **gate is the SECOND half** |
| `ffn_down` | 1152 | 768 | bias-free |

## Head GEMMs — `nn.TransformerEncoderLayer(768, 12, 3072)`, **with** biases

| op | K | N | design |
|---|---|---|---|
| `qkv` | 768 | 2304 | `in_proj_weight` + `in_proj_bias` |
| `attn_out` | 768 | 768 | `out_proj.weight` + `.bias` |
| `ffn_up` | 768 | 3072 | `linear1` + bias, **ReLU** |
| `ffn_down` | 3072 | 768 | `linear2` + bias |

**Identical to `BERT-h768-bfp16`.** `design_fits(768, 3072, gated=false, qkv_n=2304)`
returns true against the shipping set. No new xclbin.

Head tail is **host-only** and must stay there — it is not array-tileable:

| tensor | shape | why not on the array |
|---|---|---|
| `scorer.1` | [768,768] | could go, but see below |
| `scorer.3` | [1,768] | N=1 fails `N % (tile_n*8) == 0` for every tile |
| `act_head.0` | [256,772] | K=772 fails `K % 64 == 0` |
| `act_head.2` | [2,256] | N=2, same failure |

`scorer.1` is kept on the host with the rest of the tail: it runs on K marker
positions (≤20), not on rows, so there is nothing to tile. Host it.

## Prompt format — exact, from `laya/common.py`

```
[CLS] "<type> question: <instructions>" [SEP] [MASK] opt0 [MASK] opt1 … [SEP] <state> [SEP]
 ^1                                                        ^4                    ^1
```

Assembly, in order, from `build_sequence` (`laya/common.py:135-199`):

1. `head_ids = encode(f"{type} question: {instructions}")`, `add_special_tokens=False`
2. for each option in order: `opt_ids[i] = [mask_token_id] + encode(" " + opt_text, truncation=True, max_length=48)`
3. `opt_budget = head_max_len - sum(len(o) for o in opt_ids)`
   - if `opt_budget < 16`: `per = max(4, (head_max_len - 16) // len(opt_ids))`; truncate every option to `per`; recompute
4. `head_ids = head_ids[: max(8, opt_budget)]`
5. `ids = [cls] + head_ids + [sep]`; `markers[i] = len(ids)`; `ids += opt_ids[i]` for each
6. `ids.append(sep)`
7. `room = max(0, max_len - len(ids) - 1)`; `state_ids = state_ids[:room]` (right-truncate, keep the **head** of the state)
8. `ids = ids + state_ids + [sep]`; `ids = ids[:max_len]`; `markers = [m for m in markers if m < max_len]`

**Non-negotiable details:**

- The `[MASK]` is a **standalone token** at `markers[i]`. The option text follows it.
- Every encode is `add_special_tokens=False`. Specials are inserted by hand.
- Option order is semantic. For `score` the order *is* the scale.
- `str(k)` when a choice label has no description; `"%s: %s" % (k, desc)` when it does.
- `score` options render as `"level %d: %s" % (i, crit)`.
- `noul` renders exactly two options, semantic order `[false, true]`:
  - `f"{false_label}: " + (crit or "no, the statement does not hold")`
  - `f"{true_label}: " + (crit or "yes, the statement holds")`
- **Mask-token sanitising:** the literal mask token string is replaced with a space
  in the instructions, in every option, and in the state. This is a
  prompt-injection guard and it is part of the format.

## State / criterion serialisation — this is where llama.cpp lost 17 decisions

`serialize_state` (`laya/common.py:74-77`):

```python
def serialize_state(state):
    if isinstance(state, str): return state
    return json.dumps(state, ensure_ascii=False)
```

`render_criterion` (`laya/common.py:80-89`):

```python
def render_criterion(value):
    if isinstance(value, str): return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)
```

Both use `ensure_ascii=False` and separators `(", ", ": ")`. A non-string
`state` or `criteria` value is **structured JSON**, not a string. Reproduce
Python's `json.dumps` exactly, including:

- **shortest round-trip float formatting** (`repr`-equivalent). `%.17g` is
  *wrong* and flipped 17 of 208 decisions in llama.cpp.
- `ensure_ascii=False` — non-ASCII is emitted raw, not `\uXXXX`-escaped.
- `default=str` for non-JSON-native values.

Phase 1 ships the `state: string` path. Phase 1's gate is the string path only;
the structured path is a follow-on and is **not** required to ship.

## Tokenizer — the gating dependency

`multilingual/tokenizer/tokenizer.json` is **Metaspace** (`prepend_scheme:
"always"`, `replacement: "▁"`, `split: true`) over byte-level BPE with
`byte_fallback: true`, plus an NFC-free normaliser that does `Replace(" " →
"▁")`, and a `TemplateProcessing` post-processor of `<bos> $A <eos>`.

Two consequences that will silently corrupt every answer if missed:

1. **`prepend_scheme: "always"` inserts a leading `▁` on every call.** So
   encoding `" " + opt` produces a doubled space marker. Reproduce HF's exact
   per-segment call sequence; do not "normalise" the inputs.
2. **Added-token pre-matching must happen before Metaspace.** Without it,
   2783/4595 strings tokenised differently in the llama.cpp port — and any
   state or question containing a **newline** is affected.

---

# Phases

Each phase has an exit gate that can fail. Do not start a phase until the
previous gate passes.

## Phase 0 — Fail closed on an unrecognised `model_type`

**Problem.** `prepare_model_auto` (`src/open_npue/npue_pack.cpp:1971-2049`)
dispatches on `model_type` with the BERT packer as the *deliberate* last
branch. `model_type: "modernbert"` therefore packs as `bert_abs_gelu_postln` —
GELU + absolute position embeddings for a GeGLU + RoPE model — and the runtime
loads it and returns wrong vectors. This is a live wrong-answer path today,
independent of everything else here.

**Change.** One branch, inserted between `npue_pack.cpp:2033` and `:2035`:

```cpp
if (model_type == "modernbert") {
  throw std::runtime_error(
      "modernbert_rope_geglu is not packed by this build. The BERT fallback "
      "below would emit a valid-looking arch=0 container for a GeGLU/RoPE "
      "model; refusing rather than writing the wrong answer. See "
      "npu_offload/gemm_rtp/npue.py:106-141 for the architecture and "
      "specs/open-engine/plans/laya-decision-encoder.md for the port.");
}
```

**Gate.** Packing any `modernbert` checkpoint raises an error naming
`model_type`. `all-minilm`, `bge-*`, `nomic`, `gte`, `embeddinggemma` all still
pack and still produce byte-identical containers. **Re-run
`utilities/test_open_npue.ps1` for all seven.**

## Phase 1 — Tokenizer: Metaspace byte-BPE

**State.** `src/open_npue/tokenizer_bbpe.{hpp,cpp}` and
`bbpe_tokenizer_gen.cpp` are written, generated, compiled and linked
(`src/CMakeLists.txt:338-339`, both objects present in `build/src/CMakeFiles/`).
**Nothing calls them.** `BbpeTokenizer`, `BbpeEncoded` and
`generate_bbpe_tokenizer_table` have zero call sites; `nm -C build/src/oflm`
shows the symbols linked in as global text with no references. The sibling
Gemma and XLM-R generators *are* called (`npue_pack.cpp:1006`, `:1724`), which
is the proof the omission is accidental, not a decision.

**Change.**

1. Read `bbpe_tokenizer_gen.hpp` and add a **Metaspace mode** to
   `generate_bbpe_tokenizer_table`. The generator currently assumes
   `ByteLevel` pre-tokenisation; the multilingual checkpoint is `Metaspace`.
   Store `prepend_scheme`, `replacement` and `split` in the generated blob.
2. Add the `BbpeTokenizer` construction to `load_tokenizer`
   (`npue_encoder.hpp:374-418`) under a new arch branch, mirroring
   `npue_encoder.hpp:388`'s construction of `XlmrTokenizer`.
3. Add `modernbert_rope_geglu` to `encoder_implemented`
   (`npue_encoder.hpp:301-314`) — **not in this phase**; the packer does not
   exist yet and enabling it would make Phase 0 unreachable. The arch string is
   added in Phase 2.
4. Tokenise **per segment with `add_special_tokens=false`**, specials inserted
   by hand. Never one pass over the assembled string.

**Gate.** For a fixture of ≥200 strings (must include: strings with newlines,
tabs, runs of spaces, HTML tags, `<unusedN>` tokens, non-Latin scripts, emoji,
and the literal `[MASK]` string), the C++ tokenizer's ids are **exactly equal**
to HF `tokenizer(text, add_special_tokens=False)["input_ids"]`. Any single
divergence fails the phase. The generated unicode tables are already measured
against HF's splitter, so this is a wiring check, not a re-derivation.

## Phase 2 — The packer

Model `prepare_model_gte` (`npue_pack.cpp:1516-1900`) — it is the only packer
that carries a *computed* RoPE set, a prose half-order key, a zero-filled bias
and an embedded tokenizer blob.

**Signature** — identical to `prepare_model_gte`
(`npue_pack.hpp:132-139`):

```cpp
void prepare_model_modernbert(const std::string &model_dir, const std::string &pooling,
                              const std::string &source_repo, const std::string &out,
                              const std::string &layout_json, const std::string &layout_hash,
                              int64_t tile_k, int64_t tile_n, int64_t max_seq,
                              void (*log)(const std::string &));
```

**Workload handling.** Laya ships no root `config.json`; it is at
`<dir>/encoder/config.json`. The packer reads the encoder config from there and
the weights from `<dir>/model.safetensors`. `prepare_model_auto` is invoked
with `checkpoint_dir` = the *checkpoint subdirectory* (`multilingual/`), and
`resolve_pooling` is satisfied by writing a `1_Pooling/config.json` into the
served model directory at pack time (see Phase 9) — **not** by relaxing
`resolve_pooling`, which is a deliberate name-checked contract.

**Container config keys to emit**, in this order (the packers build raw JSON by
string append and the key order must be stable):

```
{"arch":"modernbert_rope_geglu",
 "hidden":768,"layers":22,"heads":12,"head_dim":64,"intermediate":1152,
 "gated_ffn":true,"qkv_n":2304,
 "pre_layernorm":true,"final_norm":true,
 "layer_types":[...22 entries...],
 "rope_theta":160000,
 "identity_attn_norm_layer0":true,
 "sliding_window":64,"global_attn_every_n_layers":3,
 "swiglu_halves":"gate|up",
 "norm_eps":1e-5,"norm_eps_layer":1e-5,
 "pooling":"mean","l2_normalize":false,
 "head_layers":2,"head_intermediate":3072,
 "n_qtype":3,"marker_token_id":4,
 "max_len":1024,"head_max_len":256,
 "not_implemented":[...]}
```

`"l2_normalize": false` — Laya never L2-normalises. Reusing `pool_rows`'s
`true` default would corrupt every answer.

**Tensors.** Emit these, and **no others**. Every `*.bias` and
`embeddings.position` / `embeddings.token_type` is **zero-filled** — the runtime
dereferences all of them unconditionally (`npue_encoder.hpp:4559-4562`,
`:4682-4684`), and a zero tensor of the right shape is exact.

```
embeddings.tok_embeddings  [256000, 768] BF16   (gather)
embeddings.position        [max_seq, 768]  F32   ZERO
embeddings.token_type      [1, 768]        F32   ZERO
embeddings.ln.weight       [768]           F32
embeddings.ln.bias         [768]           F32   ZERO
final_norm.weight          [768]           F32
final_norm.bias            [768]           F32   ZERO
layer.{L}.ln1.weight      [768]           F32     L in 1..21
layer.{L}.ln1.bias        [768]           F32     ZERO
layer.{L}.ln2.weight      [768]           F32
layer.{L}.ln2.bias        [768]           F32     ZERO
layer.{L}.qkv.weight      [2304, 768]     gemm_b
layer.{L}.qkv.bias        [2304]          F32     ZERO
layer.{L}.attn_out.weight [768, 768]      gemm_b
layer.{L}.attn_out.bias   [768]           F32     ZERO
layer.{L}.ffn_up.weight   [2304, 768]     gemm_b   concat[gate, up]
layer.{L}.ffn_up.bias     [2304]          F32     ZERO
layer.{L}.ffn_down.weight [768, 1152]     gemm_b
layer.{L}.ffn_down.bias   [768]           F32     ZERO
```

**Layer 0 has no `ln1`** in the checkpoint. The tensor set stays uniform: emit a
zeros `layer.0.ln1.weight` and carry `identity_attn_norm_layer0: true` so the
runtime *skips the norm*, rather than normalising. A weight-1 norm still
centres and scales — this is why it is a config field and not a unit tensor.

**The one new helper.** `mlp.Wi` is stored fused `[2304, 768]`. The runtime
wants `concat([Wi_gate_half, up], N=0)`. `add_gemm_b_concat2`
(`npue_pack.cpp:1111-1130`) already does exactly this concat but takes whole
`Tensor`s. Add a row-slice variant:

```cpp
// like add_gemm_b_concat2, but `a` is a row range of one tensor rather than a whole tensor.
// ModernBERT stores Wi fused; the runtime wants the gate half and the up half concatenated
// in that order, so this slices [r0, r0+rows) out of `fused` and prepends it to `b`.
static void add_gemm_b_concat_rows(Writer &w, const std::string &name,
                                   const Tensor &fused, int64_t r0, int64_t rows,
                                   const Tensor &b, int64_t tk, int64_t tn,
                                   const std::string &layout_json,
                                   const std::string &layout_hash);
```

**Head tensors** are packed too (Phase 7 consumes them), named:

```
head.layers.{H}.norm1.{weight,bias}        [768]        F32
head.layers.{H}.self_attn.in_proj.weight  [2304, 768]  gemm_b   H in 0..1
head.layers.{H}.self_attn.in_proj.bias    [2304]       F32
head.layers.{H}.self_attn.out_proj.weight [768, 768]   gemm_b
head.layers.{H}.self_attn.out_proj.bias   [768]        F32
head.layers.{H}.norm2.{weight,bias}        [768]        F32
head.layers.{H}.linear1.weight            [3072, 768]  gemm_b
head.layers.{H}.linear1.bias              [3072]       F32
head.layers.{H}.linear2.weight            [768, 3072]  gemm_b
head.layers.{H}.linear2.bias              [768]        F32
type_emb.weight                           [3, 768]     F32
scorer.0.{weight,bias}                    [768]        F32
scorer.1.weight                           [768, 768]   F32
scorer.1.bias                             [768]        F32
scorer.3.weight                           [1, 768]     F32
scorer.3.bias                             [1]          F32
act_head.0.weight                         [256, 772]   F32
act_head.0.bias                           [256]        F32
act_head.2.weight                         [2, 256]     F32
act_head.2.bias                           [2]          F32
```

`in_proj`/`out_proj`/`linear1`/`linear2` are `gemm_b`; the `N=1`, `N=2` and
`K=772` tail is F32 and host-only.

**`not_implemented` — be honest, in prose, in the container.** Minimum entries:
the 8192-token context against a seq=1024 design; `predict_long` windowing with
50 % overlap; the structured-`state` JSON path (Phase 1 ships strings only);
and the Laya `Router`'s language routing. Every one of these is a real
behavioural difference between this container and the Python package.

**Gate.** A synthetic `modernbert` fixture packs to a container that (a) passes
the byte-parity check against `npu_offload/gemm_rtp/npue.py`'s `Writer` for
every shared field, (b) round-trips every emitted tensor through
`npue::Reader::raw()` at the right shape and dtype, and (c) whose
`gemm_b_layout` hash equals the `b_layout_hash` of the design built in Phase 3.
Note `tools/verify_pack_parity.py` lives upstream in `vegah/Npu-Embeddings` and
is **not vendored here** — parity must be re-established, not assumed.

## Phase 3 — The design family

Add to `npu_offload/gemm_rtp/families.json`:

```json
{
  "name": "BERT-h768-gated-i1152-bfp16",
  "serves": ["laya-decision:multilingual"],
  "note": "gated 768 with intermediate 1152, not the 3072 of BERT-h768-gated-bfp16. ffn_up is 2304. 48 is legal and matches every shipping family. The head runs on BERT-h768-bfp16, which must never be purged concurrently with this one: they share 8 of 16 cache markers.",
  "args": ["--hidden","768","--intermediate","1152","--qkv-n","2304",
           "--gated-ffn","--emulate-bfp16","--c-bf16","-n","48"]
}
```

`common` changes from `"--batches","4,16,32,128"` to
`"--batches","1,4,16,32,64"` **for this family only** — do not edit `common`,
because the six existing encoders were gated at 4/16/32/128 and changing it
invalidates their `design.json` check. Put the tiers in this family's `args`.
`--seq 1024`.

Constraints to satisfy (`gemm_pretiled.py:155-158`):
`M = batch x 1024` must be a multiple of 256 → 1024, 4096, 16384, 32768, 65536 ✓.
All N ∈ {2304, 768} divide `48 x 8 = 384` ✓. All K ∈ {768, 1152} divide 64 ✓.

**Build serially.** `build.ps1` loops families one at a time because `purge()`
deletes from the shared `~/.npu/cache` on content markers, and the two hidden-768
families own identical markers for 8 of 16 entries. Parallel builds have
corrupted each other's output before.

**Gate.** `check_design_sets.py --xclbins src/xclbins` passes for all seven
families. The new `design.json` carries `streams[]` for 4 shapes × 5 tiers = 20
entries, all four N values legal, `b_layout_hash` matching Phase 2's container,
and `tile: 48`. Re-run the existing six designs' check to prove `common` was not
disturbed.

## Phase 4 — Pre-LN encoder

**Shape.** `GemmaNpuEncoder::encode_batch` (`npue_encoder.hpp:3825-3887`) with
norms 2 and 4 deleted and norm 1 unconditional except at layer 0.

**Changes to `Encoder`:**

1. New `std::vector<float> hbuf;` member — the norm output must not overwrite
   `x`. `Encoder`'s scratch list is currently `qkvbuf, ctx, proj, up, down,
   scores` (`npue_encoder.hpp:1447`).
2. New `run_preln(const std::vector<float>&) -> std::vector<float>`
   alongside `run()` (`:2809-2995`), not a modification of it — arch=0 through
   3 must stay bit-identical.
3. **`add_norm_quant` / `add_norm_bf16` (`:2343-2515`) must be bypassed
   entirely.** They end with `_mm256_storeu_ps(res + j, yv)`: they store the
   *normalised* value as the residual. Under post-LN that is correct; under
   pre-LN the next block would add to a normalised stream. This is a
   correctness bug, not a performance one.
4. `final_norm` at norm-site `s_ln[1 + 2*g_layers]` — the next free slot after
   the loop in `stage_all` (`:1606-1626`). Emits `final_norm.weight` /
   `final_norm.bias`.
5. Layer 0's `ln1` skipped when `identity_attn_norm_layer0` is set.
6. Two-table RoPE. Copy `GemmaNpuEncoder::ensure_tables` (`:3748-3757`) and its
   per-layer selection (`:3827-3829`). **mmBERT has one theta, so the two tables
   are identical — build one and select it; do not add a per-layer-theta code
   path in this phase.** The English checkpoint's two thetas are Phase 10+.
7. `swiglu_halves == "gate|up"` → `gelu(lo) * hi`. Add the third key beside
   `geglu_halves` (`:3139-3144`) and `swiglu_halves` (`:589-594`). An
   unrecognised string **throws**, as the other two do.

**Loop body, pre-LN:**

```
residual = x
h = LN1(x)  -> hbuf                      (skip at L==0)
qkv = GEMM(qkv, hbuf)
rope(qkv)                                 (one theta)
qk -> scores; band mask; softmax; av -> ctx
proj = GEMM(attn_out, ctx)
x += proj                                 (no norm)
h = LN2(x) -> hbuf
up = GEMM(ffn_up, hbuf)
gated = geglu(up)                         gelu(first half) * second half
down = GEMM(ffn_down, gated)
x += down                                 (no norm)
final = LN_final(x)
```

**Gate.** Per-layer hidden states within the `oflm-test` cosine discipline
against a fp64 Python reference, for all 22 layers plus the final norm.
**Calibrate the threshold from a bf16 replica, never in advance** — the whisper
precedent is that 0.999 on `enc.out` is not reachable at 32-layer depth in
bf16 (`specs/open-whisper/spec.md:45-49`). Report the replica's number and the
engine's number side by side; fail only on a *gap*, not on an absolute value.

## Phase 5 — Band mask, and the S=1024 measurement

Two edits, both small:

1. `softmax_cpu` (`:1757-1810`) currently loops `(b,h)` uniformly via
   `rows_per_seq = g_heads * g_seq` (`:1764`). Index the mask by layer: pass the
   layer index, and for sliding layers mask `|i-j| > 64` to `-INFINITY`.
2. Clamp `j` in `qk_impl` (`:2538-2611`) and `av_impl` (`:2624-2704`) to
   `[i-64, i+64]` on sliding layers, so the MACs are **skipped**. Without this
   the 129/1024 saving is mask arithmetic, not work.

**The half-window is 64, not 65.** `local_attention` is 128 and the mask uses
`local_attention // 2`. The `+1` in HF's
`self.sliding_window = config.sliding_window + 1` is a FlashAttention
inclusive-boundary convention and does **not** apply to a dense/sparse mask.
llama.cpp shipped `half = n_swa/2 + 1`, which allowed `|Δ| ≤ 65` and moved
1967/2000 → 2000/2000 argmax agreement when fixed.

**Safety property.** Outside the band is exactly `-inf`; inside it is
**untouched**, so in-band values are bit-identical to the unmasked path
restricted to the band. `open_whisper/host_ops.cpp:531-536` records that a 1-ulp
softmax change cost a golden token path.

**Then measure.** With the phase-timers already in `npue_encoder.hpp` (`:2847`,
`:2861`, `:1809`), record at S=1024, batch 1 and 4: total wall, NPU GEMM time,
host QK, host softmax, host AV. Replace the standing warning at
`export_gemm_rtp.py:400-406` with the measured numbers, in that file's comment
and in this plan.

**Gate.** The measurement exists and is written into the repo. **Phase 7 does
not start without it.**

## Phase 6 — The readout and plan layer (pure host, no NPU)

**This phase needs no NPU, no model, and no hardware.** It is pure data
transformation, and it is where the biggest unknown lives — whether the wire
format is right. Build it first, gate it on a golden fixture, and the hardware
phases afterwards are mechanical.

**New:** `src/include/AutoDecisionModel/decision_types.hpp` and
`src/common/AutoDecisionModel/decision_types.cpp`.

```cpp
enum decision_readout { DECISION_READOUT_LETTER_SLOT, DECISION_READOUT_MASKED_SLOT,
                        DECISION_READOUT_SCORED_SLOT, DECISION_READOUT_RANK_HEAD };
enum decision_kind    { DECISION_KIND_NOUL, DECISION_KIND_CHOICE, DECISION_KIND_SCORE };

struct decision_question {
  decision_kind kind = DECISION_KIND_NOUL;
  std::string   key;           // caller-chosen name; the response is keyed by it
  std::string   instructions;  // empty is legal
  // choice: label -> description (may be empty). score: ordered levels, index 0 == 0.
  std::vector<std::string> criteria;                       // score, noul
  std::vector<std::pair<std::string,std::string>> labels;  // choice, ordered
};

struct decision_answer {
  decision_kind kind;
  float              noul;        // noul only
  std::string        choice;      // choice only: the argmax label
  float              score;       // score only: sum(i * p_i)
  std::vector<float> probabilities;   // choice: name-keyed via labels; score: index-keyed
  float              confidence = 0.f; // choice, score only. NEVER set for noul.
  std::vector<float> logits;      // raw, always
};
```

**Rules, all enforced with a named error:**

- `noul` has **no** `confidence` field in the response. (`1 - H(p)/ln 2`
  degenerates to 1.0, so it would be a constant lie.) Serialising it is an error.
- `choice`/`score` require **≥ 2** criteria. `score` criteria are **ordered** and
  the order is the scale.
- `choice` supports up to **255** options. Above that, refuse by name.
- **Question order is semantic.** `nlohmann::json` is a `std::map` and will sort
  keys alphabetically, silently reordering both questions and a choice's
  criteria. Use `nlohmann::ordered_json` for anything touching the request or
  the response.
- An unknown `type` is **dropped with a warning, not fatal** — forward
  compatibility is an explicit requirement of the SDK behaviour.
- `output_tokens` is **always 0**. It is the definitional property of the class.
- `state` as a list of content parts: refuse by name, `"'state' as a list of
  content parts is not supported"`.

**Confidence — a documented divergence from llama.cpp.** Laya ships **two**
confidences (`laya/common.py:470-496`): `answer_confidence = max(p)` (calibrated)
and `confidence = 1 - H(p)/ln K` (uncalibrated). llama.cpp #29321 emits the
latter. TypeSafe's `confidence` field is documented as calibrated, and the
payload's own thesis is calibration. **Emit `max(p)` after temperature.**
The raw `logits` and `probabilities` are both in the response, so a caller that
wants the entropy form can compute it. Record the divergence in the README.

**Temperature.** One value for the whole request — a per-question temperature is
an error (`"one value for the whole request, not per question"`), matching the
TypeSafe schema. Applied on the host after the forward pass, from
`rl_agent_config.json`'s 3-vector, with the `temperature_by_options` buckets
(`"<type>:2"`, `"<type>:3-5"`, `"<type>:6-10"`, `"<type>:11+"`) taking
precedence, clamped to `[0.5, 5.0]`.

**`readout` derivation.** From the container, exactly as llama.cpp derives it
from GGUF metadata: a `scored_slot` is an arch with a decision head, a marker
token, and non-causal attention. A model whose `readout` is not
`DECISION_READOUT_SCORED_SLOT` **refuses by name** when this build is asked to
serve it.

**Gate.** A golden JSON fixture — the request and the exact expected response,
plus a synthetic logit tensor — round-trips byte-for-byte, with `ordered_json`
preserving question and criteria order. The fixture is the TypeSafe
`SystemOneRequest`/`SystemOneResponse` schema from
`typesafe-sdk-python/src/typesafe_sdk/_schemas/models.py`:
`noul` → `{type, noul}`; `choice` → `{type, choice, confidence, probabilities{name:float}}`;
`score` → `{type, score, confidence, legend{int:...}, probabilities{int:float}}`.
Unit tests in `src/common/AutoDecisionModel/` for every refusal above.

## Phase 7 — The decision engine

`src/include/AutoDecisionModel/auto_decision_model.hpp` +
`auto_decision_model.hpp` registry, a sibling of `AutoEmbeddingModel`
(`src/include/AutoEmbeddingModel/all_embedding_model.hpp:30-85`). Same shape:
`load_model`, `decide(state, questions, temperature) -> answers`, plus
`readout()`, `supported_kinds()`, `prompt_names()`.

Behind it, a `Decider` in `src/open_npue/` that:

1. Builds the prompt per `build_sequence` (Phase 1's tokenizer). **One row per
   (state, question) pair** — `collate_items` flattens question groups, and each
   row is independent, so a batch of 5 questions is 5 rows through the same
   encoder.
2. Runs `Encoder::run_preln` + the head transformer.
3. **The head's 2 layers on the NPU via `BERT-h768-bfp16`.** A second
   `Stack`/`Design` pair over the same container, with its **own**
   `intermediate` (3072) — it must **not** read the container's single
   `g_ffn` (1152). This is why `design_fits`' "every occurrence must match" rule
   (`npue_encoder.hpp:4009-4017`) does not bite: the two geometries live in two
   design sets, not one.
4. Gathers the `markers[i]` positions, runs the host tail (`scorer` LN+Linear+
   GELU+Linear, `masked_fill(~marker_mask, -1e4)`).
5. Applies temperature, softmax, argmax, `max(p)` confidence, and
   `score = sum(i * p_i)`.

**The `act_head` is computed and discarded.** Laya's own issue tracker records
`action.act_probability` at AUROC 0.30 versus 0.77 for `confidence`. It is dead
weight. Compute it (it is in the checkpoint, and skipping it is a silent
divergence) but do not return it; record why in the `not_implemented` list.

**Gate.** Against the PyTorch reference: for ≥ 50 (state, question) pairs, the
argmax agrees on **100 %**, `confidence` within 1e-3, and raw logits within the
bf16 tolerance established in Phase 4. **Argmax agreement is the gate; logits
are diagnostics.** Given the llama.cpp finding that this architecture's
pre-softmax scores reach ~55 and amplify summation-order differences, a
logit-tolerance gate would be both unmeetable and meaningless.

## Phase 8 — The surface

**CLI first.** `oflm decide <tag> --input-file <f.json> [--temperature T] [--json]`.
Modelled on `bench-embed` (`src/src/benchmark_embed.hpp:263-478`): a new
`src/src/decision_cli.hpp`, one `main.cpp` branch near `:635`, one help line
near `src/include/utils/vm_args.hpp:31`, three `program_args_t` fields. No server
work at all. Refuses `--decisionmodel` together with a chat model, by name.

**Then the server.** `POST /v1/systemone` in `create_lm_server`
(`src/server/server.cpp:912-1143`), beside `/v1/embeddings` at `:1049`.

**The route must be added to `requires_npu_access()` (`server.cpp:184-194`) or
it runs outside the NPU lock** and can be served concurrently with
`/v1/chat/completions`, which will corrupt or crash the shared `hw_context`.

A new `handle_systemone` in `rest_handler.cpp` beside `handle_embeddings`
(`:1039-1359`). The **model-identity guard** from `:1103-1137` is mandatory and
must be copied: a request naming a different model must be refused, because
echoing the asked-for name over the loaded model's answers is the worst version
of a wrong answer.

`capabilities` at `rest_handler.cpp:752` is a hardcoded literal that already
omits `embeddings`; add `systemone`.

**Gate.** CLI end-to-end against the Phase 7 reference **before** the server
route is written. Then the server route returns the identical bytes for the
identical request.

## Phase 9 — Packaging, tests, skill

1. **`1_Pooling/config.json`** written into the served model directory at pack
   time with `{"pooling_mode_mean_tokens": true, "pooling_mode_cls_token":
   false}` — `resolve_pooling` (`npue_pack.cpp:1949-1966`) refuses anything
   else, and that is a deliberate contract.
2. **`model_list.json`** entry, `laya-decision` / `multilingual`, with
   `npue_design_family: "BERT-h768-gated-i1152-bfp16"`, `npue_tile_n: 48`,
   `default_context_length: 1024`, `label: ["systemone"]`, and `files`
   including `1_Pooling/config.json`. `oflm_min_version: "0.0.0"`.
   Register in `all_embedding_model.hpp`'s sibling registry.
3. **`model_info.json`** manifest entry. **Known gap:** the downloader has no
   `allow_patterns` or subfolder support, and Laya's files live in
   `multilingual/`, `typed-decisions/` and `tokenizer/`. Either vendor the
   three needed files into our own repo, or extend `model_downloader.cpp` to
   fetch a subfolder. **Decision deferred to the implementer: extend the
   downloader** — vendoring 600 MB of weights into this repo is worse.
4. **`oflm-test` suite** `decisions`: D1 shape, D2 probabilities sum to 1 ±
   tol, D3 determinism over 10 draws, D4 batch/index integrity, D5 argmax
   stability, D6 model identity (an impossible tag must be refused), D7 unknown
   question type dropped-with-warning, D8 `noul` has no `confidence`, D9
   **reference agreement** against a bundled golden fixture. Register `"decisions"`
   in `SUITE_NAMES` and make it mutually exclusive with `--embedding` — both take
   the `ShapeLease`.
5. **`src/test/laya_decision_npu/`** — the 4-file harness
   (`test.cpp`, `CMakeLists.txt`, `Makefile`, `test.sh`), using
   `gemma_embedding`'s **standalone** CMake style, not `add_npu_test`, which
   unconditionally links the closed `q4_npu_eXpress`/`mha`/`lm_head` stack.
   Needs `-mavx2 -mfma` and the per-source `open_npue` include dir.
6. **Skill file** `.opencode/skill/open-laya-decision-kernels/SKILL.md`, per
   `AGENTS.md`. `open-phi3-nanbeige-kernels` and `open-qwen36-kernels` are the
   templates. It must record: the prompt format, the twelve traps, the
   `tile_n` reasoning, the fact that T43/T44 of `vegah/Npu-Embeddings`
   (`research/OPEN-THREADS.md`) are the same work seen from upstream, and that
   arch=4 is reusable by any ModernBERT checkpoint.

---

# File manifest

| file | change | phase |
|---|---|---|
| `src/open_npue/npue_pack.cpp` | `modernbert` branch; `prepare_model_modernbert`; `add_gemm_b_concat_rows` | 0, 2 |
| `src/open_npue/npue_pack.hpp` | declare the two | 0, 2 |
| `src/open_npue/bbpe_tokenizer_gen.{hpp,cpp}` | Metaspace mode | 1 |
| `src/open_npue/tokenizer_bbpe.{hpp,cpp}` | Metaspace pre-tokenisation, added-token pre-match | 1 |
| `src/open_npue/npue_encoder.hpp` | `load_tokenizer` arm; `encoder_implemented`; `hbuf`; `run_preln`; `final_norm`; two-table RoPE; `gate\|up`; band mask; `j` clamp; `Decider` | 1, 2, 4, 5, 7 |
| `npu_offload/gemm_rtp/families.json` | one family | 3 |
| `npu_offload/gemm_rtp/export_gemm_rtp.py` | seq-1024 warning → measured numbers | 5 |
| `src/xclbins/BERT-h768-gated-i1152-bfp16/` | built artefacts | 3 |
| `src/include/AutoDecisionModel/*` | the interface | 6, 7, 8 |
| `src/common/AutoDecisionModel/*` | the plan/answer layer + unit tests | 6 |
| `src/open_npue/decision_engine.*` | prompt build, head, readout | 7 |
| `src/src/decision_cli.hpp`, `main.cpp`, `vm_args.hpp`, `program_args.hpp` | `oflm decide` | 8 |
| `src/server/server.cpp`, `rest_handler.{hpp,cpp}` | route, NPU-lock enrolment, identity guard, `capabilities` | 8 |
| `src/model_list.json`, `src/model_info.json` | registry | 9 |
| `utilities/oflm-test/oflm_test/{__init__,tasks}.py` | `decisions` suite | 9 |
| `src/test/laya_decision_npu/` | harness | 9 |
| `.opencode/skill/open-laya-decision-kernels/SKILL.md` | the skill | 9 |

---

# Not in scope

- `laya` (English, ModernBERT-large) and `laya-typed-decisions` — the
  `tile_n=16` family and the two-theta RoPE path. Deferred; the arch=4 packer
  is reusable and only the geometry and the theta selection change.
- The Laya `Router`'s language routing. One model per process, chosen by the
  caller.
- `predict_long` (50 %-overlap windowing for 8k documents).
- Structured (non-string) `state` and `criteria` values.
- `letter_slot`, `masked_slot`, `rank_head` readouts — declared, refusing by
  name.
- NPU-side attention (`whisper_fa` generalisation). Gated on Phase 5's
  measurement.
- Any performance work. Decision 9.
- llama.cpp PR 29363/29321 as a dependency. Read them; do not depend on them.
  #29363 is `mergeable_state: unstable`; #29321 is `draft: true` and cannot
  merge in that state.
