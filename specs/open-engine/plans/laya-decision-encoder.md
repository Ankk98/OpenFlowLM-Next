# Plan: Laya on NPU2 — the `scored_slot` readout

**Status:** designed 2026-09-29. Nothing built, nothing run.
**Spec impact:** five new requirements, **whose verbatim text is in this plan**
under "New requirements": `OPEN-ENC-MODERNBERT` (arch=4 encoder),
`OPEN-DECISION-SYSTEMONE` (the wire schema), `OPEN-DECISION-READOUT` (the
plan/answer layer), `OPEN-DECISION-HEAD` (the decision head),
`OPEN-DECISION-ACCURACY` (the gate). They are appended to
`specs/open-engine/spec.md` in Phase 0, before its last requirement; no existing
requirement changes.
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
  DECISION_READOUT_SCORED_SLOT,  // bidirectional; one NUMBER per option    [PHASE 1, 7]  <- Laya
  DECISION_READOUT_RANK_HEAD,    // classification head on a pooled vector  [not implemented]
};
```

An unimplemented readout **refuses by name**. It does not silently fall back to
`scored_slot` — the repo's rule, and llama.cpp's: routing a model to the nearest
recipe "would emit kernels that drop a whole stage of the layer and then report
parity against a replica making the same mistake."

## The wire schema, pinned

`/v1/systemone` is byte-compatible with TypeSafe's `SystemOneRequest` /
`SystemOneResponse`. **The schema is a pinned vendored copy, not a URL:**

| what | where |
|---|---|
| repository | `github.com/typesafe-ai/typesafe-sdk-python` |
| commit | `0ffd094c72ed9445223060b24ffd7a56aa781fb4` (captured 2026-09-21) |
| the two models | `src/typesafe_sdk/_schemas/models.py` |
| SDK version documented at that commit | 0.7.1 |
| HTTP reference | `POST https://api.typesafe.ai/v1/systemone`; OpenAPI 0.2.0, spec 3.1.0 |

Vendor `_schemas/models.py` (and nothing else — the transport, the retries and
the Pydantic plumbing are not the contract) under
`specs/open-engine/plans/typesafe-systemone-0ffd094c.py`, with the commit in the
header. Phase 1's golden fixture is generated from that file. An unpinned
upstream path is not a fixture.

**Request** — `SystemOneRequest`:

| field | type | required | notes |
|---|---|---|---|
| `state` | `string \| object \| array` | **yes** | the string form ships (Phases 1 and 7); structured is `not_implemented` |
| `model` | `string` | **yes** | the model tag; a *different* tag from the loaded one is refused, not served |
| `questions` | `map<string, Question>` | **yes**, `minProperties: 1` | caller-chosen keys; the key is never sent to the model and is not used in inference |

**Response envelope** — `SystemOneResponse`. This is the part the first draft of
this plan omitted, and omitting it is the difference between "byte-compatible"
and "compatible with the answers":

```json
{"model": "<the tag that actually answered>",
 "answers": { "<question key>": { …one Answer… } },
 "usage": {"input_tokens": <int>, "output_tokens": 0}}
```

`model` and `usage` are both **required**. `model` is *not* the string the
caller sent — it is the resolved model, which may differ from an alias — which
is the second reason the model-identity guard in Phase 8 is not optional.
`usage.input_tokens` is reported from the request; `output_tokens` is **always
0** and that is definitional, not a stub.

**Answers** — a discriminated union on `type`, three members:

| type | fields |
|---|---|
| `noul` | `noul` (0–1) — **no `confidence`**, confirmed by the SDK's own `NoulAnswer` |
| `choice` | `choice` (argmax label), `confidence`, `probabilities: {label: p}` |
| `score` | `score` (probability-weighted mean, may land between levels), `confidence`, `legend: {index: label}`, `probabilities: {index: p}` |

Two serialisation details that "byte-for-byte" turns on:

- **Index-keyed maps serialise with string keys.** The SDK types are
  `dict[int, …]`, and a JSON object key is always a string, so a 3-level score
  emits `{"0": …, "1": …, "2": …}`, not `[{…}]` and not `{0: …}`. The plan's
  earlier shorthand `legend{int:…}` is a type, not JSON.
- **Option order is carried by the array, not the object.** `criteria` for a
  `choice` arrives as an object in the SDK's typed form but its iteration order
  is the answer space; `nlohmann::ordered_json` preserves it, `nlohmann::json`
  does not.

**Bounds the ecosystem actually imposes** (llmgateway's `/v1/systemone`
reference, 2026-09-20 — confirm both against the vendored `models.py` in Phase 1
and let the vendored file win if they differ): `choice` takes **1–255** options,
`score` takes **2–10** ordered levels. The 255 is the same number Laya's
`k / 255.0` act feature is calibrated against (`laya/common.py:346`), which is a
coincidence of two independent designs that happen to agree — worth a comment in
the code so nobody "fixes" one toward the other.

**Unknown kinds are dropped, not fatal, in both directions** — and the two
directions are different mechanisms. The *client* deletes an answer whose `type`
is outside `{noul, choice, score}` and logs a warning (verified in the SDK's
decoder: a pre-pass over `answers` that drops and warns before Pydantic
validation). The *server* drops a **question** whose `type` it does not
implement, also with a warning. Forward compatibility in this ecosystem is
defined as "ignore what you do not know, do not fail the batch".

## Fixed decisions

These are settled. An implementer does not re-open them.

| # | Decision | Rationale |
|---|---|---|
| 1 | **Wire format is TypeSafe `/v1/systemone`, byte-compatible.** | It is the contract of a shipped ecosystem, not an OFLM invention. Verified 2026-09-29: **llmgateway** serves it natively (`POST https://api.llmgateway.io/v1/systemone`, shipped 2026-09-20); **`yijunyu/jev-rs`** serves it natively from `tools/laya`-style engines (`jev serve` exposes `/v1/systemone`, and the official Python/JS SDKs run against it unchanged via `TYPESAFE_BASE_URL`); **Mesh-LLM** ports it (`mesh-llm#2083`, `model_support/0006`); **OpenRouter** routes it under the `typesafe/` namespace. **LiteLLM is a pass-through, not a second implementation** — it forwards `/typesafe/v1/systemone` to `api.typesafe.ai` and returns the response unchanged (`v1.103.0-rc`), which strengthens the argument rather than weakening it: there is no second implementation free to drift from the schema, only pipes. Being a drop-in for that ecosystem is worth more than an OFLM-native route name. |
| 2 | **All four readouts are declared; only `scored_slot` is implemented.** | The four-way split is this plan's, not a copy: the TypeSafe wire format has exactly three question kinds (`noul`/`choice`/`score`), all scored-slot, and `letter_slot` / `masked_slot` / `rank_head` are placeholders for readout families the ecosystem has not standardised. Unimplemented ones refuse by name. |
| 3 | **`laya-multilingual` only.** | One RoPE base (160000 for both layer types), `tile_n=48` matches shipping families, default 1024 ctx. English is deferred — it forces `tile_n=16` and two thetas for no proof-of-surface. |
| 4 | **Head on the host.** | It is 8 GEMMs against the encoder's 88, and running it on the array in the same process is not implementable against today's `Encoder` — see "Why the head runs on the host". |
| 5 | **Dedicated process, no co-residency.** | `ShapeLease` makes the engine geometry process-wide, and two resident `hw_context` objects is explicitly unmeasured in this repo. |
| 6 | **Extend `open_npue` upstream; add `AutoDecisionModel` as a sibling seam.** | `SYNCED.md` says edit upstream, not the synced copy — see "Where the work happens", because Phases 2, 3, 5, 6 and 7 all land inside that synced directory. The abstraction that needs a new interface is the *app* seam, not the engine. |
| 7 | **`oflm pull` from a `model_list.json` entry.** | Consistent with the six existing encoders. |
| 8 | **S=1024, batch tiers 1,4,16,32,64.** | Laya's own default `max_len`. Tiers capped at 64 because host attention materialises `batch x heads x S^2 x 4` = 3.2 GB at 64. |
| 9 | **Performance is a separate project.** | Build for correctness. `tile_n` and the pad-vs-16 trade-off are re-opened only after this ships. |

## Where the work happens

**Phases 2, 3, 5, 6 and 7 land inside `src/open_npue/`, which is a synced copy.**
`src/open_npue/SYNCED.md:1-6` is unambiguous: *"This directory is a **copy**,
written by `tools/sync_openflowlm.py` in
[NpuEmbeddings](https://github.com/vegardberget/NpuEmbeddings). Edit it there,
not here: the accuracy gates that make these numbers mean anything live in that
repository, and a local edit here silently detaches this code from them."* The
directory also carries a per-file sha256 table that a local edit invalidates, and
upstream is Apache-2.0 while this copy is relicensed MIT — so a local edit is also
a licensing question, not only a hygiene one.

Two consequences for this plan:

- **Do the work upstream, in `vegardberget/NpuEmbeddings`, then re-sync.** This
  document describes the change; it is not the site of the change. `tools/
  sync_openflowlm.py` is **not in this tree** — it lives upstream — so the sync is
  a clone-and-run, not a build target here. Phase 9 item 6's note that T43/T44
  upstream are "the same work seen from upstream" is the same fact: this
  architecture is already being built there, and the right move is to add to that
  work rather than to fork it.
- **Fork-owned files are the exception and they are already named.**
  `src/open_npue_adapter/npue_embedding.cpp` is listed in `OPEN_NPUE_SOURCES`
  with the comment "Fork-owned, not synced: the ONLY translation unit that
  includes the engine's header" (`src/CMakeLists.txt:344-348`). The subdirectory
  keys in Phase 3 and the `1_Pooling/config.json` shim in Phase 9 go **there**,
  and can land without an upstream round-trip. Likewise `npu_offload/gemm_rtp/`,
  `src/server/`, `src/common/`, `src/include/` and `utilities/` are this tree's
  own.

If an upstream round-trip is not possible, say so in the phase report and record
which files were edited locally — silently detaching a synced directory is the one
outcome `SYNCED.md` exists to prevent, and the only way to prevent it here is to
write down when it happened.

## Why the head runs on the host

The head is arithmetically a BERT layer, and that is worth stating first
because it is true: its four GEMMs are `(768,2304) (768,768) (768,3072)
(3072,768)` and `design_fits(768, 3072, gated=false, qkv_n=2304)` returns true
against `BERT-h768-bfp16`, so **an NPU head would need no new xclbin**. It is
still not the right first shape, because "no new xclbin" is not "no new work",
and four things stand in the way that no amount of care in Phases 3 or 3
addresses:

1. **`g_ffn`, `g_layers` and `g_seq` are process-wide, written once.**
   `ShapeLease` calls `detail::apply_model_shape` (`npue_encoder.hpp:786-796`),
   which sets `g_layers = num_layers` and `g_ffn = intermediate` from **the
   container's single values** (`:552-561`). The encoder needs
   `intermediate=1152, layers=22`; the head needs `3072, 2`. There is no second
   door: the lease is the only writer and a second lease throws.
2. **`Encoder` hardcodes the encoder's tensor vocabulary.** `stage_all()` loops
   `g_layers` and names every tensor `"layer." + L + "."` (`:1586-1592`),
   unconditionally dereferences `<op>.bias` (`:1574`) and per-layer
   `ln1`/`ln2` (`:1622-1626`). The head's tensors are `head.layers.{H}.*`. There
   is no prefix parameter, no layer-count override, no ffn override, and no
   second LayerNorm site numbering — `layer_norm_cpu` indexes
   `h_gamma[site]`/`h_beta[site]` (`:1665`), which are pushed in lockstep with
   `s_ln`.
3. **A second `npu::Design` is a second `hw_context`.** Not a shared context
   with more instruction streams: `npu_device.hpp:219-222` — *"Gets its own
   hw_context … two `Design`s sharing one `Device` is how a second xclbin
   becomes resident beside the first."* Decision 5 exists because the cost of two
   resident contexts is unmeasured; putting the head on the array means shipping
   exactly that, in the process Decision 5 was supposed to keep simple. And
   `test_open_npue.ps1:88-92` records the sibling failure: two contexts on this
   NPU *"do not queue, they block."*
4. **One model resolves to one design set.** `find_artifacts(dir, info)`
   (`npue_embedding.cpp:185-219`) returns a single path from
   `npue_design_family`, and `Encoder`'s four `npu::Design&` are all bound from
   that one `Stack` (`:4198-4232`). "Two design sets, not one" needs a second
   model-list key and a second `Stack`, and the `design_fits` selection is
   per-geometry (`:4000-4018`) so the two sets genuinely cannot be merged.

Worth being explicit about what is **not** a problem, because it looks like one:
the `b_layout_hash` guard. `stage_all` (`:1556-1571`) compares each design's
`b_layout_hash` against the container tensor's `layout_hash`, and
`gemm_b_layout` only ever sees `(tile_k, tile_n, "BF16")`. Both families are
`-n 48` at the same `tile_k`, so the two sets **share one `b_layout_hash` byte
for byte** and the guard passes for both. The obstacles are the four above, not
this.

The cost is 2 layers × 4 GEMMs against the encoder's 22 × 4 — **8 of 96 GEMMs,
8.3 % of the arithmetic** — and Phase 7's gate is argmax agreement, not latency,
so nothing observable is lost. Host is also strictly *easier* to make
bit-reproducible, which matters given the attention-saturation finding recorded
in `not_implemented`.

**The NPU head is a real follow-on, not a fantasy.** When someone wants it, the
complete list is: (a) a per-`Encoder` geometry override replacing the four
globals with members, threaded through `run_preln`, `stage_all`,
`swiglu_cpu` and `pool_rows`; (b) a `prefix` + `first_layer` parameter on
`stage_all`; (c) a `head_design_family` key in `model_list.json` and a second
`find_artifacts` call; (d) an acceptance measurement that two resident
`hw_context`s on one device neither block nor thrash. Gate it on that
measurement, not on a FLOP argument.

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

**Arithmetically identical to `BERT-h768-bfp16`.** `design_fits(768, 3072,
gated=false, qkv_n=2304)` returns true against the shipping set, so an NPU head
would need no new xclbin. That is a fact about xclbins, not about work — see
"Why the head runs on the host" for the four obstacles, which are process-wide
geometry, a hardcoded tensor prefix, a second `hw_context`, and a
one-design-set-per-model lookup.

The head tail is host-only and must stay there — it is not array-tileable:

| tensor | shape | why not on the array |
|---|---|---|
| `scorer.1` | [768,768] | could go, but see below |
| `scorer.3` | [1,768] | N=1 fails `N % (tile_n*8) == 0` for every tile |
| `act_head.0` | [256,772] | K=772 fails `K % 64 == 0` |
| `act_head.2` | [n_act,256] | N=2 for this checkpoint, same failure |

`scorer.1` is kept on the host with the rest of the tail: it runs on K marker
positions (≤20), not on rows, so there is nothing to tile. Host it.

## Prompt format — exact, from `laya/common.py`

```
[CLS] "<type> question: <instructions>" [SEP] [MASK] opt0 [MASK] opt1 … [SEP] <state> [SEP]
 ^1                                                        ^4                    ^1
```

(`[MASK]` above is upstream's own notation from the `build_sequence` docstring,
a placeholder for **the mask token id 4** — whose literal string is `<mask>`, see
below. The `^4` marks that id, not a four-character string.)

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

- The mask token is a **standalone token** at `markers[i]` — id 4, inserted by
  hand, never encoded from text. The option text follows it as a separate
  encode.
- Every encode is `add_special_tokens=False`. Specials are inserted by hand.
- Option order is semantic. For `score` the order *is* the scale.
- `str(k)` when a choice label has no description; `"%s: %s" % (k, desc)` when it does.
- `score` options render as `"level %d: %s" % (i, crit)`.
- `noul` renders exactly two options, semantic order `[false, true]`:
  - `f"{false_label}: " + (render_criterion(c) if c not in (None, "") else "no, the statement does not hold")`
  - `f"{true_label}: " + (render_criterion(c) if c not in (None, "") else "yes, the statement holds")`

  **`not in (None, "")`, not `or`.** `0`, `False`, `0.0`, `[]` and `{}` are falsy
  but meaningful criterion values; `c or default` would silently replace a
  criterion of `0` with the English sentence. The `choice` branch's own comment
  in `render_options` calls out precisely this ("only None/`""` mean 'no
  description'; 0 and False are legitimate criterion values"), so the `noul`
  branch gets the same treatment.
- `noul`'s two labels come from `q["labels"]`, which must map exactly
  `{"false", "true"}` to two distinct non-empty strings (defaulted by
  `_DEFAULT_NOUL_LABELS` when absent). **`labels` is rejected outright on any
  other question type** — `render_options` opens with
  `if t != "noul" and "labels" in q: raise ValueError("labels is only supported
  for noul questions")`. A wire type that accepts `labels` on a `choice` is
  accepting something upstream refuses.
- **Mask-token sanitising:** `tok.mask_token` is replaced with a space in the
  instructions, in every option, and in the state. This is a prompt-injection
  guard and it is part of the format.
  **The literal is `<mask>`, not `[MASK]`.** Verified in
  `multilingual/tokenizer/tokenizer.json`'s `added_tokens`: the entry with
  `id: 4` has `"content": "<mask>"`, and the string `[MASK]` does not occur
  anywhere in the vocabulary. An implementer who sanitises `[MASK]` sanitises a
  string that cannot occur and leaves the real injection vector open.

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

Phase 7 ships the `state: string` path, and that path's gate is the string path
only. The structured path is a follow-on and is **not** required to ship.

## Tokenizer — the gating dependency

`multilingual/tokenizer/tokenizer.json` is **Metaspace** (`prepend_scheme:
"always"`, `replacement: "▁"`, `split: true`) over byte-level BPE with
`byte_fallback: true`, plus an NFC-free normaliser that does `Replace(" " →
"▁")`, and a `TemplateProcessing` post-processor of `<bos> $A <eos>`.

Three consequences that will silently corrupt every answer if missed:

1. **`prepend_scheme: "always"` inserts a leading `▁` on every call.** So
   encoding `" " + opt` produces a doubled space marker. Reproduce HF's exact
   per-segment call sequence; do not "normalise" the inputs.
2. **Added-token pre-matching must happen before Metaspace, and must be
   leftmost-longest.** Without it, 2783/4595 strings tokenised differently in
   the llama.cpp port — and any state or question containing a **newline** is
   affected. This checkpoint makes it worse than "newline" implies: it carries
   **249 added tokens**, of which **88 are strict prefixes of other added
   tokens** — `\n`×1..31, `\t`×1..~90, the `<2mass>`-style mass tokens,
   `[@BOS@]`, `<unused0..98>` at ids 7..105 and **`<unused99>` at id 255999**
   (non-contiguous). Shortest-match would turn a 4-space run into four
   single-space tokens. The runtime is already correct here —
   `tokenizer_bbpe.cpp:563` ("left to right, leftmost-longest") and `:583`
   ("`added_` is sorted longest-first") — and the generator already emits the
   table (`bbpe_tokenizer_gen.cpp:223-240`). Note all 249 added contents *also*
   appear in `model.vocab`, so the pre-match is not an id remap that could be
   skipped for "already in vocab" entries: it changes the segmentation.
3. **The blob's derived special ids are wrong for this model — take them from
   the config.** `generate_bbpe_tokenizer_table` derives
   `cls_id = named({"[CLS]","<s>",...})` and `sep_id = named({"[SEP]","</s>"})`
   (`:257-258`). This vocabulary has `<s>` at **204** and `</s>` at **213**, so
   the blob will say `cls_id=204, sep_id=213`. The encoder config says
   `cls_token_id: 1, sep_token_id: 1` (both `<eos>`), plus `pad=0`, `unk=3`,
   `mask=4` — of which the blob gets only the last three right. **Read all five
   from the encoder config, never from the blob, and throw if they disagree.**
   Likewise the post-processor: the blob will carry `<bos>`=2 / `<eos>`=1 as
   prefix/suffix ids, and every Laya encode passes `add_special_tokens=False`,
   so **the runtime must not apply them** even though the blob has them.


---

# Phases

Each phase has an exit gate that can fail. Do not start a phase until the
previous gate passes.

## Phase 0 — Fail closed on an unrecognised `model_type`

**This is two separable things and the plan keeps them separable.** Part (a) is
a 25-line fail-closed guard in the packer. Part (b) is the five requirement
blocks from "New requirements", appended to `specs/open-engine/spec.md`. They
share a number because they share a morning, not because one depends on the
other: **(a) can and should land on its own**, and if the spec text is still being
argued about, that is not a reason to hold the guard back. The gate below is for
(a); (b) is reviewed as prose.

### Part (a) — the guard

**Problem.** `prepare_model_auto` resolves the architecture from one line:

```cpp
const std::string model_type =
    json_string_field(opt.checkpoint_dir + "/config.json", "model_type");
```

`json_string_field` (`npue_pack.cpp:1909-1916`) returns `std::string()` — the
**empty string** — when the file cannot be opened, cannot be parsed, has no
`model_type`, or has a non-string one. It does not throw.

Laya ships **no `config.json` at the repo root or under `multilingual/`** — the
real tree is `multilingual/{encoder/config.json, model.safetensors,
tokenizer/tokenizer.json}` (verified against the Hub). So for a Laya checkpoint
`model_type` is `""`, the dispatch falls through to the BERT **last branch**,
and the first thing that branch does is `slurp(checkpoint_dir + "/config.json")`
and `slurp(checkpoint_dir + "/vocab.txt")` (`npue_pack.cpp:2043-2044`) — both of
which throw `cannot open …`.

So a Laya checkpoint never produces a valid arch-0 container: it produces a
`cannot open …/config.json`. The two failure modes are different, and both are
`model_type` bugs, so they need two different guards. A missing config is the
one Laya actually hits; the `modernbert` arm is the one that stops a
valid-looking arch-0 container for a checkpoint that *does* put its config at
the root and names an architecture no packer handles.

**Change.** Two guards, inserted after `npue_pack.cpp:1982` and before the
`gemma3_text` branch:

```cpp
// A missing, unparseable or typeless config is NOT an empty model_type to
// dispatch on. Laya (and any repo that nests its config) lands here, and
// would otherwise be packed by the BERT branch's LAST arm -- which then dies
// in slurp() on a file the checkpoint does not have, in a directory it does
// not name. Name the file we looked for.
if (model_type.empty()) {
  throw std::runtime_error(
      "no usable \"model_type\" in " + opt.checkpoint_dir + "/config.json. "
      "That file is missing, unparseable, or has no string \"model_type\"; "
      "this packer reads the config from the checkpoint root and refuses to "
      "guess a subdirectory. If the checkpoint nests its config (e.g. "
      "<dir>/encoder/config.json), point this at the directory that holds "
      "config.json, model.safetensors and any 1_Pooling/config.json -- or "
      "set \"npue_checkpoint_subdir\" in the model entry. See "
      "specs/open-engine/plans/laya-decision-encoder.md.");
}
if (model_type == "modernbert") {
  throw std::runtime_error(
      "modernbert_rope_geglu is not packed by this build. The BERT fallback "
      "below would emit a valid-looking arch=0 container for a GeGLU/RoPE "
      "model; refusing rather than writing the wrong answer. See "
      "npu_offload/gemm_rtp/npue.py:106-141 for the architecture and "
      "specs/open-engine/plans/laya-decision-encoder.md for the port.");
}
```

The subdirectory problem is Phase 3's, and it needs a key. See "Workload
handling" there. It is named in the message because the two failures are the
same user-visible symptom.

**Gate (a).** Packing any checkpoint with no root `config.json` raises an error
naming that path; packing one with `model_type: "modernbert"` raises an error
naming `model_type`. `all-minilm`, `bge-*`, `nomic`, `gte`, `embeddinggemma`
all still pack and still produce **byte-identical** containers — the guard only
touches inputs that already failed. Re-run `utilities/test_open_npue.ps1`: six
cases (`:51-58`), all six must pass. That script needs PowerShell, which is not
installed on a Linux dev host; on Linux, run `export_gemm_rtp.py`'s pack path
directly and diff the container against a known-good one, or defer (a) to a host
with `pwsh`. **Do not skip the gate by asserting it.**

### Part (b) — the spec text

Append the five blocks from "New requirements" to
`specs/open-engine/spec.md`, after `OPEN-DECODE-PIPELINE` and before the
`## What it cannot reach.` closer, in the order arch, wire, readout, head,
accuracy. Copy them verbatim — the `**Applies to:**` file lists and the
`**Tests:**` paths are the parts a reviewer checks against reality, so an
approximate path is a defect, not a typo.

**Gate (b).** For each of the five: the ID is unique in the file; the block has
all three metadata lines in the file's order; every path in `**Applies to:**`
and `**Tests:**` exists or is named in this plan's file manifest; and the five
`**Tests:**` files exist under `specs/open-engine/tests/`, each opening with
`# Traces: <its IDs> (canonical spec: specs/open-engine/spec.md)`. That last one
is a real gate, not a formality: the file's convention is that a requirement is
reachable from a test, and a requirement whose test does not exist is a
statement nobody can check.

## Phase 1 — The readout and plan layer (pure host, no NPU)

**This phase needs no NPU, no model, and no hardware, and that is why it is
numbered 1.** It is pure data transformation, and it carries the plan's largest
stated unknown — whether the wire format is right. Every later phase is
mechanical by comparison, and the phases that can fail expensively — packer,
design set, hardware — all come after it, so the cheapest thing that can be
wrong is found first.

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
  // choice: (label, description) in the caller's order, description may be empty
  //        -- this is upstream's `crit` dict for a choice.
  // score:  the ordered levels, index 0 == 0. Order IS the scale.
  // noul:   two entries, always in semantic order [false, true], description
  //        optional and independently defaulted per side.
  std::vector<std::pair<std::string,std::string>> options;
  // noul ONLY. The caller's own false/true wording. Empty means "use
  // _DEFAULT_NOUL_LABELS". Supplying it for choice or score is an error, because
  // upstream raises ValueError on exactly that.
  std::vector<std::string> noul_labels;   // 0, or 2, or an error
};

struct decision_answer {
  decision_kind kind;
  float              noul;        // noul only
  std::string        choice;      // choice only: the argmax label
  float              score;       // score only: sum(i * p_i)
  // Length is k, this row's marker count -- NOT the batch's kmax, and NOT the
  // number of options if the head budget truncated one. Serialised as
  // {label: p} for choice (keyed off `options`) and {index: p} for score.
  std::vector<float> probabilities;
  float              confidence = 0.f; // choice, score only. NEVER set for noul.
  std::vector<float> logits;      // raw, length k
};
```

**Rules, all enforced with a named error:**

- `noul` has **no** `confidence` field in the response. (`1 - H(p)/ln 2`
  degenerates to 1.0, so it would be a constant lie.) Serialising it is an error.
- `choice`/`score` require **≥ 2** options. `score` options are **ordered** and
  the order is the scale, and the ecosystem caps `score` at **10** levels and
  `choice` at **255** — refuse above those, by name, so a request the ecosystem
  would reject is refused here rather than answered with a wrong-shaped body.
- `choice` supports up to **255** options. Above that, refuse by name. The cap is
  upstream's, not ours: the `act_head` feature vector carries `k / 255.0`
  (`laya/common.py:346`), so `k` is calibrated against 255 and a 256th option
  pushes that feature outside the range it was fitted on.
- `noul` takes exactly 2 options and 0 or 2 `noul_labels`; anything else is an
  error, matching `_resolve_noul_labels`'s `set(labels) != {"false","true"}`
  check.
- **Question order is semantic, and so is option order.** `nlohmann::json` is a
  `std::map`, so it sorts keys alphabetically — which silently reorders the
  questions in a request and, worse, a choice's options among themselves, where
  order is the answer space. Use `nlohmann::ordered_json` for anything touching
  the request or the response, and carry option order in an array, never in an
  object.

- An unknown `type` is **dropped with a warning, not fatal** — forward
  compatibility is an explicit requirement of the SDK behaviour.
- `output_tokens` is **always 0**. It is the definitional property of the class.
- `state` as a list of content parts: refuse by name, `"'state' as a list of
  content parts is not supported"`.

**Confidence — a divergence, and the schema does not settle it.** Laya ships
**two** confidences (`laya/common.py:470-496`): `answer_confidence = max(p[:k])`,
which is the one temperature scaling is fitted to and the one every calibration
figure upstream is computed on, and `confidence_from_probs = 1 - H(p)/ln k`,
which its own docstring says "carries no such guarantee" and "must not be
compared against the same threshold". Engines in this ecosystem differ: jev-rs
describes its `confidence` as "TypeSafe's formula" without saying which, and
llama.cpp #29321 emits the entropy form.

**Neither does TypeSafe.** The vendored schema types `confidence` as a plain
0–1 float with no formula; the SDK's field doc says only "how certain the model
is in this answer". The *blog* says answers carry "calibrated probabilities and
confidence scores" — which is a claim about the product, not a definition.

So this is a real choice, and the plan makes it: **emit `max(p[:k])` after
temperature**, because it is the calibrated quantity and because a
`confidence` field that callers will threshold on should be the one that means
what its name says. The raw `logits` and `probabilities` are both in the
response, so a caller that wants the entropy form can compute it in one line.
**Record the divergence in the README**, and record it in the container too —
this is the single most likely place for a caller to find a number that differs
from the hosted API's.

**Temperature.** One value for the whole request — a per-question temperature is
an error (`"one value for the whole request, not per question"`). **The
TypeSafe request schema has no temperature field at all**: it is `{state, model,
questions}`, so a per-question or ad-hoc temperature is not merely unsupported,
it has nowhere to live on the wire. That is the real reason for the rule.

Applied on the host after the forward pass, from the **container's**
`temperature` and `temperature_by_options` (Phase 3), not from a file: the
bucket key is `"%s:%s" % (QTYPE_NAMES[qtype], size)` with `size` in
`2 / 3-5 / 6-10 / 11+` (`laya/common.py:499-501`), buckets take precedence over
the per-qtype 3-vector, and the result is clamped to `[0.5, 5.0]`
(`TEMP_MIN`/`TEMP_MAX`, `laya/common.py:508-509`). A bucket key that is not one
of those four sizes is **ignored with a warning**, not an error: a new upstream
bucket size should not take the model down.

**`readout` derivation.** From the container, exactly as llama.cpp derives it
from GGUF metadata: a `scored_slot` is an arch with a decision head, a marker
token, and non-causal attention. A model whose `readout` is not
`DECISION_READOUT_SCORED_SLOT` **refuses by name** when this build is asked to
serve it.

**Gate.** A golden JSON fixture — the request and the exact expected response,
plus a synthetic logit tensor — round-trips **byte-for-byte**, with
`ordered_json` preserving question and option order. Generate the expected
response from the **vendored** `typesafe-systemone-0ffd094c.py`, including the
envelope, and assert:

- the full body is `{model, answers, usage}` — not the bare answers;
- `usage.output_tokens` is `0` and `usage.input_tokens` is the request's;
- `noul` → `{type, noul}` with **no** `confidence` key present at all;
- `choice` → `{type, choice, confidence, probabilities:{label:p}}`;
- `score` → `{type, score, confidence, legend:{"0":…}, probabilities:{"0":…}}`
  with **string** index keys;
- a request naming a different `model` is refused, and the refusal is not a
  200 with somebody else's answers in it.

Unit tests in `src/common/AutoDecisionModel/` for every refusal above, and one
that re-reads the vendored file and fails if the fixture and the vendored models
disagree — that is what stops the pin from going stale silently.

## Phase 2 — Tokenizer: Metaspace byte-BPE

**State.** `src/open_npue/tokenizer_bbpe.{hpp,cpp}` and
`bbpe_tokenizer_gen.cpp` are written, generated, compiled and linked
(`src/CMakeLists.txt:338-339`, both objects present in `build/src/CMakeFiles/`).
**Nothing calls them.** `BbpeTokenizer`, `BbpeEncoded` and
`generate_bbpe_tokenizer_table` have zero call sites; `nm -C build/src/oflm`
shows the symbols linked in as global text with no references. The sibling
Gemma and XLM-R generators *are* called (`npue_pack.cpp:1006`, `:1724`), which
is the proof the omission is accidental, not a decision.

**The generator refuses this checkpoint at four separate gates, not one.**
`generate_bbpe_tokenizer_table` is written to fail closed and the multilingual
`tokenizer.json` trips every check that encodes an assumption about a
GPT-2/RoBERTa-shaped byte-level BPE. Read them off the file, not off the prose:

| gate | file says | code | what "ByteLevel BPE" assumed |
|---|---|---|---|
| normalizer | `{"type":"Replace","pattern":{"String":" "},"content":"▁"}` | `:100-110` accepts only `null` or `NFC` | the vocab has **no raw space** — `' '` is not a vocabulary entry, so space *must* be rewritten or nothing tokenises |
| pre_tokenizer | `Metaspace` | `:113-121` requires `ByteLevel` | the GPT-2 regex is a different segmenter and needs its own scanner |
| `model.byte_fallback` | `true` | `:161-166` refuses true | a byte-level model "needs no fallback" — true of a *closed* byte alphabet, not of a Metaspace one |
| `model.unk_token` | `"<unk>"` | `:170-174` refuses any `unk_token` | a byte-level model "has no unknown pieces" — this one has `<unk>`=3 and `fuse_unk: true` |

Two more that this checkpoint *passes*, so do not spend time on them: the
merges are `["a","b"]` pairs rather than `"a b"` strings, which `:311-333`
already reads in both forms; and all 256 GPT-2 byte characters are present in
the vocabulary, which `:298-314` requires.

**Change.**

1. `bbpe_tokenizer_gen.cpp`:
   - **Normaliser.** Add a `Replace(String→String)` mode. It is not a
     general regex engine and must not become one: accept *only* a
     `{"String": s}` pattern, store `norm = 2` plus the replacement string in
     the blob, and throw on any other pattern type. The value of `s` is
     `" "` and the content is `"▁"`.
   - **Metaspace pre-tokeniser.** Accept `Metaspace` as a second
     pre-tokeniser kind; store `prepend_scheme` (an enum: `always` /
     `never` / `first`), `replacement` and `split` alongside the existing
     `add_prefix_space`. Anything else still throws, with the same "this is a
     DIFFERENT regex and needs its own scanner" reasoning.
   - **Bump the blob to version 2.** `BBPETOK1` currently writes
     `norm, add_prefix_space, vocab, merges, added, cls, sep, pad, unk, mask,
     prefix, suffix` (`bbpe_tokenizer_gen.cpp:317-337`). Version 2 appends
     `norm_replacement` (string), `prepend_scheme` (u32), `replacement`
     (string), `split` (u32). **`tokenizer_bbpe.cpp` must keep reading v1** —
     the shipped `BERT-h384-bfp16`/`bge-base` containers embed v1 blobs and
     changing the reader is a breaking change to a validated set. Version is
     read at `tokenizer_bbpe.cpp:314-321`; branch there and throw a named error
     on a version this build does not implement.
   - **`byte_fallback`.** The check at `:161-166` is correct for a closed byte
     alphabet and wrong here. Replace the refusal with a **recorded** flag in
     the blob and have the runtime honour it — for this vocabulary it is
     inert in the common case, because all 256 byte characters *are* present,
     so the fallback never fires; it exists for characters the Metaspace
     segmenter can emit that have no merge. Do not simply delete the check:
     that is how a checkpoint that *does* need it ships wrong.
   - **`unk_token`.** Record `unk_id` (already emitted) and let the runtime use
     it. `tokenizer_bbpe.cpp:435` currently says "there is no `<unk>` in this
     family to substitute" — that arm needs a real implementation, and Phase
     1's fixture must include a string that actually hits it, or the arm ships
     unexercised. If you cannot produce a hitting string in the fixture, say so
     in the phase report and keep the refusal.
2. `tokenizer_bbpe.cpp`: the Metaspace scan and the `Replace` normaliser, in
   that order — normalise, then match added tokens against the *normalised*
   text but emit the added token's own id, then replace every remaining space
   with `replacement` and prepend `replacement` per `prepend_scheme`. Keep
   the existing leftmost-longest added-token stage exactly as it is
   (`:503`, `:563`, `:583`); it is already right and re-deriving it is the
   mistake.
3. `npue_encoder.hpp`'s `load_tokenizer` (`:374-418`): add the
   `BbpeTokenizer` construction under the arch=4 arm, mirroring `:388`'s
   construction of `XlmrTokenizer`. **Read `cls`/`sep`/`pad`/`unk`/`mask` from
   the encoder `config.json`, not from the blob** — see the tokenizer section's
   point 3 — and refuse if the blob disagrees.
4. **Do not touch `encoder_implemented` (`:301-314`) in this phase.** Adding
   `modernbert_rope_geglu` before the packer exists would make Phase 0's guard
   unreachable in the one path that matters, because the guard is a *packer*-side
   refusal and the runtime-side list is a separate switch. The arch string is
   registered in Phase 3, in the same change that adds the packer.
5. Tokenise **per segment with `add_special_tokens=false`**, specials inserted
   by hand. Never one pass over the assembled string, and never apply the
   blob's post-processor prefix/suffix.

**Gate.** For a fixture of ≥200 strings (must include: strings with newlines,
**runs of 2..31 newlines and 2..90 tabs** — the added-token prefix chains;
`single_word`/`rstrip` boundary cases; HTML tags; `<unusedN>` tokens including
`<unused99>`; non-Latin scripts; emoji; the literal `<mask>`; the literal
`[MASK]`, which must tokenise as ordinary text; and at least one string that
reaches `<unk>` if the arm is implemented), the C++ tokenizer's ids are
**exactly equal** to HF `tokenizer(text, add_special_tokens=False)["input_ids"]`.
Any single divergence fails the phase. Generate the fixture **from
`/resolve/main/`**, not `/raw/main/` — the `raw` path returns a 133-byte
git-lfs pointer for this file (verified), so a fixture generated from it is
testing the pointer. The generated unicode tables are already measured against
HF's splitter, so this is a wiring check, not a re-derivation.

## Phase 3 — The packer

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

**Register the arch in the same change.** `prepare_model_auto` grows a
`model_type == "modernbert"` branch that calls this, and
`encoder_implemented` (`npue_encoder.hpp:301-314`) grows
`arch == "modernbert_rope_geglu"`. Both or neither: a container the packer can
write but the runtime refuses is a wasted pack, and one the runtime accepts
without a packer to produce it is the arch=0 fail-open this plan exists to close.
Pass `max_seq = 1024`, which is what the Phase 4 design is compiled at and what
`set_design_seq` will be handed (`:814-823`).

**Workload handling — Laya nests its config, and nothing in the current code
path can reach it.** Verified Hub tree for `convaiinnovations/laya`:

```
encoder/{config.json, …}   model.safetensors   rl_agent_config.json   tokenizer/tokenizer.json
multilingual/encoder/config.json   multilingual/model.safetensors
multilingual/rl_agent_config.json  multilingual/tokenizer/tokenizer.json
typed-decisions/{encoder/config.json, model.safetensors, …}
```

So with `checkpoint_dir = <served>/multilingual`: `config.json` is at
`encoder/config.json`, the weights are at `model.safetensors` (in the same
directory — this one *is* convenient), and the tokenizer is at
`tokenizer/tokenizer.json`. **No path in the repo can express that today:**

- `prepare_model_auto` reads `<dir>/config.json` and `<dir>/vocab.txt` and
  `<dir>/model.safetensors` and `<dir>/1_Pooling/config.json` — all flat
  (`npue_pack.cpp:1982`, `:2033`, `:2043-2044`, `:1949-1954`). Laya has no
  `vocab.txt` at all, which is fine: arch=4 is a BBPE-tokened container whose
  blob is embedded, exactly as arch=2 and arch=3 embed `tokenizer.xlmr_table`.
- `npue_embedding.cpp:140-150` — the served-model pack path — hardcodes
  `po.checkpoint_dir = dir` and refuses unless `dir/config.json` is a regular
  file. It has no subdirectory parameter.

**So add one, mirroring the key that is already there.** `npue_container`
(`npue_embedding.cpp:112-135`) already exists to name a file in a
heterogeneous model directory; `npue_checkpoint_subdir` is the same idea for a
nested checkpoint. Concretely:

- `PrepareOptions` gains `std::string config_subdir;` and
  `std::string tokenizer_subdir;` (default `""` and `"tokenizer"` — the latter
  matching the shape the two BBPE/XLM-R containers use, and checkable against
  the real layout).
- `prepare_model_auto` resolves `config_subdir + "/config.json"`,
  `tokenizer_subdir + "/tokenizer.json"`, `1_Pooling/config.json` and
  `model.safetensors` through them, and `prepare_model_modernbert` does the
  same. It must **not** silently ignore an empty `model_type` because a
  subdirectory was set — if `config_subdir` is non-empty and the file is still
  not there, Phase 0's error fires with the composed path.
- `npue_embedding.cpp` reads `npue_checkpoint_subdir` (and
  `npue_tokenizer_subdir`) from the model entry into those fields, exactly as
  it reads `npue_tile_n` at `:172-173` and `npue_container` at `:112`.
- The model entry's `url` is the repo root, so `oflm pull`'s `files` list uses
  **repo-root-relative** paths with the subdirectory in them
  (`multilingual/encoder/config.json`). The downloader already handles nested
  relative paths — `1_Pooling/config.json` is an entry in six shipped models
  (`model_list.json:1160-1166`) — so **nothing needs extending in
  `model_downloader.cpp`**: `files` is an explicit list of relative paths and
  there is no allow-list concept to add.

**`resolve_pooling` is satisfied by writing a file, not by relaxing it.**
`resolve_pooling` (`npue_pack.cpp:1949-1966`) is a deliberate name-checked
contract and stays as it is. It reads `<dir>/1_Pooling/config.json`; see Phase 9
item 1, which writes one — and note there that what it says is a fiction,
because Laya does not mean-pool.

**Container config keys to emit**, in this order (the packers build raw JSON by
string append and the key order must be stable). **These names are not
free choice: `apply_model_shape` reads these exact keys
(`npue_encoder.hpp:552-561`), and a misspelled or missing one is not a warning
— it is `throw "the .npue reports a non-positive shape"` (`:742`) or a
zero-shaped geometry that fails somewhere less obvious.**

```
{"arch":"modernbert_rope_geglu",
 "num_layers":22,               NOT "layers"
 "hidden":768,
 "num_heads":12,                 NOT "heads"
 "head_dim":64,
 "intermediate":1152,
 "max_seq_len":1024,             NOT "max_len" -- set_design_seq() REFUSES seq > max_seq_len
 "source_repo":"…",              read by apply_model_shape at :557
 "gated_ffn":true,"qkv_n":2304,
 "pre_layernorm":true,"final_norm":true,
 "layer_types":[...22 entries...],
 "rope_theta":160000,
 "identity_attn_norm_layer0":true,
 "sliding_window":64,"global_attn_every_n_layers":3,
 "swiglu_halves":"gate|up",
 "activation":"gelu",
 "position_embedding_type":"rope",
 "vocab_size":256000,
 "layer_norm_eps":1e-5,
 "pooling":"mean","l2_normalize":false,
 "head_layers":2,"head_intermediate":3072,
 "n_qtype":3,"marker_token_id":4,
 "max_len":1024,"head_max_len":256,
 "temperature":[<t_choice>,<t_score>,<t_noul>],
 "temperature_by_options":{…},
 "tile_k":64,"tile_n":48,
 "fusions":{…},
 "not_implemented":[…]}
```

Four notes on that list:

- **`layer_norm_eps`, not `norm_eps`/`norm_eps_layer`.** The gte packer emits
  `layer_norm_eps` (`npue_pack.cpp:1758`) and nothing reads `norm_eps`.
  Emitting a key the runtime ignores while the runtime uses a **hardcoded
  `1e-12f`** in `layer_norm_cpu` (`npue_encoder.hpp:1685`) is a false claim in
  the container: it would say 1e-5 and compute 1e-12. Two choices, both
  acceptable, pick one and say which in the phase report: (a) emit
  `layer_norm_eps: 1e-5` for provenance and add a Phase 5 note that the host
  norm uses 1e-12, or (b) make `layer_norm_cpu` read the container's value.
  (a) is what this plan assumes; the delta is ~5e-6 relative, far under the
  bf16 operand's 8-bit mantissa, and Phase 5's gate is a bf16 comparison.
- **`rope_theta` is provenance, not the mechanism.** The checkpoint carries
  thetas as `rope_parameters.{full_attention,sliding_attention}.rope_theta`
  (both 160000, verified) — a *nested* key, so the packer must read it from
  there, not from a top-level `rope_theta` that does not exist in this config.
  Phase 5 builds **one** table from it.
- **`"pooling":"mean"` is a fiction and `l2_normalize:false` is the honest
  half.** Laya pools by `gather` at `marker_pos`, not mean and not CLS — see
  the research doc, and `laya/common.py:327-330`. `pool_rows` accepts only
  `cls`/`mean` and throws otherwise (`npue_encoder.hpp:719-723`), so the
  decision engine must **never call `pool_rows`**. Emit `mean` only because
  `apply_model_shape` demands one of the two, and record in `not_implemented`
  that pooling for this model is a marker gather, not a mean.
  `l2_normalize: false` is correct and load-bearing: `pool_rows`'s default is
  `true` (`:740`) and Laya never L2-normalises.
- **`fusions`** — copy arch=3's verbatim (`npue_pack.cpp:1792-1799`), minus
  nothing; the deltas for arch=4 are the keys above.
- **`temperature` / `temperature_by_options` are packed, not read at run time.**
  Copy the 3-vector from `rl_agent_config.json` in `QTYPES` order
  (`{choice:0, score:1, noul:2}` — `laya/common.py:17`) and the bucket map
  verbatim. They are 3 floats and a small object, so they go in the config as
  data rather than as a tensor. **The packer still needs the file on disk to
  read them**, so `rl_agent_config.json` is in the model entry's `files`
  (Phase 9) — but once packed, the container is self-contained and a later
  upstream temperature change does not silently apply to an already-served
  model. `temperature_by_options` is `{}` in this checkpoint; the key is emitted
  regardless, because a future release may not be, and "the key is missing" and
  "the map is empty" must not mean the same thing to a reader.
  The dead `temperature` *buffer* in the checkpoint (registered, never read by
  `forward`) is deliberately **not** packed as a tensor: it is the same three
  numbers, and packing it twice is how a container ends up with two sources of
  truth for one value.

**Tensors.** Emit these, and **no others**. Every `*.bias` and
`embeddings.position` / `embeddings.token_type` is **zero-filled** — the
runtime dereferences all of them unconditionally (`npue_encoder.hpp:4559-4562`,
`:4682-4684`), and a zero tensor of the right shape is exact because the
embedding build is `dst[c] = wv[c] + pv[c] + w_typ[c]` (`:4560-4561`).

```
embeddings.word          [256000, 768]  F32   <- the NAME and DTYPE are both
                                                load-bearing; see below
embeddings.position      [1024, 768]    F32   ZERO   (max_seq_len rows)
embeddings.token_type    [1, 768]       F32   ZERO
embeddings.ln.weight     [768]          F32
embeddings.ln.bias       [768]          F32   ZERO
final_norm.weight        [768]          F32
final_norm.bias          [768]          F32   ZERO
layer.{L}.ln1.weight     [768]          F32     L in 1..21
layer.{L}.ln1.bias       [768]          F32     ZERO
layer.{L}.ln2.weight     [768]          F32     L in 0..21
layer.{L}.ln2.bias       [768]          F32     ZERO
layer.{L}.qkv.weight      [2304, 768]   gemm_b
layer.{L}.qkv.bias        [2304]        F32     ZERO
layer.{L}.attn_out.weight [768, 768]    gemm_b
layer.{L}.attn_out.bias   [768]         F32     ZERO
layer.{L}.ffn_up.weight   [2304, 768]   gemm_b   concat[gate, up]
layer.{L}.ffn_up.bias     [2304]        F32     ZERO
layer.{L}.ffn_down.weight [768, 1152]   gemm_b
layer.{L}.ffn_down.bias   [768]         F32     ZERO
```

**`embeddings.word`, in F32. Not `embeddings.tok_embeddings`, not BF16.** All
three existing packers emit `embeddings.word` as **F32**
(`npue_pack.cpp:645`, `:1370`, `:1815`), and the runtime reads exactly that
name and casts it: `model_.raw("embeddings.word").as<float>()`
(`npue_encoder.hpp:4682`, `:4559`). The checkpoint's own key is
`embeddings.tok_embeddings`, which is the trap — the packer **renames on
purpose**, and a packer that emits the checkpoint's key name loads nothing.
BF16 is not an option either: `.as<float>()` on a BF16 tensor reinterprets
bytes. It is also the wrong economy — 786 MB in F32 against 393 MB in BF16 for
a container this repo already mmaps whole, and the 256k-row table is read once
per row-gather.

**Layer 0 has no `ln1`** in the checkpoint. The tensor set stays uniform: emit a
zeros `layer.0.ln1.weight` / `.bias` and carry `identity_attn_norm_layer0: true`
so the runtime *skips the norm*, rather than normalising. A weight-1 norm still
centres and scales — this is why it is a config field and not a unit tensor.
Note `layer.0.ln1.bias` must be emitted too: `stage_all` dereferences
`<weight>` and `<bias>` for every layer unconditionally (`:1616-1617`).


**The one new helper is a row *permutation*, not a concat.** `mlp.Wi` is stored
fused `[2304, 768]` and is the model's **only** up-projection — there is no
second tensor to concatenate with. `add_gemm_b_concat2`
(`npue_pack.cpp:1111-1130`) takes two whole `Tensor`s `a` and `b` and lays them
out as `[a; b]` along N, so it cannot express "the same tensor, rows reordered".

What is wanted is: take `fused`, emit rows `[1152, 2304)` first and `[0, 1152)`
second, transpose to `[K, N]` the way `add_gemm_b` does, then tile. Name it for
what it does:

```cpp
// Emit `fused` (an [N, K] checkpoint tensor) with its row range [r0, r0+rows)
// moved ahead of the rest: layout is [fused[r0..r0+rows) ; fused[rest]] along N.
// ModernBERT stores mlp.Wi fused with the GATE half second (chunk(2,-1) ->
// input, gate), while the runtime computes lo * act(hi) over the packed order,
// so the gate half has to lead. The helper is a permutation, not a concat:
// ModernBERT has no second up-projection tensor to concatenate with.
static void add_gemm_b_reorder_rows(Writer &w, const std::string &name,
                                    const Tensor &fused, int64_t r0, int64_t rows,
                                    int64_t tk, int64_t tn,
                                    const std::string &layout_json,
                                    const std::string &layout_hash);
```

Call it as `add_gemm_b_reorder_rows(w, "layer.{L}.ffn_up", Wi, 1152, 1152, …)` —
`r0 = rows = intermediate`. Derive both from the config's `intermediate`, never
from a literal: `intermediate` is the one number a differently-pruned mmBERT
would change, and a hardcoded 1152 would silently pack the wrong halves in the
right order. Record the swap in `config["swiglu_halves"] = "gate|up"` so the
container says what it did rather than leaving the reader to infer it.


**Head tensors** are packed too — Phase 7 runs them **on the host** (Decision 4),
so these are plain F32 host weights, not `gemm_b`: no layout hash, no design, no
pre-tiling. They are still packed so that the container is
self-contained, so a re-pack is not needed when the NPU head lands, and so the
head's arithmetic is gated by the same parity check as the encoder's.

```
head.layers.{H}.norm1.{weight,bias}        [768]        F32   H in 0..1
head.layers.{H}.self_attn.in_proj.weight  [2304, 768]  F32
head.layers.{H}.self_attn.in_proj.bias    [2304]       F32
head.layers.{H}.self_attn.out_proj.weight [768, 768]   F32
head.layers.{H}.self_attn.out_proj.bias   [768]        F32
head.layers.{H}.norm2.{weight,bias}        [768]        F32
head.layers.{H}.linear1.weight            [3072, 768]  F32
head.layers.{H}.linear1.bias              [3072]       F32
head.layers.{H}.linear2.weight            [768, 3072]  F32
head.layers.{H}.linear2.bias              [768]        F32
type_emb.weight                           [3, 768]     F32
scorer.0.{weight,bias}                    [768]        F32
scorer.1.weight                           [768, 768]   F32
scorer.1.bias                             [768]        F32
scorer.3.weight                           [1, 768]     F32
scorer.3.bias                             [1]          F32
act_head.0.weight                         [256, 772]   F32
act_head.0.bias                           [256]        F32
act_head.2.weight                         [n_act, 256] F32
act_head.2.bias                           [n_act]      F32
```

`norm1`/`norm2` are the pre-LN norms of the `norm_first=True`
`nn.TransformerEncoderLayer`; `in_proj`/`out_proj`/`linear1`/`linear2` are
PyTorch's names for qkv/attn_out/ffn_up/ffn_down. `n_act` is
`len(rl_agent_config.json["act_costs"]) + 1` — **2 for this checkpoint**
(`act_costs: {"escalate": 0.5}`), not a constant. Read it; do not hardcode it,
or the packer breaks on the next Laya release.

`act_head.0`'s K is **d + 4 = 772** because `forward` concatenates four
hand-built features onto `h[:,0]` (`laya/common.py:346`). The plan's
host-only justification for the whole tail stands and is now the whole story:
`scorer.3` (N=1), `act_head.2` (N=2) and `act_head.0` (K=772) all fail
`N % (tile_n*8) == 0` or `K % 64 == 0` at every legal tile, and `scorer.1` runs
on K marker positions (≤20) rather than rows, so there is nothing to tile.

**`not_implemented` — be honest, in prose, in the container.** Minimum entries:
the 8192-token context against a seq=1024 design; `predict_long` windowing with
50 % overlap; the structured-`state` JSON path (Phase 7 ships strings only);
the Laya `Router`'s language routing; the `truncate_left` and `option_order`
arguments of `build_sequence`; that `pooling` says `mean` because the runtime
accepts only `cls`/`mean` and this model gathers at `marker_pos`; that
`cls_token_id`/`sep_token_id` come from the config and not the blob; and that
`act_head` is computed and discarded (Phase 7). Every one of these is a real
behavioural difference between this container and the Python package.

**Gate.** A synthetic `modernbert` fixture packs to a container that (a) passes
the byte-parity check against `npu_offload/gemm_rtp/npue.py`'s `Writer` for
every shared field, (b) round-trips **every** emitted tensor through
`npue::Reader::raw()` at the right shape and dtype, and (c) **loads through
`ShapeLease` and `apply_model_shape`** — that is, `load_tokenizer` +
`apply_model_shape` + `encoder_implemented` all accept it and `g_layers=22,
g_ffn=1152, g_max_positions=1024` come out. Gate (c) is the one that catches a
misspelled config key, and it is cheap; add it rather than discovering the
problem in Phase 5.
Note `tools/verify_pack_parity.py` lives upstream in `vegah/Npu-Embeddings` and
is **not vendored here** — parity must be re-established, not assumed. The
`b_layout_hash` half of (c) is deferred to Phase 4, because Phase 4 is what
builds the design whose hash it is: `gemm_b_layout(64, 48)` is the same function
for every `-n 48` family, so the container's `layout_hash` and Phase 4's
`b_layout_hash` are the same value by construction, and Phase 4's gate is where
that gets confirmed rather than assumed.

## Phase 4 — The design family

Add **two** families to `npu_offload/gemm_rtp/families.json` — the datapath is
decided by measurement in Phase 7, not by inheritance, and a decision you cannot
reverse without a rename needs both arms built up front:

```json
{
  "name": "BERT-h768-gated-i1152-bfp16",
  "serves": ["laya-decision:multilingual"],
  "note": "gated 768 with intermediate 1152, not the 3072 of BERT-h768-gated-bfp16. ffn_up is 2304. 48 is legal and matches every shipping family. Its qkv (768x2304) and attn_out (768x768) are byte-identical to BERT-h768-bfp16's, so the two share 8 of 16 cache markers and must never be built concurrently. DATAPATH IS A MEASUREMENT: this is the --emulate-bfp16 arm and the -bf16 arm below differ only in that flag, and Phase 7's gate picks one. bge-small failed its MTEB gate on bfp16 at -0.5010 bit-reproducibly and had to be rebuilt on plain bf16, so this is not a formality.",
  "args": ["--hidden","768","--intermediate","1152","--qkv-n","2304",
           "--gated-ffn","--emulate-bfp16","--c-bf16","-n","48",
           "--batches","1,4,16,32,64","--seq","1024"]
},
{
  "name": "BERT-h768-gated-i1152-bf16",
  "serves": [],
  "note": "The same geometry on the plain bf16 datapath: no --emulate-bfp16, everything else identical. It is a SEPARATE FAMILY rather than a flag override because check_design_sets.py keys on emulate_bfp16 (it is the one field that differs in design.json), and because both must be able to sit in src/xclbins at once for Phase 7 to A/B them. `serves` is empty until Phase 7 picks; whichever wins gets the tag and the other stays built for the next model of this shape.",
  "args": ["--hidden","768","--intermediate","1152","--qkv-n","2304",
           "--gated-ffn","--c-bf16","-n","48",
           "--batches","1,4,16,32,64","--seq","1024"]
}
```

**Two families, not one, and the names are load-bearing.** They differ in
exactly one flag, so a reviewer must be able to tell them apart at a glance in
`src/xclbins/`, and `check_design_sets.py` must be able to tell them apart in
`design.json`. `--emulate-bfp16` is the field it compares (`:57-58`), so reusing
one name for both would make the checker report the second build as a stale set
of the first.

**The second family is speculative and this plan says so, so nobody later
mistakes it for a requirement.** A build is 3–4 minutes, so building both is
cheap insurance against inheriting a datapath that fails its gate — but Phase 6's
measurement can still invalidate the whole approach (if host attention dominates
at S=1024, the design needs rethinking, not a different `emulate` flag). If that
happens, **the losing family is deleted, not kept "just in case"**: a design set
nobody selected is a maintenance cost and a future reader's puzzle. The gate that
selects one is Phase 7's, and until it runs, `serves: []` on both is the honest
state.

**`--batches` must move out of `common` and into every family's `args`,** and
this is a correctness requirement, not tidiness. A family cannot override
`common` today, because of how the two are joined:

- `build.ps1:93` runs `export_gemm_rtp.py @($f.args + $common)` — **family args
  first, `common` last.** argparse takes the *last* occurrence of a flag, so
  `common`'s `--batches 4,16,32,128` silently overrides a family's
  `1,4,16,32,64`. The build completes, prints `ok`, and produces a design with
  the wrong tiers.
- `check_design_sets.py:107` builds its expectation from the *same* order but
  `val()` reads the **first** occurrence (`:46`). So the checker would expect
  `1,4,16,32,64`, find `4,16,32,128`, and report a `MISMATCH` — whose own advice
  is "fix it there [families.json] and nowhere else", i.e. move the flag into
  `common` and invalidate all six existing models' tiers.

So the change is one edit plus five: delete `--batches` from `common`, and put
`"--batches","4,16,32,128"` explicitly in each of the five existing families'
`args`. The tier list then lives in exactly one place per family and a family can
set its own without a precedence fight. Add a `check_design_sets.py` case for a
family whose `--batches` differs from the others, so this failure mode cannot
return silently.

Two smaller things the gate must not overclaim. `check_design_sets.py`'s
`expected()`/`actual()` **never check `seq`**, so "`--seq 1024`" and the
20-stream count are unverified by the tool the gate names — assert them
separately by reading `design.json`'s `streams[]` and `tiers[]` directly. And
`gemm_pretiled.py` asserts four things, not two: `M % (m*rows) == 0` (`:156`),
`K % k == 0`, `N % (n*cols) == 0` (`:158`) **and** `m % r == 0 && k % s == 0 &&
n % t == 0` (`:159`, the MAC dims of the chosen kernel). Check all four.

**Build serially.** `build.ps1` loops families one at a time because `purge()`
deletes from the shared `~/.npu/cache` on content markers, and the two hidden-768
families own identical markers for 8 of 16 entries. Parallel builds have
corrupted each other's output before — and this family is a third hidden-768
set, so the hazard is now three-way.

**Gate.** `check_design_sets.py --xclbins src/xclbins` passes for all **seven**
families: `families.json` has five entries over four distinct names today, plus
the two above; the registry in `all_embedding_model.hpp:37-46` carries seven
*tags* of which six are Npue — the seventh is `embed-gemma:300m`, which is served
by `OpenGemma` and has no design family. Each new `design.json` carries
`streams[]` for 4 shapes × 5 tiers = 20 entries, all four N values legal,
`b_layout_hash` matching Phase 3's container, and `tile: 48`; and the two must
differ in `emulate_bfp16` and be identical in every other checked field, which
is the checker's own comparison doing the work. Re-run the existing five to prove
the `common` edit was not missed by one of them.

Which of the two the model actually *uses* is Phase 7's gate, not this one.

## Phase 5 — Pre-LN encoder

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
4. `final_norm` at norm-site `1 + 2*g_layers` — the next free index after the
   per-layer `ln1`/`ln2` loop in `stage_all` (`:1606-1626`). **Three vectors,
   not one:** `s_ln`, `h_gamma` *and* `h_beta`, because `layer_norm_cpu` indexes
   `h_gamma[site]`/`h_beta[site]` directly (`:1665`) and those are pushed in
   lockstep with `s_ln`. Push into **both** arms — the staged arm and the
   `host_ln` arm, which number their sites differently (`--host-ln` assigns
   `s_ln.size() + 1` and `layer_norm` then does `layer_norm_cpu(x, slot - 1)`,
   `:1955`). A site pushed into one arm only is an out-of-range read in the
   other, and `h_gamma[site]` on a short vector is not a bounds check.

5. Layer 0's `ln1` skipped when `identity_attn_norm_layer0` is set.
6. RoPE tables. Copy `GemmaNpuEncoder::ensure_tables` (`:3748-3757`) and its
   per-layer selection (`:3827-3829`). **mmBERT has one theta, so build one
   table and select it; do not add a per-layer-theta code path in this phase.**
   The English checkpoint's two thetas are out of scope — see "Not in scope".
7. `swiglu_halves == "gate|up"` → `gelu(lo) * hi`, because the packer wrote the
   gate half first. An unrecognised string **throws**, as the other two do.

   **This is one key and THREE code changes, and getting two of them wrong is a
   silent wrong answer on the default path.** The gated activation is
   implemented three times over, each a copy of the others:

   | copy | line | on the default path? |
   |---|---|---|
   | `swiglu_cpu` | `:1918-1921` | no — this is the *unfused* fallback |
   | bf16 fused epilogue | `:2086-2091` | **yes** — `fuse_ffn_epilogue` defaults to `true` (`:1501`) |
   | int8 fused epilogue | `:2223-2228` | only under `--int8`, which this model does not use |

   All three read `g_gated_act` and branch on `GatedAct::GeluErf`; none of them
   consults a half-order key, so a new key alone changes nothing. The third
   activation (`gelu_cpu`, ungated) is not involved. Two of the three comments
   say the copy is deliberate ("COPIED VERBATIM … not re-derived", ":2220-2221"),
   which is the right instinct and exactly why the third copy must be updated in
   the same commit — otherwise `--no-fuse-ffn` and the default disagree, and the
   difference is a gate swap, not a rounding difference.

   The key itself is read in `apply_model_shape` beside the other two
   (`:589-594` for `swiglu_halves`, `:658-679` for `activation`), **not** beside
   `geglu_halves` (`:3139-3144`) — that one is in `GemmaNpuEncoder`'s
   constructor (class opens at `:3030`), a different code path that arch=4 never
   reaches.

**Loop body, pre-LN** (`x` is updated in place; pre-LN never needs the
post-LN residual copy, which is why `add_norm_*` has no place here):

```
h = LN1(x)  -> hbuf                        (skip at L==0)
qkv = GEMM(qkv, hbuf)
rope(qkv)                                  (one theta)
qk -> scores; band mask; softmax; av -> ctx  (band arrives in Phase 6; this
                                               phase runs full attention)
proj = GEMM(attn_out, ctx)
x += proj                                  (no norm)
h = LN2(x) -> hbuf
up = GEMM(ffn_up, hbuf)
gated = geglu(up)                          gelu(first half) * second half
down = GEMM(ffn_down, gated)
x += down                                  (no norm)
final = LN_final(x)
```

The two `x += …` lines are the whole difference from post-LN, and they are the
reason item 3 is not optional: with `add_norm_*` in the loop, `x` would receive
the *normalised* value at each of those two points and the residual stream would
be re-centred twice per layer.

**Gate.** Per-layer hidden states within the `oflm-test` cosine discipline
against a fp64 Python reference, for all 22 layers plus the final norm.
**Calibrate the threshold from a bf16 replica, never in advance** — the whisper
precedent is that 0.999 on `enc.out` is not reachable at 32-layer depth in
bf16 (`specs/open-whisper/spec.md:45-49`). Report the replica's number and the
engine's number side by side; fail only on a *gap*, not on an absolute value.

## Phase 6 — Band mask, and the S=1024 measurement

Two edits, both small — **and both land in functions that arch=0 through 3 also
call**, so both need a flag that is off by default. This is the sharpest edge in
the phase: `run()` drives the same `softmax_cpu`/`qk_impl`/`av_impl` that
`run_preln` will, for six shipping encoders that have no locality term at all.
An unconditional clamp is not "a band mask for the new model", it is a band mask
on `bge-base`, and it returns a correctly-sized, correctly-normed, plausible
vector.

1. **The band cannot go in `add_mask`.** That vector is `[batch, g_seq]`
   (`:1374`) and both mask paths index it as
   `add_mask.data() + (r / rows_per_seq) * g_seq` with
   `rows_per_seq = g_heads * g_seq` (`:1751`, `:1771`) — so it has **no head
   axis and no layer axis**, only a per-sequence offset. A band depends on
   `(layer, i, j)`, so it must be computed inside the mask-adding loop, where
   `r` decodes to `b = r/(g_heads*g_seq)`, `h = (r/g_seq) % g_heads`,
   `i = r % g_seq`.
2. **There are TWO mask paths and the phase must cover both.** `run()` picks
   between them at `:2848-2855`: `host_sm` selects `softmax_cpu(scores)`,
   otherwise `add_additive_mask(scores)` feeds the NPU softmax design. Banding
   only `softmax_cpu` leaves a model that is *silently full-attention on all 22
   layers* whenever the design runs the array softmax — wrong answers, and the
   "129/1024 saving" measurement in the same phase becomes fiction. The band
   term goes in **both** (`:1745-1755` and `:1757-1810`).
3. Both edits need a member `int64_t band_half = 0;` — **zero means no band**,
   so every existing caller and every existing container is bit-identical
   without touching `run()` — plus the current layer index. `run()` (`:2840-2862`)
   drives the same `qk`/`softmax`/`av` calls `run_preln` will, for six shipping
   encoders that have no locality term at all; an unconditional clamp is not "a
   band mask for the new model", it is a band mask on `bge-base`, returning a
   correctly-sized, correctly-normed, plausible vector. Set `band_half` in
   `apply_model_shape`'s arch=4 arm only, and **refuse a non-zero band with no
   `sliding_window` key** rather than defaulting one.
4. Clamp `j` in `qk_impl` (`:2538-2611`) and `av_impl` (`:2624-2704`) to
   `[i-band_half, i+band_half]` when `band_half != 0`, so the MACs are **skipped**
   on the encoder's local layers. Both are templates on `NV`; the clamp goes in
   the shared outer `p` loop, not in an unrolled arm, or the head's full-band
   attention (Phase 7) and the encoder's banded one need different
   instantiations of the same function.


**The head must never be banded.** Phase 7's head layers are global attention.
Give them their own attention entry point rather than setting `band_half = 0`
around a call — a caller that forgets is a silent wrong answer, and the flag is
exactly the sort of thing that gets left set.

**The half-window is 64, not 65.** `local_attention` is 128 and the mask uses
`local_attention // 2`. The `+1` in HF's
`self.sliding_window = config.sliding_window + 1` is a FlashAttention
inclusive-boundary convention and does **not** apply to a dense/sparse mask.
This one is settled by reading the source rather than by ablating: the sdpa mask
is built from `config.sliding_window`, a property equal to `local_attention //
2` (`configuration_modernbert.py:160-162`), and reaches
`abs(q_idx - kv_idx) <= sliding_window` via
`sliding_window_bidirectional_overlay` (`masking_utils.py:143-151`, `:1297-1307`);
only the eager/flash attention *interface* receives the `+1`
(`modeling_modernbert.py:253`, `:294`), and Laya pins `sdpa`.
(`npue.py:106-141` says "only an ablation settles it" — the transformers source
settles it, and the ablation is still worth running as a check, not as the
arbiter.)

**Mask fill: one value, chosen deliberately.** The BERT path currently fills the
padding mask `-1.0e30f` (`:4559`) and the Gemma path `-3.4028235e38f` (`:3811`).
A band that used `-inf` would sit in the same `add_mask` vector as a padding mask
using `-1.0e30f`, and the two are added in the same pass. Reuse the BERT path's
`-1.0e30f` for the band so one buffer carries one convention and the two
failure modes are indistinguishable in the code.

**Safety property.** Outside the band the mask term is exactly the fill value and
inside it the term is `0.0f`, so in-band values are bit-identical to the unmasked
path restricted to the band — the guarantee rests on *adding exact zero*, which
holds for any finite fill. `open_whisper/host_ops.cpp:531-536` records that a
1-ulp softmax change cost a golden token path.


**Then measure.** With the phase-timers already in `npue_encoder.hpp` (`:2847`,
`:2861`, `:1809`), record at S=1024, batch 1 and 4: total wall, NPU GEMM time,
host QK, host softmax, host AV. Replace the standing warning at
`export_gemm_rtp.py:400-406` with the measured numbers, in that file's comment
and in this plan.

**Gate.** The measurement exists and is written into the repo. **Phase 7 does
not start without it.**

## Phase 7 — The decision engine

`src/include/AutoDecisionModel/auto_decision_model.hpp` +
`auto_decision_model.hpp` registry, a sibling of `AutoEmbeddingModel`
(`src/include/AutoEmbeddingModel/all_embedding_model.hpp:30-85`). Same shape:
`load_model`, `decide(state, questions, temperature) -> answers`, plus
`readout()`, `supported_kinds()`, `prompt_names()`.

Behind it, a `Decider` in `src/open_npue/` that:

1. Builds the prompt per `build_sequence` (Phase 2's tokenizer). **One row per
   (state, question) pair** — `collate_items` flattens question groups, and each
   row is independent, so a batch of 5 questions is 5 rows through the same
   encoder. Reuse `EmbedService::plan()`'s greedy tier split
   (`npue_encoder.hpp:4508-4525`) rather than calling `use_tier(n)` directly:
   `use_tier` pads to the tier (`:1390-1402`), so a naive `use_tier(5)` costs a
   whole 16-row pass. With tiers {1,4,16,32,64} `plan()` turns 5 into 4+1.
2. Runs `Encoder::run_preln` for the 22 encoder layers, producing the
   post-`final_norm` hidden states. That is the value the head consumes —
   upstream is `h = self.encoder(...).last_hidden_state`, i.e. **after**
   `final_norm` — so the head's input is the encoder's output, not `x` from
   inside the loop.
3. **The head's 2 layers on the HOST.** Not a second `Stack`/`Design`: see
   "Why the head runs on the host" for the four reasons, the shortest of which
   is that `g_ffn`/`g_layers` are process-wide and a second `npu::Design` is a
   second `hw_context`. The head is a small, self-contained host transformer over
   `[rows, 1024, 768]`:
   - `h += type_emb[qtype]` broadcast over the sequence;
   - 2 × (pre-LN → qkv → **bidirectional full-band** attention, no sliding
     window, no causal mask, padding-masked only → `+out_proj` → pre-LN →
     `linear1`+bias → **ReLU** → `linear2`+bias → `+`);
   - gather at `markers[i]`.

   **The head's attention scale is explicit and the encoder's is not.** For
   arch=0/2/3 the packer folds `1/sqrt(head_dim)` into the Q block of the qkv
   weight (`npue_pack.cpp:1849-1856`) and `qk_impl` therefore computes a raw dot
   product — the repo says so twice, in the packer comment and in `run()`'s own
   ("1/sqrt(head_dim) is already folded into Q by the .npue", `:2841`). The head's weights are
   packed as plain F32 host tensors with no such fold, so the host head GEMM
   must apply `1/sqrt(64)` itself. Omitting it leaves the pre-softmax scores 8×
   too large, which saturates the softmax and looks like a confident model
   rather than a bug. Fold it into the Q rows of `in_proj` **after** the GEMM,
   not before, or it is applied twice on the K side too.

   **`in_proj_weight` is `[3d, d]` in `[Q | K | V]` order**, Q rows `[0,d)`, K
   `[d,2d)`, V `[2d,3d)` — de-interleave before the attention, and note this is
   *not* the `(3, Nh, Dh)` interleave of the encoder's fused ModernBERT qkv
   (research-doc trap 5). The head is plain PyTorch MHA, not ModernBERT, so it
   has no per-head layout and no RoPE.

   The attention reuses the encoder's own `qk_impl`/`av_impl`/`softmax_cpu`
   with `band_half` **not** set — a head layer is global attention, and applying
   the ±64 band to it is a silent wrong answer, not a slow one. Give the head its
   own attention entry point rather than a global toggle, so the two cannot be
   confused.
4. Runs the host tail: `scorer` (LayerNorm → Linear+GELU → Linear) over the
   gathered marker rows, then `masked_fill(~marker_mask, -1e4)`.
5. Applies temperature, softmax, argmax, `max(p)` confidence, and
   `score = sum(i * p_i)`. **The softmax runs over the batch's `kmax` marker
   slots, not over `k`** — upstream masks to `-1e4` and softmaxes the padded
   axis (`laya/common.py:330-332`) — and `answer_confidence` is
   `max(p[:k])` with a **per-row** `k` (`laya/common.py:483`). Return only the
   first `k` per row. The `oflm-test` "probabilities sum to 1" check cannot
   catch returning `kmax` entries, so the reference-agreement fixture must
   assert the length.


**The `act_head` is computed and discarded.** Laya's own issue tracker records
`action.act_probability` at AUROC 0.30 versus 0.77 for `confidence`. It is dead
weight. Compute it (it is in the checkpoint, and skipping it is a silent
divergence) but do not return it; record why in the `not_implemented` list.

**The reference, stated precisely, because the gate is meaningless without it.**
The oracle is the upstream PyTorch model run the way Laya runs it, which is
**bf16 autocast** — `rl_agent_config.json` says `"amp_dtype": "bf16"`, not a
default this port gets to pick. So the reference's `nn.Linear` layers are bf16
while this port's host tail is F32. That is a deliberate precision difference and
it has to be priced, not assumed away: measure the reference **twice**, once
under autocast and once in fp32, and report the fp32-vs-autocast argmax
disagreement as the **reference's own irreducible noise floor**. Any
disagreement this port has with the fp32 run that the autocast run does not is
this port's error; the rest is the reference's.

**Gate — argmax agreement, stratified by how decided the reference was.** For
≥ 50 (state, question) pairs, split by the reference's own top-2 probability gap
and require two different things of the two strata:

| stratum | reference's top-2 gap | requirement |
|---|---|---|
| **decided** | ≥ 0.20 | argmax agrees on **100 %**. No exceptions — a disagreement here is a bug, and this is where every real defect will show up. |
| **undecided** | < 0.20 | no pass/fail. **Record the agreement rate** in the repo alongside the fixture. |

A 0.20 threshold is a starting point, not a constant: measure the distribution
of reference gaps over the fixture first, and if no natural separation exists,
say so and set the threshold at the point where this port's disagreements stop
clustering. The point of the split is that "100 % argmax agreement" is
unachievable as a *flat* claim and unnecessary as a goal: the undecided pairs are
pairs where the reference itself is a coin flip, and a port that matches a coin
flip half the time has not failed. What the split preserves is the property that
matters — **zero disagreements on every pair where the model had an opinion** —
while producing a number for the rest instead of a hope.

`confidence` within 1e-3, and raw logits within the bf16 tolerance established in
Phase 5. **Argmax is the gate; logits are diagnostics** — with pre-softmax scores
reaching ~55 and amplifying summation order, a logit-tolerance gate would be both
unmeetable and meaningless.

**The datapath decision, made here.** Run the whole gate twice, once per family
from Phase 4. Record both numbers. Pick the bfp16 arm only if it matches the
decided-stratum requirement *and* its undecided-stratum rate is not worse than
the bf16 arm's; otherwise ship `BERT-h768-gated-i1152-bf16` and record why in the
container's `not_implemented` and in the skill. This is the same decision
`bge-small` was rebuilt for, made the same way — by measurement, on this model,
not by inheriting it from four other families. Whichever wins, write the losing
arm's number into the repo next to the winning one: a datapath chosen by
ablation is only trustworthy if the ablation is legible later.

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

1. **`1_Pooling/config.json` is a shim, and it has to exist before the packer is
   called — not "at pack time".** `prepare_model_auto` resolves pooling at
   `npue_pack.cpp:2006`, **before** the arch dispatch at `:2013-2048`; a packer
   that wrote the file would be writing it long after `resolve_pooling` had
   already thrown. The file also cannot come from `oflm pull`, because it does
   not exist upstream.

   So: the adapter writes it. In `npue_embedding.cpp`, after the model directory
   is known and **before** `npue::prepare_model_auto(po)` (`:175`), if
   `<dir>/1_Pooling/config.json` is absent, create the directory and write
   `{"pooling_mode_mean_tokens": true, "pooling_mode_cls_token": false}`. Never
   overwrite an existing one — a real sentence-transformers checkpoint ships its
   own and that one is authoritative. `resolve_pooling` (`npue_pack.cpp:1949-1966`)
   stays exactly as it is; the name-checked contract is preserved and every
   existing model is unaffected.

   Two things follow, and both belong in item 2 and in `not_implemented`:
   - the file is **not** in the model entry's `files` list, because `oflm pull`
     can only fetch what upstream has;
   - what it asserts is a fiction. Laya pools by `gather` at `marker_pos`
     (`laya/common.py:327-330`), not by mean. `mean` is the only value
     `apply_model_shape` will accept besides `cls` (`:719-723`), and the
     decision engine never calls `pool_rows` (Phase 7), so nothing reads the
     lie — but a future reader will, so the container says so in prose.
2. **`model_list.json`** entry, `laya-decision` / `multilingual`, with
   `npue_design_family:` **whichever of `BERT-h768-gated-i1152-bfp16` /
   `BERT-h768-gated-i1152-bf16` Phase 7's gate selected**, `npue_tile_n: 48`,
   `default_context_length: 1024`, `label: ["systemone"]`, and `files` —
   repo-root-relative, **not** including `1_Pooling/config.json`:
   `config.json` is `multilingual/encoder/config.json`,
   `multilingual/model.safetensors`, `multilingual/tokenizer/tokenizer.json`,
   and `multilingual/rl_agent_config.json` — the last one because the **packer**
   reads the temperature vector and buckets out of it (Phase 3); after packing,
   the container carries them and the file is not needed again.
   `npue_checkpoint_subdir: "multilingual/encoder"` and
   `npue_tokenizer_subdir: "multilingual/tokenizer"` name them for the packer.
   `oflm_min_version: "0.0.0"`. Register in `all_embedding_model.hpp`'s sibling
   registry.
3. **`model_info.json`** manifest entry. **No downloader work is needed, and none
   should be attempted:** `files` is an explicit list of repo-root-relative paths
   and nested paths already ship — `1_Pooling/config.json` is an entry in six
   models today (`model_list.json:1160-1166`). There is no `allow_patterns` and no
   subfolder concept in `model_downloader.cpp` to add. When generating this
   entry, fetch the sha256 of each file from the Hub's `/resolve/main/` URLs:
   the `/raw/main/` path returns a 133-byte git-lfs pointer for every one of
   them, which would pin a hash of the pointer.

4. **`oflm-test` suite** `decisions`: D1 shape, D2 probabilities sum to 1 ±
   tol, D3 determinism over 10 draws, D4 batch/index integrity, D5 argmax
   stability, D6 model identity (an impossible tag must be refused), D7 unknown
   question type dropped-with-warning, D8 `noul` has no `confidence`, D9
   **reference agreement** against a bundled golden fixture. `D` is the suite
   initial, per the existing `E`/`A`/`L`/`V`/`T` convention, and the numbers are
   stable once assigned.

   **Five registration points, not one.** `oflm_test/__init__.py` has no
   decorator and no plugin scan, so a new suite is an `if` ladder and adding only
   `SUITE_NAMES` gets you a flag that dispatches to nothing:
   1. `DecisionTask` in the `from .tasks import (…)` list (`:6-7`);
   2. `"decisions"` in `SUITE_NAMES` (`:9`);
   3. `parser.add_argument('--decisions', action='store_true', …)` (`:81-91`);
   4. a `if suites["decisions"]: results.append(DecisionTask(...).run(...))`
      block in `run_suites` (`:145-175`);
   5. **`resolve_suites` must be restructured, not extended.** Today the
      mutual exclusion is hardcoded around the literal `"embedding"` with a
      `CHAT_SUITES` tuple (`:11`, `:14-42`), and `--all` means
      "everything except embedding". A *second* exclusive suite does not fit that
      shape: the cleanest form is an `EXCLUSIVE_SUITES = ("embedding",
      "decisions")` tuple, `--all` meaning "everything not in it", and the
      conflict rule generalised to "if more than one exclusive suite is
      requested, run none of them and say so". Both take the `ShapeLease`, so
      running them in one process is a hard conflict, not a preference.
      Add `test_embedding_checks.py`-style coverage in
      `utilities/oflm-test/tests/` for the new resolution rules.
5. **`src/test/laya_decision_npu/`** — the 4-file harness
   (`test.cpp`, `CMakeLists.txt`, `Makefile`, `test.sh`), using
   `gemma_embedding`'s **standalone** CMake style, not `add_npu_test`, which
   unconditionally links the closed `q4_npu_eXpress`/`mha`/`lm_head` stack.
   Needs `-mavx2 -mfma` and the per-source `open_npue` include dir.

   **Copy its structure, not its rigour.** `gemma_embedding/test.cpp` has no
   assertions at all: it prints error metrics and `return 0` unconditionally, so
   its verdict is a number a human reads. That is the right shape for a
   benchmark and the wrong shape for a gate. For the assertions, copy
   `gemma4_tool_parser/test.cpp:11-18` instead — a `CHECK(cond)` macro that
   prints `FAIL <file>:<line>` and increments a counter, one `test_*()` per case,
   and a non-zero exit — and register it with
   `add_test(NAME laya_decision_npu COMMAND …)`, which is what
   `gemma4_tool_parser/CMakeLists.txt` does and what makes it reachable from
   `ctest`. Also note `gemma_embedding/test.sh` is a **stale leftover** that
   references a `../../detail/` tree which no longer exists; the live path is the
   Makefile's `test:` target, which copies `model_list.json` beside the binary
   first. Do not treat `test.sh` as load-bearing in the new harness either.
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
| `src/open_npue/npue_pack.cpp` | `model_type` guards; `prepare_model_modernbert`; `add_gemm_b_reorder_rows`; subdir-aware paths | 0, 2 |
| `src/open_npue/npue_pack.hpp` | declare the two; `PrepareOptions.config_subdir` / `.tokenizer_subdir` | 0, 2 |
| `src/open_npue/bbpe_tokenizer_gen.{hpp,cpp}` | `Replace` normaliser, Metaspace, blob **v2**, `byte_fallback` + `unk_token` recorded | 1 |
| `src/open_npue/tokenizer_bbpe.{hpp,cpp}` | v1 **and** v2 readers, Metaspace scan, `<unk>` arm | 1 |
| `src/open_npue/npue_encoder.hpp` | `load_tokenizer` arm; `encoder_implemented`; `hbuf`; `run_preln`; `final_norm` in 3 vectors × 2 arms; `gate\|up` in **all three** activation copies; `band_half`; head transformer | 1, 2, 4, 5, 7 |
| `src/open_npue_adapter/npue_embedding.cpp` | `npue_checkpoint_subdir` / `npue_tokenizer_subdir`; write `1_Pooling/config.json` before packing | 2, 9 |
| `npu_offload/gemm_rtp/families.json` | **two** families (bfp16 + bf16); `--batches` moved out of `common` into all six | 3 |
| `npu_offload/gemm_rtp/check_design_sets.py` | a case for a family whose `--batches` differs | 3 |
| `npu_offload/gemm_rtp/export_gemm_rtp.py` | seq-1024 warning → measured numbers | 5 |
| `src/xclbins/BERT-h768-gated-i1152-bfp16/`, `…-bf16/` | built artefacts, both arms | 3 |
| `src/include/AutoDecisionModel/*` | the interface | 6, 7, 8 |
| `src/common/AutoDecisionModel/*` | the plan/answer layer + unit tests | 6 |
| `src/open_npue/decision_engine.{hpp,cpp}` | prompt build, **host** head, readout | 7 |
| `src/CMakeLists.txt` | **add `decision_engine.cpp` to `OPEN_NPUE_SOURCES` explicitly** — `open_npue/` is not globbed | 7 |
| `src/src/decision_cli.hpp`, `main.cpp`, `vm_args.hpp`, `program_args.hpp` | `oflm decide` | 8 |
| `src/server/server.cpp`, `rest_handler.{hpp,cpp}` | route, NPU-lock enrolment, identity guard, `capabilities` | 8 |
| `src/model_list.json`, `src/model_info.json` | registry | 9 |
| `utilities/oflm-test/oflm_test/{__init__,tasks}.py` | `decisions` suite | 9 |
| `src/test/laya_decision_npu/` | harness | 9 |
| `specs/open-engine/spec.md` | the five `OPEN-DECISION-*` / `OPEN-NPUE-MODERNBERT` requirements | 0 |
| `specs/open-engine/plans/typesafe-systemone-0ffd094c.py` | the pinned wire schema (vendored, `_schemas/models.py` only) | 0, 6 |
| `.opencode/skill/open-laya-decision-kernels/SKILL.md` | the skill | 9 |


---

# New requirements

The text below is what lands in `specs/open-engine/spec.md`, in the file's own
format: `### OPEN-…`, then `**Applies to:**` / `**Test category:**` /
`**Tests:**`, then the normative paragraph, then `**Acceptance criteria:**`.
IDs follow `spec.md:3`'s `Prefix OPEN` and the `OPEN-<AREA>-<THING>` shape the
file already uses (`OPEN-QUANT-Q8`, `OPEN-PACK-PLAN`, `OPEN-PREFILL-ATTN`).
The five are appended after `OPEN-DECODE-PIPELINE`, the current last
requirement; the file has no table of contents, so nothing else needs updating.

Each `**Tests:**` line names a file under `specs/open-engine/tests/`, and each
of those opens with `# Traces: <these IDs> (canonical spec: specs/open-engine/spec.md)`
— the convention 39 of the 48 existing test files follow.

---

### OPEN-ENC-MODERNBERT: a ModernBERT checkpoint packs to an arch-4 container and the runtime runs it
**Applies to:** openflowlm-next (`src/open_npue/npue_pack.cpp`, `src/open_npue/npue_encoder.hpp`, `src/open_npue/tokenizer_bbpe.cpp`, `src/open_npue/bbpe_tokenizer_gen.cpp`, `src/open_npue_adapter/npue_embedding.cpp`)
**Test category:** unit (packer, tokenizer, shape) + manual (the NPU run)
**Tests:** `specs/open-engine/tests/test_modernbert_pack.py`

A checkpoint with `model_type: "modernbert"` shall pack to a
`modernbert_rope_geglu` container, and that container shall load and run. The
architecture is pre-LayerNorm with a final norm, bias-free throughout, no
position table (position enters only through RoPE), GeGLU whose packed order is
`gate|up`, and a sliding window of `local_attention // 2` on the layers whose
`layer_types` entry is `sliding_attention`. Layer 0's attention norm is
`nn.Identity()` and is **skipped**, not replaced by a weight-1 norm — a norm with
weight 1 still centres and scales.

The packer emits the container keys `apply_model_shape` reads, and the runtime
reads the tokenizer's special ids from the encoder `config.json` rather than from
the generated blob. It shall refuse, by name, on: a `model_type` it does not
implement; a config it cannot find; a `swiglu_halves` / `pre_tokenizer` /
`normalizer` value it does not implement; and a requested sequence length above
the packed `max_seq_len`.

**Acceptance criteria:**
- A synthetic `modernbert` fixture packs, and the container loads through `ShapeLease`/`apply_model_shape` with `num_layers`, `hidden`, `num_heads`, `head_dim`, `intermediate` and `max_seq_len` read back as packed.
- `embeddings.position` and `embeddings.token_type` and every `*.bias` are zero-filled and present; `embeddings.word` is the embedding table's packed name and dtype.
- The GeGLU gate half is packed **first**, and the container says so in `swiglu_halves`.
- With `identity_attn_norm_layer0` set, layer 0 applies no attention norm; without it, a weight-1 norm is still refused as a substitute.
- The band mask is off for every `sliding_attention: false` layer, off for the decision head entirely, and on for `sliding_attention: true` layers at exactly `local_attention // 2` — not `+1`.
- A checkpoint with a nested `config.json` is packable through the subdirectory keys; one with no `config.json` is refused naming the path.
- `bge-*`, `nomic`, `gte` and `all-minilm` still pack to byte-identical containers after every change above.

---

### OPEN-DECISION-SYSTEMONE: `/v1/systemone` is byte-compatible with the pinned TypeSafe schema
**Applies to:** openflowlm-next (`src/common/AutoDecisionModel/decision_types.cpp`, `src/server/rest_handler.cpp`, `src/server/server.cpp`, `specs/open-engine/plans/typesafe-systemone-0ffd094c.py`)
**Test category:** unit (serialisation, refusals) + integration (the route)
**Tests:** `specs/open-engine/tests/test_systemone_wire.py`

`POST /v1/systemone` shall accept a `SystemOneRequest` and return a
`SystemOneResponse` as pinned by
`typesafe-ai/typesafe-sdk-python @ 0ffd094c72ed9445223060b24ffd7a56aa781fb4`,
`src/typesafe_sdk/_schemas/models.py`, vendored beside this plan. The response
carries the full envelope — `model`, `answers`, `usage` — and `model` is the tag
that answered, not the tag that was asked for. `usage.output_tokens` is always
`0`, which is definitional for a non-autoregressive readout. `noul` answers carry
no `confidence` key at all. Index-keyed maps serialise with string keys.

Question order and option order are semantic and are preserved end to end: a
`std::map`-backed JSON object is not used anywhere on this path.

**Acceptance criteria:**
- A golden request/response pair round-trips byte-for-byte against the vendored models, and a test re-reads the vendored file and fails if the fixture and it disagree.
- The response body is `{model, answers, usage}`; `usage.output_tokens == 0`.
- A `noul` answer has no `confidence` key; serialising one is an error.
- A 3-level `score` emits `legend` and `probabilities` keyed `{"0","1","2"}`.
- `choice` accepts 1–255 options and `score` 2–10 levels; outside those, the request is refused by name.
- A request naming a model other than the loaded one is refused; the response never carries another model's answers under the asked-for name.
- A question whose `type` is not implemented is dropped with a warning and the rest of the batch is served.
- The route is enrolled in `requires_npu_access()` and is never served concurrently with a NPU route.

---

### OPEN-DECISION-READOUT: one forward pass returns a typed answer per question, or refuses by name
**Applies to:** openflowlm-next (`src/common/AutoDecisionModel/decision_types.cpp`, `src/include/AutoDecisionModel/auto_decision_model.hpp`, `src/common/AutoDecisionModel/all_decision_model.hpp`)
**Test category:** unit
**Tests:** `specs/open-engine/tests/test_decision_plan.py`

A decision model shall take a state plus typed questions and return a
probability distribution per question in one forward pass, with no generation and
no parsing. The readout kind is derived from the container, and a model whose
readout this build does not implement **refuses by name** — it does not fall
back to a neighbouring recipe. `output_tokens` is always `0`.

`noul` renders exactly two options in the semantic order `[false, true]`; `score`
options are ordered and the order is the scale; `choice` preserves the caller's
option order. A criterion value that is falsy but meaningful — `0`, `False`,
`0.0`, `[]` — is rendered, not replaced by the type's default sentence.

`confidence` is `max(p[:k])` **after temperature**, the quantity Laya's
calibration is fitted to. This is a recorded divergence: the pinned schema types
`confidence` as a bare 0–1 float and does not define a formula, and engines in
this ecosystem differ on which they emit.

**Acceptance criteria:**
- A 2-option `noul`, an n-option `choice` and an m-level `score` each produce the right option strings, in the right order, from the same prompt builder.
- A `noul` criterion of `0` or `False` renders its value; only `None` and `""` fall back to the default sentence.
- `noul_labels` is accepted on `noul` only; supplying it on `choice` or `score` is an error, matching upstream's refusal.
- `confidence` equals `max(p[:k])` after temperature, and is absent for `noul`.
- `probabilities` and `logits` have length `k` (this row's marker count), never the batch's `kmax`.
- A container whose readout is not `scored_slot` is refused by name, naming the readout.
- `output_tokens` is `0` in every response.

---

### OPEN-DECISION-HEAD: the decision head is arithmetically a BERT layer and runs where it can be made reproducible
**Applies to:** openflowlm-next (`src/open_npue/decision_engine.cpp`, `src/open_npue/npue_encoder.hpp`)
**Test category:** unit (head arithmetic against the reference) + manual (the NPU encoder run)
**Tests:** `specs/open-engine/tests/test_decision_head.py`

The head shall reproduce the checkpoint's 2-layer `nn.TransformerEncoderLayer`
exactly: `in_proj`=qkv, `out_proj`=attn_out, `linear1`+`linear2` the FFN, **with
biases**, `norm_first=True`, and torch-default **ReLU** — not GELU, which is the
encoder's activation and not the head's. `h += type_emb[qtype]` is broadcast over
the sequence before the head; the marker gather is after it. The head consumes the
encoder's post-`final_norm` output.

The head runs on the **host**. This is a placement decision, not a capability
limit: the encoder's geometry globals are process-wide, the encoder hardcodes its
tensor prefix, every `npu::Design` is its own `hw_context`, and one model resolves
to one design set. The head is 8 of the model's 96 GEMMs.

`act_head` is computed and discarded, and the container records why. It is not
silently skipped: a missing tensor and a deliberately-unused one are different
things.

**Acceptance criteria:**
- The head's per-layer output matches the reference on a fixed input, including both biases at every site and ReLU in the FFN.
- The head's attention is **full-band**: no sliding window and no causal mask, with the padding mask only. A banded head is a wrong answer, not a slow one.
- The head applies `1/sqrt(head_dim)` explicitly; the encoder's scale is folded into its weight at pack time and the host head's is not.
- `qkv` is de-interleaved from `in_proj_weight` in `[Q|K|V]` order (the head is plain MHA, not the encoder's `(3, Nh, Dh)` interleave, and has no RoPE).
- The head never calls `pool_rows`; pooling for this model is a gather at `marker_pos`, not a mean.
- `type_emb` is indexed by `qtype` in `QTYPES` order and broadcast over the sequence.
- `act_head` runs and its result is dropped, with the reason in the container's `not_implemented`.

---

### OPEN-DECISION-ACCURACY: the gate is argmax agreement, stratified by how decided the reference was
**Applies to:** openflowlm-next (`src/test/laya_decision_npu/test.cpp`, `npu_offload/gemm_rtp/families.json`, `utilities/oflm-test/oflm_test/tasks.py`)
**Test category:** manual (the NPU run against the PyTorch reference) + integration (the `decisions` suite)
**Tests:** `specs/open-engine/tests/test_decision_accuracy_fixture.py`

A decision model's answers shall be gated against the upstream PyTorch reference
by **argmax agreement**, stratified by the reference's own top-2 probability gap.
On pairs where the reference had an opinion (gap ≥ 0.20) agreement shall be
100 %; below that threshold the agreement rate is recorded, not passed or failed.
The threshold is calibrated from the fixture's measured gap distribution, not set
in advance, and if no natural separation exists the plan says so and sets it
where this port's disagreements stop clustering.

The reference is run **twice** — once under the bf16 autocast that
`rl_agent_config.json` specifies, once in fp32 — and the gap between those two
runs is the reference's own noise floor. Raw logits are diagnostics, not a gate:
this architecture's pre-softmax scores reach ~55 and amplify summation order, so
a logit tolerance is both unmeetable and meaningless.

**The datapath is decided here, by measurement.** `--emulate-bfp16` is not
inherited from the four families that use it. Both arms are built, the gate runs
on both, and the winner is named in `model_list.json` — the same decision
`bge-small` was rebuilt for, made the same way. The losing arm's number is
written into the repo beside the winning one's.

**Acceptance criteria:**
- On the fixture's decided stratum, argmax agreement is 100 % against the bf16-autocast reference; `confidence` within 1e-3.
- The undecided stratum's agreement rate is recorded in the repo with the fixture, and the gap distribution that set the 0.20 threshold is recorded with it.
- The fp32-vs-autocast reference disagreement is measured and reported as the noise floor, and every disagreement this port has with fp32 that autocast does not is treated as this port's error.
- Raw-logit deltas are reported, not gated.
- Both datapaths are built and gated; `model_list.json`'s `npue_design_family` names the winner, and the loser's numbers are in the repo.
- `oflm-test --decisions` runs D1–D9 (shape, probabilities sum to 1, determinism over 10 draws, batch/index integrity, argmax stability, model identity, unknown-type drop, `noul` has no `confidence`, reference agreement) and is mutually exclusive with `--embedding`, both taking the `ShapeLease`.

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
- NPU-side attention (`whisper_fa` generalisation). Gated on Phase 6's
  measurement.
- **The head on the NPU.** The first ship runs the head on the host; the NPU head
  is sequenced behind it, not forbidden. It needs per-`Encoder` geometry, a
  tensor-prefix parameter, a second `model_list.json` key, and an acceptance
  measurement that two resident `hw_context`s neither block nor thrash — all
  four listed under "Why the head runs on the host".
- Any performance work. Decision 9.
- llama.cpp PR 29363/29321 as a dependency. Read them; do not depend on them.
  #29363 is `mergeable_state: unstable`; #29321 is `draft: true` and cannot
  merge in that state.

---

# Open decisions

None outstanding. The five that were open are now decided, and each fact lives
where it is used rather than in a summary of itself: the schema pin and the
response envelope in "The wire schema, pinned" and Phase 1's gate; the
temperature source in Phase 3's container config; the two-family datapath in
Phase 4 and Phase 7's gate; the stratified argmax gate in `OPEN-DECISION-ACCURACY`
and Phase 7; and the ecosystem verification in Decision 1.

Two things surfaced while closing them and are now requirements rather than
questions: the ecosystem caps `score` at **10** levels, and index-keyed maps
serialise with **string** keys. Both are in `OPEN-DECISION-SYSTEMONE`.
