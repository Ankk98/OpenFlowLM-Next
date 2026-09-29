# Plan: Laya on NPU2 — the `scored_slot` readout

**Status:** designed 2026-09-29. Nothing built, nothing run.
**Spec impact:** five new requirements —
`OPEN-NPUE-MODERNBERT` (arch=4 encoder), `OPEN-DECISION-READOUT` (the plan/answer
layer), `OPEN-DECISION-SYSTEMONE` (the wire schema), `OPEN-DECISION-HEAD`
(the decision head), `OPEN-DECISION-ACCURACY` (the gate). These are **declared
here and written into `specs/open-engine/spec.md` in Phase 0**; no existing
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
| 4 | **Head on the host.** | It is 8 GEMMs against the encoder's 88, and running it on the array in the same process is not implementable against today's `Encoder` — see "Why the head runs on the host". |
| 5 | **Dedicated process, no co-residency.** | `ShapeLease` makes the engine geometry process-wide. Two resident `hw_context` objects is explicitly unmeasured in this repo — which is now also *why* Decision 4 was revised. |
| 6 | **Extend `open_npue`; add `AutoDecisionModel` as a sibling seam.** | `SYNCED.md` says edit upstream, not the synced copy. The abstraction that needs a new interface is the *app* seam, not the engine. |
| 7 | **`oflm pull` from a `model_list.json` entry.** | Consistent with the six existing encoders. |
| 8 | **S=1024, batch tiers 1,4,16,32,64.** | Laya's own default `max_len`. Tiers capped at 64 because host attention materialises `batch x heads x S^2 x 4` = 3.2 GB at 64. |
| 9 | **Performance is a separate project.** | Build for correctness. `tile_n` and the pad-vs-16 trade-off are re-opened only after this ships. |

## Why the head runs on the host

The head is arithmetically a BERT layer, and that is worth stating first
because it is true: its four GEMMs are `(768,2304) (768,768) (768,3072)
(3072,768)` and `design_fits(768, 3072, gated=false, qkv_n=2304)` returns true
against `BERT-h768-bfp16`, so **an NPU head would need no new xclbin**. It is
still not the right first shape, because "no new xclbin" is not "no new work",
and four things stand in the way that no amount of care in Phases 2 or 3
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
gated=false, qkv_n=2304)` returns true against the shipping set, so the NPU head
would need no new xclbin — and that fact is the *reason* Decision 4 was revised
rather than kept, not a reason to keep it: "no new xclbin" is not "no new work".
The head runs on the host; see "Why the head runs on the host" for the four obstacles,
which are process-wide geometry, a hardcoded tensor prefix, a second
`hw_context`, and a one-design-set-per-model lookup — none of which a new xclbin
would fix.

The head tail is host-only and must stay there — it is not array-tileable, and
under the revised Decision 4 that is no longer a partial exception:

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

Phase 1 ships the `state: string` path. Phase 1's gate is the string path only;
the structured path is a follow-on and is **not** required to ship.

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

## Phase 0 — Fail closed on an unrecognised or absent `model_type`

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

The subdirectory problem is Phase 2's, and it needs a key. See "Workload
handling" there. It is named in the message because the two failures are the
same user-visible symptom.

**Gate.** Packing any checkpoint with no root `config.json` raises an error
naming that path; packing one with `model_type: "modernbert"` raises an error
naming `model_type`. `all-minilm`, `bge-*`, `nomic`, `gte`, `embeddinggemma`
all still pack and still produce **byte-identical** containers — the guard only
touches inputs that already failed. Re-run `utilities/test_open_npue.ps1`: six
cases (`:51-58`), all six must pass.


## Phase 1 — Tokenizer: Metaspace byte-BPE

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
4. Tokenise **per segment with `add_special_tokens=false`**, specials inserted
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
  `layer_norm_eps: 1e-5` for provenance and add a Phase 4 note that the host
  norm uses 1e-12, or (b) make `layer_norm_cpu` read the container's value.
  (a) is what this plan assumes; the delta is ~5e-6 relative, far under the
  bf16 operand's 8-bit mantissa, and Phase 4's gate is a bf16 comparison.
- **`rope_theta` is provenance, not the mechanism.** The checkpoint carries
  thetas as `rope_parameters.{full_attention,sliding_attention}.rope_theta`
  (both 160000, verified) — a *nested* key, so the packer must read it from
  there, not from a top-level `rope_theta` that does not exist in this config.
  Phase 4 builds **one** table from it.
- **`"pooling":"mean"` is a fiction and `l2_normalize:false` is the honest
  half.** Laya pools by `gather` at `marker_pos`, not mean and not CLS — see
  the research doc, and `laya/common.py:328-330`. `pool_rows` accepts only
  `cls`/`mean` and throws otherwise (`npue_encoder.hpp:719-723`), so the
  decision engine must **never call `pool_rows`**. Emit `mean` only because
  `apply_model_shape` demands one of the two, and record in `not_implemented`
  that pooling for this model is a marker gather, not a mean.
  `l2_normalize: false` is correct and load-bearing: `pool_rows`'s default is
  `true` (`:740`) and Laya never L2-normalises.
- **`fusions`** — copy arch=3's verbatim (`npue_pack.cpp:1792-1799`), minus
  nothing; the deltas for arch=4 are the keys above.

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

**Head tensors** are packed too — Phase 7 runs them **on the host** (Decision 4,
revised), so these are plain F32 host weights, not `gemm_b`: no layout hash, no
design, no pre-tiling. They are still packed so that the container is
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
50 % overlap; the structured-`state` JSON path (Phase 1 ships strings only);
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
problem in Phase 4.
Note `tools/verify_pack_parity.py` lives upstream in `vegah/Npu-Embeddings` and
is **not vendored here** — parity must be re-established, not assumed. The
`b_layout_hash` half of (c) is deferred to Phase 3, because Phase 3 is what
builds the design whose hash it is: `gemm_b_layout(64, 48)` is the same function
for every `-n 48` family, so the container's `layout_hash` and Phase 3's
`b_layout_hash` are the same value by construction, and Phase 3's gate is where
that gets confirmed rather than assumed.


## Phase 3 — The design family

Add to `npu_offload/gemm_rtp/families.json`:

```json
{
  "name": "BERT-h768-gated-i1152-bfp16",
  "serves": ["laya-decision:multilingual"],
  "note": "gated 768 with intermediate 1152, not the 3072 of BERT-h768-gated-bfp16. ffn_up is 2304. 48 is legal and matches every shipping family. Its qkv (768x2304) and attn_out (768x768) are byte-identical to BERT-h768-bfp16's, so the two share 8 of 16 cache markers and must never be built concurrently.",
  "args": ["--hidden","768","--intermediate","1152","--qkv-n","2304",
           "--gated-ffn","--emulate-bfp16","--c-bf16","-n","48",
           "--batches","1,4,16,32,64","--seq","1024"]
}
```

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

**Gate.** `check_design_sets.py --xclbins src/xclbins` passes for all **six**
families: `families.json` has five entries over four distinct names today, and
the registry in `all_embedding_model.hpp:37-46` carries seven *tags* of which six
are Npue — the seventh is `embed-gemma:300m`, which is served by `OpenGemma` and
has no design family. The new `design.json` carries `streams[]` for 4 shapes ×
5 tiers = 20 entries, all four N values legal, `b_layout_hash` matching Phase 2's
container, and `tile: 48`. Re-run the existing five to prove the `common` edit
was not missed by one of them.


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
6. RoPE tables. Copy `GemmaNpuEncoder::ensure_tables` (`:3748-3757`) and its
   per-layer selection (`:3827-3829`). **mmBERT has one theta, so build one
   table and select it; do not add a per-layer-theta code path in this phase.**
   The English checkpoint's two thetas are out of scope — see "Not in scope".
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
   The attention reuses the encoder's own `qk_impl`/`av_impl`/`softmax_cpu`
   with the band mask **disabled** — a head layer is global attention, and
   applying the ±64 band to it is a silent wrong answer, not a slow one. Give
   the head its own attention entry point rather than a global toggle, so the
   two cannot be confused.
4. Runs the host tail: `scorer` (LayerNorm → Linear+GELU → Linear) over the
   gathered marker rows, then `masked_fill(~marker_mask, -1e4)`.
5. Applies temperature, softmax, argmax, `max(p)` confidence, and
   `score = sum(i * p_i)`. **The softmax runs over the batch's `kmax` marker
   slots, not over `k`** — upstream masks to `-1e4` and softmaxes the padded
   axis (`laya/common.py:331-333`) — and `answer_confidence` is
   `max(p[:k])` with a **per-row** `k` (`laya/common.py:483`). Return only the
   first `k` per row. The `oflm-test` "probabilities sum to 1" check cannot
   catch returning `kmax` entries, so the reference-agreement fixture must
   assert the length.


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
| `src/open_npue/npue_pack.cpp` | `model_type` guards; `prepare_model_modernbert`; `add_gemm_b_concat_rows`; subdir-aware paths | 0, 2 |
| `src/open_npue/npue_pack.hpp` | declare the two; `PrepareOptions.config_subdir` / `.tokenizer_subdir` | 0, 2 |
| `src/open_npue/bbpe_tokenizer_gen.{hpp,cpp}` | `Replace` normaliser, Metaspace, blob **v2**, `byte_fallback` + `unk_token` recorded | 1 |
| `src/open_npue/tokenizer_bbpe.{hpp,cpp}` | v1 **and** v2 readers, Metaspace scan, `<unk>` arm | 1 |
| `src/open_npue/npue_encoder.hpp` | `load_tokenizer` arm; `encoder_implemented`; `hbuf`; `run_preln`; `final_norm`; `gate\|up`; band mask; `j` clamp; head transformer | 1, 4, 5, 7 |
| `src/open_npue_adapter/npue_embedding.cpp` | `npue_checkpoint_subdir` / `npue_tokenizer_subdir` | 2 |
| `npu_offload/gemm_rtp/families.json` | one family; **`--batches` moved out of `common` into all six** | 3 |
| `npu_offload/gemm_rtp/check_design_sets.py` | a case for a family whose `--batches` differs | 3 |
| `npu_offload/gemm_rtp/export_gemm_rtp.py` | seq-1024 warning → measured numbers | 5 |
| `src/xclbins/BERT-h768-gated-i1152-bfp16/` | built artefacts | 3 |
| `src/include/AutoDecisionModel/*` | the interface | 6, 7, 8 |
| `src/common/AutoDecisionModel/*` | the plan/answer layer + unit tests | 6 |
| `src/open_npue/decision_engine.*` | prompt build, **host** head, readout | 7 |
| `src/src/decision_cli.hpp`, `main.cpp`, `vm_args.hpp`, `program_args.hpp` | `oflm decide` | 8 |
| `src/server/server.cpp`, `rest_handler.{hpp,cpp}` | route, NPU-lock enrolment, identity guard, `capabilities` | 8 |
| `src/model_list.json`, `src/model_info.json` | registry | 9 |
| `utilities/oflm-test/oflm_test/{__init__,tasks}.py` | `decisions` suite | 9 |
| `src/test/laya_decision_npu/` | harness | 9 |
| `specs/open-engine/spec.md` | the five `OPEN-DECISION-*` / `OPEN-NPUE-MODERNBERT` requirements | 0 |
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

Not blockers — each has a defensible default, and each is a judgement call the
implementer may want to make deliberately rather than by inheritance.

- **`noul` criterion defaulting.** Upstream is
  `render_criterion(c) if c not in (None, "") else "<default>"`. `crit or
  "<default>"` is not equivalent: `0`, `False`, `0.0` and `[]` are falsy but
  meaningful, and the `choice` branch's own comment calls out exactly this.
  Default to `not in (None, "")`.
- **`decision_question.labels`.** Upstream *rejects* `labels` on anything but
  `noul`, and *requires* `{false, true}` there. Structuring the C++ type
  around `criteria` + `labels` as two vectors loses the custom noul labels;
  prefer a shape that can express them.
- **The `typesafe-sdk-python` schema.** `/v1/systemone` is byte-compatible with
  an external, unpinned schema. Vendor a copy and pin a commit, or the
  "byte-compatible" claim has no referent.
- **Where the temperature vector lives.** `rl_agent_config.json` is in neither
  the container config, the tensor list, nor the model entry's `files`. Pack it,
  or read it from the model directory — decide, and say which.
- **`--emulate-bfp16`.** Five of six shipping encoders use it; `bge-small`
  failed its gate on it. The research doc is right that this must be ablated
  against a bf16 replica rather than inherited, and that ablation is not yet a
  phase.
- **The Phase 7 argmax gate.** 100 % is a strong claim against a reference
  whose `rl_agent_config.json` sets `amp_dtype: "bf16"`; the same documents
  report pre-softmax scores near 55 amplifying summation order. Consider
  pricing a tolerance band against a bf16-autocast reference before fixing the
  gate at 100 %.
