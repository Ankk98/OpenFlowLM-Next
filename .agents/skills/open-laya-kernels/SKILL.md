---
name: open-laya-kernels
description: Build, verify and ship the open XDNA2 kernel sets (BERT-h768-gated-i1152) that run laya-decision:multilingual's encoder. Use when rebuilding those xclbins, adding another hidden-768 mmBERT checkpoint, choosing between the bf16 and bfp16 datapaths, or debugging "no open kernels found" for a Laya tag installed with oflm-add.
---

# Laya decision (mmBERT) on the hidden-768 gated recipe

## The fold — read this before anything else

`laya-decision:multilingual` is **not a decoder**. It is a multilingual
classification model: an mmBERT-base encoder plus a two-layer decision head and a
scorer over the option set. Only the encoder goes to the NPU.

```
container  ~/.oflm/models/laya/laya-decision:multilingual.npue
              |
              +-- ENCODER  -> NPU, gemm_rtp dense recipe, bf16 operands
              |     hidden 768, intermediate 1152, qkv-n 2304, 12 heads @ hd 64
              |     gated GeGLU, seq 1024, -n 48
              |
              +-- HEAD     -> HOST, float32, decision_engine.cpp
```

**intermediate is 1152, not 3072.** Every other hidden-768 family in
`families.json` carries 3072, so `ffn_up` is 2304 wide and `ffn_down` consumes
K=1152. This is an ordinary new geometry, not a forced one — `-n 48` is legal for
all four N values (2304, 768, 2304, 768) and matches every shipping family. That
is the whole reason it is a new family name rather than a variant of
`BERT-h768-gated-bfp16`.

**Two families, one geometry, two datapaths.** They differ in exactly one flag,
`--emulate-bfp16`:

| family | datapath | `serves` |
|---|---|---|
| `BERT-h768-gated-i1152-bfp16` | `--emulate-bfp16` | `laya-decision:multilingual` |
| `BERT-h768-gated-i1152-bf16` | plain bf16 | **`[]` — on purpose** |

`serves` is empty on the bf16 arm *until the accuracy gate picks*. Whichever
wins gets the tag; the other stays built for the next checkpoint of this shape.
**Do not "fix" the empty list.** It is a separate family rather than a flag
override for two reasons: `check_design_sets.py` selects at load time by geometry
**and** datapath, and `emulate_bfp16` is a top-level key in the shipped
`design.json` -- so reusing one name would make the checker report the second
build as a stale set of the first; and both have to sit in `src/xclbins` at once
for the gate to A/B them.

The first claim is checkable in one command. The two shipped specs are identical
in every field except the datapath:

```
  BERT-h768-gated-i1152-bf16/gemm_rtp/design.json   emulate_bfp16: false
  BERT-h768-gated-i1152-bfp16/gemm_rtp/design.json  emulate_bfp16: true
  both: M=32768  seq=1024  tiers=[4,16,32]  batch=32  c_dtype=bf16  a_dtype=bf16
```

**The datapath is a measurement, not an inheritance.** `bge-small` failed its
MTEB gate on bfp16 at **-0.5010 bit-reproducibly** and had to be rebuilt on plain
bf16. A bf16 arm inherited without a gate is a formality, not a result.

## Build — on Linux, and the opposite of the Granite skill

`open-granite-kernels` says build on Windows, not WSL. **For these families,
build on Linux** with `utilities/build-design-sets.py`, which is the Linux driver
for the PowerShell `npu_offload/gemm_rtp/build.ps1` — same input, same output,
same order. Both statements are correct for their own families; do not merge them.

```bash
source ironvenv/bin/activate
python utilities/build-design-sets.py --list
python utilities/build-design-sets.py --only BERT-h768-gated-i1152-bf16
```

**One family at a time, always.** `purge()` deletes matching entries from the
**shared** `~/.npu/cache` on content markers, and `qkv`/`attn_out` depend on
neither `--gated-ffn` nor `--intermediate`. So `BERT-h768-bfp16`,
`BERT-h768-gated-bfp16` and `BERT-h768-gated-i1152-bfp16` own **identical markers
for 8 of their 12 streams**, and starting two at once deletes the other's output.
With these families the hazard is **three-way**, not two. The exporter holds a
lock and refuses in under a second rather than racing, so build serially.

The driver always runs `check_design_sets.py` afterwards, because a build that
succeeds while the spec disagrees is invisible from the artifact. **That script
lives at `npu_offload/gemm_rtp/check_design_sets.py`, not in `utilities/`** —
the docstring names it without a path and you will look in the wrong directory.

## The two stride bugs — the reason this section exists

Both are the same error: treating a packed `[rows, 3*d]` tensor as flat. `d` is
768 here, and one row is `[ Q(768) | K(768) | V(768) ]`.

The Q scaling applies `1/sqrt(head_dim)` to **Q only, per token**. Written flat,
it looks equivalent and is not:

```
  rows x 3d,  one row = [ Q | K | V ]

  WRONG:  for i < R*S*d:  qkv[i] *= sc        <- contiguous, no row stride
     row 0            [ Q | K | V ]  all three scaled
     row 1            [ Q | K | V ]  all three scaled
     ...
     row R*S/3         [ Q | K | V ]  all three scaled
     row R*S/3 + 1     untouched  ...  and so on to the end

  RIGHT:  for i < R*S:  row = qkv + i*3*d;  for j < d: row[j] *= sc
     every row         [ Q | K | V ]  Q only
```

A contiguous prefix of `R*S*d` elements is exactly `(R*S)/3` **whole rows**, so it
scaled Q, K *and* V for the first third of the tokens and nothing for the rest.

The second bug is the mirror image, in the accumulator: `acc` points at head
`h`'s slice of a `d`-wide row, so zeroing `t < d` instead of `t < head_dim_`
**erased the other 11 heads' outputs** for that `(batch, token)`. Whichever head
ran last won the zeroing, so which 11 survived depended on thread interleaving
— which is why two identical runs disagreed.

**The lesson that generalises past this model, corrected by measurement:** the
cosine gate could not see the Q bug. Gathered marker rows still measured **0.999**
against the float64 oracle, because scaling all of Q, K and V for a third of the
tokens perturbs the stream only slightly.

An earlier version of this section went further and said the *argmax* gate did see
it — 0.60 agreement over 60 pairs — and that the "datapath error versus model
margin" explanation was therefore wrong. **That was also wrong, and it was written
before the measurement was taken.** Measured, both arms, one instance each on the
NPU:

| | argmax agreement | residual logit error (max) |
|---|---:|---:|
| before the Q fix | 36/60 = 0.600 | 0.8434 |
| after the Q fix | 36/60 = 0.600 | **0.6258** |

The fix is real — the residual fell 26% — and it moved agreement by **nothing**. So
the Q bug was never the cause of the disagreements, and the margin reading of them
is still open.

Two rules, and the second is the one that was broken here:

- A gate that fires is information even when you can explain it away — but a gate's
  firing is not evidence about a *particular* defect until you have measured with
  and without that defect. Both directions of that mistake are errors, and the
  second is easier to commit because it feels like rigour.
- **Never write the conclusion of a measurement you have not taken.** Both wrong
  claims in this section were written in the past tense about a re-measurement that
  had not happened. An A/B needs both arms; one arm is an anecdote.

What the disagreements actually are: all 24 sit where the reference's own top-2 gap
is under **0.043**, while the engine's residual there is ~0.44 median. A near-tie
plus a residual an order of magnitude larger than the margin is a coin flip, and
**the fixture contains no decided pair to test.** That is the current state of the
accuracy gate: an open question, not a solved one.

## Verify

Same ladder as the siblings, in order, and the build is not done until it passes:

1. `python utilities/build-design-sets.py --only <family>` — exits non-zero on any
   family failure and always runs the checker.
2. `npu_offload/gemm_rtp/check_design_sets.py` — spec against artifact.
3. End-to-end: the accuracy fixture is the gate, and it needs **both** datapath
   arms built to be meaningful.

`utilities/laya_preln_reference.py` is the float64 oracle. `--dump-rows` writes
rows that `utilities/laya_head_rows_rig.cpp` replays into the head, which is how
the readout was proven exact to seven significant figures. **That proves the
readout only, not attention** — the Q bug above passed it.

## Result — no hardware gate run yet

Toolchain as shipped in
`src/xclbins/BERT-h768-gated-i1152-bf16/gemm_rtp/toolchain.json`: mlir-aie
**1.4.2**, Peano **21.0.0.2026080301+c9c5ecb7**.

**No datapath has been chosen, and the accuracy gate cannot currently choose one.**
The one hardware measurement that exists — 60-pair fixture, one instance, NPU —
is in the table above: 36/60 argmax agreement with a 0.6258 max residual, and
**no decided pair in the fixture at all** (the reference's widest top-2 gap over
all 60 is 0.043). The plan's gate is "100% argmax agreement where the reference
gap is >= 0.20", and that stratum is empty, so the gate's own primary requirement
is untestable rather than met. Nothing in this skill quotes a throughput or
latency figure for these families; when one exists it belongs here with its run
date and the arm it came from.

## Rules

- Never commit the xclbins; the distributed package ships them pre-built.
- Never build two `BERT-h768-*` families concurrently. They share cache markers.
- Do not populate `serves` on the losing datapath arm before the gate has run.
- Do not quote a Laya accuracy number from before the Q-scale fix. It measures a
  bug, not a datapath.
- The decision head is **host float32, not the NPU**, and its LayerNorm eps is
  **1e-5**, against the encoder's hard-coded 1e-12. That is not an inconsistency:
  upstream's head norms are `nn.LayerNorm(d)` and upstream's encoder norms use
  the config's `layer_norm_eps`, which for this checkpoint is *also* 1e-5. The
  encoder's bf16 operands hide the difference; the head's float32 residual
  stream does not — channels reach 242, so ~5e-6 relative is ~1e-3 absolute, and
  it compounds over two layers plus the scorer.
- ModernBERT/mmBERT is `arch=4`: pre-LN, RoPE, and a gated GeGLU whose **gate
  half is packed SECOND**. The loader moves it first. That is recorded in the
  **model container's** fusion spec -- `"gated_ffn_gate_half_moved_first":true`
  in the `fusions` block `npue_pack.cpp` writes -- and in `add_gemm_b_reorder_rows`
  in the same file. Do not look for it in the shipped xclbin: that
  `gemm_rtp/design.json` has no `fusions` key at all, only `name`, `kind`,
  `kernel`, `M`, `buffers`, `c_dtype`, `a_dtype`, `emulate_bfp16`,
  `b_layout_hash`, `b_layout`, `cols`, `batch`, `tiers`, `seq`. Assume the half
  you want is the half that moved.
- Before hand-building another hidden-768 family, check `amd/IRON`
  `iron/operators/flm/gemm/` first. It is a maintained Python API over the same
  MLIR-AIE, its GEMM takes **M, K and N as runtime parameters**, and it has a
  fused-activation epilogue — so it very likely covers this geometry without a
  new design set. See `npu-profiling` for why fusion is where the host time is.