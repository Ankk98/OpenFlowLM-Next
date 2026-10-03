# Laya NPU2 — handoff for the long-term integration project

Purpose: bring every LayerNorm / GELU / softmax / fused-attention op off the
host and onto the NPU so the whole encoder forward pass is device-resident.
This doc is the durable state-of-the-project marker + the remaining work, to
be revalidated against the repo before each session rather than trusted as-is.

Branch: `add-laya-support-implement`.

---

## Validated today

- **Laya multilingual answers end-to-end** on the open NPU path:
  `oflm decide laya-decision:multilingual -i req.json` produces a valid
  `choice` + confidence. Measured 2026-10-02: NPU column util peaks at **21%**,
  not the 98% a saturated kernel hits. GELU / softmax / LayerNorm run on the
  HOST (fp32): 45 + 22 + 22 host calls per encode.
- **Fused-attention designs exist and run** for both shapes:
  - Reference point `FA_LQ=1536 FA_LK=1536 FA_VALID_LEN=1500 NUM_HEADS=20`:
    state 4 with cos ≈ 1.0.
  - **Laya shape compiled and run**: `FA_LQ=1024 FA_LK=1024 FA_VALID_LEN=1024
    NUM_HEADS=12 HEADS_PER_UNROLL=2 CASCADE_STAGES=4` — builds, loads, reaches
    `state 4` in ~3.9 ms at cos 0.99963.
- **LayerNorm at LN_N=768** (`designs/ln/`) validates against the float64
  oracle in `utilities/laya_preln_reference.py`: y maxrel 5.4e-8.

## Structural gap (why LN-on-NPU is not "flip a flag")

1. **`host_ln` is *forced*** in the unified config (`npue_encoder.hpp:4999`).
2. **No eltwise design ships** in what the engine loads. The unified `Stack`
   branch loads one xclbin with 4 GEMM shapes × 3 batch tiers = 12 streams.
   The seven `art + "/{qkv,attn_out,ffn_up,ffn_down,gelu,layernorm,softmax}"`
   dirs named in the non-unified branch are not created by any
   `build-design-sets.py` invocation and are not in any `families.json` entry.
3. **`designs/ln/` and `designs/ln_bf16/` exist standalone but are not wired
   in.** Both are validated kernels, and the bf16 one is buffer-ABI-mismatched
   with `layer_norm()` (which assumes one whole-buffer dispatch with weights
   chosen by container slot — `bind(1, slot)`). `designs/ln_bf16/` processes
   one row per dispatch and deadlocks on the second call; it's a stub, not a
   candidate.
4. **`whisper_fa` is validated standalone but is not in the dispatch path.**
   The `qk()`/`av()` calls still travel through the old encrypted-ish path.

## Gotchas that already cost time

1. `/opt/xilinx/xrt/bin` must be on PATH (`aiebu-asm`, `xclbinutil`, `aie-opt`).
   The `AGENTS.md` pointer at `utilities/mlir-aie/utils/env_setup.sh` is stale;
   that file does not exist.
2. `build_design.py <src> <outdir>` puts outputs in `<outdir>`, not
   `designs/<name>/build/` unless that's the second argument.
3. `run_kernel` needs `-DOFLM_BUILD_OPEN_KERNELS_HARNESS=ON`,
   `-DOFLM_VERSION`, `-DNPU_VERSION`, and the cfg file beside the artifacts for
   relative paths to resolve.
4. The cfg grammar is `kernelx <name> <xclbin-kind> <path>` — THREE tokens.
   Supplying two shows the misleading "missing kernelx insts.bin".
5. **Device-state semantics:** 4 = COMPLETED, 8 = FAILED/timed-out. A hung
   kernel reports state 8, not a compilation error.
6. Power dose-response is confounded: running a kernel also scales host work,
   so a rising reading is not proof the NPU did work.
7. `model.q4nx` is a *weights* container, not a kernel container. Never look
   for `.xclbin`/`.insts` inside it. Laya ships no `.q4nx`; you must build
   it or point the engine at the `.safetensors`.
8. `oflm decide` uses `-i <request.json>`, not `--input-file`.

## Required decision (no in-repo answer today)

> What runtime model should a whole-buffer LN / softmax / GELU have on the NPU?

The encoder expects one dispatch over the *whole* activation buffer, with the
weight chosen per-call from the container via `bind(1, slot)`. Every design in
`open_kernels/designs/` today is row-oriented with the weight as a fixed
argument. These two ABIs do not meet.

- **Whole-buffer ABI** (what `layer_norm()` reads / writes): a single call
  over `rows × N`, tiled inside the kernel's own sequencer or a fixed tiling.
  Straighten the LN path, but need to write one new kernel per eltwise op in
  this shape.
- **Row ABI** (what every existing design uses): one call per stream element,
  tiled by IRON's `Pipeline` / TAP. Then `layer_norm()` would need a per-row
  drive. No one has written that.

**Recommendation:** pick the whole-buffer ABI — it is the only one that lets
one `Program` hold GEMMs + LN + GELU + softmax, which is the cost of one
xclbin load, not four. Do not reconcile `designs/ln` with `layer_norm()`; it is
a dead ABI mismatch.

## What to do, in order

1. **Pick the ABI**, record it in `specs/open-engine/laya-npu-layer-migration.md`
   (the section "BLOCKED: how LayerNorm actually reaches the NPU" should be the
   anchor for the decision).
2. **Re-classify `designs/ln_bf16/`** as a stub that documents the mismatch, or
   delete it if you don't want a trap in the repo. Do not merge it into the
   engine.
3. **Write one whole-buffer kernel** per eltwise op and a paired driver that
   produces a single xclbin containing GEMM + LN + GELU + softmax. Use
   `designs/whisper_fa/` as the ABI template; it already does whole-buffer
   execution.
4. **Validate via the engine**, not just `run_kernel`: one
   `oflm decide laya-decision:multilingual -i req.json`, check the answer and
   the new column-utilization number.
5. **Re-test precision.** If the on-device LN math is bf16 and the reference
   is fp32, gate on the worst pair, not the average.

## Tests that already exist and must pass

- `utilities/laya_preln_rig.cpp` — LN slice vs `laya_preln_reference.py`.
- `utilities/laya_decision_rig.cpp` — full request shape and prompt parity.
- `specs/open-engine/tests/test_laya_preln.py` — preln slice specifically.
- `open_kernels/designs/ln/compare.py` — fp32 add-fused vs oracle (fp32 x fp32
  add vs y). This is the *validated* kernel; the bf16 variant is not.
- `open_kernels/designs/whisper_fa/` — fused attention validated standalone;
  fold into the encoder is the unfinished piece.

## Known, left visible

- `oflm-add` and `q4nx-build` are the wrong install paths for Laya: `oflm-add`
  is GGUF→Q4NX for decoder LLMs, and Laya ships `.safetensors`. The working
  three-step path is in the migration doc under "Integration".
- `designs/ln/` and `designs/ln_bf16/` are standalone kernels, not wired in.
- `designs/whisper_fa/` is validated but not in dispatch.
- No design set ships all four eltwise designs for the Laya encoder. No encoder
  layer does all four on the NPU yet.
