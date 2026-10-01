# Help a model out!

You can install or request toolchain components

Each time you build a new set of xclbins and libraries, document your process and create a new skill file for the next to build on.

Skills provide specialized instructions and workflows for specific tasks.
Use the skill tool to load a skill when a task matches its description.

The new format Q4NX models for this will be located on huggingface `Atomic-Germ/*-nB-OpenNPU2`, each will need a builder in q4nx-build as they gain xclbin support. Family xclbins will be distributed with the application, and a converted model ships with the rest. Each model links to a family xclbin, regardless of fine-tuning or derivitive models. The purpose of this is to allow quick additions of new but shape-identical models without updates needed, distributed through `oflm-add` to install and link properly.

As you encounter closed-source requirements, note them and replace with "not implemented" messaging before coding a replacement.

When investigating pre-compiled xclbins and libraries, save any useful tools you may create to the `utilities/` directory rather than leaving them in a temp folder.

The `oflm-test` tool at `utilities/oflm-test` is capable of running a full test suite for `--llm`, `--vision`, `--embed`, or `--tools`.

The `q4nx-build` tool at `utilities/q4nx-build` should always be in-sync with the expected formats of the models as each gains support, there should be no manual steps left. If unavoidable, the user should always recieve instruction for it.

For each successfully packed model, create or update a skill to ensure the next agent does not need to reproduce research for the next addition.

Note: Peano (llvm-aie) has not been added to PATH to avoid conflict with
      system clang/clang++. It can be found in:
      ./ironvenv/lib/python3.12/site-packages/llvm-aie/bin

Activate the ironvenv/bin/activate; use source utilities/mlir-aie/utils/env_setup.sh also *if needed*

<available_skills>
  <skill>
    <name>npu-offload-pipeline</name>
    <description>AMD NPU2 Iron/mlir-aie kernel compilation, XCLBIN and closed-library replacement, open model engine integration, XRT dispatch, and CPU/NPU validation. Use when investigating or replacing precompiled NPU artifacts, building family XCLBIN bundles, extending q4nx-build, or debugging NPU offload.</description>
    <location>.agents/skills/npu-offload-pipeline/SKILL.md</location>
  </skill>
  <skill>
    <name>upstream-prior-art</name>
    <description>Check upstream for merged, open and in-flight work before starting a task, before editing anything in a synced or vendored directory, and again before any PR-ready verdict. Use at task start, when porting or rebasing a branch, on resume after a pause, and before declaring anything new or ship-ready.</description>
    <location>.agents/skills/upstream-prior-art/SKILL.md</location>
  </skill>
  <skill>
    <name>locked-clock-bench</name>
    <description>Enforce honest benchmarking when the accelerator cannot be clock-locked: verify instruments are alive and the machine is quiet before every arm, discard warmup, interleave and bracket A/B arms with cooldowns, label every number with its operating point, and refuse cross-arm comparisons that are not like-for-like. Use for any throughput or latency delta claim, and for any timing number that will be quoted.</description>
    <location>.agents/skills/locked-clock-bench/SKILL.md</location>
  </skill>
  <skill>
    <name>prove-correctness</name>
    <description>Prove output identity and work conservation before believing any speedup, and prove your gate can see the bug you might have introduced. Greedy identical, counts conserved, precision bounds met, gate shown to have teeth: speed without identity is skipped work until proven otherwise. Use after engagement proof, before reading any delta, or when a change is engaged but its output may still be wrong.</description>
    <location>.agents/skills/prove-correctness/SKILL.md</location>
  </skill>
  <skill>
    <name>prove-engaged</name>
    <description>Prove the change actually ran before any A/B comparison or performance claim. Name the observable that must differ, in the predicted direction, and refuse the delta if the arms are indistinguishable. Use before reading any speedup, before benchmarking a build or a flag, and when a change may be silently inactive.</description>
    <location>.agents/skills/prove-engaged/SKILL.md</location>
  </skill>
  <skill>
    <name>prove-precise</name>
    <description>Precision is contract, never a tuning knob. The single-threaded host path is the spec; every fast or reduced-precision path must reproduce it bit-for-bit or within a bound stated before the run. Diverging inputs are already suspect. Use wherever precisions mix, wherever a batch or tier selects a path, and before widening any accuracy gate.</description>
    <location>.agents/skills/prove-precise/SKILL.md</location>
  </skill>
  <skill>
    <name>ground-truth-verify</name>
    <description>Verify every load-bearing plan claim against primary sources before implementing or benching: spec text, in-tree code at working HEAD, vendor docs, driver and kernel sources. Clone or fetch dependencies pinned by SHA as needed. Use before any kernel, scheduler, driver-facing, or contract-touching change, and whenever a plan cites a spec, a register or arch number, or an external limit.</description>
    <location>.agents/skills/ground-truth-verify/SKILL.md</location>
  </skill>
  <skill>
    <name>teach-first</name>
    <description>Explain technical work so the reader can check it - reasoning and assumptions stated before acting, every term defined at first use, every number carrying units and direction, one idea per message, and a worked example from real numbers. Use for any explanation, result summary, investigation write-up or handoff addressed to the human. Fires by default, not by request.</description>
    <location>.agents/skills/teach-first/SKILL.md</location>
  </skill>
  <skill>
    <name>project-scaffold</name>
    <description>Start any multi-session task with a structured .local/&lt;slug&gt;/ folder so results, experiments, logs, scripts and status have an honest home, and keep a baseline row before the first candidate arm. Use when beginning work that will span sessions or produce numbers worth keeping, and when resuming one.</description>
    <location>.agents/skills/project-scaffold/SKILL.md</location>
  </skill>
  <skill>
    <name>host-discipline</name>
    <description>Run heavy builds, model loads, test suites and benchmarks on a shared workstation without filling its disk or exhausting its RAM. Use when a job writes gigabytes, when choosing a scratch directory, setting -j, backgrounding long work, or setting up a benchmark. Also for "the disk is full", "the machine froze", "the screen went blank but it is still running".</description>
    <location>.agents/skills/host-discipline/SKILL.md</location>
  </skill>
  <skill>
    <name>npu-profiling</name>
    <description>Profile the NPU stack at every level that has a working tool - host C++, host/NPU boundary, AIE kernel, AIE graph, whole-system - and know which tools are present but inert. Use when asking where time goes, why an encode is slow, which host phase dominates, or when a profiling attempt produced artifacts but no data.</description>
    <location>.agents/skills/npu-profiling/SKILL.md</location>
  </skill>
  <skill>
    <name>open-granite-kernels</name>
    <description>Build, verify and ship the open XDNA2 kernel sets (dx ln lm_head_q4) that run IBM Granite 4.2 3B on the dense recipe. Use when rebuilding those xclbins, adding another Granite size, debugging "no open kernels found" for a Granite tag installed with oflm-add, or when a Granite container's attention_multiplier is refused at load.</description>
    <location>.agents/skills/open-granite-kernels/SKILL.md</location>
  </skill>
  <skill>
    <name>open-laya-kernels</name>
    <description>Build, verify and ship the open XDNA2 kernel sets (BERT-h768-gated-i1152) that run laya-decision:multilingual's encoder. Use when rebuilding those xclbins, adding another hidden-768 mmBERT checkpoint, choosing between the bf16 and bfp16 datapaths, or debugging "no open kernels found" for a Laya tag installed with oflm-add.</description>
    <location>.agents/skills/open-laya-kernels/SKILL.md</location>
  </skill>
  <skill>
    <name>open-phi3-nanbeige-kernels</name>
    <description>Build, verify and serve the open XDNA2 kernel sets for Phi-4-mini (the phi3 recipe: a 96-of-128 rotation and longrope) and Nanbeige4.1-3B (the llama3 recipe at 20 heads over 4). Use when re-exporting either, adding another Phi-3 or Nanbeige size, or when `oflm serve` segfaults on the first request for a model whose adapter casts to its closed engine class.</description>
    <location>.agents/skills/open-phi3-nanbeige-kernels/SKILL.md</location>
  </skill>
  <skill>
    <name>open-qwen36-kernels</name>
    <description>Build, verify and ship the open XDNA2 kernel sets (lx0 lx1 ax0 ax1 ln lm_head_q8) that the open Qwen3.6-MoE engine (src/open_qwen36) loads. Use when rebuilding those xclbins after a design change, checking a rebuild against a previous one, packaging them for a release, or debugging "no open kernels found" at model load.</description>
    <location>.agents/skills/open-qwen36-kernels/SKILL.md</location>
  </skill>
  <skill>
    <name>openflowlm-packaging</name>
    <description>Build and ship the OpenFlowLM Linux distribution (engine + open xclbins + bundled utilities) as RPM/TGZ (and DEB on Debian/Ubuntu). Use when producing a release, changing the install prefix or PATH handling, editing src/CMakeLists.txt install/CPack rules or CMakePresets package/workflow presets, or debugging "no manual steps" install problems.</description>
    <location>.agents/skills/openflowlm-packaging/SKILL.md</location>
  </skill>
</available_skills>

Skills live in `.agents/skills/<name>/SKILL.md` as the single source (the Agent Skills
open standard). `.claude/skills/<name>` and `.opencode/skill` are checked-in
symlinks to it, never copies. Cursor, Codex and Gemini read `.agents/skills/`
natively. The flat `npu_offload_pipeline.md` became
`.agents/skills/npu-offload-pipeline/references/full-workflow.md`, which the
skill references; skill bodies stay under 500 lines.
