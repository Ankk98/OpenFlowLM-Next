# Which sources to trust for NPU facts

Every load-bearing number in an NPU change should be traceable to a source whose
standing you can state. This file records the standing of the sources reachable
for XDNA2 / Strix Halo work, and the traps in each. Verified 2026-10-02 against
this workstation.

The rule: **a number's confidence is the weakest link in its provenance, not the
number.** Say which it is.

## Tier 1 — authoritative, quote freely

### `docs.kernel.org/accel/amdxdna/amdnpu.html`

The kernel driver's own documentation (`amdnpu.rst`, in-tree upstream). This is
the best public description of what the driver and NPU firmware actually do, and
it is the source for *runtime* facts that no header exposes:

- array geometry per part — Strix Point is 4x8 (4 compute rows x 8 columns),
  Phoenix/Hawk Point 4x5, with one memtile row plus 4 compute rows per column;
- total L2 — 4096 KB on Strix Point, 2560 KB on Phoenix/Hawk Point;
- concurrent workload contexts — **16** on Strix Point, **6** on Phoenix/Hawk;
- a **64 MB host-resident instruction buffer per workload context**, PASID
  protected, mapped into both the firmware context and user space;
- partitioning is at **column granularity**, programmed by the microcontroller
  via column-isolation registers, each partition with its own PASID;
- a **Resource Solver** allocates columns from workload-declared hints plus its
  own heuristics, and firmware enforces the binding;
- the submission path: overlay + `ctrlcode`, mailbox submit, ERT executes,
  MSI-X completion;
- **DMA is encoded as `XAIE_IO_BLOCKWRITE` opcodes inside `ctrlcode`** — a host
  dispatch call is not the same event as a data movement being issued;
- firmware telemetry exists (L1 interrupt, DMA, deep-sleep counters) even where
  this box exposes no userspace path to it;
- its worked `drm-total-memory` example shows the value is a **usage sum**
  (heap + internal + external), which settles a common misreading of that field.

Caveat: it describes the in-tree driver, so check it against the running module's
version before assuming a field exists.

### `third_party/mlir-aie/include/aie/Dialect/AIE/IR/AIETargetModel.h`

The most reliable geometry source available, and it is right there in the tree.
For XDNA2 read `BaseNPU2TargetModel` and `NPU2TargetModel`:

- `rows() = 6` — "1 Shim row, 1 memtile row, and 4 Core rows";
- `getTileType()` maps row 0 to ShimNOC, row 1 to MemTile, rows 2–5 to CoreTile
  (this row numbering is why cascade designs compute `row = 2 + ty`);
- `columns() = 8` for the full NPU;
- `getNumMemTileRows() = 1` and `getMemTileSize() = 0x80000` (**512 KB**);
- `getLocalMemorySize() = 0x10000` (**64 KB**), inherited from `AIE2TargetModel`
  — NPU2 does not override it, so do not assume a new value without checking;
- `getComputeTileMaxVectorAlignBits() = 512` — **a full-width vector access needs
  512-bit alignment even though the load/store bus is 256-bit**. New relative to
  AIE2, and an easy way to get a split or a fault;
- `getNumBanks()` — 8 for a memtile, 4 for a core tile;
- `VirtualizedNPU2TargetModel` plus `TK_AIE2_NPU2_1Col`..`_7Col` — **the NPU can
  be presented as 1–7 columns**, so a design that wants all 8 must ask and cannot
  assume it.

Note the trap that cost real time: per-column L2 is **512 KB**, not the
"3 MB addressable per column" an earlier note in this repo claimed. There is one
memtile row. When a per-column figure looks generous, check `getNumMemTileRows()`
before believing it.

### `github.com/Xilinx/llvm-aie`

Peano. The README names the targets, which is worth stating because it is a
common mix-up:

| target | part |
|---|---|
| `aie2-none-unknown-elf` | XDNA — Phoenix, Hawk Point |
| `aie2p-none-unknown-elf` | XDNA2 — **Strix Point** |

This repo builds `aie2p` exclusively. Verify rather than assume:

    ./ironvenv/lib/python3.*/site-packages/llvm-aie/bin/llc --version | grep aie

The architectural statements that change how you write kernels:

- these are **in-order, exposed-pipeline VLIW cores with no stall logic** —
  instructions continue in program order regardless of pipeline state, so the
  *compiler* hides latency, not the hardware. A kernel that stalls is a kernel
  that was scheduled wrong;
- **20-bit pointer registers**;
- **varying instruction-slot widths**, so branch and return targets carry
  alignment restrictions;
- the fork carries its own caveat: maturity "similar to other Experimental LLVM
  architectures". Do not present Peano output as production-hardened.

## Tier 2 — useful concepts, wrong chip. Do not merge numbers

### `pp4fpgas.readthedocs.io/en/latest/project_aie.html`

A university course page, not vendor documentation. Its **concepts** are good and
match what this repo already does: ObjectFIFOs as the data-movement and
synchronisation primitive, TensorAccessPatterns (the wrap/stride lists in
`ln.py` and `attn_fa.py` are exactly this), and two-level tiling
(M,K,N) -> (m,k,n) -> (r,s,t).

Its **numbers describe AIE1** — 4 columns, 16 compute tiles, Versal AI Core — not
the 8-column AIE2P. Its "512 int8 or 64 int16 MACs per clock" is AIE1's vector
width; quoting it would silently halve our per-core throughput. The rest of the
page is coursework, cloud-VM setup and a grading rubric.

### `amd.com/.../technologies/ai-engine.html`

Marketing, and for a **different silicon family** (Versal AIE / AIE-ML).

It does corroborate three things *directionally*: **512 KB memory tiles**,
**64 KB local data memory per tile**, and explicitly that **FP32 is "SW
emulation"** rather than a native unit — which independently supports an
`__AIE_API_FP32_SUPPORT__ = 0` finding. But its per-tile tables are AIE-MLv2's,
not AIE2P's, and it is **self-inconsistent**: the FFT/FIR bullet claims "128 INT8
MACs per tile" while the AIE table further up says INT8 256.

A page that disagrees with itself is not a source for a number. Use it as a
signpost to the real manuals — AM009 (Versal AI Engine), AM020 (AIE-ML), UG1079
(kernel coding) — which are gated.

## The honest gap, stated plainly

**AMD publishes no XDNA2 architecture manual.** That is why per-core MAC
throughput and cascade rules come from generated Peano disassembly and
`aie_api` config headers rather than from a specification. When several
authoritative sources agree on array geometry, confidence in *the geometry* rises
to high — it does not transfer to the throughput table. Label them separately.
Do not let a well-corroborated neighbour number launder an uncorroborated one.

## Method

1. Prefer Tier 1, and prefer the **in-tree header over a web page** — it cannot
   drift from the code that will actually compile your design.
2. When two sources disagree, find the third. `AIETargetModel.h` + `amdnpu.rst` +
   `xrt-smi examine` agreeing on 8 columns x 6 rows is a strong result; two
   marketing pages agreeing on nothing is not.
3. Record the *disagreement* in the artifact, not just the verdict. A reader who
   knows a number was contested can re-check it; a reader who sees a bare figure
   cannot.
4. When a figure cannot be sourced, say so and give the weaker provenance
   explicitly ("from disassembly", "from `aie_api` config") instead of dropping
   it or promoting it.
