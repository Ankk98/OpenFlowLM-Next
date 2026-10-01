# Upstreams consulted by this skill

Kept here rather than in the body so the skill's section order stays fixed
(R5) and so a future repo can replace this file without touching the
workflow (R9: one owner per rule).

| what | where | why it matters |
|---|---|---|
| canonical repo | git remote `upstream` -> `Atomic-Germ/OpenFlowLM-Next` | every "is this new / ready" claim is about divergence from here |
| fork | git remote `origin` -> `Ankk98/OpenFlowLM-Next` | where the work is pushed |
| engine source | `vegardberget/NpuEmbeddings` | `src/open_npue/` is a **copy** with a pinned file table in `src/open_npue/SYNCED.md` |
| model reference | `NandhaKishorM/laya` | the `common.py` / `agent.py` the decision port mirrors |
| model port prior art | `zzhdbw/laya-Ascend` | a third-party port of the same model to a **different accelerator**; read before claiming a port is novel |
| NPU stack | `Xilinx/XRT`, `Xilinx/mlir-aie`, `amd/xdna-driver` | kernel toolchain and driver; local checkouts may match the installed build by hash |

`origin` and `upstream` may point at forks. Fetch **by URL** when a specific
upstream matters, so the fetch cannot be satisfied by a fork.

## Reachability, as measured 2026-10-01

| repository | HTTPS unauthenticated |
|---|---|
| `Atomic-Germ/OpenFlowLM-Next` (the `upstream` remote) | reachable |
| `NandhaKishorM/laya` | reachable |
| `vegardberget/NpuEmbeddings` (named in `src/open_npue/SYNCED.md` and `PR_open_npue.md`) | **not reachable** |

The NpuEmbeddings URL is not reachable unauthenticated under `vegardberget`,
`Xilinx` or `AMD`. That is consistent with either a rename or a repository that
is private; an unauthenticated client cannot tell those apart. **A human has to
resolve which**, because until it is resolved the pin check in step 2 is the only
half of this skill that can run.

## Prior-art state of the Laya work, at that date

- The `upstream` remote had **never been fetched** on this branch, so every
  "is this new" claim about the port had no upstream comparison behind it.
- `git log HEAD..FETCH_HEAD -- src/open_npue/ src/include/AutoDecisionModel/`
  returned nothing once the fetch worked, i.e. upstream has not moved under the
  paths this project touches. That is a `new` classification for the port, on the
  evidence available -- subject to the NpuEmbeddings caveat above.
- `zzhdbw/laya-Ascend` is a third-party port of the same model to a different
  accelerator and is the closest prior art that should be read before any
  "first NPU port of Laya" claim.
