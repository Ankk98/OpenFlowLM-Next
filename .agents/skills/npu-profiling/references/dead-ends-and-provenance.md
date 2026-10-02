# npu-profiling: dead ends and source provenance

Moved out of `SKILL.md` to keep the skill body under 500 lines. These are
the paths that were walked and found empty, plus where the driver/XRT source
that matches the running modules lives. Read this before re-investigating any
of them.

---

### 5.1 `xrt-capture` / `xrt-replay` capture nothing

```bash
/opt/xilinx/xrt/bin/unwrapped/xrt-capture --frames 3 --output-dir $S/cap -- ./rig ...
# rc=0, two artifacts, 130711 bytes. Looks like a profile.
```

It is not one:

```
final.xclbin  : 130526 bytes  md5=d6646add2f2159c0d655045635fe3c41
capture_0.bin : 130526 bytes  md5=d6646add2f2159c0d655045635fe3c41   IDENTICAL
replay.json   : frames/threads/buffers/hwctxs/kernels/runs = all null
```

`capture_0.bin` **is the xclbin, byte for byte**, and the replay manifest is empty.
XRT's profiling support is not compiled into this build -- consistent with
`pyxrt` containing **zero** profile-related strings. So there is no AIE execution
trace, no PC sampling, and no way to get one without an XRT rebuild.

**This is the cleanest example in the repo of the rule in 6: exit 0 plus
plausible artifacts is not measurement.** Always `md5sum` the capture and check it
is not the input.

### 5.2 The `mlir_aie` Python API does not import

```python
import mlir_aie   # succeeds
len([x for x in dir(mlir_aie) if not x.startswith('_')])   # -> 0
```

Three faults, all silent:

1. both `ironvenv/lib*/python3.14/site-packages/aie.pth` contain the **relative**
   path `mlir_aie/python`, so they resolve only from some CWDs;
2. there are two site-packages trees (`lib` and `lib64`) and the `.pth` is
   duplicated across both;
3. `mlir_aie/python/aie/compiler/` has **no `__init__.py`**, so
   `from aie.compiler import compile_xclbin` fails with *unknown location* -- a
   namespace package, not a real one.

The package is also named **`aie`**, not `mlir_aie`. An empty import is the symptom
of all three.

**This does not block kernel building**, because the repo does not use the Python
API: `utilities/export-kernels.py` shells out to **`clang` (Peano), `xclbinutil`
and `aiebu-asm`**, and uses `third_party/mlir-aie` only for `MLIR_AIE_ROOT`.

### 5.3 Two claims this skill previously made, both of which were MY measurement errors

Recorded because the corrections are more useful than the claims were.

**"NPU power is 1000x wrong."** It is not. The driver is correct and the channel
works:

```
/sys/class/hwmon/hwmon12/power1_input   0 -> 885000 -> 25000   (uW)
sensors -> NPU_power                    0.00 mW -> 885.00 mW -> 25.00 mW
```

`amdxdna_sensors.c` does `npu_power * MICROWATT_PER_MILLIWATT` into hwmon's uW
convention, and the field is a `u16` of milliwatts, so 885 mW is 0.885 W of NPU
rail. The "755 W" that started this was a shell line of mine that read the number
and then appended a hardcoded `W`:

```bash
printf "%s W" "$(sensors | grep NPU_power | grep -oE '[0-9.]+')"   # 755 mW -> "755 W"
```

**Print the unit the tool gives you.** The bug was in the measurement.

**"The fdinfo busy counter is frozen."** Not established. The accounting *is*
wired -- `amdxdna_io_stats_job_start()` at `drivers/accel/amdxdna/aie2_ctx.c:448`,
`job_done()` at `:298` and `:353` -- and the accessor adds in-flight time whenever
`job_depth > 0`. The counter is per-`drm_file` and reads
`job->hwctx->client`, so it belongs to **the client that owns the hardware
context**. My sampling read only the *first* `/dev/accel/accel0` fd it found, and
XRT opens several; I was probably reading a context that was not dispatching. To
settle it, sample **every** accel fd of the process, twice.

### 5.4 Where the source is, and that it matches what is running

`~/repos/xdna-driver` is the open driver stack, and the correspondence is exact:

```
installed module : amdxdna 2.25.0_20260628,e2d8f832ad1745bab50b236bff4d8550c85b7f78
local HEAD       : e2d8f83  (origin https://github.com/amd/xdna-driver.git)
```

So source-level conclusions from that tree apply to the running driver without
version anxiety. It also carries `xrt/` as a submodule of `https://github.com/Xilinx/XRT.git`,
and that submodule is **exactly the installed XRT**:

```
submodule HEAD : 0026185de81a179dc7d886197d3e35f9a179b4e8
installed XRT  : XRT Build Version: 2.25.0 (HEAD)
                 Hash ID: 0026185de81a179dc7d886197d3e35f9a179b4e8
```

(`xclbinutil` prints the hash on any invocation, which is how to check this
without guessing.) Its `CHANGELOG.rst` stops at 2.17.0, so **the changelog is
stale, not the checkout** -- an earlier version of this file said the submodule was
out of date on the strength of that changelog, and was wrong. So source-level
conclusions from this tree apply to the running driver *and* the running XRT.



| thing | symptom |
|---|---|
| `aiebu-dump` with no `-p`/`-d` and no `-m` | exit 0, **empty output** |
| `aiebu-dump`, `aiebu-asm`, `xclbinutil` | **not on PATH**; `/opt/xilinx/xrt/bin/{,unwrapped/}` |
| `aiebu-dump` (bare name) | "No such file or directory" -- the PATH entry that looks right is `bin/`, and the tool is in `unwrapped/` for some, `bin/` for others. Check both. |
| `AGENTS.md`'s `source utilities/mlir-aie/utils/env_setup.sh` | **that path does not exist.** Peano is `ironvenv/lib/python3.14/site-packages/llvm-aie/bin` (21 tools, `clang` 21.0.0 Xilinx llvm-aie), and AGENTS.md also says Python 3.12 where it is 3.14. |
| `/sys/kernel/debug/amdxdna/` | debugfs is mounted but **empty** -- no driver counters |
| `intel-rapl` | present, `enabled=0`, reads nothing on AMD |
| `sensors` -> `NPU_power` | ~1000x too large; see `host-discipline` 7.2.1 |
| `drm-engine-amdxdna_accel_driver` in fdinfo | per-**file** submit time; you must read the fd owning the hwctx |
| `perf`, `valgrind`, `py-spy`, `ltrace`, `turbostat`, `sysstat`, `numactl`, `cpupower`, `ryzen_smu` | **all absent** -- no uncore/IMC hardware counters, no `mpstat`/`pidstat` |

---
