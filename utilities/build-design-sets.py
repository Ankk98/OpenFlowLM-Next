#!/usr/bin/env python3
"""Build the open_npue design sets from families.json, on Linux.

THE LINUX DRIVER for npu_offload/gemm_rtp/build.ps1, which is PowerShell and
therefore unavailable here. Same input, same output, same order, same
one-family-at-a-time rule -- and the reason that rule exists is repeated here
because it is the one thing a parallel driver gets wrong:

  `purge()` deletes matching entries from the SHARED ~/.npu/cache on content
  markers, and qkv/attn_out depend on neither --gated-ffn nor --intermediate.
  So BERT-h768-bfp16, BERT-h768-gated-bfp16 and BERT-h768-gated-i1152-bfp16 own
  IDENTICAL markers for 8 of their 12 streams, and building two at once deletes
  the other's output. export_gemm_rtp.py holds a lock and refuses in under a
  second, so this driver builds serially rather than racing the lock.

  And with the new families that hazard is THREE-WAY, not two.

Usage:
    source ironvenv/bin/activate          # mlir-aie + Peano
    python utilities/build-design-sets.py [--only NAME] [--dst DIR] [--force]

    python utilities/build-design-sets.py --list

Exits non-zero if any family fails to build, and always runs
check_design_sets.py afterwards -- a build that succeeds and a spec that
disagrees is the failure mode this whole file exists to prevent, and it is
invisible from the artifact.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GEMM_RTP = REPO / "npu_offload" / "gemm_rtp"


def _pyxrt_importable() -> bool:
    try:
        import pyxrt  # noqa: F401
    except Exception:
        return False
    return True


def _find_pyxrt() -> Path | None:
    """Look for a pyxrt extension module built for THIS interpreter.

    Narrow and deliberate: an XRT source tree's build directory, because a
    pyxrt built for a different CPython will import by filename and then fail
    inside the extension loader with a message about a symbol rather than about
    the ABI. The interpreter's own tag is part of the search for that reason.
    """
    tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
    roots = [Path.home() / "repos", Path("/opt/xilinx/xrt/python"),
             Path("/usr/lib/xrt/python")]
    for root in roots:
        if not root.is_dir():
            continue
        for so in root.rglob(f"pyxrt.{tag}-*.so"):
            return so.parent
    return None


def toolchain_present() -> bool:
    try:
        import aie.iron  # noqa: F401
    except Exception:
        return False
    return True


def build_one(family: dict, common: list[str], dst: Path, force: bool) -> tuple[str, bool, int]:
    name = family["name"]
    out = dst / name
    if (out / "gemm_rtp" / "design.json").is_file() and not force:
        print(f"  {name:<28} already built (use --force to rebuild)")
        return name, True, 0
    if out.exists():
        shutil.rmtree(out)

    print(f"  {name:<28} {', '.join(family.get('serves', [])) or '(no tag yet)'}")
    t0 = time.time()
    # FAMILY ARGS FIRST, common LAST -- the same order build.ps1 uses. It
    # stopped mattering when --batches moved out of `common`, because the two
    # readers could not agree about precedence, and keeping the order identical
    # is what makes that fix auditable rather than asserted.
    argv = [sys.executable, "export_gemm_rtp.py",
            *[str(a) for a in family["args"]], *common, "--out", str(out)]
    p = subprocess.run(argv, cwd=GEMM_RTP, capture_output=True, text=True)
    secs = int(time.time() - t0)
    if p.returncode != 0:
        print(f"    FAILED (exit {p.returncode}) after {secs}s", file=sys.stderr)
        tail = (p.stdout + p.stderr).strip().splitlines()[-25:]
        for line in tail:
            print(f"      {line}", file=sys.stderr)
        return name, False, secs
    n = len(list((out / "gemm_rtp").glob("*"))) if (out / "gemm_rtp").is_dir() else 0
    print(f"    ok  {n} files, {secs}s")
    # The export prints a per-stream trace that is worth keeping in the log:
    # it is where a shape that failed the tiling asserts says so.
    for line in p.stdout.strip().splitlines():
        if line.strip().startswith(("b4", "b16", "b32", "b64", "identity", "MISMATCH")):
            print(f"      {line.strip()}")
    return name, True, secs


def main() -> int:
    # Line buffering, because this script's own progress lines and the
    # subprocess output it interleaves are both diagnostics and an out-of-order
    # log of a failing build costs more than it takes to fix.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:                       # pragma: no cover
        pass
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="build one family by name")
    ap.add_argument("--dst", default=str(REPO / "src" / "xclbins"),
                    help="where the sets go (default src/xclbins)")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if the set is already there")
    ap.add_argument("--list", action="store_true", help="list the families and exit")
    ap.add_argument("--skip-check", action="store_true",
                    help="do not run check_design_sets.py afterwards (NOT "
                         "recommended: a build that succeeds and a spec that "
                         "disagrees is invisible from the artifact)")
    args = ap.parse_args()

    spec = json.loads((GEMM_RTP / "families.json").read_text(encoding="utf-8"))
    families = spec["families"]
    if args.list:
        for f in families:
            print(f"{f['name']:<28} {', '.join(f.get('serves', [])) or '-'}")
        return 0
    if args.only:
        families = [f for f in families if f["name"] == args.only]
        if not families:
            print(f"No family named {args.only!r}. Known:", file=sys.stderr)
            for f in spec["families"]:
                print(f"  {f['name']}", file=sys.stderr)
            return 1

    # aiecc shells out to `xclbinutil`, which ships with XRT and is NOT on a
    # default PATH. Without this the build dies 35 steps into the FIRST shape
    # with `tool 'xclbinutil' not found in search paths or PATH`, after
    # everything that could have been checked cheaply has already been checked.
    for cand in ("/opt/xilinx/xrt/bin", "/opt/Xilinx/xrt/bin"):
        if (Path(cand) / "xclbinutil").is_file():
            os.environ["PATH"] = cand + os.pathsep + os.environ.get("PATH", "")
            break
    else:
        if shutil.which("xclbinutil") is None:
            print("xclbinutil is not on PATH and not under /opt/xilinx/xrt/bin.\n"
                  "aiecc shells out to it to assemble the final.xclbin.",
                  file=sys.stderr)
            return 1

    # PyXRT, and WHY it has to be found.
    #
    # mlir-aie picks its tensor class at import time: if `pyxrt` imports it uses
    # the XRT-backed tensor (which accepts device="npu"), and if not it silently
    # falls back to a CPU-only tensor whose device list is ["cpu"]. The export
    # then fails on the FIRST shape with `Cannot run kernel; DefaultNPURuntime
    # not set` -- because a CPU-only tensor makes pretiled_array() try to EXECUTE
    # the design instead of only compiling it.
    #
    # So this is not an optimisation knob, it is a build prerequisite, and the
    # message names both halves of the consequence because the error does not.
    if shutil.which("pyxrt") is None and not _pyxrt_importable():
        found = _find_pyxrt()
        if found:
            os.environ["PYTHONPATH"] = (str(found) + os.pathsep
                                        + os.environ.get("PYTHONPATH", ""))
            print(f"  pyxrt      {found} (added to PYTHONPATH)")
        else:
            print("PyXRT is not importable and was not found on this host.\n"
                  "Without it mlir-aie silently selects a CPU-only tensor class, "
                  "iron.zeros(device=\"npu\") is rejected, and using \"cpu\" "
                  "instead makes the export try to RUN the design rather than "
                  "compile it -- so the build cannot start at all.\n"
                  "Install XRT's python bindings for THIS interpreter, or point "
                  "PYTHONPATH at the directory holding pyxrt*.so.",
                  file=sys.stderr)
            return 1

    if not toolchain_present():
        print("The IRON toolchain is not importable on this interpreter.",
              file=sys.stderr)
        print("", file=sys.stderr)
        print("    source ironvenv/bin/activate", file=sys.stderr)
        print("", file=sys.stderr)
        print("Without it the failure is `ModuleNotFoundError: No module named "
              "'aie'\nwhich reads as a broken checkout rather than a shell that "
              "was never set up.", file=sys.stderr)
        return 1

    dst = Path(os.path.abspath(args.dst))
    common = spec["common"]
    print(f"Building into {dst}")
    print(f"  toolchain: {sys.executable}")
    print()

    t_all = time.time()
    built = skipped = 0
    failed: list[str] = []
    for fam in families:
        # SERIAL, on purpose. See the module docstring.
        name, ok, _ = build_one(fam, common, dst, args.force)
        if not ok:
            failed.append(name)
        elif any(f["name"] == name and f is not fam for f in families):
            skipped += 1
        else:
            built += 1

    print()
    print(f"built {built}, skipped {skipped}, failed {len(failed)} "
          f"({int(time.time() - t_all)} s total)")
    if failed:
        print(f"failed: {', '.join(failed)}", file=sys.stderr)

    rc = 1 if failed else 0
    if not args.skip_check:
        print()
        print("Checking the built sets against families.json:")
        p = subprocess.run([sys.executable, "check_design_sets.py", "--xclbins", str(dst)],
                           cwd=GEMM_RTP)
        rc = rc or p.returncode
    return rc


if __name__ == "__main__":
    sys.exit(main())
