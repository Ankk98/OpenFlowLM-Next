#!/usr/bin/env python3
"""Generate a model_info.json entry from the HuggingFace (or ModelScope) tree API.

WHY THIS EXISTS. `src/model_info.json` is a map of model tag -> the file list
the model consists of, with each file's git oid, byte size, and LFS oid. It is
hand-maintained, and hand-maintaining an oid is how you end up with a manifest
that disagrees with the repository it claims to describe. `model_info.json` is
also the reason a fresh `oflm pull` can silently fetch nothing:

  * `src/pull/model_downloader.cpp` looks the tag up with `.at()`, which THROWS
    on a missing key. The throw was caught by a handler that printed an error
    and then returned an empty download list, so the pull reported success and
    downloaded nothing.
  * The loop that walks the model_list `files` array `continue`s when a file is
    absent from the manifest, so a partially-described model also reported
    success.

So adding a model needs an entry in BOTH `model_list.json` (which files the
model is) and `model_info.json` (where each one is and how big it is). This tool
produces the second one from the source of truth rather than from memory.

It reads the repository's own file list, so the oids cannot drift from the repo
they describe. It does not invent, and it does not guess: a file the model needs
that the repository does not contain is reported as an error, not skipped.

Usage:
    python utilities/sync_model_info.py --tag laya-decision:multilingual \\
        --repo convaiinnovations/laya

    # audit an existing entry without writing anything
    python utilities/sync_model_info.py --tag laya-decision:multilingual \\
        --repo convaiinnovations/laya --check

    # list what a model needs vs what the manifest describes
    python utilities/sync_model_info.py --tag laya-decision:multilingual \\
        --repo convaiinnovations/laya --report

Writes src/model_info.json in place unless --check or --report is given. Refuses
to write if the repository is missing any file that model_list.json requires.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_INFO = os.path.join(REPO_ROOT, "src", "model_info.json")
MODEL_LIST = os.path.join(REPO_ROOT, "src", "model_list.json")
USER_AGENT = "oflm-sync-model-info/1.0"


def fetch_hf_tree(repo: str, revision: str, timeout: int) -> list[dict]:
    """The file list for a HuggingFace repository, as the API returns it."""
    url = f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return json.load(fh)


def fetch_modelscope_tree(repo: str, revision: str, timeout: int) -> list[dict]:
    """ModelScope's tree API. Key names differ from HF's, so normalise here."""
    url = f"https://modelscope.cn/api/v1/models/{repo}/repo/files?Revision={revision}&Root="
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        payload = json.load(fh)
    files = payload.get("Data", {}).get("Files", [])
    out = []
    for f in files:
        if f.get("Type") != "blob":
            continue
        entry = {
            "type": "file",
            "oid": f.get("Sha256") or f.get("RevisionId") or "",
            "size": f.get("Size", 0),
            "path": f.get("Path", ""),
        }
        if f.get("RevisionId"):
            entry["lfs"] = {"oid": f["RevisionId"], "size": f.get("Size", 0)}
        out.append(entry)
    return out


def required_files(tag: str) -> list[str]:
    """The files model_list.json says this tag consists of, and the repo it comes from.

    `name` in model_list.json is a DIRECTORY name, not a repository id, so the
    repo id has to come from the tag and the `url`, never from `name`.
    """
    with open(MODEL_LIST, encoding="utf-8") as fh:
        models = json.load(fh)["models"]
    family, _, size = tag.partition(":")
    if family not in models:
        raise SystemExit(f"model_list.json has no family {family!r} (tag {tag!r})")
    variants = models[family]
    key = size or next(iter(variants))
    if key not in variants:
        raise SystemExit(f"model_list.json family {family!r} has no variant {key!r}")
    entry = variants[key]
    files = list(entry.get("files") or [])
    if not files:
        raise SystemExit(f"model_list.json entry for {tag!r} lists no files")
    url = entry.get("url") or ""
    if not url:
        raise SystemExit(
            f"model_list.json entry for {tag!r} has no `url`, so the repository "
            f"cannot be determined. Pass --repo explicitly."
        )
    repo = url.rstrip("/").split("huggingface.co/")[-1]
    return files, repo, entry


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", required=True, help="model tag as in model_list.json, e.g. laya-decision:multilingual")
    ap.add_argument("--repo", help="HF repo id; default derived from model_list.json `url`")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--provider", choices=("hf", "modelscope"), default="hf")
    ap.add_argument("--check", action="store_true", help="verify the existing entry, write nothing")
    ap.add_argument("--report", action="store_true", help="print required vs described, write nothing")
    ap.add_argument("--timeout", type=int, default=60)
    args = ap.parse_args()

    try:
        files, derived_repo, entry = required_files(args.tag)
    except SystemExit as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    repo = args.repo or derived_repo

    print(f"tag     {args.tag}")
    print(f"repo    {repo}  (revision {args.revision}, provider {args.provider})")
    print(f"model   {entry.get('name')}  family={entry.get('npue_design_family', '-')}")
    print(f"needs   {len(files)} file(s)")

    fetch = fetch_hf_tree if args.provider == "hf" else fetch_modelscope_tree
    try:
        tree = fetch(repo, args.revision, args.timeout)
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: could not read the repository tree: {exc}", file=sys.stderr)
        print(
            "       Refusing to write a manifest from anything but the real tree. "
            "Hand-written oids are how a manifest drifts from its repository.",
            file=sys.stderr,
        )
        return 1

    by_path = {e["path"]: e for e in tree if e.get("type") == "file"}
    print(f"tree    {len(by_path)} file(s) in the repository")

    missing = [f for f in files if f not in by_path]
    if missing:
        print()
        print(f"ERROR: the repository does not contain {len(missing)} required file(s):", file=sys.stderr)
        for m in missing:
            print(f"         {m}", file=sys.stderr)
        print(
            "       Refusing to write. Either model_list.json names a file the "
            "repository does not have, or the revision is wrong.",
            file=sys.stderr,
        )
        return 1

    # Only the files the model consists of. The manifest's job is to describe
    # THIS model, and a file the model does not use is a file the downloader
    # will never look up.
    described = [by_path[f] for f in files]

    if args.report:
        print()
        print("required vs described:")
        for f in files:
            e = by_path[f]
            oid = e.get("lfs", {}).get("oid", e.get("oid", ""))
            print(f"  {f:<52} {e.get('size', 0):>13,}  {oid[:16]}")
        total = sum(e.get("size", 0) for e in described)
        print(f"  {'TOTAL':<52} {total:>13,}")
        return 0

    with open(MODEL_INFO, encoding="utf-8") as fh:
        manifest = json.load(fh)

    if args.check:
        have = manifest.get(args.tag)
        if have is None:
            print(f"\nCHECK  FAIL: {args.tag!r} is absent from model_info.json")
            return 1
        have_by_path = {e.get("path"): e for e in have}
        bad = []
        for f in files:
            want, got = by_path[f], have_by_path.get(f)
            if got is None:
                bad.append(f"{f}: not described")
                continue
            for key in ("oid", "size"):
                if got.get(key) != want.get(key):
                    bad.append(f"{f}: {key} {got.get(key)!r} != repository {want.get(key)!r}")
            w_oid = want.get("lfs", {}).get("oid")
            g_oid = got.get("lfs", {}).get("oid")
            if (w_oid or None) != (g_oid or None):
                bad.append(f"{f}: lfs oid {g_oid!r} != repository {w_oid!r}")
        if bad:
            print(f"\nCHECK  FAIL: {len(bad)} disagreement(s) with the repository:")
            for b in bad:
                print(f"         {b}")
            return 1
        print(f"\nCHECK  OK: {args.tag!r} matches the repository at {args.revision}")
        return 0

    if args.tag in manifest:
        print(f"\n{args.tag!r} already present; replacing its entry with the current tree.")
    manifest[args.tag] = described
    tmp = MODEL_INFO + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=4)
        fh.write("\n")
    os.replace(tmp, MODEL_INFO)
    print(f"\nwrote {len(described)} file entries for {args.tag!r} into {os.path.relpath(MODEL_INFO, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
