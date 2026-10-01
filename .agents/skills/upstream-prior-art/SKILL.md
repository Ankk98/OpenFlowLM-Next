---
name: upstream-prior-art
description: Check upstream for merged, open and in-flight work before starting a task, before editing anything in a synced or vendored directory, and again before any PR-ready verdict. Fetch, diff the paths you plan to touch, search PRs and issues in all states, verify vendored-file pins. Use at task start, when porting or rebasing, on resume, and before claiming anything is new or ship-ready.
---

# Upstream Prior Art

Checking is cheap. Not checking is how you spend a day rebuilding something that
merged while you worked, or how you edit a file whose whole point is that it is a
copy.

Two real examples from this repo, both of which a first-step check would have
caught:

* **`upstream/main` had never been fetched.** The `upstream` remote is configured
  and pointed at the canonical repository, so every `git log upstream/main` failed
  and nothing looked wrong -- the branch simply had no upstream to compare against.
  Nine vendor-port commits old, and nobody had asked whether the Laya work
  existed upstream.
* **Nine of the thirty-one pinned files in `src/open_npue/` no longer match their
  recorded hashes**, because work was done in the copy rather than in its source.
  `SYNCED.md` in that directory says, in its first paragraph, to edit it upstream
  and not here -- and the accuracy gates that make the numbers mean anything live
  upstream, so a local edit does not just risk a conflict, it detaches the code
  from the checks that justify it.

Neither is a judgement call about whether the work was any good. Both are
mechanical, and both are checkable in under a minute.

## Trigger

- Starting any task, before the baseline row (`project-scaffold`).
- **Before editing any file in a directory that declares itself synced, vendored
  or generated.** Find the declaration first: `SYNCED.md`, `VENDOR.md`, a header
  banner, or a generator script named in a comment.
- Porting, rebasing, or picking up someone else's branch.
- Before any PROMOTE, PR-ready, or "this is new work" claim.
- Resuming after a pause in an active area.

## Preconditions + refusal conditions

- The **touched-path list**: files or directories you expect to change.
- Three to five **keyword sets**: model name, architecture, the feature, planned
  new file names.
- The **upstream table below**, filled in with the real remotes for this repo.
- **Refuse to start** if you cannot name which upstream owns a file you plan to
  edit. "It is probably ours" is the exact state this skill exists to end.

- The **upstream table** in `references/upstreams.md` -- which remote owns
  which path, and which file hashes are pinned where.

## Requires + Never

Requires: shell `git fetch`, `git log`, `git rev-parse`, `gh search prs`,
`gh search issues`, `gh pr view`, `sha256sum` -- all **read-only**; read the
upstream table above.

Never: add or change remotes, or push; comment on, react to, or close a PR or
issue; treat zero search hits as proof without the `git log` check, because titles
miss; port a branch before its provenance is known; **edit a pinned file in place
"just to try something"** -- that is the failure this skill is named for.

## Workflow

1. **Fetch over HTTPS, by URL, and record the SHA immediately.** The next fetch
   overwrites `FETCH_HEAD`, and two agents on one repo overwrite each other.
   ```bash
   set -e                      # a failed fetch must abort, not print and continue
   git fetch https://github.com/OWNER/REPO main
   UP=$(git rev-parse --short FETCH_HEAD); date -Is        # write both down now
   ```
   **Why HTTPS by URL and not the `upstream` remote:** this check is read-only,
   so it needs no credentials, and it must work when the SSH path does not. On
   this host the SSH route is broken -- `ssh-agent` is running with keys loaded,
   but signing fails with `sign_and_send_pubkey: ... communication with agent
   failed` followed by `Permission denied (publickey)`, so `git fetch upstream`
   exits 128. HTTPS works for public repositories.

   **Check the exit code.** A `git fetch ... | tail` swallows a 128 and leaves
   `FETCH_HEAD` empty, which then reads as "no upstream changes" rather than
   "the check did not run". That is the same failure shape as every other inert
   tool in this project: it looked like a clean result.
2. **Pin check, before any edit.** If the target directory declares a file table,
   verify the files you intend to touch still match it:
   ```bash
   python3 - <<'EOF'
   import re,hashlib,pathlib
   t=pathlib.Path("src/open_npue/SYNCED.md").read_text()
   for f,pin in re.findall(r"^\\|\\s*`([\\w.]+)`\\s*\\|\\s*`?([0-9a-f]{8,})`?\\s*\\|",t,re.M):
       p=pathlib.Path("src/open_npue")/f
       if p.is_file():
           got=hashlib.sha256(p.read_bytes()).hexdigest()[:len(pin)]
           print(("MATCH   " if got==pin else "MISMATCH"),f,pin,got)
   EOF
   ```
   A `MISMATCH` is not an error to tidy away -- it is a **process question**: was
   this file already edited locally, and if so does the change belong upstream?
   A file in the table that does **not** match means the copy is already detached;
   say so before adding to it.
3. **Code drift on the paths you plan to touch.**
   `git log --oneline HEAD..FETCH_HEAD -- <paths>`. Read every hit before writing
   code. A new upstream file whose name matches one you planned to add means stop.
4. **Search, in all states, including merged and closed.**
   `gh search prs --repo <owner>/<repo> "<kw>" --state all`, and the same for
   issues, per keyword set. Record number, state and date per hit. Merged is the
   state that matters most: it is the one that makes your work a duplicate.
5. **Provenance, for anything you did not write.** Find the author and any PR from
   them. A maintainer's work-in-progress branch usually means a PR is coming --
   ask the human before building on it.
6. **Classify, in writing:** `duplicate` (stop; measure upstream instead),
   `overlap` (rebase onto it, scope only the delta), or `new` (proceed).
7. **Re-run steps 1-4 before every PROMOTE / PR-ready verdict and on resume.**
   Put the fetched SHA next to the verdict, so a later reader can tell how fresh
   the novelty claim was.

## Verification

- Every task note and every verdict carries the fetched SHA, the fetch date, and
  the hits -- or the literal string `none` **with the exact queries that produced
  it**.
- The pin check ran, and its output is in the task note, including any pre-existing
  `MISMATCH`.
- A class (`duplicate` / `overlap` / `new`) is stated with the evidence that
  supports it, not asserted.
- A third party can re-run steps 1-4 and get the same answer from the recorded
  SHA.

## Non-goals

- Whether the idea is good, or whether the numbers are right. That is
  `prove-correctness` and `ground-truth-verify`.
- Performance claims. `perf-profile` owns those.
- Any GitHub **write**. The human writes every word that leaves the machine.
- Reconciling a detached copy. This skill surfaces the question; deciding whether
  to upstream a change or fork it deliberately is the human's call.

## Concurrency (read-only skill)

Parallelizes freely -- it reads, and fetches are additive.

`FETCH_HEAD` is per-repo and **shared**, so two agents on one repo will overwrite
each other's SHA. Record the SHA and date in your own notes immediately after
fetching, and treat another agent's newer SHA as authoritative for "has upstream
moved" while keeping yours for "what did I compare against".
