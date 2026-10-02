---
name: ground-truth-verify
description: Verify every load-bearing plan claim against primary sources before implementing or benching: spec text, in-tree code at working HEAD, vendor docs, driver and kernel sources. Clone or fetch dependencies pinned by SHA as needed. Use before any kernel, scheduler, driver-facing, or contract-touching change, and whenever a plan cites a spec, a register or arch number, or an external limit.
---

# Ground Truth Verify

Three near-filed driver bugs on this branch dissolved the moment the driver
source was read instead of a runtime reading: a context-handling path that
exists, a sensor whose value field is narrower than assumed, and a provenance
table that did not match the files it describes. A fourth claim -- "NPU power is
1000x too high" -- was a unit error, settled by reading the sysfs attribute's
units rather than by arguing about the magnitude.

A claim that sounds like a defect is more often a wrong-layer inference, a
missing unit, or a stale table. This skill forces source verification before
code, because the cost of the wrong order is a confident bug report against
working code.

## Trigger

- Before implementing any kernel, scheduler, tile, driver-facing call, or
  contract-touching change.
- Before benching a mechanism claim, or attributing a delta to a code path.
- Whenever a plan cites a specification, an arch or register number, a limit, a
  supported feature, or a source file.
- Whenever a number is described as "obviously wrong", "far too large", or
  "should be", with no unit attached.
- Before reporting a defect in code you did not write.

## Preconditions + refusal conditions

- **The plan's load-bearing claims are written down**, each as one checkable
  sentence: the contract rule, the figure, the serving path, the limit, the
  provenance. If they are not written down, write them first. An unwritten plan
  cannot be verified, and unverifiable plans stop here.
- **Working HEAD recorded**, and somewhere to store quotes. The project's
  investigation directory (`project-scaffold` layout) is the right home: an
  evidence file is only useful if a later reader can check it.

Refuse to proceed, and say which claim is unverified, when:

- The only source available is a discussion, an issue, a commit message, a
  blog, or a model's memory. These are leads, not ground truth. An issue
  describing a fix is evidence the reporter *believed* something.
- A cited source cannot be reached. Record it as **unverifiable**, name what
  would settle it, and carry the risk explicitly. An unreachable upstream is not
  licence to assume the local copy is canonical, and it is never evidence that
  the local copy agrees with upstream.
- The claim is a capability claim ("supports", "never faults", "driver
  absorbs") with no source chain behind it.

## Requires + Never

Requires:

- Shell: `git clone --depth 1` for a dependency, deepened only far enough to
  reach the cited commit, then `git rev-parse HEAD` to record the SHA. Record
  remote and SHA at the time of reading. Clone into a recorded path, never
  into a temp directory that will be cleaned up before the claim is checked.
- Shell: `grep`, `rg`, and a read of the file at the cited commit -- not at
  whatever the working tree has drifted to.
- Read: specification text, in-tree source at working HEAD, vendor
  documentation, and the driver or kernel sources behind any hardware claim.
- Record: the quote or the code line, with its SHA and line numbers, in the
  investigation file.

Never:

- Cite a discussion, issue, or commit message as a specification.
- Trust a line number carried over from an earlier reading, a plan, or another
  document. Re-grep. Line numbers are the single most common stale citation.
- Transfer a number across architectures, vendors, or versions without a local
  measurement. Other-configuration numbers transfer **qualitatively only**.
- Assert a hardware or driver tolerance from lore, from another project, or from
  the fact that it has never failed for you.
- Report a defect in unread code. Read the function, including its error paths.
- Interpret a runtime observation as a capability. See step 7.
- Push, open or edit a PR or issue, write a review reply, or use sudo.

## Workflow

1. **List the load-bearing claims.** Every assertion the change depends on. If
   the change has no such list, the change is not specified well enough to
   verify.

2. **Obtain each source, pinned.** Clone or fetch every dependency the claims
   rest on, at a specific SHA, into a recorded path. Shallow first; deepen only
   to reach the commit actually being cited.

3. **Contract claims: quote the specification verbatim**, with its revision or
   issue number and a link. Numbers-first, spec-second is refused -- a number
   in a plan that has no quoted spec behind it is a guess with a citation
   added later.

4. **Code claims: re-verify at working HEAD.** Re-grep every file, line and
   symbol the plan names, and record SHA plus line numbers. When a named symbol
   is missing, classify it **at the spot** as new work or as a confirmation that
   already exists; never drop it silently, because a symbol that turns out to
   already exist has usually been implemented twice. If working HEAD is behind
   upstream on any path the change touches, run `upstream-prior-art` before
   calling anything new.

5. **Architecture and limit claims: probe locally.** Read the device's own
   capability surface rather than inferring it, and compare its native types
   and layouts against the cited source. A field's declared width is the kind
   of thing that is wrong in a plan and decisive in a diagnosis: read the
   struct, not the printout.

6. **Tolerance claims: name the chain or concede.** "The driver absorbs it" and
   "the hardware never faults" are only acceptable with the mechanism named --
   which field, which check, which code path -- and quoted. Otherwise concede
   the claim. Many projects already carry a machine-readable list of
   unimplemented paths; if one exists, it is a source here, not a footnote.

7. **Path and capability claims: identify the serving pipeline, and keep layers
   apart.** Before attributing any delta or any behavior to a code path,
   establish that the path is actually the one that runs -- a table name, a
   registry entry, and a function that exists are all evidence that code was
   *written*, not that it is *reached*.

   For each claim, name the layer it lives at -- specification text, host code,
   build flags, device pass, driver source, runtime behavior -- and take
   evidence **only** at that layer:

   | claim about | only valid evidence |
   |---|---|
   | what the spec permits | spec text at a named revision |
   | what the code does | the code at a recorded SHA |
   | what the build enabled | build flags, or the built artifact |
   | what the hardware/driver offers | driver source, or a probe that returns |
   | what a run did | the run's own output or counters |

   **Wrong-layer evidence is refused however thorough it looks.** Three
   specific traps, all hit on this branch:
   - A probe that returns zero records does not show the feature is absent. It
     shows the probe is wrong, or the node is unselected, or the protocol
     version is. Only the driver source distinguishes those.
   - A tool being installed, documented, or listed in a man page is not
     evidence it works. Run it and inspect what it produced: one capture tool
     here emitted a file byte-identical to its own input and a manifest of null
     fields, and would have been reported as working on its presence.
   - A reading with no unit is a units bug until proven otherwise. A value in
     milliwatts compared against an expectation in watts is off by 1000 and
     reads as a hardware fault.
   - **Documentation is rendered against some release, not necessarily yours.** A
     vendor or kernel doc page states the version it was built from; your host
     runs something else. Record both and treat the delta as unverified. Check
     the running component's own reported version rather than assuming your
     checkout matches it -- and the reverse: a checkout can sit at a commit the
     running module never saw. Here the driver checkout and the running module
     happened to agree exactly, which is worth confirming rather than assuming.
   - **Resolve toolchain paths by globbing, never by hardcoding a version.**
     Interpreter minor versions and package install roots both drift. An
     `AGENTS.md` here named `python3.12` for a venv that had moved to `3.14`,
     so the documented path did not exist. Glob, then *verify what you got* --
     print the version, confirm the target triple you need is registered -- rather
     than assuming the glob matched.
   - **A version macro may not exist in the version you expect.** Two installs of
     one library exposed disjoint macros, so a check written against the older
     layout passed *vacuously* against the newer. Handle every layout you might
     meet, and treat "macro absent" as a distinct outcome from "version wrong" --
     they need different fixes. A silently-never-matching regex looks identical
     to a missing file, so test the matcher against both layouts.
   - **A hash field may not be a hash of the bytes.** A manifest's `oid` turned
     out to be a git `blob_id`, which for LFS objects is the hash of the
     *pointer*, so it cannot be reproduced from a download. Verify against the
     field that is a content hash (`lfs.sha256`), and confirm your computed
     digest equals the *live* upstream value. When your own verification
     disagrees with a manifest, suspect the verification method first -- that is
     what happened here, twice, before the files turned out to be correct.

8. **No implementation and no bench** until every load-bearing claim has a
   quote with an SHA or a link. Write the stop conditions next to the claims, so
   the person who finds the claim false knows what it was supposed to enable.

## Verification

- Every load-bearing claim traces to one of: a spec quote with a revision, code
  lines anchored to a SHA, a probed capability, or a named mechanism chain.
- Unverifiable claims are listed as such, with the risk accepted in writing
  rather than absorbed silently.
- Dependency sources recorded with remote and SHA at reading time.
- A reader can check every claim from the document alone, without re-running
  anything and without trusting the author.

## Non-goals

- Proving the change is engaged. `prove-engaged` owns "did it run".
- Proving the change is correct. `prove-correctness` owns identity, conservation
  and gate resolution; this skill only establishes that the plan was grounded.
- Bench validity. `host-discipline` and `locked-clock-bench` own whether a
  number is comparable.
- Prior art and novelty. `upstream-prior-art` owns whether the work already
  exists somewhere.

## Concurrency

Read-only. Parallelizes without limit: no locks, no shared mutable state.
Clones go to per-task recorded paths, and two agents verifying the same source
share one clone rather than re-cloning over each other. Evidence files should be
written per task, not appended to a shared log.
