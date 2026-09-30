# Traces: OPEN-ENC-MODERNBERT (canonical spec: specs/open-engine/spec.md)
"""The byte-level BPE tokenizer against HuggingFace, id for id.

The corpus is generated FROM the real `multilingual/tokenizer/tokenizer.json` of
convaiinnovations/laya at `/resolve/main/` -- NOT `/raw/main/`, which returns a
133-byte git-lfs pointer for this file, so a fixture generated from `raw` is a
fixture for the pointer.

The oracle is `tokenizers` itself, run right now, through the same call the
model's own `encode_text()` makes: `add_special_tokens=False`. That is the whole
gate. Not a spot check, not a round trip through this repository's own decoder:
every id, on every string, or the phase fails.

WHAT IS ACTUALLY BEING EXERCISED, because "the tokenizer agrees with HF" is not
one thing:

  * Metaspace pre-tokenization. Runs no regex: it cuts in front of every U+2581
    keeping it with the piece that follows. `▁▁x` is `["▁", "▁x"]`, which is the
    boundary this test exists to pin.
  * A `Replace(" " -> "▁")` normalizer whose `pattern` is a literal string.
    Without it this vocabulary does not tokenize at all -- it has no raw space.
  * Added-token pre-matching, leftmost-LONGEST, on the RAW text. All 249 added
    tokens here carry `normalized: false`, which in HuggingFace means "match the
    ORIGINAL input", and the space-run tokens are stored as U+2581 runs -- so a
    space run can never match once the normalizer has rewritten it. This is the
    one place the design plan's stated order was wrong, and
    `test_the_added_token_stage_runs_before_the_normaliser` is the test that
    says so out loud.
  * 88 added tokens that are strict PREFIXES of other added tokens, so
    leftmost-longest is load-bearing rather than an optimisation.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
GEN = REPO / "src/open_npue/bbpe_tokenizer_gen.cpp"
TOK = REPO / "src/open_npue/tokenizer_bbpe.cpp"
TOKH = REPO / "src/open_npue/tokenizer_bbpe.hpp"
CACHE = Path(os.environ.get("NPUE_FIXTURE_CACHE", "/tmp/oflm-fixtures"))

REPO_ID = "convaiinnovations/laya"
COMMIT = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
#: /resolve/main/, NOT /raw/main/. The raw path serves an lfs pointer for this
#: file, so a fixture built from it tests the pointer.
TOKENIZER_URL = f"https://huggingface.co/{REPO_ID}/resolve/{COMMIT}/multilingual/tokenizer/tokenizer.json"

CXX = shutil.which("g++") or shutil.which("clang++")

MIN_STRINGS = 200


def _tokenizer_json() -> Path:
    p = CACHE / "laya-multilingual-tokenizer.json"
    if p.exists() and p.stat().st_size > 1_000_000:
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(TOKENIZER_URL, timeout=300) as r:
        data = r.read()
    # An lfs pointer is ~130 bytes. A 34 MB file is not one, and asserting it is
    # how a future run cannot quietly start testing the wrong bytes.
    assert len(data) > 1_000_000, f"got {len(data)} bytes -- this looks like an lfs pointer"
    p.write_bytes(data)
    return p


# --------------------------------------------------------------------------
# the corpus


def build_corpus(tok: dict) -> list[str]:
    """Every shape this tokenizer can get wrong, in one list.

    Generated FROM the checkpoint's own added_tokens and normalizer, so it cannot
    drift from the file it is testing -- a hand-written corpus for a 256k
    vocabulary is a corpus that happens to miss whatever the vocabulary has."""
    R = "\u2581"
    added = [a["content"] for a in tok["added_tokens"]]
    strings: list[str] = []

    # 1. Every added token, alone and wrapped, so each is reached by the matcher
    #    and not only as a side effect of a neighbour.
    strings += added
    for a in added[:60]:
        strings.append(f"prefix {a} suffix")

    # 2. The prefix chains. 88 added tokens are strict prefixes of others, so a
    #    shortest-first matcher splits a run of markers into single characters
    #    and every one of these diverges.
    by_len = sorted(added, key=len)
    for a in by_len:
        longer = [b for b in added if b.startswith(a) and len(b) > len(a)]
        if longer:
            strings.append(longest_marker_run_probe(a, longer))

    # 3. The space runs, which is where the added-token stage's ORDER matters:
    #    a raw run of spaces normalises to a run of U+2581 and must NOT then be
    #    re-matched as the U+2581 added token.
    for n in range(1, 12):
        strings.append(" " * n)
        strings.append(" " * n + "word")
        strings.append("word" + " " * n)
        strings.append("word " + " " * n + " word")

    # 4. Newlines and tabs, in runs, because those added tokens exist and are
    #    long enough to be reached only from a run.
    for n in (1, 2, 3, 5, 8, 31):
        strings.append("a" + "\n" * n + "b")
        strings.append("a" + "\t" * n + "b")
        strings.append("\n" * n)
        strings.append("\t" * n)

    # 5. Ordinary text, with and without a leading space, because
    #    prepend_scheme="always" only prepends when the segment does not already
    #    begin with the marker -- and after the Replace normalizer a leading
    #    space HAS become one.
    strings += [
        "hello world", " hello world", "hello", "a", " ", "",
        "The quick brown fox jumps over the lazy dog.",
        "  leading and trailing  ",
        "\ttabbed\tvalue\t", "line1\nline2\nline3",
        "choice: yes", " choice: yes", "choice:yes",
        "score: 3", "level 0: Can wait", " level 2: Needs attention today",
        "is this spam?", " is this a duplicate charge?",
        "true", " false", "no, the statement does not hold",
    ]

    # 6. The mask token and its literal neighbours. `<mask>` is the ONLY added
    #    token with lstrip:true, and `[MASK]` is ordinary text -- a string that
    #    contains it and gets it treated as special is a wrong prompt.
    strings += ["<mask>", "a <mask> b", "  <mask>  ", "[MASK]", "a [MASK] b",
                "<MASK>", "MASK", "<unused0>", "<unused99>", "<2mass>",
                "[@BOS@]", "<pad>", "<unk>", "<s>", "</s>", "<bos>", "<eos>"]

    # 7. Non-Latin scripts, emoji, and combining marks. Each is a multi-byte
    #    codepoint, so each exercises the byte map and the replacement-character
    #    boundary at once.
    strings += ["héllo wörld", "日本語のテキスト", "Привет мир", "العربية",
                "עברית", "हिन्दी", "한국어", "中文测试", "🎯 target 🏁",
                "family 👨‍👩‍👧‍👦 emoji", "écombining", "ｆｕｌｌｗｉｄｔｈ",
                "mixed ascii 日本語 🎯 end"]

    # 8. JSON-ish and markup-ish text: the prompt builder renders criteria as
    #    JSON, so braces and quotes are ordinary input.
    strings += ['{"a": 1, "b": [2, 3]}', "a: 1", "level 0: {\"k\": \"v\"}",
                "<b>bold</b>", "<div class=\"x\">y</div>", "a &amp; b",
                "100%", "$12.50", "3/4", "C:\\path\\to\\file"]

    # Deduplicate, keep order, and make sure there is enough of it: a corpus
    # that quietly shrank below MIN_STRINGS would still pass every comparison.
    seen, out = set(), []
    for s in strings:
        if s not in seen:
            seen.add(s)
            out.append(s)
    assert len(out) >= MIN_STRINGS, f"corpus is only {len(out)} strings"
    return out


def longest_marker_run_probe(a: str, longer: list[str]) -> str:
    """A string whose marker run is ambiguous unless the matcher is
    leftmost-longest: `a` followed by the tail of the longest extension."""
    longest = max(longer, key=len)
    if not longest.startswith(a):        # pragma: no cover
        return a + longest
    return longest + "x"


@pytest.fixture(scope="module")
def corpus_and_oracle():
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer

    p = _tokenizer_json()
    raw = json.loads(p.read_text())
    tk = Tokenizer.from_file(str(p))
    strings = build_corpus(raw)
    oracle = {s: tk.encode(s, add_special_tokens=False).ids for s in strings}
    return raw, strings, oracle


# --------------------------------------------------------------------------
# the C++ driver


DRIVER = r"""
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "tokenizer_bbpe.hpp"

// Reads a UTF-8 file of NUL-separated strings, tokenizes each, and prints the
// ids NUL-separated. One process, one table load: a 34 MB blob per string would
// make the gate slow enough that people stop running it.
int main(int argc, char **argv) {
  if (argc < 3) return 2;
  std::ifstream f(argv[1], std::ios::binary);
  std::stringstream ss; ss << f.rdbuf();
  const std::string blob = ss.str();

  npue::BbpeTokenizer t;
  try { t = npue::BbpeTokenizer::from_table_bytes(blob.data(), blob.size()); }
  catch (const std::exception &e) {
    std::fprintf(stderr, "TABLE REFUSED: %s\n", e.what()); return 3;
  }
  std::fprintf(stderr, "TABLE vocab=%zu unk=%d mask=%d norm=%u pretok=%u prepend=%u split=%u bf=%u fu=%u\n",
               t.vocab_size(), t.unk_id, t.mask_id, t.normalizer,
               t.pre_tokenizer, t.prepend_scheme, (unsigned)t.metaspace_split,
               (unsigned)t.byte_fallback, (unsigned)t.fuse_unk);

  std::ifstream g(argv[2], std::ios::binary);
  std::stringstream gs; gs << g.rdbuf();
  const std::string in = gs.str();
  size_t start = 0;
  std::string out;
  while (start <= in.size()) {
    const size_t z = in.find('\0', start);
    if (z == std::string::npos) break;
    const std::string s = in.substr(start, z - start);
    start = z + 1;
    std::vector<int32_t> ids;
    try { ids = t.tokenize(s); }
    catch (const std::exception &e) {
      std::fprintf(stderr, "TOKENIZE REFUSED for one string: %s\n", e.what());
      return 4;
    }
    for (int32_t id : ids) { out += std::to_string(id); out += ' '; }
    out += '\0';
  }
  std::cout << out;
  return 0;
}
"""

GEN_DRIVER = r"""
#include <cstdio>
#include <fstream>
#include <string>
#include "bbpe_tokenizer_gen.hpp"
int main(int argc, char **argv) {
  if (argc < 3) return 2;
  std::vector<uint8_t> t;
  try { t = npue::generate_bbpe_tokenizer_table(argv[1]); }
  catch (const std::exception &e) {
    std::fprintf(stderr, "GEN REFUSED: %s\n", e.what()); return 1;
  }
  std::ofstream o(argv[2], std::ios::binary);
  o.write(reinterpret_cast<const char *>(t.data()), (std::streamsize)t.size());
  return 0;
}
"""


def _compile(tmp_path: Path, src: str, name: str, extra: list[str]) -> Path:
    f = tmp_path / f"{name}.cpp"
    f.write_text(src)
    exe = tmp_path / name
    p = subprocess.run([CXX, "-std=c++17", "-O2", "-o", str(exe), str(f)]
                       + [str(REPO / "src/open_npue" / s) for s in
                          ("bbpe_tokenizer_gen.cpp", "tokenizer_bbpe.cpp",
                           "json_min.cpp")]
                       + extra + ["-I", str(REPO / "src/open_npue")],
                       capture_output=True, text=True)
    assert p.returncode == 0, "driver failed to build:\n" + p.stderr[-6000:]
    return exe


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    if CXX is None:                             # pragma: no cover
        pytest.skip("no C++ compiler on this host")
    d = tmp_path_factory.mktemp("bbpe")
    gen = _compile(d, GEN_DRIVER, "gen", [])
    out = d / "table.bin"
    p = subprocess.run([str(gen), str(_tokenizer_json()), str(out)],
                       capture_output=True, text=True)
    assert p.returncode == 0, "the generator refused this checkpoint:\n" + p.stderr
    return out


@pytest.fixture(scope="module")
def tokenizer(table, tmp_path_factory):
    d = tmp_path_factory.mktemp("bbpe_run")
    exe = _compile(d, DRIVER, "run", [])
    return exe


def _run_all(exe: Path, table: Path, strings: list[str]) -> dict:
    inp = exe.parent / "in.bin"
    inp.write_bytes(b"".join(s.encode() + b"\0" for s in strings))
    p = subprocess.run([str(exe), str(table), str(inp)], capture_output=True)
    assert p.returncode == 0, p.stderr.decode()[-4000:]
    rows = p.stdout.split(b"\0")
    assert rows[-1] == b"", "the driver must end every row with a NUL"
    return {s: [int(x) for x in rows[i].split()] if rows[i] else []
            for i, s in enumerate(strings)}, p.stderr.decode()


# --------------------------------------------------------------------------
# the gate


def test_the_generator_accepts_this_checkpoint(table):
    """It used to refuse at FOUR independent gates, and each of them encoded an
    assumption about a GPT-2-shaped byte-level BPE that this checkpoint does not
    share. That is why this is worth a phase on its own: the tokenizer for this
    model was written, generated, compiled and linked into the binary before
    this work and had never been called."""
    assert table.exists() and table.stat().st_size > 1_000_000


def test_every_id_matches_huggingface(tokenizer, table, corpus_and_oracle):
    """"THE GATE. The C++ tokenizer's ids are exactly
    `tokenizer(text, add_special_tokens=False)["input_ids"]`.

    Any single divergence fails the phase."""
    _, strings, oracle = corpus_and_oracle
    got, banner = _run_all(tokenizer, table, strings)
    # The banner is a claim about what the table says, printed on every run so a
    # reader can see the flags the comparison was made under.
    assert "pretok=1" in banner, banner
    assert "norm=2" in banner, banner

    bad = [(s, got[s], oracle[s]) for s in strings if got[s] != oracle[s]]
    assert not bad, (
        f"{len(bad)} of {len(strings)} strings differ from HuggingFace.\n"
        + "\n".join(f"  {s!r}\n    got    {g}\n    want   {o}"
                    for s, g, o in bad[:12])
    )


def test_the_added_token_stage_runs_before_the_normaliser(corpus_and_oracle):
    """The design plan said "normalise, then match added tokens against the
    NORMALISED text". That is wrong for this checkpoint, and the correction is
    measurable rather than arguable.

    Every added token here carries `normalized: false`, which in HuggingFace
    means "match the ORIGINAL input". And the space-run added tokens are stored
    as U+2581 runs (ids 139..168), so after `Replace(" " -> "▁")` a run of
    SPACES can no longer match anything -- while a run of literal U+2581 still
    does. Both are asserted against tokenizers, because the whole point is that
    the two differ."""
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(_tokenizer_json()))
    R = "\u2581"
    # Two spaces -> two single markers, NOT the `▁▁` added token.
    assert tk.encode("  x", add_special_tokens=False).tokens == [R, R + "x"]
    # The same two markers, written literally, ARE the added token.
    assert tk.encode(R * 2 + "x", add_special_tokens=False).tokens[0] == R * 2
    ids = tk.encode(R * 2, add_special_tokens=False).ids
    assert ids == [139], ids


def test_the_added_token_prefix_chains_are_leftmost_longest(corpus_and_oracle):
    """88 of the 249 added tokens are strict prefixes of others. A matcher that
    takes the first match found rather than the longest splits a run of markers
    into single characters, and a four-space run is the case a reader would
    notice if any reader looked at ids."""
    raw, _, _ = corpus_and_oracle
    contents = {a["content"] for a in raw["added_tokens"]}
    chains = [c for c in contents if any(o != c and o.startswith(c) for o in contents)]
    assert len(chains) >= 80, f"expected the documented prefix chains, found {len(chains)}"
    assert all(a["single_word"] is False and a["rstrip"] is False
               for a in raw["added_tokens"]), \
        "the generator refuses single_word and rstrip; if this checkpoint set one, that refusal is live"


def test_only_the_mask_token_has_lstrip(corpus_and_oracle):
    """`<mask>` is the only added token with lstrip:true in this checkpoint, and
    lstrip is the branch that OVERRIDES a match already accepted inside the
    whitespace it swallows -- the reason the added-token stage is two phases and
    not one."""
    raw, _, _ = corpus_and_oracle
    ls = [(a["content"], a["id"]) for a in raw["added_tokens"] if a["lstrip"]]
    assert ls == [("<mask>", 4)], ls


def test_the_blob_records_what_it_claims_and_the_reader_agrees(tokenizer, table):
    """The flags the comparison runs under are printed by the runtime itself, so
    a reader of a failure can see them without re-deriving them from the JSON."""
    p = subprocess.run([str(tokenizer), str(table), "/dev/null"],
                       capture_output=True)
    banner = p.stderr.decode()
    for field in ("norm=2", "pretok=1", "prepend=2", "split=1", "bf=1", "fu=1"):
        assert field in banner, f"{field} not in the runtime banner: {banner}"
    assert "unk=3" in banner and "mask=4" in banner, banner


def _v1_prefix(blob: bytes) -> bytes:
    """The version-1 layout, derived from the version-2 blob by parsing the
    version-1 prefix the same way the reader does.

    Derived rather than stored, so the test and the reader cannot disagree about
    where the tail starts without this failing -- which is the failure mode that
    would make "v1 is still readable" a claim about a byte offset nobody checks.
    """
    import struct

    p = 8 + 4                                   # magic + version
    _vocab, merges, added = struct.unpack_from("<III", blob, p + 8)
    p += 20                                     # norm aps vocab merges added
    p += 20                                     # cls sep pad unk mask
    for _ in range(2):                          # prefix, suffix counts + ids
        n = struct.unpack_from("<I", blob, p)[0]
        p += 4 + 4 * n
    for _ in range(_vocab):
        n = struct.unpack_from("<H", blob, p)[0]
        p += 2 + n
    p += 12 * merges
    for _ in range(added):
        n = struct.unpack_from("<H", blob, p)[0]
        p += 2 + n + 8
    return blob[:p]


def test_a_v1_table_is_still_readable(table, tmp_path):
    """The version-2 fields are APPENDED, so a version-1 table is a strict
    prefix of a version-2 one and a version-1 reader stops before them.

    No container in this tree embeds a BBPETOK1 blob today -- the tokenizer was
    written, generated, compiled and linked and had never been called -- but the
    blob is DATA, and data outlives the writer. A reader that only understood the
    version it was written with would turn a re-pack into a re-download for
    every checkpoint packed by an older build.
    """
    import struct

    blob = bytearray(table.read_bytes())
    assert len(_v1_prefix(bytes(blob))) < len(blob), \
        "the version-2 tail must be an APPEND"
    v1 = bytearray(_v1_prefix(bytes(blob)))
    # A real version-1 table: same prefix, version 1, no tail. Setting the
    # version back is the whole point -- a truncated version-2 table would be read
    # as a version-2 table with a truncated tail, which is a different (and much
    # louder) failure.
    struct.pack_into("<I", v1, 8, 1)
    v1 = bytes(v1)

    (tmp_path / "v1.bin").write_bytes(v1)
    p = subprocess.run([str(_RUN_EXE[0]), str(tmp_path / "v1.bin"), "/dev/null"],
                       capture_output=True)
    assert p.returncode == 0, p.stderr.decode()
    banner = p.stderr.decode()
    # Everything the version-2 tail carries defaults, by name and visibly. These
    # are the assertions that a reader is not walking off the end into whatever
    # follows: an uninitialised `prepend_scheme` would be a prepend nobody asked
    # for, and an uninitialised `byte_mode` would be either alphabet at random.
    assert "pretok=0" in banner, banner
    assert "prepend=0" in banner and "split=0" in banner, banner
    assert "bf=0" in banner and "fu=0" in banner, banner
    # `norm` is a VERSION-1 field, so it survives the truncation -- and that is
    # the compatibility claim in one line: the fields that could have been added
    # in the MIDDLE of the layout were not.
    assert "norm=2" in banner, banner
    # ...and it must still produce ids, in the byte_mode 0 alphabet.
    assert _tokenize_blob(v1, "hello world"), "a v1 table tokenized to nothing"


def test_an_unknown_blob_version_is_refused_by_name(table, tmp_path):
    """Reading a version this build does not implement as if it were the one it
    does is how a table's tail becomes somebody's vocabulary. Refuse, and name
    the version."""
    import struct

    blob = bytearray(table.read_bytes())
    struct.pack_into("<I", blob, 8, 3)
    (tmp_path / "v3.bin").write_bytes(bytes(blob))
    p = subprocess.run([str(_RUN_EXE[0]), str(tmp_path / "v3.bin"), "/dev/null"],
                       capture_output=True)
    assert p.returncode == 3, p.stdout.decode() + p.stderr.decode()
    assert "version 3" in p.stderr.decode(), p.stderr.decode()
    assert "Rebuild the table" in p.stderr.decode(), p.stderr.decode()


def test_the_raw_alphabet_is_the_one_this_checkpoint_uses(corpus_and_oracle):
    """THE FINDING THIS PHASE EXISTS FOR, stated so a later reader cannot
    mistake it for an implementation detail.

    This checkpoint is NOT a GPT-2 byte-level BPE. It is a byte-fallback BPE over
    raw characters, and the three facts below are what settle it -- all read off
    the checkpoint's own vocabulary, none of them a guess:

      * ' ' (U+0020) is not a vocabulary entry, so a space must be REWRITTEN
        before it can tokenize at all. In a GPT-2 byte-level vocabulary the space
        would be U+0120 and the question would not arise.
      * U+2581 is a single vocabulary entry, and 'hello', 'U+2581h' and
        'U+2581hello' are all entries. A GPT-2 byte-level vocabulary contains none
        of them: it would have written U+2581 as the three printable stand-ins for
        0xE2 0x96 0x81.
      * the merges are over RAW characters ('\u2581\u2581\u2581...' + '\u2581').

    The trap this creates is worth spelling out: this 256k vocabulary HAPPENS to
    contain all 256 GPT-2 byte characters, so the "the byte characters are all
    present" check passes for it, and a runtime that maps bytes anyway produces
    ids that are all plausible and none of them right. Which is exactly what the
    first version of this port did.
    """
    raw, _, oracle = corpus_and_oracle
    v = raw["model"]["vocab"]
    assert " " not in v, "a raw space is an entry: this is a GPT-2 byte-level vocab"
    assert v["\u2581"] == 235248
    assert v["hello"] == 17534 and v["\u2581h"] == 531 and v["\u2581hello"] == 25612
    assert raw["model"]["byte_fallback"] is True
    assert not any(d.get("type") == "ByteLevel"
                   for d in raw["decoder"]["decoders"]), \
        "the decoder has no ByteLevel stage, so the alphabet is not the byte map"
    # And the consequence, in ids: the marker is ONE token inside a word.
    assert oracle["hello world"] == [25612, 2134]


def test_the_generator_refuses_what_it_does_not_implement(tmp_path):
    """The refusals are the point of a generator: a checkpoint this build cannot
    tokenize exactly must stop HERE rather than produce a table that looks fine."""
    import json as _json

    base = _json.loads(_tokenizer_json().read_text())

    def gen(raw: dict) -> subprocess.CompletedProcess:
        src = tmp_path / "t.json"
        src.write_text(_json.dumps(raw))
        out = tmp_path / "t.bin"
        p = subprocess.run([str(_GEN_EXE[0]), str(src), str(out)],
                           capture_output=True, text=True)
        return p

    # A Replace with a REGEX pattern is a different matcher, and accepting it
    # would mean accepting every regex this build has never seen.
    raw = _json.loads(_json.dumps(base))
    raw["normalizer"] = {"type": "Replace",
                         "pattern": {"Regex": "\\s+"}, "content": "X"}
    p = gen(raw)
    assert p.returncode == 1 and "non-literal pattern" in p.stderr, p.stderr

    # An unknown pre_tokenizer is still refused -- ByteLevel and Metaspace only.
    raw = _json.loads(_json.dumps(base))
    raw["pre_tokenizer"] = {"type": "Sequence",
                            "pretokenizers": [{"type": "Split", "pattern": {"Regex": "x"}}]}
    p = gen(raw)
    assert p.returncode == 1 and "needs its own scanner" in p.stderr, p.stderr

    # A single_word added token constrains the match to word boundaries, and is
    # not implemented.
    raw = _json.loads(_json.dumps(base))
    raw["added_tokens"][10]["single_word"] = True
    p = gen(raw)
    assert p.returncode == 1 and "single_word" in p.stderr, p.stderr

    # byte_fallback whose alphabet is not closed is refused, because
    # byte_fallback is the promise that ANY byte tokenizes.
    # Renamed rather than deleted: the count has to stay put or the
    # contiguity check fires first and the test stops measuring what it means to.
    raw = _json.loads(_json.dumps(base))
    ren = {}
    dropped = 0
    for k, v in raw["model"]["vocab"].items():
        if k.startswith("<0x") and dropped < 100:
            ren["<z" + k[2:]] = v
            dropped += 1
        else:
            ren[k] = v
    assert dropped == 100
    raw["model"]["vocab"] = ren
    p = gen(raw)
    assert p.returncode == 1 and "byte_fallback" in p.stderr, p.stderr

    # And the real checkpoint still generates, so none of the above is a refusal
    # that has quietly become a blanket one.
    p = gen(base)
    assert p.returncode == 0, p.stderr


_GEN_EXE: list[Path] = []
_RUN_EXE: list[Path] = []


def _tokenize_blob(blob: bytes, s: str) -> list[int]:
    import tempfile

    d = Path(tempfile.mkdtemp())
    (d / "t.bin").write_bytes(blob)
    (d / "i.bin").write_bytes(s.encode() + b"\0")
    p = subprocess.run([str(_RUN_EXE[0]), str(d / "t.bin"), str(d / "i.bin")],
                       capture_output=True)
    assert p.returncode == 0, p.stderr.decode()
    row = p.stdout.split(b"\0")[0]
    return [int(x) for x in row.split()] if row else []


@pytest.fixture(scope="module", autouse=True)
def _exes(tmp_path_factory):
    d = tmp_path_factory.mktemp("bbpe_exe")
    _GEN_EXE.append(_compile(d, GEN_DRIVER, "gen2", []))
    _RUN_EXE.append(_compile(d, DRIVER, "run2", []))
    yield
