//===- tokenizer_bbpe.cpp ------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- byte-level BPE. See tokenizer_bbpe.hpp for the design.
// SPDX-License-Identifier: MIT
//
// THE PIPELINE, in the order it runs. Each stage was read out of a real
// tokenizer.json and then confirmed against HuggingFace's own output before
// being written here; the fuzz numbers are in tasks/0153.
//
//   1. ADDED TOKENS. The checkpoint's added_tokens are matched literally
//      against the raw text, leftmost-longest, splitting it into "this span
//      IS id N" pieces and ordinary pieces. The OLMo family uses this for
//      runs of 2-24 spaces, so skipping the stage would silently change the
//      tokenization of every indented text. `lstrip` extends a match over the
//      whitespace before it, which is then swallowed rather than tokenized --
//      BERT's [MASK] carries it, and "a [MASK] b" gives ['a', ' [MASK]',
//      'Gb'], one token fewer than the naive reading. (`rstrip` is refused by
//      the generator: nothing here sets it, so it could only ship unverified.)
//
//   2. NFC, if the checkpoint asks for it. Composition is the standard
//      algorithm over bbpe_unicode_tables.hpp's canonical data, with Hangul
//      handled algorithmically.
//
//   3. PRE-TOKENIZE. The GPT-2 regex
//         's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//      as a hand-written scanner over the four-way character class. The one
//      subtle alternative is `\s+(?!\S)`: inside a maximal whitespace run it
//      can only stop where the next character is also whitespace, so greedily
//      it takes the whole run at end-of-text and everything-but-the-last
//      space otherwise -- which is precisely how the space before a word ends
//      up attached to that word as "Gword".
//
//   4. BYTE MAP. Each word's UTF-8 bytes become printable codepoints via
//      GPT-2's bytes_to_unicode. Every byte that can occur in valid UTF-8 has
//      a vocabulary entry, so there is no fallback and no <unk> -- the
//      generator verifies that rather than trusting it.
//
//   5. BPE, per word, merges applied lowest-rank first. Same engine as
//      tokenizer_gemma.cpp: doubly-linked list plus a min-heap whose entries
//      are validated on pop, because an earlier merge may have consumed an
//      endpoint. Merges never cross a word boundary, which is the whole point
//      of stage 3.

#include "tokenizer_bbpe.hpp"

#include "bbpe_unicode_tables.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <queue>
#include <sstream>
#include <stdexcept>

namespace npue {
namespace {

uint16_t read_u16(const char *p) {
  uint16_t v;
  std::memcpy(&v, p, 2);
  return v;
}
uint32_t read_u32(const char *p) {
  uint32_t v;
  std::memcpy(&v, p, 4);
  return v;
}

// --- UTF-8, the same conventions as tokenizer_gemma.cpp's ---------------
//
// An invalid sequence becomes U+FFFD, exactly as Python's decoder would with
// errors="replace". That matters for more than tidiness: re-encoding the
// replacement character produces valid UTF-8, which is what guarantees the
// byte map below never sees 0xC0, 0xC1 or 0xF5-0xFF -- the five bytes a
// byte-level vocabulary is allowed to omit.
std::vector<uint32_t> utf8_decode(const std::string &s) {
  std::vector<uint32_t> out;
  out.reserve(s.size());
  size_t i = 0;
  const size_t n = s.size();
  while (i < n) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    uint32_t cp = 0xFFFD;
    int len = 1;
    if (c < 0x80) { cp = c; len = 1; }
    else if ((c & 0xE0) == 0xC0 && i + 1 < n) { cp = c & 0x1F; len = 2; }
    else if ((c & 0xF0) == 0xE0 && i + 2 < n) { cp = c & 0x0F; len = 3; }
    else if ((c & 0xF8) == 0xF0 && i + 3 < n) { cp = c & 0x07; len = 4; }
    else { out.push_back(0xFFFD); ++i; continue; }
    bool ok = true;
    for (int k = 1; k < len; ++k) {
      const unsigned char cc = static_cast<unsigned char>(s[i + k]);
      if ((cc & 0xC0) != 0x80) { ok = false; break; }
      cp = (cp << 6) | (cc & 0x3F);
    }
    if (!ok) { out.push_back(0xFFFD); ++i; continue; }
    out.push_back(cp);
    i += len;
  }
  return out;
}

void utf8_append(std::string &s, uint32_t cp) {
  if (cp < 0x80) {
    s.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    s.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    s.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

// --- Unicode lookups over the generated tables --------------------------

uint8_t char_class(uint32_t cp) {
  int lo = 0, hi = bbpe_uni::kClass_n - 1;
  while (lo <= hi) {
    const int mid = lo + (hi - lo) / 2;
    if (cp < bbpe_uni::kClass[mid].lo) hi = mid - 1;
    else if (cp > bbpe_uni::kClass[mid].hi) lo = mid + 1;
    else return bbpe_uni::kClass[mid].cls;
  }
  return bbpe_uni::kOther;      // absent means Other, by construction
}

uint8_t ccc_of(uint32_t cp) {
  int lo = 0, hi = bbpe_uni::kCcc_n - 1;
  while (lo <= hi) {
    const int mid = lo + (hi - lo) / 2;
    if (cp < bbpe_uni::kCcc[mid].cp) hi = mid - 1;
    else if (cp > bbpe_uni::kCcc[mid].cp) lo = mid + 1;
    else return bbpe_uni::kCcc[mid].ccc;
  }
  return 0;
}

const bbpe_uni::Decomp *decomp_of(uint32_t cp) {
  int lo = 0, hi = bbpe_uni::kDecomp_n - 1;
  while (lo <= hi) {
    const int mid = lo + (hi - lo) / 2;
    if (cp < bbpe_uni::kDecomp[mid].cp) hi = mid - 1;
    else if (cp > bbpe_uni::kDecomp[mid].cp) lo = mid + 1;
    else return &bbpe_uni::kDecomp[mid];
  }
  return nullptr;
}

uint32_t compose_pair(uint32_t a, uint32_t b) {
  int lo = 0, hi = bbpe_uni::kCompose_n - 1;
  while (lo <= hi) {
    const int mid = lo + (hi - lo) / 2;
    const auto &e = bbpe_uni::kCompose[mid];
    if (a < e.a || (a == e.a && b < e.b)) hi = mid - 1;
    else if (a > e.a || (a == e.a && b > e.b)) lo = mid + 1;
    else return e.cp;
  }
  return 0;
}

// Canonical decomposition, plus algorithmic Hangul.
//
// NO RECURSION: the table holds HuggingFace's own NFD output, which is
// already fully decomposed and canonically ordered, so one lookup gives the
// final sequence.
void decompose_cp(uint32_t cp, std::vector<uint32_t> &out) {
  using namespace bbpe_uni;
  if (cp >= kSBase && cp < kSBase + kSCount) {
    const uint32_t s = cp - kSBase;
    out.push_back(kLBase + s / kNCount);
    out.push_back(kVBase + (s % kNCount) / kTCount);
    const uint32_t t = s % kTCount;
    if (t) out.push_back(kTBase + t);
    return;
  }
  const Decomp *d = decomp_of(cp);
  if (d == nullptr) { out.push_back(cp); return; }
  for (uint32_t i = 0; i < d->len; ++i)
    out.push_back(kDecompData[d->off + i]);
}

// NFC = canonical decomposition, canonical ordering, canonical composition.
// The standard algorithm (Unicode 15.0 sections 3.11 and 3.12); the only
// thing worth pointing at is the `last_ccc` blocking rule in the composition
// pass, which is what stops a starter from reaching across an intervening
// mark of equal or higher combining class.
std::vector<uint32_t> nfc(const std::vector<uint32_t> &cps) {
  std::vector<uint32_t> d;
  d.reserve(cps.size() + cps.size() / 4);
  for (uint32_t cp : cps) decompose_cp(cp, d);

  // Canonical ordering: a stable bubble over runs of nonzero ccc.
  for (size_t i = 1; i < d.size(); ++i) {
    const uint8_t k = ccc_of(d[i]);
    if (k == 0) continue;
    size_t j = i;
    while (j > 0) {
      const uint8_t kp = ccc_of(d[j - 1]);
      if (kp == 0 || kp <= k) break;
      std::swap(d[j - 1], d[j]);
      --j;
    }
  }

  using namespace bbpe_uni;
  std::vector<uint32_t> out;
  out.reserve(d.size());
  size_t starter = static_cast<size_t>(-1);
  // `last_ccc` is the combining class of the last character emitted SINCE the
  // current starter, and it is -1 -- not 0 -- immediately after a starter.
  // The blocking rule is "C is blocked from the starter if something between
  // them has ccc 0 or ccc >= ccc(C)", so a character DIRECTLY after the
  // starter is never blocked, whatever its class. Initialising to 0 instead
  // makes `last_ccc < ccc(C)` false for every C with ccc 0 -- which is every
  // Hangul jamo, so "한글" decomposed to jamo and never came back, and
  // tokenized as 18 raw byte pieces against HuggingFace's 3 (tasks/0153).
  int last_ccc = -1;
  for (size_t i = 0; i < d.size(); ++i) {
    const uint32_t c = d[i];
    const uint8_t k = ccc_of(c);
    if (starter != static_cast<size_t>(-1) && last_ccc < static_cast<int>(k)) {
      const uint32_t s = out[starter];
      uint32_t composed = 0;
      // Hangul, algorithmically: L+V and LV+T.
      if (s >= kLBase && s < kLBase + kLCount && c >= kVBase &&
          c < kVBase + kVCount) {
        composed = kSBase + ((s - kLBase) * kVCount + (c - kVBase)) * kTCount;
      } else if (s >= kSBase && s < kSBase + kSCount &&
                 (s - kSBase) % kTCount == 0 && c > kTBase &&
                 c < kTBase + kTCount) {
        composed = s + (c - kTBase);
      } else {
        composed = compose_pair(s, c);
      }
      if (composed) {
        out[starter] = composed;
        continue;                      // c consumed; last_ccc unchanged
      }
    }
    if (k == 0) {
      starter = out.size();
      last_ccc = -1;                 // nothing between the starter and what
    } else {                         // comes next, so nothing can block it
      last_ccc = static_cast<int>(k);
    }
    out.push_back(c);
  }
  return out;
}

// --- GPT-2's byte <-> printable-codepoint map ---------------------------
//
// Built once, the same construction the generator uses. Byte -> codepoint is
// a 256-entry array; codepoint -> byte is only needed by the table generator,
// not here.
struct ByteMap {
  uint32_t to_cp[256];
  ByteMap() {
    bool used[256] = {false};
    int n = 0;
    auto take = [&](int lo, int hi) {
      for (int b = lo; b <= hi; ++b) { to_cp[b] = static_cast<uint32_t>(b); used[b] = true; }
    };
    take('!', '~');
    take(0xA1, 0xAC);
    take(0xAE, 0xFF);
    for (int b = 0; b < 256; ++b)
      if (!used[b]) to_cp[b] = static_cast<uint32_t>(256 + n++);
  }
};
const ByteMap &byte_map() {
  static const ByteMap m;
  return m;
}

uint64_t pair_key(int32_t a, int32_t b) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) |
         static_cast<uint32_t>(b);
}

const char *kContractions[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};

}  // namespace

// --- table loading ------------------------------------------------------

void BbpeTokenizer::build_index(const std::string &blob) {
  const char *p = blob.data();
  const char *end = p + blob.size();
  auto need = [&](size_t n) {
    if (static_cast<size_t>(end - p) < n)
      throw std::runtime_error("tokenizer_bbpe: truncated table");
  };

  need(8);
  if (std::memcmp(p, "BBPETOK1", 8) != 0)
    throw std::runtime_error("tokenizer_bbpe: bad magic");
  p += 8;

  need(4); const uint32_t version = read_u32(p); p += 4;
  if (version != 1 && version != 2)
    throw std::runtime_error("tokenizer_bbpe: unsupported table version " +
                             std::to_string(version) +
                             ". This build reads BBPETOK1 versions 1 and 2; a "
                             "higher version was written by a newer generator, "
                             "and reading its tail as vocabulary would produce "
                             "plausible ids that are wrong. Rebuild the table "
                             "rather than guessing the layout.");

  need(4); normalizer = read_u32(p); p += 4;
  if (normalizer > 2)
    throw std::runtime_error("tokenizer_bbpe: unknown normalizer id " +
                             std::to_string(normalizer));
  need(4); add_prefix_space = read_u32(p) != 0; p += 4;

  need(12);
  const uint32_t vocab_size = read_u32(p); p += 4;
  const uint32_t num_merges = read_u32(p); p += 4;
  const uint32_t num_added = read_u32(p); p += 4;

  need(20);
  cls_id = static_cast<int32_t>(read_u32(p)); p += 4;
  sep_id = static_cast<int32_t>(read_u32(p)); p += 4;
  pad_id = static_cast<int32_t>(read_u32(p)); p += 4;
  unk_id = static_cast<int32_t>(read_u32(p)); p += 4;
  mask_id = static_cast<int32_t>(read_u32(p)); p += 4;

  need(4); const uint32_t n_prefix = read_u32(p); p += 4;
  prefix_ids_.resize(n_prefix);
  for (uint32_t i = 0; i < n_prefix; ++i) {
    need(4); prefix_ids_[i] = static_cast<int32_t>(read_u32(p)); p += 4;
  }
  need(4); const uint32_t n_suffix = read_u32(p); p += 4;
  suffix_ids_.resize(n_suffix);
  for (uint32_t i = 0; i < n_suffix; ++i) {
    need(4); suffix_ids_[i] = static_cast<int32_t>(read_u32(p)); p += 4;
  }

  id_to_token_.resize(vocab_size);
  token_to_id_.reserve(vocab_size * 2);
  for (uint32_t i = 0; i < vocab_size; ++i) {
    need(2);
    const uint16_t len = read_u16(p); p += 2;
    need(len);
    id_to_token_[i].assign(p, len);
    p += len;
    token_to_id_.emplace(id_to_token_[i], static_cast<int32_t>(i));
  }

  merge_of_.reserve(static_cast<size_t>(num_merges) * 2);
  for (uint32_t rank = 0; rank < num_merges; ++rank) {
    need(12);
    const uint32_t a = read_u32(p); p += 4;
    const uint32_t b = read_u32(p); p += 4;
    const uint32_t merged = read_u32(p); p += 4;
    merge_of_.emplace(pair_key(static_cast<int32_t>(a), static_cast<int32_t>(b)),
                      MergeInfo{rank, static_cast<int32_t>(merged)});
  }

  added_.resize(num_added);
  for (uint32_t i = 0; i < num_added; ++i) {
    need(2);
    const uint16_t len = read_u16(p); p += 2;
    need(len + 8u);
    added_[i].content.assign(p, len); p += len;
    added_[i].id = static_cast<int32_t>(read_u32(p)); p += 4;
    const uint32_t flags = read_u32(p); p += 4;
    added_[i].lstrip = (flags & 1u) != 0;
    if (flags & ~1u)
      throw std::runtime_error(
          "tokenizer_bbpe: added-token flag bit set that this build does not "
          "implement (only lstrip = bit 0). The table was written by a newer "
          "generator; rebuild the runtime rather than ignoring the bit.");
    added_max_bytes_ = std::max(added_max_bytes_, added_[i].content.size());
  }
  // Longest first, so a linear scan at each position is leftmost-LONGEST
  // without a second pass. HuggingFace's matcher is an Aho-Corasick; with
  // ~100 patterns the difference is not measurable and the ordering is the
  // part that has to be right.
  std::sort(added_.begin(), added_.end(),
            [](const Added &a, const Added &b) {
              if (a.content.size() != b.content.size())
                return a.content.size() > b.content.size();
              return a.content < b.content;
            });

  // --- version 2 tail. v1 stops here, and that is the compatibility story:
  // the new fields are APPENDED, so a v1 table is a strict prefix and a v1
  // reader never walks off the end looking for them.
  if (version >= 2) {
    auto read_str = [&](std::string &dst) {
      need(2);
      const uint16_t len = read_u16(p); p += 2;
      need(len);
      dst.assign(p, len);
      p += len;
    };
    need(4); pre_tokenizer = read_u32(p); p += 4;
    if (pre_tokenizer > 1)
      throw std::runtime_error("tokenizer_bbpe: unknown pre_tokenizer id " +
                               std::to_string(pre_tokenizer));
    read_str(norm_replacement);
    need(4); prepend_scheme = read_u32(p); p += 4;
    if (prepend_scheme > 2)
      throw std::runtime_error("tokenizer_bbpe: unknown prepend_scheme " +
                               std::to_string(prepend_scheme));
    read_str(replacement);
    need(4); metaspace_split = read_u32(p) != 0; p += 4;
    need(4); byte_fallback = read_u32(p) != 0; p += 4;
    need(4); fuse_unk = read_u32(p) != 0; p += 4;
    need(4); byte_mode = read_u32(p); p += 4;
    if (byte_mode > 1)
      throw std::runtime_error("tokenizer_bbpe: unknown byte_mode " +
                               std::to_string(byte_mode) +
                               ". This build implements 0 (GPT-2 byte map) and 1 "
                               "(raw characters with byte_fallback); the field "
                               "says which ALPHABET the vocabulary is indexed "
                               "by, and reading it as the other one produces ids "
                               "that are all plausible and none of them right.");

    // Two of the new fields are only meaningful together with something that
    // uses them, and a missing value would make the corresponding stage a
    // no-op rather than an error: Metaspace with no replacement splits on
    // nothing, and norm == 2 with an empty replacement rewrites no spaces --
    // which for this checkpoint means no space has a symbol and nothing
    // tokenizes. Both are refused here, at load, by name.
    if (pre_tokenizer == 1 && replacement.empty())
      throw std::runtime_error(
          "tokenizer_bbpe: the table declares the Metaspace pre-tokenizer and "
          "carries no replacement character. Refusing rather than splitting on "
          "nothing.");
    if (normalizer == 2 && norm_replacement.empty())
      throw std::runtime_error(
          "tokenizer_bbpe: the table declares the Replace normalizer and "
          "carries no replacement string, so a space would not be rewritten "
          "and nothing would tokenise the way the checkpoint does.");
  } else {
    // v1 tables predate both flags. Recording them as false keeps every
    // pre-existing table's behaviour bit-identical, which is the whole reason
    // the version branches here rather than defaulting in one place.
    pre_tokenizer = 0;
    prepend_scheme = 0;
    metaspace_split = false;
    byte_fallback = false;
    fuse_unk = false;
    byte_mode = 0;      // every v1 table this generator wrote was ByteLevel
  }

  if (id_to_token_.empty())
    throw std::runtime_error("tokenizer_bbpe: empty vocabulary");
}

BbpeTokenizer BbpeTokenizer::from_table_file(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("tokenizer_bbpe: cannot open " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  BbpeTokenizer t;
  t.build_index(ss.str());
  return t;
}

BbpeTokenizer BbpeTokenizer::from_table_bytes(const char *data, size_t bytes) {
  BbpeTokenizer t;
  t.build_index(std::string(data, bytes));
  return t;
}

int32_t BbpeTokenizer::id_of(const std::string &token) const {
  auto it = token_to_id_.find(token);
  return it == token_to_id_.end() ? unk_id : it->second;
}

const std::string &BbpeTokenizer::token_of(int32_t id) const {
  static const std::string kEmpty;
  if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) return kEmpty;
  return id_to_token_[static_cast<size_t>(id)];
}

// --- BPE over one pre-tokenized word ------------------------------------
//
// Stage 4 (build symbols) and stage 5 (merge) are SEPARATE, and that split is
// this phase's whole finding. The merge engine is alphabet-agnostic: it takes a
// sequence of ids and applies merges lowest-rank first. What decides the ids is
// the vocabulary's alphabet, and there are two of them --

// byte_mode 0: GPT-2 bytes_to_unicode(). The WORD has already been through the
// map by the caller; every codepoint in it is one of the 256 printable
// stand-ins and all 256 are vocabulary entries, which the generator checks.
void BbpeTokenizer::bpe_word(const std::string &mapped,
                             std::vector<int32_t> &out) const {
  const std::vector<uint32_t> cps = utf8_decode(mapped);
  std::vector<int32_t> symbols;
  symbols.reserve(cps.size());
  for (uint32_t cp : cps) {
    std::string ch;
    utf8_append(ch, cp);
    auto it = token_to_id_.find(ch);
    if (it != token_to_id_.end()) {
      symbols.push_back(it->second);
      continue;
    }
    // Unreachable: the generator checks that every byte which can occur in
    // UTF-8 is an entry in this alphabet. A table that reaches here was written
    // by something else.
    throw std::runtime_error(
        "tokenizer_bbpe: byte_mode 0 and a character is missing from the "
        "vocabulary. The generator checks that all 256 GPT-2 byte characters "
        "are present, so this table was not written by it.");
  }
  bpe_symbols(symbols, out);
}

// byte_mode 1: the vocabulary is indexed by CHARACTERS, and a character nothing
// can represent falls back to its UTF-8 BYTES -- one `<0xNN>` token per byte --
// which is what `byte_fallback: true` promises and why it exists at all. This is
// the SentencePiece mechanism, and the old generator's refusal message used to
// point at tokenizer_gemma.cpp for it while being wrong about which tokenizer
// this was.
void BbpeTokenizer::bpe_word_raw(const std::string &word,
                                std::vector<int32_t> &out) const {
  const std::vector<uint32_t> cps = utf8_decode(word);
  std::vector<int32_t> symbols;
  symbols.reserve(cps.size());
  bool prev_was_unk = false;
  for (uint32_t cp : cps) {
    std::string ch;
    utf8_append(ch, cp);
    auto it = token_to_id_.find(ch);
    if (it != token_to_id_.end()) {
      symbols.push_back(it->second);
      prev_was_unk = false;
      continue;
    }
    if (byte_fallback) {
      // One `<0xNN>` per UTF-8 BYTE. The generator allows ONE piece to be
      // absent, and that is sound rather than convenient: a missing piece can
      // only be a byte whose single-byte CHARACTER already has an entry (this
      // checkpoint has no `<0x09>` and tab is id 226), and a multi-byte
      // character only ever contains continuation bytes >= 0x80. So this arm,
      // which is reached only for a character with no entry of its own, cannot
      // be the one that needs the piece that is missing.
      bool any = false;
      for (unsigned char b : ch) {
        char key[8];
        std::snprintf(key, sizeof(key), "<0x%02X>", b);
        auto bit = token_to_id_.find(key);
        if (bit == token_to_id_.end())
          throw std::runtime_error(
              std::string("tokenizer_bbpe: the vocabulary declares byte_fallback "
                          "but has no '") + key + "' piece, and no <unk> either. "
                          "byte_fallback is the promise that ANY byte tokenizes; "
                          "this table does not keep it.");
        symbols.push_back(bit->second);
        any = true;
      }
      prev_was_unk = false;
      (void)any;
      continue;
    }
    if (unk_id < 0)
      throw std::runtime_error(
          "tokenizer_bbpe: the vocabulary has no entry for a character that "
          "occurs in this text, records no byte_fallback to split it into "
          "<0xNN> pieces, and names no <unk> token to substitute for it. There "
          "is nothing correct to emit, and dropping the character would shorten "
          "the word by one token without saying so.");
    // `fuse_unk` decides whether a RUN of them is one token or one each, and
    // it is RECORDED rather than assumed: this checkpoint sets it and a
    // neighbouring one might not.
    if (fuse_unk && prev_was_unk) continue;
    symbols.push_back(unk_id);
    prev_was_unk = true;
  }
  bpe_symbols(symbols, out);
}

// Stage 5. Doubly-linked list plus a lazily-invalidated min-heap on merge rank;
// entries are re-checked on pop because an earlier merge may have consumed one
// of their endpoints. This is tokenizer_gemma.cpp's engine deliberately kept as
// the same code shape, so the two can be read against each other.
void BbpeTokenizer::bpe_symbols(const std::vector<int32_t> &symbols,
                                std::vector<int32_t> &out) const {
  if (symbols.empty()) return;

  struct Node { int32_t id; int prev, next; bool alive; };
  std::vector<Node> nodes(symbols.size());
  for (size_t i = 0; i < symbols.size(); ++i) {
    nodes[i].id = symbols[i];
    nodes[i].prev = static_cast<int>(i) - 1;
    nodes[i].next = (i + 1 < symbols.size()) ? static_cast<int>(i + 1) : -1;
    nodes[i].alive = true;
  }

  struct Candidate {
    uint32_t rank;
    int left, right;
    int32_t left_id, right_id;
    int32_t merged_id;
  };
  struct Cmp {
    bool operator()(const Candidate &a, const Candidate &b) const {
      if (a.rank != b.rank) return a.rank > b.rank;
      return a.left > b.left;
    }
  };
  std::priority_queue<Candidate, std::vector<Candidate>, Cmp> heap;

  auto try_queue = [&](int left, int right) {
    if (left < 0 || right < 0) return;
    auto it = merge_of_.find(pair_key(nodes[left].id, nodes[right].id));
    if (it == merge_of_.end()) return;
    heap.push(Candidate{it->second.rank, left, right, nodes[left].id,
                        nodes[right].id, it->second.merged_id});
  };

  for (size_t i = 0; i + 1 < nodes.size(); ++i)
    try_queue(static_cast<int>(i), static_cast<int>(i) + 1);

  while (!heap.empty()) {
    Candidate c = heap.top();
    heap.pop();
    if (!nodes[c.left].alive || !nodes[c.right].alive) continue;
    if (nodes[c.left].id != c.left_id || nodes[c.right].id != c.right_id) continue;
    if (nodes[c.left].next != c.right) continue;

    nodes[c.left].id = c.merged_id;
    nodes[c.right].alive = false;
    const int after = nodes[c.right].next;
    nodes[c.left].next = after;
    if (after >= 0) nodes[after].prev = c.left;

    try_queue(nodes[c.left].prev, c.left);
    try_queue(c.left, nodes[c.left].next);
  }

  for (int i = 0; i >= 0 && static_cast<size_t>(i) < nodes.size();
       i = nodes[i].next)
    if (nodes[i].alive) out.push_back(nodes[i].id);
}

// --- the pre-tokenizers, one function each
//
// Two DIFFERENT segmenters, not one with a mode. The GPT-2 regex below is a
// character-class scanner over four classes plus a contractions list; Metaspace
// runs no regex at all -- it cuts in front of every occurrence of one character,
// keeps that character at the head of the piece that follows, and drops the
// empty leading piece. Putting a Metaspace branch inside the regex scanner would
// put a per-character test on the hot path for the checkpoints that do not use
// it, in exchange for sharing nothing.

// GPT-2:
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
void BbpeTokenizer::emit_gpt2_words(const std::vector<uint32_t> &cps,
                                    std::vector<int32_t> &out) const {
  const size_t n = cps.size();
  size_t i = 0;
  while (i < n) {
    size_t j = i;
    bool matched_contraction = false;
    for (const char *c : kContractions) {
      const size_t len = std::strlen(c);
      if (i + len > n) continue;
      bool eq = true;
      for (size_t k = 0; k < len; ++k)
        if (cps[i + k] != static_cast<uint32_t>(static_cast<unsigned char>(c[k]))) {
          eq = false;
          break;
        }
      if (eq) { j = i + len; matched_contraction = true; break; }
    }
    if (!matched_contraction) {
      j = i;
      // The regex's ` ?` is a literal U+0020, not `\s?`.
      if (cps[j] == 0x20 && j + 1 < n && char_class(cps[j + 1]) != bbpe_uni::kSpace)
        ++j;
      const uint8_t k = char_class(cps[j]);
      if (k != bbpe_uni::kSpace) {
        while (j < n && char_class(cps[j]) == k) ++j;
      } else {
        // `\s+(?!\S)` then `\s+`: hold the last space back for the word
        // that follows it, unless the run ends the text.
        while (j < n && char_class(cps[j]) == bbpe_uni::kSpace) ++j;
        if (j < n && j - 1 > i) --j;
      }
    }
    byte_map_and_bpe(cps, i, j, out);
    i = j;
  }
}

// Metaspace, `split: true`.
//
// MEASURED against tokenizers 0.22, not derived: the boundary is "cut in front
// of every replacement character, keep it with the piece that follows, drop the
// empty leading piece". So "▁▁x" is ["▁", "▁x"] and "▁▁▁x" is ["▁", "▁", "▁x"] --
// three pieces, not one and not two. The obvious alternative ("the delimiter is
// removed and the pieces are what is left") gives ["", "", "x"], which is two
// empty pieces and a word, and the difference is one token per leading marker.
//
// With `split: false` there is one piece, the whole segment.
void BbpeTokenizer::emit_metaspace_words(const std::vector<uint32_t> &cps,
                                         const std::vector<uint32_t> &reps,
                                         std::vector<int32_t> &out) const {
  if (!metaspace_split || reps.size() != 1) {
    byte_map_and_bpe(cps, 0, cps.size(), out);
    return;
  }
  const uint32_t R = reps[0];
  size_t start = 0;
  for (size_t i = 1; i < cps.size(); ++i) {
    if (cps[i] != R) continue;
    byte_map_and_bpe(cps, start, i, out);
    start = i;
  }
  byte_map_and_bpe(cps, start, cps.size(), out);
}

// Stages 4 and 5: build this word's symbols in the vocabulary's alphabet, then
// merge them. The alphabet is the only difference between the two paths, and it
// is one branch rather than two copies of the scanner.
void BbpeTokenizer::byte_map_and_bpe(const std::vector<uint32_t> &cps,
                                     size_t lo, size_t hi,
                                     std::vector<int32_t> &out) const {
  if (hi <= lo) return;
  std::string word;
  word.reserve((hi - lo) * 2);
  for (size_t k = lo; k < hi; ++k) utf8_append(word, cps[k]);
  if (byte_mode == 0) {
    const ByteMap &bm = byte_map();
    std::string mapped;
    mapped.reserve(word.size() * 2);
    for (char ch : word)
      utf8_append(mapped, bm.to_cp[static_cast<unsigned char>(ch)]);
    bpe_word(mapped, out);
  } else {
    bpe_word_raw(word, out);
  }
}

void BbpeTokenizer::emit_words(const std::vector<uint32_t> &cps,
                               std::vector<int32_t> &out) const {
  if (pre_tokenizer == 1) {
    // Decoded ONCE per segment rather than per character in the loop above.
    const std::vector<uint32_t> reps = utf8_decode(replacement);
    emit_metaspace_words(cps, reps, out);
  } else {
    emit_gpt2_words(cps, out);
  }
}

// --- the whole pipeline --------------------------------------------------

std::vector<int32_t> BbpeTokenizer::tokenize(const std::string &text) const {
  std::vector<int32_t> out;
  // Metaspace's prepend_scheme == "first" prepends to the FIRST segment only,
  // and segments here are the pieces between added-token matches -- so "first"
  // needs to know whether it has already had its turn.
  bool first_segment = true;

  // Stage 1: split on added tokens. Everything between matches is an
  // ordinary piece and goes through stages 2-5; a match contributes its id
  // and nothing else.
  auto run_ordinary = [&](const std::string &piece) {
    if (piece.empty()) return;   // not a segment; see first_segment below

    // Stage 2: normalize.
    std::vector<uint32_t> cps = utf8_decode(piece);
    if (normalizer == 1) {
      cps = nfc(cps);
    } else if (normalizer == 2) {
      // Replace(literal -> literal). The generator pins the pattern to a plain
      // STRING, so this is a substring rewrite and not a regex match -- which is
      // the only reason this is not a regex engine.
      //
      // And rewriting the spaces is not cosmetic here, it is the whole reason
      // this checkpoint has a normalizer at all: ' ' (U+0020) is NOT a
      // vocabulary entry in this model, so a space has no symbol and nothing
      // tokenizes until it has been rewritten. In a GPT-2 byte-level vocabulary
      // the space would be U+0120 and this stage would not exist.
      const std::vector<uint32_t> to = utf8_decode(norm_replacement);
      if (!to.empty()) {
        std::vector<uint32_t> rewritten;
        rewritten.reserve(cps.size() + 4);
        for (uint32_t cp : cps) {
          if (cp == 0x20)
            rewritten.insert(rewritten.end(), to.begin(), to.end());
          else
            rewritten.push_back(cp);
        }
        cps.swap(rewritten);
      }
    }
    if (pre_tokenizer == 1) {
      // Stage 3 (Metaspace). The prepend is here and the boundary rule is in
      // emit_metaspace_words; this is the whole of the prepend.
      const std::vector<uint32_t> reps = utf8_decode(replacement);
      if (reps.size() == 1) {
        const uint32_t R = reps[0];
        const bool want = prepend_scheme == 2 ||
                          (prepend_scheme == 1 && first_segment);
        if (want && !cps.empty() && cps[0] != R) cps.insert(cps.begin(), R);
      }
    } else if (add_prefix_space && !cps.empty() && cps[0] != 0x20) {
      cps.insert(cps.begin(), 0x20);
    }
    // An EMPTY piece is not a segment for the purpose of prepend_scheme
    // "first": HuggingFace skips empty splits entirely, and counting one here
    // would spend the single prepend on nothing.
    first_segment = false;

    // Stage 3: pre-tokenize. The two scanners are separate functions rather
    // than one with a branch per character: Metaspace runs no regex and shares
    // no rule with the GPT-2 pattern, and merging them would put a per-character
    // test on the hot path to serve a checkpoint shape that does not use it.
    emit_words(cps, out);
  };

  // The added-token spans, left to right, leftmost-longest -- then LSTRIP,
  // which is why this is two phases and not one.
  //
  // An lstrip match extends its start left over whitespace, and that
  // extension OVERRIDES a match already accepted inside it. Measured:
  // "a  [MASK]  [MASK]  b" gives ['a', '  [MASK]', '  [MASK]', '  ', 'b'] --
  // the two-space runs before each [MASK] are swallowed while the one before
  // "b" survives as its own added token, and "a  |||IP_ADDRESS|||  b" (same
  // shape, no lstrip) keeps both. A single-pass matcher accepts the
  // whitespace token first and can never take it back, which is exactly the
  // extra token the first version emitted.
  struct Span { size_t start, stop; int32_t id; };
  std::vector<Span> spans;
  size_t pos = 0;
  while (pos < text.size()) {
    const Added *hit = nullptr;
    for (const Added &a : added_) {
      if (a.content.size() > text.size() - pos) continue;
      if (std::memcmp(text.data() + pos, a.content.data(), a.content.size()) == 0) {
        hit = &a;
        break;                       // added_ is sorted longest-first
      }
    }
    if (hit == nullptr) { ++pos; continue; }
    size_t start = pos;
    if (hit->lstrip) {
      // Walk back over whole codepoints while they are whitespace.
      while (start > 0) {
        size_t k = start;
        while (k > 0 && (static_cast<unsigned char>(text[k - 1]) & 0xC0) == 0x80) --k;
        if (k == 0) break;
        --k;
        const std::vector<uint32_t> prev = utf8_decode(text.substr(k, start - k));
        if (prev.size() != 1 || char_class(prev[0]) != bbpe_uni::kSpace) break;
        start = k;
      }
      while (!spans.empty() && spans.back().stop > start) spans.pop_back();
    }
    spans.push_back(Span{start, pos + hit->content.size(), hit->id});
    pos += hit->content.size();
  }

  size_t cursor = 0;
  for (const Span &sp : spans) {
    if (sp.start > cursor) run_ordinary(text.substr(cursor, sp.start - cursor));
    out.push_back(sp.id);
    cursor = sp.stop;
  }
  if (cursor < text.size()) run_ordinary(text.substr(cursor));
  return out;
}

BbpeEncoded BbpeTokenizer::encode(const std::string &text, int max_len) const {
  BbpeEncoded e;
  const std::vector<int32_t> body = tokenize(text);
  const int wrap = n_special();
  const int room = max_len - wrap;
  const int take = std::min<int>(static_cast<int>(body.size()),
                                 std::max(0, room));

  e.input_ids.reserve(max_len);
  for (int32_t id : prefix_ids_) e.input_ids.push_back(id);
  for (int i = 0; i < take; ++i) e.input_ids.push_back(body[i]);
  for (int32_t id : suffix_ids_) e.input_ids.push_back(id);
  e.n_tokens = static_cast<int32_t>(e.input_ids.size());
  e.n_tokens_full = static_cast<int32_t>(body.size()) + wrap;
  e.truncated = take < static_cast<int>(body.size());

  e.attention_mask.assign(e.input_ids.size(), 1);
  if (static_cast<int>(e.input_ids.size()) < max_len && pad_id < 0)
    throw std::runtime_error(
        "tokenizer_bbpe: this checkpoint's table records no padding token, "
        "so a sequence shorter than max_len cannot be padded. Refusing "
        "rather than inventing an id -- padded positions are masked out, so "
        "a wrong one stays invisible until something reads them.");
  while (static_cast<int>(e.input_ids.size()) < max_len) {
    e.input_ids.push_back(pad_id);
    e.attention_mask.push_back(0);
  }
  return e;
}

std::vector<BbpeEncoded> BbpeTokenizer::encode_batch(
    const std::vector<std::string> &texts, int max_len) const {
  std::vector<BbpeEncoded> out;
  out.reserve(texts.size());
  for (const auto &t : texts) out.push_back(encode(t, max_len));
  return out;
}

}  // namespace npue
