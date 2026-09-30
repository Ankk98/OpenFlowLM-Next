//===- bbpe_tokenizer_gen.cpp --------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- C++ port of tools/gen_bbpe_tokenizer_table.py.
// See bbpe_tokenizer_gen.hpp.
// SPDX-License-Identifier: MIT
//
// The Python script is the reference; this file follows it statement for
// statement, including the order of its refusals, so a diff between the two
// stays readable. Where a message differs it is only in wording that names
// C++ rather than Python.

#include "bbpe_tokenizer_gen.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "json_min.hpp"

namespace npue {
namespace {

[[noreturn]] void fail(const std::string &msg) {
  throw std::runtime_error("gen_bbpe: " + msg);
}

void put_u16(std::vector<uint8_t> &o, uint16_t v) {
  o.push_back(static_cast<uint8_t>(v & 0xFF));
  o.push_back(static_cast<uint8_t>(v >> 8));
}
void put_u32(std::vector<uint8_t> &o, uint32_t v) {
  o.push_back(static_cast<uint8_t>(v & 0xFF));
  o.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
  o.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
  o.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
}
void put_i32(std::vector<uint8_t> &o, int32_t v) {
  put_u32(o, static_cast<uint32_t>(v));
}
void put_str(std::vector<uint8_t> &o, const std::string &s) {
  if (s.size() > 0xFFFF) fail("a string exceeds 65535 bytes");
  put_u16(o, static_cast<uint16_t>(s.size()));
  o.insert(o.end(), s.begin(), s.end());
}

// GPT-2's byte -> printable-codepoint map, as UTF-8 strings. The same
// construction as the Python `bytes_to_unicode()`.
std::vector<std::string> byte_chars() {
  bool used[256] = {false};
  uint32_t cp[256];
  auto take = [&](int lo, int hi) {
    for (int b = lo; b <= hi; ++b) { cp[b] = static_cast<uint32_t>(b); used[b] = true; }
  };
  take('!', '~');
  take(0xA1, 0xAC);
  take(0xAE, 0xFF);
  int n = 0;
  for (int b = 0; b < 256; ++b)
    if (!used[b]) cp[b] = static_cast<uint32_t>(256 + n++);

  std::vector<std::string> out(256);
  for (int b = 0; b < 256; ++b) {
    std::string s;
    const uint32_t c = cp[b];
    if (c < 0x80) {
      s.push_back(static_cast<char>(c));
    } else if (c < 0x800) {
      s.push_back(static_cast<char>(0xC0 | (c >> 6)));
      s.push_back(static_cast<char>(0x80 | (c & 0x3F)));
    } else {
      s.push_back(static_cast<char>(0xE0 | (c >> 12)));
      s.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
      s.push_back(static_cast<char>(0x80 | (c & 0x3F)));
    }
    out[b] = s;
  }
  return out;
}

}  // namespace

std::vector<uint8_t> generate_bbpe_tokenizer_table(
    const std::string &tokenizer_json_path) {
  std::ifstream f(tokenizer_json_path, std::ios::binary);
  if (!f) fail("cannot open " + tokenizer_json_path);
  std::stringstream ss;
  ss << f.rdbuf();
  const json::Value j = json::parse(ss.str());

  // --- normalizer -------------------------------------------------------
  //
  // Three kinds, and the third one (norm == 2, "Replace a literal string with
  // another literal string") is here for convaiinnovations/laya's multilingual
  // tokenizer, whose normalizer is
  //     {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}
  // The pattern TYPE is pinned to the literal form. Replace also takes a Regex
  // and a char/char pair, and those are a DIFFERENT matcher: this one is a
  // single-substring rewrite, and building a regex engine to serve it would be
  // a way to be quietly wrong about a dozen regexes this generator has never
  // seen. Anything else still throws.
  //
  // WHY a vocabulary needs it at all. `' '` (U+0020) is NOT a vocabulary entry
  // in this model -- U+2581 is -- so a space has no symbol and nothing
  // tokenises until it has been rewritten. Without this stage the failure looks
  // like a bug in the BPE rather than a missing normalizer.
  uint32_t norm = 0;
  std::string norm_replacement;
  if (const json::Value *n = j.find("normalizer")) {
    if (!n->is_null()) {
      const std::string ntype =
          (!n->is_object() || !n->contains("type") || !n->at("type").is_string())
              ? std::string()
              : n->at("type").as_string();
      if (ntype == "NFC") {
        norm = 1;
      } else if (ntype == "Replace") {
        const json::Value *pat = n->find("pattern");
        if (pat == nullptr || !pat->is_object() || !pat->contains("String") ||
            !pat->at("String").is_string())
          fail("normalizer is a Replace with a non-literal pattern. This "
               "generator implements `NFC`, `null`, and Replace(\"String\" -> "
               "String\"); the Regex and char/char forms are a DIFFERENT "
               "matcher, and refusing is cheaper than being wrong about regex "
               "syntax this build has never seen.");
        norm = 2;
        norm_replacement = n->at("content").as_string();
        if (norm_replacement.empty())
          fail("normalizer is Replace(\"String\" -> \"\") -- an empty "
               "replacement would delete the matched text rather than rewrite "
               "it, which is a different normalizer than the one this "
               "generator implements");
      } else {
        fail("unsupported normalizer -- this generator implements `null`, "
             "`NFC`, and Replace(\"String\" -> \"String\"). Add it to "
             "tokenizer_bbpe.cpp AND here, or the runtime will normalize "
             "differently from HuggingFace and every id will look reasonable "
             "and be wrong.");
      }
    }
  }

  // --- pre_tokenizer ----------------------------------------------------
  //
  // Two kinds. ByteLevel (the GPT-2 regex scanner) and Metaspace, which is a
  // DIFFERENT segmenter and not a variant of it: it splits on ONE character
  // (the replacement, U+2581 for sentencepiece-style vocabularies) keeping that
  // character at the head of the piece that follows, and optionally prepends
  // one per input. It never runs a regex at all, which is why it is cheap to
  // add here and why it cannot be reached by "just also allow Metaspace" in the
  // ByteLevel branch.
  uint32_t pre_tok = 0;                 // 0 = ByteLevel, 1 = Metaspace
  uint32_t add_prefix_space = 0;
  uint32_t prepend_scheme = 0;          // 0 never, 1 first, 2 always
  std::string replacement;              // Metaspace's replacement character
  uint32_t metaspace_split = 1;
  {
    const json::Value *pre = j.find("pre_tokenizer");
    if (pre == nullptr || !pre->is_object() || !pre->contains("type") ||
        !pre->at("type").is_string())
      fail("pre_tokenizer has no string \"type\"");
    const std::string ptype = pre->at("type").as_string();
    if (ptype == "ByteLevel") {
      if (const json::Value *ur = pre->find("use_regex"))
        if (ur->as_bool())
          fail("pre_tokenizer ByteLevel has use_regex=true -- the GPT-2 "
               "pattern is what this generator implements and turning it off "
               "makes the whole text one word, which is a different "
               "segmentation");
      if (const json::Value *ap = pre->find("add_prefix_space"))
        add_prefix_space = ap->as_bool() ? 1u : 0u;
    } else if (ptype == "Metaspace") {
      pre_tok = 1;
      if (const json::Value *ap = pre->find("add_prefix_space"))
        add_prefix_space = ap->as_bool() ? 1u : 0u;
      const json::Value *rep = pre->find("replacement");
      if (rep == nullptr || !rep->is_string() || rep->as_string().empty())
        fail("Metaspace pre_tokenizer has no replacement string");
      replacement = rep->as_string();
      if (const json::Value *sp = pre->find("split"))
        metaspace_split = sp->as_bool() ? 1u : 0u;
      const json::Value *sc = pre->find("prepend_scheme");
      if (sc != nullptr && sc->is_string()) {
        const std::string s = sc->as_string();
        if (s == "always") prepend_scheme = 2;
        else if (s == "never") prepend_scheme = 0;
        else if (s == "first") prepend_scheme = 1;
        else fail("Metaspace prepend_scheme is '" + s +
                  "' -- this generator implements never / first / always");
      }
      // A Metaspace replacement that is more than one CHARACTER cannot be
      // split on as an atom by the runtime's scanner, which is why the runtime
      // compares codepoints one at a time. Refuse rather than silently split
      // on the first.
      size_t ncp = 0;
      for (unsigned char c : replacement)
        if ((c & 0xC0) != 0x80) ++ncp;
      if (ncp != 1)
        fail("Metaspace replacement is not a single codepoint; this generator "
             "implements the one-character form only");
    } else {
      fail("pre_tokenizer is '" + ptype +
           "' -- this generator implements ByteLevel and Metaspace. A "
           "`Sequence` or an explicit `Split` pattern (Llama-3, Qwen-2.5 and "
           "tekken use one) is a DIFFERENT regex and needs its own scanner; "
           "refusing rather than pretending the GPT-2 pattern is universal.");
    }
  }

  // --- model ------------------------------------------------------------
  const json::Value &m = j.at("model");
  // `model.type` is ABSENT in the pre-0.10 tokenizer.json format
  // (roberta-base still ships one). The Python generator reports that as
  // "model.type is None, expected BPE"; this said "object has no key 'type'",
  // which is the same refusal wearing a parser's clothes. The port is meant
  // to be faithful down to its messages.
  const json::Value *mtype = m.find("type");
  if (mtype == nullptr || !mtype->is_string())
    fail("model.type is None, expected BPE");
  if (mtype->as_string() != "BPE")
    fail("model.type is '" + mtype->as_string() + "', expected BPE");
  for (const char *key : {"continuing_subword_prefix", "end_of_word_suffix"}) {
    const json::Value *v = m.find(key);
    if (v != nullptr && v->is_string() && !v->as_string().empty())
      fail(std::string("model.") + key +
           " is set -- affix-marked BPE is a different segmentation and is "
           "not implemented");
  }
  auto flag = [&](const char *key) {
    const json::Value *v = m.find(key);
    return v != nullptr && v->is_bool() && v->as_bool();
  };
  // --- THE ALPHABET, from the DECODER ----------------------------------
  //
  // The single most consequential fact about a BPE tokenizer, and the one that
  // decides whether the runtime maps bytes to printable codepoints at all.
  //
  // A GPT-2 / RoBERTa byte-level BPE's vocabulary is indexed by the 256
  // bytes_to_unicode() codepoints, and its DECODER carries a ByteLevel stage to
  // invert that. A SentencePiece-shaped vocabulary is indexed by the CHARACTERS,
  // with `<0xNN>` pieces existing so a character nothing can represent still
  // tokenizes. Same `model.type: BPE`, same merge engine, DIFFERENT alphabet --
  // and applying the GPT-2 map to the second one produces ids that are all
  // plausible and none of them right.
  //
  // The decoder says which, because it is the stage that inverts the mapping.
  // The pre-tokenizer is only a hint: Metaspace and ByteLevel can coexist in a
  // Sequence, and a SentencePiece alphabet can sit behind a ByteLevel
  // pre-tokenizer.
  //
  // convaiinnovations/laya's multilingual tokenizer is the second kind, and it
  // is easy to get wrong because its 256k vocabulary HAPPENS to contain all 256
  // GPT-2 byte characters -- so the byte-character check below passes for it,
  // and a runtime that maps bytes anyway looks right until someone reads the
  // ids. Three facts settle it, all read off the checkpoint's own vocabulary:
  //   * `' '` (U+0020) is NOT an entry. In a bytes_to_unicode alphabet the space
  //     IS U+0120 and this would be a different question; here it is absent, so
  //     a space has no symbol and must be REWRITTEN -- which is what the Replace
  //     normalizer above does.
  //   * `\u2581`, `hello`, `\u2581h` and `\u2581hello` are all entries. A
  //     byte-level vocabulary contains none of them: it would have written
  //     U+2581 as the three printable stand-ins for 0xE2 0x96 0x81.
  //   * the merges are over RAW characters ('\u2581\u2581\u2581...' + '\u2581').
  uint32_t byte_mode = 1;              // 1 = raw characters, 0 = GPT-2 byte map
  {
    bool decoder_has_bytelevel = false;
    const json::Value *dec = j.find("decoder");
    if (dec != nullptr && dec->is_object() && dec->contains("type") &&
        dec->at("type").is_string()) {
      const std::string dt = dec->at("type").as_string();
      if (dt == "ByteLevel") {
        decoder_has_bytelevel = true;
      } else if (dt == "Sequence") {
        const json::Value *ds = dec->find("decoders");
        if (ds != nullptr && ds->is_array())
          for (const json::Value &d : ds->as_array())
            if (d.is_object() && d.contains("type") &&
                d.at("type").is_string() && d.at("type").as_string() == "ByteLevel")
              decoder_has_bytelevel = true;
      } else if (dt != "Fuse" && dt != "Replace" && dt != "ByteFallback" &&
                 dt != "Strip" && dt != "ReplaceSequence" &&
                 dt != "ByteFallbackSequence") {
        fail("decoder is '" + dt + "' -- this generator implements a ByteLevel "
             "decoder (byte_mode 0, the GPT-2 alphabet) and the raw-character "
             "kinds (byte_mode 1). The decoder is what says which alphabet the "
             "vocabulary is indexed by, and guessing it wrong produces ids that "
             "are all plausible and none of them right.");
      }
    }
    byte_mode = decoder_has_bytelevel ? 0u : 1u;
  }

  // byte_fallback and unk_token are RECORDED, not refused, and the old
  // refusals were wrong for a reason worth keeping: they asserted "a byte-level
  // model needs no fallback because every byte is already a vocabulary entry",
  // which is a claim about a CLOSED alphabet and says nothing about this one.
  //
  // Here byte_fallback is the LIVE mechanism, not a floor: byte_mode 1 means a
  // character with no entry is split into its UTF-8 bytes and each byte becomes
  // a `<0xNN>` token. convaiinnovations/laya's multilingual tokenizer sets it,
  // and the vocabulary carries 255 of the 256 pieces -- the missing one is
  // `<0x09>`, because tab has an entry of its own and so never needs it.
  //
  // unk_token: this vocabulary HAS one (`<unk>` = 3) with fuse_unk set, and
  // tokenizers substitutes it for a character neither the vocabulary nor the
  // fallback can represent. fuse_unk decides whether a RUN of them is one token
  // or one each, so it is recorded rather than assumed.
  uint32_t byte_fallback = 0;
  if (flag("byte_fallback")) byte_fallback = 1u;
  const uint32_t fuse_unk = flag("fuse_unk") ? 1u : 0u;

  if (flag("ignore_merges"))
    fail("model.ignore_merges is true -- a word present in the vocabulary "
         "bypasses the merge loop entirely. Not implemented; it changes the "
         "segmentation of exactly the common words, so ignoring it would be "
         "undetectable in a spot check and wrong in a corpus.");
  if (const json::Value *d = m.find("dropout"))
    if (!d->is_null()) fail("model.dropout is set -- BPE-dropout is stochastic");

  // --- vocabulary -------------------------------------------------------
  const auto &vocab_obj = m.at("vocab").as_object();
  std::unordered_map<std::string, int32_t> vocab;
  vocab.reserve(vocab_obj.size() * 2);
  int32_t max_id = -1;
  for (const auto &kv : vocab_obj) {
    const int32_t id = static_cast<int32_t>(kv.second.as_number());
    vocab.emplace(kv.first, id);
    max_id = std::max(max_id, id);
  }
  if (static_cast<size_t>(max_id) + 1 != vocab.size())
    fail("vocab ids are not contiguous 0..n-1");
  std::vector<std::string> id_to_token(vocab.size());
  std::vector<bool> seen(vocab.size(), false);
  for (const auto &kv : vocab) {
    if (kv.second < 0 || static_cast<size_t>(kv.second) >= id_to_token.size())
      fail("vocab id out of range");
    if (seen[static_cast<size_t>(kv.second)]) fail("duplicate vocab id");
    seen[static_cast<size_t>(kv.second)] = true;
    id_to_token[static_cast<size_t>(kv.second)] = kv.first;
  }

  // --- merges -----------------------------------------------------------
  //
  // `tokenizers` >= 0.20 writes ["a", "b"] pairs; older files use "a b".
  // Byte-level pieces never contain a space (it is mapped to U+0120), so the
  // split is unambiguous either way, and both forms are read.
  const auto &merges_arr = m.at("merges").as_array();
  std::vector<std::array<uint32_t, 3>> merges;
  merges.reserve(merges_arr.size());
  for (const json::Value &e : merges_arr) {
    std::string a, b;
    if (e.is_array()) {
      const auto &p = e.as_array();
      if (p.size() != 2) fail("a merge entry is not a pair");
      a = p[0].as_string();
      b = p[1].as_string();
    } else {
      const std::string &s = e.as_string();
      const size_t sp = s.find(' ');
      if (sp == std::string::npos) fail("a merge entry has no separator");
      a = s.substr(0, sp);
      b = s.substr(sp + 1);
    }
    auto ia = vocab.find(a), ib = vocab.find(b), im = vocab.find(a + b);
    if (ia == vocab.end() || ib == vocab.end() || im == vocab.end())
      fail("merge '" + a + "'+'" + b +
           "' references a piece not in the vocabulary");
    merges.push_back({static_cast<uint32_t>(ia->second),
                      static_cast<uint32_t>(ib->second),
                      static_cast<uint32_t>(im->second)});
  }

  // --- added tokens -----------------------------------------------------
  struct Added { std::string content; int32_t id; uint32_t flags; };
  std::vector<Added> added;
  if (const json::Value *at = j.find("added_tokens")) {
    for (const json::Value &t : at->as_array()) {
      auto boolean = [&](const char *k) {
        const json::Value *v = t.find(k);
        return v != nullptr && v->is_bool() && v->as_bool();
      };
      if (boolean("single_word"))
        fail("an added token sets single_word -- that constrains the match to "
             "word boundaries and is not implemented");
      if (boolean("rstrip"))
        fail("an added token sets rstrip -- not implemented, because no "
             "checkpoint available here sets it and the branch would ship "
             "unverified (lstrip IS implemented and IS exercised, by [MASK])");
      const uint32_t flags = boolean("lstrip") ? 1u : 0u;
      added.push_back(Added{t.at("content").as_string(),
                            static_cast<int32_t>(t.at("id").as_number()),
                            flags});
    }
  }

  // --- named special ids ------------------------------------------------
  //
  // Searched in the added tokens too, not just model.vocab: granite-4.2-3b
  // keeps `<|padding|>` only in added_tokens.
  std::unordered_map<std::string, int32_t> lookup = vocab;
  for (const Added &a : added) lookup.emplace(a.content, a.id);
  auto named = [&](std::initializer_list<const char *> names) -> int32_t {
    for (const char *n : names) {
      auto it = lookup.find(n);
      if (it != lookup.end()) return it->second;
    }
    return -1;
  };
  const int32_t cls_id = named({"[CLS]", "<s>", "<|endoftext|>"});
  const int32_t sep_id = named({"[SEP]", "</s>"});
  const int32_t pad_id = named({"[PAD]", "<pad>", "<|padding|>"});
  const int32_t unk_id = named({"[UNK]", "<unk>"});
  const int32_t mask_id = named({"[MASK]", "<mask>"});

  // --- post processor ---------------------------------------------------
  std::vector<int32_t> prefix_ids, suffix_ids;
  const json::Value *post = j.find("post_processor");
  if (post != nullptr && !post->is_null()) {
    if (post->at("type").as_string() != "TemplateProcessing")
      fail("post_processor is not TemplateProcessing -- ByteLevel and "
           "RobertaProcessing exist in this family and are NOT the same "
           "wrapping");
    const json::Value *specials = post->find("special_tokens");
    bool seen_sequence = false;
    const json::Value *single = post->find("single");
    if (single == nullptr) fail("post_processor has no `single` template");
    for (const json::Value &item : single->as_array()) {
      if (item.contains("Sequence")) {
        if (seen_sequence) fail("post_processor template has two sequences");
        seen_sequence = true;
        continue;
      }
      if (item.contains("SpecialToken")) {
        const std::string &name = item.at("SpecialToken").at("id").as_string();
        if (specials == nullptr || !specials->contains(name))
          fail("template names special token '" + name +
               "' with no entry in special_tokens");
        const auto &ids = specials->at(name).at("ids").as_array();
        if (ids.size() != 1)
          fail("special token '" + name + "' does not map to exactly one id");
        (seen_sequence ? suffix_ids : prefix_ids)
            .push_back(static_cast<int32_t>(ids[0].as_number()));
        continue;
      }
      fail("unrecognised post_processor template item");
    }
    if (!seen_sequence)
      fail("post_processor template never places the sequence");
  }

  // --- the alphabet must be closed -------------------------------------
  //
  // The right invariant depends on the alphabet, and getting this backwards is
  // how a byte-mode table passes a check that means nothing for it.
  //
  // byte_mode 0 (GPT-2 byte map): every byte that can occur in valid UTF-8 must
  // be a vocabulary entry. 0xC0, 0xC1 and 0xF5-0xFF cannot occur, so a
  // vocabulary may legitimately omit them.
  //
  // byte_mode 1 (raw characters): there is no fixed 256-entry requirement at
  // all -- there are 1.1M codepoints -- which is precisely why `byte_fallback`
  // exists. What CAN be checked is that the fallback alphabet itself is
  // closed, because byte_fallback is the promise that any byte tokenizes: the
  // `<0xNN>` pieces must be there. One of them may legitimately be absent when
  // that byte already has a character of its own (this checkpoint has no
  // `<0x09>` because tab is id 226), so the check is "at least 255 of 256" and
  // the exception is named rather than silent.
  if (byte_mode == 0) {
    const std::vector<std::string> bc = byte_chars();
    std::vector<int> unexpected;
    for (int b = 0; b < 256; ++b) {
      if (vocab.count(bc[static_cast<size_t>(b)])) continue;
      if (b == 0xC0 || b == 0xC1 || b >= 0xF5) continue;
      unexpected.push_back(b);
    }
    if (!unexpected.empty()) {
      char buf[8];
      std::snprintf(buf, sizeof(buf), "0x%02X", unexpected[0]);
      fail(std::to_string(unexpected.size()) +
           " byte characters are absent from the vocabulary (first: " + buf +
           ") -- a byte-level BPE vocabulary must contain every byte that can "
           "occur in UTF-8");
    }
  } else if (byte_fallback) {
    int present = 0, absent_first = -1;
    for (int b = 0; b < 256; ++b) {
      char key[8];
      std::snprintf(key, sizeof(key), "<0x%02X>", b);
      if (vocab.count(key)) ++present;
      else if (absent_first < 0) absent_first = b;
    }
    if (present < 255) {
      char buf[8];
      std::snprintf(buf, sizeof(buf), "0x%02X", absent_first);
      fail("byte_fallback is true but only " + std::to_string(present) +
           " of the 256 <0xNN> pieces are in the vocabulary (first missing: " +
           buf + "). byte_fallback is the promise that ANY byte tokenizes; "
           "with fewer than 256 it does not, and a character that needs it "
           "would silently become <unk>.");
    }
  }

  // --- emit -------------------------------------------------------------
  //
  // VERSION 2 appends to the end of the version-1 layout. Appending rather than
  // inserting is the whole point: a v1 reader reads a fixed prefix and stops,
  // so every already-validated v1 table this generator has ever written keeps
  // working, and a v1 table inside a v2 container is still readable. The magic
  // stays BBPETOK1 and the VERSION field is what branches -- see
  // tokenizer_bbpe.cpp's build_index, which refuses a version it does not
  // implement by name rather than misreading the tail as vocabulary.
  std::vector<uint8_t> o;
  const char magic[8] = {'B', 'B', 'P', 'E', 'T', 'O', 'K', '1'};
  o.insert(o.end(), magic, magic + 8);
  put_u32(o, 2);                                   // version
  put_u32(o, norm);
  put_u32(o, add_prefix_space);
  put_u32(o, static_cast<uint32_t>(id_to_token.size()));
  put_u32(o, static_cast<uint32_t>(merges.size()));
  put_u32(o, static_cast<uint32_t>(added.size()));
  put_i32(o, cls_id);
  put_i32(o, sep_id);
  put_i32(o, pad_id);
  put_i32(o, unk_id);
  put_i32(o, mask_id);
  put_u32(o, static_cast<uint32_t>(prefix_ids.size()));
  for (int32_t i : prefix_ids) put_i32(o, i);
  put_u32(o, static_cast<uint32_t>(suffix_ids.size()));
  for (int32_t i : suffix_ids) put_i32(o, i);
  for (const std::string &t : id_to_token) put_str(o, t);
  for (const auto &mg : merges) {
    put_u32(o, mg[0]);
    put_u32(o, mg[1]);
    put_u32(o, mg[2]);
  }
  for (const Added &a : added) {
    put_str(o, a.content);
    put_i32(o, a.id);
    put_u32(o, a.flags);
  }
  // --- version 2 tail
  put_u32(o, pre_tok);            // 0 ByteLevel, 1 Metaspace
  put_str(o, norm_replacement);   // used only when norm == 2
  put_u32(o, prepend_scheme);     // used only when pre_tok == 1
  put_str(o, replacement);        // used only when pre_tok == 1
  put_u32(o, metaspace_split);    // used only when pre_tok == 1
  put_u32(o, byte_fallback);
  put_u32(o, fuse_unk);
  put_u32(o, byte_mode);          // 0 GPT-2 byte map, 1 raw characters
  return o;
}

}  // namespace npue
