//===- tokenizer_bbpe.hpp ------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- byte-level BPE (GPT-2 / OLMo / Qwen / tekken family).
// SPDX-License-Identifier: MIT
//
// The fourth tokenizer in this tree, and the one T43 named as the gate on
// every modern encoder:
//
//   WordPiece             tokenizer.hpp        arch 0, arch 2
//   SentencePiece BPE     tokenizer_gemma.hpp  arch 1
//   SentencePiece Unigram tokenizer_xlmr.hpp   arch 3
//   byte-level BPE        this                 -- ModernBERT, Qwen3, Ministral
//
// WHAT IS ACTUALLY NEW. The merge engine is the same one tokenizer_gemma.cpp
// runs -- linked list plus a lazily-invalidated min-heap on merge rank -- and
// it is deliberately the same code shape so the two can be read against each
// other. What is new is everything around it:
//
//   * an ADDED-TOKEN matcher (leftmost-longest, with lstrip), which the
//     SentencePiece tokenizers do not need,
//   * an NFC normalizer,
//   * the GPT-2 PRE-TOKENIZER -- a regex split into words, each of which is
//     BPE'd independently so no merge ever crosses a word boundary, and
//   * the BYTE-TO-UNICODE map, which replaces byte-fallback: every byte is
//     already a vocabulary entry, so there is no unknown piece and no <unk>.
//
// The pre-tokenizer's character classes come from bbpe_unicode_tables.hpp,
// which is MEASURED against HuggingFace's own splitter rather than computed
// from Python's unicodedata -- the two disagree on 4,386 codepoints, and the
// generator's docstring says exactly which. That is the difference between
// this file agreeing with HuggingFace and merely being written carefully.
//
// The vocabulary, merges, added tokens and post-processor wrapping are
// GENERATED offline by tools/gen_bbpe_tokenizer_table.py (and by its C++ port
// in bbpe_tokenizer_gen.hpp, so a fresh clone can pack without Python) into a
// flat `BBPETOK1` binary this class reads with ifstream + memcpy -- no JSON at
// runtime, per CLAUDE.md rule 5.
//
// VERSION 2 (convaiinnovations/laya's multilingual tokenizer) adds a `Replace`
// normalizer, the Metaspace pre-tokenizer, and two recorded flags. It APPENDS to
// the v1 layout rather than inserting into it, and the version field is what
// branches: a v1 reader reads its fixed prefix and stops. BOTH versions are read
// here, because a blob is data and data outlives the writer.
//
// Correctness is measured, not asserted: tools/verify_tokenizer_bbpe.py runs
// this implementation and HuggingFace's over the same corpus and requires
// exact agreement on every id.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace npue {

struct BbpeEncoded {
  std::vector<int32_t> input_ids;
  std::vector<int32_t> attention_mask;   // 1 for real tokens, 0 for padding
  int32_t n_tokens = 0;                  // before padding, incl. specials

  // The untruncated count, and whether anything was dropped -- the same pair
  // as npue::Encoded and GemmaEncoded carry, for the same reason: since
  // tasks/0110 an input that does not fit is an ERROR, and the caller cannot
  // raise it without knowing the real length.
  int32_t n_tokens_full = 0;
  bool truncated = false;
};

class BbpeTokenizer {
public:
  static BbpeTokenizer from_table_file(const std::string &path);
  static BbpeTokenizer from_table_bytes(const char *data, size_t bytes);

  // Ids for one text WITHOUT the post-processor's wrapping -- the added-token
  // split, normalization, pre-tokenization and BPE only. Exposed for
  // debugging and for tools/verify_tokenizer_bbpe.py, which compares against
  // HuggingFace's `add_special_tokens=False`.
  std::vector<int32_t> tokenize(const std::string &text) const;

  // The template's prefix ids + tokenize(text) + its suffix ids, padded or
  // truncated to `max_len`. Throws if the table records no padding token and
  // padding is actually needed -- a decoder checkpoint has none, and
  // inventing one is invisible until something reads the padded positions.
  BbpeEncoded encode(const std::string &text, int max_len) const;
  std::vector<BbpeEncoded> encode_batch(const std::vector<std::string> &texts,
                                        int max_len) const;

  size_t vocab_size() const { return id_to_token_.size(); }
  int32_t id_of(const std::string &token) const;
  const std::string &token_of(int32_t id) const;

  // How many ids the wrapping costs, so a caller can budget `max_len`.
  int32_t n_special() const {
    return static_cast<int32_t>(prefix_ids_.size() + suffix_ids_.size());
  }

  int32_t cls_id = -1, sep_id = -1, pad_id = -1, unk_id = -1, mask_id = -1;
  // 0 = none, 1 = NFC, 2 = Replace(literal -> literal). Read from the table,
  // never assumed -- all three occur in this family and the difference is
  // silent. For 2, `norm_replacement` is what every literal space becomes.
  uint32_t normalizer = 0;
  std::string norm_replacement;
  bool add_prefix_space = false;

  // Which pre-tokenizer this table was generated for. 0 = ByteLevel (the GPT-2
  // regex scanner), 1 = Metaspace. They are DIFFERENT segmenters rather than
  // two settings of one: Metaspace runs no regex at all, it splits on a single
  // replacement character keeping it at the head of the piece that follows, and
  // optionally prepends one per input. A checkpoint that wants one and gets the
  // other produces plausible ids that are wrong, so this is data.
  uint32_t pre_tokenizer = 0;
  // 0 never, 1 first, 2 always. Only meaningful for Metaspace.
  uint32_t prepend_scheme = 0;
  std::string replacement;      // the Metaspace replacement character
  bool metaspace_split = true;

  // Which ALPHABET the vocabulary is indexed by. This is the single most
  // consequential field in the table, and it comes from the checkpoint's
  // DECODER rather than from its pre-tokenizer.
  //
  //   0 = GPT-2 / RoBERTa byte-level: every byte is mapped through
  //       bytes_to_unicode() first, so a space becomes U+0120 and U+2581 is
  //       three bytes written as three odd codepoints.
  //   1 = raw characters (SentencePiece-shaped): the vocabulary is indexed by
  //       the characters themselves, U+2581 is one entry, and the `<0xNN>`
  //       pieces exist so a character nothing can represent still tokenizes.
  //
  // Same `model.type: BPE` and the same merge engine either way, and applying
  // the GPT-2 map to a mode-1 vocabulary produces ids that are all plausible and
  // none of them right. The trap is that a 256k SentencePiece-shaped vocabulary
  // of this kind HAPPENS to contain all 256 GPT-2 byte characters, so a
  // "the byte characters are all present" check passes for it and a runtime
  // that maps bytes anyway looks right until someone reads the ids.
  uint32_t byte_mode = 1;
  // Recorded rather than refused (bbpe_tokenizer_gen.cpp explains why the old
  // refusals were wrong for a vocabulary like this one).
  bool byte_fallback = false;
  // tokenizers' `fuse_unk`: a RUN of characters with no vocabulary entry is one
  // <unk>, not one each. Recorded because convaiinnovations/laya's multilingual
  // tokenizer sets it and a neighbouring checkpoint might not.
  bool fuse_unk = false;

private:
  void build_index(const std::string &blob);

  // Stage 3+4+5: split `cps` into words with this table's pre-tokenizer, then
  // byte-map and BPE each. One entry point so the two scanners cannot drift on
  // what happens to a word AFTER it is cut.
  void emit_words(const std::vector<uint32_t> &cps,
                  std::vector<int32_t> &out) const;
  void emit_gpt2_words(const std::vector<uint32_t> &cps,
                       std::vector<int32_t> &out) const;
  void emit_metaspace_words(const std::vector<uint32_t> &cps,
                            const std::vector<uint32_t> &reps,
                            std::vector<int32_t> &out) const;
  void byte_map_and_bpe(const std::vector<uint32_t> &cps, size_t lo, size_t hi,
                        std::vector<int32_t> &out) const;

  // One pre-tokenized, byte-mapped word -> its ids, appended to `out`.
  void bpe_word(const std::string &mapped, std::vector<int32_t> &out) const;
  // The merge engine itself, over an already-built symbol sequence. Shared by
  // both alphabets: it does not know or care which one produced the symbols,
  // which is the point -- the alphabet is a property of stage 4, not of BPE.
  void bpe_symbols(const std::vector<int32_t> &symbols,
                   std::vector<int32_t> &out) const;
  // byte_mode 1: one codepoint at a time, raw, with byte_fallback and <unk>.
  void bpe_word_raw(const std::string &word, std::vector<int32_t> &out) const;

  std::vector<std::string> id_to_token_;
  std::unordered_map<std::string, int32_t> token_to_id_;

  struct MergeInfo { uint32_t rank; int32_t merged_id; };
  std::unordered_map<uint64_t, MergeInfo> merge_of_;

  struct Added {
    std::string content;
    int32_t id;
    bool lstrip;
  };
  std::vector<Added> added_;
  size_t added_max_bytes_ = 0;

  std::vector<int32_t> prefix_ids_, suffix_ids_;
};

}  // namespace npue
