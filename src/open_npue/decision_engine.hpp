// The decision head: two host nn.TransformerEncoderLayer-equivalents, the
// marker gather, and the readout.
//
// WHY THE HOST, and this is a placement decision rather than a capability
// limit -- the four reasons, shortest first: the encoder's geometry globals
// (`g_ffn`, `g_layers`, `g_heads`) are PROCESS-WIDE, so a second Encoder over
// the head's 3072-wide FFN would redefine the encoder's; every `npu::Design` is
// its own `hw_context`, and one model resolves to one design set; the encoder
// hardcodes its tensor prefix, so the head's names (`head.layers.N.*`,
// `scorer.*`, `act_head.*`) are not reachable through it at all; and the head is
// 8 of the model's 96 GEMMs. Running it here costs one host pass over
// [rows, seq, 768] and buys a second geometry nobody asked for.
//
// WHAT IS DIFFERENT FROM THE ENCODER, and every one of these is a wrong answer
// if you inherit the encoder's version:
//
//   * ReLU, not GELU. torch's DEFAULT nn.TransformerEncoderLayer activation.
//     The head is 2 layers deep and the encoder's is 22, so a swapped
//     activation is a slightly wrong model rather than an obviously broken one.
//   * WITH biases, at every site. The encoder is bias-free (the packer
//     zero-fills them); the head's are real numbers in the checkpoint.
//   * The attention scale is EXPLICIT here. For arch=0/2/3 the packer folds
//     1/sqrt(head_dim) into the Q block of the qkv weight and `qk_impl` computes
//     a raw dot product; these weights are packed as plain host F32 with no fold,
//     so the scale is applied HERE, to the Q rows only, AFTER the projection.
//     Folding it into the weight before the GEMM would put it on K and V too.
//   * `in_proj` is [3d, d] in [Q|K|V] order and the head is plain MHA: Q, K and
//     V are each [rows, seq, d] and are then VIEWED as [rows, seq, Nh, Dh].
//     That is NOT the encoder's fused ModernBERT (3, Nh, Dh) interleave, and no
//     RoPE is applied -- nn.MultiheadAttention has none.
//   * FULL-BAND attention: no sliding window, no causal mask, padding mask only.
//     `head_attention` below does not read `band_now` at all. The plan's own
//     reason is the one worth repeating: a banded head is a wrong answer, not a
//     slow one, and a caller that forgets to clear a shared flag gets exactly
//     that. There is no flag to forget.
//   * norm_first=True, and the residual adds are on the UNNORMALISED stream.
#ifndef OPENFLOWLM_DECISION_ENGINE_H
#define OPENFLOWLM_DECISION_ENGINE_H

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "npue_encoder.hpp"

namespace npue {
namespace dec {

/// QTYPES order, and it is an ORDER and not a set: `type_emb.weight` is
/// [3, d] indexed by it, so a different numbering silently embeds the wrong
/// row for every question. Upstream's `laya/common.py:17`.
enum : int64_t { QTYPE_CHOICE = 0, QTYPE_SCORE = 1, QTYPE_NOUL = 2 };

/// One (state, question) row, already tokenised. The prompt builder's output;
/// the engine does not know how a prompt is formatted, which is deliberate --
/// see `build_prompt` for why.
/// Upstream's temperature bucket key, `common.py:543`:
///   size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
///   key  = "<qtype name>:<size>"
///
/// It is keyed on k and NOT on the question type alone because the fitted
/// temperatures are per (type, option count): a 12-option choice is a different
/// calibration from a 2-option noul, and upstream fits them separately. Reading
/// the map by qtype alone would silently apply a 2-option calibration to a
/// 12-option question, which is the error the bucket exists to prevent.
inline std::string temp_bucket(int64_t qtype, int64_t k) {
  const char *name = "choice";
  if (qtype == QTYPE_SCORE) name = "score";
  else if (qtype == QTYPE_NOUL) name = "noul";
  const char *size = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+";
  return std::string(name) + ":" + size;
}

struct PromptRow {
  std::vector<int32_t> ids;
  /// Where the [MASK] markers are, one per option, in `ids`. Length is this
  /// row's k. A marker at or past `max_seq_len` was truncated away upstream and
  /// is NOT here: `k` is the number of options the head actually reached.
  std::vector<int64_t> markers;
  int64_t qtype = QTYPE_CHOICE;
  /// The caller's option order, for the unpermute. Empty means identity.
  std::vector<int> option_order;
};

/// What one row produced. Probabilities and logits are length k, this row's own
/// marker count -- never the batch's kmax, which is what upstream softmaxes over
/// and then slices.
struct RowAnswer {
  int64_t k = 0;
  std::vector<float> logits;          ///< raw, pre-temperature, pre-mask
  std::vector<float> probabilities;   ///< after temperature, length k
  float confidence = 0.f;             ///< max(p) after temperature
  float score = 0.f;                  ///< sum(i * p_i), i from 0
  int64_t argmax = 0;                 ///< index into probabilities
  /// act_head's output. COMPUTED AND DISCARDED, and both halves matter.
  ///
  /// Discarded: Laya's own issue tracker records act_probability at AUROC 0.30
  /// against 0.77 for confidence, so it is dead weight.
  ///
  /// Computed: it is in the checkpoint, and skipping it is a SILENT divergence
  /// -- a missing tensor and a deliberately-unused one are different things,
  /// and only the second is visible in this struct. Kept here so a future
  /// surface that wants it does not have to re-derive it from the checkpoint.
  std::vector<float> act;
};

/// One host LayerNorm + Linear + activation chain, in the shape the head needs.
/// Weights are [N, K] row-major exactly as the container stores them
/// (`gemm_b_host`, untiled), so `w` is read with NO transpose: the packer
/// deliberately did not tile these, and tiling is what makes every other B
/// operand in this runtime need a layout-aware read.
struct HostLinear {
  const float *w = nullptr;   ///< [N, K]
  const float *b = nullptr;   ///< [N] or null
  int64_t n = 0, k = 0;
};

struct HostLayerNorm {
  const float *w = nullptr;   ///< [d]
  const float *b = nullptr;   ///< [d]
  int64_t d = 0;
};

/// One head layer: the checkpoint's `head.layers.N.*`, renamed to what the maths
/// calls them. The container's names are PyTorch's; these are ours, and the
/// mapping is stated in the comment above.
struct HeadLayer {
  HostLayerNorm norm1, norm2;
  HostLinear in_proj, out_proj, linear1, linear2;
};

/// The host head. Constructed from a container; every weight is a pointer into
/// the mapped file, so this owns no tensor memory and CopyConstruct is
/// harmless -- but the Encoder it shares with is not, which is why `Decider`
/// holds both and is non-copyable.
class Head {
 public:
  explicit Head(const npue::File &m);

  int64_t layers() const { return static_cast<int64_t>(w_.size()); }
  int64_t hidden() const { return hidden_; }
  int64_t heads() const { return heads_; }
  int64_t head_dim() const { return head_dim_; }
  int64_t act_out() const { return act_out_; }

  /// `h` is [rows, seq, d] in, and out. Overwrites it in place -- the head is a
  /// residual stack, so every intermediate is a temporary and keeping `h` as the
  /// only caller-visible buffer is the whole point.
  ///
  /// `pad` is [rows, seq], 1 for real tokens. `workers` threads; 0 means one.
  void forward(std::vector<float> &h, const std::vector<float> &pad,
               int64_t rows, int64_t seq, int workers) const;

  /// The readout, over gathered marker rows: LayerNorm -> Linear -> GELU ->
  /// Linear(->1), on [rows, k, d]. Returns [rows, k] raw logits, NOT masked and
  /// NOT temperature-scaled -- both of those are the caller's, because the
  /// mask depends on the caller's marker counts.
  void score(const std::vector<float> &gathered, int64_t rows, int64_t k,
             int64_t d, int workers, std::vector<float> &logits) const;

  /// act_head, computed and discarded. `pooled` is [rows, d] (h[:, 0]),
  /// `feats` is [rows, 4] -- top1, top1-top2, normalised entropy, k/255 -- and
  /// the two are concatenated because the checkpoint's first layer is
  /// [256, d + 4]. The feature scale is not arbitrary: k/255 is calibrated
  /// against the 255-option cap.
  void act(const std::vector<float> &pooled, const std::vector<float> &feats,
           int64_t rows, int64_t d, int workers, std::vector<float> &out) const;

 private:
  std::vector<HeadLayer> w_;
  HostLayerNorm scorer_ln_;
  HostLinear scorer_fc1_, scorer_fc2_;
  HostLinear act_fc1_, act_fc2_;
  const float *type_emb_ = nullptr;   ///< [3, d]
  int64_t hidden_ = 0, heads_ = 0, head_dim_ = 0, act_hidden_ = 0,
          act_out_ = 0;
  // MUTABLE, and the reason is worth one line: forward/score/act are const
  // because they do not change the MODEL -- the weights are pointers into a
  // mapped file and cannot change -- while they do allocate and reuse a
  // workspace. Making them non-const instead would mean the caller could not
  // hold a `const Head&` for a Decider it only wants to read, which is the
  // normal way to use it. The scratch is not observable state, so `mutable` is
  // the honest annotation rather than a loophole.
  mutable std::vector<float> scratch_, scratch2_, scratch3_, scores_, ctx_, qkv_;
};

/// The engine: a container, the NPU encoder that runs the 22 encoder layers,
/// and the host head that runs the 2 head layers. One model, one design set,
/// one hw_context.
class Decider {
 public:
  struct Options {
    std::string container;      ///< the .npue
    std::string artifacts_dir;  ///< the design set
    int threads = 1;
    int lanes = 1;
  };
  explicit Decider(const Options &o);
  ~Decider();

  /// `rows` is one entry per (state, question) pair. Each is independent, so
  /// they batch: `EmbedService::plan()` splits them against the tier ladder
  /// greedily, which is what keeps 5 questions at 4+1 rather than one padded
  /// 16-row pass.
  ///
  /// Serialised by a mutex: the Encoder holds one hw_context and two concurrent
  /// decides would interleave dispatches into it.
  /// `temperature` is the PER-QTYPE base, `temperature[qtype]`.
  ///
  /// `temperature_is_override` says the caller means it for every row, which
  /// bypasses the container's per-bucket `temperature_by_options`. Without the
  /// flag a caller override would be applied as the base and then overwritten by
  /// the bucket for exactly the option counts the checkpoint fitted -- which is
  /// the bug this parameter exists to make impossible: an override that is
  /// silently ignored for most of what it was sent for.
  std::vector<RowAnswer> decide(const std::vector<PromptRow> &rows,
                                const std::vector<float> &temperature,
                                bool temperature_is_override = false);

  /// The container's `temperature_by_options`, verbatim. Empty for this
  /// checkpoint, which is why the plumbing is untested by the default model and
  /// why the fixture in the handoff has to run against a synthetic one.
  const std::map<std::string, float> &temperature_by_options() const {
    return temperature_by_options_;
  }

  int64_t max_seq() const;
  int64_t hidden() const;
  int64_t head_max_len() const { return head_max_len_; }
  int64_t max_seq_len() const { return max_seq_len_; }
  const Head &head() const { return *head_; }
  /// Workers for the HOST tail. The encoder's own thread budget is separate:
  /// it is `Options::threads` too, but the head's work is arithmetic and
  /// parallelises differently from a design's dispatch stream.
  int o_threads() const { return o_.threads; }

  /// `[CLS] <type> question: <ins> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP]
  /// <state> [SEP]`, with the head budget applied.
  ///
  /// A port of upstream's `build_sequence` (laya/common.py:135-228), and the
  /// three parts of it that are easy to get wrong:
  ///
  ///   * the marker is recorded BEFORE its option's tokens are appended, so it
  ///     points AT the [MASK] and the gather reads the marker's own embedding;
  ///   * the option budget `head_max_len - sum(len(o))` and the `>= 16` floor
  ///     with `per_option = max(4, (head_max_len - 16) / n_opts)` are what stops
  ///     a long instruction from consuming the room the options need, and
  ///     upstream applies them in that order;
  ///   * the state is clamped to the room LEFT, so a long state loses tokens
  ///     silently -- and `state_dropped` is reported so a caller can tell.
  /// The option TEXTS, in the caller's option order. A port of upstream's
  /// `render_options` (laya/common.py:106-132), and the LABELS are load-bearing
  /// in a way that is easy to miss: the option text is not the criterion, it is
  /// `"<label>: <criterion>"`, and the label is what tells the model which slot
  /// it is reading. Sending the bare criterion is a well-formed prompt that
  /// puts a confident model at chance -- the marker positions still land, the
  /// row still has the right k, and the logits come back flat.
  ///
  ///   choice  "<label>: <criterion>", or bare `label` when there is none
  ///   score   "level <i>: <criterion>", i from 0 -- the ORDER IS THE SCALE
  ///   noul    "<false label>: <criterion>" then "<true label>: <criterion>",
  ///           always in the semantic order [false, true], with the default
  ///           labels "false"/"true" and the fallbacks "no, the statement does
  ///           not hold" / "yes, the statement holds"
  ///
  /// `criteria` is one entry per option and is IGNORED for a noul with none.
  static std::vector<std::string> render_options(
      int64_t qtype, const std::vector<std::string> &labels,
      const std::vector<std::string> &criteria,
      const std::vector<std::string> &noul_labels = {});

  struct Prompt {
    std::vector<int32_t> ids;
    std::vector<int64_t> markers;
    int64_t state_tokens = 0, state_used = 0;
  };
  Prompt build_prompt(const std::string &type_text,
                      const std::string &instruction, const std::string &state,
                      const std::vector<std::string> &options,
                      int64_t qtype) const;

 private:
  Decider(const Decider &) = delete;
  Decider &operator=(const Decider &) = delete;

  // DECLARATION ORDER IS DESTRUCTION ORDER, REVERSED. The encoders hold
  // pointers into the pools, the designs and the shape globals, so they are
  // declared FIRST and therefore destroyed LAST; a Pool outliving its Encoder is
  // a use-after-free at exit. Same rule as npue_encoder.hpp's own Stack.
  // FIRST, and therefore the first destroyed -- see the member-order comment.
  // It is read by o_threads() after everything else is gone, which is why it is
  // a value and not a reference to the caller's Options.
  Options o_;
  npue::File model_;
  npue::enc::ShapeLease lease_;
  npue::npu::Device dev_;
  npue::enc::Stack stack_;
  std::unique_ptr<npue::enc::EmbedService> svc_;
  std::unique_ptr<Head> head_;
  /// Private, with a public const getter above. It is MODEL STATE read from the
  /// container at load, not part of the caller's contract: a public data member
  /// invites `dec.temperature_by_options_["choice:11+"] = 0.1` from outside, and
  /// 0.1 is precisely the value upstream refuses because it sharpens logits
  /// ~10x. The range check in the constructor is then bypassable from outside.
  std::map<std::string, float> temperature_by_options_;
  int64_t head_max_len_ = 0, max_seq_len_ = 0;
  std::mutex call_mu_;
};

}  // namespace dec
}  // namespace npue

#endif
