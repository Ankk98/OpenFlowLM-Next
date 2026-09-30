#include "decision_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace npue {
namespace dec {
namespace {

// torch's DEFAULT LayerNorm eps, and it is 1e-5 rather than the encoder's
// hard-coded 1e-12 (`npue_encoder.hpp:2019` and its three siblings) because
// upstream's head norms are `nn.LayerNorm(d)` / the default inside
// nn.TransformerEncoderLayer, and upstream's ENCODER norms use the config's
// layer_norm_eps -- which for this checkpoint is ALSO 1e-5.
//
// The plan's note calls the 1e-12 difference "~5e-6 relative, far under the
// bf16 operand's 8-bit mantissa". That argument covers the ENCODER, whose
// operands are bf16 on the NPU. The head runs here, in float32, with nothing to
// hide behind: at 5e-6 relative on a residual stream whose channels reach 242
// that is ~1e-3 absolute, and it compounds over the head's two layers plus the
// scorer. So the head takes 1e-5, and the only thing the encoder's 1e-12 buys is
// a difference nobody wanted.
constexpr float kHeadEps = 1e-5f;

// --------------------------------------------------------------------------
// The host primitives. Plain F32, no tiling, no bf16 -- the head's weights are
// `gemm_b_host` in the container, i.e. exactly [N, K] row-major float, which is
// the one B operand in this runtime that needs no layout-aware read. The
// attention is in float32 because upstream autocasts to bf16 and the difference
// is a DIAGNOSTIC number, not a gate (see the plan): argmax is the gate.
// --------------------------------------------------------------------------

/// out[m, n] = sum_k x[m, k] * w[n, k] + b[n], for m in [0, rows).
///
/// `w` is [N, K] ROW-MAJOR, which is what the container stores and what
/// PyTorch's `Linear.weight` is: out_features first. The k loop therefore walks
/// BOTH operands contiguously and n is the outer loop, because a [N, K] weight
/// indexed as `w[k * N + n]` is the TRANSPOSE -- and a transposed weight
/// produces a model that answers, with flat logits and a confidence near 0.5,
/// which is exactly what the first run of this showed.
///
/// Written this way rather than accumulating over k in the inner loop with n
/// outside for vectorisation: the k loop is already a contiguous dot product
/// the compiler can vectorise, and getting the index right is worth more than
/// the last factor of two. The head is 8 of the model's 96 GEMMs and runs on
/// the host by decision, not by necessity.
void gemm_nk(const float *x, const float *w, const float *b, int64_t rows,
             int64_t K, int64_t N, int64_t xstride, float *out,
             int64_t ostride) {
  for (int64_t m = 0; m < rows; ++m) {
    const float *xm = x + m * xstride;
    float *om = out + m * ostride;
    for (int64_t n = 0; n < N; ++n) {
      const float *wn = w + n * K;
      float acc = b ? b[n] : 0.f;
      for (int64_t k = 0; k < K; ++k) acc += xm[k] * wn[k];
      om[n] = acc;
    }
  }
}

void layer_norm_inplace(float *x, int64_t rows, int64_t d, const float *w,
                        const float *b, float eps) {
  for (int64_t m = 0; m < rows; ++m) {
    float *r = x + m * d;
    double mu = 0.0;
    for (int64_t i = 0; i < d; ++i) mu += r[i];
    mu /= static_cast<double>(d);
    double var = 0.0;
    for (int64_t i = 0; i < d; ++i) {
      const double t = r[i] - mu;
      var += t * t;
    }
    var /= static_cast<double>(d);
    const double inv = 1.0 / std::sqrt(var + eps);
    for (int64_t i = 0; i < d; ++i)
      r[i] = static_cast<float>((r[i] - mu) * inv) * w[i] + b[i];
  }
}

void relu_inplace(float *x, size_t n) {
  for (size_t i = 0; i < n; ++i)
    if (x[i] < 0.f) x[i] = 0.f;
}

void gelu_exact_inplace(float *x, size_t n) {
  for (size_t i = 0; i < n; ++i)
    x[i] = static_cast<float>(
        0.5 * static_cast<double>(x[i]) *
        (1.0 + std::erf(static_cast<double>(x[i]) *
                        0.70710678118654752440)));
}

void add_inplace(float *dst, const float *src, size_t n) {
  for (size_t i = 0; i < n; ++i) dst[i] += src[i];
}

/// Run `body(worker, nworkers)` over a thread pool of `workers` threads and join.
/// Zero threads means one, and a body that throws inside a thread would call
/// std::terminate, so nothing here throws: the callers are arithmetic on
/// validated shapes.
template <typename F>
void parallel(int workers, F body) {
  if (workers <= 1) {
    body(0, 1);
    return;
  }
  std::vector<std::thread> ts;
  ts.reserve(static_cast<size_t>(workers));
  for (int i = 0; i < workers; ++i)
    ts.emplace_back([&body, i, workers] { body(i, workers); });
  for (auto &t : ts) t.join();
}

}  // namespace

// --------------------------------------------------------------------------
// Head
// --------------------------------------------------------------------------

namespace {

HostLinear linear_of(const npue::File &m, const std::string &name, int64_t n,
                     int64_t k) {
  HostLinear L;
  L.w = m.raw(name + ".weight").as<float>();
  L.b = m.has(name + ".bias") ? m.raw(name + ".bias").as<float>() : nullptr;
  L.n = n;
  L.k = k;
  const auto &info = m.info(name + ".weight");
  if (info.logical_shape.size() != 2 ||
      info.logical_shape[0] != n || info.logical_shape[1] != k)
    throw std::runtime_error(
        name + ".weight is " +
        std::to_string(info.logical_shape[0]) + "x" +
        std::to_string(info.logical_shape.size() > 1 ? info.logical_shape[1] : 0) +
        ", expected " + std::to_string(n) + "x" + std::to_string(k) +
        ". A head weight with the wrong shape is a head that computes something, "
        "so the shape is checked against the CONTAINER rather than trusted from "
        "the config the caller also read.");
  if (info.dtype != "F32")
    throw std::runtime_error(name + ".weight is " + info.dtype +
                             ", expected F32: the head runs on the host and the "
                             "host arithmetic here is float32");
  return L;
}

HostLayerNorm ln_of(const npue::File &m, const std::string &name, int64_t d) {
  HostLayerNorm L;
  L.w = m.raw(name + ".weight").as<float>();
  L.b = m.has(name + ".bias") ? m.raw(name + ".bias").as<float>() : nullptr;
  L.d = d;
  if (m.info(name + ".weight").logical_shape.size() != 1 ||
      m.info(name + ".weight").logical_shape[0] != d)
    throw std::runtime_error(name + ".weight is not [" + std::to_string(d) +
                             "]");
  return L;
}

}  // namespace

Head::Head(const npue::File &m) {
  hidden_ = enc::g_hidden;
  heads_ = enc::g_heads;
  head_dim_ = enc::g_head_dim;
  // Upstream allows head_layers == 0, which makes the head IDENTITY: the
  // gathered rows go straight to the scorer. That is a model, and refusing it
  // would refuse a checkpoint this packer accepted -- so a negative value is
  // the refusal and zero is a configuration.
  const int64_t nl = m.config_int("head_layers");
  if (nl < 0)
    throw std::runtime_error("head_layers is " + std::to_string(nl) +
                             "; a negative layer count is not a configuration");
  const int64_t ffn = 4 * hidden_;
  w_.resize(static_cast<size_t>(nl));
  for (int64_t l = 0; l < nl; ++l) {
    const std::string p = "head.layers." + std::to_string(l) + ".";
    HeadLayer &L = w_[static_cast<size_t>(l)];
    L.norm1 = ln_of(m, p + "norm1", hidden_);
    L.norm2 = ln_of(m, p + "norm2", hidden_);
    // PyTorch's names, not ours. `in_proj` is the FUSED [3d, d] qkv in
    // [Q|K|V] order -- NOT the encoder's (3, Nh, Dh) interleave and NOT the
    // interleaved (Nh, 3, Dh) some transformers write.
    L.in_proj = linear_of(m, p + "self_attn.in_proj", 3 * hidden_, hidden_);
    L.out_proj = linear_of(m, p + "self_attn.out_proj", hidden_, hidden_);
    L.linear1 = linear_of(m, p + "linear1", ffn, hidden_);
    L.linear2 = linear_of(m, p + "linear2", hidden_, ffn);
  }
  scorer_ln_ = ln_of(m, "scorer.0", hidden_);
  scorer_fc1_ = linear_of(m, "scorer.1", hidden_, hidden_);
  scorer_fc2_ = linear_of(m, "scorer.3", 1, hidden_);
  act_hidden_ = m.info("act_head.0.weight").logical_shape[0];
  act_out_ = m.info("act_head.2.weight").logical_shape[0];
  act_fc1_ = linear_of(m, "act_head.0", act_hidden_, hidden_ + 4);
  act_fc2_ = linear_of(m, "act_head.2", act_out_, act_hidden_);
  type_emb_ = m.raw("type_emb.weight").as<float>();
  if (m.info("type_emb.weight").logical_shape.size() != 2 ||
      m.info("type_emb.weight").logical_shape[0] != 3 ||
      m.info("type_emb.weight").logical_shape[1] != hidden_)
    throw std::runtime_error(
        "type_emb.weight is not [3, hidden]. Its row index is the QTYPE order, "
        "so a wrong first dimension embeds the wrong question type for every "
        "row rather than refusing one.");
}

void Head::forward(std::vector<float> &h, const std::vector<float> &pad,
                   int64_t rows, int64_t seq, int workers) const {
  const int64_t d = hidden_, S = seq, R = rows;
  if (static_cast<int64_t>(h.size()) != R * S * d)
    throw std::runtime_error("head.forward: h is " +
                             std::to_string(h.size()) + " floats, expected " +
                             std::to_string(R * S * d));
  const float eps = kHeadEps;
  scratch_.resize(static_cast<size_t>(R * S * d));
  scratch2_.resize(static_cast<size_t>(R * S * d));
  qkv_.resize(static_cast<size_t>(R * S * 3 * d));
  scores_.resize(static_cast<size_t>(R * heads_ * S * S));
  ctx_.resize(static_cast<size_t>(R * S * d));

  // type_emb is added to the WHOLE sequence, before the head and before the
  // marker gather, and it is added by Decider rather than here: its row index is
  // the per-ROW question type, and threading an int per row through an
  // arithmetic function is how it ends up applied to the wrong axis. A reader
  // looking for the type embedding inside the head will not find it and should
  // know why.
  (void)pad;

  for (const HeadLayer &L : w_) {
    // --- pre-LN 1
    std::memcpy(scratch_.data(), h.data(), h.size() * sizeof(float));
    layer_norm_inplace(scratch_.data(), R * S, d, L.norm1.w, L.norm1.b, eps);
    gemm_nk(scratch_.data(), L.in_proj.w, L.in_proj.b, R * S, d, 3 * d, d,
            qkv_.data(), 3 * d);

    // --- FULL-BAND attention, padding-masked only.
    //
    // This reads no band width and has no flag to set: a head layer is global
    // attention, and a banded head is a wrong answer rather than a slow one.
    // scores[b, h, i, j] with j over the real tokens only.
    // 1/sqrt(head_dim) folded into the Q ROWS, ONCE, after the projection --
    // not into the weight before the GEMM (which would scale K and V too, since
    // in_proj is fused) and not inside the reduction (which rounds every
    // product separately). The encoder's packer folds the same factor into its
    // Q block; this one is packed plain, so the fold happens here.
    {
      const float sc = 1.0f / std::sqrt(static_cast<float>(head_dim_));
      for (int64_t i = 0; i < R * S * d; ++i) qkv_[static_cast<size_t>(i)] *= sc;
    }
    parallel(workers, [&](int wi, int ni) {
      const int64_t pairs = R * heads_;
      for (int64_t p = wi; p < pairs; p += ni) {
        const int64_t b = p / heads_, h = p % heads_;
        const float *qb = qkv_.data() + b * S * 3 * d + h * head_dim_;
        const float *kb = qkv_.data() + b * S * 3 * d + d + h * head_dim_;
        const float *vb = qkv_.data() + b * S * 3 * d + 2 * d + h * head_dim_;
        float *ctxb = ctx_.data() + b * S * d + h * head_dim_;
        for (int64_t i = 0; i < S; ++i) {
          const float *qi = qb + i * 3 * d;
          // ONE HEAD'S SLICE, and `head_dim_` floats of it -- not `d`.
          //
          // This was `t < d`, and it is the worst bug in the phase: `acc` points
          // at head h's slice of a d-wide row, so zeroing `d` floats from there
          // ERASES THE OTHER Nh-1 HEADS' OUTPUTS for this (b, i). Whichever
          // head happened to run last won the zeroing, so 11 of 12 heads'
          // attention outputs were destroyed -- and which 11 depended on thread
          // interleaving, which is why two identical runs disagreed.
          //
          // It did not crash and it did not look wrong. The model answered, with
          // confidences near 0.5, and every number was a plausible float.
          float *acc = ctxb + i * d;
          for (int64_t t = 0; t < head_dim_; ++t) acc[t] = 0.f;
          // max over j, then exp, then normalise. Two passes over j per i, which
          // is the price of not needing a scratch row per (i, h).
          float mx = -INFINITY;
          float *sc = scores_.data() + (p * S + i) * S;
          for (int64_t j = 0; j < S; ++j) {
            if (pad[b * S + j] <= 0.f) {
              sc[j] = -1.0e30f;
              continue;
            }
            const float *kj = kb + j * 3 * d;
            double a = 0.0;
            for (int64_t t = 0; t < head_dim_; ++t)
              a += static_cast<double>(qi[t]) * static_cast<double>(kj[t]);
            sc[j] = static_cast<float>(a);
            if (sc[j] > mx) mx = sc[j];
          }
          if (mx == -INFINITY) {
            // A fully-masked query row. Upstream's sdpa would produce NaN here
            // and softmax(NaN) is NaN, which is a real upstream behaviour -- but
            // a NaN here propagates into the residual stream and out of the
            // gather, so the row is zeroed and the caller's answer for it is
            // uniform. Upstream cannot reach this: it never builds an
            // all-padding row, because collate_items refuses when
            // marker_mask.sum() would be zero. `head_dim_` floats, for the same
            // reason as the zeroing above: `d` would erase the other heads.
            for (int64_t t = 0; t < head_dim_; ++t) acc[t] = 0.f;
            continue;
          }
          float sum = 0.f;
          for (int64_t j = 0; j < S; ++j) {
            const float e = pad[b * S + j] > 0.f
                                ? std::exp(sc[j] - mx)
                                : 0.f;
            sc[j] = e;
            sum += e;
          }
          const float inv = sum > 0.f ? 1.0f / sum : 0.f;
          for (int64_t j = 0; j < S; ++j) {
            const float w8 = sc[j] * inv;
            if (w8 == 0.f) continue;
            const float *vj = vb + j * 3 * d;
            for (int64_t t = 0; t < head_dim_; ++t) acc[t] += w8 * vj[t];
          }
        }
      }
    });

    gemm_nk(ctx_.data(), L.out_proj.w, L.out_proj.b, R * S, d, d, d,
            scratch2_.data(), d);
    add_inplace(h.data(), scratch2_.data(), static_cast<size_t>(R * S * d));

    // --- pre-LN 2 and the FFN. ReLU, not GELU: torch's DEFAULT
    // nn.TransformerEncoderLayer activation. WITH biases.
    std::memcpy(scratch_.data(), h.data(), h.size() * sizeof(float));
    layer_norm_inplace(scratch_.data(), R * S, d, L.norm2.w, L.norm2.b, eps);
    scratch2_.resize(static_cast<size_t>(R * S) * static_cast<size_t>(L.linear1.n));
    gemm_nk(scratch_.data(), L.linear1.w, L.linear1.b, R * S, d, L.linear1.n, d,
            scratch2_.data(), L.linear1.n);
    relu_inplace(scratch2_.data(),
                 static_cast<size_t>(R * S) * static_cast<size_t>(L.linear1.n));
    scratch3_.resize(static_cast<size_t>(R * S * d));
    gemm_nk(scratch2_.data(), L.linear2.w, L.linear2.b, R * S, L.linear1.n, d,
            L.linear1.n, scratch3_.data(), d);
    add_inplace(h.data(), scratch3_.data(), static_cast<size_t>(R * S * d));
  }
  (void)pad;
}

void Head::score(const std::vector<float> &gathered, int64_t rows, int64_t k,
                 int64_t d, int workers, std::vector<float> &logits) const {
  const int64_t n = rows * k;
  if (static_cast<int64_t>(gathered.size()) != n * d)
    throw std::runtime_error("Head::score: gathered is " +
                             std::to_string(gathered.size()) + " floats, "
                             "expected " + std::to_string(n * d));
  const float eps = kHeadEps;
  scratch_.assign(gathered.begin(), gathered.end());
  layer_norm_inplace(scratch_.data(), n, d, scorer_ln_.w, scorer_ln_.b, eps);
  scratch2_.resize(static_cast<size_t>(n) * static_cast<size_t>(scorer_fc1_.n));
  gemm_nk(scratch_.data(), scorer_fc1_.w, scorer_fc1_.b, n, d, scorer_fc1_.n, d,
          scratch2_.data(), scorer_fc1_.n);
  gelu_exact_inplace(scratch2_.data(),
                     static_cast<size_t>(n) * static_cast<size_t>(scorer_fc1_.n));
  logits.assign(static_cast<size_t>(n), 0.f);
  scratch3_.resize(static_cast<size_t>(n));
  gemm_nk(scratch2_.data(), scorer_fc2_.w, scorer_fc2_.b, n, scorer_fc1_.n, 1,
          scorer_fc1_.n, scratch3_.data(), 1);
  for (int64_t i = 0; i < n; ++i) logits[static_cast<size_t>(i)] = scratch3_[static_cast<size_t>(i)];
  (void)workers;
  (void)k;
}

void Head::act(const std::vector<float> &pooled, const std::vector<float> &feats,
               int64_t rows, int64_t d, int workers,
               std::vector<float> &out) const {
  // [rows, d + 4]: h[:, 0] with the four hand-built features appended. The
  // concatenation is the checkpoint's own shape -- act_head.0 is [256, d + 4]
  // -- so it is done here rather than in two GEMMs, which would need an
  // intermediate the container does not have.
  const int64_t K = d + 4;
  scratch_.resize(static_cast<size_t>(rows) * static_cast<size_t>(K));
  for (int64_t r = 0; r < rows; ++r) {
    std::memcpy(scratch_.data() + r * K, pooled.data() + r * d,
                static_cast<size_t>(d) * sizeof(float));
    std::memcpy(scratch_.data() + r * K + d, feats.data() + r * 4,
                4 * sizeof(float));
  }
  scratch2_.resize(static_cast<size_t>(rows) * static_cast<size_t>(act_hidden_));
  gemm_nk(scratch_.data(), act_fc1_.w, act_fc1_.b, rows, K, act_hidden_, K,
          scratch2_.data(), act_hidden_);
  gelu_exact_inplace(scratch2_.data(),
                     static_cast<size_t>(rows) * static_cast<size_t>(act_hidden_));
  out.assign(static_cast<size_t>(rows) * static_cast<size_t>(act_out_), 0.f);
  gemm_nk(scratch2_.data(), act_fc2_.w, act_fc2_.b, rows, act_hidden_, act_out_,
          act_hidden_, out.data(), act_out_);
  (void)workers;
}

// --------------------------------------------------------------------------
// Decider
// --------------------------------------------------------------------------

Decider::Decider(const Options &o)
    : o_(o),
      model_(o.container),
      lease_(model_),
      stack_(dev_, model_, o.artifacts_dir, [&] {
        enc::StackOptions s;
        s.threads = o.threads;
        s.lanes = o.lanes;
        return s;
      }()),
      svc_(std::make_unique<enc::EmbedService>(enc::EmbedService{
          enc::load_tokenizer(model_, o.container),
          model_.raw("embeddings.word").as<float>(),
          model_.raw("embeddings.position").as<float>(),
          model_.raw("embeddings.token_type").as<float>(),
          stack_.lead.get(), {}, stack_.batch})) {
  if (model_.config_string("arch") != "modernbert_rope_geglu")
    throw std::runtime_error(
        "this decision engine is arch=4 only; the container says '" +
        model_.config_string("arch") +
        "'. The head reads type_emb/scorer/act_head/head.layers.*, none of "
        "which exist in another arch's container, and a refusal here is one "
        "line rather than a scatter of missing-tensor errors.");
  head_ = std::make_unique<Head>(model_);
  head_max_len_ = model_.config_int("head_max_len");
  max_seq_len_ = model_.config_int("max_seq_len");
  if (head_max_len_ <= 0 || head_max_len_ > max_seq_len_)
    throw std::runtime_error(
        "head_max_len " + std::to_string(head_max_len_) + " against max_seq_len " +
        std::to_string(max_seq_len_) +
        ". The option budget is subtracted from the sequence, so a head budget "
        "larger than the sequence would make every option a marker at the cut "
        "and the row's k would be 1 for any real question.");
}

Decider::~Decider() = default;

int64_t Decider::max_seq() const { return enc::g_max_positions; }
int64_t Decider::hidden() const { return enc::g_hidden; }

std::vector<std::string> Decider::render_options(
    int64_t qtype, const std::vector<std::string> &labels,
    const std::vector<std::string> &criteria,
    const std::vector<std::string> &noul_labels) {
  // A criterion value that is FALSY BUT MEANINGFUL -- "0", "false", "[]" --
  // renders. Only null/absent and the empty string fall back, because
  // `if (v)` would silently drop a rubric level of 0 into the "no description"
  // form, which is a different option text and a different token count.
  auto crit = [&](size_t i) -> std::string {
    return i < criteria.size() ? criteria[i] : std::string();
  };
  if (qtype == QTYPE_CHOICE) {
    if (!noul_labels.empty())
      throw std::runtime_error(
          "noul labels were supplied for a choice question. The pinned schema "
          "has no `labels` field for a choice and upstream raises ValueError "
          "for it, so this port refuses it rather than ignoring a field the "
          "caller believed was in effect.");
    std::vector<std::string> out;
    for (size_t i = 0; i < labels.size(); ++i) {
      const std::string c = crit(i);
      out.push_back(c.empty() ? labels[i] : labels[i] + ": " + c);
    }
    return out;
  }
  if (qtype == QTYPE_SCORE) {
    // THE CRITERIA, not the labels. A score's level text IS its criterion and
    // the label is the index: upstream enumerates `crit`, so a caller that
    // passed the levels as labels and left the criteria empty would get N
    // options reading "level 0: " with nothing after the colon.
    std::vector<std::string> out;
    for (size_t i = 0; i < criteria.size(); ++i)
      out.push_back("level " + std::to_string(i) + ": " + criteria[i]);
    if (!labels.empty() && labels.size() != criteria.size())
      throw std::runtime_error(
          "a score question has " + std::to_string(labels.size()) +
          " labels and " + std::to_string(criteria.size()) +
          " criteria. The LEVEL TEXT is the criterion and the label is the "
          "index, so a mismatch is a caller that passed its levels in the "
          "wrong field -- and which field it passed them in decides whether "
          "the answer is right or merely well-formed.");
    return out;
  }
  if (qtype != QTYPE_NOUL)
    throw std::runtime_error("qtype " + std::to_string(qtype) +
                             " is outside the QTYPES order; render_options "
                             "has no form for it");
  // noul: ALWAYS [false, true], whatever the caller's order. The semantic order
  // is the model's, and a swapped pair flips every noul answer while looking
  // like a calibration change.
  std::string fl = "false", tl = "true";
  if (noul_labels.size() == 2) {
    fl = noul_labels[0];
    tl = noul_labels[1];
  } else if (!noul_labels.empty()) {
    throw std::runtime_error(
        "noul labels must be exactly two, [false, true]; got " +
        std::to_string(noul_labels.size()) +
        ". One label cannot be paired with a criterion and two is the number "
        "the semantic order has.");
  }
  if (fl == tl)
    throw std::runtime_error(
        "noul labels are both '" + fl +
        "'. They must be DISTINCT: the option text is built as label + "
        "criterion, so equal labels make the two options the same text, the two "
        "markers the same token span, and the row unanswerable in a way no "
        "downstream check can see.");
  const std::string fc = crit(0), tc = crit(1);
  return {fl + ": " + (fc.empty() ? "no, the statement does not hold" : fc),
          tl + ": " + (tc.empty() ? "yes, the statement holds" : tc)};
}

Decider::Prompt Decider::build_prompt(const std::string &type_text,
                                      const std::string &instruction,
                                      const std::string &state,
                                      const std::vector<std::string> &options,
                                      int64_t qtype) const {
  // A port of upstream's build_sequence (laya/common.py:135-228), and the
  // constants are the CONTAINER's, not this file's: max_len is the design's
  // sequence length, and a prompt built against one length and encoded against
  // another is a truncation nobody reported.
  const int64_t max_len = std::min<int64_t>(enc::g_max_positions, model_.config_int("max_seq_len"));
  const int64_t head_max = model_.config_int("head_max_len");
  const auto &tok = svc_->tok;
  if (!tok.bbpe)
    throw std::runtime_error(
        "no BBPETOK blob on this container. arch=4's prompt is assembled token "
        "by token with its specials in hand, so there is no WordPiece or Unigram "
        "path to fall back to -- and the fallback would be wrong, not just "
        "different.");
  const int64_t cls = tok.bbpe->cls_id, sep = tok.bbpe->sep_id,
                mask = tok.bbpe->mask_id;

  // `tokenize()`, NOT `encode()`. tokenize() is the add_special_tokens=False
  // path -- the added-token split, normalisation, pre-tokenisation and BPE --
  // and it does NOT truncate, which is what this function needs: upstream caps
  // an option at 48 TOKENS with `truncation=True, max_length=48`, and
  // `encode(text, 48)` would instead add the post-processor's wrapping before
  // truncating, spending the budget on a prefix and a suffix.
  //
  // That is also what `AnyTokenizer::encode()`'s refusal on this arch is
  // pointing at when it says "Call tokenize() per segment".
  auto ids_of = [&](const std::string &t) { return tok.bbpe->tokenize(t); };
  auto cap = [](std::vector<int32_t> v, int64_t n) {
    if (n > 0 && static_cast<int64_t>(v.size()) > n) v.resize(static_cast<size_t>(n));
    return v;
  };

  // `add_special_tokens=False` upstream, because Laya inserts every special by
  // hand in a specific order. The C++ tokenizer's encode() adds the blob's
  // post-processor's prefix/suffix, which for this checkpoint is nothing -- so
  // the two agree, and Phase 2's 418-string parity is what established that.
  // Stated because it is the one place where "looks equivalent" was verified
  // rather than assumed.
  // The mask token's SURFACE FORM, from the table -- upstream scrubs the literal
  // "[MASK]" out of caller text before tokenizing it, so a state or an option
  // that happens to contain the marker must not inject a real one and shift
  // every marker index after it. Deriving it from the id rather than writing
  // the literal is what makes that true for a checkpoint whose marker surface
  // form is not "[MASK]".
  const std::string mask_str = tok.bbpe->token_of(mask);
  auto scrub = [&](std::string t) {
    size_t at = 0;
    while ((at = t.find(mask_str, at)) != std::string::npos) {
      t.replace(at, mask_str.size(), " ");
      at += 1;
    }
    return t;
  };

  std::vector<int32_t> head_ids = cap(
      ids_of(type_text + " question: " + scrub(instruction)), head_max);
  std::vector<std::vector<int32_t>> opt_ids;
  for (const auto &o : options) {
    auto v = cap(ids_of(" " + scrub(o)), 48);
    v.insert(v.begin(), static_cast<int32_t>(mask));
    opt_ids.push_back(std::move(v));
  }
  int64_t used = 0;
  for (const auto &o : opt_ids) used += static_cast<int64_t>(o.size());
  int64_t opt_budget = head_max - used;
  if (opt_budget < 16) {
    // The per-option cap. Upstream's floor of 16 and its max(4, ...) are both
    // reproduced: the floor is what stops a long instruction from eating the
    // room the options need, and the cap is what stops one long option from
    // doing the same to the others.
    const int64_t per =
        std::max<int64_t>(4, (head_max - 16) /
                                 std::max<int64_t>(1, static_cast<int64_t>(opt_ids.size())));
    for (auto &o : opt_ids) {
      if (static_cast<int64_t>(o.size()) > per) o.resize(static_cast<size_t>(per));
    }
    used = 0;
    for (const auto &o : opt_ids) used += static_cast<int64_t>(o.size());
    opt_budget = head_max - used;
  }
  if (static_cast<int64_t>(head_ids.size()) > std::max<int64_t>(8, opt_budget))
    head_ids.resize(static_cast<size_t>(std::max<int64_t>(8, opt_budget)));

  Prompt out;
  out.ids.push_back(static_cast<int32_t>(cls));
  out.ids.insert(out.ids.end(), head_ids.begin(), head_ids.end());
  out.ids.push_back(static_cast<int32_t>(sep));
  for (const auto &o : opt_ids) {
    // The marker is the position of the [MASK] ITSELF, recorded BEFORE the
    // option's tokens are appended. Recording it after would point the gather
    // at the first option token instead, which is a plausible model and the
    // wrong one.
    out.markers.push_back(static_cast<int64_t>(out.ids.size()));
    out.ids.insert(out.ids.end(), o.begin(), o.end());
  }
  out.ids.push_back(static_cast<int32_t>(sep));

  // The state is clamped to the room LEFT, and upstream's `-room:` is not
  // `room:` -- with no room left, `ids[-0:]` is the whole state rather than none
  // of it, which is a 1024-token document in a 256-token budget.
  const int64_t room = std::max<int64_t>(0, max_len - static_cast<int64_t>(out.ids.size()) - 1);
  auto st = ids_of(scrub(state));
  out.state_tokens = static_cast<int64_t>(st.size());
  if (static_cast<int64_t>(st.size()) > room) st.resize(static_cast<size_t>(room));
  out.state_used = static_cast<int64_t>(st.size());
  out.ids.insert(out.ids.end(), st.begin(), st.end());
  out.ids.push_back(static_cast<int32_t>(sep));
  if (static_cast<int64_t>(out.ids.size()) > max_len)
    out.ids.resize(static_cast<size_t>(max_len));
  // And a marker past the cut is DROPPED, not clamped: the option it named is
  // no longer in the sequence, so the row has fewer options than the caller
  // asked about. Reporting k as the option count would return a probability for
  // an option the model never saw.
  out.markers.erase(std::remove_if(out.markers.begin(), out.markers.end(),
                                   [&](int64_t m) { return m >= max_len; }),
                    out.markers.end());
  (void)qtype;
  return out;
}

std::vector<RowAnswer> Decider::decide(const std::vector<PromptRow> &rows_in,
                                       const std::vector<float> &temperature) {
  if (rows_in.empty()) return {};
  std::lock_guard<std::mutex> lk(call_mu_);
  enc::Encoder &e = *stack_.lead;
  const int64_t S = enc::g_max_positions, H = enc::g_hidden;
  const int64_t d = head_->hidden();

  // kmax is the batch's widest row, and it is needed BEFORE the encode because
  // the readout's slot layout depends on it.
  int64_t kmax = 0;
  for (const auto &r : rows_in)
    kmax = std::max<int64_t>(kmax, static_cast<int64_t>(r.markers.size()));
  if (kmax <= 0)
    throw std::runtime_error(
        "every row has zero markers. Upstream's collate_items refuses the same "
        "case, because a row with no option cannot be answered: the softmax "
        "over an empty axis is NaN and a NaN in the response is worse than a "
        "refusal.");

  std::vector<RowAnswer> out(rows_in.size());
  // `plan()` rather than use_tier(n) directly: use_tier pads to the tier, so a
  // naive use_tier(5) is a whole 16-row pass for five rows. The greedy ladder
  // turns 5 into 4 + 1.
  const auto jobs = svc_->plan(static_cast<int64_t>(rows_in.size()));
  const int64_t pad_id = svc_->tok.bbpe ? svc_->tok.bbpe->pad_id : 0;

  for (const auto &job : jobs) {
    const int64_t base = job.first, take = job.second;
    const int64_t bt = e.use_tier(take);
    std::vector<float> buf(static_cast<size_t>(bt) * S * H, 0.f);
    std::vector<float> cmask(static_cast<size_t>(bt) * S, -1.0e30f);
    for (int64_t b = 0; b < bt; ++b) {
      const bool real = b < take;
      const PromptRow *row = real ? &rows_in[static_cast<size_t>(base + b)] : nullptr;
      const size_t n = row ? row->ids.size() : 0;
      for (int64_t s = 0; s < S; ++s) {
        const bool keep = real && s < static_cast<int64_t>(n);
        const int32_t id = keep ? row->ids[static_cast<size_t>(s)] : pad_id;
        cmask[static_cast<size_t>(b) * S + s] = keep ? 0.f : -1.0e30f;
        float *dst = buf.data() + (b * S + s) * H;
        const float *wv = svc_->w_word + static_cast<size_t>(id) * H;
        const float *pv = svc_->w_pos + static_cast<size_t>(s) * H;
        for (int64_t c = 0; c < H; ++c) dst[c] = wv[c] + pv[c] + svc_->w_typ[c];
      }
    }
    e.add_mask = cmask;
    // The encoder's OUTPUT -- post-final_norm -- is the head's input. Upstream
    // reads last_hidden_state, so handing the head the residual stream from
    // inside the loop is a different model.
    std::vector<float> h = e.run_dispatch(buf);

    // type_emb, broadcast over the sequence, BEFORE the head and before the
    // gather: the markers have to know their question's type, and a gather
    // that happens first cannot see it.
    for (int64_t b = 0; b < bt; ++b) {
      const int64_t qt = (b < take) ? rows_in[static_cast<size_t>(base + b)].qtype
                                    : QTYPE_CHOICE;
      if (qt < 0 || qt > 2)
        throw std::runtime_error("row " + std::to_string(base + b) +
                                 " has qtype " + std::to_string(qt) +
                                 ", outside the QTYPES order [choice=0, "
                                 "score=1, noul=2]. type_emb has three rows and "
                                 "is indexed by that order, so this would embed "
                                 "the wrong question type rather than fail.");
      const float *te = model_.raw("type_emb.weight").as<float>() + qt * d;
      for (int64_t s = 0; s < S; ++s) {
        float *dst = h.data() + (b * S + s) * d;
        for (int64_t c = 0; c < d; ++c) dst[c] += te[c];
      }
    }

    std::vector<float> pad(static_cast<size_t>(bt) * S);
    for (int64_t b = 0; b < bt; ++b)
      for (int64_t s = 0; s < S; ++s)
        pad[static_cast<size_t>(b) * S + s] = cmask[static_cast<size_t>(b) * S + s] == 0.f ? 1.f : 0.f;
    head_->forward(h, pad, bt, S, o_threads());

    // --- the gather, then the readout
    std::vector<float> gathered(static_cast<size_t>(bt) * kmax * d, 0.f);
    for (int64_t b = 0; b < bt; ++b) {
      for (int64_t j = 0; j < kmax; ++j) {
        int64_t at = 0;
        if (b < take && j < static_cast<int64_t>(rows_in[static_cast<size_t>(base + b)].markers.size()))
          at = rows_in[static_cast<size_t>(base + b)].markers[static_cast<size_t>(j)];
        float *dst = gathered.data() + (b * kmax + j) * d;
        if (at > 0 && at < S)
          std::memcpy(dst, h.data() + (b * S + at) * d,
                      static_cast<size_t>(d) * sizeof(float));
        // A slot with no marker reads row 0. It is masked to -1e4 below, so it
        // never reaches the returned probabilities -- the length assertion is
        // what keeps that honest.
      }
    }
    std::vector<float> logits;
    head_->score(gathered, bt, kmax, d, o_threads(), logits);

    // act_head, computed and discarded. `ent` and the top-2 gap come from the
    // softmax of the RAW logits -- upstream computes them before the caller
    // applies a temperature, and the act head was fitted on that.
    std::vector<float> pooled(static_cast<size_t>(bt) * d);
    for (int64_t b = 0; b < bt; ++b)
      std::memcpy(pooled.data() + b * d, h.data() + b * S * d,
                  static_cast<size_t>(d) * sizeof(float));
    std::vector<float> feats(static_cast<size_t>(bt) * 4, 0.f);
    for (int64_t b = 0; b < bt; ++b) {
      const int64_t k = (b < take) ? static_cast<int64_t>(rows_in[static_cast<size_t>(base + b)].markers.size())
                                   : kmax;
      std::vector<float> p(static_cast<size_t>(k));
      float mx = -INFINITY;
      for (int64_t j = 0; j < k; ++j) mx = std::max(mx, logits[static_cast<size_t>(b) * kmax + j]);
      double sum = 0.0;
      for (int64_t j = 0; j < k; ++j) {
        p[static_cast<size_t>(j)] = std::exp(logits[static_cast<size_t>(b) * kmax + j] - mx);
        sum += p[static_cast<size_t>(j)];
      }
      for (int64_t j = 0; j < k; ++j) p[static_cast<size_t>(j)] /= static_cast<float>(sum);
      float top1 = 0.f, top2 = 0.f;
      for (int64_t j = 0; j < k; ++j) {
        if (p[static_cast<size_t>(j)] > top1) {
          top2 = top1;
          top1 = p[static_cast<size_t>(j)];
        } else if (p[static_cast<size_t>(j)] > top2) {
          top2 = p[static_cast<size_t>(j)];
        }
      }
      if (k < 2) top2 = 0.f;   // a one-option row: softmax over one logit is 1.0
      double ent = 0.0;
      for (int64_t j = 0; j < k; ++j)
        ent -= static_cast<double>(p[static_cast<size_t>(j)]) *
               std::log(std::max(1e-9f, p[static_cast<size_t>(j)]));
      const double kk = std::max(2.0, static_cast<double>(k));
      feats[static_cast<size_t>(b) * 4 + 0] = top1;
      feats[static_cast<size_t>(b) * 4 + 1] = top1 - top2;
      feats[static_cast<size_t>(b) * 4 + 2] = static_cast<float>(ent / std::log(kk));
      feats[static_cast<size_t>(b) * 4 + 3] = static_cast<float>(static_cast<double>(k) / 255.0);
    }
    std::vector<float> act;
    head_->act(pooled, feats, bt, d, o_threads(), act);

    // --- the readout, per row, with THIS row's k
    for (int64_t b = 0; b < take; ++b) {
      const size_t ri = static_cast<size_t>(base + b);
      const int64_t k = static_cast<int64_t>(rows_in[ri].markers.size());
      RowAnswer &A = out[ri];
      A.k = k;
      A.logits.assign(logits.begin() + static_cast<long>(b) * kmax,
                      logits.begin() + static_cast<long>(b) * kmax + k);
      // masked_fill(~marker_mask, -1e4), then the softmax over kmax -- not over
      // k. So a row with fewer options carries padded slots.
      //
      // AND THE PADDED SLOTS CONTRIBUTE NOTHING, which is worth stating because
      // the opposite is the natural assumption and it was the first version of
      // this comment. exp(-1e4 - max) underflows float32 to exactly 0 -- not
      // "very small": the smallest normal float32 is 1.2e-38 and exp(-1e4) is
      // far below it. So normalising over kmax and over k agree to float32
      // precision (measured 3e-8 on a 2-of-5 row), and the kmax layout is
      // about WHERE the mask is applied, not about a different normaliser.
      //
      // The fill is -1e4 and not -1e30 precisely because it has to survive a
      // float32 exp: a -1e30 would overflow the subtraction to -inf, which is
      // also 0 after exp, but -1e30 - (-1e30) is a NaN waiting for a row where
      // every slot is padded. -1e4 is upstream's value and it is the right one.
      std::vector<float> z(static_cast<size_t>(kmax), -1.0e4f);
      for (int64_t j = 0; j < k; ++j) z[static_cast<size_t>(j)] = A.logits[static_cast<size_t>(j)];
      const float t = temperature.empty() ? 1.0f
                                           : temperature[static_cast<size_t>(rows_in[ri].qtype)];
      if (!(t > 0.f))
        throw std::runtime_error(
            "temperature[" + std::to_string(rows_in[ri].qtype) + "] is " +
            std::to_string(t) + ". Upstream clamps temperatures to a positive "
            "range; zero would divide by zero and a negative one would flip the "
            "argmax, so neither is a caller error to pass on silently.");
      float mx = -INFINITY;
      for (int64_t j = 0; j < kmax; ++j) mx = std::max(mx, z[static_cast<size_t>(j)] / t);
      double sum = 0.0;
      std::vector<float> p(static_cast<size_t>(kmax));
      for (int64_t j = 0; j < kmax; ++j) {
        p[static_cast<size_t>(j)] = std::exp(z[static_cast<size_t>(j)] / t - mx);
        sum += p[static_cast<size_t>(j)];
      }
      for (int64_t j = 0; j < kmax; ++j) p[static_cast<size_t>(j)] /= static_cast<float>(sum);
      // Returned length is k -- THIS row's own marker count. The oflm-test
      // "probabilities sum to 1" check cannot catch returning kmax.
      // Upstream unpermutes IMMEDIATELY after the softmax and before anything
      // that indexes by option (`agent.py:1039`), so argmax, score and every
      // probability here are in the CALLER's order. `p` is slot-ordered, and
      // `canonical[option_order[s]] = p[s]` is upstream's
      // `unpermute_probs` -- slot s held option option_order[s], so the inverse
      // is a scatter, not a gather.
      //
      // Skipping this is silent: the probabilities still sum to 1, the
      // confidence is still plausible, and the labels are the caller's, so
      // nothing looks broken. Every probability is just attached to the wrong
      // option, which is exactly the failure upstream's docstring warns about.
      std::vector<float> q(static_cast<size_t>(k));
      const std::vector<int> &ord = rows_in[ri].option_order;
      if (ord.empty()) {
        q.assign(p.begin(), p.begin() + k);
      } else {
        // Validated at the edge, and re-checked here because the engine is
        // reachable without the wire parser (the rigs, and any future caller).
        if (static_cast<int64_t>(ord.size()) != k)
          throw std::runtime_error(
              "option_order has " + std::to_string(ord.size()) + " entries for " +
              std::to_string(k) + " options on a row the head actually reached. "
              "The order is validated against the REQUEST's option count; if "
              "these differ, the head truncated an option and the permutation "
              "no longer describes the slots. Refusing rather than answering "
              "with probabilities attached to the wrong options.");
        for (int64_t s = 0; s < k; ++s) {
          const int dst = ord[static_cast<size_t>(s)];
          if (dst < 0 || dst >= k)
            throw std::runtime_error("option_order entry " + std::to_string(dst) +
                                     " is outside [0, " + std::to_string(k) + ")");
          q[static_cast<size_t>(dst)] = p[static_cast<size_t>(s)];
        }
      }
      A.probabilities = std::move(q);
      A.argmax = 0;
      for (int64_t j = 1; j < k; ++j)
        if (A.probabilities[static_cast<size_t>(j)] >
            A.probabilities[static_cast<size_t>(A.argmax)])
          A.argmax = j;
      A.confidence = A.probabilities[static_cast<size_t>(A.argmax)];
      double sc = 0.0;
      for (int64_t j = 0; j < k; ++j)
        sc += static_cast<double>(j) * A.probabilities[static_cast<size_t>(j)];
      A.score = static_cast<float>(sc);
      // act_head's output, sliced to this row. The row stride is act_out, not
      // n_act, because the two differ the moment a checkpoint changes it --
      // and slicing by n_act would read into the NEXT row's activations.
      const int64_t na = head_->act_out();
      A.act.assign(act.begin() + static_cast<long>(b) * na,
                   act.begin() + static_cast<long>(b) * na + na);
    }
  }
  return out;
}



}  // namespace dec
}  // namespace npue
