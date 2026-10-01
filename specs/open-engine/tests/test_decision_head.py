# Traces: OPEN-DECISION-READOUT, OPEN-DECISION-HEAD
# (specs/open-engine/spec.md); Phase 7 of specs/open-engine/plans/laya-decision-encoder.md
"""The decision head's arithmetic, against a second implementation of it.

`utilities/laya_preln_reference.py`'s `head()` is a float64 numpy port of the same
model read from the same checkpoint, so the two are independent readings rather
than two runs of the same code. Here it is checked on synthetic weights small
enough to reason about by hand, which is the only way to catch the class of bug
that this phase actually produced: a matrix indexed the wrong way round.

That class is worth a test of its own. `gemm_nk` reads its weight as [N, K]
because that is what the container stores and what PyTorch's `Linear.weight` is.
An earlier version indexed it as `w[k * N + n]` -- the transpose -- and the
symptom was not a crash: the engine answered every question, with a confidence
near 0.55 on all of them. A wrong orientation is a plausible model.

The measured gate against the real checkpoint is in the Phase 7 commit: on three
probes the engine and the float64 oracle agree on the ARGMAX for every row,
which is the plan's gate, and the logits differ by ~0.2, which the plan calls a
diagnostic. The accuracy fixture -- >= 50 pairs, stratified by the reference's
top-2 gap -- is Phase 9's and does not exist yet.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "utilities"))
REPO = Path(__file__).resolve().parents[3]
ENC = REPO / "src/open_npue/decision_engine.hpp"
CPP = REPO / "src/open_npue/decision_engine.cpp"

# The head's arithmetic with the container substituted, so the maths can be
# checked without an NPU, a design set or a 1.4 GB checkpoint. `render_options`
# and the readout's masking/temperature are pure host functions and are used
# verbatim; only the weights change.
HOST_DRIVER = r"""
#include <cstdio>
#include <cmath>
#include <vector>
#include "decision_engine.hpp"

// The readout's per-row arithmetic, verbatim from Decider::decide: the -1e4
// fill for the padded marker slots, the softmax over kmax (NOT over k), the
// temperature, and the returned length being k.
std::vector<float> readout(const std::vector<float> &logits, int64_t k,
                           int64_t kmax, float t, float *conf, float *score,
                           long *argmax) {
  std::vector<float> z((size_t)kmax, -1.0e4f);
  for (int64_t j = 0; j < k; ++j) z[(size_t)j] = logits[(size_t)j];
  float mx = -INFINITY;
  for (int64_t j = 0; j < kmax; ++j) mx = std::max(mx, z[(size_t)j] / t);
  double sum = 0.0;
  std::vector<float> p((size_t)kmax);
  for (int64_t j = 0; j < kmax; ++j) {
    p[(size_t)j] = std::exp(z[(size_t)j] / t - mx);
    sum += p[(size_t)j];
  }
  for (int64_t j = 0; j < kmax; ++j) p[(size_t)j] /= (float)sum;
  *argmax = 0;
  for (int64_t j = 1; j < k; ++j)
    if (p[(size_t)j] > p[(size_t)*argmax]) *argmax = j;
  *conf = p[(size_t)*argmax];
  double s = 0.0;
  for (int64_t j = 0; j < k; ++j) s += (double)j * p[(size_t)j];
  *score = (float)s;
  return std::vector<float>(p.begin(), p.begin() + k);
}

int main(int argc, char **argv) {
  if (argc < 2) return 2;
  const std::string mode = argv[1];
  try {
    if (mode == "render") {
      // argv[2]=qtype, argv[3]=labels(csv), argv[4]=criteria(csv),
      // argv[5]=noul_labels(csv, may be empty)
      auto split = [](const std::string &s) {
        std::vector<std::string> v;
        size_t at = 0;
        while (at <= s.size()) {
          size_t c = s.find(',', at);
          if (c == std::string::npos) { if (at < s.size()) v.push_back(s.substr(at)); break; }
          v.push_back(s.substr(at, c - at));
          at = c + 1;
        }
        return v;
      };
      auto out = npue::dec::Decider::render_options(
          atoll(argv[2]), split(argv[3]), split(argv[4]), split(argv[5]));
      for (const auto &o : out) std::printf("OPT %s\n", o.c_str());
      return 0;
    }
    if (mode == "buckets") {
      // `temp_bucket` is a pure function and this checkpoint's
      // temperature_by_options is `{}`, so nothing else in the tree exercises it.
      // A bucket key that is spelled differently from upstream's is a silently
      // missing calibration rather than an error, which is the worst kind.
      for (int64_t qt = 0; qt < 3; ++qt)
        for (int64_t k : {1, 2, 3, 5, 6, 10, 11, 40})
          std::printf("BUCKET %lld %lld %s\n", (long long)qt, (long long)k,
                      npue::dec::temp_bucket(qt, k).c_str());
      return 0;
    }
    if (mode == "repro") {
      // The head is a PURE FUNCTION of its input. This was not: the attention
      // accumulator was zeroed with `t < d` where it must be `t < head_dim_`,
      // so each head erased the other eleven. With one worker the last head
      // always won -- wrong but STABLE; with twelve it was wrong and UNSTABLE,
      // and two identical `oflm decide` calls disagreed on a score by 1.5e-02.
      //
      // No cosine and no argmax gate can see this: every value was a plausible
      // float and the answers were merely wrong. Only equality can.
      npue::File mf(argv[2]);
      // The geometry globals come from the lease; without it hidden/heads are 0
      // and every shape check in Head's constructor fires at once.
      npue::enc::ShapeLease lease(mf);
      npue::dec::Head h(mf);
      const int64_t R = 4, S = 256, D = h.hidden();
      std::vector<float> in((size_t)(R * S * D));
      std::vector<float> pad((size_t)(R * S));
      unsigned sd = 12345u;
      for (size_t i = 0; i < in.size(); ++i) {
        sd = sd * 1664525u + 1013904223u;
        in[i] = (float)((sd >> 8) & 0xffff) / 32768.0f - 1.0f;
      }
      for (int64_t i = 0; i < R * S; ++i)
        pad[(size_t)i] = (i % S) < 200 ? 1.0f : 0.0f;
      auto run = [&](int workers) {
        auto a = in;
        h.forward(a, pad, R, S, workers);
        return a;
      };
      const auto w1 = run(1), w1b = run(1);
      const auto w12 = run(12), w12b = run(12);
      double worst = 0;
      for (size_t i = 0; i < w1.size(); ++i) {
        const double dd = std::fabs((double)w1[i] - (double)w12[i]);
        if (dd > worst) worst = dd;
      }
      std::printf("repeat_w1 %d cross %d repeat_w12 %d maxdiff %g\n",
                  int(w1 == w1b), int(w1 == w12), int(w12 == w12b), worst);
      return 0;
    }
    if (mode == "readout") {
      // stdin: k, kmax, t, then k logits
      long k, kmax; double t;
      if (scanf("%ld %ld %lf", &k, &kmax, &t) != 3) return 3;
      std::vector<float> l((size_t)k);
      for (long i = 0; i < k; ++i) { double v; if (scanf("%lf", &v) != 1) return 3; l[(size_t)i] = (float)v; }
      float c, s; long a;
      auto p = readout(l, k, kmax, (float)t, &c, &s, &a);
      std::printf("ARGMAX %ld CONF %.9g SCORE %.9g\n", a, c, s);
      for (float v : p) std::printf("P %.9g\n", v);
      return 0;
    }
    std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 2;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "REFUSED: %s\n", e.what());
    return 1;
  }
}
"""


@pytest.fixture(scope="module")
def host(tmp_path_factory):
    if shutil_which("g++") is None:                # pragma: no cover
        pytest.skip("no C++ compiler on this host")
    d = tmp_path_factory.mktemp("dechead")
    src = d / "host.cpp"
    src.write_text(HOST_DRIVER)
    exe = d / "dechead"
    # -mavx2 -mfma because npue_encoder.hpp's qk_impl/av_impl do not compile
    # without them: their #if defined(__AVX2__) arms leave the __m256
    # declarations unguarded, which is pre-existing and not this test's to fix.
    cmd = ([shutil_which("g++"), "-std=c++17", "-O1", "-mavx2", "-mfma",
            "-o", str(exe), str(src),
            str(CPP), str(REPO / "src/open_npue/npue_encoder.cpp"),
            str(REPO / "src/open_npue/npue.cpp"),
            str(REPO / "src/open_npue/npu_device.cpp"),
            str(REPO / "src/open_npue/npue_pack.cpp"),
            str(REPO / "src/open_npue/json_min.cpp"),
            str(REPO / "src/open_npue/xlmr_tokenizer_gen.cpp"),
            str(REPO / "src/open_npue/gemma_tokenizer_gen.cpp"),
            str(REPO / "src/open_npue/bbpe_tokenizer_gen.cpp"),
            str(REPO / "src/open_npue/tokenizer_bbpe.cpp"),
            str(REPO / "src/open_npue/tokenizer_xlmr.cpp"),
            str(REPO / "src/open_npue/tokenizer_gemma.cpp"),
            str(REPO / "src/open_npue/tokenizer.cpp"),
            str(REPO / "src/open_npue/gemma_kernels.cpp"),
            str(REPO / "src/open_npue/gemma_encode.cpp"),
            "-I", str(REPO / "src/open_npue"),
            "-I", "/opt/xilinx/xrt/include",
            "-L", "/opt/xilinx/xrt/lib64", "-lxrt_coreutil",
            "-Wl,-rpath,/opt/xilinx/xrt/lib64"])
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-5000:]
    return exe


def shutil_which(name):
    import shutil
    return shutil.which(name)


def _run(exe, *args, stdin=None):
    p = subprocess.run([str(exe), *[str(a) for a in args]],
                       capture_output=True, text=True, input=stdin)
    return p


def _opts(p):
    return [l[4:] for l in p.stdout.splitlines() if l.startswith("OPT ")]


# --------------------------------------------------------------------------
# render_options
# --------------------------------------------------------------------------

def test_a_SCORE_renders_its_INDEX_and_the_index_is_the_scale(host):
    """`"level %d: %s"`. The index is not decoration: the readout returns
    `sum(i * p_i)`, so the model's i-th slot has to BE level i. Rendering the
    levels bare would leave the scorer choosing on text alone and every score
    question would be a choice question with extra steps."""
    p = _run(host, "render", 1, "", "minimal,moderate,severe", "")
    assert p.returncode == 0, p.stderr
    assert _opts(p) == ["level 0: minimal", "level 1: moderate", "level 2: severe"]


def test_a_NOUL_renders_TWO_options_in_the_semantic_order(host):
    """[false, true] always. The pair is the model's, not the caller's: a
    swapped noul flips every noul answer and looks like a calibration change."""
    p = _run(host, "render", 2, "false,true",
             "This is not it.,This is it.", "")
    assert p.returncode == 0, p.stderr
    assert _opts(p) == ["false: This is not it.", "true: This is it."]


def test_a_NOUL_with_no_criteria_uses_the_fallback_sentences(host):
    """Upstream's "no, the statement does not hold" / "yes, the statement
    holds". They are not filler: they are what the model reads when the caller
    gives a bare yes/no question, and substituting empty strings puts a bare
    `[MASK]` option in front of it."""
    p = _run(host, "render", 2, "false,true", ",", "")
    assert p.returncode == 0, p.stderr
    assert _opts(p) == ["false: no, the statement does not hold",
                        "true: yes, the statement holds"]


def test_a_FALSY_criterion_is_rendered_not_replaced(host):
    """A criterion of `0` is a rubric level, not an absent one. `if (v)` would
    drop it into the bare-label form, which is a different option text AND a
    different token count -- so the marker positions move and nothing reports
    it."""
    src = (REPO / "utilities/laya_preln_reference.py")
    assert src.exists()
    p = _run(host, "render", 0, "none,one", "0,", "")
    assert p.returncode == 0, p.stderr
    assert _opts(p) == ["none: 0", "one"], _opts(p)


def test_a_CHOICE_with_no_criterion_is_the_BARE_label(host):
    p = _run(host, "render", 0, "alpha,beta", ",", "")
    assert p.returncode == 0, p.stderr
    assert _opts(p) == ["alpha", "beta"]


def test_noul_labels_on_a_choice_are_REFUSED(host):
    """The pinned schema has no `labels` for a choice and upstream raises
    ValueError. Ignoring it would render a prompt the caller did not ask for."""
    p = _run(host, "render", 0, "alpha", "", "no,yes")
    assert p.returncode == 1
    assert "noul labels" in p.stderr, p.stderr


def test_two_IDENTICAL_noul_labels_are_REFUSED(host):
    """The option text is label + criterion, so equal labels make the two
    options the same text, the two markers the same token span, and the row
    unanswerable in a way nothing downstream can see."""
    p = _run(host, "render", 2, "yes,yes", "a,b", "yes,yes")
    assert p.returncode == 1
    assert "DISTINCT" in p.stderr, p.stderr


def test_a_noul_needs_EXACTLY_two_labels(host):
    p = _run(host, "render", 2, "no", "a,b", "no")
    assert p.returncode == 1
    assert "exactly two" in p.stderr, p.stderr


# --------------------------------------------------------------------------
# the readout's arithmetic
# --------------------------------------------------------------------------

def _readout(exe, k, kmax, t, logits):
    p = _run(exe, "readout", stdin=f"{k} {kmax} {t}\n" +
             "".join(f"{v}\n" for v in logits))
    assert p.returncode == 0, p.stderr
    head = p.stdout.splitlines()[0].split()
    probs = [float(l[2:]) for l in p.stdout.splitlines()[1:]]
    return int(head[1]), float(head[3]), float(head[5]), probs


def test_the_returned_length_is_THIS_ROWS_k_and_not_kmax(host):
    exe = host
    """The check that cannot catch it. `oflm-test`'s "probabilities sum to 1"
    passes for kmax entries too, since they are normalised over kmax; the length
    assertion is the only thing standing between a caller and a probability for
    an option that does not exist."""
    _, _, _, probs = _readout(exe, k=2, kmax=5, t=1.0, logits=[1.0, 0.0])
    assert len(probs) == 2
    _, _, _, probs = _readout(exe, k=5, kmax=5, t=1.0, logits=[1.0, 0, 0, 0, 0])
    assert len(probs) == 5


def test_the_padded_slots_are_filled_and_they_CONTRIBUTE_NOTHING(host):
    """-1e4 on the padded slots, softmax over kmax -- and exp(-1e4 - max)
    UNDERFLOWS float32 to exactly zero, so the padded slots add nothing to the
    normaliser and the kmax result equals the k result to float32 precision.

    That is worth a test because the opposite is the natural assumption: it
    looks as though a mixed-width batch must dilute the reported probabilities,
    and an implementation that "corrected" for it by normalising over k would be
    changing a number that was already right. Measured 3e-8, so the assertion
    is float32 epsilon rather than a visible difference.

    The fill being -1e4 and not -1e30 is the same point from the other side: a
    fill large enough to survive a subtraction must not be large enough to make
    `-1e30 - (-1e30)` a NaN."""
    exe = host
    _, conf, _, probs = _readout(exe, k=2, kmax=5, t=1.0, logits=[1.0, 0.0])
    _, _, _, p2 = _readout(exe, k=2, kmax=2, t=1.0, logits=[1.0, 0.0])
    assert abs(sum(probs) - 1.0) < 1e-6, (
        "the padded slots underflow, so the real probabilities must sum to 1 "
        f"to float32 precision; got {sum(probs)}")
    for a, b in zip(probs, p2):
        assert abs(a - b) < 1e-7, (a, b)
    src = CPP.read_text()
    assert "underflows float32 to exactly 0" in src
    assert "-1.0e4f" in src


def test_the_temperature_sharpens_and_flips_nothing_at_1(host):
    exe = host
    """temperature is a DIVISOR on the logits (`z / t`), so t < 1 sharpens.
    A negative t would flip the argmax and a zero would divide by zero; both are
    refused at the boundary, and both are the caller's error rather than
    something to pass on."""
    a1, _, _, p1 = _readout(exe, k=2, kmax=2, t=1.0, logits=[0.4, 0.0])
    a2, c2, _, p2 = _readout(exe, k=2, kmax=2, t=0.5, logits=[0.4, 0.0])
    assert a1 == a2 == 0
    assert c2 > p1[0], "t < 1 must sharpen"
    src = CPP.read_text()
    assert "would divide by zero" in src and "flip the" in src


def test_score_is_the_EXPECTED_LEVEL_not_the_argmax(host):
    """`sum(i * p_i)`. Checked against the closed form rather than a remembered
    constant, so a change in the arithmetic shows up in THIS expression and not
    as a stale literal -- and the band it lands in is the assertion that it is
    not the argmax, which would be 0.0 on a row that is 87% on option 0."""
    exe = host
    _, _, score, probs = _readout(exe, k=4, kmax=4, t=1.0,
                                  logits=[2.0, -1.0, -1.0, -1.0])
    z = np.exp(np.array([2.0, -1.0, -1.0, -1.0]))
    z /= z.sum()
    assert np.allclose(probs, z, atol=1e-6), probs
    expect = float((np.arange(4) * z).sum())
    assert abs(score - expect) < 1e-5, (score, expect)
    assert 0.05 < score < 0.35, (
        f"sum(i * p_i) on a row 87% concentrated on option 0 lands near {expect}; "
        f"got {score}, and an argmax readout would have said 0.0")


# --------------------------------------------------------------------------
# the source-level contracts
# --------------------------------------------------------------------------

def test_the_weight_is_read_as_N_by_K_and_the_ARRANGEMENT_IS_STATED():
    """The transpose. `Linear.weight` is [out_features, in_features] and the
    container's `gemm_b_host` tensors are [N, K] row-major, so the k loop must
    walk `w + n * K`.

    This is asserted on the source because a test that catches it at runtime
    needs a trained model, and the symptom -- a confident-looking model at
    chance -- is exactly the kind that survives review."""
    src = CPP.read_text()
    i = src.index("void gemm_nk(")
    body = src[i:src.index("\n}\n", i)]
    assert "const float *wn = w + n * K;" in body, (
        "gemm_nk indexes the weight as w[k * N + n], which is the TRANSPOSE of "
        "what the container stores")
    assert "w + k;" not in body
    # and the N, K order is the CONTAINER's, stated where it is used
    assert "[N, K]" in src or "[N, K] ROW-MAJOR" in src


def test_the_head_norm_eps_is_torchs_default_and_NOT_the_encoders():
    """1e-5, not the encoder's hard-coded 1e-12.

    The plan's note calls that difference ~5e-6 relative and "far under the
    bf16 operand's 8-bit mantissa" -- which is true of the ENCODER, whose
    operands are bf16 on the NPU. The head runs here in float32 with nothing to
    hide behind, and upstream's head norms are `nn.LayerNorm(d)` and torch's
    default inside nn.TransformerEncoderLayer. Carrying 1e-12 into the head
    would be a deliberate divergence from the reference in the one place the
    reference is exactly reproducible."""
    src = CPP.read_text()
    assert "constexpr float kHeadEps = 1e-5f;" in src
    assert src.count("kHeadEps") == 3, "the constant must be the only eps used"
    # the encoder's 1e-12 must NOT have leaked in here
    head = src[src.index("void Head::forward("):]
    assert "1e-12" not in head


def test_the_head_attention_reads_NO_band_and_has_NO_flag():
    """Full-band, padding-masked only. A banded head is a wrong answer rather
    than a slow one, and the robust form of "do not band the head" is for the
    head's attention not to have a band parameter at all -- a caller that
    forgets to clear a shared flag gets exactly the wrong model."""
    src = CPP.read_text()
    head = src[src.index("void Head::forward("):src.index("void Head::score(")]
    assert "band_now" not in head and "band_lo" not in head and "band_hi" not in head
    assert "band_half" not in head
    assert "g_sliding_layer" not in head
    # and the encoder's own entry points are untouched by the head
    assert "band_now = 0" not in head, (
        "setting band_now to 0 and calling the encoder's qk would be the "
        "toggle the plan rules out")


def test_the_head_uses_ReLU_and_the_encoder_uses_GELU():
    """torch's DEFAULT nn.TransformerEncoderLayer activation. Two wrong ways
    here: GELU (the encoder's) and SiLU (this runtime's gated default). The
    head is 2 layers deep against the encoder's 22, so a swapped activation is
    a slightly wrong model rather than an obviously broken one."""
    src = CPP.read_text()
    head = src[src.index("void Head::forward("):src.index("void Head::score(")]
    assert "relu_inplace(scratch2_.data()" in head
    assert "gelu" not in head.split("void Head::score(")[0].lower() or \
        "gelu_erf" not in head
    enc = (REPO / "src/open_npue/npue_encoder.hpp").read_text()
    assert "gelu_erf_exact" in enc and "relu_inplace" not in enc


def test_the_attention_scale_is_applied_to_the_Q_ROWS_once():
    """Explicit here, and on Q only. The encoder's packer folds 1/sqrt(head_dim)
    into its Q block; the head's weights are packed plain, so the fold has to
    happen -- and folding it into the weight BEFORE the GEMM would scale K and V
    too, because in_proj is fused. Omitting it entirely leaves the pre-softmax
    scores 8x too large, which saturates the softmax and reads as confidence.

    This asserts the STRIDE, and structurally rather than numerically, because
    the stride is a syntactic property and the numeric comparison cannot see it.
    Measured on this branch, post-head state against the float64 oracle:

        correct    cosine 0.978966   max|d|/max|want| 0.4116
        Q bug      cosine 0.981232   max|d|/max|want| 0.4643

    The bug scored HIGHER on cosine. A 0.99 threshold -- which this test used to
    assert, on the reasoning that "even after the softmax it is nowhere near
    0.99" -- was unreachable on correct code, which measures 0.9790, so the test
    failed on the fix and could not have failed on the bug. There is no
    threshold on this comparison that separates the two arms with enough margin
    to survive a checkpoint change, so the numeric gate is not here. The stride
    assertion below is the gate for this bug, and it is exact.
    """
    src = CPP.read_text()
    head = src[src.index("void Head::forward("):src.index("void Head::score(")]
    assert "1.0f / std::sqrt(static_cast<float>(head_dim_))" in head

    # THE STRIDE. One row of qkv_ is [Q(d) | K(d) | V(d)], so the Q third of row i
    # starts at i*3*d. A contiguous run of R*S*d elements -- the shape this
    # assertion used to require -- is not "the Q third of every row"; it is the
    # first R*S/3 WHOLE rows, so it scales Q, K and V of the first third of the
    # tokens and nothing at all for the rest.
    assert "float *row = qkv_.data() + i * 3 * d;" in head, (
        "the 1/sqrt(head_dim) fold must walk rows with stride 3*d. A flat loop "
        "over a [R*S, 3*d] buffer scales Q, K and V of the first third of the "
        "tokens instead of Q alone of every token -- the bug this test pins.")
    assert "for (int64_t j = 0; j < d; ++j) row[j] *= sc;" in head, (
        "exactly d floats of each row, which is that row's Q third")
    assert "qkv_[static_cast<size_t>(i)] *= sc" not in head, (
        "this is the flat-prefix form of the fold, and it is the bug. If this "
        "fires, the stride assertion above has been rewritten to permit it.")

    # V must be read UNSCALED: the fold is on the Q third only.
    assert "qkv_.data() + b * S * 3 * d + 2 * d" in head


def test_type_emb_is_added_before_the_head_and_outside_it():
    """Broadcast over the whole sequence, before the head and before the
    gather. A gather that happens first cannot see the question type, and a
    head that adds it internally would need a per-ROW integer threaded through
    an arithmetic function -- which is how it ends up on the wrong axis."""
    src = CPP.read_text()
    dec = src[src.index("std::vector<RowAnswer> Decider::decide("):]
    add = dec.index("type_emb.weight")
    fwd = dec.index("head_->forward(")
    assert add < fwd, "type_emb must be added BEFORE the head runs"
    head = src[src.index("void Head::forward("):src.index("void Head::score(")]
    code = [l for l in head.splitlines() if not l.lstrip().startswith("//")]
    assert "type_emb" not in "\n".join(code), (
        "the head must not add type_emb itself; the row's qtype belongs to the "
        "caller, and threading it in here would put it on the wrong axis")


def test_the_marker_is_recorded_BEFORE_its_option_is_appended():
    """The marker is the position OF the [MASK], and the gather reads the
    marker's own embedding. Recording it after the option's tokens would point
    the gather at the first option token -- a plausible model, the wrong one,
    and one whose only symptom is a slightly weaker signal."""
    src = CPP.read_text()
    i = src.index("for (const auto &o : opt_ids) {", src.index("Decider::build_prompt("))
    body = src[i:i + 400]
    assert body.index("out.markers.push_back") < body.index("out.ids.insert")


def test_a_marker_past_the_cut_is_DROPPED_not_clamped():
    """The option it named is no longer in the sequence. Reporting k as the
    option count would return a probability for an option the model never saw,
    and the length check passes because k is self-consistent."""
    src = CPP.read_text()
    i = src.index("out.markers.erase(")
    assert "m >= max_len" in src[i:i + 200]


def test_act_head_is_COMPUTED_and_not_returned():
    """Both halves. It is computed because skipping it is a silent divergence
    from the checkpoint, and a missing tensor and a deliberately-unused one are
    different things. It is not in the answer because Laya's own tracker records
    act_probability at AUROC 0.30 against 0.77 for confidence."""
    src = CPP.read_text()
    assert "head_->act(pooled, feats," in src
    assert "A.act.assign(" in src
    hdr = ENC.read_text()
    i = hdr.index("std::vector<float> act;")
    # The note is ABOVE the member, so read the window before it rather than
    # after: a test that looked forward from the declaration would be asserting
    # on whatever the NEXT member's comment happens to say.
    j = max(0, i - 900)
    assert "COMPUTED AND DISCARDED" in hdr[j:i + 40], hdr[j:i + 40]
    assert "AUROC 0.30" in hdr[j:i + 40]


# --------------------------------------------------------------------------
# the head is a PURE FUNCTION of its input
# --------------------------------------------------------------------------

def test_the_heads_output_does_not_depend_on_its_thread_count(host, monkeypatch):
    """Bit-identical across repeats AND across worker counts."""
    import glob
    import os
    import struct
    monkeypatch.setenv("OFLM_LAYA_CONTAINER", "")
    cands = []
    if os.environ.get("OFLM_LAYA_CONTAINER"):
        cands.append(os.environ["OFLM_LAYA_CONTAINER"])
    for root in (os.environ.get("OFLM_MODEL_PATH", ""),
                 os.path.expanduser("~/.config/oflm/models"),
                 os.path.expanduser("~/.oflm/models"),
                 "/home/ankk98/models"):
        if root and os.path.isdir(root):
            cands += sorted(glob.glob(os.path.join(root, "**", "*.npue"),
                                      recursive=True))
    chosen = None
    for c in (x for x in cands if os.path.isfile(x)):
        try:
            with open(c, "rb") as f:
                f.seek(16)
                off, ln = struct.unpack("<QQ", f.read(16))
                blob = f.read(ln).decode("utf-8", "ignore")
            if "modernbert" in blob:
                chosen = c
                break
        except Exception:
            continue
    if chosen is None:
        pytest.skip("no packed laya container on this host; set "
                    "OFLM_LAYA_CONTAINER to run this")
    p = _run(host, "repro", chosen)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "repeat_w1 1 cross 1 repeat_w12 1 maxdiff 0" in p.stdout, (
        "the head's output depends on its thread count, or is not repeatable "
        "run to run -- which is the signature of a write wider than the slice it "
        "owns:\n" + p.stdout)


# --------------------------------------------------------------------------
# temp_bucket: the per-(type, option-count) calibration key
# --------------------------------------------------------------------------

BUCKET_CASES = [
    # (qtype, k, expected) -- upstream common.py:543-545, verbatim.
    (0, 1, "choice:2"), (0, 2, "choice:2"), (0, 3, "choice:3-5"),
    (0, 5, "choice:3-5"), (0, 6, "choice:6-10"), (0, 10, "choice:6-10"),
    (0, 11, "choice:11+"), (0, 40, "choice:11+"),
    (1, 2, "score:2"), (1, 6, "score:6-10"), (1, 11, "score:11+"), (1, 40, "score:11+"),
    (2, 2, "noul:2"), (2, 3, "noul:3-5"),
]


def test_temp_bucket_matches_upstream_exactly(host):
    """`size` is "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else
    "11+", and the key is "<qtype name>:<size>".

    The boundaries are the whole test: `k <= 2` rather than `k == 2` means a
    ONE-option question lands in the same bucket as a two-option one, and an
    off-by-one here is a silently missing calibration rather than an error.
    """
    out = _run(host, "buckets").stdout
    got = {}
    for line in out.splitlines():
        if line.startswith("BUCKET "):
            _, qt, k, key = line.split()
            got[(int(qt), int(k))] = key
    for qt, k, want in BUCKET_CASES:
        assert got.get((qt, k)) == want, (qt, k, want, got.get((qt, k)))


def test_the_three_qtype_names_are_distinct(host):
    """`type_emb.weight` is [3, d] indexed by QTYPES order, so a bucket name that
    collides across types would apply one type's calibration to another."""
    out = _run(host, "buckets").stdout
    names = {line.split()[3].split(":")[0]
             for line in out.splitlines() if line.startswith("BUCKET ")}
    assert names == {"choice", "score", "noul"}, names


# --------------------------------------------------------------------------
# The head's ATTENTION, against the oracle -- not just the readout
# --------------------------------------------------------------------------

def _container_path():
    root = os.environ.get("OFLM_MODEL_PATH")
    if not root:
        return None, None
    for base in (Path(root) / "models" / "laya", Path(root) / "laya"):
        hits = sorted(base.glob("laya-decision:*.npue"))
        if hits and (base / "multilingual" / "encoder" / "config.json").is_file():
            return hits[0], base / "multilingual"
    return None, None


def _head_rows_rig(tmp_path):
    """A rig that can run the head's ATTENTION (`LAYA_HEAD_FWD`) as well as its
    readout, because the readout alone cannot validate the head."""
    if shutil_which("g++") is None:                # pragma: no cover
        pytest.skip("no C++ compiler on this host")
    src = REPO / "utilities" / "laya_head_rows_rig.cpp"
    if not src.is_file():
        pytest.skip("utilities/laya_head_rows_rig.cpp is absent")
    exe = tmp_path / "headrows"
    units = ["decision_engine.cpp", "npue_encoder.cpp", "npue.cpp",
             "npu_device.cpp", "npue_pack.cpp", "json_min.cpp",
             "xlmr_tokenizer_gen.cpp", "gemma_tokenizer_gen.cpp",
             "bbpe_tokenizer_gen.cpp", "tokenizer_bbpe.cpp",
             "tokenizer_xlmr.cpp", "tokenizer_gemma.cpp", "tokenizer.cpp",
             "gemma_kernels.cpp", "gemma_encode.cpp"]
    cmd = [shutil_which("g++"), "-std=c++17", "-O2", "-mavx512f", "-mavx2",
           "-mfma", "-o", str(exe), str(src),
           *[str(REPO / "src/open_npue" / u) for u in units],
           "-I", str(REPO / "src/open_npue"),
           "-I", "/opt/xilinx/xrt/include",
           "-L", "/opt/xilinx/xrt/lib64", "-lxrt_coreutil",
           "-Wl,-rpath,/opt/xilinx/xrt/lib64"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-4000:]
    return exe


# A GATE THAT WAS HERE, AND WHY IT IS NOT ONE ANY MORE
#
# `test_the_head_attention_matches_the_oracle_not_just_the_readout` compared the
# post-head STATE against the float64 oracle and asserted cosine > 0.99, on the
# reasoning that an 8x Q-scaling error "is nowhere near 0.99 cosine". Measured on
# this branch, by temporarily reintroducing the bug:
#
#     correct    cosine 0.978966    max|d|/max|want| 0.4116
#     Q bug      cosine 0.981232    max|d|/max|want| 0.4643
#
# The bug scored HIGHER on cosine. So the threshold was unreachable on correct
# code -- the test failed on the fix -- and it could not have failed on the bug
# either. The residual metric points the right way but separates the arms by
# 0.4116 against 0.4643, about 6% of a signal that is itself 41% error on
# CORRECT code, which is not a margin that survives a checkpoint change.
#
# The post-head state is a large residual stream with outliers, and the attention
# contribution is a small perturbation of it, so this comparison has a resolution
# and the defect is below it. That is a property of the INSTRUMENT, not a
# threshold to be retuned, and rethresholding is how the wrong number gets
# installed. The stride assertion in
# `test_the_attention_scale_is_applied_to_the_Q_ROWS_once` is the gate for this
# bug, and it is exact: the stride is syntactic, so no numeric tolerance is
# needed to see it.
#
# To measure again, re-run the rig (utilities/laya_head_rows_rig.cpp) with
# LAYA_HEAD_FWD=1 against utilities/laya_preln_reference.py, and record BOTH
# arms. Do not add a single-arm threshold from a correct-code number.
