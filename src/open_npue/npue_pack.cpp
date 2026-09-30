//===- npue_pack.cpp ----------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- build the .npue container from an upstream checkpoint.
// SPDX-License-Identifier: MIT
//
// WHY THIS EXISTS IN C++ AT ALL
// -----------------------------
// tools/pack_npue.py already does this, and Python at BUILD time is fine by
// this project's rules. But the release does not ship the model: the weights
// belong to sentence-transformers/all-MiniLM-L6-v2, and a 66 MB binary blob
// that a user cannot easily check against the original is a worse deal than
// a two-line download from the canonical source with a sha256 to verify.
//
// That trade only works if preparing the model is easy. A Python step would
// have made "download, unzip, run" false. So the packing is here too, and the
// release ships a script that curls two files and calls this.
//
// TWO IMPLEMENTATIONS OF THE SAME LAYOUT IS A RISK, SO IT IS TESTED
// -----------------------------------------------------------------
// A silent disagreement between this and pack_npue.py would produce
// correctly-sized weights in the wrong order -- the exact failure tasks/0022
// hit, and one that a size check cannot catch. So the gate is not "it looks
// right", it is `--prepare-model` and `pack_npue.py` producing a
// BYTE-IDENTICAL file. tools/verify_pack_parity.py runs that comparison.
//
// The container format is documented in docs/04-model/npue-format.md and
// implemented for reference in tools/npue.py.

#include "npue_pack.hpp"

#include "bbpe_tokenizer_gen.hpp"
#include "gemma_tokenizer_gen.hpp"
#include "json_min.hpp"
#include "xlmr_tokenizer_gen.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
// <memory> for std::shared_ptr below. MSVC pulls it in transitively and
// GCC does not, so this file compiled on Windows and not on Linux until
// tasks/0156 A2 built it there (three errors, one missing include).
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace npue {
namespace {

constexpr uint32_t kAlign = 4096;
constexpr int kMacS = 8, kMacT = 8;      // bf16 MMAC sub-tile on aie2p

// --- safetensors ----------------------------------------------------------
// The format is deliberately simple: 8 bytes of little-endian header length,
// that many bytes of JSON, then the tensor bytes. The JSON maps a name to
// {dtype, shape, data_offsets}. Only F32 appears in this checkpoint, and an
// unexpected dtype is refused rather than reinterpreted.
struct Tensor {
  std::string dtype;
  std::vector<int64_t> shape;
  const uint8_t *data = nullptr;
  size_t bytes = 0;
  // Set only when the source dtype was F16 or BF16 and this tensor was
  // widened to fp32 at read time (0076). A shared_ptr so `data` survives the
  // copies and moves a std::map does: the vector object is heap-allocated, so
  // its `data()` is stable for as long as any Tensor holds a reference.
  //
  // WHY WIDEN AT ALL. Every one of the six built-in checkpoints happens to
  // ship F32 safetensors, so the packer only ever needed to read F32. That is
  // a property of those six, not of HuggingFace: the first finetune `add`
  // reached (TaylorAI/bge-micro-v2) ships F16, and refusing it would have
  // made `add` a demo rather than a feature. Widening is exact -- every F16
  // and BF16 value is representable in fp32 -- and it happens once, offline,
  // on a path that then bf16-rounds anyway.
  std::shared_ptr<std::vector<float>> owned;
  const float *f32() const { return reinterpret_cast<const float *>(data); }
  int64_t rows() const { return shape.size() > 1 ? shape[0] : 1; }
  int64_t cols() const { return shape.empty() ? 0 : shape.back(); }
  int64_t count() const {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
  }
};

std::string json_str_field(const std::string &s, size_t from, size_t to,
                           const char *key) {
  const std::string k = std::string("\"") + key + "\"";
  size_t i = s.find(k, from);
  if (i == std::string::npos || i > to) return {};
  i = s.find('"', s.find(':', i) + 1) + 1;
  return s.substr(i, s.find('"', i) - i);
}

std::vector<int64_t> json_int_array(const std::string &s, size_t from,
                                    size_t to, const char *key) {
  std::vector<int64_t> out;
  const std::string k = std::string("\"") + key + "\"";
  size_t i = s.find(k, from);
  if (i == std::string::npos || i > to) return out;
  i = s.find('[', i);
  const size_t end = s.find(']', i);
  size_t p = i + 1;
  while (p < end) {
    while (p < end && !(std::isdigit(static_cast<unsigned char>(s[p])) ||
                        s[p] == '-')) ++p;
    if (p >= end) break;
    out.push_back(std::stoll(s.substr(p)));
    while (p < end && (std::isdigit(static_cast<unsigned char>(s[p])) ||
                       s[p] == '-')) ++p;
  }
  return out;
}

std::map<std::string, Tensor> read_safetensors(const std::vector<uint8_t> &buf) {
  if (buf.size() < 8) throw std::runtime_error("safetensors: file too short");
  uint64_t hlen = 0;
  std::memcpy(&hlen, buf.data(), 8);
  if (8 + hlen > buf.size())
    throw std::runtime_error("safetensors: header length exceeds file");
  const std::string js(reinterpret_cast<const char *>(buf.data() + 8),
                       static_cast<size_t>(hlen));
  const uint8_t *base = buf.data() + 8 + hlen;

  std::map<std::string, Tensor> out;
  size_t p = 0;
  while (true) {
    // Each entry is  "name":{...}. Find the next name at brace depth 1.
    const size_t q1 = js.find('"', p);
    if (q1 == std::string::npos) break;
    const size_t q2 = js.find('"', q1 + 1);
    if (q2 == std::string::npos) break;
    const std::string name = js.substr(q1 + 1, q2 - q1 - 1);
    const size_t ob = js.find('{', q2);
    if (ob == std::string::npos) break;
    const size_t cb = js.find('}', ob);
    if (cb == std::string::npos) break;
    p = cb + 1;
    if (name == "__metadata__") continue;

    Tensor t;
    t.dtype = json_str_field(js, ob, cb, "dtype");
    // A non-F32 tensor is not an error by itself -- this checkpoint carries
    // embeddings.position_ids as I64 and nothing reads it. It becomes an
    // error only if something asks for it, which get() enforces. Rejecting
    // the whole file here would refuse a checkpoint that is perfectly usable;
    // ignoring the dtype at read time would reinterpret integers as floats.
    t.shape = json_int_array(js, ob, cb, "shape");
    const auto off = json_int_array(js, ob, cb, "data_offsets");
    if (off.size() != 2)
      throw std::runtime_error("safetensors: " + name + " has no data_offsets");
    t.data = base + off[0];
    t.bytes = static_cast<size_t>(off[1] - off[0]);
    if (t.dtype == "F32" && t.bytes != static_cast<size_t>(t.count()) * 4)
      throw std::runtime_error("safetensors: " + name + " size disagrees with "
                               "its shape");

    // Widen F16/BF16 to fp32 here, once, so every consumer below sees F32 and
    // none of them has to know. Both conversions are EXACT (fp32 has more
    // exponent range and more mantissa than either), so this cannot be the
    // source of any error measured downstream.
    if (t.dtype == "F16" || t.dtype == "BF16") {
      const size_t n = static_cast<size_t>(t.count());
      if (t.bytes != n * 2)
        throw std::runtime_error("safetensors: " + name + " size disagrees "
                                 "with its shape");
      t.owned = std::make_shared<std::vector<float>>(n);
      const uint16_t *src = reinterpret_cast<const uint16_t *>(t.data);
      float *dst = t.owned->data();
      if (t.dtype == "BF16") {
        // bf16 IS the top half of an fp32: shift, done. No rounding, no
        // special cases -- NaN and Inf patterns come across unchanged.
        for (size_t i = 0; i < n; ++i) {
          const uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
          std::memcpy(&dst[i], &bits, 4);
        }
      } else {
        // IEEE half -> float. Subnormals and the Inf/NaN exponent both need
        // their own arm; getting either wrong is silent, so both are here
        // rather than approximated by the fast path.
        for (size_t i = 0; i < n; ++i) {
          const uint16_t h = src[i];
          const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
          uint32_t exp = (h >> 10) & 0x1Fu;
          uint32_t man = h & 0x3FFu;
          uint32_t bits;
          if (exp == 0) {
            if (man == 0) {
              bits = sign;                       // +-0
            } else {
              // Subnormal half: normalise it into a normal float.
              int shift = 0;
              while (!(man & 0x400u)) { man <<= 1; ++shift; }
              man &= 0x3FFu;
              bits = sign | ((127 - 15 - shift + 1) << 23) | (man << 13);
            }
          } else if (exp == 0x1F) {
            bits = sign | 0x7F800000u | (man << 13);   // Inf / NaN
          } else {
            bits = sign | ((exp + (127 - 15)) << 23) | (man << 13);
          }
          std::memcpy(&dst[i], &bits, 4);
        }
      }
      t.data = reinterpret_cast<const uint8_t *>(dst);
      t.bytes = n * 4;
      t.dtype = "F32";
    }
    out.emplace(name, t);
  }
  return out;
}

// --- sha256 ---------------------------------------------------------------
// The container records the checksum of the checkpoint it was built from, and
// the goldens assert against it. Without it a .npue cannot say which weights
// it came from, which is the whole point of pinning a source.
struct Sha256 {
  uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  uint64_t len = 0;
  uint8_t buf[64] = {};
  size_t have = 0;

  static uint32_t ror(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

  void block(const uint8_t *p) {
    static const uint32_t K[64] = {
      0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,
      0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
      0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,
      0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
      0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,
      0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
      0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,
      0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
      0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,
      0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
      0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = (uint32_t(p[i * 4]) << 24) | (uint32_t(p[i * 4 + 1]) << 16) |
             (uint32_t(p[i * 4 + 2]) << 8) | uint32_t(p[i * 4 + 3]);
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = ror(w[i-15],7) ^ ror(w[i-15],18) ^ (w[i-15] >> 3);
      const uint32_t s1 = ror(w[i-2],17) ^ ror(w[i-2],19) ^ (w[i-2] >> 10);
      w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t S1 = ror(e,6) ^ ror(e,11) ^ ror(e,25);
      const uint32_t ch = (e & f) ^ (~e & g);
      const uint32_t t1 = hh + S1 + ch + K[i] + w[i];
      const uint32_t S0 = ror(a,2) ^ ror(a,13) ^ ror(a,22);
      const uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t t2 = S0 + mj;
      hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
  }

  void update(const uint8_t *p, size_t n) {
    len += n;
    while (n) {
      const size_t take = std::min(n, size_t(64) - have);
      std::memcpy(buf + have, p, take);
      have += take; p += take; n -= take;
      if (have == 64) { block(buf); have = 0; }
    }
  }

  std::string hex() {
    const uint64_t bits = len * 8;
    uint8_t pad = 0x80;
    update(&pad, 1);
    const uint8_t zero = 0;
    while (have != 56) update(&zero, 1);
    uint8_t be[8];
    for (int i = 0; i < 8; ++i) be[i] = uint8_t(bits >> (56 - 8 * i));
    len -= 8;                       // the length field is not part of the data
    update(be, 8);
    std::string s;
    static const char *H = "0123456789abcdef";
    for (uint32_t v : h)
      for (int i = 28; i >= 0; i -= 4) s.push_back(H[(v >> i) & 0xF]);
    return s;
  }
};

// --- the container --------------------------------------------------------

// fp32 -> bf16 bits, round-to-nearest-even. Must match tools/npue.py exactly:
// truncation would bias every one of 10.6 M weights toward zero.
inline uint16_t bf16_rne(float x) {
  uint32_t u;
  std::memcpy(&u, &x, sizeof u);
  return static_cast<uint16_t>(((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF);
}

// The pre-tiling, matching tools/npue.py's tile_b(order="k,n"):
//   [K,N] -> [kb][nb][tk/s][tn/t][s][t]
// Both re-layouts the runtime design would otherwise do at load time are
// absorbed here, which is the whole point of the format.
std::vector<uint16_t> tile_b(const float *mat, int64_t K, int64_t N,
                             int64_t tk, int64_t tn) {
  if (K % tk || N % tn)
    throw std::runtime_error(
        "operand [" + std::to_string(K) + "," + std::to_string(N) +
        "] does not tile evenly by (" + std::to_string(tk) + "," +
        std::to_string(tn) + "): K%tk=" + std::to_string(K % tk) +
        ", N%tn=" + std::to_string(N % tn));
  const int64_t kb_n = K / tk, nb_n = N / tn;
  std::vector<uint16_t> out(static_cast<size_t>(K) * N);
  size_t w = 0;
  for (int64_t kb = 0; kb < kb_n; ++kb)
    for (int64_t nb = 0; nb < nb_n; ++nb)
      for (int64_t si = 0; si < tk / kMacS; ++si)
        for (int64_t ti = 0; ti < tn / kMacT; ++ti)
          for (int64_t s = 0; s < kMacS; ++s)
            for (int64_t t = 0; t < kMacT; ++t) {
              const int64_t r = kb * tk + si * kMacS + s;
              const int64_t c = nb * tn + ti * kMacT + t;
              out[w++] = bf16_rne(mat[r * N + c]);
            }
  return out;
}

struct Entry {
  std::string name, role, dtype, layout_json, layout_hash;
  std::vector<int64_t> shape;
  uint64_t offset = 0, nbytes = 0;
};

class Writer {
public:
  void add(const std::string &name, const void *data, size_t bytes,
           const char *dtype, const char *role,
           const std::vector<int64_t> &shape,
           const std::string &layout_json = {},
           const std::string &layout_hash = {}) {
    Entry e;
    e.name = name;
    e.role = role;
    e.dtype = dtype;
    e.shape = shape;
    e.offset = offset_;
    e.nbytes = bytes;
    e.layout_json = layout_json;
    e.layout_hash = layout_hash;
    entries_.push_back(e);
    const uint8_t *p = static_cast<const uint8_t *>(data);
    blob_.insert(blob_.end(), p, p + bytes);
    offset_ += bytes;
    // Pad AFTER each tensor so the next one starts aligned.
    const uint64_t pad = (kAlign - (offset_ % kAlign)) % kAlign;
    blob_.insert(blob_.end(), static_cast<size_t>(pad), 0);
    offset_ += pad;
  }

  void write(const std::string &path, const std::string &config_json,
            uint32_t arch = 0) const {
    std::string js = "{\"config\":" + config_json + ",\"tensors\":[";
    for (size_t i = 0; i < entries_.size(); ++i) {
      const Entry &e = entries_[i];
      if (i) js += ',';
      js += "{\"name\":\"" + e.name + "\",\"role\":\"" + e.role +
            "\",\"dtype\":\"" + e.dtype + "\",\"logical_shape\":[";
      for (size_t k = 0; k < e.shape.size(); ++k)
        js += (k ? "," : "") + std::to_string(e.shape[k]);
      js += "],\"padded_shape\":[";
      for (size_t k = 0; k < e.shape.size(); ++k)
        js += (k ? "," : "") + std::to_string(e.shape[k]);
      js += "],\"offset\":" + std::to_string(e.offset) +
            ",\"nbytes\":" + std::to_string(e.nbytes);
      if (!e.layout_json.empty())
        js += ",\"layout\":" + e.layout_json +
              ",\"layout_hash\":\"" + e.layout_hash + "\"";
      js += "}";
    }
    js += "]}";

    const uint64_t json_offset = 64;
    uint64_t data_offset = json_offset + js.size();
    data_offset += (kAlign - (data_offset % kAlign)) % kAlign;

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path);
    uint8_t head[64] = {};
    std::memcpy(head, "NPUE", 4);
    const uint32_t version = 1, flags = 1;   // FLAG_PRETILED
    std::memcpy(head + 4, &version, 4);
    std::memcpy(head + 8, &arch, 4);
    std::memcpy(head + 12, &flags, 4);
    const uint64_t jlen = js.size();
    std::memcpy(head + 16, &json_offset, 8);
    std::memcpy(head + 24, &jlen, 8);
    std::memcpy(head + 32, &data_offset, 8);
    std::memcpy(head + 40, &offset_, 8);
    f.write(reinterpret_cast<char *>(head), 64);
    f.write(js.data(), static_cast<std::streamsize>(js.size()));
    const std::string pad(static_cast<size_t>(data_offset - json_offset -
                                              js.size()), '\0');
    f.write(pad.data(), static_cast<std::streamsize>(pad.size()));
    f.write(reinterpret_cast<const char *>(blob_.data()),
            static_cast<std::streamsize>(blob_.size()));
    if (!f) throw std::runtime_error("write failed: " + path);
  }

  size_t count() const { return entries_.size(); }
  uint64_t data_bytes() const { return offset_; }

private:
  std::vector<Entry> entries_;
  std::vector<uint8_t> blob_;
  uint64_t offset_ = 0;
};

// Compose one path inside the checkpoint from an OPTIONAL subdirectory, so an
// empty subdirectory is the flat layout and a non-empty one is a nested
// checkpoint -- with no "//" in either case, because these strings are printed
// in refusal messages and the reader is meant to be able to open what is named.
std::string sub(const std::string &dir, const std::string &subdir,
                const std::string &name) {
  std::string p = dir;
  while (!p.empty() && p.back() == '/') p.pop_back();
  if (!subdir.empty()) {
    p += '/';
    size_t b = 0, e = subdir.size();
    while (b < e && subdir[b] == '/') ++b;
    while (e > b && subdir[e - 1] == '/') --e;
    p += subdir.substr(b, e - b);
  }
  p += '/';
  p += name;
  return p;
}

std::vector<uint8_t> slurp(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open " + path);
  const std::streamsize n = f.tellg();
  f.seekg(0);
  std::vector<uint8_t> buf(static_cast<size_t>(n));
  f.read(reinterpret_cast<char *>(buf.data()), n);
  return buf;
}

}  // namespace

// The one C++ SHA-256, exposed. The downloader must verify a checkpoint with
// EXACTLY the implementation that later records `source_sha256` into the
// container -- a second copy is how the Python side ended up with four.
// Streamed rather than slurped: `model.safetensors` is 438 MB for bge-base
// and there is no reason to hold it twice.
std::string sha256_file(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  Sha256 s;
  std::vector<char> buf(1 << 20);
  while (f) {
    f.read(buf.data(), (std::streamsize)buf.size());
    const std::streamsize got = f.gcount();
    if (got > 0)
      s.update(reinterpret_cast<const uint8_t *>(buf.data()), (size_t)got);
  }
  return s.hex();
}

// The [K,N] operand for a GEMM: transpose the checkpoint's [N,K] weight,
// optionally fold a scale, convert and pre-tile.
static void add_gemm_b(Writer &w, const std::string &name, const Tensor &t,
                       int64_t tk, int64_t tn, const std::string &layout_json,
                       const std::string &layout_hash, float fold = 1.0f,
                       int64_t fold_cols = 0) {
  const int64_t N = t.rows(), K = t.cols();     // checkpoint stores [out, in]
  std::vector<float> m(static_cast<size_t>(K) * N);
  const float *src = t.f32();
  for (int64_t r = 0; r < K; ++r)
    for (int64_t c = 0; c < N; ++c) {
      float v = src[c * K + r];                 // transpose to [K, N]
      if (fold != 1.0f && c < fold_cols) v = v * fold;
      m[r * N + c] = v;
    }
  const auto tiled = tile_b(m.data(), K, N, tk, tn);
  w.add(name, tiled.data(), tiled.size() * 2, "BF16", "gemm_b", {K, N},
        layout_json, layout_hash);
}

// Mirror of tools/pack_npue.py's add_gemm_b_host: the [K,N] operand stored
// PLAIN -- F32, row-major, no tiling. Used only by prepare_model_gemma
// (arch=1), which has no NPU kernel to pre-tile for (tasks/0064). The
// checkpoint's nn.Linear weight is [out, in]; transpose to [in, out] = [K, N]
// so the host does y = x @ W with no runtime transpose, matching add_gemm_b's
// convention above.
static void add_gemm_b_host(Writer &w, const std::string &name,
                            const Tensor &t) {
  const int64_t N = t.rows(), K = t.cols();     // checkpoint stores [out, in]
  std::vector<float> m(static_cast<size_t>(K) * N);
  const float *src = t.f32();
  for (int64_t r = 0; r < K; ++r)
    for (int64_t c = 0; c < N; ++c)
      m[r * N + c] = src[c * K + r];             // transpose to [K, N]
  w.add(name, m.data(), m.size() * sizeof(float), "F32", "gemm_b_host",
        {K, N});
}

// Fuse several [out, in] checkpoint tensors along N into ONE [K, N] operand
// and ZERO-PAD the tail to `n_padded`. Mirror of tools/pack_npue.py's
// pack_gemma() qkv assembly (tasks/0074).
//
// The padding is the whole reason this model reaches the array. MQA gives K
// and V a width of 256, which caps the legal `tile_n` at 16 across the whole
// design; padding the fused operand to a multiple of `tile_n * n_aie_cols`
// removes the cap. Zero columns of B produce exactly-zero columns of C, so the
// host slices Q/K/V off the front by offset and ignores the tail -- exact, not
// approximate. The zeros are written by the value-initialised vector below,
// which matters: an uninitialised tail would tile whatever was on the heap.
static void add_gemm_b_concat_pad(Writer &w, const std::string &name,
                                  const std::vector<const Tensor *> &parts,
                                  int64_t n_padded, int64_t tk, int64_t tn,
                                  const std::string &layout_json,
                                  const std::string &layout_hash) {
  if (parts.empty()) throw std::runtime_error(name + ": no parts to fuse");
  const int64_t K = parts[0]->cols();
  int64_t used = 0;
  for (const Tensor *t : parts) {
    if (t->cols() != K)
      throw std::runtime_error(name + ": fused parts disagree on the `in` dim");
    used += t->rows();
  }
  if (n_padded < used)
    throw std::runtime_error(name + ": padded N is narrower than its parts");
  std::vector<float> m(static_cast<size_t>(K) * n_padded, 0.0f);
  int64_t base = 0;
  for (const Tensor *t : parts) {
    const int64_t Np = t->rows();
    const float *s = t->f32();
    for (int64_t r = 0; r < K; ++r)
      for (int64_t c = 0; c < Np; ++c)
        m[r * n_padded + base + c] = s[c * K + r];    // transpose to [K, N]
    base += Np;
  }
  const auto tiled = tile_b(m.data(), K, n_padded, tk, tn);
  w.add(name, tiled.data(), tiled.size() * 2, "BF16", "gemm_b", {K, n_padded},
        layout_json, layout_hash);
}

Layout gemm_b_layout(int64_t tile_k, int64_t tile_n, int64_t mac_s,
                     int64_t mac_t) {
  const std::string k = std::to_string(tile_k), n = std::to_string(tile_n);
  const std::string s = std::to_string(mac_s), tt = std::to_string(mac_t);
  Layout L;
  // Insertion order, matching tools/npue.py's dict literal.
  L.json = "{\"kind\":\"block_panel\",\"tile_k\":" + k + ",\"tile_n\":" + n +
           ",\"order\":\"k,n,kt,nt\",\"inner\":\"s,t\"" +
           ",\"mac_s\":" + s + ",\"mac_t\":" + tt + ",\"dtype\":\"BF16\"}";
  // json.dumps(..., sort_keys=True, separators=(",", ":")) -- keys sorted
  // alphabetically: dtype, inner, kind, mac_s, mac_t, order, tile_k, tile_n.
  const std::string canonical =
      "{\"dtype\":\"BF16\",\"inner\":\"s,t\",\"kind\":\"block_panel\",\"mac_s\":" + s +
      ",\"mac_t\":" + tt + ",\"order\":\"k,n,kt,nt\",\"tile_k\":" + k +
      ",\"tile_n\":" + n + "}";
  Sha256 h;
  h.update(reinterpret_cast<const uint8_t *>(canonical.data()),
           canonical.size());
  L.hash = h.hex();
  return L;
}

void prepare_model(const std::string &safetensors, const std::string &vocab,
                   const std::string &config_json_path,
                   const std::string &pooling,
                   const std::string &source_repo,
                   const std::string &out, const std::string &source_sha,
                   const std::string &layout_json,
                   const std::string &layout_hash,
                   int64_t tile_k, int64_t tile_n, int64_t max_seq,
                   void (*log)(const std::string &)) {
  const auto st_buf = slurp(safetensors);
  const auto src = read_safetensors(st_buf);
  std::string sha = source_sha;
  if (sha.empty()) {
    Sha256 s;
    s.update(st_buf.data(), st_buf.size());
    sha = s.hex();
  }
  auto get = [&](const std::string &n) -> const Tensor & {
    auto it = src.find(n);
    if (it == src.end())
      throw std::runtime_error("checkpoint has no tensor '" + n + "'");
    if (it->second.dtype != "F32")
      throw std::runtime_error("checkpoint tensor '" + n + "' is " +
                               it->second.dtype + "; this packer reads F32");
    return it->second;
  };

  // The architecture is read from config.json rather than assumed, so a
  // different BERT-family checkpoint fails loudly instead of silently
  // packing the wrong shapes.
  const std::string cfg(reinterpret_cast<const char *>(
      slurp(config_json_path).data()), slurp(config_json_path).size());
  auto cfg_int = [&](const char *key) -> int64_t {
    const std::string k = std::string("\"") + key + "\"";
    const size_t i = cfg.find(k);
    if (i == std::string::npos)
      throw std::runtime_error(std::string("config.json has no ") + key);
    return std::stoll(cfg.substr(cfg.find(':', i) + 1));
  };
  const int64_t L = cfg_int("num_hidden_layers");
  const int64_t H = cfg_int("num_attention_heads");
  const int64_t hidden = cfg_int("hidden_size");
  const int64_t inter = cfg_int("intermediate_size");
  const int64_t vocab_size = cfg_int("vocab_size");
  const int64_t type_vocab = cfg_int("type_vocab_size");
  const int64_t head_dim = hidden / H;
  // FLOAT, not double. numpy multiplies a float32 array by a Python float in
  // float32; computing in double and rounding once at the end is a DIFFERENT
  // rounding, and it moved 86 of 2304 bias values by 1 ULP -- found only
  // because the parity test compares bytes rather than tolerances.
  const float scale = static_cast<float>(1.0 / std::sqrt(
      static_cast<double>(head_dim)));

  std::ostringstream cj;
  cj.precision(17);
  cj << "{\"arch\":\"bert_abs_gelu_postln\""
     << ",\"source_repo\":\"" << source_repo << "\""
     << ",\"source_sha256\":\"" << sha << "\""
     << ",\"num_layers\":" << L << ",\"num_heads\":" << H
     << ",\"hidden\":" << hidden << ",\"head_dim\":" << head_dim
     << ",\"intermediate\":" << inter
     << ",\"layer_norm_eps\":1e-12"
     << ",\"vocab_size\":" << vocab_size
     << ",\"max_seq_len\":" << max_seq
     << ",\"pooling\":\"" << pooling << "\",\"l2_normalize\":true"
     << ",\"activation\":\"gelu_erf_exact\""
     << ",\"tile_k\":" << tile_k << ",\"tile_n\":" << tile_n
     << ",\"mac_s\":" << kMacS << ",\"mac_t\":" << kMacT
     // The operand datapath (tasks/0078). This C++ packer only produces bf16
     // -- int8 needs a SmoothQuant calibration pass that runs the numpy
     // oracle, which is build-time Python by design (CLAUDE.md rule 5) -- but
     // it must still WRITE the key, in the same position tools/pack_npue.py
     // writes it, or the two packers stop being byte-identical and
     // tools/verify_pack_parity.py fails for a reason that is not a bug.
     << ",\"a_dtype\":\"bf16\""
     << ",\"fusions\":{\"qkv_fused\":true,\"transposed_to_kn\":true,"
        "\"qk_scale_folded_into_q\":true,\"gemm_operands_bf16\":true,"
        "\"biases_and_layernorm_fp32\":true,"
        "\"position_embeddings_presliced_to\":" << max_seq << "}"
     << ",\"not_implemented\":[\"pooler.dense (unused by "
        "sentence-transformers)\"]}";

  Writer w;
  auto add_f32 = [&](const std::string &name, const Tensor &t,
                     const char *role, const std::vector<int64_t> &shape,
                     int64_t limit_rows = 0) {
    const size_t n = static_cast<size_t>(
        limit_rows ? limit_rows * t.cols() : t.count());
    w.add(name, t.data, n * 4, "F32", role, shape);
  };

  add_f32("embeddings.word", get("embeddings.word_embeddings.weight"),
          "embedding", {vocab_size, hidden});
  add_f32("embeddings.position", get("embeddings.position_embeddings.weight"),
          "embedding", {max_seq, hidden}, max_seq);
  add_f32("embeddings.token_type",
          get("embeddings.token_type_embeddings.weight"), "embedding",
          {type_vocab, hidden});
  add_f32("embeddings.ln.weight", get("embeddings.LayerNorm.weight"),
          "layernorm", {hidden});

  const auto vb = slurp(vocab);
  w.add("tokenizer.vocab", vb.data(), vb.size(), "U8", "tokenizer",
        {static_cast<int64_t>(vb.size())});

  add_f32("embeddings.ln.bias", get("embeddings.LayerNorm.bias"), "layernorm",
          {hidden});

  for (int64_t i = 0; i < L; ++i) {
    const std::string p = "encoder.layer." + std::to_string(i) + ".";
    const std::string sa = p + "attention.self.";
    const std::string ao = p + "attention.output.";
    const std::string tag = "layer." + std::to_string(i) + ".";

    // QKV fused along `out`, transposed to [K, N], and 1/sqrt(head_dim)
    // folded into the Q block only -- the first `hidden` columns.
    {
      const Tensor &q = get(sa + "query.weight");
      const Tensor &k = get(sa + "key.weight");
      const Tensor &v = get(sa + "value.weight");
      const int64_t N = 3 * hidden;
      std::vector<float> m(static_cast<size_t>(hidden) * N);
      const Tensor *parts[3] = {&q, &k, &v};
      for (int b = 0; b < 3; ++b)
        for (int64_t o = 0; o < hidden; ++o)
          for (int64_t in = 0; in < hidden; ++in) {
            float val = parts[b]->f32()[o * hidden + in];
            if (b == 0) val = val * scale;             // float32 throughout
            m[in * N + b * hidden + o] = val;
          }
      const auto tiled = tile_b(m.data(), hidden, N, tile_k, tile_n);
      w.add(tag + "qkv", tiled.data(), tiled.size() * 2, "BF16", "gemm_b",
            {hidden, N}, layout_json, layout_hash);

      std::vector<float> bias(static_cast<size_t>(N));
      const char *bn[3] = {"query.bias", "key.bias", "value.bias"};
      for (int b = 0; b < 3; ++b) {
        const float *s = get(sa + bn[b]).f32();
        for (int64_t o = 0; o < hidden; ++o)
          bias[b * hidden + o] = b == 0 ? s[o] * scale : s[o];
      }
      w.add(tag + "qkv.bias", bias.data(), bias.size() * 4, "F32", "bias", {N});
    }

    add_gemm_b(w, tag + "attn_out", get(ao + "dense.weight"), tile_k, tile_n,
               layout_json, layout_hash);
    add_f32(tag + "attn_out.bias", get(ao + "dense.bias"), "bias", {hidden});
    add_f32(tag + "ln1.weight", get(ao + "LayerNorm.weight"), "layernorm",
            {hidden});
    add_f32(tag + "ln1.bias", get(ao + "LayerNorm.bias"), "layernorm",
            {hidden});

    add_gemm_b(w, tag + "ffn_up", get(p + "intermediate.dense.weight"),
               tile_k, tile_n, layout_json, layout_hash);
    add_f32(tag + "ffn_up.bias", get(p + "intermediate.dense.bias"), "bias",
            {inter});
    add_gemm_b(w, tag + "ffn_down", get(p + "output.dense.weight"),
               tile_k, tile_n, layout_json, layout_hash);
    add_f32(tag + "ffn_down.bias", get(p + "output.dense.bias"), "bias",
            {hidden});
    add_f32(tag + "ln2.weight", get(p + "output.LayerNorm.weight"),
            "layernorm", {hidden});
    add_f32(tag + "ln2.bias", get(p + "output.LayerNorm.bias"), "layernorm",
            {hidden});
  }

  w.write(out, cj.str());
  if (log) {
    std::ostringstream s;
    s << "  packed " << w.count() << " tensors, "
      << (w.data_bytes() / 1e6) << " MB of tensor data";
    log(s.str());
  }
}

// arch=1 mirror of tools/pack_npue.py's pack_gemma() (tasks/0064,
// tasks/0065). Deliberately NOT threaded through prepare_model() above --
// 4 RMSNorms/layer (not 2 LayerNorms), MQA, q_norm/k_norm, per-layer RoPE
// base, separate gate/up GeGLU matrices, two post-pool Dense heads, no
// biases anywhere, no token_type embedding, no position table -- different
// enough that sharing code would mean threading Gemma-only branches through
// every step of the BERT path. Every GEMM operand is stored PLAIN (F32,
// row-major, no tiling): there is no NPU kernel for this arch, so nothing
// here ever becomes a DMA descriptor.
void prepare_model_gemma(const std::string &model_dir, const std::string &out,
                         const std::string &source_repo,
                         void (*log)(const std::string &), int64_t tile_k,
                         int64_t tile_n, bool host_only) {
  const auto st_buf = slurp(model_dir + "/model.safetensors");
  const auto src = read_safetensors(st_buf);
  Sha256 sh;
  sh.update(st_buf.data(), st_buf.size());
  const std::string sha = sh.hex();

  auto get = [&](const std::string &n) -> const Tensor & {
    auto it = src.find(n);
    if (it == src.end())
      throw std::runtime_error("checkpoint has no tensor '" + n + "'");
    if (it->second.dtype != "F32")
      throw std::runtime_error("checkpoint tensor '" + n + "' is " +
                               it->second.dtype + "; this packer reads F32");
    return it->second;
  };
  auto get_from = [](const std::map<std::string, Tensor> &m,
                     const std::string &n) -> const Tensor & {
    auto it = m.find(n);
    if (it == m.end())
      throw std::runtime_error("checkpoint has no tensor '" + n + "'");
    if (it->second.dtype != "F32")
      throw std::runtime_error("checkpoint tensor '" + n + "' is " +
                               it->second.dtype + "; this packer reads F32");
    return it->second;
  };

  const auto cfg_buf = slurp(model_dir + "/config.json");
  const std::string cfg(reinterpret_cast<const char *>(cfg_buf.data()),
                        cfg_buf.size());

  auto cfg_int = [&](const char *key) -> int64_t {
    const std::string k = std::string("\"") + key + "\"";
    const size_t i = cfg.find(k);
    if (i == std::string::npos)
      throw std::runtime_error(std::string("config.json has no ") + key);
    return std::stoll(cfg.substr(cfg.find(':', i) + 1));
  };
  auto cfg_str = [&](const char *key) -> std::string {
    const std::string k = std::string("\"") + key + "\"";
    const size_t i = cfg.find(k);
    if (i == std::string::npos)
      throw std::runtime_error(std::string("config.json has no ") + key);
    const size_t c = cfg.find(':', i) + 1;
    const size_t q1 = cfg.find('"', c);
    const size_t q2 = cfg.find('"', q1 + 1);
    return cfg.substr(q1 + 1, q2 - q1 - 1);
  };
  // The value's EXACT literal text, copied verbatim rather than reparsed and
  // reformatted. json.dumps(json.loads(x)) only reproduces x byte-for-byte
  // when x is already Python's canonical shortest-round-trip form -- true of
  // every numeric field this reads (checked directly against this
  // checkpoint's config.json: "1e-06", "1000000.0", "10000.0", "512", "256",
  // "262144", "2048" all round-trip unchanged through Python's json module).
  // Reformatting these ourselves would risk drifting from Python's float
  // printer for zero benefit -- the source text already IS the answer.
  auto cfg_raw = [&](const char *key) -> std::string {
    const std::string k = std::string("\"") + key + "\"";
    const size_t i = cfg.find(k);
    if (i == std::string::npos)
      throw std::runtime_error(std::string("config.json has no ") + key);
    size_t c = cfg.find(':', i) + 1;
    while (c < cfg.size() && std::isspace(static_cast<unsigned char>(cfg[c])))
      ++c;
    size_t e = c;
    while (e < cfg.size() && cfg[e] != ',' && cfg[e] != '}' &&
          cfg[e] != '\n' && cfg[e] != '\r')
      ++e;
    while (e > c && std::isspace(static_cast<unsigned char>(cfg[e - 1])))
      --e;
    return cfg.substr(c, e - c);
  };

  const std::string model_type = cfg_str("model_type");
  const int64_t L = cfg_int("num_hidden_layers");
  const int64_t hidden = cfg_int("hidden_size");
  const int64_t heads = cfg_int("num_attention_heads");
  const int64_t kv_heads = cfg_int("num_key_value_heads");
  const int64_t head_dim = cfg_int("head_dim");
  const int64_t inter = cfg_int("intermediate_size");
  // cfg.get("_sliding_window_pattern", 6) -- default 6 if the checkpoint
  // does not carry it.
  int64_t swp = 6;
  if (cfg.find("\"_sliding_window_pattern\"") != std::string::npos)
    swp = cfg_int("_sliding_window_pattern");

  const auto d2_buf = slurp(model_dir + "/2_Dense/model.safetensors");
  const auto d2 = read_safetensors(d2_buf);
  const auto d3_buf = slurp(model_dir + "/3_Dense/model.safetensors");
  const auto d3 = read_safetensors(d3_buf);
  const Tensor &d2w = get_from(d2, "linear.weight");
  const Tensor &d3w = get_from(d3, "linear.weight");
  const int64_t dense_hidden = d2w.shape[0];

  if (log) {
    std::ostringstream s;
    s << "packing " << model_dir << " -> " << out
      << "  (arch=gemma3, HOST-only GEMMs)\n"
      << "  hidden=" << hidden << " heads=" << heads
      << " kv_heads=" << kv_heads << " head_dim=" << head_dim
      << " layers=" << L << " inter=" << inter;
    log(s.str());
  }

  // Exact key order and formatting of tools/pack_npue.py's pack_gemma()
  // config dict -- json.dumps(..., separators=(",", ":")) preserves
  // insertion order, and this must match it byte for byte.
  std::string cj;
  cj += "{\"arch\":\"gemma3_mqa_rope_geglu\"";
  cj += ",\"model_type\":\"" + model_type + "\"";
  cj += ",\"source_repo\":\"" + source_repo + "\"";
  cj += ",\"source_sha256\":\"" + sha + "\"";
  cj += ",\"num_layers\":" + std::to_string(L);
  cj += ",\"hidden\":" + std::to_string(hidden);
  cj += ",\"num_heads\":" + std::to_string(heads);
  cj += ",\"num_key_value_heads\":" + std::to_string(kv_heads);
  cj += ",\"head_dim\":" + std::to_string(head_dim);
  cj += ",\"intermediate\":" + std::to_string(inter);
  cj += ",\"dense_hidden\":" + std::to_string(dense_hidden);
  cj += ",\"rms_norm_eps\":" + cfg_raw("rms_norm_eps");
  cj += ",\"rope_theta\":" + cfg_raw("rope_theta");
  cj += ",\"rope_local_base_freq\":" + cfg_raw("rope_local_base_freq");
  cj += ",\"sliding_window\":" + cfg_raw("sliding_window");
  cj += ",\"sliding_window_pattern\":" + std::to_string(swp);
  cj += ",\"query_pre_attn_scalar\":" + cfg_raw("query_pre_attn_scalar");
  cj += ",\"vocab_size\":" + cfg_raw("vocab_size");
  cj += ",\"max_seq_len\":" + cfg_raw("max_position_embeddings");
  cj += ",\"pooling\":\"mean_include_prompt\",\"l2_normalize\":true";
  cj += ",\"activation\":\"gelu_pytorch_tanh\"";
  cj += ",\"attention_bias\":false,\"dense_bias\":false";
  cj += ",\"not_implemented\":[\"sliding-window mask (exact for "
        "seq_len<=512, see reference/encoder_gemma.py's file header)\"]";

  // The task-prefix table (0075). See pack_gemma()'s comment for WHY it is in
  // the container at all when gemma_tokenizer.bin already carries it: the MTEB
  // harness reads it from here and applies it to both sides, so a prefix
  // mismatch cannot masquerade as a datapath difference.
  //
  // SORTED BY KEY, matching Python. json_min.hpp stores objects in an
  // unordered_map and says so in its own header, so source order is not
  // available to this side at all -- sorting is what makes the two packers
  // byte-identical rather than accidentally-agreeing.
  {
    const auto cst_buf = slurp(model_dir + "/config_sentence_transformers.json");
    const std::string cst_txt(reinterpret_cast<const char *>(cst_buf.data()),
                              cst_buf.size());
    const npue::json::Value cst = npue::json::parse(cst_txt);
    if (!cst.is_object() || !cst.as_object().count("prompts"))
      throw std::runtime_error(
          model_dir + "/config_sentence_transformers.json has no 'prompts' "
          "table -- refusing to pack without this model's own prefixes rather "
          "than inventing them");
    const auto &pr = cst.at("prompts").as_object();
    std::vector<std::string> keys;
    keys.reserve(pr.size());
    for (const auto &kv : pr) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());
    // Python's json.dumps escapes " \ and the C0 controls. None of these
    // prefixes contains any of them today; escaping anyway means a future
    // checkpoint that does cannot silently produce two different files.
    auto esc = [](const std::string &s) {
      std::string o;
      for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') o += "\\n";
        else if (c == '\r') o += "\\r";
        else if (c == '\t') o += "\\t";
        else o += c;
      }
      return o;
    };
    cj += ",\"prompts\":{";
    for (size_t i = 0; i < keys.size(); ++i) {
      if (i) cj += ",";
      cj += "\"" + esc(keys[i]) + "\":\"" +
            esc(pr.at(keys[i]).as_string()) + "\"";
    }
    cj += "}";
    if (!pr.count("document"))
      throw std::runtime_error("this checkpoint's prompts table has no "
                               "'document' row");
    cj += ",\"prompt_default\":\"document\"";
  }

  // tasks/0074. Key ORDER below mirrors tools/pack_npue.py's insertion order
  // exactly (gemm_layout, then the update() block), because json.dumps
  // preserves it and tools/verify_pack_parity.py compares the two containers
  // byte for byte.
  const int64_t kv_w = kv_heads * head_dim;
  const int64_t qkv_used = hidden + 2 * kv_w;
  const int64_t gran = tile_n * 8;             // n_aie_cols = 8
  const int64_t qkv_n = ((qkv_used + gran - 1) / gran) * gran;
  const Layout glay = gemm_b_layout(tile_k, tile_n);

  cj += ",\"gemm_layout\":\"";
  cj += host_only ? "host" : "pretiled_bf16";
  cj += "\"";
  if (!host_only) {
    cj += ",\"tile_k\":" + std::to_string(tile_k);
    cj += ",\"tile_n\":" + std::to_string(tile_n);
    cj += ",\"mac_s\":8,\"mac_t\":8";
    cj += ",\"gated_ffn\":true";
    cj += ",\"geglu_halves\":\"gate|up\"";
    cj += ",\"qkv_n\":" + std::to_string(qkv_n);
    cj += ",\"qkv_blocks\":{\"q\":[0," + std::to_string(hidden) + "]";
    cj += ",\"k\":[" + std::to_string(hidden) + "," +
          std::to_string(hidden + kv_w) + "]";
    cj += ",\"v\":[" + std::to_string(hidden + kv_w) + "," +
          std::to_string(qkv_used) + "]";
    cj += ",\"pad\":[" + std::to_string(qkv_used) + "," +
          std::to_string(qkv_n) + "]}";
    cj += ",\"fusions\":{\"qkv_fused\":true"
          ",\"qkv_zero_padded_to_tile\":true"
          ",\"transposed_to_kn\":true"
          ",\"qk_scale_folded_into_q\":false"
          ",\"qk_scale_folded_into_q_note\":\"ILLEGAL for this architecture: "
          "q_norm (RMSNorm) runs after q_proj and is scale-invariant, so a "
          "fold into Wq would be annihilated and attention would run unscaled "
          "with no shape error. The scale stays on the scores.\""
          ",\"gemm_operands_bf16\":true"
          ",\"norms_embeddings_dense_fp32\":true"
          ",\"gated_ffn_fused_gate_up\":true"
          ",\"biases_zero_filled\":true}";
  }
  cj += "}";

  Writer w;
  // Zero biases, one per fused operand width. Gemma has no biases anywhere;
  // the runtime's GEMM epilogue adds one unconditionally, so zero-filling is
  // exact and keeps this arch on the same dispatch path as every other model
  // (same reasoning as pack_nomic's).
  auto add_zero_bias = [&](const std::string &name, int64_t n) {
    const std::vector<float> z(static_cast<size_t>(n), 0.0f);
    w.add(name, z.data(), z.size() * sizeof(float), "F32", "bias", {n});
  };
  auto add_f32 = [&](const std::string &name, const Tensor &t,
                     const char *role, const std::vector<int64_t> &shape) {
    w.add(name, t.data, static_cast<size_t>(t.count()) * 4, "F32", role,
          shape);
  };

  const Tensor &embed = get("embed_tokens.weight");
  add_f32("embed_tokens.weight", embed, "embedding", embed.shape);
  const Tensor &normw = get("norm.weight");
  add_f32("norm.weight", normw, "layernorm", normw.shape);

  {
    // Prefer an already-cached table on disk (byte-identical either way it
    // got there -- Python or this generator, tasks/0067's verification
    // confirms the two agree). Otherwise generate it here, in C++, from the
    // checkpoint's own tokenizer.json + config_sentence_transformers.json --
    // and write it out to the same cache path, so a second pack of this
    // model is as fast as the "have" case and the file is inspectable on
    // disk exactly like the Python tool's output always was. This closes
    // the one gap tasks/0066 left open: a fresh clone that fetches
    // EmbeddingGemma had no way to produce gemma_tokenizer.bin without
    // manually running the Python-only build tool.
    const std::string tok_path = model_dir + "/gemma_tokenizer.bin";
    std::ifstream tf(tok_path, std::ios::binary);
    std::vector<uint8_t> tb;
    bool generated = false;
    if (tf.good()) {
      tf.close();
      tb = slurp(tok_path);
    } else {
      tb = generate_gemma_tokenizer_table(
          model_dir + "/tokenizer.json",
          model_dir + "/config_sentence_transformers.json");
      generated = true;
      std::ofstream of(tok_path, std::ios::binary);
      if (!of)
        throw std::runtime_error("cannot write " + tok_path);
      of.write(reinterpret_cast<const char *>(tb.data()),
              static_cast<std::streamsize>(tb.size()));
      if (!of)
        throw std::runtime_error("error writing " + tok_path);
    }
    w.add("tokenizer.gemma_table", tb.data(), tb.size(), "U8", "tokenizer",
          {static_cast<int64_t>(tb.size())});
    if (log) {
      std::ostringstream s;
      if (generated)
        s << "  generated tokenizer.gemma_table (no cached "
             "gemma_tokenizer.bin found)  " << (tb.size() / 1e6) << " MB";
      else
        s << "  tokenizer.gemma_table  " << (tb.size() / 1e6) << " MB";
      log(s.str());
    }
  }

  for (int64_t i = 0; i < L; ++i) {
    const std::string p = "layers." + std::to_string(i) + ".";
    const std::string sa = p + "self_attn.";
    const std::string tag = "layer." + std::to_string(i) + ".";

    if (host_only) {
      add_gemm_b_host(w, tag + "q_proj", get(sa + "q_proj.weight"));
      add_gemm_b_host(w, tag + "k_proj", get(sa + "k_proj.weight"));
      add_gemm_b_host(w, tag + "v_proj", get(sa + "v_proj.weight"));
    } else {
      add_gemm_b_concat_pad(w, tag + "qkv",
                            {&get(sa + "q_proj.weight"),
                             &get(sa + "k_proj.weight"),
                             &get(sa + "v_proj.weight")},
                            qkv_n, tile_k, tile_n, glay.json, glay.hash);
      add_zero_bias(tag + "qkv.bias", qkv_n);
    }
    {
      const Tensor &qn = get(sa + "q_norm.weight");
      add_f32(tag + "q_norm.weight", qn, "layernorm", qn.shape);
    }
    {
      const Tensor &kn = get(sa + "k_norm.weight");
      add_f32(tag + "k_norm.weight", kn, "layernorm", kn.shape);
    }
    if (host_only) {
      add_gemm_b_host(w, tag + "o_proj", get(sa + "o_proj.weight"));
    } else {
      add_gemm_b(w, tag + "attn_out", get(sa + "o_proj.weight"), tile_k,
                 tile_n, glay.json, glay.hash);
      add_zero_bias(tag + "attn_out.bias", hidden);
    }

    for (const char *ln : {"input_layernorm", "post_attention_layernorm",
                           "pre_feedforward_layernorm",
                           "post_feedforward_layernorm"}) {
      const Tensor &t = get(p + ln + ".weight");
      add_f32(tag + ln + ".weight", t, "layernorm", t.shape);
    }

    const std::string mp = p + "mlp.";
    if (host_only) {
      add_gemm_b_host(w, tag + "gate_proj", get(mp + "gate_proj.weight"));
      add_gemm_b_host(w, tag + "up_proj", get(mp + "up_proj.weight"));
      add_gemm_b_host(w, tag + "down_proj", get(mp + "down_proj.weight"));
    } else {
      // config["geglu_halves"] == "gate|up": the FIRST half gets the GELU.
      add_gemm_b_concat_pad(w, tag + "ffn_up",
                            {&get(mp + "gate_proj.weight"),
                             &get(mp + "up_proj.weight")},
                            2 * inter, tile_k, tile_n, glay.json, glay.hash);
      add_zero_bias(tag + "ffn_up.bias", 2 * inter);
      add_gemm_b(w, tag + "ffn_down", get(mp + "down_proj.weight"), tile_k,
                 tile_n, glay.json, glay.hash);
      add_zero_bias(tag + "ffn_down.bias", hidden);
    }
  }

  add_gemm_b_host(w, "dense2.weight", d2w);
  add_gemm_b_host(w, "dense3.weight", d3w);

  w.write(out, cj, /*arch=*/1);
  if (log) {
    std::ostringstream s;
    s << "\n  tensors    : " << w.count()
      << "\n  data       : " << (w.data_bytes() / 1e6) << " MB"
      << "\n  source     : " << sha.substr(0, 16) << "...";
    log(s.str());
  }
}

// nomic's gated FFN: fc11 (untouched "up") and fc12 (SiLU "gate") are two
// SEPARATE [inter,hidden] checkpoint tensors (nn.Linear [out,in]), fused
// into ONE [hidden, 2*inter] ffn_up operand along the N axis -- mirrors
// tools/pack_npue.py's pack_nomic():
//   up = fc11.weight.T; gate = fc12.weight.T; ffn_up = concat([up,gate],1)
// add_gemm_b() above only takes one source tensor, so this transposes both
// into the SAME [K,N] buffer at their own column offset, then tiles once --
// the same "manual assembly, then tile_b, then w.add" shape prepare_model()
// uses inline for BERT's 3-way qkv fusion, just for 2 sources instead of 3.
static void add_gemm_b_concat2(Writer &w, const std::string &name,
                               const Tensor &a, const Tensor &b,
                               int64_t tk, int64_t tn,
                               const std::string &layout_json,
                               const std::string &layout_hash) {
  const int64_t K = a.cols();               // both share `hidden` as `in`
  if (b.cols() != K)
    throw std::runtime_error(name + ": fc11/fc12 disagree on `in` dim");
  const int64_t Na = a.rows(), Nb = b.rows();
  const int64_t N = Na + Nb;
  std::vector<float> m(static_cast<size_t>(K) * N);
  const float *sa = a.f32(), *sb = b.f32();
  for (int64_t r = 0; r < K; ++r) {
    for (int64_t c = 0; c < Na; ++c) m[r * N + c] = sa[c * K + r];
    for (int64_t c = 0; c < Nb; ++c) m[r * N + Na + c] = sb[c * K + r];
  }
  const auto tiled = tile_b(m.data(), K, N, tk, tn);
  w.add(name, tiled.data(), tiled.size() * 2, "BF16", "gemm_b", {K, N},
        layout_json, layout_hash);
}

// Python's str.splitlines() count for plain-LF ASCII text (vocab.txt has no
// CR and no exotic Unicode line separators): one line per '\n', plus a final
// unterminated line if the file does not end with one. Matches
// tools/pack_npue.py's `len(vocab_path.read_bytes().decode("utf-8")
// .splitlines())` for this specific input shape -- not a general splitlines
// reimplementation.
static int64_t count_lines(const std::vector<uint8_t> &b) {
  if (b.empty()) return 0;
  int64_t n = 0;
  for (uint8_t c : b) if (c == '\n') ++n;
  if (b.back() != '\n') ++n;
  return n;
}

// arch=2 mirror of tools/pack_npue.py's pack_nomic() (tasks/0069, 0070,
// 0071). See npue_pack.hpp for the departures from prepare_model() above.
// Every architectural fact asserted below was settled EMPIRICALLY against
// the real checkpoint in tasks/0068 -- this function only implements that
// already-settled architecture and asserts the config facts it depends on,
// so a checkpoint that silently changed underneath it refuses to pack
// rather than packing wrong (same discipline as pack_nomic()'s docstring).
void prepare_model_nomic(const std::string &model_dir,
                         const std::string &pooling,
                         const std::string &source_repo,
                         const std::string &out,
                         const std::string &layout_json,
                         const std::string &layout_hash,
                         int64_t tile_k, int64_t tile_n, int64_t max_seq,
                         void (*log)(const std::string &)) {
  const auto st_buf = slurp(model_dir + "/model.safetensors");
  const auto src = read_safetensors(st_buf);
  Sha256 sh;
  sh.update(st_buf.data(), st_buf.size());
  const std::string sha = sh.hex();

  auto get = [&](const std::string &n) -> const Tensor & {
    auto it = src.find(n);
    if (it == src.end())
      throw std::runtime_error("checkpoint has no tensor '" + n + "'");
    if (it->second.dtype != "F32")
      throw std::runtime_error("checkpoint tensor '" + n + "' is " +
                               it->second.dtype + "; this packer reads F32");
    return it->second;
  };

  const auto cfg_buf = slurp(model_dir + "/config.json");
  const std::string cfg(reinterpret_cast<const char *>(cfg_buf.data()),
                        cfg_buf.size());
  auto cfg_int = [&](const char *key) -> int64_t {
    const std::string k = std::string("\"") + key + "\"";
    const size_t i = cfg.find(k);
    if (i == std::string::npos)
      throw std::runtime_error(std::string("config.json has no ") + key);
    return std::stoll(cfg.substr(cfg.find(':', i) + 1));
  };
  auto cfg_str = [&](const char *key) -> std::string {
    const std::string k = std::string("\"") + key + "\"";
    const size_t i = cfg.find(k);
    if (i == std::string::npos)
      throw std::runtime_error(std::string("config.json has no ") + key);
    const size_t c = cfg.find(':', i) + 1;
    const size_t q1 = cfg.find('"', c);
    const size_t q2 = cfg.find('"', q1 + 1);
    return cfg.substr(q1 + 1, q2 - q1 - 1);
  };
  // The value's EXACT literal text, verbatim -- see prepare_model_gemma()'s
  // cfg_raw for why this is safe rather than reparsing and reformatting:
  // both "layer_norm_eps" (1e-12) and "rotary_emb_base" (1000) round-trip
  // unchanged through Python's json module, checked directly against this
  // checkpoint's config.json.
  auto cfg_raw = [&](const char *key) -> std::string {
    const std::string k = std::string("\"") + key + "\"";
    const size_t i = cfg.find(k);
    if (i == std::string::npos)
      throw std::runtime_error(std::string("config.json has no ") + key);
    size_t c = cfg.find(':', i) + 1;
    while (c < cfg.size() && std::isspace(static_cast<unsigned char>(cfg[c])))
      ++c;
    size_t e = c;
    while (e < cfg.size() && cfg[e] != ',' && cfg[e] != '}' &&
          cfg[e] != '\n' && cfg[e] != '\r')
      ++e;
    while (e > c && std::isspace(static_cast<unsigned char>(cfg[e - 1])))
      --e;
    return cfg.substr(c, e - c);
  };
  auto cfg_is_false = [&](const char *key) { return cfg_raw(key) == "false"; };

  const std::string model_type = cfg_str("model_type");
  const int64_t L = cfg_int("num_hidden_layers");
  const int64_t H = cfg_int("num_attention_heads");
  const int64_t hidden = cfg_int("hidden_size");
  const int64_t head_dim = cfg_int("head_dim");
  const int64_t inter = cfg_int("intermediate_size");
  const int64_t vocab_size = cfg_int("vocab_size");
  const int64_t type_vocab = cfg_int("type_vocab_size");

  if (hidden != H * head_dim)
    throw std::runtime_error("hidden=" + std::to_string(hidden) +
                             " != num_heads*head_dim");
  if (head_dim % 2)
    throw std::runtime_error("head_dim is odd -- RoPE cannot half-split it "
                             "into rotation pairs");

  // rope_theta: ASSERT, never default -- tasks/0068 measured a wrong theta
  // subtle enough to slip past a loose gate. Compared as a NUMBER (not text)
  // so "1000" and "1000.0" are both accepted the way Python's `!= 1000`
  // comparison is type-agnostic; the literal text used in the OUTPUT config
  // below is still cfg_raw's verbatim copy, whatever its exact spelling.
  const std::string theta_raw = cfg_raw("rotary_emb_base");
  if (std::stod(theta_raw) != 1000.0)
    throw std::runtime_error("rotary_emb_base=" + theta_raw + ", expected "
                             "1000 -- refusing to pack against an unverified "
                             "RoPE base");

  // layer_norm_epsilon and layer_norm_eps are two keys for the same value in
  // this checkpoint's config.json -- read one, assert they agree.
  const std::string eps_a = cfg_raw("layer_norm_epsilon");
  const std::string eps_b = cfg_raw("layer_norm_eps");
  if (std::stod(eps_a) != std::stod(eps_b))
    throw std::runtime_error("layer_norm_epsilon (" + eps_a +
                             ") != layer_norm_eps (" + eps_b + ")");

  for (const char *flag : {"qkv_proj_bias", "mlp_fc1_bias", "mlp_fc2_bias"})
    if (!cfg_is_false(flag))
      throw std::runtime_error(std::string(flag) + "=" + cfg_raw(flag) +
                               ", expected false -- this packer zero-fills "
                               "every bias on the assumption nomic has none");
  if (!cfg_is_false("prenorm"))
    throw std::runtime_error("prenorm=" + cfg_raw("prenorm") + ", expected "
                             "false (post-LN block order)");
  if (cfg_str("activation_function") != "swiglu" ||
      cfg_str("hidden_act") != "silu")
    throw std::runtime_error("expected activation_function=swiglu, "
                             "hidden_act=silu");
  if (!cfg_is_false("rotary_emb_interleaved"))
    throw std::runtime_error("rotary_emb_interleaved=" +
                             cfg_raw("rotary_emb_interleaved") + " -- this "
                             "packer/runtime assumes NeoX-style RoPE");
  if (std::stod(cfg_raw("rotary_emb_fraction")) != 1.0)
    throw std::runtime_error("rotary_emb_fraction=" +
                             cfg_raw("rotary_emb_fraction") + ", expected "
                             "1.0 (whole head rotated)");

  const float scale = static_cast<float>(1.0 / std::sqrt(
      static_cast<double>(head_dim)));

  if (log) {
    std::ostringstream s;
    s << "packing " << model_dir << " -> " << out
      << "  (arch=nomic_bert_rope_swiglu)\n"
      << "  hidden=" << hidden << " heads=" << H << " head_dim=" << head_dim
      << " layers=" << L << " inter=" << inter << " rope_theta=" << theta_raw;
    log(s.str());
  }

  // How many embedding rows the tokenizer can actually reach -- nomic pads
  // vocab_size up to a multiple of 64, and those extra rows are not zero.
  int64_t n_reachable = 0;
  std::vector<uint8_t> vb;
  {
    std::ifstream vf(model_dir + "/vocab.txt", std::ios::binary);
    if (vf.good()) {
      vf.close();
      vb = slurp(model_dir + "/vocab.txt");
      n_reachable = count_lines(vb);
    }
  }

  // Exact key order of tools/pack_npue.py's pack_nomic() config dict --
  // json.dumps(..., separators=(",", ":")) preserves insertion order, and
  // this must match it byte for byte (tools/verify_pack_parity.py's gate).
  std::string cj;
  cj += "{\"arch\":\"nomic_bert_rope_swiglu\"";
  cj += ",\"model_type\":\"" + model_type + "\"";
  cj += ",\"source_repo\":\"" + source_repo + "\"";
  cj += ",\"source_sha256\":\"" + sha + "\"";
  cj += ",\"num_layers\":" + std::to_string(L);
  cj += ",\"num_heads\":" + std::to_string(H);
  cj += ",\"hidden\":" + std::to_string(hidden);
  cj += ",\"head_dim\":" + std::to_string(head_dim);
  cj += ",\"intermediate\":" + std::to_string(inter);
  cj += ",\"layer_norm_eps\":" + eps_b;
  cj += ",\"vocab_size\":" + std::to_string(vocab_size);
  cj += ",\"max_seq_len\":" + std::to_string(max_seq);
  cj += ",\"pooling\":\"" + pooling + "\",\"l2_normalize\":true";
  cj += ",\"activation\":\"silu\",\"gated_ffn\":true";
  cj += ",\"swiglu_halves\":\"fc11_up|fc12_gate\"";
  cj += ",\"position_embedding_type\":\"rope\"";
  cj += ",\"rope_theta\":" + theta_raw;
  cj += ",\"attention_bias\":false,\"mlp_bias\":false";
  cj += ",\"tile_k\":" + std::to_string(tile_k) +
        ",\"tile_n\":" + std::to_string(tile_n) +
        ",\"mac_s\":" + std::to_string(kMacS) +
        ",\"mac_t\":" + std::to_string(kMacT);
  cj += ",\"prompts\":{\"search_document\":\"search_document: \","
        "\"search_query\":\"search_query: \","
        "\"clustering\":\"clustering: \","
        "\"classification\":\"classification: \"}";
  cj += ",\"prompt_default\":\"search_document\"";
  cj += ",\"prompts_source\":\"npuembeddings, NOT from the checkpoint -- "
        "config_sentence_transformers.json carries no 'prompts' dict for "
        "this checkpoint, so presenting this table as the model's own "
        "would be a lie in a file other tools read. Same precedent as "
        "tools/gen_gemma_tokenizer_table.py:63-77.\"";
  cj += ",\"l2_normalize_note\":\"sentence-transformers does NOT "
        "L2-normalize this model (measured output norm 20.93, tasks/0068 "
        "sec 5b) -- l2_normalize:true here matches THIS RUNTIME's own "
        "hardcoded behaviour (main.cpp g_l2_normalize) and nomic's own "
        "documented usage (F.normalize), not sentence-transformers' "
        "default pipeline for this particular model.\"";
  cj += ",\"fusions\":{\"qkv_fused\":true,\"transposed_to_kn\":true,"
        "\"qk_scale_folded_into_q\":true,\"gemm_operands_bf16\":true,"
        "\"biases_and_layernorm_fp32\":true,"
        "\"gated_ffn_fused_fc11_fc12\":true,"
        "\"position_embeddings_zeroed_rope_instead\":true}";
  cj += ",\"not_implemented\":[\"Matryoshka truncation (layer_norm(768) -> "
        "slice -> normalize is a different post-processing chain, not "
        "just a shorter vector)\",\"vocab rows " +
        std::to_string(n_reachable) + "-" + std::to_string(vocab_size - 1) +
        " are pad_vocab_size_multiple padding: non-zero but unreachable "
        "from the tokenizer (max id " + std::to_string(n_reachable - 1) +
        "), packed only so vocab_size and the tensor agree\"]}";

  Writer w;
  auto add_f32 = [&](const std::string &name, const Tensor &t,
                     const char *role, const std::vector<int64_t> &shape) {
    w.add(name, t.data, static_cast<size_t>(t.count()) * 4, "F32", role,
          shape);
  };
  auto add_zero_bias = [&](const std::string &name, int64_t n) {
    std::vector<float> z(static_cast<size_t>(n), 0.f);
    w.add(name, z.data(), z.size() * 4, "F32", "bias", {n});
  };

  // -- embeddings: SAME order as prepare_model() above, including the odd
  // ln.weight -> tokenizer.vocab -> ln.bias interleaving, which is
  // load-bearing for byte parity with tools/pack_npue.py. -----------------
  add_f32("embeddings.word", get("embeddings.word_embeddings.weight"),
          "embedding", {vocab_size, hidden});
  // nomic has NO position table -- RoPE is computed inside attention
  // instead. Zero-filled rather than omitted: Encoder::stage_all() and the
  // --embed path both dereference "embeddings.position" unconditionally, so
  // a zero tensor of the right shape is exact (adds nothing) and keeps that
  // read path untouched.
  {
    std::vector<float> zpos(static_cast<size_t>(max_seq) * hidden, 0.f);
    w.add("embeddings.position", zpos.data(), zpos.size() * 4, "F32",
          "embedding", {max_seq, hidden});
  }
  add_f32("embeddings.token_type", get("embeddings.token_type_embeddings.weight"),
          "embedding", {type_vocab, hidden});
  // emb_ln lives at the TOP LEVEL upstream (not embeddings.LayerNorm, as in
  // BERT).
  add_f32("embeddings.ln.weight", get("emb_ln.weight"), "layernorm",
          {hidden});
  if (!vb.empty()) {
    w.add("tokenizer.vocab", vb.data(), vb.size(), "U8", "tokenizer",
          {static_cast<int64_t>(vb.size())});
    if (log) {
      std::ostringstream s;
      s << "  tokenizer.vocab   " << (vb.size() / 1024.0) << " KB";
      log(s.str());
    }
  } else if (log) {
    log("  WARNING: " + model_dir + "/vocab.txt not found -- .npue will "
                                    "have no vocab");
  }
  add_f32("embeddings.ln.bias", get("emb_ln.bias"), "layernorm", {hidden});

  for (int64_t i = 0; i < L; ++i) {
    const std::string p = "encoder.layers." + std::to_string(i) + ".";
    const std::string attn = p + "attn.";
    const std::string mp = p + "mlp.";
    const std::string tag = "layer." + std::to_string(i) + ".";

    // Fused upstream already: Wqkv is [2304,768] three-major
    // [Q(768)|K(768)|V(768)] -- tasks/0068 sec 5 Wqkv row-order check.
    // 1/sqrt(head_dim) folded into the Q block ONLY (the first `hidden`
    // columns of the transposed [768,2304] operand) -- legal here because
    // RoPE is linear in q: rope(s*q) = s*rope(q), so folding the scale
    // before the GEMM and before RoPE is exact (tools/verify_npue_nomic.py
    // check E). No qkv bias exists to fold.
    add_gemm_b(w, tag + "qkv", get(attn + "Wqkv.weight"), tile_k, tile_n,
               layout_json, layout_hash, scale, hidden);
    add_zero_bias(tag + "qkv.bias", 3 * hidden);

    add_gemm_b(w, tag + "attn_out", get(attn + "out_proj.weight"), tile_k,
               tile_n, layout_json, layout_hash);
    add_zero_bias(tag + "attn_out.bias", hidden);
    add_f32(tag + "ln1.weight", get(p + "norm1.weight"), "layernorm",
            {hidden});
    add_f32(tag + "ln1.bias", get(p + "norm1.bias"), "layernorm", {hidden});

    // Gated ffn_up: [fc11 (up, untouched) | fc12 (gate, gets SiLU)] fused
    // along N. Runtime computes out = lo * silu(hi) -- see
    // config["swiglu_halves"]. ONE GEMM, so the array still sees four GEMMs
    // per layer, not five.
    add_gemm_b_concat2(w, tag + "ffn_up", get(mp + "fc11.weight"),
                       get(mp + "fc12.weight"), tile_k, tile_n, layout_json,
                       layout_hash);
    add_zero_bias(tag + "ffn_up.bias", 2 * inter);

    add_gemm_b(w, tag + "ffn_down", get(mp + "fc2.weight"), tile_k, tile_n,
               layout_json, layout_hash);
    add_zero_bias(tag + "ffn_down.bias", hidden);
    add_f32(tag + "ln2.weight", get(p + "norm2.weight"), "layernorm",
            {hidden});
    add_f32(tag + "ln2.bias", get(p + "norm2.bias"), "layernorm", {hidden});
  }

  w.write(out, cj, /*arch=*/2);
  if (log) {
    std::ostringstream s;
    s << "\n  tensors    : " << w.count()
      << "\n  data       : " << (w.data_bytes() / 1e6) << " MB"
      << "\n  source     : " << sha.substr(0, 16) << "...";
    log(s.str());
  }
}

// Python's repr(float) -- the exact text json.dumps writes for a double --
// reimplemented for the one config field this packer must FORMAT rather than
// copy verbatim: "rope_inv_freq", whose 32 values are COMPUTED here (the
// same float32 arithmetic as pack_gte(), verified bit-for-bit against the
// Python-packed container in tasks/0138) and so have no source text to copy
// the way cfg_raw copies "1e-12" or "8.0".
//
// Both sides print the SHORTEST decimal string that round-trips the double
// (CPython: format_float_short mode 'r'; MSVC: std::to_chars, [charconv]'s
// "minimal representation" guarantee) -- so the digit sequences agree by
// construction, and only the DRESSING differs. CPython's rules, mirrored
// here: fixed notation when the decimal point lands in (-4, 16] digits from
// the front, else scientific with a sign and at least two exponent digits;
// a fixed integer value gets a trailing ".0"; a single-digit scientific
// mantissa gets NO ".0" (repr(1e-05) == '1e-05').
static std::string py_double_repr(double v) {
  char buf[64];
  const auto res =
      std::to_chars(buf, buf + sizeof buf, v, std::chars_format::scientific);
  std::string s(buf, res.ptr);
  std::string sign;
  if (!s.empty() && s[0] == '-') { sign = "-"; s.erase(0, 1); }
  const size_t ep = s.find('e');
  if (ep == std::string::npos)
    throw std::runtime_error("py_double_repr: non-finite value");  // inf/nan
  std::string digits = s.substr(0, ep);
  const size_t dot = digits.find('.');
  if (dot != std::string::npos) digits.erase(dot, 1);
  const int exp10 = std::stoi(s.substr(ep + 1));
  const int decpt = exp10 + 1;      // digits before the decimal point
  if (decpt > -4 && decpt <= 16) {
    std::string o = sign;
    if (decpt <= 0) {
      o += "0.";
      o.append(static_cast<size_t>(-decpt), '0');
      o += digits;
    } else if (static_cast<size_t>(decpt) >= digits.size()) {
      o += digits;
      o.append(static_cast<size_t>(decpt) - digits.size(), '0');
      o += ".0";
    } else {
      o += digits.substr(0, static_cast<size_t>(decpt)) + "." +
           digits.substr(static_cast<size_t>(decpt));
    }
    return o;
  }
  std::string o = sign + digits.substr(0, 1);
  if (digits.size() > 1) o += "." + digits.substr(1);
  o += 'e';
  o += exp10 < 0 ? '-' : '+';
  const int ae = exp10 < 0 ? -exp10 : exp10;
  if (ae < 10) o += '0';
  o += std::to_string(ae);
  return o;
}

// arch=3 mirror of tools/pack_npue.py's pack_gte() (tasks/0135, 0138). See
// npue_pack.hpp for the departures from the nomic shape it otherwise
// mirrors. Every architectural fact asserted below was settled EMPIRICALLY
// in tasks/0134 (per-layer probe against the repaired fp32 reference, with
// negative controls on the wrong-theta and wrong-half readings) -- this
// function only implements that already-settled architecture, and refuses a
// checkpoint that silently changed underneath it rather than packing wrong.
void prepare_model_gte(const std::string &model_dir,
                       const std::string &pooling,
                       const std::string &source_repo,
                       const std::string &out,
                       const std::string &layout_json,
                       const std::string &layout_hash,
                       int64_t tile_k, int64_t tile_n, int64_t max_seq,
                       void (*log)(const std::string &)) {
  const auto st_buf = slurp(model_dir + "/model.safetensors");
  const auto src = read_safetensors(st_buf);   // widens this checkpoint's
                                               // F16 to F32 at read time
  Sha256 sh;
  sh.update(st_buf.data(), st_buf.size());
  const std::string sha = sh.hex();

  // pack_gte() strips the leading "new." from every checkpoint key (only
  // classifier.weight/classifier.bias lack it, and those are deliberately
  // not packed). Mirrored in the lookup rather than by rebuilding the map.
  auto get = [&](const std::string &n) -> const Tensor & {
    auto it = src.find("new." + n);
    if (it == src.end()) it = src.find(n);
    if (it == src.end())
      throw std::runtime_error("checkpoint has no tensor '" + n + "'");
    if (it->second.dtype != "F32")
      throw std::runtime_error("checkpoint tensor '" + n + "' is " +
                               it->second.dtype + "; this packer reads F32 "
                               "(F16/BF16 are widened at read time)");
    return it->second;
  };

  const auto cfg_buf = slurp(model_dir + "/config.json");
  const std::string cfg(reinterpret_cast<const char *>(cfg_buf.data()),
                        cfg_buf.size());
  auto find_key = [&](const char *key) -> size_t {
    return cfg.find(std::string("\"") + key + "\"");
  };
  auto cfg_int = [&](const char *key) -> int64_t {
    const size_t i = find_key(key);
    if (i == std::string::npos)
      throw std::runtime_error(std::string("config.json has no ") + key);
    return std::stoll(cfg.substr(cfg.find(':', i) + 1));
  };
  auto cfg_str = [&](const char *key) -> std::string {
    const size_t i = find_key(key);
    if (i == std::string::npos)
      throw std::runtime_error(std::string("config.json has no ") + key);
    const size_t c = cfg.find(':', i) + 1;
    const size_t q1 = cfg.find('"', c);
    const size_t q2 = cfg.find('"', q1 + 1);
    return cfg.substr(q1 + 1, q2 - q1 - 1);
  };
  // The value's EXACT literal text, verbatim -- see prepare_model_gemma()'s
  // cfg_raw for why this is safe rather than reparsing and reformatting:
  // every numeric field this copies into the output ("1e-12", "20000",
  // "8.0", "250048") is already Python's canonical shortest-round-trip form,
  // checked directly against this checkpoint's config.json.
  auto cfg_raw = [&](const char *key) -> std::string {
    const size_t i = find_key(key);
    if (i == std::string::npos)
      throw std::runtime_error(std::string("config.json has no ") + key);
    size_t c = cfg.find(':', i) + 1;
    while (c < cfg.size() && std::isspace(static_cast<unsigned char>(cfg[c])))
      ++c;
    size_t e = c;
    while (e < cfg.size() && cfg[e] != ',' && cfg[e] != '}' &&
          cfg[e] != '\n' && cfg[e] != '\r')
      ++e;
    while (e > c && std::isspace(static_cast<unsigned char>(cfg[e - 1])))
      --e;
    return cfg.substr(c, e - c);
  };

  const std::string model_type = cfg_str("model_type");
  const int64_t L = cfg_int("num_hidden_layers");
  const int64_t H = cfg_int("num_attention_heads");
  const int64_t hidden = cfg_int("hidden_size");
  const int64_t inter = cfg_int("intermediate_size");
  const int64_t vocab_size = cfg_int("vocab_size");
  const int64_t type_vocab = cfg_int("type_vocab_size");
  const int64_t head_dim = hidden / H;    // gte's config carries no head_dim

  // -- fail-closed assertions: every fact tasks/0134's probe settled, in
  // pack_gte()'s order. ----------------------------------------------------
  if (model_type != "new")
    throw std::runtime_error("model_type='" + model_type +
                             "', expected 'new'");
  if (cfg_str("hidden_act") != "gelu")
    throw std::runtime_error("hidden_act='" + cfg_str("hidden_act") +
                             "', expected 'gelu' (exact erf -- tasks/0134)");
  if (cfg_str("position_embedding_type") != "rope")
    throw std::runtime_error("position_embedding_type='" +
                             cfg_str("position_embedding_type") +
                             "', expected 'rope'");
  // rope_theta / rope_scaling: read SCOPED to the rope_scaling object --
  // "type" as a bare key search would also match rope_scaling's own row in
  // some other object, and this file has several *_type keys.
  const std::string theta_raw = cfg_raw("rope_theta");
  std::string rs_type, rs_factor_raw;
  bool rs_has_mixed_b = false;
  {
    const size_t i = find_key("rope_scaling");
    if (i == std::string::npos)
      throw std::runtime_error("config.json has no rope_scaling");
    const size_t ob = cfg.find('{', i);
    const size_t cb = cfg.find('}', ob);
    if (ob == std::string::npos || cb == std::string::npos)
      throw std::runtime_error("config.json: rope_scaling is not an object");
    const std::string rs = cfg.substr(ob, cb - ob + 1);
    auto rs_find = [&](const char *key) -> size_t {
      return rs.find(std::string("\"") + key + "\"");
    };
    const size_t ti = rs_find("type");
    if (ti != std::string::npos) {
      const size_t q1 = rs.find('"', rs.find(':', ti) + 1);
      const size_t q2 = rs.find('"', q1 + 1);
      rs_type = rs.substr(q1 + 1, q2 - q1 - 1);
    }
    const size_t fi = rs_find("factor");
    if (fi != std::string::npos) {
      size_t c = rs.find(':', fi) + 1;
      while (c < rs.size() && std::isspace(static_cast<unsigned char>(rs[c])))
        ++c;
      size_t e = c;
      while (e < rs.size() && rs[e] != ',' && rs[e] != '}' &&
             rs[e] != '\n' && rs[e] != '\r')
        ++e;
      while (e > c && std::isspace(static_cast<unsigned char>(rs[e - 1])))
        --e;
      rs_factor_raw = rs.substr(c, e - c);
    }
    const size_t mi = rs_find("mixed_b");
    if (mi != std::string::npos) {
      size_t c = rs.find(':', mi) + 1;
      while (c < rs.size() && std::isspace(static_cast<unsigned char>(rs[c])))
        ++c;
      // Python: `rs.get("mixed_b") is not None` -- only an explicit null
      // (or the key's absence) passes.
      rs_has_mixed_b = rs.compare(c, 4, "null") != 0;
    }
  }
  if (std::stod(theta_raw) != 20000.0 || rs_type != "ntk" ||
      rs_factor_raw.empty() || std::stod(rs_factor_raw) != 8.0 ||
      rs_has_mixed_b)
    throw std::runtime_error(
        "rope_theta=" + theta_raw + ", rope_scaling{type='" + rs_type +
        "', factor=" + rs_factor_raw + "} -- expected 20000 / ntk / 8.0 / "
        "mixed_b None. The baked inv_freq below is derived for exactly that "
        "configuration (tasks/0134); refusing to pack an unverified RoPE "
        "against it");
  if (type_vocab != 1)
    throw std::runtime_error("type_vocab_size=" + std::to_string(type_vocab) +
                             ", expected 1");
  if (find_key("layer_norm_type") != std::string::npos &&
      cfg_str("layer_norm_type") != "layer_norm")
    throw std::runtime_error("layer_norm_type='" +
                             cfg_str("layer_norm_type") + "'");
  if (find_key("logn_attention_scale") != std::string::npos) {
    const std::string v = cfg_raw("logn_attention_scale");
    if (v != "false" && v != "null" && std::stod("0" + v) != 0.0)
      throw std::runtime_error("logn_attention_scale is set -- tasks/0134's "
                               "probe validated the plain 1/sqrt(head_dim) "
                               "scale only");
  }
  if (find_key("pack_qkv") == std::string::npos ||
      cfg_raw("pack_qkv") != "true")
    throw std::runtime_error("pack_qkv is false -- this packer reads the "
                             "fused qkv_proj tensor");
  const std::string eps_raw = cfg_raw("layer_norm_eps");

  // FLOAT, not double, exactly as prepare_model()'s scale is -- but note
  // that for head_dim 64 the value is 0.125, a power of two, so the fold
  // below is EXACT regardless (no rounding anywhere in x * 0.125f).
  const float scale = static_cast<float>(1.0 / std::sqrt(
      static_cast<double>(head_dim)));

  // The NTK frequency set, float32 arithmetic exactly as pack_gte() (and
  // torch) compute it: inv_freq_i = (theta*factor)^(-2i/head_dim), then
  // divided by factor^(2/head_dim). MSVC's powf reproduces numpy's float32
  // power bit-for-bit on all 32 values -- verified against the
  // Python-packed container in tasks/0138, which is what licenses computing
  // rather than transcribing them.
  const int64_t half = head_dim / 2;
  std::vector<float> inv_freq(static_cast<size_t>(half));
  {
    const float tf = static_cast<float>(std::stod(theta_raw) *
                                        std::stod(rs_factor_raw)); // 160000
    const float corr = std::pow(static_cast<float>(std::stod(rs_factor_raw)),
                                2.0f / static_cast<float>(head_dim));
    for (int64_t j = 0; j < half; ++j) {
      const float e = static_cast<float>(2 * j) /
                      static_cast<float>(head_dim);
      inv_freq[static_cast<size_t>(j)] = (1.0f / std::pow(tf, e)) / corr;
    }
  }

  // The XLMRTOK1 tokenizer blob: prefer the cached file (byte-identical
  // either way it got there -- tasks/0133's sha256 identity), else generate
  // it here in C++ and write it back to the same cache path, exactly as
  // prepare_model_gemma() does for its own table.
  const std::string tok_path = model_dir + "/xlmr_tokenizer.bin";
  std::vector<uint8_t> tb;
  bool tok_generated = false;
  {
    std::ifstream tf(tok_path, std::ios::binary);
    if (tf.good()) {
      tf.close();
      tb = slurp(tok_path);
    } else {
      tb = generate_xlmr_tokenizer_table(model_dir + "/tokenizer.json");
      tok_generated = true;
      std::ofstream of(tok_path, std::ios::binary);
      if (!of) throw std::runtime_error("cannot write " + tok_path);
      of.write(reinterpret_cast<const char *>(tb.data()),
               static_cast<std::streamsize>(tb.size()));
      if (!of) throw std::runtime_error("error writing " + tok_path);
    }
  }

  if (log) {
    std::ostringstream s;
    s << "packing " << model_dir << " -> " << out
      << "  (arch=gte_new_rope_geglu)\n"
      << "  hidden=" << hidden << " heads=" << H << " head_dim=" << head_dim
      << " layers=" << L << " inter=" << inter
      << " rope=ntk(20000 x 8.0, " << half << " baked inv_freq)";
    log(s.str());
  }

  // Exact key order of tools/pack_npue.py's pack_gte() config dict --
  // json.dumps(..., separators=(",", ":")) preserves insertion order, and
  // this must match it byte for byte (tools/verify_pack_parity.py's gate,
  // held for arch=3 in tasks/0138).
  std::string cj;
  cj += "{\"arch\":\"gte_new_rope_geglu\"";
  cj += ",\"a_dtype\":\"bf16\"";
  cj += ",\"model_type\":\"" + model_type + "\"";
  cj += ",\"source_repo\":\"" + source_repo + "\"";
  cj += ",\"source_sha256\":\"" + sha + "\"";
  cj += ",\"num_layers\":" + std::to_string(L);
  cj += ",\"num_heads\":" + std::to_string(H);
  cj += ",\"hidden\":" + std::to_string(hidden);
  cj += ",\"head_dim\":" + std::to_string(head_dim);
  cj += ",\"intermediate\":" + std::to_string(inter);
  cj += ",\"layer_norm_eps\":" + eps_raw;
  cj += ",\"vocab_size\":" + std::to_string(vocab_size);
  cj += ",\"max_seq_len\":" + std::to_string(max_seq);
  cj += ",\"pooling\":\"" + pooling + "\",\"l2_normalize\":true";
  cj += ",\"l2_normalize_note\":\"genuinely the checkpoint's own: "
        "modules.json lists a 2_Normalize module (unlike nomic, where true "
        "records this runtime's behaviour).\"";
  cj += ",\"activation\":\"gelu\",\"gated_ffn\":true";
  cj += ",\"glu_halves\":\"up_first|gate_second -- runtime computes "
        "lo * gelu(hi), same half order as nomic's fc11_up|fc12_gate with "
        "GELU for SiLU\"";
  cj += ",\"position_embedding_type\":\"rope\"";
  cj += ",\"rope_theta\":" + theta_raw;
  cj += ",\"rope_scaling\":{\"type\":\"" + rs_type + "\",\"factor\":" +
        rs_factor_raw + "}";
  cj += ",\"rope_inv_freq\":[";
  for (int64_t j = 0; j < half; ++j) {
    if (j) cj += ",";
    cj += py_double_repr(static_cast<double>(inv_freq[static_cast<size_t>(j)]));
  }
  cj += "]";
  cj += ",\"rope_note\":\"rope_inv_freq IS the model -- inv_freq_i = "
        "160000^(-i/32) / 8^(1/32), NOT expressible as any single theta "
        "(tasks/0134, verified bit-for-bit). rope_theta/rope_scaling above "
        "are provenance only; a consumer that derives frequencies from "
        "rope_theta alone is wrong by 1.9e-02 relfro at layer 0.\"";
  cj += ",\"attention_bias\":true";
  cj += ",\"mlp_bias\":\"down_only -- up_gate_proj is genuinely bias-free\"";
  cj += ",\"tile_k\":" + std::to_string(tile_k) +
        ",\"tile_n\":" + std::to_string(tile_n) +
        ",\"mac_s\":" + std::to_string(kMacS) +
        ",\"mac_t\":" + std::to_string(kMacT);
  cj += ",\"fusions\":{\"qkv_fused\":true,\"transposed_to_kn\":true,"
        "\"qk_scale_folded_into_q\":true,"
        "\"qk_scale_folded_into_q_bias\":true,"
        "\"gemm_operands_bf16\":true,"
        "\"biases_and_layernorm_fp32\":true,"
        "\"gated_ffn_fused_upstream\":true,"
        "\"position_embeddings_zeroed_rope_instead\":true}";
  cj += ",\"not_implemented\":[\"int8 datapath (calibrate_smoothing has no "
        "'gte' oracle)\",\"vocab rows 250002-" +
        std::to_string(vocab_size - 1) +
        " are padding: unreachable from the tokenizer, packed only so "
        "vocab_size and the tensor agree\",\"classifier.weight/"
        "classifier.bias (a task head this encoder never runs) are "
        "deliberately NOT packed\"]}";

  Writer w;
  auto add_f32 = [&](const std::string &name, const Tensor &t,
                     const char *role, const std::vector<int64_t> &shape) {
    w.add(name, t.data, static_cast<size_t>(t.count()) * 4, "F32", role,
          shape);
  };

  // -- embeddings: SAME order as arch=0/2, including the ln.weight ->
  // tokenizer -> ln.bias interleaving (load-bearing for byte parity). ------
  add_f32("embeddings.word", get("embeddings.word_embeddings.weight"),
          "embedding", {vocab_size, hidden});
  {
    // No position table -- RoPE instead. Zero-filled, not omitted: the same
    // reasoning (and bytes) as prepare_model_nomic() above.
    std::vector<float> zpos(static_cast<size_t>(max_seq) * hidden, 0.f);
    w.add("embeddings.position", zpos.data(), zpos.size() * 4, "F32",
          "embedding", {max_seq, hidden});
  }
  add_f32("embeddings.token_type",
          get("embeddings.token_type_embeddings.weight"), "embedding",
          {type_vocab, hidden});
  add_f32("embeddings.ln.weight", get("embeddings.LayerNorm.weight"),
          "layernorm", {hidden});
  w.add("tokenizer.xlmr_table", tb.data(), tb.size(), "U8", "tokenizer",
        {static_cast<int64_t>(tb.size())});
  if (log) {
    std::ostringstream s;
    s << (tok_generated
              ? "  generated tokenizer.xlmr_table (no cached "
                "xlmr_tokenizer.bin found)  "
              : "  tokenizer.xlmr_table  ")
      << (tb.size() / 1e6) << " MB (XLMRTOK1, tasks/0127)";
    log(s.str());
  }
  add_f32("embeddings.ln.bias", get("embeddings.LayerNorm.bias"),
          "layernorm", {hidden});

  for (int64_t i = 0; i < L; ++i) {
    const std::string p = "encoder.layer." + std::to_string(i) + ".";
    const std::string at = p + "attention.";
    const std::string tag = "layer." + std::to_string(i) + ".";

    // Fused upstream already: qkv_proj is [2304,768] [Q|K|V]-major.
    // 1/sqrt(head_dim) folded into the Q block (first `hidden` columns of
    // the transposed [768,2304] operand) -- exact, RoPE is linear in q.
    // Unlike nomic, gte HAS a qkv bias, so its Q third scales too:
    // (xW + b)*s == x(Ws) + (bs).
    add_gemm_b(w, tag + "qkv", get(at + "qkv_proj.weight"), tile_k, tile_n,
               layout_json, layout_hash, scale, hidden);
    {
      const Tensor &b = get(at + "qkv_proj.bias");
      std::vector<float> bias(b.f32(), b.f32() + b.count());
      for (int64_t o = 0; o < hidden; ++o)
        bias[static_cast<size_t>(o)] *= scale;
      w.add(tag + "qkv.bias", bias.data(), bias.size() * 4, "F32", "bias",
            {3 * hidden});
    }

    add_gemm_b(w, tag + "attn_out", get(at + "o_proj.weight"), tile_k,
               tile_n, layout_json, layout_hash);
    add_f32(tag + "attn_out.bias", get(at + "o_proj.bias"), "bias", {hidden});
    add_f32(tag + "ln1.weight", get(p + "attn_ln.weight"), "layernorm",
            {hidden});
    add_f32(tag + "ln1.bias", get(p + "attn_ln.bias"), "layernorm", {hidden});

    // up_gate_proj is already the fused [2*inter, hidden] the runtime
    // wants: transposed to [hidden, 2*inter] by add_gemm_b; up columns
    // [0, inter), gate columns [inter, 2*inter) -- the lo/hi order of
    // `lo * act(hi)`. Genuinely bias-free upstream -- zero-filled.
    add_gemm_b(w, tag + "ffn_up", get(p + "mlp.up_gate_proj.weight"),
               tile_k, tile_n, layout_json, layout_hash);
    {
      std::vector<float> z(static_cast<size_t>(2 * inter), 0.f);
      w.add(tag + "ffn_up.bias", z.data(), z.size() * 4, "F32", "bias",
            {2 * inter});
    }

    add_gemm_b(w, tag + "ffn_down", get(p + "mlp.down_proj.weight"), tile_k,
               tile_n, layout_json, layout_hash);
    add_f32(tag + "ffn_down.bias", get(p + "mlp.down_proj.bias"), "bias",
            {hidden});
    add_f32(tag + "ln2.weight", get(p + "mlp_ln.weight"), "layernorm",
            {hidden});
    add_f32(tag + "ln2.bias", get(p + "mlp_ln.bias"), "layernorm", {hidden});
  }

  w.write(out, cj, /*arch=*/3);
  if (log) {
    std::ostringstream s;
    s << "\n  tensors    : " << w.count()
      << "\n  data       : " << (w.data_bytes() / 1e6) << " MB"
      << "\n  source     : " << sha.substr(0, 16) << "...";
    log(s.str());
  }
}

namespace {

// Read one top-level string field, through the real JSON parser rather than a
// scan. main.cpp used http.hpp's json_field_string() for this, which is a
// string scanner and is not in the library subset (http.hpp brings winsock).
// A DOM parse is also strictly harder to fool: a scanner finds `"model_type"`
// wherever it appears, including nested one level down.
std::string json_string_field(const std::string &path, const std::string &key) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return std::string();
  std::stringstream ss;
  ss << f.rdbuf();
  try {
    const npue::json::Value v = npue::json::parse(ss.str());
    const npue::json::Value *f = v.find(key);
    if (!f || !f->is_string()) return std::string();
    return f->as_string();
  } catch (const std::exception &) {
    return std::string();
  }
}

bool json_bool_field(const npue::json::Value &v, const char *key) {
  const npue::json::Value *f = v.find(key);
  return f && f->is_bool() && f->as_bool();
}

void say(const PrepareOptions &opt, const std::string &s) {
  if (opt.log) opt.log(s);
}

std::string resolve_source_repo(const PrepareOptions &opt) {
  if (!opt.source_repo.empty()) return opt.source_repo;
  const std::string repo =
      json_string_field(opt.checkpoint_dir + "/CHECKPOINT.json", "repo_id");
  if (repo.empty())
    throw std::runtime_error(
        "no CHECKPOINT.json with a repo_id under " + opt.checkpoint_dir +
        " and no source_repo given -- refusing to guess which repository "
        "these weights came from. A container that misattributes its own "
        "weights is a licensing statement.");
  return repo;
}

// Pooling comes from the checkpoint's own 1_Pooling/config.json, the same
// source tools/pack_npue.py reads. Both packers must agree or
// verify_pack_parity fails, which is the point of having the gate.
std::string resolve_pooling(const PrepareOptions &opt) {
  const std::string path = opt.checkpoint_dir + "/1_Pooling/config.json";
  std::ifstream pf(path, std::ios::binary);
  if (!pf)
    throw std::runtime_error(
        "no 1_Pooling/config.json under " + opt.checkpoint_dir +
        " -- cannot tell whether this checkpoint pools by mean or by CLS, and "
        "the two are different models");
  std::stringstream ps;
  ps << pf.rdbuf();
  const npue::json::Value v = npue::json::parse(ps.str());
  const bool cls = json_bool_field(v, "pooling_mode_cls_token");
  const bool mean = json_bool_field(v, "pooling_mode_mean_tokens");
  if (cls == mean)
    throw std::runtime_error(
        "1_Pooling/config.json asks for neither or both of cls and mean; this "
        "runtime implements exactly those two");
  return cls ? "cls" : "mean";
}

}  // namespace

// Emit `fused` (an [N, K] checkpoint tensor) with its row range
// [r0, r0+rows) moved AHEAD of the rest: the layout along N is
// [fused[r0..r0+rows) ; fused[everything else]], transposed to [K, N] the way
// add_gemm_b() does, then pre-tiled.
//
// WHY A PERMUTATION AND NOT A CONCATENATION. add_gemm_b_concat2() above takes
// TWO whole tensors and lays them out as [a; b], which is the right helper for
// nomic (fc11 and fc12 are two separate [inter, hidden] tensors) and cannot
// express "the same tensor, rows reordered". ModernBERT has no second
// up-projection tensor to concatenate with: mlp.Wi is stored FUSED at
// [2*inter, hidden] and is the model's only up-projection.
//
// WHY ModernBERT NEEDS IT AT ALL. Upstream computes
//     x, gate = self.Wi(h).chunk(2, dim=-1)
// so the GATE is the SECOND half -- chunk(2, -1) yields [first, second] and they
// bind them as (input, gate). This runtime computes `lo * gelu(hi)` over the
// packed order, i.e. the ACTIVE half comes SECOND, so the gate half has to be
// packed FIRST. LLaMA's convention is the reverse of ModernBERT's here, and
// getting it backwards still produces a model that emits fluent plausible
// garbage -- which is why the container records what it did in
// config["glu_halves"] rather than leaving a reader to infer it.
//
// Exact, not approximate: every element that moves is the same float32 value and
// every element is still present exactly once.
static void add_gemm_b_reorder_rows(Writer &w, const std::string &name,
                                    const Tensor &fused, int64_t r0, int64_t rows,
                                    int64_t tk, int64_t tn,
                                    const std::string &layout_json,
                                    const std::string &layout_hash) {
  const int64_t N = fused.rows(), K = fused.cols();
  if (r0 < 0 || rows <= 0 || r0 + rows > N)
    throw std::runtime_error(
        name + ": row range [" + std::to_string(r0) + ", " +
        std::to_string(r0 + rows) + ") is outside the fused tensor's " +
        std::to_string(N) + " rows");
  const float *src = fused.f32();
  std::vector<float> m(static_cast<size_t>(K) * N);
  for (int64_t r = 0; r < K; ++r) {
    // Source and destination are different index spaces, so this is a
    // per-element copy with no aliasing to reason about.
    for (int64_t c = 0; c < rows; ++c)
      m[r * N + c] = src[(r0 + c) * K + r];
    for (int64_t c = rows; c < N; ++c)
      m[r * N + c] = src[(c - rows) * K + r];
  }
  const auto tiled = tile_b(m.data(), K, N, tk, tn);
  w.add(name, tiled.data(), tiled.size() * 2, "BF16", "gemm_b", {K, N},
        layout_json, layout_hash);
}

// The .npue header's `arch` number for this architecture.
//
// It is written into the 64-byte header (Writer::write's argument) and
// npu_offload/gemm_rtp/npue.py reads it back as ARCH_MODERNBERT_ROPE_GEGLU = 4.
// That constant was written when the architecture was specified and long before
// anything here could pack one; the container format carried the slot from the
// start. 4 in ONE place, because a second literal here would be a second place
// for the header and the reader to disagree.
constexpr uint32_t kArchModernBertRopeGeGLU = 4;

// arch=4 (ModernBERT / mmBERT): pre-LN, RoPE, gated GeGLU, bidirectional with a
// sliding window on most layers.
//
// NOT A BERT WITH A NEW ACTIVATION. Four properties make this architecture
// different from arch=0 in ways that each produce a PLAUSIBLE WRONG ANSWER
// rather than an error, so each is asserted from the checkpoint rather than
// assumed:
//
//  * PRE-LAYERNORM, with a final norm after the last layer. arch=0/2/3 fuse
//    `residual + y` and then normalise, storing the NORMALISED value as the
//    residual -- correct under post-LN, wrong under pre-LN, where the next block
//    would add to an already-centred stream and the residual stream would be
//    re-centred twice per layer.
//  * LAYER 0'S ATTENTION NORM IS nn.Identity(). The checkpoint has no
//    `attn_norm` tensor for layer 0 at all (21 of them for 22 layers). It must
//    be SKIPPED, not replaced by a weight-1 norm: a norm with weight 1 still
//    subtracts the mean and divides by the standard deviation.
//  * NO BIASES ANYWHERE IN THE ENCODER. attention_bias, mlp_bias and norm_bias
//    are all false and the checkpoint ships no `*.bias` tensor for the encoder
//    at all. The runtime dereferences `<op>.bias` for every GEMM
//    unconditionally, so they are emitted zero-filled -- exact, because the
//    embedding build is dst = word + position + token_type and a zero add
//    changes nothing.
//  * NO POSITION TABLE. position enters only through RoPE; `sans_pos` (mmBERT)
//    says so and ModernBERT's own "absolute" is a dead key read by no code path.
//    Zero-filled placeholders, same reasoning as nomic.
//
// And one that is a TRAP rather than a difference: the GeGLU gate is the SECOND
// half of mlp.Wi, the opposite of LLaMA's convention and of arch=2/3 here.
// add_gemm_b_reorder_rows() puts it first and the container says so.
//
// The decision head's tensors are packed here TOO, as plain F32 host weights --
// no layout hash, no design, no pre-tiling -- because the decision engine runs
// the head on the HOST. The reasons are structural rather than arithmetic: the
// geometry globals are process-wide and written once by ShapeLease, the encoder
// hardcodes its "layer." tensor prefix, every npu::Design is its own
// hw_context, and one model resolves to one design set. The head is 8 of this
// model's 96 GEMMs. Packing them keeps the container self-contained, so moving
// the head onto the array later is not a re-pack.
void prepare_model_modernbert(const std::string &model_dir,
                              const std::string &pooling,
                              const std::string &source_repo,
                              const std::string &out,
                              const std::string &layout_json,
                              const std::string &layout_hash,
                              int64_t tile_k, int64_t tile_n, int64_t max_seq,
                              const std::string &config_subdir,
                              const std::string &tokenizer_subdir,
                              const std::string &rl_config_path,
                              void (*log)(const std::string &)) {
  const auto st_buf = slurp(model_dir + "/model.safetensors");
  const auto src = read_safetensors(st_buf);   // widens this checkpoint's F16
  Sha256 sh;                                   // to F32 at read time
  sh.update(st_buf.data(), st_buf.size());
  const std::string sha = sh.hex();

  // The encoder's tensors carry an `encoder.` prefix on top of ModernBERT's own
  // names -- `attn.Wqkv`, `mlp.Wi`, `attn_norm`, `final_norm`. Try the prefixed
  // form first and fall back to the bare one, the same shape as gte's `get()`
  // above: a checkpoint saved by transformers has the prefix, and one saved by a
  // converter may not.
  auto get = [&](const std::string &n) -> const Tensor & {
    auto it = src.find("encoder." + n);
    if (it == src.end()) it = src.find(n);
    if (it == src.end())
      throw std::runtime_error("checkpoint has no tensor 'encoder." + n +
                               "' (nor '" + n + "')");
    if (it->second.dtype != "F32")
      throw std::runtime_error("checkpoint tensor '" + n + "' is " +
                               it->second.dtype + "; this packer reads F32 "
                               "(F16/BF16 are widened at read time)");
    return it->second;
  };

  const auto cfg_buf = slurp(sub(model_dir, config_subdir, "config.json"));
  const std::string cfg(reinterpret_cast<const char *>(cfg_buf.data()),
                        cfg_buf.size());
  auto find_key = [&](const char *key) -> size_t {
    return cfg.find(std::string("\"") + key + "\"");
  };
  // The exact literal text of a value, or the fallback when the key is absent or
  // explicitly null. Both are needed: `bos_token_id` is optional in some configs
  // and `null` in others, and `std::stoll("null")` is a throw with a message
  // about neither.
  auto cfg_raw_or = [&](const char *key, const char *fallback) -> std::string {
    const size_t i = find_key(key);
    if (i == std::string::npos) return fallback;
    size_t c = cfg.find(':', i) + 1;
    while (c < cfg.size() && std::isspace(static_cast<unsigned char>(cfg[c]))) ++c;
    size_t e = c;
    while (e < cfg.size() && cfg[e] != ',' && cfg[e] != '}' && cfg[e] != '\n' &&
           cfg[e] != '\r')
      ++e;
    while (e > c && std::isspace(static_cast<unsigned char>(cfg[e - 1]))) --e;
    const std::string v = cfg.substr(c, e - c);
    return v == "null" ? std::string(fallback) : v;
  };
  auto cfg_int = [&](const char *key) -> int64_t {
    const std::string v = cfg_raw_or(key, nullptr);
    if (v.empty())
      throw std::runtime_error(std::string("config.json has no ") + key);
    return std::stoll(v);
  };
  auto cfg_bool = [&](const char *key, bool fallback) -> bool {
    const std::string v = cfg_raw_or(key, fallback ? "true" : "false");
    if (v == "true") return true;
    if (v == "false") return false;
    return fallback;
  };
  auto cfg_str = [&](const char *key) -> std::string {
    const size_t i = find_key(key);
    if (i == std::string::npos)
      throw std::runtime_error(std::string("config.json has no ") + key);
    const size_t c = cfg.find(':', i) + 1;
    const size_t q1 = cfg.find('"', c);
    const size_t q2 = cfg.find('"', q1 + 1);
    return cfg.substr(q1 + 1, q2 - q1 - 1);
  };

  // --- geometry, and the fail-closed assertions --------------------------
  const std::string model_type = cfg_str("model_type");
  if (model_type != "modernbert")
    throw std::runtime_error("model_type='" + model_type +
                             "', expected 'modernbert'");
  const int64_t L = cfg_int("num_hidden_layers");
  const int64_t H = cfg_int("num_attention_heads");
  const int64_t hidden = cfg_int("hidden_size");
  const int64_t inter = cfg_int("intermediate_size");
  const int64_t vocab_size = cfg_int("vocab_size");
  const int64_t head_dim = hidden / H;      // this config carries no head_dim
  if (head_dim != 64)
    throw std::runtime_error(
        "hidden_size/num_attention_heads = " + std::to_string(head_dim) +
        ", expected 64. The host attention kernels step head_dim/8 AVX2 "
        "vectors and the design's MAC tile is chosen against it, so any other "
        "width is a DIFFERENT geometry that needs its own design family -- not "
        "this one, and not a variant of it.");

  // Every bias off. Each is a plural flag in this config and each has a tensor
  // behind it, so a config that sets one while the checkpoint ships none is
  // exactly the fail-open the arch-4 guard exists to stop.
  for (const char *k : {"attention_bias", "mlp_bias", "norm_bias"}) {
    if (cfg_bool(k, false))
      throw std::runtime_error(
          std::string("config.json sets ") + k +
          " -- this packer emits a bias-free encoder and zero-fills every "
          "*.bias, because a ModernBERT checkpoint carries none. A config that "
          "says otherwise is a mismatch; refusing rather than dropping weights.");
  }
  if (cfg_str("hidden_activation") != "gelu")
    throw std::runtime_error(
        "hidden_activation='" + cfg_str("hidden_activation") +
        "', expected 'gelu' -- this is the EXACT erf form (ACT2FN['gelu']), not "
        "the tanh approximation");
  const std::string pet = cfg_str("position_embedding_type");
  if (pet != "sans_pos" && pet != "rope" && pet != "absolute")
    throw std::runtime_error(
        "position_embedding_type='" + pet +
        "' -- ModernBERT carries no position table (mmBERT says so with "
        "'sans_pos', and ModernBERT's own 'absolute' is a dead key read by no "
        "code path). A config naming a table this packer would have to read is "
        "a checkpoint whose embeddings would be wrong.");

  // layer_types is DERIVED in ModernBERT (`"full_attention" if i % n == 0`) and
  // mmBERT ships it explicitly. Read it when present -- 22 explicit entries beat
  // a re-derivation -- and derive it otherwise. Either way the resolved list goes
  // into the container, because that is what the runtime's band mask keys on.
  const int64_t every_n = cfg_int("global_attn_every_n_layers");
  const int64_t local_attention = cfg_int("local_attention");
  const int64_t half_window = local_attention / 2;
  if (half_window <= 0)
    throw std::runtime_error(
        "local_attention=" + std::to_string(local_attention) +
        " -- the half-window is local_attention / 2 and this leaves no band");
  std::vector<std::string> layer_types;
  {
    const size_t i = find_key("layer_types");
    if (i == std::string::npos) {
      for (int64_t l = 0; l < L; ++l)
        layer_types.push_back((l % every_n == 0) ? "full_attention"
                                                : "sliding_attention");
    } else {
      const size_t ob = cfg.find('[', i), cb = cfg.find(']', ob);
      if (ob == std::string::npos || cb == std::string::npos)
        throw std::runtime_error("config.json: layer_types is not an array");
      const std::string body = cfg.substr(ob + 1, cb - ob - 1);
      size_t p = 0;
      while (true) {
        const size_t q1 = body.find('"', p);
        if (q1 == std::string::npos) break;
        const size_t q2 = body.find('"', q1 + 1);
        if (q2 == std::string::npos) break;
        layer_types.push_back(body.substr(q1 + 1, q2 - q1 - 1));
        p = q2 + 1;
      }
      if (static_cast<int64_t>(layer_types.size()) != L)
        throw std::runtime_error("layer_types has " +
                                 std::to_string(layer_types.size()) +
                                 " entries, num_hidden_layers is " +
                                 std::to_string(L));
      // The derived rule is a FACT about this architecture, not a preference,
      // so the shipped list is cross-checked against it. A config whose
      // layer_types was edited without the geometry following it would band the
      // wrong layers, which is a wrong answer and not a warning.
      for (int64_t l = 0; l < L; ++l) {
        const bool want_global = (l % every_n == 0);
        const std::string &got = layer_types[static_cast<size_t>(l)];
        if (got != "full_attention" && got != "sliding_attention")
          throw std::runtime_error("layer_types[" + std::to_string(l) +
                                   "] is '" + got +
                                   "', expected full_attention or "
                                   "sliding_attention");
        if (want_global != (got == "full_attention"))
          throw std::runtime_error(
              "layer_types[" + std::to_string(l) + "] is '" + got +
              "' but i % " + std::to_string(every_n) +
              " == 0 says it must be '" +
              (want_global ? "full_attention" : "sliding_attention") +
              "'. ModernBERT DERIVES this list, and a config where the two "
              "disagree would band the wrong layers");
      }
    }
  }

  // RoPE theta is NESTED here -- rope_parameters.{full_attention,sliding_
  // attention}.rope_theta -- and ModernBERT-large's two values DIFFER (160000
  // global, 10000 local) while mmBERT's are both 160000. The runtime's pre-LN
  // path builds ONE table and selects it per layer, which is exact for a
  // checkpoint whose two agree and wrong by up to 1.9e-02 relfro at layer 0 for
  // one whose they do not. So this REFUSES rather than picking a winner.
  const double theta = [&] {
    const size_t i = find_key("rope_parameters");
    if (i == std::string::npos)
      throw std::runtime_error("config.json has no rope_parameters");
    const size_t ob = cfg.find('{', i);
    if (ob == std::string::npos)
      throw std::runtime_error("config.json: rope_parameters is not an object");
    // One balanced-brace scan per sub-object. A bare substring search would
    // always find the first `rope_theta`, which is the whole reason a
    // differently-theta'd ModernBERT would be silently wrong here.
    auto theta_in = [&](const char *key) -> double {
      const std::string k = std::string("\"") + key + "\"";
      const size_t at = cfg.find(k, ob);
      if (at == std::string::npos)
        throw std::runtime_error("rope_parameters has no " + std::string(key));
      const size_t ob2 = cfg.find('{', at);
      if (ob2 == std::string::npos)
        throw std::runtime_error(std::string("rope_parameters.") + key + " is not an object");
      size_t depth = 0, end = std::string::npos;
      for (size_t p = ob2; p < cfg.size(); ++p) {
        if (cfg[p] == '{') ++depth;
        else if (cfg[p] == '}' && --depth == 0) { end = p; break; }
      }
      if (end == std::string::npos)
        throw std::runtime_error(std::string("rope_parameters.") + key + " has no closing brace");
      const std::string body = cfg.substr(ob2, end - ob2 + 1);
      const size_t t = body.find("\"rope_theta\"");
      if (t == std::string::npos)
        throw std::runtime_error(std::string("rope_parameters.") + key + " has no rope_theta");
      return std::stod(body.substr(body.find(':', t) + 1));
    };
    const double full = theta_in("full_attention");
    const double local = theta_in("sliding_attention");
    if (full != local)
      throw std::runtime_error(
          "rope_parameters.full_attention.rope_theta=" + py_double_repr(full) +
          " and .sliding_attention.rope_theta=" + py_double_repr(local) +
          " differ. This build's pre-LN path builds ONE RoPE table and selects it "
          "per layer, which is exact for a checkpoint whose two agree (mmBERT: "
          "both 160000) and wrong for one whose they do not (ModernBERT-large: "
          "160000 global / 10000 local). Refusing rather than picking one.");
    if (!(full > 0.0))
      throw std::runtime_error("rope_theta must be positive");
    return full;
  }();

  // The host LayerNorm uses a HARDCODED 1e-12 (layer_norm_cpu), so emitting
  // 1e-5 and calling it layer_norm_eps would be a claim the runtime does not
  // honour. Recorded for provenance, with the delta spelled out in
  // not_implemented: ~5e-6 relative, far under the bf16 operand's 8-bit
  // mantissa, and a Phase-5 comparison against a bf16 replica rather than
  // against an absolute threshold.
  const std::string eps_raw = cfg_raw_or("layer_norm_eps", "1e-5");

  // --- the RL agent config: temperatures and head geometry ---------------
  //
  // PACKED, NOT READ AT RUN TIME. `temperature` is three floats and
  // `temperature_by_options` is a small object, so they go in the config as
  // data. Three reasons that is right:
  //  * the container becomes self-contained, so a later upstream recalibration
  //    does not silently apply to an already-served model;
  //  * the checkpoint's own `temperature` BUFFER is deliberately NOT packed --
  //    it is the same three numbers, upstream's forward() never reads it, and
  //    packing it twice is how a container ends up with two sources of truth
  //    for one value;
  //  * the file is still needed ON DISK to read them, so it is in the model
  //    entry's `files` list -- but not after the pack.
  int64_t head_layers = 2, n_act = 2, max_len = 512, head_max_len = 192;
  double temperature[3] = {1.0, 1.0, 1.0};   // QTYPES: choice, score, noul
  std::string temperature_by_options = "{}";
  {
    const auto rl_buf = slurp(rl_config_path);
    const std::string rl(reinterpret_cast<const char *>(rl_buf.data()),
                         rl_buf.size());
    auto rl_find = [&](const char *key) -> size_t {
      return rl.find(std::string("\"") + key + "\"");
    };
    auto rl_int = [&](const char *key, int64_t fallback) -> int64_t {
      const size_t i = rl_find(key);
      if (i == std::string::npos) return fallback;
      return std::stoll(rl.substr(rl.find(':', i) + 1));
    };
    head_layers = rl_int("head_layers", head_layers);
    max_len = rl_int("max_len", max_len);
    head_max_len = rl_int("head_max_len", head_max_len);
    if (head_layers <= 0)
      throw std::runtime_error(rl_config_path + ": head_layers=" +
                               std::to_string(head_layers) +
                               " -- a decision model with no head is not a "
                               "decision model");
    {
      // n_act = len(act_costs) + 1, and it is the WIDTH of the act head's
      // output. Read, not hardcoded: the next Laya release may add a cost, and a
      // hardcoded 2 would silently drop it.
      const size_t i = rl_find("act_costs");
      if (i == std::string::npos)
        throw std::runtime_error(rl_config_path +
                                 " has no act_costs, so the act head's width "
                                 "cannot be derived");
      const size_t ob = rl.find('{', i), cb = rl.find('}', ob);
      if (ob == std::string::npos || cb == std::string::npos)
        throw std::runtime_error(rl_config_path + ": act_costs is not an object");
      int64_t n = 0;
      for (size_t p = ob + 1; p < cb;) {
        const size_t q1 = rl.find('"', p);
        if (q1 == std::string::npos || q1 >= cb) break;
        const size_t q2 = rl.find('"', q1 + 1);
        if (q2 == std::string::npos || q2 > cb) break;
        ++n;
        p = q2 + 1;
      }
      n_act = n + 1;
    }
    {
      const size_t i = rl_find("temperature");
      if (i == std::string::npos)
        throw std::runtime_error(rl_config_path + " has no temperature vector");
      const size_t ob = rl.find('[', i), cb = rl.find(']', ob);
      if (ob == std::string::npos || cb == std::string::npos)
        throw std::runtime_error(rl_config_path + ": temperature is not an array");
      const std::string body = rl.substr(ob + 1, cb - ob - 1);
      size_t p = 0;
      for (int k = 0; k < 3; ++k) {
        const size_t comma = body.find(',', p);
        temperature[k] = std::stod(
            body.substr(p, comma == std::string::npos ? std::string::npos : comma - p));
        if (comma == std::string::npos) break;
        p = comma + 1;
      }
      for (int k = 0; k < 3; ++k)
        if (!(temperature[k] > 0.0))
          throw std::runtime_error(rl_config_path + ": temperature[" +
                                   std::to_string(k) + "]=" +
                                   py_double_repr(temperature[k]) +
                                   " -- zero or negative divides the softmax "
                                   "by zero");
      // temperature_by_options is copied VERBATIM, braces included, because its
      // keys are Python-formatted strings ("choice:11+") and re-emitting them
      // from parsed pieces is a way to spell one of them differently.
      const size_t j = rl_find("temperature_by_options");
      if (j == std::string::npos)
        throw std::runtime_error(
            rl_config_path + " has no temperature_by_options. The key is emitted "
            "whether or not it is empty, because 'the key is missing' and 'the "
            "map is empty' must not mean the same thing to a reader.");
      const size_t b2 = rl.find('{', j);
      if (b2 == std::string::npos)
        throw std::runtime_error(rl_config_path +
                                 ": temperature_by_options is not an object");
      int depth = 0;
      size_t e = b2;
      for (; e < rl.size(); ++e) {
        if (rl[e] == '{') ++depth;
        else if (rl[e] == '}' && --depth == 0) break;
      }
      if (depth != 0)
        throw std::runtime_error(rl_config_path +
                                 ": temperature_by_options has unbalanced braces");
      temperature_by_options = rl.substr(b2, e - b2 + 1);
    }
  }

  // FLOAT, not double, exactly as the other packers' scale is -- and for
  // head_dim 64 the value is 0.125, a power of two, so the fold below is EXACT
  // (no rounding anywhere in x * 0.125f).
  const float scale =
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));

  // --- the tokenizer blob -----------------------------------------------
  // Prefer a cached file, else generate here and write it back, exactly as
  // prepare_model_gte() does for its XLM-R table. The generator is the C++ port
  // of tools/gen_bbpe_tokenizer_table.py, so a cold clone packs without Python.
  std::vector<uint8_t> tb;
  bool tok_generated = false;
  {
    const std::string bin = sub(model_dir, tokenizer_subdir, "tokenizer.bin");
    const std::string js = sub(model_dir, tokenizer_subdir, "tokenizer.json");
    std::ifstream tf(bin, std::ios::binary);
    if (tf.good()) {
      tf.close();
      tb = slurp(bin);
    } else {
      tb = generate_bbpe_tokenizer_table(js);
      tok_generated = true;
      std::ofstream of(bin, std::ios::binary);
      if (!of) throw std::runtime_error("cannot write " + bin);
      of.write(reinterpret_cast<const char *>(tb.data()),
               static_cast<std::streamsize>(tb.size()));
      if (!of) throw std::runtime_error("error writing " + bin);
    }
  }

  if (log) {
    int n_full = 0;
    for (const auto &t : layer_types)
      if (t == "full_attention") ++n_full;
    std::ostringstream s;
    s << "packing " << model_dir << " -> " << out
      << "  (arch=modernbert_rope_geglu)\n"
      << "  hidden=" << hidden << " heads=" << H << " head_dim=" << head_dim
      << " layers=" << L << " inter=" << inter << " ffn_up=" << 2 * inter
      << "\n  rope=" << py_double_repr(theta) << " window=" << half_window
      << " (local_attention/2, NOT +1)  " << n_full << " full / "
      << (L - n_full) << " sliding\n"
      << "  head_layers=" << head_layers << " head_ffn=" << 4 * hidden
      << " n_act=" << n_act << " max_len=" << max_len
      << " head_max_len=" << head_max_len << "\n  temperature=["
      << py_double_repr(temperature[0]) << ", " << py_double_repr(temperature[1])
      << ", " << py_double_repr(temperature[2]) << "] (QTYPES order)";
    log(s.str());
  }

  // --- the container config ----------------------------------------------
  //
  // KEY ORDER IS FIXED, and the NAMES ARE NOT FREE CHOICE. apply_model_shape()
  // reads `num_layers`, `num_heads`, `hidden`, `head_dim`, `intermediate` and
  // `max_seq_len` (npue_encoder.hpp), and a misspelled or missing one is not a
  // warning: it is `the .npue reports a non-positive shape` two thousand lines
  // from the cause, or a zero-shaped geometry that fails somewhere less obvious.
  std::string cj;
  cj += "{\"arch\":\"modernbert_rope_geglu\"";
  cj += ",\"a_dtype\":\"bf16\"";
  cj += ",\"model_type\":\"" + model_type + "\"";
  cj += ",\"source_repo\":\"" + source_repo + "\"";
  cj += ",\"source_sha256\":\"" + sha + "\"";
  cj += ",\"num_layers\":" + std::to_string(L);
  cj += ",\"num_heads\":" + std::to_string(H);
  cj += ",\"hidden\":" + std::to_string(hidden);
  cj += ",\"head_dim\":" + std::to_string(head_dim);
  cj += ",\"intermediate\":" + std::to_string(inter);
  cj += ",\"qkv_n\":" + std::to_string(3 * hidden);
  cj += ",\"ffn_up_n\":" + std::to_string(2 * inter);
  cj += ",\"vocab_size\":" + std::to_string(vocab_size);
  cj += ",\"max_seq_len\":" + std::to_string(max_seq);
  cj += ",\"checkpoint_max_position_embeddings\":" +
        cfg_raw_or("max_position_embeddings", "8192");
  cj += ",\"layer_norm_eps\":" + eps_raw;
  cj += ",\"activation\":\"gelu\",\"gated_ffn\":true";
  cj += ",\"glu_halves\":\"gate_first|up_second -- ModernBERT binds "
        "x, gate = Wi(h).chunk(2, dim=-1), so the GATE is the SECOND half, and "
        "the packer moved it FIRST because this runtime computes lo * gelu(hi) "
        "over the packed order. LLaMA's convention is the reverse of ModernBERT's "
        "here, and reading it the other way still produces fluent plausible "
        "garbage.\"";
  cj += ",\"position_embedding_type\":\"rope\"";
  cj += ",\"rope_theta\":" + py_double_repr(theta);
  cj += ",\"rope_theta_note\":\"ONE table, selected per layer. This packer "
        "REFUSES a checkpoint whose full_attention and sliding_attention thetas "
        "differ, because it only implements the single-table path; "
        "ModernBERT-large (160000 / 10000) is therefore out of reach until a "
        "per-layer selection lands, and mmBERT (both 160000) is exact.\"";
  cj += ",\"pre_layernorm\":true,\"final_norm\":true";
  cj += ",\"identity_attn_norm_layer0\":true";
  cj += ",\"attention_bias\":false,\"mlp_bias\":false,\"norm_bias\":false";
  cj += ",\"bias_note\":\"zero-filled, not omitted: the runtime dereferences "
        "<op>.bias for every GEMM unconditionally, and a zero add to the "
        "embedding build is exact.\"";
  cj += ",\"sliding_window\":" + std::to_string(half_window);
  cj += ",\"sliding_window_note\":\"local_attention/2, NOT +1. The +1 in "
        "transformers' `self.sliding_window = config.sliding_window + 1` is a "
        "FlashAttention inclusive-boundary convention and does NOT apply to the "
        "dense/sdpa mask this checkpoint pins, which reaches "
        "`abs(q_idx - kv_idx) <= config.sliding_window` through "
        "sliding_window_bidirectional_overlay. The window is 64: a 129-token "
        "band.\"";
  cj += ",\"global_attn_every_n_layers\":" + std::to_string(every_n);
  cj += ",\"layer_types\":[";
  for (int64_t l = 0; l < L; ++l)
    cj += std::string(l ? "," : "") + "\"" +
          layer_types[static_cast<size_t>(l)] + "\"";
  cj += "]";
  // "pooling":"mean" is a FICTION and l2_normalize:false is the honest half.
  // pool_rows() accepts only cls and mean (npue_encoder.hpp), so the key has to
  // name one of the two -- but this model pools by GATHER at marker_pos, not by
  // mean and not by CLS, and the decision engine never calls pool_rows at all.
  // Recorded here because a future reader WILL reach for this key.
  cj += ",\"pooling\":\"" + pooling + "\",\"l2_normalize\":false";
  cj += ",\"pooling_note\":\"a fiction, and the only value apply_model_shape "
        "will accept. This model pools by `gather` at the [MASK] marker "
        "positions -- the decision head's readout -- not by mean and not by CLS, "
        "and Laya never L2-normalises. The decision engine must NOT call "
        "pool_rows; see not_implemented.\"";
  cj += ",\"head_layers\":" + std::to_string(head_layers);
  cj += ",\"head_intermediate\":" + std::to_string(4 * hidden);
  cj += ",\"head_n_act\":" + std::to_string(n_act);
  cj += ",\"head_activation\":\"relu\"";
  cj += ",\"head_bias\":true";
  cj += ",\"head_attention\":\"full-band: no sliding window, no causal mask, "
        "padding mask only\"";
  cj += ",\"n_qtype\":3";
  cj += ",\"marker_token_id\":" + cfg_raw_or("mask_token_id", "4");
  cj += ",\"cls_token_id\":" + cfg_raw_or("cls_token_id", "1");
  cj += ",\"sep_token_id\":" + cfg_raw_or("sep_token_id", "1");
  cj += ",\"pad_token_id\":" + cfg_raw_or("pad_token_id", "0");
  cj += ",\"unk_token_id\":" + cfg_raw_or("unk_token_id", "3");
  cj += ",\"bos_token_id\":" + cfg_raw_or("bos_token_id", "2");
  cj += ",\"special_ids_note\":\"read from THIS config, never from the "
        "tokenizer blob, and the runtime refuses if the two disagree. The blob "
        "derives cls_id from the first of {[CLS], <s>, <|endoftext|>} and sep_id "
        "from the first of {[SEP], </s>}; this vocabulary has <s> at 204 and "
        "</s> at 213, so the blob would say 204/213 while this config says 1 and "
        "1 (both <eos>).\"";
  cj += ",\"max_len\":" + std::to_string(max_len);
  cj += ",\"head_max_len\":" + std::to_string(head_max_len);
  cj += ",\"temperature\":[" + py_double_repr(temperature[0]) + "," +
        py_double_repr(temperature[1]) + "," + py_double_repr(temperature[2]) + "]";
  cj += ",\"temperature_by_options\":" + temperature_by_options;
  cj += ",\"temperature_note\":\"packed, not read at run time, in QTYPES order "
        "{choice:0, score:1, noul:2}. The checkpoint's own `temperature` BUFFER "
        "is deliberately NOT packed: it is the same three numbers, upstream's "
        "forward() never reads it, and packing it twice is how a container ends "
        "up with two sources of truth for one value.\"";
  cj += ",\"tile_k\":" + std::to_string(tile_k) +
        ",\"tile_n\":" + std::to_string(tile_n) +
        ",\"mac_s\":" + std::to_string(kMacS) +
        ",\"mac_t\":" + std::to_string(kMacT);
  cj += ",\"fusions\":{\"qkv_fused\":true,\"transposed_to_kn\":true,"
        "\"qk_scale_folded_into_q\":true,"
        "\"gemm_operands_bf16\":true,"
        "\"biases_and_layernorm_fp32\":true,"
        "\"gated_ffn_fused_upstream\":true,"
        "\"gated_ffn_gate_half_moved_first\":true,"
        "\"position_embeddings_zeroed_rope_instead\":true,"
        "\"token_type_embeddings_zeroed\":true}";
  cj += ",\"not_implemented\":[";
  cj += "\"context above " + std::to_string(max_seq) +
        " tokens: the design is compiled at that seq and set_design_seq() "
        "REFUSES more. The checkpoint's max_position_embeddings is " +
        cfg_raw_or("max_position_embeddings", "8192") +
        " and is a RoPE cache length only -- there is no table -- so this is a "
        "throughput choice, not a correctness one.\"";
  cj += ",\"predict_long's 50%-overlap windowing for long documents\"";
  cj += ",\"structured (non-string) `state` and `criteria` values: Phase 1's "
        "render_criterion reproduces Python's json.dumps exactly, but the "
        "structured-STATE path is a separate serialiser and ships as a refusal "
        "by name\"";
  cj += ",\"Laya's Router language routing: one model per process, chosen by "
        "the caller\"";
  cj += ",\"build_sequence's truncate_left and option_order arguments\"";
  cj += ",\"pooling says mean because the runtime accepts only cls/mean and "
        "this model gathers at marker_pos; see pooling_note\"";
  cj += ",\"cls_token_id/sep_token_id come from the config and not from the "
        "tokenizer blob; see special_ids_note\"";
  cj += ",\"act_head is computed and DISCARDED: Laya's own issue tracker "
        "records action.act_probability at AUROC 0.30 against 0.77 for "
        "confidence, so it is dead weight. It is computed -- it is in the "
        "checkpoint, and skipping it would be a silent divergence -- and its "
        "result is not returned.\"";
  cj += ",\"the decision head runs on the HOST, not the array: 8 of this "
        "model's 96 GEMMs, and the four obstacles are process-wide geometry "
        "globals, a hardcoded 'layer.' tensor prefix, one hw_context per "
        "npu::Design, and one design set per model -- none of which is about "
        "arithmetic\"";
  cj += ",\"layer_norm_eps is recorded at " + eps_raw +
        " for provenance while layer_norm_cpu() uses a HARDCODED 1e-12. The "
        "delta is ~5e-6 relative, far under the bf16 operand's 8-bit mantissa, "
        "but a container is not allowed to claim a precision the runtime does "
        "not implement.\"";
  cj += "]}";

  Writer w;
  auto add_f32 = [&](const std::string &name, const Tensor &t,
                     const char *role, const std::vector<int64_t> &shape) {
    w.add(name, t.data, static_cast<size_t>(t.count()) * 4, "F32", role, shape);
  };
  auto add_zeros = [&](const std::string &name, const char *role,
                       const std::vector<int64_t> &shape) {
    size_t n = 1;
    for (int64_t d : shape) n *= static_cast<size_t>(d);
    const std::vector<float> z(n, 0.f);
    w.add(name, z.data(), n * 4, "F32", role, shape);
  };

  // -- embeddings ---------------------------------------------------------
  // `embeddings.word` is the NAME the runtime reads and casts to float; the
  // checkpoint's own key is `embeddings.tok_embeddings.weight`, so the packer
  // RENAMES ON PURPOSE and a packer that emits the checkpoint's key loads
  // nothing. F32, not the checkpoint's F16: the runtime does
  // model_.raw("embeddings.word").as<float>(), and .as<float>() on a BF16
  // tensor reinterprets bytes rather than converting them.
  add_f32("embeddings.word", get("embeddings.tok_embeddings.weight"),
          "embedding", {vocab_size, hidden});
  add_zeros("embeddings.position", "embedding", {max_seq, hidden});
  add_zeros("embeddings.token_type", "embedding", {1, hidden});
  add_f32("embeddings.ln.weight", get("embeddings.norm.weight"), "layernorm",
          {hidden});
  add_zeros("embeddings.ln.bias", "layernorm", {hidden});
  w.add("tokenizer.bbpe_table", tb.data(), tb.size(), "U8", "tokenizer",
        {static_cast<int64_t>(tb.size())});
  if (log) {
    std::ostringstream s;
    s << (tok_generated
              ? "  generated tokenizer.bbpe_table (no cached tokenizer.bin)  "
              : "  tokenizer.bbpe_table  ")
      << (tb.size() / 1e6) << " MB (BBPETOK1 v2)";
    log(s.str());
  }

  // -- the decision head, then the encoder -------------------------------
  //
  // EMISSION ORDER IS LOAD-BEARING: embeddings.ln.weight, then the tokenizer
  // blob, then embeddings.ln.bias -- the same interleaving arch=0/2/3 use. It is
  // not cosmetic: Writer::add pads after every tensor, so moving one shifts
  // every offset after it.
  for (int64_t h = 0; h < head_layers; ++h) {
    const std::string p = "head.layers." + std::to_string(h) + ".";
    // norm1/norm2 are the pre-LN norms of the norm_first=True
    // nn.TransformerEncoderLayer; in_proj/out_proj/linear1/linear2 are
    // PyTorch's names for qkv / attn_out / ffn_up / ffn_down.
    add_f32(p + "norm1.weight", get(p + "norm1.weight"), "layernorm", {hidden});
    add_f32(p + "norm1.bias", get(p + "norm1.bias"), "layernorm", {hidden});
    add_f32(p + "self_attn.in_proj.weight",
            get(p + "self_attn.in_proj_weight"), "gemm_b_host",
            {3 * hidden, hidden});
    add_f32(p + "self_attn.in_proj.bias", get(p + "self_attn.in_proj_bias"),
            "bias", {3 * hidden});
    add_f32(p + "self_attn.out_proj.weight",
            get(p + "self_attn.out_proj.weight"), "gemm_b_host",
            {hidden, hidden});
    add_f32(p + "self_attn.out_proj.bias", get(p + "self_attn.out_proj.bias"),
            "bias", {hidden});
    add_f32(p + "norm2.weight", get(p + "norm2.weight"), "layernorm", {hidden});
    add_f32(p + "norm2.bias", get(p + "norm2.bias"), "layernorm", {hidden});
    // The head's FFN is ReLU, not the encoder's GELU: torch's DEFAULT
    // nn.TransformerEncoderLayer activation. The two are different functions,
    // the head is only 2 layers deep, and getting it wrong produces a model
    // that is still confidently wrong.
    add_f32(p + "linear1.weight", get(p + "linear1.weight"), "gemm_b_host",
            {4 * hidden, hidden});
    add_f32(p + "linear1.bias", get(p + "linear1.bias"), "bias", {4 * hidden});
    add_f32(p + "linear2.weight", get(p + "linear2.weight"), "gemm_b_host",
            {hidden, 4 * hidden});
    add_f32(p + "linear2.bias", get(p + "linear2.bias"), "bias", {hidden});
  }
  // type_emb is broadcast over the sequence BEFORE the head and the marker gather
  // is after it, so its row order is the QTYPES order: {choice:0, score:1,
  // noul:2}.
  add_f32("type_emb.weight", get("type_emb.weight"), "embedding", {3, hidden});
  // scorer: LayerNorm -> Linear(d,d) -> GELU -> Linear(d,1). N=1 for the last
  // layer, which fails N % (tile_n*8) == 0 at EVERY legal tile, and K=772 for
  // act_head.0, which fails K % 64 == 0. The whole tail is host arithmetic,
  // and it runs on at most a couple of dozen MARKER positions rather than on
  // rows, so there is nothing to tile in the first place.
  add_f32("scorer.0.weight", get("scorer.0.weight"), "layernorm", {hidden});
  add_f32("scorer.0.bias", get("scorer.0.bias"), "layernorm", {hidden});
  add_f32("scorer.1.weight", get("scorer.1.weight"), "gemm_b_host",
          {hidden, hidden});
  add_f32("scorer.1.bias", get("scorer.1.bias"), "bias", {hidden});
  add_f32("scorer.3.weight", get("scorer.3.weight"), "gemm_b_host", {1, hidden});
  add_f32("scorer.3.bias", get("scorer.3.bias"), "bias", {1});
  // act_head.0's K is d + 4 = 772: forward() concatenates four hand-built
  // features onto h[:,0] first (top1, top1-top2, normalised entropy, k/255.0).
  // The 255 is not a coincidence with the 255-option cap on a choice: the
  // feature is calibrated against it.
  add_f32("act_head.0.weight", get("act_head.0.weight"), "gemm_b_host",
          {256, hidden + 4});
  add_f32("act_head.0.bias", get("act_head.0.bias"), "bias", {256});
  add_f32("act_head.2.weight", get("act_head.2.weight"), "gemm_b_host",
          {n_act, 256});
  add_f32("act_head.2.bias", get("act_head.2.bias"), "bias", {n_act});

  // final_norm, after the last layer. Its OUTPUT is what the head consumes:
  // upstream reads self.encoder(...).last_hidden_state, which is after
  // final_norm, so the head's input is the encoder's output and not the residual
  // stream from inside the loop.
  add_f32("final_norm.weight", get("final_norm.weight"), "layernorm", {hidden});
  add_zeros("final_norm.bias", "layernorm", {hidden});

  for (int64_t l = 0; l < L; ++l) {
    const std::string tag = "layer." + std::to_string(l) + ".";
    // RELATIVE to the `encoder.` prefix `get()` adds -- ModernBERT's own naming,
    // with no prefix of its own written here.
    const std::string p = "layers." + std::to_string(l) + ".";

    // qkv is FUSED upstream as [3H, H] in [Q|K|V] order -- ModernBERT's own
    // layout, which is (3, Nh, Dh) and NOT the interleaved (Nh, 3, Dh) some
    // transformers write. 1/sqrt(head_dim) folded into the Q block (the first
    // `hidden` columns of the transposed [H, 3H] operand); exact, RoPE is linear
    // in q, and bias-free so there is no bias third to scale.
    add_gemm_b(w, tag + "qkv", get(p + "attn.Wqkv.weight"), tile_k, tile_n,
               layout_json, layout_hash, scale, hidden);
    add_zeros(tag + "qkv.bias", "bias", {3 * hidden});

    add_gemm_b(w, tag + "attn_out", get(p + "attn.Wo.weight"), tile_k, tile_n,
               layout_json, layout_hash);
    add_zeros(tag + "attn_out.bias", "bias", {hidden});

    // Layer 0's attn_norm IS nn.Identity() -- there is no tensor for it in the
    // checkpoint at all. A zero-filled placeholder plus
    // identity_attn_norm_layer0 lets the runtime SKIP the norm, which is not the
    // same thing: a norm with weight 1 still subtracts the mean and divides by
    // the standard deviation. The bias is emitted anyway, because stage_all
    // dereferences <weight> AND <bias> for every layer unconditionally.
    if (l == 0) {
      add_zeros(tag + "ln1.weight", "layernorm", {hidden});
      add_zeros(tag + "ln1.bias", "layernorm", {hidden});
    } else {
      add_f32(tag + "ln1.weight", get(p + "attn_norm.weight"), "layernorm",
              {hidden});
      add_zeros(tag + "ln1.bias", "layernorm", {hidden});
    }
    add_f32(tag + "ln2.weight", get(p + "mlp_norm.weight"), "layernorm",
            {hidden});
    add_zeros(tag + "ln2.bias", "layernorm", {hidden});

    // THE GATE HALF MOVE. See add_gemm_b_reorder_rows' header: ModernBERT's gate
    // is the SECOND half of Wi and this runtime's activation applies to the
    // second half, so the packer puts the gate first. r0 = rows = intermediate,
    // DERIVED from the config rather than written as 1152, because intermediate
    // is the one number a differently-sized mmBERT would change and a hardcoded
    // literal would pack the wrong halves in the right order.
    add_gemm_b_reorder_rows(w, tag + "ffn_up", get(p + "mlp.Wi.weight"), inter, inter,
                            tile_k, tile_n, layout_json, layout_hash);
    add_zeros(tag + "ffn_up.bias", "bias", {2 * inter});

    add_gemm_b(w, tag + "ffn_down", get(p + "mlp.Wo.weight"), tile_k, tile_n,
               layout_json, layout_hash);
    add_zeros(tag + "ffn_down.bias", "bias", {hidden});
  }

  w.write(out, cj, kArchModernBertRopeGeGLU);
  if (log) {
    std::ostringstream s;
    s << "\n  tensors    : " << w.count()
      << "\n  data       : " << (w.data_bytes() / 1e6) << " MB"
      << "\n  source     : " << sha.substr(0, 16) << "...";
    log(s.str());
  }
}

std::string prepare_model_auto(const PrepareOptions &opt) {
  namespace fs = std::filesystem;
  if (opt.checkpoint_dir.empty())
    throw std::runtime_error("prepare_model_auto: no checkpoint directory");

  std::string out = opt.out_path;
  if (out.empty())
    out = opt.checkpoint_dir + "/" +
          fs::path(opt.checkpoint_dir).filename().string() + ".npue";

  // A nested checkpoint puts its files somewhere this packer is not looking, so
  // every path is composed ONCE, here, and every branch below reads the same
  // composed roots. `checkpoint_subdir` is where the CHECKPOINT is; the other
  // two are relative to it. All three default to the flat layout, so every
  // existing caller stays byte-identical.
  const std::string ckpt_dir = sub(opt.checkpoint_dir, opt.checkpoint_subdir, "");
  const std::string cfg_path = sub(ckpt_dir, opt.config_subdir, "config.json");
  const std::string model_type = json_string_field(cfg_path, "model_type");

  // FAIL CLOSED on an architecture this build does not pack, in TWO arms,
  // because the two failures are different and neither is the BERT fallback's.
  //
  // json_string_field() returns the EMPTY string for a file it cannot open,
  // cannot parse, has no `model_type` in, or has a non-string one -- and it does
  // not throw. So `model_type == ""` is three distinct situations collapsed into
  // one value, and the one Laya actually hits is the first: it ships no
  // config.json at its root or under `multilingual/`, because the real tree is
  // multilingual/{encoder/config.json, model.safetensors, tokenizer/...}.
  // Left alone, that falls through to the BERT LAST branch, whose first act is
  // slurp(config.json) and slurp(vocab.txt) -- so the user gets "cannot open
  // <dir>/config.json", naming a file the checkpoint does not have in a
  // directory it does not name.
  if (model_type.empty())
    throw std::runtime_error(
        "no usable \"model_type\" in " + cfg_path +
        ". That file is missing, unparseable, or has no string \"model_type\"; "
        "this packer reads the config from the checkpoint root and refuses to "
        "guess a subdirectory. If the checkpoint nests its config (e.g. "
        "<dir>/encoder/config.json), point \"npue_checkpoint_subdir\" in the "
        "model entry at the directory that holds config.json, model.safetensors "
        "and any 1_Pooling/config.json -- or pass PrepareOptions::"
        "config_subdir. See specs/open-engine/plans/laya-decision-encoder.md.");
  // arch=1 (EmbeddingGemma / Gemma3 family): a completely different tensor
  // shape and container, routed to its own packer rather than threaded through
  // the BERT logic below. It resolves source_repo and the two tile knobs and
  // nothing else -- no pooling (its own is fixed) and no layout hash.
  // tasks/0065.
  if (model_type == "gemma3_text") {
    const std::string repo = resolve_source_repo(opt);
    say(opt, "  source     " + repo);
    say(opt, "NpuEmbeddings -- preparing " + out);
    prepare_model_gemma(opt.checkpoint_dir, out, repo, opt.log,
                        opt.tile_k, opt.tile_n, opt.gemma_host_only);
    say(opt, "  wrote " + out);
    return out;
  }

  // Everything below shares tile size, layout, pooling and source_repo; only
  // the packer differs. That is why they are resolved once, here, rather than
  // per branch.
  const Layout lay = gemm_b_layout(opt.tile_k, opt.tile_n);
  say(opt, "  layout     tile (" + std::to_string(opt.tile_k) + ", " +
               std::to_string(opt.tile_n) + "), hash " +
               lay.hash.substr(0, 16) + "...");
  // resolve_pooling reads <dir>/1_Pooling/config.json, which lives at the
  // CHECKPOINT root, so it gets the rooted view rather than the served one.
  PrepareOptions rooted = opt;
  rooted.checkpoint_dir = ckpt_dir;
  rooted.checkpoint_subdir.clear();
  // ModernBERT carries no 1_Pooling/config.json, because it has no sentence
  // embedding head to configure: this model pools by GATHER at the [MASK]
  // markers and the decision engine never reaches pool_rows at all. The key
  // still has to name one of the two modes apply_model_shape accepts, and
  // "mean" is the one whose recorded note says it is a fiction.
  //
  // Gated on model_type rather than on the file being absent, because the
  // absence is what the OTHER five arches refuse on: a BERT checkpoint whose
  // 1_Pooling went missing must not quietly become a mean-pooler.
  const std::string pooling =
      model_type == "modernbert" ? std::string("mean") : resolve_pooling(rooted);
  say(opt, "  pooling    " + pooling +
               (model_type == "modernbert"
                    ? " (FICTION -- ModernBERT pools by marker gather; see "
                      "pooling_note in the container)"
                    : " (from 1_Pooling/config.json)"));
  const std::string repo = resolve_source_repo(rooted);
  say(opt, "  source     " + repo);

  // arch=2 (nomic-embed-text-v1.5): RoPE + gated SwiGLU rather than BERT's
  // absolute-position + GELU. tasks/0071.
  if (model_type == "nomic_bert") {
    say(opt, "NpuEmbeddings -- preparing " + out +
                 " (arch=nomic_bert_rope_swiglu)");
    prepare_model_nomic(ckpt_dir, pooling, repo, out, lay.json, lay.hash,
                        opt.tile_k, opt.tile_n, 256, opt.log);
    say(opt, "  wrote " + out);
    return out;
  }

  // arch=3 (gte-multilingual-base): model_type "new". max_seq 64 matches the
  // Python-packed container this mirror is held byte-identical to (tasks/0135
  // packed --max-seq 64; under RoPE the position table is zeros, so max_seq
  // only caps request length). tasks/0138.
  if (model_type == "new") {
    say(opt, "NpuEmbeddings -- preparing " + out +
                 " (arch=gte_new_rope_geglu)");
    prepare_model_gte(ckpt_dir, pooling, repo, out, lay.json, lay.hash,
                      opt.tile_k, opt.tile_n, 64, opt.log);
    say(opt, "  wrote " + out);
    return out;
  }

  // arch=4 (ModernBERT / mmBERT: pre-LN, RoPE, a gated GeGLU whose gate half is
  // SECOND upstream, bidirectional with a sliding window on most layers).
  //
  // This branch and encoder_implemented()'s `modernbert_rope_geglu` arm land in
  // the SAME change, both or neither. A container the packer can write but the
  // runtime refuses is a wasted pack; a runtime that accepts an arch no packer
  // produces is the arch-0 fail-open the guard above exists to close.
  if (model_type == "modernbert") {
    say(opt, "NpuEmbeddings -- preparing " + out +
                  " (arch=modernbert_rope_geglu)");
    prepare_model_modernbert(ckpt_dir, pooling, repo, out, lay.json, lay.hash,
                             opt.tile_k, opt.tile_n, opt.modernbert_max_seq,
                             opt.config_subdir, opt.tokenizer_subdir,
                             sub(ckpt_dir, "", "rl_agent_config.json"),
                             opt.log);
    say(opt, "  wrote " + out);
    return out;
  }

  // arch=0 (BERT family) is the LAST branch rather than a catch-all guess: an
  // unrecognised model_type reaching here is packed as BERT, which is correct
  // for the family (bert, xlm-roberta and the sentence-transformers forks all
  // report their own names) and is where a genuinely new architecture will
  // first show up as a wrong answer rather than an error. The container's
  // `arch` field is what the runtime later refuses on, which is the guard that
  // catches it.
  say(opt, "NpuEmbeddings -- preparing " + out);
  prepare_model(sub(ckpt_dir, "", "model.safetensors"),
                sub(ckpt_dir, "", "vocab.txt"),
                sub(ckpt_dir, opt.config_subdir, "config.json"), pooling, repo,
                out, "", lay.json, lay.hash, opt.tile_k, opt.tile_n, 256,
                opt.log);
  say(opt, "  wrote " + out);
  return out;
}

}  // namespace npue
