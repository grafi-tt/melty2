// SPDX-License-Identifier: Apache-2.0
//
// ARX structure on the 6x32bit melty2 state.
//
// The counter (hi, lo) is injected as v0 ^= lo, v1 += hi, v2 ^= hi, v3 += lo, and then the ops run.
// Op types: Add: v[d] += v[s];  Xor: v[d] ^= v[s];  Rot: v[d] = rotl(v[d] ^ v[s], r);  Sub: v[d] -= v[s].
// Text form: "A d s", "X d s", "R d s r" and "S d s" separated by ';', e.g. "R1 0 5;A2 1;...".
// The output of a prefix of m ops is the word written by op m - 1.
#ifndef ARX_HPP_
#define ARX_HPP_

#include <array>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "number.hpp"

namespace arx {

// the maximum number of ops (the GPU metric keeps the ops in a constant array of this size)
inline constexpr int kMaxOps = 48;

// the six words of a key
using Key = std::array<uint32_t, 6>;

// the values are those of the packed ops of the GPU metric
enum OpType : int { kAdd = 0, kXor = 1, kRot = 2, kSub = 3 };

struct Op {
    int t = kAdd, d = 0, s = 0, r = 0;
    bool operator==(const Op &o) const { return t == o.t && d == o.d && s == o.s && r == o.r; }
    bool operator!=(const Op &o) const { return !(*this == o); }
    bool additive() const { return t == kAdd || t == kSub; }
};

using Arx = std::vector<Op>;

constexpr uint32_t rotl(uint32_t x, int r) { return r ? (x << r | x >> (32 - r)) : x; }

// output of the prefix of m ops for the key k and the counter ctr
inline uint32_t eval(const Arx &a, int m, const Key &k, uint64_t ctr) {
    const uint32_t lo = static_cast<uint32_t>(ctr), hi = static_cast<uint32_t>(ctr >> 32);
    uint32_t v[6] = {k[0] ^ lo, k[1] + hi, k[2] ^ hi, k[3] + lo, k[4], k[5]};
    int d = 0;
    for (int i = 0; i < m; ++i) {
        const Op &o = a[i];
        d = o.d;
        const uint32_t s = v[o.s];
        switch (o.t) {
        case kAdd: v[d] += s; break;
        case kXor: v[d] ^= s; break;
        case kSub: v[d] -= s; break;
        default: v[d] = rotl(v[d] ^ s, o.r); break;
        }
    }
    return v[d];
}

// instruction counts with and without a rotation instruction: 4 for the injection, 1 per op, and
// 1 (rotate) or 3 (two shifts and or) more per rotation
struct Cost {
    int rot1, rot3, nrot;
};
inline Cost cost(const Arx &a) {
    int r = 0;
    for (const Op &o : a) r += o.t == kRot;
    const int n = static_cast<int>(a.size());
    return {4 + n + r, 4 + n + 3 * r, r};
}

// Whether an addition, a subtraction or an xor of v[s] into v[d] is repeated without any write to v[d]
// or v[s] in between. Such ops merge into one op or cancel, so they are wasted:
// - additions and subtractions, e.g. "A3 4;A3 4" is v[3] += 2 * v[4] (losing the MSB of v[4]), and
//   "A3 4;S3 4" does nothing;
// - xors, e.g. "X3 4;X3 4" does nothing.
// valid() doesn't reject them, so that the search reproduces the kernel of melty2 1.0; sa -R does.
inline bool repeated(const Arx &a) {
    for (size_t j = 1; j < a.size(); ++j) {
        const Op &b = a[j];
        if (!b.additive() && b.t != kXor) continue;
        for (size_t i = j; i-- > 0;) {
            if (a[i].d == b.d) {
                const bool same_kind = b.additive() ? a[i].additive() : a[i].t == kXor;
                if (same_kind && a[i].s == b.s) return true;
                break;  // v[d] is changed by another op
            }
            if (a[i].d == b.s) break;  // v[s] is changed
        }
    }
    return false;
}

// Whether every key word propagates into the other words by the first k ops: it is read as a source
// operand (or a word depending on it is) by one of them. A key word read only in the last few ops is
// little more than a whitening of the output, even if it is read before the last op: e.g. with
// "A1 4;A1 3;X0 1" as the last 3 ops and no other use of v[4], the output is v0 ^ (v1 + k4 + v3), and
// the outputs of two keys differing only in k4 are related simply. The counter-differential metric
// doesn't see it, as the key is fixed; sa -D k rejects such structures.
inline bool keys_propagated_by(const Arx &a, int k) {
    int dep[6], read = 0;
    for (int j = 0; j < 6; ++j) dep[j] = 1 << j;
    for (int i = 0; i < k && i < static_cast<int>(a.size()); ++i) {
        read |= dep[a[i].s];
        dep[a[i].d] |= dep[a[i].s];
    }
    return read == 63;
}

// Whether every key word propagates into the other words: it is read as a source operand (or a word
// depending on it is) by an op before the last one. A key word which only receives a word at the last
// op, e.g. "A5 2" as the last op without any other use of v[5], just adds a constant of the key to the
// output, and hardly takes part in the mixing; the outputs of two keys differing only in that word
// differ by a constant. valid() doesn't reject such structures, as it would change the search
// reproducing the kernel of melty2 1.0; sa -P does.
inline bool keys_propagated(const Arx &a) { return keys_propagated_by(a, static_cast<int>(a.size()) - 1); }

// Validity used by the search:
// - every op affects the output, and every one of the six key words reaches the output;
// - every key word passes through an addition or a subtraction (as either operand) before the output,
//   so that no word only whitens the output by xor.
inline bool valid(const Arx &a) {
    if (a.empty()) return false;
    bool need[6] = {};
    need[a.back().d] = true;
    for (size_t i = a.size(); i-- > 0;) {
        if (!need[a[i].d]) return false;
        need[a[i].s] = true;
    }
    for (bool b : need)
        if (!b) return false;
    // dep[w]: initial words the value of w depends on; mixed[w]: those which have passed through an
    // addition or a subtraction
    int dep[6], mixed[6];
    for (int j = 0; j < 6; ++j) {
        dep[j] = 1 << j;
        mixed[j] = 0;
    }
    for (const Op &o : a) {
        const int nd = dep[o.d] | dep[o.s];
        int nm = mixed[o.d] | mixed[o.s];
        if (o.additive()) nm |= nd;
        dep[o.d] = nd;
        mixed[o.d] = nm;
    }
    return mixed[a.back().d] == 63;
}

// Parse the text form; spaces around the ops are ignored. Throws std::invalid_argument for a malformed
// op, a word index out of 0..5, a rotation amount out of 0..31, or more than kMaxOps ops.
inline Arx parse(std::string_view str) {
    Arx a;
    size_t p = 0;
    const auto bad = [&](std::string_view what) { throw std::invalid_argument(std::format("invalid structure ({}): {}", what, str)); };
    const auto skip = [&] {
        while (p < str.size() && std::isspace(static_cast<unsigned char>(str[p]))) ++p;
    };
    const auto number = [&](int lo, int hi) {
        skip();
        size_t q = p;
        while (q < str.size() && std::isdigit(static_cast<unsigned char>(str[q]))) ++q;
        if (q == p) bad("a number is expected");
        const int v = number::parse<int>(str.substr(p, q - p));
        if (v < lo || v > hi) bad("a number is out of range");
        p = q;
        return v;
    };
    for (;;) {
        while (p < str.size() && (str[p] == ';' || std::isspace(static_cast<unsigned char>(str[p])))) ++p;
        if (p == str.size()) break;
        if (static_cast<int>(a.size()) == kMaxOps) bad("too many ops");
        Op o;
        switch (str[p++]) {
        case 'A': o.t = kAdd; break;
        case 'X': o.t = kXor; break;
        case 'R': o.t = kRot; break;
        case 'S': o.t = kSub; break;
        default: bad("an op is A, X, R or S");
        }
        o.d = number(0, 5);
        o.s = number(0, 5);
        if (o.t == kRot) o.r = number(0, 31);
        skip();
        if (p < str.size() && str[p] != ';') bad("ops are separated by ';'");
        a.push_back(o);
    }
    return a;
}

// the text form, every op followed by ';'
inline std::string to_string(const Arx &a) {
    std::string s;
    for (const Op &o : a) {
        static constexpr char kName[] = {'A', 'X', 'R', 'S'};
        s += kName[o.t];
        s += std::to_string(o.d) + ' ' + std::to_string(o.s);
        if (o.t == kRot) s += ' ' + std::to_string(o.r);
        s += ';';
    }
    return s;
}

// the kernel of melty2 1.0 (20 steps), and the original 22-step kernel it was derived from
inline constexpr std::string_view kMelty2 =
    "R1 0 5;A2 1;R3 2 9;A1 3;R5 1 13;A3 5;R2 3 30;A4 2;R1 4 17;A4 1;R3 4 25;A0 3;R1 0 5;R0 1 17;A5 0;R3 5 14;A3 2;R0 3 12;X3 0;A3 5";
inline constexpr std::string_view kMelty2Orig22 =
    "R1 0 7;A2 1;R3 2 9;A4 3;R5 4 13;A0 5;R2 0 18;A4 2;R1 4 16;A3 1;R3 5 12;A0 3;R1 0 8;A2 1;R3 2 7;A3 4;R0 3 9;A1 0;R2 1 13;A3 2;R0 3 18;A0 2";

}  // namespace arx

#endif
