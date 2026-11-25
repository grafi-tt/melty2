// SPDX-License-Identifier: Apache-2.0
// CPU implementation of metric::bias_per_key() in metric.hpp. The statistic is the same as the GPU
// version, though the random counters differ, so the results agree only statistically.
#include <array>
#include <cstring>

#include "metric.hpp"

namespace metric {

namespace {

constexpr int kLanes = 64;

using Lanes = std::array<uint32_t, kLanes>;

// run the first mhi ops on kLanes counters; out[m - mlo] receives the word written by op m - 1
void trace(const arx::Arx &a, int mlo, int mhi, const Key &k, const std::array<uint64_t, kLanes> &ctr, Lanes *out) {
    Lanes v[6];
    for (int l = 0; l < kLanes; ++l) {
        const uint32_t lo = static_cast<uint32_t>(ctr[l]), hi = static_cast<uint32_t>(ctr[l] >> 32);
        v[0][l] = k[0] ^ lo; v[1][l] = k[1] + hi; v[2][l] = k[2] ^ hi; v[3][l] = k[3] + lo; v[4][l] = k[4]; v[5][l] = k[5];
    }
    for (int i = 0; i < mhi; ++i) {
        const arx::Op &o = a[i];
        Lanes &x = v[o.d];
        const Lanes &y = v[o.s];
        switch (o.t) {
        case arx::kAdd: for (int l = 0; l < kLanes; ++l) x[l] += y[l]; break;
        case arx::kXor: for (int l = 0; l < kLanes; ++l) x[l] ^= y[l]; break;
        case arx::kSub: for (int l = 0; l < kLanes; ++l) x[l] -= y[l]; break;
        default: for (int l = 0; l < kLanes; ++l) x[l] = arx::rotl(x[l] ^ y[l], o.r); break;
        }
        if (i + 1 >= mlo) out[i + 1 - mlo] = x;
    }
}

}  // namespace

// See metric.hpp. For every key and delta, n (rounded up to a multiple of kLanes) random base counters
// c and c + delta are traced, and the differences of their outputs at every depth are histogrammed
// (h[depth][addition or xor][window][top 8 bits]). The chi-square statistic of every histogram,
// normalized as (chi2 - 255) / n, is summed into the bias of the key at the depth.
std::vector<double> bias_per_key(const arx::Arx &a, int mlo, int mhi, const std::vector<Key> &keys, int n, uint64_t seed) {
    const int nd = mhi - mlo + 1, nk = static_cast<int>(keys.size());
    using Hist = std::array<std::array<std::array<uint32_t, 256>, kWindows>, 2>;
    std::vector<Hist> h(nd);
    std::vector<Lanes> o0(nd), o1(nd);
    std::array<uint64_t, kLanes> c0, c1;
    const int nreal = (n + kLanes - 1) / kLanes * kLanes;  // n rounded up
    std::vector<double> per_key(static_cast<size_t>(nk) * nd, 0.0);
    SplitMix64 next(seed);
    for (int key = 0; key < nk; ++key) {
        const Key &k = keys[key];
        double *kb = per_key.data() + static_cast<size_t>(key) * nd;
        for (int di = 0; di < kDeltas; ++di) {
            const int j = di < 8 ? di : 32 + di - 8;
            const uint64_t delta = UINT64_C(1) << j;
            std::memset(h.data(), 0, sizeof(Hist) * nd);
            for (int s = 0; s < nreal; s += kLanes) {
                for (int l = 0; l < kLanes; ++l) {
                    c0[l] = next();
                    c1[l] = c0[l] + delta;
                }
                trace(a, mlo, mhi, k, c0, o0.data());
                trace(a, mlo, mhi, k, c1, o1.data());
                for (int m = 0; m < nd; ++m)
                    for (int l = 0; l < kLanes; ++l) {
                        const uint32_t ad = o1[m][l] - o0[m][l], xd = o1[m][l] ^ o0[m][l];
                        for (int w = 0; w < kWindows; ++w) {
                            h[m][0][w][arx::rotl(ad, 4 * w) >> 24]++;
                            h[m][1][w][arx::rotl(xd, 4 * w) >> 24]++;
                        }
                    }
            }
            const double e = nreal / 256.0;
            for (int m = 0; m < nd; ++m)
                for (const auto &hq : h[m])
                    for (const auto &hw : hq) {
                        double chi = 0;
                        for (const uint32_t c : hw) {
                            const double d = c - e;
                            chi += d * d;
                        }
                        kb[m] += (chi / e - 255) / nreal;
                    }
        }
    }
    return per_key;
}

}  // namespace metric
