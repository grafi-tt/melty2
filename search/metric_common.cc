// SPDX-License-Identifier: Apache-2.0
// The parts of metric.hpp shared by the CPU and the GPU metric.
#include <algorithm>

#include "metric.hpp"

namespace metric {

std::vector<Key> random_keys(int nk, uint64_t seed, int weak_words) {
    SplitMix64 next(seed * UINT64_C(0x2545f4914f6cdd1d) + 12345);
    std::vector<Key> keys(nk);
    const auto ww = static_cast<uint64_t>(weak_words);
    for (int key = 0; key < nk; ++key) {
        Key &k = keys[key];
        for (uint32_t &w : k) w = static_cast<uint32_t>(next());
        if (weak_words && key % 4 == 3) k[next() % ww] = 0;
        if (weak_words && key % 4 == 2) {
            const auto i = next() % ww;
            uint64_t j;
            do j = next() % ww;
            while (j == i);
            const int r = static_cast<int>(1 + next() % 31);
            k[i] = arx::rotl(k[j], r);
        }
    }
    return keys;
}

namespace {

// a word with 1 to 4 random bits set (fewer if the positions coincide), from the 64bit random r
uint32_t sparse_word(uint64_t r) {
    uint32_t w = 0;
    const int nbits = static_cast<int>(1 + r % 4);
    r >>= 2;
    for (int b = 0; b < nbits; ++b, r >>= 5) w |= UINT32_C(1) << (r % 32);
    return w;
}

}  // namespace

std::vector<Key> sparse45_keys(int nk, uint64_t seed, int weak_words) {
    std::vector<Key> keys = random_keys(nk, seed, weak_words);
    uint64_t r = seed * UINT64_C(0x9e3779b97f4a7c15) + 777;
    for (int key = 1; key < nk; key += 4) {
        Key &k = keys[key];
        r = r * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
        switch (key / 4 % 4) {
        case 0: k[4] = 0; k[5] = 0; break;
        case 1: k[4] = UINT32_C(1) << (r >> 32) % 32; k[5] = UINT32_C(1) << (r >> 40) % 32; break;
        case 2: k[4] = sparse_word(r); k[5] = sparse_word(r >> 23); break;
        default: k[4] = r >> 63 ? ~UINT32_C(0) : 0; k[5] = sparse_word(r >> 13); break;
        }
    }
    return keys;
}

Profile profile(const arx::Arx &a, int mlo, int mhi, int nk, int n, uint64_t seed, int weak_words) {
    const std::vector<Key> keys = random_keys(nk, seed, weak_words);
    const int nd = mhi - mlo + 1;
    const std::vector<double> pk = bias_per_key(a, mlo, mhi, keys, n, seed);
    Profile p{std::vector<double>(nd, 0.0), std::vector<double>(nd, 0.0)};
    for (int key = 0; key < nk; ++key)
        for (int m = 0; m < nd; ++m) {
            const double v = pk[key * nd + m];
            p.mean[m] += v / nk;
            p.max[m] = std::max(p.max[m], v);
        }
    return p;
}

}  // namespace metric
