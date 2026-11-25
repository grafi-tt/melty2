// SPDX-License-Identifier: Apache-2.0
//
// Counter-differential bias metric.
//
// For a key k and random base counters c, the outputs at every prefix depth m are compared between
// the counters c and c + delta, for delta = 2^j with j in 0..7 (lower word) and 32..39 (upper word).
// The top 8 bits of rotl(diff, 4w) are histogrammed for the addition difference and the xor
// difference, and for the 8 windows w = 0..7. The bias of the key at the depth m is the sum over
// (delta, kind, window) of (chi2 - 255) / n, where n is the number of counter pairs per delta.
// For an ideal generator, the bias is 0 with a standard deviation noise(n).
//
// bias_per_key is implemented for CPU (metric_cpu.cc) and GPU (metric_gpu.hip, HIP); programs are
// linked with either of them.
#ifndef METRIC_HPP_
#define METRIC_HPP_

#include <cmath>
#include <cstdint>
#include <vector>

#include "arx.hpp"

namespace metric {

using arx::Key;

inline constexpr int kDeltas = 16;  // the deltas 2^0..2^7 and 2^32..2^39
inline constexpr int kWindows = 8;  // the windows of the top 8 bits of rotl(diff, 4w)

// The bias of every key at every depth m in [mlo, mhi] (1 <= mlo <= mhi <= a.size()), with n counter
// pairs per key and delta: result[key * (mhi - mlo + 1) + (m - mlo)].
std::vector<double> bias_per_key(const arx::Arx &a, int mlo, int mhi, const std::vector<Key> &keys, int n, uint64_t seed);

// standard deviation of the bias of one key for an ideal generator
inline double noise(int n) { return std::sqrt(2.0 * 255 * kDeltas * 2 * kWindows) / n; }

// nk random keys from the seed. If weak_words > 0, every 4th key (index 3 mod 4) has a zero word, and
// every 4th key (index 2 mod 4) has a word equal to a rotation of another word, among the first
// weak_words words.
std::vector<Key> random_keys(int nk, uint64_t seed, int weak_words);

// The keys of random_keys, except that the keys of index 1 mod 4 have sparse v[4] and v[5] (both zero,
// both of weight 1, both of weight 1 to 4, or zero or all-one v[4] and v[5] of weight 1 to 4, in turn),
// which are the most sensitive keys of the melty2 kernel.
std::vector<Key> sparse45_keys(int nk, uint64_t seed, int weak_words);

// the mean and the maximum over the keys of the bias at every depth in [mlo, mhi], indexed by m - mlo
struct Profile {
    std::vector<double> mean, max;
};

// the profile over nk random keys (see random_keys)
Profile profile(const arx::Arx &a, int mlo, int mhi, int nk, int n, uint64_t seed, int weak_words);

// SplitMix64
class SplitMix64 {
  public:
    explicit SplitMix64(uint64_t state) : state_(state) {}
    uint64_t operator()() {
        uint64_t z = (state_ += UINT64_C(0x9e3779b97f4a7c15));
        z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
        return z ^ (z >> 31);
    }

  private:
    uint64_t state_;
};

}  // namespace metric

#endif
