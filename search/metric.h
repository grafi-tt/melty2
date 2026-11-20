/* SPDX-License-Identifier: MIT */
/*
 * Counter-differential bias metric.
 *
 * For a key k and random base counters c, the outputs at every prefix depth m are compared between
 * the counters c and c + delta, for delta = 2^j with j in 0..7 (lower word) and 32..39 (upper word).
 * The top 8 bits of rotl(diff, 4w) are histogrammed for the addition difference and the xor
 * difference, and for the 8 windows w = 0..7. The bias of the key at the depth m is the sum over
 * (delta, kind, window) of (chi2 - 255) / n, where n is the number of counter pairs per delta.
 * For an ideal generator, the bias is 0 with a standard deviation metric_noise(n).
 *
 * The metric is implemented for CPU (metric_cpu.c) and GPU (metric_gpu.hip, HIP); programs are
 * linked with either of them.
 */
#ifndef METRIC_H_
#define METRIC_H_

#include "arx.h"

#ifdef __cplusplus
extern "C" {
#endif

/* per_key[key * (mhi - mlo + 1) + (m - mlo)] = bias of the key at the depth m, for m in [mlo, mhi] */
void metric_keys(const arx_t *a, int mlo, int mhi, const uint32_t *keys, int nk, int n, uint64_t seed, double *per_key);

/* standard deviation of the bias of one key for an ideal generator */
double metric_noise(int n);

/* nk random keys (6 words each) from the seed. If weak_words > 0, every 4th key (index 3 mod 4) has a
 * zero word, and every 4th key (index 2 mod 4) has a word equal to a rotation of another word, among
 * the first weak_words words. */
void metric_random_keys(uint32_t *keys, int nk, uint64_t seed, int weak_words);

/* mean and maximum over nk random keys (see metric_random_keys) of the bias at every depth in [mlo, mhi] */
void metric(const arx_t *a, int mlo, int mhi, int nk, int n, uint64_t seed, int weak_words, double *bias_mean, double *bias_max);

#ifdef __cplusplus
}
#endif

#endif
