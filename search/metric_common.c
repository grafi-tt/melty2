/* SPDX-License-Identifier: MIT */
#include "metric.h"

/* SplitMix64 */
static uint64_t sm_state;
static uint64_t sm_next(void) {
    uint64_t z = (sm_state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

void metric_random_keys(uint32_t *keys, int nk, uint64_t seed, int weak_words) {
    sm_state = seed * 0x2545f4914f6cdd1dULL + 12345;
    for (int key = 0; key < nk; ++key) {
        uint32_t *k = keys + 6 * key;
        for (int i = 0; i < 6; ++i) k[i] = (uint32_t)sm_next();
        if (weak_words && key % 4 == 3) k[sm_next() % weak_words] = 0;
        if (weak_words && key % 4 == 2) {
            int i = sm_next() % weak_words, j;
            do j = sm_next() % weak_words; while (j == i);
            int r = 1 + sm_next() % 31;
            k[i] = k[j] << r | k[j] >> (32 - r);
        }
    }
}

void metric(const arx_t *a, int mlo, int mhi, int nk, int n, uint64_t seed, int weak_words, double *bias_mean, double *bias_max) {
    uint32_t *keys = malloc(sizeof(uint32_t) * 6 * nk);
    metric_random_keys(keys, nk, seed, weak_words);
    int nd = mhi - mlo + 1;
    double *pk = malloc(sizeof(double) * nk * nd);
    metric_keys(a, mlo, mhi, keys, nk, n, seed, pk);
    for (int m = 0; m < nd; ++m) {
        bias_mean[m] = 0;
        bias_max[m] = 0;
    }
    for (int key = 0; key < nk; ++key)
        for (int m = 0; m < nd; ++m) {
            double v = pk[key * nd + m];
            bias_mean[m] += v / nk;
            if (v > bias_max[m]) bias_max[m] = v;
        }
    free(pk);
    free(keys);
}
