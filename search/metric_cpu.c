/* SPDX-License-Identifier: MIT */
/* CPU implementation of metric_keys() in metric.h. The statistic is the same as the GPU version,
 * though the random counters differ, so the results agree only statistically. */
#include <math.h>

#include "metric.h"

#define LANES 64
#define NDELTA 16
#define NWIN 8

static uint64_t sm_state;
static uint64_t sm_next(void) {
    uint64_t z = (sm_state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

/* run the first mhi ops on LANES counters; out[m - mlo] receives the word written by op m - 1 */
static void trace(const arx_t *a, int mlo, int mhi, const uint32_t *k, const uint64_t *ctr, uint32_t out[][LANES]) {
    uint32_t v[6][LANES];
    for (int l = 0; l < LANES; ++l) {
        uint32_t lo = (uint32_t)ctr[l], hi = (uint32_t)(ctr[l] >> 32);
        v[0][l] = k[0] ^ lo; v[1][l] = k[1] + hi; v[2][l] = k[2] ^ hi; v[3][l] = k[3] + lo; v[4][l] = k[4]; v[5][l] = k[5];
    }
    for (int i = 0; i < mhi; ++i) {
        uint32_t *x = v[a->d[i]];
        const uint32_t *y = v[a->s[i]];
        int r = a->r[i];
        if (a->t[i] == 0) for (int l = 0; l < LANES; ++l) x[l] += y[l];
        else if (a->t[i] == 1) for (int l = 0; l < LANES; ++l) x[l] ^= y[l];
        else for (int l = 0; l < LANES; ++l) x[l] = arx_rotl(x[l] ^ y[l], r);
        if (i + 1 >= mlo) memcpy(out[i + 1 - mlo], x, sizeof(uint32_t) * LANES);
    }
}

void metric_keys(const arx_t *a, int mlo, int mhi, const uint32_t *keys, int nk, int n, uint64_t seed, double *per_key) {
    int nd = mhi - mlo + 1;
    static uint32_t h[MAXOPS][2][NWIN][256];
    static uint32_t o0[MAXOPS][LANES], o1[MAXOPS][LANES];
    uint64_t c0[LANES], c1[LANES];
    int nreal = (n + LANES - 1) / LANES * LANES;  /* n rounded up */
    sm_state = seed;
    for (int key = 0; key < nk; ++key) {
        const uint32_t *k = keys + 6 * key;
        double *kb = per_key + key * nd;
        for (int m = 0; m < nd; ++m) kb[m] = 0;
        for (int di = 0; di < NDELTA; ++di) {
            int j = di < 8 ? di : 32 + di - 8;
            uint64_t delta = 1ULL << j;
            memset(h, 0, sizeof(h[0]) * nd);
            for (int s = 0; s < nreal; s += LANES) {
                for (int l = 0; l < LANES; ++l) {
                    c0[l] = sm_next();
                    c1[l] = c0[l] + delta;
                }
                trace(a, mlo, mhi, k, c0, o0);
                trace(a, mlo, mhi, k, c1, o1);
                for (int m = 0; m < nd; ++m)
                    for (int l = 0; l < LANES; ++l) {
                        uint32_t ad = o1[m][l] - o0[m][l], xd = o1[m][l] ^ o0[m][l];
                        for (int w = 0; w < NWIN; ++w) {
                            h[m][0][w][arx_rotl(ad, 4 * w) >> 24]++;
                            h[m][1][w][arx_rotl(xd, 4 * w) >> 24]++;
                        }
                    }
            }
            double e = nreal / 256.0;
            for (int m = 0; m < nd; ++m)
                for (int q = 0; q < 2; ++q)
                    for (int w = 0; w < NWIN; ++w) {
                        double chi = 0;
                        for (int b = 0; b < 256; ++b) {
                            double d = h[m][q][w][b] - e;
                            chi += d * d;
                        }
                        kb[m] += (chi / e - 255) / nreal;
                    }
        }
    }
}

double metric_noise(int n) { return sqrt(2.0 * 255 * NDELTA * 2 * NWIN) / n; }
