/* SPDX-License-Identifier: MIT */
/* fperm "structure" M seed swap N r t : permutation test of N t-tuples of consecutive outputs, after
 * dropping the top r bits (the statistic of sknuth_Permutation of TestU01). Prints chi2, and the z
 * score and the upper-tail p-value by the Wilson-Hilferty approximation. */
#include <math.h>

#include "keymod.h"
#include "melty2.h"

#include "kern.h"

int main(int argc, char **argv) {
    if (argc < 8) {
        fprintf(stderr, "usage: fperm structure M seed swap N r t\n");
        return 1;
    }
    int M = atoi(argv[2]);
    uint32_t seed = strtoul(argv[3], 0, 0);
    int swp = atoi(argv[4]);
    long N = atol(argv[5]);
    int r = atoi(argv[6]), t = atoi(argv[7]);
    if (t < 2 || t > 8) {
        fprintf(stderr, "t must be in [2, 8]\n");
        return 1;
    }
    melty2_key key;
    melty2_init(&key, seed, 0, 0, 0);
    keymod(key.v_);
    int fact = 1;
    for (int i = 2; i <= t; ++i) fact *= i;
    uint64_t *cnt = calloc(fact, sizeof *cnt);
    uint32_t buf[LANES], u[8];
    int idx = LANES;
    uint64_t ctr = 0;
    for (long n = 0; n < N; ++n) {
        for (int j = 0; j < t; ++j) {
            if (idx == LANES) {
                block(key.v_, ctr, swp, buf);
                ctr += LANES;
                idx = 0;
            }
            u[j] = buf[idx++] << r;
        }
        int code = 0; /* Lehmer code of the order */
        for (int j = 0; j < t - 1; ++j) {
            int c = 0;
            for (int q = j + 1; q < t; ++q) c += u[q] < u[j];
            code = code * (t - j) + c;
        }
        cnt[code]++;
    }
    double e = (double)N / fact, chi = 0;
    for (int i = 0; i < fact; ++i) {
        double d = cnt[i] - e;
        chi += d * d / e;
    }
    double df = fact - 1;
    double z = (cbrt(chi / df) - (1 - 2 / (9 * df))) / sqrt(2 / (9 * df));
    double p = 0.5 * erfc(z / sqrt(2));
    printf("M=%d seed=%u swap=%d N=%ld r=%d t=%d chi2=%.1f z=%.2f p=%.3g\n", M, seed, swp, N, r, t, chi, z, p);
    return 0;
}
