/* SPDX-License-Identifier: MIT */
/* check : ARX_MELTY2 in arx.h computes the same outputs as the kernel of the library (melty2_kernel.h) */
#include "arx.h"
#include "melty2_kernel.h"

int main(void) {
    arx_t a;
    arx_parse(&a, ARX_MELTY2);
    uint64_t x = 1;
    long bad = 0, n = 1000000;
    for (long i = 0; i < n; ++i) {
        uint32_t k[6], v[6];
        for (int j = 0; j < 6; ++j) {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            k[j] = v[j] = (uint32_t)x;
        }
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        uint64_t ctr = x;
        uint32_t expected = melty2_kernel(v, (uint32_t)ctr, (uint32_t)(ctr >> 32));
        bad += arx_eval(&a, a.n, k, ctr) != expected;
    }
    printf("ARX_MELTY2 vs melty2_kernel.h: %ld mismatches of %ld\n", bad, n);
    return bad != 0;
}
