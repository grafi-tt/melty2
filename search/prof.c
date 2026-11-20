/* SPDX-License-Identifier: MIT */
/* prof "structure" mlo mhi nkeys n [seed [weak_words]] : counter-differential bias profile at the depths mlo..mhi
 * over random keys (see metric.h). The margin of a structure is its length minus the last depth whose
 * maximum bias is clearly above the noise. */
#include "metric.h"

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: prof structure mlo mhi nkeys n [seed [weak_words]]\n");
        return 1;
    }
    arx_t a;
    arx_parse(&a, argv[1]);
    int mlo = atoi(argv[2]), mhi = atoi(argv[3]), nk = atoi(argv[4]), n = atoi(argv[5]);
    uint64_t seed = argc > 6 ? strtoull(argv[6], 0, 0) : 1;
    int weak = argc > 7 ? atoi(argv[7]) : 0;
    if (mlo < 1 || mhi > a.n || mlo > mhi) {
        fprintf(stderr, "depths must satisfy 1 <= mlo <= mhi <= %d\n", a.n);
        return 1;
    }
    double bm[MAXOPS], bx[MAXOPS];
    metric(&a, mlo, mhi, nk, n, seed, weak, bm, bx);
    printf("noise(1 key)=%.2e\n", metric_noise(n));
    for (int m = mlo; m <= mhi; ++m) printf("depth %2d  mean %.3e  max %.3e\n", m, bm[m - mlo], bx[m - mlo]);
    return 0;
}
