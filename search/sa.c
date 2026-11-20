/* SPDX-License-Identifier: MIT */
/*
 * sa L seed iters "init" [nkeys n weak_words] : simulated annealing of an L-op structure.
 *
 * The first L ops of init are the initial structure, and the number of rotations is kept. A move
 * changes the destination or the source of an op, the rotation amount (or add <-> xor), swaps two
 * ops, or moves a rotation to another op; one or two moves are applied per iteration. Invalid
 * structures (see arx_valid) are rejected.
 *
 * The objective is the sum of log10(max(bias_max(m), T)) over the last W + 1 depths m, where T is
 * twice the noise level (lower is better). It is evaluated on nkeys random keys, a quarter of which
 * have a zero word and another quarter a rotation relation (weak_words > 0).
 *
 * The defaults (16 keys, 2^20 pairs, weak_words = 6) are those used to find the melty2 kernel with
 * the GPU metric; e.g. the kernel is the result of "sa 20 504 3000 ORIG22".
 */
#include <math.h>

#include "metric.h"

#define W 10

static uint64_t rng = 1;
/* xorshift64 */
static inline uint64_t rr(void) {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return rng;
}

static int nk = 16, n = 1 << 20, weak = 6;
static double T;

static double objective(const arx_t *a, int nkeys, int npairs, uint64_t seed, double *prof) {
    double bm[MAXOPS], bx[MAXOPS];
    metric(a, a->n - W, a->n, nkeys, npairs, seed, weak, bm, bx);
    double f = 0;
    for (int i = 0; i <= W; ++i) {
        if (prof) prof[i] = bx[i];
        f += log10(fmax(bx[i], T));
    }
    return f;
}

static void mutate(arx_t *a) {
    int kind = rr() % 5, i = rr() % a->n, j;
    switch (kind) {
    case 0: /* change the destination */
        do a->d[i] = rr() % 6; while (a->d[i] == a->s[i]);
        break;
    case 1: /* change the source */
        do a->s[i] = rr() % 6; while (a->d[i] == a->s[i]);
        break;
    case 2: /* change the rotation amount, or flip add <-> xor */
        if (a->t[i] == 2) a->r[i] = 1 + rr() % 31;
        else a->t[i] ^= 1;
        break;
    case 3: /* swap two ops */
        j = rr() % a->n;
        {
            int t;
            t = a->t[i]; a->t[i] = a->t[j]; a->t[j] = t;
            t = a->d[i]; a->d[i] = a->d[j]; a->d[j] = t;
            t = a->s[i]; a->s[i] = a->s[j]; a->s[j] = t;
            t = a->r[i]; a->r[i] = a->r[j]; a->r[j] = t;
        }
        break;
    case 4: /* move a rotation to a non-rotation op, keeping the number of rotations */
        j = rr() % a->n;
        if (a->t[i] == 2 && a->t[j] != 2) {
            a->t[j] = 2;
            a->r[j] = a->r[i];
            a->t[i] = rr() % 2;
            a->r[i] = 0;
        }
        break;
    }
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: sa L seed iters init [nkeys n weak_words]\n"
                        "  init may be ORIG22 for the original 22-step kernel\n");
        return 1;
    }
    int L = atoi(argv[1]);
    rng = strtoull(argv[2], 0, 0) * 2654435761u + 1;
    int iters = atoi(argv[3]);
    const char *init = strcmp(argv[4], "ORIG22") == 0 ? ARX_MELTY2_ORIG22 : argv[4];
    if (argc > 5) nk = atoi(argv[5]);
    if (argc > 6) n = atoi(argv[6]);
    if (argc > 7) weak = atoi(argv[7]);
    arx_t cur;
    arx_parse(&cur, init);
    if (L <= W || L > cur.n) {
        fprintf(stderr, "L must satisfy %d < L <= (ops of init)\n", W);
        return 1;
    }
    cur.n = L;
    T = 2 * metric_noise(n);

    double fc = objective(&cur, nk, n, 1, 0), f0 = fc;
    arx_t best = cur;
    double fb = fc;
    for (int it = 0; it < iters; ++it) {
        double temp = 0.5 * (1.0 - (double)it / iters) + 0.02;
        arx_t cand = cur;
        int nm = 1 + rr() % 2;
        for (int q = 0; q < nm; ++q) mutate(&cand);
        if (!arx_valid(&cand)) continue;
        double f = objective(&cand, nk, n, 1 + it % 4, 0);
        if (f < fc || exp((fc - f) / temp) > (double)(rr() % 1000000) / 1e6) {
            cur = cand;
            fc = f;
        }
        if (fc < fb) {
            best = cur;
            fb = fc;
        }
    }

    /* verify the best structure with more keys and pairs */
    double prof[W + 1];
    double fv = objective(&best, 2 * nk, 4 * n, 12345, prof);
    int c1, c3, nr;
    arx_cost(&best, &c1, &c3, &nr);
    printf("valid=%d L=%d R=%d cost=%d/%d init=%.2f best=%.2f verify=%.2f prof(d%d..%d):",
           arx_valid(&best), best.n, nr, c1, c3, f0, fb, fv, L - W, L);
    for (int i = 0; i <= W; ++i) printf(" %.1e", prof[i]);
    printf("\n  ");
    arx_print(stdout, &best);
    printf("\n");
    return 0;
}
