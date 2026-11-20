/* SPDX-License-Identifier: MIT */
/*
 * weakkeys "structure" mlo mhi n list [nbase [dump]] : differential bias of structured keys.
 *
 * Every line of the list file is a key modification spec (see keymod.h), applied to the keys of the
 * seeds 1..nbase (default 2). 64 keys of the seeds 1000..1063 are the control. For every depth, the
 * control distribution, the number of structured keys above the control maximum, and the worst 8
 * structured keys are printed. With dump, the bias of every key at every depth is written to the file
 * as tab-separated values.
 */
#include "keymod.h"
#include "melty2.h"
#include "metric.h"

#define NCTL 64
#define MAXSPEC 128

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: weakkeys structure mlo mhi n list [nbase [dump]]\n");
        return 1;
    }
    arx_t a;
    arx_parse(&a, argv[1]);
    int mlo = atoi(argv[2]), mhi = atoi(argv[3]), n = atoi(argv[4]);
    const char *list = argv[5];
    int nbase = argc > 6 ? atoi(argv[6]) : 2;
    const char *dump = argc > 7 ? argv[7] : 0;

    int cap = 1024, nm = 0;
    char (*mods)[MAXSPEC] = malloc(cap * MAXSPEC);
    FILE *f = fopen(list, "r");
    if (!f) {
        perror(list);
        return 1;
    }
    char line[MAXSPEC];
    while (fgets(line, sizeof line, f)) {
        line[strcspn(line, "\r\n")] = 0;
        if (!*line || *line == '#') continue;
        if (nm == cap) mods = realloc(mods, (cap *= 2) * MAXSPEC);
        strcpy(mods[nm++], line);
    }
    fclose(f);

    int nk = NCTL + nm * nbase, nd = mhi - mlo + 1;
    uint32_t *keys = malloc(sizeof(uint32_t) * 6 * nk);
    for (int c = 0; c < NCTL; ++c) {
        melty2_key k;
        melty2_init(&k, 1000 + c, 0, 0, 0);
        memcpy(keys + 6 * c, k.v_, 24);
    }
    for (int m = 0; m < nm; ++m)
        for (int b = 0; b < nbase; ++b) {
            melty2_key k;
            melty2_init(&k, 1 + b, 0, 0, 0);
            keymod_apply(k.v_, mods[m]);
            memcpy(keys + 6 * (NCTL + m * nbase + b), k.v_, 24);
        }
    double *pk = malloc(sizeof(double) * nk * nd);
    for (int s = 0; s < nk; s += 256) { /* in chunks, to bound the memory of the GPU metric */
        int c = nk - s < 256 ? nk - s : 256;
        metric_keys(&a, mlo, mhi, keys + 6 * s, c, n, 77, pk + (size_t)s * nd);
    }

    printf("noise(1 key)=%.2e  structured keys=%d x %d bases\n", metric_noise(n), nm, nbase);
    if (dump) {
        FILE *o = fopen(dump, "w");
        for (int k = 0; k < nk; ++k) {
            if (k < NCTL) fprintf(o, "CTRL\t%d", k);
            else fprintf(o, "%s\t%d", mods[(k - NCTL) / nbase], (k - NCTL) % nbase + 1);
            for (int d = 0; d < nd; ++d) fprintf(o, "\t%.4e", pk[(size_t)k * nd + d]);
            fprintf(o, "\n");
        }
        fclose(o);
    }
    for (int d = 0; d < nd; ++d) {
        double mx = -1e9, sum = 0;
        for (int c = 0; c < NCTL; ++c) {
            double v = pk[c * nd + d];
            sum += v;
            if (v > mx) mx = v;
        }
        double wmx = -1e9;
        int wi = -1, cnt = 0;
        for (int m = 0; m < nm * nbase; ++m) {
            double v = pk[(NCTL + m) * nd + d];
            if (v > wmx) { wmx = v; wi = m; }
            if (v > mx) cnt++;
        }
        printf("depth %2d  control mean %.2e max %.2e | structured max %.2e (%s) ; %d above control max\n",
               mlo + d, sum / NCTL, mx, wmx, wi >= 0 ? mods[wi / nbase] : "-", cnt);
    }
    for (int d = 0; d < nd && nm > 0; ++d) {
        double *col = malloc(sizeof(double) * nm * nbase);
        for (int m = 0; m < nm * nbase; ++m) col[m] = pk[(NCTL + m) * nd + d];
        for (int t = 0; t < 8 && t < nm * nbase; ++t) {
            int bi = 0;
            for (int m = 1; m < nm * nbase; ++m) if (col[m] > col[bi]) bi = m;
            printf("  worst@%d: %-24s base %d  bias %.2e\n", mlo + d, mods[bi / nbase], bi % nbase + 1, col[bi]);
            col[bi] = -1e9;
        }
        free(col);
    }
    return 0;
}
