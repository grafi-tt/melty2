/* SPDX-License-Identifier: MIT */
/*
 * ARX structure on the 6x32bit melty2 state.
 *
 * The counter (hi, lo) is injected as v0 ^= lo, v1 += hi, v2 ^= hi, v3 += lo, and then the ops run.
 * Op types: 0 (A) v[d] += v[s];  1 (X) v[d] ^= v[s];  2 (R) v[d] = rotl(v[d] ^ v[s], r).
 * Text form: "A d s", "X d s" and "R d s r" separated by ';', e.g. "R1 0 5;A2 1;...".
 * The output of a prefix of m ops is the word written by op m - 1.
 */
#ifndef ARX_H_
#define ARX_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXOPS 48

typedef struct {
    int n;
    int t[MAXOPS], d[MAXOPS], s[MAXOPS], r[MAXOPS];
} arx_t;

static inline uint32_t arx_rotl(uint32_t x, int r) { return r ? (x << r | x >> (32 - r)) : x; }

/* output of the prefix of m ops for the key k and the counter ctr */
static inline uint32_t arx_eval(const arx_t *a, int m, const uint32_t *k, uint64_t ctr) {
    uint32_t lo = (uint32_t)ctr, hi = (uint32_t)(ctr >> 32);
    uint32_t v[6] = {k[0] ^ lo, k[1] + hi, k[2] ^ hi, k[3] + lo, k[4], k[5]};
    int d = 0;
    for (int i = 0; i < m; ++i) {
        d = a->d[i];
        uint32_t s = v[a->s[i]];
        if (a->t[i] == 0) v[d] += s;
        else if (a->t[i] == 1) v[d] ^= s;
        else v[d] = arx_rotl(v[d] ^ s, a->r[i]);
    }
    return v[d];
}

/* instruction counts with and without a rotation instruction: 4 for the injection, 1 per op, and
 * 1 (rotate) or 3 (two shifts and or) more per rotation */
static inline void arx_cost(const arx_t *a, int *c_rot1, int *c_rot3, int *nrot) {
    int r = 0;
    for (int i = 0; i < a->n; ++i) r += a->t[i] == 2;
    *nrot = r;
    *c_rot1 = 4 + a->n + r;
    *c_rot3 = 4 + a->n + 3 * r;
}

/* Validity used by the search:
 * - every op affects the output, and every one of the six key words reaches the output;
 * - every key word passes through an addition (as either operand) before the output, so that no
 *   word only whitens the output by xor. */
static inline int arx_valid(const arx_t *a) {
    int need[6] = {0};
    need[a->d[a->n - 1]] = 1;
    for (int i = a->n - 1; i >= 0; --i) {
        if (!need[a->d[i]]) return 0;
        need[a->s[i]] = 1;
    }
    for (int j = 0; j < 6; ++j) if (!need[j]) return 0;
    /* dep[w]: initial words the value of w depends on; mixed[w]: those which have passed through an addition */
    int dep[6], mixed[6];
    for (int j = 0; j < 6; ++j) { dep[j] = 1 << j; mixed[j] = 0; }
    for (int i = 0; i < a->n; ++i) {
        int d = a->d[i], s = a->s[i];
        int nd = dep[d] | dep[s], nm = mixed[d] | mixed[s];
        if (a->t[i] == 0) nm |= dep[d] | dep[s];
        dep[d] = nd;
        mixed[d] = nm;
    }
    return mixed[a->d[a->n - 1]] == 63;
}

static inline int arx_parse(arx_t *a, const char *str) {
    a->n = 0;
    const char *p = str;
    while (*p) {
        while (*p == ' ' || *p == ';') ++p;
        if (!*p) break;
        if (a->n == MAXOPS) { fprintf(stderr, "too many ops (max %d)\n", MAXOPS); exit(1); }
        char c = *p++;
        int d = strtol(p, (char **)&p, 10), s = strtol(p, (char **)&p, 10), r = 0;
        if (c == 'R') r = strtol(p, (char **)&p, 10);
        a->t[a->n] = c == 'A' ? 0 : c == 'X' ? 1 : 2;
        a->d[a->n] = d;
        a->s[a->n] = s;
        a->r[a->n] = r;
        a->n++;
    }
    return a->n;
}

static inline void arx_print(FILE *f, const arx_t *a) {
    for (int i = 0; i < a->n; ++i) {
        if (a->t[i] == 0) fprintf(f, "A%d %d;", a->d[i], a->s[i]);
        else if (a->t[i] == 1) fprintf(f, "X%d %d;", a->d[i], a->s[i]);
        else fprintf(f, "R%d %d %d;", a->d[i], a->s[i], a->r[i]);
    }
}

/* the kernel of melty2 1.0 (20 steps), and the original 22-step kernel it was derived from */
#define ARX_MELTY2 "R1 0 5;A2 1;R3 2 9;A1 3;R5 1 13;A3 5;R2 3 30;A4 2;R1 4 17;A4 1;R3 4 25;A0 3;R1 0 5;R0 1 17;A5 0;R3 5 14;A3 2;R0 3 12;X3 0;A3 5"
#define ARX_MELTY2_ORIG22 "R1 0 7;A2 1;R3 2 9;A4 3;R5 4 13;A0 5;R2 0 18;A4 2;R1 4 16;A3 1;R3 5 12;A0 3;R1 0 8;A2 1;R3 2 7;A3 4;R0 3 9;A1 0;R2 1 13;A3 2;R0 3 18;A0 2"

#endif
