/* SPDX-License-Identifier: MIT */
/*
 * Structured ("weak") key modifications, applied to a key derived from a seed.
 * A spec is a comma-separated list of "i=OP" applied in order, where i is the word index (0..5):
 *   z: k_i = 0          o: k_i = ~0          bN: k_i = 1 << N
 *   eJ: k_i = k_J       nJ: k_i = -k_J       mJ: k_i = k_J ^ 0x80000000     rJ:R: k_i = rotl(k_J, R)
 *   hV: k_i = V         aV: k_i &= V         qV: k_i |= V                   xV: k_i ^= V   (V: C integer literal)
 *   dJ:C: k_i = k_J + C (C: signed decimal)
 * e.g. "4=z,5=a0x0000ffff" makes k_4 zero and the upper half of k_5 zero.
 * Programs take the spec from the environment variable MELTY2_KEYMOD by keymod().
 */
#ifndef KEYMOD_H_
#define KEYMOD_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void keymod_apply(uint32_t *k, const char *p) {
    if (!p) return;
    while (*p) {
        int i = strtol(p, (char **)&p, 10);
        if (*p++ != '=' || i < 0 || i > 5) {
            fprintf(stderr, "invalid key modification spec\n");
            exit(1);
        }
        char c = *p++;
        int j = 0, r = 0;
        if (c == 'h' || c == 'a' || c == 'q' || c == 'x') {
            uint32_t v = (uint32_t)strtoul(p, (char **)&p, 0);
            if (c == 'h') k[i] = v;
            else if (c == 'a') k[i] &= v;
            else if (c == 'q') k[i] |= v;
            else k[i] ^= v;
        } else if (c == 'd') {
            j = strtol(p, (char **)&p, 10);
            if (*p == ':') ++p;
            long cc = strtol(p, (char **)&p, 10);
            k[i] = k[j] + (uint32_t)cc;
        } else {
            if (c != 'z' && c != 'o') j = strtol(p, (char **)&p, 10);
            if (c == 'r' && *p == ':') {
                ++p;
                r = strtol(p, (char **)&p, 10);
            }
            switch (c) {
            case 'z': k[i] = 0; break;
            case 'o': k[i] = ~0u; break;
            case 'b': k[i] = 1u << j; break;
            case 'e': k[i] = k[j]; break;
            case 'n': k[i] = -k[j]; break;
            case 'm': k[i] = k[j] ^ 0x80000000u; break;
            case 'r': k[i] = r ? (k[j] << r | k[j] >> (32 - r)) : k[j]; break;
            default: fprintf(stderr, "invalid key modification '%c'\n", c); exit(1);
            }
        }
        if (*p == ',') ++p;
    }
}

static inline void keymod(uint32_t *k) { keymod_apply(k, getenv("MELTY2_KEYMOD")); }

#endif
