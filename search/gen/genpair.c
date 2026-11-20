/* SPDX-License-Identifier: MIT */
/* genpair "structure" M seed swap mask mode : a stream combining two keys which share the words in
 * mask (bit i set: word i is shared). Key A is from seed, and key B takes the other words from
 * seed + 0x10000. mode 0: A(c) ^ B(c); 1: A(c), B(c) interleaved; 2: A(c) - B(c). */
#include "keymod.h"
#include "melty2.h"

#include "kern.h"

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: genpair structure M seed swap mask mode\n");
        return 1;
    }
    uint32_t seed = strtoul(argv[3], 0, 0);
    int swp = atoi(argv[4]), mask = strtol(argv[5], 0, 0), mode = atoi(argv[6]);
    melty2_key a, b;
    melty2_init(&a, seed, 0, 0, 0);
    melty2_init(&b, seed + 0x10000, 0, 0, 0);
    keymod(a.v_);
    for (int i = 0; i < 6; ++i)
        if (mask >> i & 1) b.v_[i] = a.v_[i];
    static uint32_t out[1 << 14];
    uint32_t ba[LANES], bb[LANES];
    uint64_t ctr = 0;
    for (;;) {
        for (int o = 0; o < (1 << 14);) {
            block(a.v_, ctr, swp, ba);
            block(b.v_, ctr, swp, bb);
            ctr += LANES;
            for (int l = 0; l < LANES; ++l) {
                if (mode == 0) out[o++] = ba[l] ^ bb[l];
                else if (mode == 2) out[o++] = ba[l] - bb[l];
                else { out[o++] = ba[l]; out[o++] = bb[l]; }
            }
        }
        if (fwrite(out, 4, 1 << 14, stdout) != (1 << 14)) return 0;
    }
}
