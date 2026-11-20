/* SPDX-License-Identifier: MIT */
/* genlow "structure" M seed swap nbits [shift] : the lower nbits bits (after >> shift) of the outputs,
 * packed into 32bit words from the LSB, to stdout. The least significant bits of ARX outputs mix
 * slowly, as an addition has no carry into bit 0, so their stream is tested separately. */
#include "keymod.h"
#include "melty2.h"

#include "kern.h"

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: genlow structure M seed swap nbits [shift]\n");
        return 1;
    }
    uint32_t seed = strtoul(argv[3], 0, 0);
    int swp = atoi(argv[4]), nb = atoi(argv[5]), sh = argc > 6 ? atoi(argv[6]) : 0;
    if (nb < 1 || nb > 32 || 32 % nb != 0) {
        fprintf(stderr, "nbits must divide 32\n");
        return 1;
    }
    melty2_key key;
    melty2_init(&key, seed, 0, 0, 0);
    keymod(key.v_);
    uint32_t mask = nb == 32 ? ~0u : ((1u << nb) - 1);
    int per = 32 / nb; /* outputs per packed word */
    static uint32_t out[1 << 14];
    uint32_t buf[LANES];
    uint64_t ctr = 0;
    for (;;) {
        for (int o = 0; o < (1 << 14);) {
            block(key.v_, ctr, swp, buf);
            ctr += LANES;
            for (int base = 0; base < LANES; base += per, ++o) {
                uint32_t w = 0;
                for (int f = 0; f < per; ++f) w |= ((buf[base + f] >> sh) & mask) << (f * nb);
                out[o] = w;
            }
        }
        if (fwrite(out, 4, 1 << 14, stdout) != (1 << 14)) return 0;
    }
}
