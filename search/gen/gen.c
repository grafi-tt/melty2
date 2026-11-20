/* SPDX-License-Identifier: MIT */
/* gen "structure" M seed swap : the output stream as 32bit little-endian words to stdout, e.g. for
 * "| RNG_test stdin32" of PractRand. The key is that of melty2_init(seed, 0, 0, 0), modified by
 * MELTY2_KEYMOD; with swap = 1, the halves of the counter are swapped. */
#include "keymod.h"
#include "melty2.h"

#include "kern.h"

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: gen structure M seed swap\n");
        return 1;
    }
    int swp = atoi(argv[4]);
    melty2_key key;
    melty2_init(&key, strtoul(argv[3], 0, 0), 0, 0, 0);
    keymod(key.v_);
    static uint32_t buf[1 << 16];
    uint64_t ctr = 0;
    for (;;) {
        for (int i = 0; i < (1 << 16); i += LANES, ctr += LANES) block(key.v_, ctr, swp, buf + i);
        if (fwrite(buf, 4, 1 << 16, stdout) != (1 << 16)) return 0;
    }
}
