#ifndef MELTY2_MELTY2_DERIVE_H_
#define MELTY2_MELTY2_DERIVE_H_

/*
 * Key derivation by the ChaCha8 block function (Bernstein, 2008).
 *
 * The ChaCha key (words 4..11 of the state) is zero, and the 128bit seed is
 * put on the counter and nonce (words 12..15). The 192bit melty2 key is the
 * words 4..9 of the output block. As the initial values of these words are
 * the zero key, the final addition of the input block is omitted.
 *
 * As the outputs are uniform, keys of distinct seeds are independent, and
 * related keys of melty2 (whose difference cancels with that of counter)
 * appear only with a negligible probability.
 *
 * This header is shared by the CPU library and the GPU header. Define
 * MELTY2_INTERNAL_FUNC before including it to add function qualifiers.
 */

#include <stdint.h>

#ifndef MELTY2_INTERNAL_FUNC
#define MELTY2_INTERNAL_FUNC static inline
#endif

MELTY2_INTERNAL_FUNC void melty2_internal_chacha_qround(uint32_t s[16], int a, int b, int c, int d) {
    s[a] += s[b];
    s[d] ^= s[a];
    s[d] = s[d] << 16 | s[d] >> 16;
    s[c] += s[d];
    s[b] ^= s[c];
    s[b] = s[b] << 12 | s[b] >> 20;
    s[a] += s[b];
    s[d] ^= s[a];
    s[d] = s[d] << 8 | s[d] >> 24;
    s[c] += s[d];
    s[b] ^= s[c];
    s[b] = s[b] << 7 | s[b] >> 25;
}

/* Store the 192bit key derived from the 128bit seed to v[0..5] */
MELTY2_INTERNAL_FUNC void melty2_internal_derive_key(uint32_t v[6], uint32_t seed0, uint32_t seed1, uint32_t seed2, uint32_t seed3) {
    /* "expand 32-byte k" as little-endian 32bit integers */
    const uint32_t sigma0 = (uint32_t)'e' | (uint32_t)'x' << 8 | (uint32_t)'p' << 16 | (uint32_t)'a' << 24;
    const uint32_t sigma1 = (uint32_t)'n' | (uint32_t)'d' << 8 | (uint32_t)' ' << 16 | (uint32_t)'3' << 24;
    const uint32_t sigma2 = (uint32_t)'2' | (uint32_t)'-' << 8 | (uint32_t)'b' << 16 | (uint32_t)'y' << 24;
    const uint32_t sigma3 = (uint32_t)'t' | (uint32_t)'e' << 8 | (uint32_t)' ' << 16 | (uint32_t)'k' << 24;
    uint32_t s[16] = {
        sigma0, sigma1, sigma2, sigma3,
        0, 0, 0, 0,
        0, 0, 0, 0,
        seed0, seed1, seed2, seed3,
    };

    /* 8 rounds, as 4 double rounds of column rounds and diagonal rounds */
    for (int i = 0; i < 4; ++i) {
        melty2_internal_chacha_qround(s, 0, 4, 8, 12);
        melty2_internal_chacha_qround(s, 1, 5, 9, 13);
        melty2_internal_chacha_qround(s, 2, 6, 10, 14);
        melty2_internal_chacha_qround(s, 3, 7, 11, 15);
        melty2_internal_chacha_qround(s, 0, 5, 10, 15);
        melty2_internal_chacha_qround(s, 1, 6, 11, 12);
        melty2_internal_chacha_qround(s, 2, 7, 8, 13);
        melty2_internal_chacha_qround(s, 3, 4, 9, 14);
    }

    for (int i = 0; i < 6; ++i) {
        v[i] = s[i + 4];
    }
}

#endif  /* MELTY2_MELTY2_DERIVE_H_ */
