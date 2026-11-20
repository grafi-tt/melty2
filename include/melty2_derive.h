#ifndef MELTY2_MELTY2_DERIVE_H_
#define MELTY2_MELTY2_DERIVE_H_

/*
 * Key derivation by SipHash-2-4 (Aumasson & Bernstein, 2012).
 *
 * The 128bit seed is used as the SipHash key, and the 192bit melty2 key is
 * the concatenation of SipHash-2-4(seed, m) for the 8 bytes messages
 * m = 0, 1, 2 (as little-endian 64bit integers).
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

MELTY2_INTERNAL_FUNC void melty2_internal_sipround(uint64_t v[4]) {
    v[0] += v[1];
    v[1] = v[1] << 13 | v[1] >> 51;
    v[1] ^= v[0];
    v[0] = v[0] << 32 | v[0] >> 32;
    v[2] += v[3];
    v[3] = v[3] << 16 | v[3] >> 48;
    v[3] ^= v[2];
    v[0] += v[3];
    v[3] = v[3] << 21 | v[3] >> 43;
    v[3] ^= v[0];
    v[2] += v[1];
    v[1] = v[1] << 17 | v[1] >> 47;
    v[1] ^= v[2];
    v[2] = v[2] << 32 | v[2] >> 32;
}

/* SipHash-2-4 with 64bit output for the 8 bytes message m (given as little-endian integer) */
MELTY2_INTERNAL_FUNC uint64_t melty2_internal_siphash24_u64(uint64_t key0, uint64_t key1, uint64_t m) {
    uint64_t v[4] = {
        UINT64_C(0x736f6d6570736575),
        UINT64_C(0x646f72616e646f6d),
        UINT64_C(0x6c7967656e657261),
        UINT64_C(0x7465646279746573),
    };
    v[0] ^= key0;
    v[1] ^= key1;
    v[2] ^= key0;
    v[3] ^= key1;

    /* the message block */
    v[3] ^= m;
    for (int i = 0; i < 2; ++i) {
        melty2_internal_sipround(v);
    }
    v[0] ^= m;

    /* the last block, which only has the message length (8) as there are no rest bytes */
    const uint64_t b = UINT64_C(8) << 56;
    v[3] ^= b;
    for (int i = 0; i < 2; ++i) {
        melty2_internal_sipround(v);
    }
    v[0] ^= b;

    /* finalization */
    v[2] ^= 0xff;
    for (int i = 0; i < 4; ++i) {
        melty2_internal_sipround(v);
    }
    return v[0] ^ v[1] ^ v[2] ^ v[3];
}

/* Store the 192bit key derived from the 128bit seed to v[0..5] */
MELTY2_INTERNAL_FUNC void melty2_internal_derive_key(uint32_t v[6], uint32_t seed0, uint32_t seed1, uint32_t seed2, uint32_t seed3) {
    /* SipHash key as little-endian 64bit integers of the 16 bytes seed0..seed3 */
    const uint64_t key0 = (uint64_t)seed0 | (uint64_t)seed1 << 32;
    const uint64_t key1 = (uint64_t)seed2 | (uint64_t)seed3 << 32;

    /* written without a loop, so that GPU compilers keep the key in registers */
    const uint64_t h0 = melty2_internal_siphash24_u64(key0, key1, 0);
    const uint64_t h1 = melty2_internal_siphash24_u64(key0, key1, 1);
    const uint64_t h2 = melty2_internal_siphash24_u64(key0, key1, 2);
    v[0] = (uint32_t)h0;
    v[1] = (uint32_t)(h0 >> 32);
    v[2] = (uint32_t)h1;
    v[3] = (uint32_t)(h1 >> 32);
    v[4] = (uint32_t)h2;
    v[5] = (uint32_t)(h2 >> 32);
}

#endif  /* MELTY2_MELTY2_DERIVE_H_ */
