#include "melty2.h"

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
 */

#define MELTY2_ROTL64(x, s) ((x) << (s) | (x) >> (64 - (s)))

#define MELTY2_SIPROUND(v0, v1, v2, v3) ( \
    (v0) += (v1), (v1) = MELTY2_ROTL64((v1), 13), (v1) ^= (v0), (v0) = MELTY2_ROTL64((v0), 32), \
    (v2) += (v3), (v3) = MELTY2_ROTL64((v3), 16), (v3) ^= (v2), \
    (v0) += (v3), (v3) = MELTY2_ROTL64((v3), 21), (v3) ^= (v0), \
    (v2) += (v1), (v1) = MELTY2_ROTL64((v1), 17), (v1) ^= (v2), (v2) = MELTY2_ROTL64((v2), 32) \
)

/* SipHash-2-4 with 64bit output for a 8 bytes message m (given as little-endian integer) */
static uint64_t siphash24_u64(uint64_t k0, uint64_t k1, uint64_t m) {
    uint64_t v0 = UINT64_C(0x736f6d6570736575) ^ k0;
    uint64_t v1 = UINT64_C(0x646f72616e646f6d) ^ k1;
    uint64_t v2 = UINT64_C(0x6c7967656e657261) ^ k0;
    uint64_t v3 = UINT64_C(0x7465646279746573) ^ k1;

    /* compression of the message block */
    v3 ^= m;
    MELTY2_SIPROUND(v0, v1, v2, v3);
    MELTY2_SIPROUND(v0, v1, v2, v3);
    v0 ^= m;

    /* last block: message length (8) in the top byte, no tail bytes */
    const uint64_t b = UINT64_C(8) << 56;
    v3 ^= b;
    MELTY2_SIPROUND(v0, v1, v2, v3);
    MELTY2_SIPROUND(v0, v1, v2, v3);
    v0 ^= b;

    /* finalization */
    v2 ^= 0xff;
    MELTY2_SIPROUND(v0, v1, v2, v3);
    MELTY2_SIPROUND(v0, v1, v2, v3);
    MELTY2_SIPROUND(v0, v1, v2, v3);
    MELTY2_SIPROUND(v0, v1, v2, v3);
    return v0 ^ v1 ^ v2 ^ v3;
}

void melty2_init(melty2_key *key, uint32_t seed0, uint32_t seed1, uint32_t seed2, uint32_t seed3) {
    /* SipHash key as little-endian 64bit integers of the 16 bytes seed0..seed3 */
    uint64_t k0 = (uint64_t)seed0 | (uint64_t)seed1 << 32;
    uint64_t k1 = (uint64_t)seed2 | (uint64_t)seed3 << 32;

    for (int i = 0; i < 3; ++i) {
        uint64_t h = siphash24_u64(k0, k1, (uint64_t)i);
        key->v_[2 * i] = (uint32_t)h;
        key->v_[2 * i + 1] = (uint32_t)(h >> 32);
    }
}
