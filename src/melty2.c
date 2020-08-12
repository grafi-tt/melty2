#include "melty2.h"

static uint32_t melty2_rotl(uint32_t v, int n) {
    return (v << n) | (v >> (32 - n));
}

static void melty2_inject(uint32_t *v, uint32_t a, uint32_t b) {
    v[0] += a;
    v[1] ^= a;
    v[2] += b;
    v[3] ^= b;
}

static void melty2_round(uint32_t *v) {
    v[0] = melty2_rotl(v[0], 7);
    v[1] -= v[0];
    v[2] ^= v[1];
    v[2] = melty2_rotl(v[2], 9);
    v[3] += v[2];
    v[4] ^= v[3];
    v[4] = melty2_rotl(v[4], 13);
    v[5] += v[4];
    v[0] ^= v[5];

    v[0] = melty2_rotl(v[0], 18);
    v[1] += v[0];
    v[2] ^= v[1];
    v[2] = melty2_rotl(v[2], 7);
    v[3] -= v[2];
    v[4] ^= v[3];
    v[4] = melty2_rotl(v[4], 9);
    v[5] += v[4];
    v[0] ^= v[5];

    v[0] = melty2_rotl(v[0], 13);
    v[1] += v[0];
    v[2] ^= v[1];
    v[2] = melty2_rotl(v[2], 18);
    v[3] += v[2];
    v[4] ^= v[3];
    v[4] = melty2_rotl(v[4], 12);
    v[5] -= v[4];
    v[0] ^= v[5];
}

uint32_t melty2_gen(const melty2_key *key, uint64_t idx) {
    uint32_t v[6] = {key->v_[0], key->v_[1], key->v_[2], key->v_[3], key->v_[4], key->v_[5]};
    melty2_inject(v, (uint32_t)idx, (uint32_t)(idx >> 32));
    melty2_round(v);
    return v[0] ^ v[3];
}
