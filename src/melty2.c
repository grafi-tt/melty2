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

void melty2_initkey(melty2_key *key, const melty2_name *name, uint64_t global_seed, uint64_t global_ctr) {
    uint32_t v0[6];
    uint32_t v1[6];
    uint32_t v2[6];
    uint32_t v3[6];

    for (int i = 0; i < 6; ++i) {
        v0[i] = UINT32_C(0x6A09E667);
        v1[i] = UINT32_C(0xBB67AE85);
        v2[i] = UINT32_C(0x3C6EF372);
        v3[i] = UINT32_C(0xA54FF53A);
    }

    if (name) {
        melty2_inject(v0, name->v_[0], name->v_[4]);
        melty2_inject(v1, name->v_[1], name->v_[5]);
        melty2_inject(v2, name->v_[2], name->v_[6]);
        melty2_inject(v3, name->v_[3], name->v_[7]);
    }

    melty2_round(v0);
    melty2_round(v1);
    melty2_round(v2);
    melty2_round(v3);

    for (int i = 0; i < 6; ++i) {
        v0[i] ^= v2[i];
        v1[i] ^= v3[i];
    }

    melty2_inject(v0, (uint32_t)(global_seed >> 32), (uint32_t)global_seed);
    melty2_inject(v1, (uint32_t)(global_ctr >> 32), (uint32_t)global_ctr);

    melty2_round(v0);
    melty2_round(v1);

    for (int i = 0; i < 6; ++i) {
        key->v_[i] = v0[i] ^ v1[i];
    }

    melty2_round(key->v_);
}

void melty2_splitkey(melty2_key *key, melty2_key *newkey) {
    uint32_t v0[6];
    uint32_t v1[6];
    uint32_t v2[6];
    uint32_t v3[6];

    for (int i = 0; i < 6; ++i) {
        v0[i] = key->v_[i] ^ UINT32_C(0x6A09E667);
        v1[i] = key->v_[i] ^ UINT32_C(0xBB67AE85);
        v2[i] = key->v_[i] ^ UINT32_C(0x3C6EF372);
        v3[i] = key->v_[i] ^ UINT32_C(0xA54FF53A);
    }

    melty2_round(v0);
    melty2_round(v1);
    melty2_round(v2);
    melty2_round(v3);

    for (int i = 0; i < 6; ++i) {
        v0[i] ^= v2[i];
        v1[i] ^= v3[i];
        key->v_[i] = v0[i];
        newkey->v_[i] = v1[i];
    }

    melty2_round(v0);
    melty2_round(v1);

    for (int i = 0; i < 6; ++i) {
        key->v_[i] += v0[i];
        key->v_[i] += v1[i];
    }
}

uint32_t melty2_gen(const melty2_key *key, uint64_t idx) {
    uint32_t v[6] = {key->v_[0], key->v_[1], key->v_[2], key->v_[3], key->v_[4], key->v_[5]};
    melty2_inject(v, (uint32_t)idx, (uint32_t)(idx >> 32));
    melty2_round(v);
    return v[0] ^ v[3];
}
