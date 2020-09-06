#include "melty2.h"

static inline uint32_t melty2_rotl(uint32_t v, int n) {
    return (v << n) | (v >> (32 - n));
}

static inline void melty2_inject(uint32_t *v, uint32_t a, uint32_t b) {
    v[0] += a;
    v[1] ^= a;
    v[2] += b;
    v[3] ^= b;
}

static inline void melty2_round(uint32_t *v) {
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

/* Add suffix to melty2_gen symbol */
#ifdef MELTY2_GEN_SUFFIX
#define melty2_gen melty2_gen_EXPAND1(MELTY2_GEN_SUFFIX)
#define melty2_gen_EXPAND1(suffix) melty2_gen_EXPAND2(suffix)
#define melty2_gen_EXPAND2(suffix) melty2_gen_ ## suffix
#endif

void melty2_gen(const melty2_key *key, uint64_t ctr, size_t len, uint32_t *out) {
#ifdef _MSC_VER
#define restrict __restrict
#endif
    const uint32_t * restrict key_v = key->v_;
    uint32_t * restrict out_p = out;

    for (size_t k = 0; k < len; ++k) {
        uint32_t v[6] = {key_v[0], key_v[1], key_v[2], key_v[3], key_v[4], key_v[5]};
        uint64_t ctr_k = ctr + (uint64_t)k;
        melty2_inject(v, (uint32_t)ctr_k, (uint32_t)(ctr_k >> 32));
        melty2_round(v);
        out_p[k] = v[0] ^ v[3];
    }
}
