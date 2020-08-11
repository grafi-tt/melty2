#include "melty2.h"

#include <string.h>

static const uint64_t BLAKE2B_IV[8] = {
    UINT64_C(0x6a09e667f3bcc908),  /* fractional parts of sqrt(2)  */
    UINT64_C(0xbb67ae8584caa73b),  /* fractional parts of sqrt(3)  */
    UINT64_C(0x3c6ef372fe94f82b),  /* fractional parts of sqrt(5)  */
    UINT64_C(0xa54ff53a5f1d36f1),  /* fractional parts of sqrt(7)  */
    UINT64_C(0x510e527fade682d1),  /* fractional parts of sqrt(11) */
    UINT64_C(0x9b05688c2b3e6c1f),  /* fractional parts of sqrt(13) */
    UINT64_C(0x1f83d9abfb41bd6b),  /* fractional parts of sqrt(17) */
    UINT64_C(0x5be0cd19137e2179),  /* fractional parts of sqrt(19) */
};

static const uint8_t BLAKE2B_MSG_SCHEDULE[12][16] = {
    { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15},
    {14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3},
    {11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4},
    { 7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8},
    { 9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13},
    { 2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9},
    {12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11},
    {13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10},
    { 6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5},
    {10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13 , 0},
    { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15},
    {14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3},
};

static inline uint64_t blake2b_rotr(uint64_t v, int n) {
    return (v >> n) | (v << (64 - n));
}

static inline void blake2b_g(uint64_t x[16], int a, int b, int c, int d, uint64_t ma, uint64_t mb) {
    x[a] = x[a] + x[b] + ma;
    x[d] = blake2b_rotr(x[d] ^ x[a], 32);
    x[c] = x[c] + x[d];
    x[b] = blake2b_rotr(x[b] ^ x[c], 24);
    x[a] = x[a] + x[b] + mb;
    x[d] = blake2b_rotr(x[d] ^ x[a], 16);
    x[c] = x[c] + x[d];
    x[b] = blake2b_rotr(x[b] ^ x[c], 63);
}

static inline uint64_t blake2b_load_u64(const uint8_t m[128], uint8_t i) {
    return (uint64_t)(m[8 * i + 0]) << 56 | (uint64_t)(m[8 * i + 1]) << 48 |
           (uint64_t)(m[8 * i + 2]) << 40 | (uint64_t)(m[8 * i + 3]) << 32 |
           (uint64_t)(m[8 * i + 4]) << 24 | (uint64_t)(m[8 * i + 5]) << 16 |
           (uint64_t)(m[8 * i + 6]) <<  8 | (uint64_t)(m[8 * i + 7]);
}

static void blake2b_round(uint64_t x[16], const uint8_t m[128], const uint8_t perm[16]) {
    blake2b_g(x, 0, 4, 8,  12, blake2b_load_u64(m, perm[0]),  blake2b_load_u64(m, perm[1]));
    blake2b_g(x, 1, 5, 9,  13, blake2b_load_u64(m, perm[2]),  blake2b_load_u64(m, perm[3]));
    blake2b_g(x, 2, 6, 10, 14, blake2b_load_u64(m, perm[4]),  blake2b_load_u64(m, perm[5]));
    blake2b_g(x, 3, 7, 11, 15, blake2b_load_u64(m, perm[6]),  blake2b_load_u64(m, perm[7]));
    blake2b_g(x, 0, 5, 10, 15, blake2b_load_u64(m, perm[8]),  blake2b_load_u64(m, perm[9]));
    blake2b_g(x, 1, 6, 11, 12, blake2b_load_u64(m, perm[10]), blake2b_load_u64(m, perm[11]));
    blake2b_g(x, 2, 7, 8,  13, blake2b_load_u64(m, perm[12]), blake2b_load_u64(m, perm[13]));
    blake2b_g(x, 3, 4, 9,  14, blake2b_load_u64(m, perm[14]), blake2b_load_u64(m, perm[15]));
}

typedef struct blake2b_state_ {
    uint64_t h[8];
    uint64_t ctr;
    uint8_t block[128];
} blake2b_state;

static void blake2b_compress(blake2b_state *state, uint64_t final) {
    uint64_t x[16];
    memcpy(&x[0], state->h, sizeof(&state->h));
    memcpy(&x[8], BLAKE2B_IV, sizeof(BLAKE2B_IV));
    x[12] ^= state->ctr;
    x[14] ^= final;

    for (int r = 0; r < 12; ++r) {
        blake2b_round(x, state->block, BLAKE2B_MSG_SCHEDULE[r]);
    }

    for (int i = 0; i < 8; ++i) {
        state->h[i] = x[i] ^ x[i + 8];
    }
}

static void blake2b_init(blake2b_state *state) {
    memcpy(state->h, BLAKE2B_IV, sizeof(BLAKE2B_IV));
    state->h[0] ^= 64;
    state->ctr = 0;
}

static void blake2b_update(blake2b_state *state, const void *input, size_t len) {
    if (len == 0) return;

    const size_t block_size = sizeof(state->block);
    size_t rest_size = block_size - state->ctr % block_size;

    for (;;) {
        size_t copy_len = rest_size < len ? rest_size : len;
        memcpy(&state->block[state->ctr % block_size], input, copy_len);
        state->ctr += copy_len;
        len -= copy_len;
        if (len == 0) return;

        blake2b_compress(state, 0);
        rest_size = block_size;
    }
}

static void blake2b_finalize(blake2b_state *state) {
    const size_t block_size = sizeof(state->block);
    memset(&state->block[state->ctr % block_size], 0, block_size - state->ctr % block_size);
    blake2b_compress(state, ~(uint64_t)0);
}

typedef struct melty2_keygen_ {
    uint64_t v_[25];
} melty2_keygen;

void melty2_keygen_begin(melty2_keygen *keygen) {
    blake2b_init((blake2b_state *)keygen);
}

void melty2_keygen_null(melty2_keygen *keygen) {
    uint8_t input[1] = {0xc0};
    blake2b_update((blake2b_state *)keygen, input, 1);
}

void melty2_keygen_bool(melty2_keygen *keygen, int v) {
    uint8_t input[1] = {v ? 0xc3 : 0xc2};
    blake2b_update((blake2b_state *)keygen, input, 1);
}

void melty2_keygen_uint(melty2_keygen *keygen, uint64_t v) {
    uint8_t input[9];
    input[0] = 0xcf;
    input[4] = 0xce;
    input[6] = 0xcd;
    input[7] = 0xcc;

    int len = 1;
    input[8] = (uint8_t)v;

    if (v < UINT64_C(1) << 7) goto update;
    len += 1;

    if (v < UINT64_C(1) << 8) goto update;
    input[7] = (uint8_t)(v >> 8);
    len += 1;

    if (v < UINT64_C(1) << 16) goto update;
    input[6] = (uint8_t)(v >> 16);
    input[5] = (uint8_t)(v >> 24);
    len += 2;

    if (v < UINT64_C(1) << 32) goto update;
    input[4] = (uint8_t)(v >> 32);
    input[3] = (uint8_t)(v >> 40);
    input[2] = (uint8_t)(v >> 48);
    input[1] = (uint8_t)(v >> 56);
    len += 4;

update:
    blake2b_update((blake2b_state *)keygen, &input[9 - len], len);
}

void melty2_keygen_int(melty2_keygen *keygen, int64_t v) {
    if (v >= 0) {
        melty2_keygen_uint(keygen, v);
        return;
    }

    uint8_t input[9];
    input[0] = 0xd3;
    input[4] = 0xd2;
    input[6] = 0xd1;
    input[7] = 0xd0;

    int len = 1;
    input[8] = (uint8_t)v;

    if (v >= -(INT64_C(1) << 5)) goto update;
    len += 1;

    if (v >= -(INT64_C(1) << 8)) goto update;
    input[7] = (uint8_t)(v >> 8);
    len += 1;

    if (v >= -(INT64_C(1) << 16)) goto update;
    input[6] = (uint8_t)(v >> 16);
    input[5] = (uint8_t)(v >> 24);
    len += 2;

    if (v >= -(INT64_C(1) << 32)) goto update;
    input[4] = (uint8_t)(v >> 32);
    input[3] = (uint8_t)(v >> 40);
    input[2] = (uint8_t)(v >> 48);
    input[1] = (uint8_t)(v >> 56);
    len += 4;

update:
    blake2b_update((blake2b_state *)keygen, &input[9 - len], len);
}

void melty2_keygen_float(melty2_keygen *keygen, float v) {
    uint32_t u;
    memcpy(&u, &v, sizeof(uint32_t));
    uint8_t input[5] = {0xca, (uint8_t)(u >> 24), (uint8_t)(u >> 16), (uint8_t)(u >> 8), (uint8_t)u};
    blake2b_update((blake2b_state *)keygen, input, sizeof(input));
}

void melty2_keygen_double(melty2_keygen *keygen, double v) {
    uint64_t u;
    memcpy(&u, &v, sizeof(uint64_t));
    uint8_t input[9] = {
        0xcb, (uint8_t)(u >> 56), (uint8_t)(u >> 48), (uint8_t)(u >> 40), (uint8_t)(u >> 32),
              (uint8_t)(u >> 24), (uint8_t)(u >> 16), (uint8_t)(u >>  8), (uint8_t)u,
    };
    blake2b_update((blake2b_state *)keygen, input, sizeof(input));
}

void melty2_keygen_str(melty2_keygen *keygen, const char *str, uint32_t len) {
    uint8_t head[5];
    head[0] = 0xdb;
    head[2] = 0xda;
    head[3] = 0xd9;

    int head_len = 1;
    head[4] = (uint8_t)len;

    if (len < UINT64_C(1) << 5) {
        head[4] |= 0xa0;
        goto update;
    }
    head_len += 1;

    if (len < UINT64_C(1) << 8) goto update;
    head[3] = (uint8_t)(len >> 8);
    head_len += 1;

    if (len < UINT64_C(1) << 16) goto update;
    head[2] = (uint8_t)(len >> 16);
    head[1] = (uint8_t)(len >> 24);
    head_len += 2;

update:
    blake2b_update((blake2b_state *)keygen, &head[5 - head_len], head_len);
    blake2b_update((blake2b_state *)keygen, str, len);
}

void melty2_keygen_bin(melty2_keygen *keygen, const char *bin, uint32_t len) {
    uint8_t head[5];
    head[0] = 0xc6;
    head[2] = 0xc5;
    head[3] = 0xc4;

    int head_len = 2;
    head[4] = (uint8_t)len;

    if (len < UINT64_C(1) << 8) goto update;
    head[3] = (uint8_t)(len >> 8);
    head_len += 1;

    if (len < UINT64_C(1) << 16) goto update;
    head[2] = (uint8_t)(len >> 16);
    head[1] = (uint8_t)(len >> 24);
    head_len += 2;

update:
    blake2b_update((blake2b_state *)keygen, &head[5 - head_len], head_len);
    blake2b_update((blake2b_state *)keygen, bin, len);
}

void melty2_keygen_end(melty2_keygen *keygen, melty2_key *key) {
    blake2b_state* state = (blake2b_state *)keygen;
    blake2b_finalize(state);
    *key = (melty2_key) {{
        (uint32_t)state->h[0], (uint32_t)(state->h[0] >> 32),
        (uint32_t)state->h[1], (uint32_t)(state->h[1] >> 32),
        (uint32_t)state->h[2], (uint32_t)(state->h[2] >> 32),
    }};
}

void melty2_splitkey(melty2_key *key, melty2_key *newkey) {
    blake2b_state state[1];
    blake2b_init(state);
    blake2b_update(state, key, sizeof(*key));
    blake2b_finalize(state);
    *key = (melty2_key) {{
        (uint32_t)state->h[0], (uint32_t)(state->h[0] >> 32),
        (uint32_t)state->h[1], (uint32_t)(state->h[1] >> 32),
        (uint32_t)state->h[2], (uint32_t)(state->h[2] >> 32),
    }};
    *newkey = (melty2_key) {{
        (uint32_t)state->h[3], (uint32_t)(state->h[3] >> 32),
        (uint32_t)state->h[4], (uint32_t)(state->h[4] >> 32),
        (uint32_t)state->h[5], (uint32_t)(state->h[5] >> 32),
    }};
}