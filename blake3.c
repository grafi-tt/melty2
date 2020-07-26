#include <stdint.h>
#include <string.h>

static const uint32_t MELTY2_BLAKE3_CHUNK_START = 1 << 0;
static const uint32_t MELTY2_BLAKE3_CHUNK_END   = 1 << 1;
static const uint32_t MELTY2_BLAKE3_PARENT      = 1 << 2;
static const uint32_t MELTY2_BLAKE3_ROOT        = 1 << 3;

static const uint32_t MELTY2_BLAKE3_IV[8] = {
    UINT32_C(0x6A09E667),  /* fractional parts of sqrt(2)  */
    UINT32_C(0xBB67AE85),  /* fractional parts of sqrt(3)  */
    UINT32_C(0x3C6EF372),  /* fractional parts of sqrt(5)  */
    UINT32_C(0xA54FF53A),  /* fractional parts of sqrt(7)  */
    UINT32_C(0x510E527F),  /* fractional parts of sqrt(11) */
    UINT32_C(0x9B05688C),  /* fractional parts of sqrt(13) */
    UINT32_C(0x1F83D9AB),  /* fractional parts of sqrt(17) */
    UINT32_C(0x5BE0CD19),  /* fractional parts of sqrt(19) */
};

static const uint8_t MELTY2_BLAKE3_MSG_SCHEDULE[7][16] = {
    {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15},
    {2,  6,  3,  10, 7,  0,  4,  13, 1,  11, 12, 5,  9,  14, 15, 8},
    {3,  4,  10, 12, 13, 2,  7,  14, 6,  5,  9,  0,  11, 15, 8,  1},
    {10, 7,  12, 9,  14, 3,  13, 15, 4,  0,  11, 2,  5,  8,  1,  6},
    {12, 13, 9,  11, 15, 10, 14, 8,  7,  2,  5,  3,  0,  1,  6,  4},
    {9,  14, 11, 5,  8,  12, 15, 1,  13, 3,  0,  10, 2,  6,  4,  7},
    {11, 15, 5,  0,  1,  9,  8,  6,  14, 10, 2,  12, 3,  4,  7,  13},
};

static uint32_t melty2_blake3_rotr(uint32_t v, int n) {
    return (v >> n) | (v << (32 - n));
}

static void melty2_blake3_g(uint32_t x[16], int a, int b, int c, int d, uint32_t ma, uint32_t mb) {
    x[a] = x[a] + x[b] + ma;
    x[d] = melty2_blake3_rotr(x[d] ^ x[a], 16);
    x[c] = x[c] + x[d];
    x[b] = melty2_blake3_rotr(x[b] ^ x[c], 12);
    x[a] = x[a] + x[b] + mb;
    x[d] = melty2_blake3_rotr(x[d] ^ x[a], 8);
    x[c] = x[c] + x[d];
    x[b] = melty2_blake3_rotr(x[b] ^ x[c], 7);
}

static void melty2_blake3_round(uint32_t x[16], const uint32_t m[16], const uint8_t perm[16]) {
    melty2_blake3_g(x, 0, 4, 8,  12, m[perm[0]],  m[perm[1]]);
    melty2_blake3_g(x, 1, 5, 9,  13, m[perm[2]],  m[perm[3]]);
    melty2_blake3_g(x, 2, 6, 10, 14, m[perm[4]],  m[perm[5]]);
    melty2_blake3_g(x, 3, 7, 11, 15, m[perm[6]],  m[perm[7]]);
    melty2_blake3_g(x, 0, 5, 10, 15, m[perm[8]],  m[perm[9]]);
    melty2_blake3_g(x, 1, 6, 11, 12, m[perm[10]], m[perm[11]]);
    melty2_blake3_g(x, 2, 7, 8,  13, m[perm[12]], m[perm[13]]);
    melty2_blake3_g(x, 3, 4, 9,  14, m[perm[14]], m[perm[15]]);
}

static void melty2_blake3_compress(uint32_t chaining_value[8], const uint32_t block[16],
                                   uint64_t chunk_counter, uint32_t block_len, uint32_t flags) {
    uint32_t state[16] = {
        chaining_value[0], chaining_value[1], chaining_value[2], chaining_value[3],
        chaining_value[4], chaining_value[5], chaining_value[6], chaining_value[7],
        MELTY2_BLAKE3_IV[0], MELTY2_BLAKE3_IV[1], MELTY2_BLAKE3_IV[2], MELTY2_BLAKE3_IV[3],
        (uint32_t)chunk_counter, (uint32_t)(chunk_counter >> 32), block_len, flags
    };

    for (int r = 0; r < 7; r++) {
        melty2_blake3_round(state, block, MELTY2_BLAKE3_MSG_SCHEDULE[r]);
    }

    for (int i = 0; i < 8; i++) {
        chaining_value[i] = state[i] ^ state[i + 8];
    }
}

static void melty2_blake3_load_block(const char *input, uint32_t block_len, uint32_t block[16]) {
    uint32_t i = 0;
    while (i < block_len / sizeof(uint32_t)) {
        uint32_t v = (uint32_t)(uint8_t)input[i * sizeof(uint32_t)] |
                     (uint32_t)(uint8_t)input[i * sizeof(uint32_t) + 1] << 8 |
                     (uint32_t)(uint8_t)input[i * sizeof(uint32_t) + 2] << 16 |
                     (uint32_t)(uint8_t)input[i * sizeof(uint32_t) + 3] << 24;
        block[i++] = v;
    }

    uint32_t v = 0;
    switch (block_len % sizeof(uint32_t)) {
    case 3:
        v = (uint32_t)(uint8_t)input[i * sizeof(uint32_t) + 2] << 16;
        /* FALLTHRU */
    case 2:
        v |= (uint32_t)(uint8_t)input[i * sizeof(uint32_t) + 1] << 8;
        /* FALLTHRU */
    case 1:
        v |= (uint32_t)(uint8_t)input[i * sizeof(uint32_t)];
        block[i++] = v;
        break;
    default:
        break;
    }

    while (i < 16) {
        block[i++] = 0;
    }
}

static void melty2_blake3_hash_chunk(const char *input, uint64_t chunk_counter, uint32_t chunk_len,
                                     uint32_t chaining_value[8], int is_root) {
    memcpy(chaining_value, MELTY2_BLAKE3_IV, sizeof(MELTY2_BLAKE3_IV));

    uint32_t flags = MELTY2_BLAKE3_CHUNK_START;
    do {
        uint32_t block[16];
        uint32_t block_len = sizeof(block);
        if (chunk_len <= block_len) {
            block_len = chunk_len;
            flags |= MELTY2_BLAKE3_CHUNK_END;
            if (is_root) flags |= MELTY2_BLAKE3_ROOT;
        }
        chunk_len -= block_len;

        melty2_blake3_load_block(input, block_len, block);
        input += block_len;

        melty2_blake3_compress(chaining_value, block, chunk_counter, block_len, flags);
        flags &= ~MELTY2_BLAKE3_CHUNK_START;
    } while (chunk_len);
}

static void melty2_blake3_hash_parent(const uint32_t left_value[8], uint32_t chaining_value[8], int is_root) {
    uint32_t block[16];
    memcpy(block, left_value, sizeof(block) / 2);
    memcpy(&block[8], chaining_value, sizeof(block) / 2);

    uint32_t flags = MELTY2_BLAKE3_PARENT;
    if (is_root) flags |= MELTY2_BLAKE3_ROOT;

    memcpy(chaining_value, MELTY2_BLAKE3_IV, sizeof(MELTY2_BLAKE3_IV));
    melty2_blake3_compress(chaining_value, block, 0, sizeof(block), flags);
}

void melty2_hash(const char *input, uint64_t input_len, char output[32]) {
    const uint32_t CHUNK_SIZE = 1024;

    uint32_t chaining_value[8];
    uint32_t stack[54][8];

    uint64_t chunk_counter = 0;
    for (;;) {
        uint32_t chunk_len = CHUNK_SIZE;
        if (input_len <= chunk_len) {
            chunk_len = (uint32_t)input_len;
        }
        input_len -= chunk_len;

        melty2_blake3_hash_chunk(input, chunk_counter, chunk_len, chaining_value,
                                 !chunk_counter && !input_len);
        input += chunk_len;
        if (!input_len) break;

        unsigned int i;
        for (i = 0; chunk_counter & (uint64_t)1 << i; i++) {
            melty2_blake3_hash_parent(stack[i], chaining_value, 0);
        }
        memcpy(stack[i], chaining_value, sizeof(chaining_value));
        ++chunk_counter;
    }

    unsigned int i = 0;
    while (chunk_counter) {
        int has_left_value = !!(chunk_counter & 1);
        chunk_counter >>= 1;
        if (has_left_value) {
            melty2_blake3_hash_parent(stack[i], chaining_value, !chunk_counter);
        }
        i++;
    }

    memcpy(output, chaining_value, sizeof(chaining_value));
}
