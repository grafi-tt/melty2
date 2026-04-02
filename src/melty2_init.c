#include "melty2.h"

static uint32_t compute_crc32(const uint32_t* seed) {
    /* zlib crc32 by naive polynomial division */
    uint32_t c = ~UINT32_C(0);
    for (int i = 0; i < 4; ++i) {
        c ^= seed[i];
        for (int j = 0; j < 32; ++j) {
            int flag = c & 1;
            c >>= 1;
            if (flag) {
                c ^= UINT32_C(0xedb88320);
            }
        }
    }
    return ~c;
}

static const uint8_t code5b6b_tbl[32][2] = {
    /* this 5b/6b code is a part of widely used 8b/10b coding */
    {0x27, 0x18},
    {0x1d, 0x22},
    {0x2d, 0x12},
    {0x31, 0x31},
    {0x35, 0x0a},
    {0x29, 0x29},
    {0x19, 0x19},
    {0x38, 0x07},
    {0x39, 0x06},
    {0x25, 0x25},
    {0x15, 0x15},
    {0x34, 0x34},
    {0x0d, 0x0d},
    {0x2c, 0x2c},
    {0x1c, 0x1c},
    {0x17, 0x28},
    {0x1b, 0x24},
    {0x23, 0x23},
    {0x13, 0x13},
    {0x32, 0x32},
    {0x0b, 0x0b},
    {0x2a, 0x2a},
    {0x1a, 0x1a},
    {0x3a, 0x05},
    {0x33, 0x0c},
    {0x26, 0x26},
    {0x16, 0x16},
    {0x36, 0x09},
    {0x0e, 0x0e},
    {0x2e, 0x11},
    {0x1e, 0x21},
    {0x2b, 0x14},
};

static void encode_key(const uint32_t* seed, uint32_t* key) {
    uint32_t c = compute_crc32(seed);
    uint32_t tmp[8];

    int is_positive = 0;
    for (int i = 0; i < 8; ) {  /* runs 4 times */
        uint32_t s = *seed++;
        do {  /* runs 2 times */
            uint32_t t = 0;
            while (!(t & 0x3f)) {  /* runs 4 times */
                /*
                 * Compute 5 bits n by
                 *   n = (s' ^ (1 << sw)) | c' << 4, where c' = c & 1, s' = s & 0xf, and sw = (c >> 1) % 4 .
                 * Then apply 5b/6b encoding to n.
                 * The 6 bits code is concatenated to t.
                 */
                uint32_t n = (s & 0xf) | (c & 1) << 4;
                s >>= 4;
                c >>= 1;
                int sw = c % 4;
                n ^= 1 << sw;
                uint32_t code = code5b6b_tbl[n][!is_positive];
                uint32_t code_alt = code5b6b_tbl[n][is_positive];
                is_positive ^= (code != code_alt);
                t = code << 18 | t >> 6;
            }
            tmp[i++] = t;
        } while (i % 2);
    }

    for (int i = 0; i < 8; i += 4) {
        uint32_t a = tmp[i];
        uint32_t b = tmp[i + 1];
        uint32_t c = tmp[i + 2];
        uint32_t d = tmp[i + 3];
        key[0] = a | b << 24;
        key[1] = b >> 8 | c << 16;
        key[2] = c >> 16 | d << 8;
        key += 3;
    }
}

void melty2_init(melty2_key *key, uint32_t seed0, uint32_t seed1, uint32_t seed2, uint32_t seed3) {
    uint32_t seed[4] = {seed0, seed1, seed2, seed3};
    encode_key(seed, key->v_);
}
