#include "melty2.h"

void melty2_init(melty2_key *key, uint32_t seed0, uint32_t seed1, uint32_t seed2, uint32_t seed3) {
    /* 3/6 balanced coding in IEEE 1355 */
    static const uint8_t t[16] = { 0x1a, 0x29, 0x19, 0x31, 0x0d, 0x2c, 0x1c, 0x38, 0x0b, 0x23, 0x13, 0x32, 0x0e, 0x26, 0x16, 0x25 };

    uint32_t s[4] = {seed0, seed1, seed2, seed3};
    uint32_t v[8];
    for (uint32_t *pv = v, *ps = s; pv != v + 8; ) {
        uint32_t m = *ps++;
        for (int i = 0; i < 2; ++i) {
            *pv++ = t[m & 0xf] | t[m >> 4 & 0xf] << 6 | t[m >> 8 & 0xf] << 12 | t[m >> 12 & 0xf] << 18;
            m >>= 16;
        }
    }
    for (uint32_t *pv = v, *pk = key->v_; pv != v + 8; ) {
        *pk++ = pv[0] | pv[1] << 24;
        *pk++ = pv[1] >> 8 | pv[2] << 16;
        *pk++ = pv[2] >> 16 | pv[3] << 8;
        pv += 4;
    }
}
