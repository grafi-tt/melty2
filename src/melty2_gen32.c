#include "melty2.h"

#include "melty2_util.h"

/* Add suffix to melty2_gen32 symbol */
#ifdef MELTY2_GEN32_SUFFIX
#define melty2_gen32 melty2_gen32_EXPAND1(MELTY2_GEN32_SUFFIX)
#define melty2_gen32_EXPAND1(suffix) melty2_gen32_EXPAND2(suffix)
#define melty2_gen32_EXPAND2(suffix) melty2_gen32_ ## suffix
#endif

void melty2_gen32(const melty2_key *key, uint32_t ctr_lo, uint32_t ctr_hi, uint32_t *out) {
#ifdef _MSC_VER
#define restrict __restrict
#endif
    const uint32_t * restrict key_v = key->v_;
    uint32_t * restrict out_p = out;

    for (uint32_t k = 0; k < 32; ++k) {
        uint32_t v[6] = {key_v[0], key_v[1], key_v[2], key_v[3], key_v[4], key_v[5]};
        melty2_inject(v, ctr_lo + k, ctr_hi);
        melty2_round(v);
        out_p[k] = v[0] ^ v[3];
    }
}
