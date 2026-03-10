#include "melty2.h"

#include "melty2_kernel.h"

/* Add suffix to melty2_rawblkgen symbol */
#ifdef MELTY2_RAWBLKGEN_SUFFIX
#define melty2_rawblkgen melty2_rawblkgen_EXPAND1(MELTY2_RAWBLKGEN_SUFFIX)
#define melty2_rawblkgen_EXPAND1(suffix) melty2_rawblkgen_EXPAND2(suffix)
#define melty2_rawblkgen_EXPAND2(suffix) melty2_rawblkgen_ ## suffix
#endif

void melty2_rawblkgen(const melty2_key *key, uint32_t ctr_lo, uint32_t ctr_hi, uint32_t *out) {
#ifdef _MSC_VER
#define restrict __restrict
#endif
    const uint32_t * restrict key_v = key->v_;
    uint32_t * restrict out_p = out;

    for (uint32_t k = 0; k < MELTY2_RAWBLKLEN; ++k) {
        uint32_t v[6] = {key_v[0], key_v[1], key_v[2], key_v[3], key_v[4], key_v[5]};
        out_p[k] = melty2_kernel(v, ctr_lo + k, ctr_hi);
    }
}
