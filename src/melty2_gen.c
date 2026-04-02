#include "melty2.h"

#include "melty2_kernel.h"

void melty2_gen(const melty2_key *key, uint64_t ctr, size_t len, uint32_t *out) {
#ifdef _MSC_VER
#define restrict __restrict
#endif
    const uint32_t * restrict key_v = key->v_;
    uint32_t * restrict out_p = out;

    size_t peel_len = -ctr % MELTY2_RAWBLKLEN;
    if (len < MELTY2_RAWBLKLEN) {
        peel_len = len;
    }

    for (;;) {
        for (size_t k = 0; k < peel_len; ++k) {
            uint32_t v[6] = {key_v[0], key_v[1], key_v[2], key_v[3], key_v[4], key_v[5]};
            out_p[k] = melty2_kernel(v, (uint32_t)ctr, (uint32_t)(ctr >> 32));
            ++ctr;
        }
        if ((len -= peel_len) == 0) {
            return;
        }
        out_p += peel_len;

        while (len >= MELTY2_RAWBLKLEN) {
            len -= MELTY2_RAWBLKLEN;
            melty2_rawblkgen(key, (uint32_t)ctr, (uint32_t)(ctr >> 32), out_p);
            ctr += MELTY2_RAWBLKLEN;
            out_p += MELTY2_RAWBLKLEN;
        }
        peel_len = len;
    }
}
