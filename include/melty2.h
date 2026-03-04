#ifndef MELTY2_MELTY2_H_
#define MELTY2_MELTY2_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct melty2_key_ {
    uint32_t v_[6];
} melty2_key;

void melty2_init(melty2_key *key, uint32_t seed0, uint32_t seed1, uint32_t seed2, uint32_t seed3);
void melty2_gen(const melty2_key *key, uint64_t ctr, size_t len, uint32_t *out);

#ifdef __cplusplus
}
#endif

#endif
