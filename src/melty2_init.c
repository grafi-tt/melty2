#include "melty2.h"

#include "melty2_derive.h"

void melty2_init(melty2_key *key, uint32_t seed0, uint32_t seed1, uint32_t seed2, uint32_t seed3) {
    melty2_internal_derive_key(key->v_, seed0, seed1, seed2, seed3);
}

void melty2_init_u64(melty2_key *key, uint64_t seed0, uint64_t seed1) {
    melty2_init(key, (uint32_t)seed0, (uint32_t)(seed0 >> 32), (uint32_t)seed1, (uint32_t)(seed1 >> 32));
}
