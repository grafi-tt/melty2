#ifndef MELTY2_MELTY2_H_
#define MELTY2_MELTY2_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief melty2 key data
 */
typedef struct melty2_key_ {
    uint32_t v_[6];  /**< @private */
} melty2_key;

/**
 * @brief Initialize melty2 key.
 * @param key[out] Non-null pointer to key.
 * @param seed0    Part of 12bit seed.
 * @param seed1    Part of 12bit seed.
 * @param seed2    Part of 12bit seed.
 * @param seed3    Part of 12bit seed.
 */
void melty2_init(melty2_key *key, uint32_t seed0, uint32_t seed1, uint32_t seed2, uint32_t seed3);

/**
 * @brief Generate random numbers.
 * @param key      Pointer to key.
 * @param ctr      Counter.
 * @param len      How many random numbers are generated.
 * @param out[out] Pointer to store generated numbers.
 * @note  If len is 0, key and out can be NULL.
 */
void melty2_gen(const melty2_key *key, uint64_t ctr, size_t len, uint32_t *out);

#define MELTY2_RAWBLKLEN 32

/**
 * @brief Generate block of random numbers.
 * @param key      Pointer to key
 * @param ctr_lo   Lower 32bit of counter.
 * @param ctr_hi   Upper 32bit of counter.
 * @param out[out] Pointer to store generated numbers.
 * @note The size of block is given by MELTY2_RAWBLKLEN macro.
 *       ctr_lo cannot be larger than 2**32 - MELTY2_RAWBLKLEN, as overflow isn't handled.
 */
void melty2_rawblkgen(const melty2_key *key, uint32_t ctr_lo, uint32_t ctr_hi, uint32_t *out);

#ifdef __cplusplus
}
#endif

#endif
