#ifndef MELTY2_MELTY2_H_
#define MELTY2_MELTY2_H_

#ifdef __CUDACC_RTC__
/* stdint.h doesn't work with NVRTC. */
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
#else
#include <stdint.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct melty2_key_ {
    uint32_t v_[6];
} melty2_key;

typedef struct melty2_name_ {
    uint32_t v[8];
} melty2_name;

void melty2_initkey(melty2_key *key, const melty2_name *name, uint64_t global_seed, uint64_t global_ctr);
void melty2_splitkey(melty2_key *key, melty2_key *newkey);

uint32_t melty2_gen(const melty2_key *key, uint64_t idx);
void melty2_bulkgen(const melty2_key *key, uint64_t idx, uint64_t len, uint32_t *out);

/* `melty2_name_fromliteral("some_name_here")` returns a `melty2_name` instance.
 * `melty2_initkey_fromliteral(key, "some_nme_here", global_seed, global_ctr)` works similarly.
 *
 * The string literal MUST NOT have more than 32 chars excluding the terminating '\0'.
 * The index `-sizeof(literal) > 33` below is a trick for compile-time checking.
 */
#define melty2_initkey_fromliteral(key, literal, global_seed, global_ctr) \
    melty2_initkey(key, &melty2_name_fromliteral(literal), global_seed, global_ctr)

#define melty2_name_fromliteral(literal) melty2_name_fromliteral_constr_({{                                           \
    (uint32_t)(literal "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0")[-(sizeof(literal) > 33)] | \
    ((uint32_t)(literal "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0")[1] << 8) |                \
    ((uint32_t)(literal "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0")[2] << 16) |               \
    ((uint32_t)(literal "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0")[3] << 24),                \
    (uint32_t)(literal "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0")[4] |                       \
    ((uint32_t)(literal "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0")[5] << 8) |                \
    ((uint32_t)(literal "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0")[6] << 16) |               \
    ((uint32_t)(literal "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0")[7] << 24),                \
    (uint32_t)(literal "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0")[8] |                       \
    ((uint32_t)(literal "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0")[9] << 8) |                \
    ((uint32_t)(literal "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0")[10] << 16) |              \
    ((uint32_t)(literal "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0")[11] << 24),               \
    (uint32_t)(literal "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0")[12] |                      \
    ((uint32_t)(literal "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0")[13] << 8) |               \
    ((uint32_t)(literal "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0")[14] << 16) |              \
    ((uint32_t)(literal "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0")[15] << 24),               \
    (uint32_t)(literal "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0")[16] |                      \
    ((uint32_t)(literal "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0")[17] << 8) |               \
    ((uint32_t)(literal "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0")[18] << 16) |              \
    ((uint32_t)(literal "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0")[19] << 24),               \
    (uint32_t)(literal "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0")[20] |                      \
    ((uint32_t)(literal "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0")[21] << 8) |               \
    ((uint32_t)(literal "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0")[22] << 16) |              \
    ((uint32_t)(literal "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0")[23] << 24),               \
    (uint32_t)(literal "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0")[24] |                      \
    ((uint32_t)(literal "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0")[25] << 8) |               \
    ((uint32_t)(literal "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0")[26] << 16) |              \
    ((uint32_t)(literal "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0")[27] << 24),               \
    (uint32_t)(literal "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0")[28] |                      \
    ((uint32_t)(literal "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0")[29] << 8) |               \
    ((uint32_t)(literal "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0")[30] << 16) |              \
    ((uint32_t)(literal "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0")[31] << 24)                \
}})

#ifdef __cplusplus
#define melty2_name_fromliteral_constr_(...) static_cast<const melty2_name &>(melty2_name __VA_ARGS__)
#else
#define melty2_name_fromliteral_constr_(...) ((melty2_name) __VA_ARGS__)
#endif

#ifdef __cplusplus
}
#endif

#endif
