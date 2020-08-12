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

typedef struct melty2_keygen_ {
    uint64_t v_[25];
} melty2_keygen;

void melty2_keygen_begin(melty2_keygen *keygen);
void melty2_keygen_null(melty2_keygen *keygen);
void melty2_keygen_bool(melty2_keygen *keygen, int v);
void melty2_keygen_uint(melty2_keygen *keygen, uint64_t v);
void melty2_keygen_int(melty2_keygen *keygen, int64_t v);
void melty2_keygen_float(melty2_keygen *keygen, float v);
void melty2_keygen_double(melty2_keygen *keygen, double v);
void melty2_keygen_str(melty2_keygen *keygen, const char *str, uint32_t len);
void melty2_keygen_bin(melty2_keygen *keygen, const char *bin, uint32_t len);
void melty2_keygen_end(melty2_keygen *keygen, melty2_key *key);

void melty2_splitkey(melty2_key *key, melty2_key *newkey);

uint32_t melty2_gen(const melty2_key *key, uint64_t idx);
void melty2_bulkgen(const melty2_key *key, uint64_t idx, uint64_t len, uint32_t *out);

#ifdef __cplusplus
}
#endif

#endif
