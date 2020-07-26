#ifndef MELTY2_MELTY2_H_
#define MELTY2_MELTY2_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct melty2_name_ {
    uint32_t v_[8];
} melty2_name;

void melty2_initname(melty2_name *name, const char *str);
void melty2_initname_withlen(melty2_name *name, const char *data, size_t len);

typedef struct melty2_key_ {
    uint32_t v_[6];
} melty2_key;

void melty2_initkey(melty2_key *key, const melty2_name *name, uint64_t global_seed, uint64_t global_ctr);
void melty2_splitkey(melty2_key *key, melty2_key *newkey);

uint32_t melty2_gen(const melty2_key *key, uint64_t idx);
void melty2_bulkgen(const melty2_key *key, uint64_t idx, uint64_t len, uint32_t *out);

#ifdef __cplusplus
}
#endif

#endif
