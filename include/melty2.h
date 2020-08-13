#ifndef MELTY2_MELTY2_H_
#define MELTY2_MELTY2_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct melty2_seeder_ {
    uint64_t v_[25];
} melty2_seeder;

void melty2_initseeder(melty2_seeder *seeder);
void melty2_seed_null(melty2_seeder *seeder);
void melty2_seed_bool(melty2_seeder *seeder, int s);
void melty2_seed_uint(melty2_seeder *seeder, uint64_t s);
void melty2_seed_int(melty2_seeder *seeder, int64_t s);
void melty2_seed_float(melty2_seeder *seeder, float s);
void melty2_seed_double(melty2_seeder *seeder, double s);
void melty2_seed_str(melty2_seeder *seeder, const char *str);
void melty2_seed_strwithlen(melty2_seeder *seeder, const char *str, uint32_t len);
void melty2_seed_bin(melty2_seeder *seeder, const char *bin, uint32_t len);

typedef struct melty2_key_ {
    uint32_t v_[6];
} melty2_key;

void melty2_initkey(melty2_key *key, melty2_seeder *seeder);
void melty2_splitkey(melty2_key *key, melty2_key *newkey);

void melty2_gen(const melty2_key *key, uint64_t ctr, size_t len, uint32_t *out);

#ifdef __cplusplus
}
#endif

#endif
