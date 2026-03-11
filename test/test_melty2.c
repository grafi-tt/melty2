#include "melty2.h"

#include <inttypes.h>
#include <stdio.h>

#include "test_melty2_init.h"

int check_number32(char *title, uint32_t result, uint32_t expected) {
    if (result == expected) {
        return 0;
    } else {
        fprintf(stderr, "%s -- Result: 0x%08"PRIx32", Expected: 0x%08"PRIx32"\n", title, result, expected);
        return 1;
    }
}

int test_init(const TestCase* test_case) {
    const uint32_t *seed = test_case->seed;
    char title[55];
    sprintf(title, "seed={0x%08"PRIx32",0x%08"PRIx32",0x%08"PRIx32",0x%08"PRIx32"} i=0", seed[0], seed[1], seed[2], seed[3]);
    const uint32_t *expected = test_case->key;

    melty2_key key;
    melty2_init(&key, seed[0], seed[1], seed[2], seed[3]);
    uint32_t *result = key.v_;

    for (int i = 0; i < 6; ++i) {
        if (check_number32(title, result[i], expected[i])) {
            return 1;
        }
        title[sizeof(title) - 2]++;
    }
    return 0;
}

int main(void) {
    int n_err = 0;

    melty2_key key;
    melty2_init(&key, 0, 0, 0, 0);
    uint32_t result8[8];
    melty2_gen(&key, 0, 8, result8);
    n_err += check_number32("ctr=0", result8[0], UINT32_C(0xb374bf06));
    n_err += check_number32("ctr=1", result8[1], UINT32_C(0xd0f03e79));
    n_err += check_number32("ctr=2", result8[2], UINT32_C(0x17d9898e));
    n_err += check_number32("ctr=3", result8[3], UINT32_C(0xf7d3adca));
    n_err += check_number32("ctr=4", result8[4], UINT32_C(0x6c1c07a1));
    n_err += check_number32("ctr=5", result8[5], UINT32_C(0x1a540bc4));
    n_err += check_number32("ctr=6", result8[6], UINT32_C(0x4c97115d));
    n_err += check_number32("ctr=7", result8[7], UINT32_C(0xfee0ea5a));

    for (size_t t = 0; t < sizeof(test_cases) / sizeof(TestCase); ++t) {
        n_err += test_init(&test_cases[t]);
    }

    return !!n_err;
}
