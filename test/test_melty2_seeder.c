#include "melty2.h"

#include <inttypes.h>
#include <stdio.h>

#include "test_melty2_seeder.h"

int test(const TestCase* test_case) {
    melty2_key key;
    melty2_init(&key, test_case->seed[0], test_case->seed[1], test_case->seed[2], test_case->seed[3]);
    const uint32_t *expected = test_case->key;
    uint32_t *result = key.v_;
    for (int i = 0; i < 6; ++i) {
        if (result[i] != expected[i]) {
            fprintf(stderr, "test failed: result=0x%08"PRIx32", expected=0x%08"PRIx32", i=%d\n", result[i], expected[i], i);
            return 1;
        }
    }
    return 0;
}

int main(void) {
    int n_err = 0;
    for (size_t t = 0; t < sizeof(test_cases) / sizeof(TestCase); ++t) {
        n_err += test(&test_cases[t]);
    }
    return !!n_err;
}
