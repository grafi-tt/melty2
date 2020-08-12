#include "melty2.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_seeder.h"

int test(const TestCase* test_case) {
    unsigned char expected[24];
    for (int i = 0; i < 24; i++) {
        unsigned int c;
        if (sscanf(&test_case->expected_hex[i * 2], "%02x", &c) != 1) {
            fputs("sscanf failed\n", stderr);
            return -1;
        }
        expected[i] = (unsigned char)c;
    }

    melty2_key key;
    test_case->fn(&key);
    unsigned char result[24];
    for (int i = 0; i < 6; ++i) {
        result[i * 4 + 0] = (unsigned char)(key.v_[i]);
        result[i * 4 + 1] = (unsigned char)(key.v_[i] >> 8);
        result[i * 4 + 2] = (unsigned char)(key.v_[i] >> 16);
        result[i * 4 + 3] = (unsigned char)(key.v_[i] >> 24);
    }

    int err = !!memcmp(result, expected, sizeof(expected));
    if (err) {
        fprintf(stderr, "test failed: %s\n", test_case->name);
    }
    return err;
}

int main() {
    int n_err = 0;
    for (size_t t = 0; t < sizeof(test_cases) / sizeof(TestCase); ++t) {
        int err = test(&test_cases[t]);
        if (err < 0) return 2;
        n_err += err;
    }
    return !!n_err;
}
