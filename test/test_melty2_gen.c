#include "melty2.h"

#include <inttypes.h>
#include <stdio.h>

int check_number32(char *title, uint32_t result, uint32_t expected) {
    if (result == expected) {
        return 0;
    } else {
        fprintf(stderr, "%s -- Result: 0x%08"PRIx32", Expected: 0x%08"PRIx32"\n", title, result, expected);
        return 1;
    }
}

int main(void) {
    int n_err = 0;
    melty2_key key;
    melty2_init(&key, 0, 0, 0, 0);

    uint32_t result8[8];
    melty2_gen(&key, 0, 8, result8);
    n_err += check_number32("ctr=0", result8[0], UINT32_C(0x627106f2));
    n_err += check_number32("ctr=1", result8[1], UINT32_C(0x942da170));
    n_err += check_number32("ctr=2", result8[2], UINT32_C(0xf8f5543e));
    n_err += check_number32("ctr=3", result8[3], UINT32_C(0x8d4ae32d));
    n_err += check_number32("ctr=4", result8[4], UINT32_C(0xe49c9fb1));
    n_err += check_number32("ctr=5", result8[5], UINT32_C(0xa1349ba2));
    n_err += check_number32("ctr=6", result8[6], UINT32_C(0x2510755d));
    n_err += check_number32("ctr=7", result8[7], UINT32_C(0xe629491f));

    return !!n_err;
}
