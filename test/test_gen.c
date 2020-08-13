#include "melty2.h"

#include <inttypes.h>
#include <stdio.h>

int check_number32(char *title, uint32_t result, uint32_t expected) {
    if (result == expected) {
        return 0;
    } else {
        fprintf(stderr, "%s -- Result: 0x%08"PRIX32", Expected: 0x%08"PRIX32"\n", title, result, expected);
        return 1;
    }
}

int main() {
    int n_err = 0;
    melty2_seeder seeder;
    melty2_initseeder(&seeder);
    melty2_seed_str(&seeder, "test");
    melty2_key key;
    melty2_initkey(&key, &seeder);

    uint32_t result8[8];
    melty2_gen(&key, 0, 8, result8);
    n_err += check_number32("ctr=0", result8[0], UINT32_C(0x3DD40A39));
    n_err += check_number32("ctr=1", result8[1], UINT32_C(0x7D68202F));
    n_err += check_number32("ctr=2", result8[2], UINT32_C(0x3842C6BF));
    n_err += check_number32("ctr=3", result8[3], UINT32_C(0x58125E40));
    n_err += check_number32("ctr=4", result8[4], UINT32_C(0xA93A93E2));
    n_err += check_number32("ctr=5", result8[5], UINT32_C(0xDFA48C04));
    n_err += check_number32("ctr=6", result8[6], UINT32_C(0xBD288AED));
    n_err += check_number32("ctr=7", result8[7], UINT32_C(0x5E48F32C));

    return !!n_err;
}
