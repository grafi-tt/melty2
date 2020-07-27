#include "melty2.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

int check_number32(char *title, uint32_t result, uint32_t expected) {
    if (result == expected) {
        return 0;
    } else {
        fprintf(stderr, "%s -- Result: 0x%08"PRIX32", Expected: 0x%08"PRIX32"\n", title, result, expected);
        return 1;
    }
}

int test_init() {
    int n_err = 0;
    melty2_name name;
    melty2_initname(&name, "42");

    melty2_key key;
    melty2_initkey(&key, &name, 0, 0);
    n_err += check_number32("test_init: global_seed=0, global_ctr=0 -> v[0]", key.v_[0], UINT32_C(0x618CAA89));
    n_err += check_number32("test_init: global_seed=0, global_ctr=0 -> v[1]", key.v_[1], UINT32_C(0x17E99138));
    n_err += check_number32("test_init: global_seed=0, global_ctr=0 -> v[2]", key.v_[2], UINT32_C(0xAEDA7464));
    n_err += check_number32("test_init: global_seed=0, global_ctr=0 -> v[3]", key.v_[3], UINT32_C(0xB2AD8205));
    n_err += check_number32("test_init: global_seed=0, global_ctr=0 -> v[4]", key.v_[4], UINT32_C(0xE9EB6F42));
    n_err += check_number32("test_init: global_seed=0, global_ctr=0 -> v[5]", key.v_[5], UINT32_C(0x55EDCA92));

    return n_err;
}

int main() {
    int n_err = 0;
    n_err += test_init();
    return !!n_err;
}
