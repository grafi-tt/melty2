/* SPDX-License-Identifier: MIT */
/* bcsub "structure" M seed swap [tests...] : selected tests of BigCrush of TestU01 (by the numbers of
 * bbattery_RepeatBigCrush), by default a subset that took long for weak structures. */
#include <string.h>

#include <bbattery.h>
#include <unif01.h>

#include "keymod.h"
#include "melty2.h"

#include "kern.h"

static int swp;
static uint32_t key[6], buf[1024];
static uint64_t ctr;
static int idx = 1024;

static unsigned int next(void) {
    if (idx >= 1024) {
        for (int i = 0; i < 1024; i += LANES, ctr += LANES) block(key, ctr, swp, buf + i);
        idx = 0;
    }
    return buf[idx++];
}

static const int default_tests[] = {12, 22, 23, 24, 34, 40, 41, 42, 43, 44, 45, 66, 67, 76, 77, 78, 79, 80, 81, 82, 93, 94, 95, 96, 0};

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: bcsub structure M seed swap [tests...]\n");
        return 1;
    }
    int M = atoi(argv[2]);
    uint32_t s0 = strtoul(argv[3], 0, 0);
    swp = atoi(argv[4]);
    melty2_key k;
    melty2_init(&k, s0, 0, 0, 0);
    keymod(k.v_);
    memcpy(key, k.v_, sizeof key);
    int rep[200] = {0};
    if (argc > 5) {
        for (int i = 5; i < argc; ++i) {
            int t = atoi(argv[i]);
            if (t < 1 || t > 106) {
                fprintf(stderr, "test numbers must be in [1, 106]\n");
                return 1;
            }
            rep[t] = 1;
        }
    } else {
        for (int i = 0; default_tests[i]; ++i) rep[default_tests[i]] = 1;
    }
    char name[96];
    snprintf(name, sizeof name, "M=%d seed0=%u%s", M, s0, swp ? " swap32" : "");
    unif01_Gen *g = unif01_CreateExternGenBits(name, next);
    bbattery_RepeatBigCrush(g, rep);
    unif01_DeleteExternGenBits(g);
    return 0;
}
