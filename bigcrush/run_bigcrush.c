#include <stdio.h>
#include <string.h>

#include <bbattery.h>
#include <unif01.h>

#include "melty2.h"

#define BUFLEN 128

static int bitrev;

static melty2_key key;
static uint64_t idx = 0;

static uint32_t buf[BUFLEN];
static int pos = BUFLEN;

static uint32_t next() {
    if (pos == BUFLEN) {
        pos = 0;
        melty2_gen(&key, idx, BUFLEN, buf);
        idx += BUFLEN;
    }
    uint32_t r = buf[pos++];

    if (bitrev) {
        uint32_t t;
        t = r & UINT32_C(0x00FF00FF);
        r = (t << 16) | (t >> 16) | (t ^ r);
        t = r & UINT32_C(0x0F0F0F0F);
        r = (t << 8) | (t >> 24) | (t ^ r);
        t = r & UINT32_C(0x33333333);
        r = (t << 4) | (t >> 28) | (t ^ r);
        t = r & UINT32_C(0x55555555);
        r = (t << 2) | (t >> 30) | (t ^ r);
        r = (r << 1) | (r >> 31);
    }
    return r;
}

int main(int argc, char *argv[]) {
    if (argc == 1) {
        bitrev = 0;
    } else if (argc == 2 && !strcmp(argv[1], "--bitrev")) {
        bitrev = 1;
    } else {
        fputs("usage: bigcrush [--bitrev]", stderr);
        return 1;
    }

    melty2_seeder seeder;
    melty2_initseeder(&seeder);
    melty2_seed_str(&seeder, "melty2", 6);
    melty2_initkey(&seeder, &key);

    unif01_Gen *gen = unif01_CreateExternGenBits("melty2", next);
    bbattery_BigCrush(gen);
    unif01_DeleteExternGenBits(gen);
    return 0;
}
