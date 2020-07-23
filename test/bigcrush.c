#include <stddef.h>

#include <unif01.h>
#include <bbattery.h>

#include <melty2.h>

static uint32_t next() {
    static melty2_key key;
    static uint64_t idx = 0;
    static uint32_t buf[32];
    static int pos = 0;
    if (pos == 0) {
        melty2_initkey_fromliteral(&key, "melty2", 0, 0);
        pos = 32;
    }
    if (pos == 32) {
        pos = 0;
        melty2_bulkgen(&key, idx, 32, buf);
        idx += 32;
    }
    uint32_t r = buf[pos++];
#ifdef BITREV
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
#endif
    return r;
}

int main() {
    unif01_Gen *gen = unif01_CreateExternGenBits("melty2", next);
    bbattery_BigCrush(gen);
    unif01_DeleteExternGenBits(gen);
    return 0;
}
