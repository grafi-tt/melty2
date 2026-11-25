// SPDX-License-Identifier: Apache-2.0
// genlow [options] seed swap nbits : the lower nbits bits (after >> shift) of the outputs of the
// compiled-in structure (see run.sh), packed into 32bit words from the LSB, to stdout. The least
// significant bits of ARX outputs mix slowly, as an addition has no carry into bit 0, so their stream
// is tested separately.
//   -s, --shift N  right shift of the outputs before taking the lower bits (default 0)
#include <getopt.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "keymod.hpp"

#include "kern.hpp"

namespace {

[[noreturn]] void usage() {
    std::fprintf(stderr, "usage: genlow [-s shift] seed swap nbits\n");
    std::exit(1);
}

int run(int argc, char **argv) {
    static const option longopts[] = {
        {"shift", required_argument, nullptr, 's'},
        {nullptr, 0, nullptr, 0},
    };
    int sh = 0, opt;
    while ((opt = getopt_long(argc, argv, "s:", longopts, nullptr)) != -1) {
        switch (opt) {
        case 's': sh = std::atoi(optarg); break;
        default: usage();
        }
    }
    if (argc - optind != 3) usage();
    char **arg = argv + optind;
    const auto seed = static_cast<uint32_t>(std::strtoul(arg[0], nullptr, 0));
    const int swp = std::atoi(arg[1]), nb = std::atoi(arg[2]);
    if (nb < 1 || nb > 32 || 32 % nb != 0) {
        std::fprintf(stderr, "nbits must divide 32\n");
        return 1;
    }
    arx::Key key = keymod::seed_key(seed);
    keymod::from_env(key);
    const uint32_t mask = nb == 32 ? ~UINT32_C(0) : ((UINT32_C(1) << nb) - 1);
    const int per = 32 / nb;  // outputs per packed word
    constexpr int kOut = 1 << 14;
    std::vector<uint32_t> out(kOut);
    std::array<uint32_t, kLanes> buf;
    uint64_t ctr = 0;
    for (;;) {
        for (int o = 0; o < kOut;) {
            block(key.data(), ctr, swp, buf.data());
            ctr += kLanes;
            for (int base = 0; base < kLanes; base += per, ++o) {
                uint32_t w = 0;
                for (int f = 0; f < per; ++f) w |= ((buf[base + f] >> sh) & mask) << (f * nb);
                out[o] = w;
            }
        }
        if (std::fwrite(out.data(), 4, kOut, stdout) != kOut) return 0;
    }
}

}  // namespace

int main(int argc, char **argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
