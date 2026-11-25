// SPDX-License-Identifier: Apache-2.0
// genpair seed swap mask mode : a stream of the compiled-in structure (see run.sh) combining two keys
// which share the words in mask (bit i set: word i is shared). Key A is from seed, and key B takes the
// other words from seed + 0x10000. mode 0: A(c) ^ B(c); 1: A(c), B(c) interleaved; 2: A(c) - B(c).
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
    std::fprintf(stderr, "usage: genpair seed swap mask mode\n");
    std::exit(1);
}

int run(int argc, char **argv) {
    static const option longopts[] = {{nullptr, 0, nullptr, 0}};
    if (getopt_long(argc, argv, "", longopts, nullptr) != -1 || argc - optind != 4) usage();
    char **arg = argv + optind;
    const auto seed = static_cast<uint32_t>(std::strtoul(arg[0], nullptr, 0));
    const int swp = std::atoi(arg[1]), mask = static_cast<int>(std::strtol(arg[2], nullptr, 0)), mode = std::atoi(arg[3]);
    arx::Key a = keymod::seed_key(seed), b = keymod::seed_key(seed + 0x10000);
    keymod::from_env(a);
    for (int i = 0; i < 6; ++i)
        if (mask >> i & 1) b[i] = a[i];
    constexpr int kOut = 1 << 14;
    std::vector<uint32_t> out(kOut);
    std::array<uint32_t, kLanes> ba, bb;
    uint64_t ctr = 0;
    for (;;) {
        for (int o = 0; o < kOut;) {
            block(a.data(), ctr, swp, ba.data());
            block(b.data(), ctr, swp, bb.data());
            ctr += kLanes;
            for (int l = 0; l < kLanes; ++l) {
                if (mode == 0) out[o++] = ba[l] ^ bb[l];
                else if (mode == 2) out[o++] = ba[l] - bb[l];
                else {
                    out[o++] = ba[l];
                    out[o++] = bb[l];
                }
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
