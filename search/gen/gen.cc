// SPDX-License-Identifier: Apache-2.0
// gen seed swap : the output stream of the compiled-in structure (see run.sh) as 32bit little-endian
// words to stdout, e.g. for "| RNG_test stdin32" of PractRand. The key is that of
// melty2_init(seed, 0, 0, 0), modified by MELTY2_KEYMOD; with swap = 1, the halves of the counter are
// swapped.
#include <getopt.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "keymod.hpp"

#include "kern.hpp"

namespace {

[[noreturn]] void usage() {
    std::fprintf(stderr, "usage: gen seed swap\n");
    std::exit(1);
}

int run(int argc, char **argv) {
    static const option longopts[] = {{nullptr, 0, nullptr, 0}};
    if (getopt_long(argc, argv, "", longopts, nullptr) != -1 || argc - optind != 2) usage();
    char **arg = argv + optind;
    const int swp = std::atoi(arg[1]);
    arx::Key key = keymod::seed_key(static_cast<uint32_t>(std::strtoul(arg[0], nullptr, 0)));
    keymod::from_env(key);
    constexpr int kBuf = 1 << 16;
    std::vector<uint32_t> buf(kBuf);
    uint64_t ctr = 0;
    for (;;) {
        for (int i = 0; i < kBuf; i += kLanes, ctr += kLanes) block(key.data(), ctr, swp, buf.data() + i);
        if (std::fwrite(buf.data(), 4, kBuf, stdout) != kBuf) return 0;
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
