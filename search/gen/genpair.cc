// SPDX-License-Identifier: Apache-2.0
// genpair seed swap mask mode : a stream of the compiled-in structure (see run.sh) combining two keys
// which share the words in mask (bit i set: word i is shared). Key A is from seed, and key B takes the
// other words from seed + 0x10000. mode 0: A(c) ^ B(c); 1: A(c), B(c) interleaved; 2: A(c) - B(c).
#include <array>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "flags.hpp"
#include "keymod.hpp"
#include "logging.hpp"

#include "kern.hpp"

namespace {

int run(int argc, char **argv) {
    const std::vector<char *> arg = parse_flags(argc, argv, "genpair seed swap mask mode");
    if (arg.size() != 4) usage();
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
    init_logging();
    try {
        return run(argc, argv);
    } catch (const std::exception &e) {
        LOG(ERROR) << e.what();
        return 1;
    }
}
