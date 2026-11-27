// SPDX-License-Identifier: Apache-2.0
// gen seed swap : the output stream of the compiled-in structure (see run.sh) as 32bit little-endian
// words to stdout, e.g. for "| RNG_test stdin32" of PractRand. The key is that of
// melty2_init(seed, 0, 0, 0), modified by MELTY2_KEYMOD; with swap = 1, the halves of the counter are
// swapped.
#include <cstdio>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "flags.hpp"
#include "keymod.hpp"
#include "logging.hpp"
#include "number.hpp"

#include "kern.hpp"

namespace {

int run(int argc, char **argv) {
    const std::vector<std::string_view> arg = parse_flags(argc, argv, "gen seed swap");
    if (arg.size() != 2) usage();
    const int swp = number::parse<int>(arg[1]);
    arx::Key key = keymod::seed_key(number::parse<uint32_t>(arg[0]));
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
    init_logging();
    try {
        return run(argc, argv);
    } catch (const std::exception &e) {
        LOG(ERROR) << e.what();
        return 1;
    }
}
