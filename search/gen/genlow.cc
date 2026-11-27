// SPDX-License-Identifier: Apache-2.0
// genlow [flags] seed swap nbits : the lower nbits bits (after >> shift) of the outputs of the
// compiled-in structure (see run.sh), packed into 32bit words from the LSB, to stdout. The least
// significant bits of ARX outputs mix slowly, as an addition has no carry into bit 0, so their stream
// is tested separately.
//   --shift N      right shift of the outputs before taking the lower bits (default 0)
#include <array>
#include <cstdio>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "flags.hpp"
#include "keymod.hpp"
#include "logging.hpp"
#include "number.hpp"

#include "kern.hpp"

ABSL_FLAG(int, shift, 0, "right shift of the outputs before taking the lower bits");

namespace {

int run(int argc, char **argv) {
    const std::vector<std::string_view> arg = parse_flags(argc, argv, "genlow [flags] seed swap nbits");
    if (arg.size() != 3) usage();
    const int sh = absl::GetFlag(FLAGS_shift);
    const auto seed = number::parse<uint32_t>(arg[0]);
    const int swp = number::parse<int>(arg[1]), nb = number::parse<int>(arg[2]);
    if (nb < 1 || nb > 32 || 32 % nb != 0) {
        LOG(ERROR) << "nbits must divide 32";
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
    init_logging();
    try {
        return run(argc, argv);
    } catch (const std::exception &e) {
        LOG(ERROR) << e.what();
        return 1;
    }
}
