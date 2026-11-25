// SPDX-License-Identifier: Apache-2.0
// bcsub [-b battery] seed swap [tests...] : tests of TestU01 for the compiled-in structure (see
// run.sh), by default the selected tests of BigCrush (by the numbers of bbattery_RepeatBigCrush),
// which are a subset that took long for weak structures if no test is given.
//   -b, --battery B  small, crush or big for the whole SmallCrush, Crush or BigCrush
//   -m, --streams S  the interleaved outputs of 2^b streams "ctr:b:c" (b <= 12), as testu01_stest -m:
//                    the output at the position p is that of the stream s = p mod 2^b at its index
//                    p >> b, of the counter index + (s << c)
#include <getopt.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include <bbattery.h>
#include <unif01.h>
}

#include "keymod.hpp"

#include "kern.hpp"

namespace {

// the generator state for next(), which TestU01 calls without arguments
struct Stream {
    int swp = 0;
    arx::Key key{};
    int sbits = 0, cbit = 0;          // the streams (see -m); sbits = 0 for a single stream
    std::vector<uint32_t> buf = std::vector<uint32_t>(1024);
    std::array<uint32_t, kLanes> tmp{};
    uint64_t ctr = 0;                 // the index of the next block in every stream
    size_t idx = 1024;
} stream;

void refill() {
    auto &st = stream;
    if (st.sbits == 0) {
        for (size_t i = 0; i < st.buf.size(); i += kLanes, st.ctr += kLanes)
            block(st.key.data(), st.ctr, st.swp, st.buf.data() + i);
        return;
    }
    const uint64_t ns = uint64_t{1} << st.sbits;
    for (uint64_t s = 0; s < ns; ++s) {
        block(st.key.data(), st.ctr + (s << st.cbit), st.swp, st.tmp.data());
        for (int l = 0; l < kLanes; ++l) st.buf[(static_cast<uint64_t>(l) << st.sbits) | s] = st.tmp[l];
    }
    st.ctr += kLanes;
}

unsigned int next() {
    if (stream.idx >= stream.buf.size()) {
        refill();
        stream.idx = 0;
    }
    return stream.buf[stream.idx++];
}

constexpr int kDefaultTests[] = {12, 22, 23, 24, 34, 40, 41, 42, 43, 44, 45, 66, 67, 76, 77, 78, 79, 80, 81, 82, 93, 94, 95, 96};

[[noreturn]] void usage() {
    std::fprintf(stderr, "usage: bcsub [-b small|crush|big] [-m ctr:b:c] seed swap [tests...]\n");
    std::exit(1);
}

int run(int argc, char **argv) {
    static const option longopts[] = {
        {"battery", required_argument, nullptr, 'b'},
        {"streams", required_argument, nullptr, 'm'},
        {nullptr, 0, nullptr, 0},
    };
    std::string battery;
    int opt;
    while ((opt = getopt_long(argc, argv, "b:m:", longopts, nullptr)) != -1) {
        switch (opt) {
        case 'b': battery = optarg; break;
        case 'm':
            if (std::sscanf(optarg, "ctr:%d:%d", &stream.sbits, &stream.cbit) != 2 || stream.sbits < 1 || stream.sbits > 12 ||
                stream.cbit < 0 || stream.cbit > 63) {
                std::fprintf(stderr, "invalid streams (ctr:b:c, 1 <= b <= 12): %s\n", optarg);
                return 1;
            }
            stream.buf.assign(static_cast<size_t>(kLanes) << stream.sbits, 0);
            stream.idx = stream.buf.size();
            break;
        default: usage();
        }
    }
    if (argc - optind < 2) usage();
    if (!battery.empty() && battery != "small" && battery != "crush" && battery != "big") usage();
    if (!battery.empty() && argc - optind > 2) {
        std::fprintf(stderr, "test numbers are only for the selected tests of BigCrush\n");
        return 1;
    }
    char **arg = argv + optind;
    const int narg = argc - optind;
    const auto s0 = static_cast<uint32_t>(std::strtoul(arg[0], nullptr, 0));
    stream.swp = std::atoi(arg[1]);
    stream.key = keymod::seed_key(s0);
    keymod::from_env(stream.key);
    std::array<int, 200> rep{};
    if (narg > 2) {
        for (int i = 2; i < narg; ++i) {
            const int t = std::atoi(arg[i]);
            if (t < 1 || t > 106) {
                std::fprintf(stderr, "test numbers must be in [1, 106]\n");
                return 1;
            }
            rep[t] = 1;
        }
    } else {
        for (const int t : kDefaultTests) rep[t] = 1;
    }
    std::string name = "M=" + std::to_string(kKernM) + " seed0=" + std::to_string(s0) + (stream.swp ? " swap32" : "") +
                       (stream.sbits ? " streams=ctr:" + std::to_string(stream.sbits) + ":" + std::to_string(stream.cbit) : "");
    unif01_Gen *g = unif01_CreateExternGenBits(name.data(), next);
    if (battery.empty()) bbattery_RepeatBigCrush(g, rep.data());
    else if (battery == "small") bbattery_SmallCrush(g);
    else if (battery == "crush") bbattery_Crush(g);
    else bbattery_BigCrush(g);
    unif01_DeleteExternGenBits(g);
    return 0;
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
