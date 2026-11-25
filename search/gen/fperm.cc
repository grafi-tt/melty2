// SPDX-License-Identifier: Apache-2.0
// fperm seed swap N r t : permutation test of N t-tuples of consecutive outputs of the compiled-in
// structure (see run.sh), after dropping the top r bits (the statistic of sknuth_Permutation of
// TestU01). Prints chi2, and the z score and the upper-tail p-value by the Wilson-Hilferty
// approximation.
#include <getopt.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "keymod.hpp"

#include "kern.hpp"

namespace {

[[noreturn]] void usage() {
    std::fprintf(stderr, "usage: fperm seed swap N r t\n");
    std::exit(1);
}

int run(int argc, char **argv) {
    static const option longopts[] = {{nullptr, 0, nullptr, 0}};
    if (getopt_long(argc, argv, "", longopts, nullptr) != -1 || argc - optind != 5) usage();
    char **arg = argv + optind;
    const auto seed = static_cast<uint32_t>(std::strtoul(arg[0], nullptr, 0));
    const int swp = std::atoi(arg[1]);
    const long N = std::atol(arg[2]);
    const int r = std::atoi(arg[3]), t = std::atoi(arg[4]);
    if (t < 2 || t > 8) {
        std::fprintf(stderr, "t must be in [2, 8]\n");
        return 1;
    }
    arx::Key key = keymod::seed_key(seed);
    keymod::from_env(key);
    int fact = 1;
    for (int i = 2; i <= t; ++i) fact *= i;
    std::vector<uint64_t> cnt(fact);
    std::array<uint32_t, kLanes> buf;
    std::array<uint32_t, 8> u;
    int idx = kLanes;
    uint64_t ctr = 0;
    for (long n = 0; n < N; ++n) {
        for (int j = 0; j < t; ++j) {
            if (idx == kLanes) {
                block(key.data(), ctr, swp, buf.data());
                ctr += kLanes;
                idx = 0;
            }
            u[j] = buf[idx++] << r;
        }
        int code = 0;  // Lehmer code of the order
        for (int j = 0; j < t - 1; ++j) {
            int c = 0;
            for (int q = j + 1; q < t; ++q) c += u[q] < u[j];
            code = code * (t - j) + c;
        }
        cnt[code]++;
    }
    const double e = static_cast<double>(N) / fact;
    double chi = 0;
    for (const uint64_t c : cnt) {
        const double d = c - e;
        chi += d * d / e;
    }
    const double df = fact - 1;
    const double z = (std::cbrt(chi / df) - (1 - 2 / (9 * df))) / std::sqrt(2 / (9 * df));
    const double p = 0.5 * std::erfc(z / std::sqrt(2));
    std::printf("M=%d seed=%u swap=%d N=%ld r=%d t=%d chi2=%.1f z=%.2f p=%.3g\n", kKernM, seed, swp, N, r, t, chi, z, p);
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
