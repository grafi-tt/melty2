// SPDX-License-Identifier: Apache-2.0
//
// prof [options] "structure" mlo mhi nkeys n : counter-differential bias profile at the depths mlo..mhi
// over random keys (see metric.hpp). The margin of a structure is its length minus the last depth whose
// maximum bias is clearly above the noise.
//   -s, --seed N        seed of the keys and the counters (default 1)
//   -w, --weak-words N  words given a zero or a rotation relation in the weak keys (default 0, none)
//   -p, --pidfile F     write the process ID to the file F (see pidfile.hpp)
#include <getopt.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "metric.hpp"
#include "pidfile.hpp"

namespace {

[[noreturn]] void usage() {
    std::fprintf(stderr, "usage: prof [-s seed] [-w weak_words] [-p pidfile] structure mlo mhi nkeys n\n");
    std::exit(1);
}

int run(int argc, char **argv) {
    static const option longopts[] = {
        {"seed", required_argument, nullptr, 's'},
        {"weak-words", required_argument, nullptr, 'w'},
        {"pidfile", required_argument, nullptr, 'p'},
        {nullptr, 0, nullptr, 0},
    };
    uint64_t seed = 1;
    int weak = 0, opt;
    const char *pidfile = nullptr;
    while ((opt = getopt_long(argc, argv, "s:w:p:", longopts, nullptr)) != -1) {
        switch (opt) {
        case 's': seed = std::strtoull(optarg, nullptr, 0); break;
        case 'w': weak = std::atoi(optarg); break;
        case 'p': pidfile = optarg; break;
        default: usage();
        }
    }
    if (argc - optind != 5) usage();
    if (pidfile) pidfile::write(pidfile);
    char **arg = argv + optind;
    const arx::Arx a = arx::parse(arg[0]);
    const int L = static_cast<int>(a.size());
    const int mlo = std::atoi(arg[1]), mhi = std::atoi(arg[2]), nk = std::atoi(arg[3]), n = std::atoi(arg[4]);
    if (mlo < 1 || mhi > L || mlo > mhi) {
        std::fprintf(stderr, "depths must satisfy 1 <= mlo <= mhi <= %d\n", L);
        return 1;
    }
    const metric::Profile p = metric::profile(a, mlo, mhi, nk, n, seed, weak);
    std::printf("noise(1 key)=%.2e\n", metric::noise(n));
    for (int m = mlo; m <= mhi; ++m) std::printf("depth %2d  mean %.3e  max %.3e\n", m, p.mean[m - mlo], p.max[m - mlo]);
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
