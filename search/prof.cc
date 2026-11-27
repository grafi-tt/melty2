// SPDX-License-Identifier: Apache-2.0
//
// prof [flags] "structure" mlo mhi nkeys n : counter-differential bias profile at the depths mlo..mhi
// over random keys (see metric.hpp). The margin of a structure is its length minus the last depth whose
// maximum bias is clearly above the noise.
//   --seed N            seed of the keys and the counters (default 1)
//   --weak_words N      words given a zero or a rotation relation in the weak keys (default 0, none)
//   --pidfile F         write the process ID to the file F (see pidfile.hpp)
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "flags.hpp"
#include "logging.hpp"
#include "metric.hpp"
#include "pidfile.hpp"

ABSL_FLAG(uint64_t, seed, 1, "seed of the keys and the counters");
ABSL_FLAG(int, weak_words, 0, "words given a zero or a rotation relation in the weak keys (0: none)");
ABSL_FLAG(std::string, pidfile, "", "write the process ID to this file (see pidfile.hpp)");

namespace {

int run(int argc, char **argv) {
    const std::vector<char *> arg = parse_flags(argc, argv, "prof [flags] structure mlo mhi nkeys n: the counter-differential bias profile");
    if (arg.size() != 5) usage();
    const uint64_t seed = absl::GetFlag(FLAGS_seed);
    const int weak = absl::GetFlag(FLAGS_weak_words);
    if (const std::string pf = absl::GetFlag(FLAGS_pidfile); !pf.empty()) pidfile::write(pf.c_str());
    const arx::Arx a = arx::parse(arg[0]);
    const int L = static_cast<int>(a.size());
    const int mlo = std::atoi(arg[1]), mhi = std::atoi(arg[2]), nk = std::atoi(arg[3]), n = std::atoi(arg[4]);
    if (mlo < 1 || mhi > L || mlo > mhi) {
        LOG(ERROR) << "depths must satisfy 1 <= mlo <= mhi <= " << L;
        return 1;
    }
    const metric::Profile p = metric::profile(a, mlo, mhi, nk, n, seed, weak);
    std::printf("noise(1 key)=%.2e\n", metric::noise(n));
    for (int m = mlo; m <= mhi; ++m) std::printf("depth %2d  mean %.3e  max %.3e\n", m, p.mean[m - mlo], p.max[m - mlo]);
    return 0;
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
