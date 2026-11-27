// SPDX-License-Identifier: Apache-2.0
//
// rank [flags] list : the margin of every structure in the list file, at a high resolution.
//   --nkeys N           number of keys (default 32)
//   --pairs N           counter pairs per key and delta (default 2^24)
//   --seed N            seed of the keys and the counters (default 777, not used by sa)
//   --weak_words N      words given a zero or a rotation relation in the weak keys (default 6)
//   --keyset N          0: the keys of metric::random_keys; 1: also sparse v[4] and v[5] (default)
//   --pidfile F         write the process ID to the file F (see pidfile.hpp)
//
// This is the second stage of the search: sa generates candidates, whose objective is evaluated at a
// resolution too low to see small biases near the output, and rank judges them. Every line of the
// list is a structure, optionally after a name and a tab; empty lines and lines starting with # are
// skipped. For every structure of L ops, the maximum bias over the keys is measured at the depths
// L - 10 to L, and the last depth whose maximum bias is above 5 times the noise level is the last
// biased depth. A line is printed per structure, as soon as it is measured:
//
//   margin <TAB> last biased depth <TAB> name <TAB> structure <TAB> the maximum biases
//
// where the margin is L minus the last biased depth (L - 10 if none is biased, i.e. at least 10).
#include <algorithm>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>

#include "flags.hpp"
#include "logging.hpp"
#include "metric.hpp"
#include "pidfile.hpp"
#include "signal.hpp"

ABSL_FLAG(int, nkeys, 32, "number of keys");
ABSL_FLAG(int, pairs, 1 << 24, "counter pairs per key and delta");
ABSL_FLAG(uint64_t, seed, 777, "seed of the keys and the counters (not used by sa)");
ABSL_FLAG(int, weak_words, 6, "words given a zero or a rotation relation in the weak keys");
ABSL_FLAG(int, keyset, 1, "0: the keys of metric::random_keys; 1: also sparse v[4] and v[5]");
ABSL_FLAG(std::string, pidfile, "", "write the process ID to this file (see pidfile.hpp)");

namespace {

constexpr int kW = 10;

int run(int argc, char **argv) {
    const std::vector<char *> arg = parse_flags(argc, argv, "rank [flags] list: the margin of every structure of the list");
    if (arg.size() != 1) usage();
    const int nk = absl::GetFlag(FLAGS_nkeys), n = absl::GetFlag(FLAGS_pairs), weak = absl::GetFlag(FLAGS_weak_words),
              keyset = absl::GetFlag(FLAGS_keyset);
    const uint64_t seed = absl::GetFlag(FLAGS_seed);
    if (const std::string pf = absl::GetFlag(FLAGS_pidfile); !pf.empty()) pidfile::write(pf.c_str());
    const std::string list = arg[0];
    std::ifstream f(list);
    if (!f) throw std::runtime_error(list + ": cannot open");

    const std::vector<metric::Key> keys =
        keyset == 1 ? metric::sparse45_keys(nk, seed, weak) : metric::random_keys(nk, seed, weak);
    const double threshold = 5 * metric::noise(n);
    std::printf("# keys=%d pairs=%d seed=%llu keyset=%d threshold=%.2e\n", nk, n, static_cast<unsigned long long>(seed),
                keyset, threshold);

    std::string line;
    while (std::getline(f, line)) {
        line.erase(std::find_if(line.begin(), line.end(), [](char c) { return c == '\r' || c == '\n'; }), line.end());
        if (line.empty() || line[0] == '#') continue;
        signals::throw_if_stopped();
        const size_t tab = line.find('\t');
        const std::string name = tab == std::string::npos ? "-" : line.substr(0, tab);
        const arx::Arx a = arx::parse(tab == std::string::npos ? line : line.substr(tab + 1));
        const int L = static_cast<int>(a.size()), mlo = std::max(L - kW, 1), nd = L - mlo + 1;
        const std::vector<double> pk = metric::bias_per_key(a, mlo, L, keys, n, seed);
        std::vector<double> bx(nd, 0.0);
        int last = mlo - 1;
        for (int m = 0; m < nd; ++m) {
            for (int key = 0; key < nk; ++key) bx[m] = std::fmax(bx[m], pk[key * nd + m]);
            if (bx[m] > threshold) last = mlo + m;
        }
        std::printf("%d\t%d\t%s\t%s\t", L - last, last, name.c_str(), arx::to_string(a).c_str());
        for (int m = 0; m < nd; ++m) std::printf("%s%.1e", m ? " " : "", bx[m]);
        std::printf("\n");
        std::fflush(stdout);
    }
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    init_logging();
    signals::watch({SIGINT, SIGTERM});
    try {
        return run(argc, argv);
    } catch (const signals::Interrupted &) {
        LOG(WARNING) << "interrupted";
        return signals::kInterrupted;
    } catch (const std::exception &e) {
        LOG(ERROR) << e.what();
        return 1;
    }
}
