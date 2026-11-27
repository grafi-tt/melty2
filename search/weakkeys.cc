// SPDX-License-Identifier: Apache-2.0
//
// weakkeys [flags] "structure" mlo mhi n list : differential bias of structured keys.
//   --nbase N        number of base keys (default 2)
//   --dump FILE      file to write the bias of every key at every depth
//   --pidfile F      write the process ID to the file F (see pidfile.hpp)
//
// Every line of the list file is a key modification spec (see keymod.hpp), applied to the keys of the
// seeds 1..nbase. 64 keys of the seeds 1000..1063 are the control. For every depth, the control
// distribution, the number of structured keys above the control maximum, and the worst 8 structured
// keys are printed. With dump, the bias of every key at every depth is written to the file as
// tab-separated values.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>

#include "flags.hpp"
#include "keymod.hpp"
#include "logging.hpp"
#include "metric.hpp"
#include "pidfile.hpp"

ABSL_FLAG(int, nbase, 2, "number of base keys");
ABSL_FLAG(std::string, dump, "", "file to write the bias of every key at every depth");
ABSL_FLAG(std::string, pidfile, "", "write the process ID to this file (see pidfile.hpp)");

namespace {

constexpr int kControls = 64;
constexpr int kChunk = 256;  // keys per call of the metric, to bound the memory of the GPU metric

int run(int argc, char **argv) {
    const std::vector<char *> arg = parse_flags(argc, argv, "weakkeys [flags] structure mlo mhi n list: the differential bias of structured keys");
    if (arg.size() != 5) usage();
    const int nbase = absl::GetFlag(FLAGS_nbase);
    const std::string dump_file = absl::GetFlag(FLAGS_dump);
    const char *dump = dump_file.empty() ? nullptr : dump_file.c_str();
    if (const std::string pf = absl::GetFlag(FLAGS_pidfile); !pf.empty()) pidfile::write(pf.c_str());
    const arx::Arx a = arx::parse(arg[0]);
    const int mlo = std::atoi(arg[1]), mhi = std::atoi(arg[2]), n = std::atoi(arg[3]);
    const std::string list = arg[4];

    // read the specs, skipping empty lines and comments
    std::vector<std::string> mods;
    {
        std::ifstream f(list);
        if (!f) throw std::runtime_error(list + ": cannot open");
        std::string line;
        while (std::getline(f, line)) {
            line.erase(std::find_if(line.begin(), line.end(), [](char c) { return c == '\r' || c == '\n'; }), line.end());
            if (!line.empty() && line[0] != '#') mods.push_back(line);
        }
    }
    const int nm = static_cast<int>(mods.size()), ns = nm * nbase;

    // keys[0 .. kControls - 1]: the control keys; then nbase keys for every spec, i.e. the spec applied
    // to the keys of the seeds 1 .. nbase
    std::vector<arx::Key> keys;
    for (int c = 0; c < kControls; ++c) keys.push_back(keymod::seed_key(1000 + c));
    for (const std::string &mod : mods)
        for (int b = 0; b < nbase; ++b) {
            arx::Key k = keymod::seed_key(1 + b);
            keymod::apply(k, mod.c_str());
            keys.push_back(k);
        }
    const int nk = static_cast<int>(keys.size()), nd = mhi - mlo + 1;
    std::vector<double> pk;
    pk.reserve(static_cast<size_t>(nk) * nd);
    for (int s = 0; s < nk; s += kChunk) {
        const std::vector<arx::Key> chunk(keys.begin() + s, keys.begin() + std::min(nk, s + kChunk));
        const std::vector<double> r = metric::bias_per_key(a, mlo, mhi, chunk, n, 77);
        pk.insert(pk.end(), r.begin(), r.end());
    }
    const auto bias = [&](int key, int d) { return pk[static_cast<size_t>(key) * nd + d]; };
    // the spec of the structured key m (0-based, after the controls)
    const auto spec = [&](int m) -> const std::string & { return mods[m / nbase]; };

    std::printf("noise(1 key)=%.2e  structured keys=%d x %d bases\n", metric::noise(n), nm, nbase);
    if (dump) {
        const std::unique_ptr<std::FILE, int (*)(std::FILE *)> o(std::fopen(dump, "w"), std::fclose);
        if (!o) throw std::runtime_error(std::string(dump) + ": cannot open");
        for (int k = 0; k < nk; ++k) {
            if (k < kControls) std::fprintf(o.get(), "CTRL\t%d", k);
            else std::fprintf(o.get(), "%s\t%d", spec(k - kControls).c_str(), (k - kControls) % nbase + 1);
            for (int d = 0; d < nd; ++d) std::fprintf(o.get(), "\t%.4e", bias(k, d));
            std::fprintf(o.get(), "\n");
        }
    }
    for (int d = 0; d < nd; ++d) {
        double mx = -1e9, sum = 0;
        for (int c = 0; c < kControls; ++c) {
            sum += bias(c, d);
            mx = std::max(mx, bias(c, d));
        }
        double wmx = -1e9;
        int wi = -1, cnt = 0;
        for (int m = 0; m < ns; ++m) {
            const double v = bias(kControls + m, d);
            if (v > wmx) {
                wmx = v;
                wi = m;
            }
            if (v > mx) cnt++;
        }
        std::printf("depth %2d  control mean %.2e max %.2e | structured max %.2e (%s) ; %d above control max\n",
                    mlo + d, sum / kControls, mx, wmx, wi >= 0 ? spec(wi).c_str() : "-", cnt);
    }
    for (int d = 0; d < nd && nm > 0; ++d) {
        // the worst 8 structured keys, the first of equal ones first
        std::vector<int> order(ns);
        std::iota(order.begin(), order.end(), 0);
        const int top = std::min(8, ns);
        std::partial_sort(order.begin(), order.begin() + top, order.end(), [&](int x, int y) {
            const double bx = bias(kControls + x, d), by = bias(kControls + y, d);
            return bx > by || (bx == by && x < y);
        });
        for (int t = 0; t < top; ++t) {
            const int m = order[t];
            std::printf("  worst@%d: %-24s base %d  bias %.2e\n", mlo + d, spec(m).c_str(), m % nbase + 1,
                        bias(kControls + m, d));
        }
    }
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
