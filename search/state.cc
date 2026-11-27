// SPDX-License-Identifier: Apache-2.0
//
// state [flags] "structure" keylist : counter differentials of the state of words (default v0..v3, 128 bits)
// after the first M ops (default 12), for comparing how far the head of structures has mixed the words that
// the rest of the structure takes.
//
// For every key of keylist (lines "keymod[#seed]" as testu01_stest --keylist, "-" for none) and random counters c,
// the state S(c) is compared with S(c + 2^j) for j = 0..63 (the consecutive counters of a stream in either
// counter order, and the streams of the counters c + (s << j)). For every j:
//   bits: the flip frequency of every bit of S(c) ^ S(c + 2^j), as z-scores against 1/2 (largest |z|);
//   bytes: the chi-square of every byte of the xor difference and of the additive difference of every word
//          (S(c + 2^j) - S(c) mod 2^32), against the uniform 256 values, as z = (chi2 - 255) / sqrt(510)
//          (largest z).
// A line "delta <TAB> key=.. <TAB> j=.. <TAB> bit_z=.. <TAB> byte_z=.. <TAB> worst=.." per key and j (with --verbose),
// and a line "key <TAB> key=.. <TAB> max_z=.. <TAB> broken=.. <TAB> logsum=.. <TAB> worst_j=.. <TAB> bit_dev=.. <TAB>
// byte_dev=.. <TAB> unmixed=.." per key: the largest z over j, the number of j with a z above --broken (default 6),
// the sum over j of log(1 + z+) of the largest z of j (noise: a few per j), and the graded measures summed over j
// (as the z-scores saturate for a head far from random): bit_dev the sum over the bits of the squared bias
// (2p - 1)^2, unmixed the bits with |2p - 1| > 0.1, byte_dev the sum over the byte histograms of the chi-square
// divergence (chi2 - 255) / n (the noise subtracted). The last line "total <TAB> ..." sums them over the keys.
//   --ops M            the number of ops of the head (default 12)
//   --words LIST       the words of the state (default 0,1,2,3)
//   --samples N        the base counters per key (default 2^20)
//   --threads T        threads (default the hardware threads)
//   --seed N           the seed of the counters (default 1)
//   --broken Z         the z above which a delta counts as broken (default 6)
//   --verbose          a line per key and delta
#include <algorithm>
#include <array>
#include <csignal>
#include <cmath>
#include <format>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "arx.hpp"
#include "keymod.hpp"
#include "absl/strings/numbers.h"
#include "flags.hpp"
#include "logging.hpp"
#include "number.hpp"
#include "signal.hpp"

ABSL_FLAG(int, ops, 12, "the number of ops of the head");
ABSL_FLAG(std::vector<std::string>, words, (std::vector<std::string>{"0", "1", "2", "3"}), "the words of the state, comma-separated");
ABSL_FLAG(double, samples, 1 << 20, "the base counters per key (e.g. 1e6)");
ABSL_FLAG(int, threads, 0, "threads (0: the hardware threads)");
ABSL_FLAG(uint64_t, seed, 1, "the seed of the counters");
ABSL_FLAG(double, broken, 6, "the z above which a delta counts as broken");
ABSL_FLAG(bool, verbose, false, "a line per key and delta");

namespace {

constexpr int kDeltas = 64;

uint64_t splitmix(uint64_t &s) {
    uint64_t z = (s += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

// the words of the state after the first m ops
void state(const arx::Arx &a, int m, const arx::Key &k, uint64_t ctr, uint32_t (&v)[6]) {
    const uint32_t lo = static_cast<uint32_t>(ctr), hi = static_cast<uint32_t>(ctr >> 32);
    v[0] = k[0] ^ lo; v[1] = k[1] + hi; v[2] = k[2] ^ hi; v[3] = k[3] + lo; v[4] = k[4]; v[5] = k[5];
    for (int i = 0; i < m; ++i) {
        const arx::Op &o = a[i];
        const uint32_t s = v[o.s];
        switch (o.t) {
        case arx::kAdd: v[o.d] += s; break;
        case arx::kXor: v[o.d] ^= s; break;
        case arx::kSub: v[o.d] -= s; break;
        default: v[o.d] = arx::rotl(v[o.d] ^ s, o.r); break;
        }
    }
}

// the counts of one thread: per delta, the flips of every bit, and the byte histograms (xor, then additive)
struct Counts {
    int nw;
    std::vector<uint64_t> bits;   // [delta][32 * nw]
    std::vector<uint32_t> bytes;  // [delta][2][4 * nw][256]
    explicit Counts(int nw_) : nw(nw_), bits(size_t(kDeltas) * 32 * nw_), bytes(size_t(kDeltas) * 2 * 4 * nw_ * 256) {}
};

void count(const arx::Arx &a, int m, const std::vector<int> &words, const arx::Key &k, uint64_t seed, uint64_t n,
           Counts &c) {
    const int nw = static_cast<int>(words.size());
    uint32_t v0[6], v1[6];
    for (uint64_t i = 0; i < n; ++i) {
        // (a stop signal: the counts are dropped, see run)
        if ((i & 0xfff) == 0 && signals::stop_requested()) return;
        const uint64_t ctr = splitmix(seed);
        state(a, m, k, ctr, v0);
        for (int j = 0; j < kDeltas; ++j) {
            state(a, m, k, ctr + (UINT64_C(1) << j), v1);
            uint64_t *bits = &c.bits[size_t(j) * 32 * nw];
            uint32_t *bx = &c.bytes[size_t(j) * 2 * 4 * nw * 256], *ba = bx + 4 * nw * 256;
            for (int w = 0; w < nw; ++w) {
                const uint32_t x = v0[words[w]] ^ v1[words[w]], d = v1[words[w]] - v0[words[w]];
                for (int b = 0; b < 32; ++b) bits[32 * w + b] += x >> b & 1;
                for (int y = 0; y < 4; ++y) {
                    ++bx[(4 * w + y) * 256 + (x >> 8 * y & 255)];
                    ++ba[(4 * w + y) * 256 + (d >> 8 * y & 255)];
                }
            }
        }
    }
}

int run(int argc, char **argv) {
    const std::vector<std::string_view> arg =
        parse_flags(argc, argv, "state [flags] structure keylist: the counter differentials of the state after the head of a structure");
    const int m = absl::GetFlag(FLAGS_ops);
    const int threads = absl::GetFlag(FLAGS_threads) > 0 ? absl::GetFlag(FLAGS_threads) : static_cast<int>(std::thread::hardware_concurrency());
    const uint64_t n = static_cast<uint64_t>(absl::GetFlag(FLAGS_samples)), seed = absl::GetFlag(FLAGS_seed);
    const double broken = absl::GetFlag(FLAGS_broken);
    const bool verbose = absl::GetFlag(FLAGS_verbose);
    std::vector<int> words;
    for (const std::string &w : absl::GetFlag(FLAGS_words)) {
        int x;
        if (!absl::SimpleAtoi(w, &x)) throw std::runtime_error("invalid word: " + w);
        words.push_back(x);
    }
    if (arg.size() != 2 || threads < 1) usage();
    const arx::Arx a = arx::parse(arg[0]);
    if (m < 1 || m > static_cast<int>(a.size())) throw std::runtime_error("M out of range");
    for (int w : words)
        if (w < 0 || w > 5) throw std::runtime_error("word out of range");
    std::vector<std::string> specs;
    {
        std::ifstream f{std::string(arg[1])};
        if (!f) throw std::runtime_error("cannot read the key list");
        for (std::string line; std::getline(f, line);)
            if (!line.empty() && line[0] != '#') specs.push_back(line);
    }
    const int nw = static_cast<int>(words.size());
    double tot_max = 0, tot_log = 0, tot_bdev = 0, tot_ydev = 0;
    int tot_broken = 0, tot_unmixed = 0;
    for (const std::string &spec : specs) {
        const size_t hash = spec.find('#');
        const std::string km = spec.substr(0, hash);
        const uint32_t kseed = hash == std::string::npos ? 1 : number::parse<uint32_t>(std::string_view(spec).substr(hash + 1));
        arx::Key k = keymod::seed_key(kseed);
        if (km != "-") keymod::apply(k, km);
        std::vector<Counts> cs(threads, Counts(nw));
        std::vector<std::thread> ts;
        for (int t = 0; t < threads; ++t) {
            const uint64_t nt = n / threads + (uint64_t(t) < n % threads);
            ts.emplace_back(count, std::cref(a), m, std::cref(words), std::cref(k), seed * 1000003 + t, nt, std::ref(cs[t]));
        }
        for (auto &t : ts) t.join();
        signals::throw_if_stopped();
        for (int t = 1; t < threads; ++t) {
            for (size_t i = 0; i < cs[0].bits.size(); ++i) cs[0].bits[i] += cs[t].bits[i];
            for (size_t i = 0; i < cs[0].bytes.size(); ++i) cs[0].bytes[i] += cs[t].bytes[i];
        }
        const Counts &c = cs[0];
        // the graded measures (the state of a head is far from random, so the z-scores saturate): per delta, the
        // sum over the bits of the squared bias (2p - 1)^2, the number of bits with |2p - 1| > 0.1 ("unmixed"),
        // and the sum over the byte histograms of the chi-square divergence (chi2 - 255) / n; the noise is
        // subtracted (32 nw / n and 0)
        double kmax = 0, klog = 0, kbdev = 0, kydev = 0;
        int kbroken = 0, kworst = 0, kunmixed = 0;
        for (int j = 0; j < kDeltas; ++j) {
            double bz = 0, yz = 0, bdev = 0, ydev = 0;
            int unmixed = 0;
            std::string worst;
            for (int i = 0; i < 32 * nw; ++i) {
                const double f = double(c.bits[size_t(j) * 32 * nw + i]) / n, bias = 2 * f - 1;
                const double z = std::fabs((double(c.bits[size_t(j) * 32 * nw + i]) - n / 2.0) / std::sqrt(n / 4.0));
                bdev += bias * bias - 1.0 / n;
                unmixed += std::fabs(bias) > 0.1;
                if (z > bz) bz = z, worst = "bit v" + std::to_string(words[i / 32]) + ":" + std::to_string(i % 32);
            }
            for (int kind = 0; kind < 2; ++kind)
                for (int h = 0; h < 4 * nw; ++h) {
                    const uint32_t *hist = &c.bytes[((size_t(j) * 2 + kind) * 4 * nw + h) * 256];
                    double chi2 = 0;
                    const double e = n / 256.0;
                    for (int b = 0; b < 256; ++b) chi2 += (hist[b] - e) * (hist[b] - e) / e;
                    const double z = (chi2 - 255) / std::sqrt(510.0);
                    ydev += (chi2 - 255) / n;
                    if (z > yz && z > bz)
                        worst = std::string(kind ? "add" : "xor") + " v" + std::to_string(words[h / 4]) + " byte " + std::to_string(h % 4);
                    yz = std::max(yz, z);
                }
            const double z = std::max(bz, yz);
            if (verbose)
                std::cout << std::format("delta\tkey={}\tj={}\tbit_z={:.2f}\tbyte_z={:.2f}\tbit_dev={:.4g}\tbyte_dev={:.4g}\tunmixed={}\tworst={}\n",
                                         spec, j, bz, yz, bdev, ydev, unmixed, worst);
            if (z > kmax) kmax = z, kworst = j;
            kbroken += z > broken;
            klog += std::log1p(z);
            kbdev += bdev;
            kydev += ydev;
            kunmixed += unmixed;
        }
        std::cout << std::format("key\tkey={}\tmax_z={:.2f}\tbroken={}\tlogsum={:.2f}\tworst_j={}\tbit_dev={:.4g}\tbyte_dev={:.4g}\tunmixed={}",
                                 spec, kmax, kbroken, klog, kworst, kbdev, kydev, kunmixed)
                  << std::endl;
        tot_bdev += kbdev;
        tot_ydev += kydev;
        tot_unmixed += kunmixed;
        tot_max = std::max(tot_max, kmax);
        tot_broken += kbroken;
        tot_log += klog;
    }
    std::cout << std::format("total\tkeys={}\tmax_z={:.2f}\tbroken={}\tlogsum={:.2f}\tbit_dev={:.4g}\tbyte_dev={:.4g}\tunmixed={}\n", specs.size(),
                             tot_max, tot_broken, tot_log, tot_bdev, tot_ydev, tot_unmixed);
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
