// SPDX-License-Identifier: Apache-2.0
//
// testu01_stest [options] "structure" M : the tests of BigCrush but MatrixRank, AppearanceSpacings,
// LinearComp and LempelZiv, implemented on GPU, for the outputs of the first M ops of the
// structure.
//   -s, --seed N     the key is that of melty2_init(N, 0, 0, 0) (default 1), modified by MELTY2_KEYMOD
//   -w, --swap       swap the halves of the counter (the other counter order)
//   -t, --tests LIST comma-separated tests: poker, coupon, gap, maxoft, collision, birthday, closepairs,
//                    permutation, run, serial, collpermut, weight, savir2, gcd, hamweight, hamcorr,
//                    hamindep, bitrun, longhead, autocor, samplecorr, sampleprod, samplemean, walk,
//                    periods, sumcoll, fourier (default all)
//   -f, --scale X    multiply the sample sizes n of BigCrush by X (default 1)
//   -l, --keylist F  run the tests for every key modification spec of the file F (one per line, "-" for
//                    none; instead of MELTY2_KEYMOD) in one process; a spec "keymod@streams" takes the
//                    streams (as -m) for it, and a spec "...#seed" the key of that seed (as -s)
//   -o, --offset X   start the counters of the tests at X instead of 0 (e.g. those of the later tests
//                    of BigCrush, which TestU01 runs on a single stream from 0)
//   -m, --streams S  test the interleaved outputs of 2^b streams instead of a single stream: "ctr:b:c"
//                    for the streams of the counters index + (s << c), as with a stream number in the
//                    high bits of the counter; "key:b:w" for the streams of the keys with the word w
//                    xored with s (related keys). The output i is that of the stream i mod 2^b.
//
// After the tests of a key, a line "summary <TAB> M=.. <TAB> seed=.. <TAB> swap=.. <TAB> keymod=.. <TAB>
// min_p=.. <TAB> max_p=.. <TAB> flagged=.. <TAB> streams=.." gives the smallest and the largest p-values and the number
// of them outside [0.001, 0.999].
//   -p, --pidfile F  write the process ID to the file F (see pidfile.hpp)
//
// testu01_stest -S M[,M...] [-s seed] [-t tests] [-f scale] [-l keylist] [-m streams] [-o offset]
// [-D deadline] : the score mode for searches. For every structure of stdin (one per line), the tests
// run quietly for the outputs of its first M ops (or of each M of the list, e.g. 15,16), for every key
// (as -l, or MELTY2_KEYMOD) in both counter orders, and a line "score <TAB> max_z=.. <TAB> min_p=..
// <TAB> flagged=.. <TAB> valid=.. <TAB> propagated=.. <TAB> repeated=.. <TAB> stopped=.. <TAB> structure" gives the largest z-score of the statistics (the excess of a chi-square
// statistic over its degrees of freedom in standard deviations, or of a number of collisions over its
// mean; not for ClosePairs), the smallest p-value, the number of flagged ones, and arx::valid,
// arx::keys_propagated_by(deadline) (default 12) and arx::repeated of the whole structure. A line
// "structure <TAB> limit" stops the tests at a z-score above the limit (stopped=1; the z-score is then that
// of the tests so far, for Collision and Birthday that of their replications so far).
//
// testu01_stest -P plan [-s seed] [-m streams] [-D deadline] : the score mode of several stages, in one
// process: every line "M[,M...] tests scale keylist [offset]" of the file plan is a stage (as -S M -t tests
// -f scale -l keylist -o offset), and the stages run in their order for every structure; the line
// "score ..." is that of all of them (the limit stopping the later ones). Lines from "#" are comments.
// With -v, the score modes also print a line "key <TAB> M=.. <TAB> swap=.. <TAB> key=keymod@streams#seed
// <TAB> tests=.. <TAB> scale=.. <TAB> max_z=.. <TAB> min_p=.. <TAB> flagged=.." for every key, step count
// and counter order, before the line "score ...".
//
//
// The tests are those of the library testu01_gpu (../testu01_gpu, see its README.md), on the outputs of
// the first M ops of the structure as a counter-based generator: the position of the sequence is the
// counter (or that of the interleaved streams, see -m).
#include <hip/hip_runtime.h>
#include <getopt.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "arx.hpp"
#include "keymod.hpp"
#include "pidfile.hpp"
#include "testu01_gpu.hpp"

namespace {

using testu01_gpu::Summary;

// The key and the counter order, and the streams interleaved: the output at the position p of the
// sequence is that of the stream s = p mod 2^sbits at its index p >> sbits. With cbit >= 0, the
// stream s takes the counters index + (s << cbit) (streams of counters differing in the high bits);
// with kword >= 0, the stream s takes the key with the word kword xored with s (related keys). With
// sbits = 0, the sequence is the single stream of the key.
struct Gen {
    uint32_t k[6];
    int swp;
    int sbits = 0, cbit = -1, kword = -1;
    // the parameters of the generator source (see generator_source)
    testu01_gpu::Params params() const {
        testu01_gpu::Params p{};
        std::copy(k, k + 6, p.w);
        p.w[6] = static_cast<uint32_t>(swp);
        p.w[7] = static_cast<uint32_t>(sbits);
        p.w[8] = static_cast<uint32_t>(cbit);
        p.w[9] = static_cast<uint32_t>(kword);
        return p;
    }
};

// The structure as a generator for testu01_gpu: the ops as straight-line code on the words v[w][l] of the
// lanes l (which is several times faster than interpreting the ops on GPU), the output being the word
// written by the last op; the counter and the key as in melty2, and the streams of Gen.
std::string generator_source(const arx::Arx &a) {
    std::string ops;
    for (const arx::Op &o : a) {
        const std::string d = "v[" + std::to_string(o.d) + "][l]", s = "v[" + std::to_string(o.s) + "][l]";
        std::string e;
        switch (o.t) {
        case arx::kAdd: e = d + " += " + s + ";"; break;
        case arx::kXor: e = d + " ^= " + s + ";"; break;
        case arx::kSub: e = d + " -= " + s + ";"; break;
        default: e = o.r ? d + " = rotl_u(" + d + " ^ " + s + ", " + std::to_string(o.r) + ");" : d + " ^= " + s + ";"; break;
        }
        ops += "#pragma unroll\n    for (int l = 0; l < kLanes; ++l) " + e + "\n";
    }
    return R"(
__device__ inline uint32_t rotl_u(uint32_t x, int r) { return x << r | x >> (32 - r); }

__device__ inline void testu01_gpu_generate(const testu01_gpu::Params &g, uint64_t p0, uint32_t (&out)[testu01_gpu::kLanes]) {
    constexpr int kLanes = testu01_gpu::kLanes;
    const int swp = static_cast<int>(g.w[6]), sbits = static_cast<int>(g.w[7]), cbit = static_cast<int>(g.w[8]), kword = static_cast<int>(g.w[9]);
    uint32_t v[6][kLanes];
#pragma unroll
    for (int l = 0; l < kLanes; ++l) {
        const uint64_t p = p0 + l, s = p & ((UINT64_C(1) << sbits) - 1);
        uint64_t c = p >> sbits;
        if (cbit >= 0) c += s << cbit;
        if (swp) c = c << 32 | c >> 32;
        const uint32_t lo = static_cast<uint32_t>(c), hi = static_cast<uint32_t>(c >> 32);
        uint32_t k[6] = {g.w[0], g.w[1], g.w[2], g.w[3], g.w[4], g.w[5]};
        if (kword >= 0) k[kword] ^= static_cast<uint32_t>(s);
        v[0][l] = k[0] ^ lo; v[1][l] = k[1] + hi; v[2][l] = k[2] ^ hi; v[3][l] = k[3] + lo;
        v[4][l] = k[4]; v[5][l] = k[5];
    }
)" + ops + "#pragma unroll\n    for (int l = 0; l < kLanes; ++l) out[l] = v[" + std::to_string(a.back().d) + "][l];\n}\n";
}

// the kernels of the tests for the structure
std::unique_ptr<testu01_gpu::Generator> compile(const arx::Arx &a) { return std::make_unique<testu01_gpu::Generator>(generator_source(a), "structure"); }

struct Options {
    testu01_gpu::Options t;  // the tests, scale, first counter and limit
    bool quiet = false;      // no line per test (the score mode)
};

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// run the tests of o for the key of g, printing a line per statistic unless quiet
Summary run_tests(const Options &o, testu01_gpu::Battery &battery, const testu01_gpu::Generator &gen, const Gen &g) {
    testu01_gpu::Options t = o.t;
    if (!o.quiet)
        t.report = [](const testu01_gpu::Result &r) {
            std::printf("%s\n", testu01_gpu::format(r).c_str());
            std::fflush(stdout);
        };
    return battery.run(gen, g.params(), t);
}

// the streams of g by the spec "ctr:b:c" or "key:b:w" (see -m), or a single stream for "-"
void set_streams(Gen &g, const std::string &streams) {
    g.sbits = 0;
    g.cbit = g.kword = -1;
    if (streams == "-") return;
    char kind[8];
    int b, x;
    if (std::sscanf(streams.c_str(), "%3[a-z]:%d:%d", kind, &b, &x) != 3 || b < 1 || b > 16 ||
        !((std::strcmp(kind, "ctr") == 0 && x >= 0 && x < 64) || (std::strcmp(kind, "key") == 0 && x >= 0 && x < 6)))
        throw std::invalid_argument("invalid streams (ctr:b:c or key:b:w): " + streams);
    g.sbits = b;
    if (kind[0] == 'c') g.cbit = x;
    else g.kword = x;
}

// a key of the tests: a key spec "keymod[@streams][#seed]" (see -l)
struct KeyRun {
    std::string keymod, streams;
    Gen g;          // the key and the streams (the counter order is set later)
    uint32_t seed;
};

// the key specs of the list (one per line; "-" for none), or that of MELTY2_KEYMOD
std::vector<std::string> read_specs(const char *list) {
    std::vector<std::string> specs;
    if (list) {
        std::ifstream f(list);
        if (!f) throw std::runtime_error(std::string(list) + ": cannot open");
        std::string line;
        while (std::getline(f, line)) {
            line.erase(std::find_if(line.begin(), line.end(), [](char c) { return c == '\r' || c == '\n'; }), line.end());
            if (line.empty() || line[0] == '#') continue;
            if (line.find_first_of(" \t") != std::string::npos)
                throw std::invalid_argument(std::string(list) + ": a line holds one key spec, without spaces: " + line);
            specs.push_back(line);
        }
    } else {
        const char *km = std::getenv("MELTY2_KEYMOD");
        specs.push_back(km && *km ? km : "-");
    }
    return specs;
}

// the keys of the specs: a spec "keymod@streams" takes its own streams instead of those of -m, and a
// spec "...#seed" the key of that seed instead of that of -s
std::vector<KeyRun> make_keys(const std::vector<std::string> &specs, uint32_t seed, const std::string &streams, const Gen &g) {
    std::vector<KeyRun> keys;
    for (const std::string &spec : specs) {
        const size_t hash = spec.find('#');
        const std::string ks = spec.substr(0, hash);
        const size_t at = ks.find('@');
        KeyRun k{ks.substr(0, at), at == std::string::npos ? streams : ks.substr(at + 1), g,
                 hash == std::string::npos ? seed : static_cast<uint32_t>(std::strtoul(spec.c_str() + hash + 1, nullptr, 0))};
        arx::Key key = keymod::seed_key(k.seed);
        if (k.keymod != "-") keymod::apply(key, k.keymod.c_str());
        std::copy(key.begin(), key.end(), k.g.k);
        set_streams(k.g, k.streams);
        keys.push_back(k);
    }
    return keys;
}

// a stage of the score mode: its step counts, tests, scale, offset and keys
struct Stage {
    std::vector<int> ms;
    Options o;
    std::vector<KeyRun> keys;
};

// the step counts "M[,M...]"
std::vector<int> parse_ms(const char *arg) {
    std::vector<int> ms;
    for (const char *q = arg; *q;) {
        char *e;
        ms.push_back(static_cast<int>(std::strtol(q, &e, 10)));
        if (e == q || ms.back() <= 0 || (*e && *e != ',')) throw std::invalid_argument(std::string("invalid steps: ") + arg);
        q = *e ? e + 1 : e;
    }
    return ms;
}

[[noreturn]] void usage() {
    std::fprintf(stderr,
                 "usage: testu01_stest [-s seed] [-w] [-t tests] [-f scale] [-l keylist] [-m streams] [-o offset] [-p pidfile] structure M\n"
                 "       testu01_stest -S M[,M...] [-s seed] [-t tests] [-f scale] [-l keylist] [-m streams] [-o offset] [-D deadline] [-v] [-p pidfile]"
                 " < structures\n"
                 "       testu01_stest -P plan [-s seed] [-m streams] [-D deadline] [-v] [-p pidfile] < structures\n");
    std::exit(1);
}

int run_main(int argc, char **argv) {
    static const option longopts[] = {
        {"seed", required_argument, nullptr, 's'},
        {"swap", no_argument, nullptr, 'w'},
        {"tests", required_argument, nullptr, 't'},
        {"scale", required_argument, nullptr, 'f'},
        {"keylist", required_argument, nullptr, 'l'},
        {"streams", required_argument, nullptr, 'm'},
        {"pidfile", required_argument, nullptr, 'p'},
        {"score", required_argument, nullptr, 'S'},
        {"deadline", required_argument, nullptr, 'D'},
        {"offset", required_argument, nullptr, 'o'},
        {"plan", required_argument, nullptr, 'P'},
        {"verbose", no_argument, nullptr, 'v'},
        {nullptr, 0, nullptr, 0},
    };
    Options o;
    uint32_t seed = 1;
    int swp = 0, opt, deadline = 12;
    bool verbose = false;
    std::vector<int> score_ms;
    const char *pidfile = nullptr, *list = nullptr, *plan = nullptr;
    std::string streams = "-";
    while ((opt = getopt_long(argc, argv, "s:wt:f:l:m:p:S:D:o:P:v", longopts, nullptr)) != -1) {
        switch (opt) {
        case 's': seed = static_cast<uint32_t>(std::strtoul(optarg, nullptr, 0)); break;
        case 'w': swp = 1; break;
        case 't': o.t.tests = optarg; break;
        case 'f': o.t.scale = std::atof(optarg); break;
        case 'l': list = optarg; break;
        case 'm': streams = optarg; break;
        case 'p': pidfile = optarg; break;
        case 'S': score_ms = parse_ms(optarg); break;
        case 'P': plan = optarg; break;
        case 'v': verbose = true; break;
        case 'D': deadline = std::atoi(optarg); break;
        case 'o': o.t.first = std::strtoull(optarg, nullptr, 0); break;
        default: usage();
        }
    }
    const bool score = !score_ms.empty() || plan;
    if (argc - optind != (score ? 0 : 2) || (!score_ms.empty() && plan)) usage();
    if (pidfile) pidfile::write(pidfile);
    // wait for the device by blocking the thread, instead of spinning on a CPU core
    if (hipSetDeviceFlags(hipDeviceScheduleBlockingSync) != hipSuccess) throw std::runtime_error("hipSetDeviceFlags failed");
    Gen g;
    g.swp = swp;

    // the stages of the score mode: those of the plan, or one of the options
    std::vector<Stage> stages;
    if (plan) {
        std::ifstream f(plan);
        if (!f) throw std::runtime_error(std::string(plan) + ": cannot open");
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream in(line);
            std::string ms, tests, scale, keylist, offset = "0";
            if (!(in >> ms) || ms[0] == '#') continue;
            if (!(in >> tests >> scale >> keylist)) throw std::invalid_argument(std::string(plan) + ": M[,M...] tests scale keylist [offset]: " + line);
            in >> offset;
            Stage st{parse_ms(ms.c_str()), o, make_keys(read_specs(keylist.c_str()), seed, streams, g)};
            st.o.t.tests = tests;
            st.o.t.scale = std::atof(scale.c_str());
            st.o.t.first = std::strtoull(offset.c_str(), nullptr, 0);
            stages.push_back(st);
        }
    } else {
        stages.push_back(Stage{score_ms, o, make_keys(read_specs(list), seed, streams, g)});
    }
    // the kernels and the device memory of the tests are kept across the keys (and the stages and
    // structures of the score mode)
    testu01_gpu::Battery battery;
    if (score) {
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty() || line[0] == '#') continue;
            const size_t tab = line.find('\t');
            const double limit = tab == std::string::npos ? HUGE_VAL : std::strtod(line.c_str() + tab + 1, nullptr);
            const arx::Arx full = arx::parse(line.substr(0, tab));
            Summary s;
            std::map<int, std::unique_ptr<testu01_gpu::Generator>> kns;  // the kernels of each step count
            try {
                for (Stage &st : stages) {
                    st.o.quiet = true;
                    st.o.t.limit = limit;
                    for (const int m : st.ms) {
                        if (s.max_z > limit) break;
                        if (m > static_cast<int>(full.size())) throw std::invalid_argument("M must satisfy M <= (ops)");
                        if (!kns.count(m)) {
                            arx::Arx a = full;
                            a.resize(m);
                            kns[m] = compile(a);
                        }
                        for (int w = 0; w < 2; ++w)
                            for (KeyRun k : st.keys) {
                                if (s.max_z > limit) break;
                                k.g.swp = w;
                                const Summary ks = run_tests(st.o, battery, *kns[m], k.g);
                                s.add(ks);
                                if (verbose)
                                    std::printf("key\tM=%d\tswap=%d\tkey=%s@%s#%u\ttests=%s\tscale=%g\tmax_z=%.3f\tmin_p=%.3g\tflagged=%d\n", m, w,
                                                k.keymod.c_str(), k.streams.c_str(), k.seed, st.o.t.tests.c_str(), st.o.t.scale, ks.max_z, ks.min_p, ks.flagged);
                            }
                    }
                }
            } catch (const testu01_gpu::TestFailure &) {  // a test failing completely, e.g. running out of its counters
                s.add(0, HUGE_VAL);
            }
            std::printf("score\tmax_z=%.3f\tmin_p=%.3g\tflagged=%d\tvalid=%d\tpropagated=%d\trepeated=%d\tstopped=%d\t%s\n", s.max_z, s.min_p, s.flagged,
                        arx::valid(full), arx::keys_propagated_by(full, deadline), arx::repeated(full), s.max_z > limit,
                        arx::to_string(full).c_str());
            std::fflush(stdout);
        }
        return 0;
    }
    arx::Arx a = arx::parse(argv[optind]);
    const int M = std::atoi(argv[optind + 1]);
    if (M < 1 || M > static_cast<int>(a.size())) throw std::invalid_argument("M must satisfy 1 <= M <= (ops)");
    a.resize(M);
    const auto t0 = std::chrono::steady_clock::now();
    const auto kn = compile(a);
    std::fprintf(stderr, "compiled the kernels in %.1fs\n", seconds_since(t0));
    for (const KeyRun &k : stages[0].keys) {
        std::printf("# M=%d seed=%u%s keymod=%s scale=%g streams=%s\n", M, k.seed, swp ? " swap" : "", k.keymod.c_str(), o.t.scale, k.streams.c_str());
        const Summary s = run_tests(o, battery, *kn, k.g);
        std::printf("summary\tM=%d\tseed=%u\tswap=%d\tkeymod=%s\tmin_p=%.3g\tmax_p=%.6g\tflagged=%d\tstreams=%s\n", M, k.seed, swp, k.keymod.c_str(),
                    s.min_p, s.max_p, s.flagged, k.streams.c_str());
        std::fflush(stdout);
    }
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        return run_main(argc, argv);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
