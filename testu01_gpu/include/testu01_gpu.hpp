// SPDX-License-Identifier: Apache-2.0 AND LicenseRef-TestU01
//
// testu01_gpu: the tests of TestU01 on a GPU by HIP (ROCm) or CUDA, with the instances of its batteries SmallCrush, Crush and
// BigCrush, for a counter-based generator (a generator computing the output at any position of its sequence). See README.md.
//
// Usage:
//
//   testu01_gpu::Generator gen(source);   // compiles the kernels with the generator, by hiprtc or NVRTC
//   testu01_gpu::Params params{};          // the key, etc., as the source reads them
//   testu01_gpu::Battery battery;          // the device memory, reused across the runs
//   testu01_gpu::Options o;
//   o.report = [](const testu01_gpu::Result &r) { std::cout << testu01_gpu::format(r) << std::endl; };
//   const testu01_gpu::Summary s = battery.run(gen, params, o);
//
// The source of the generator defines, in the global namespace,
//
//   __device__ inline void testu01_gpu_generate(const testu01_gpu::Params &p, uint64_t pos,
//                                               uint32_t (&out)[testu01_gpu::kLanes])
//
// writing the 32-bit outputs at the positions pos, pos + 1, ..., pos + kLanes - 1 of the sequence. It is
// compiled after the device code of the library, so it is inlined into the kernels; hiprtc and NVRTC
// have no standard headers, but uint32_t, uint64_t, UINT32_C and UINT64_C are available.
#ifndef TESTU01_GPU_HPP_
#define TESTU01_GPU_HPP_

#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#include "testu01_gpu_params.hpp"

namespace testu01_gpu {

// the names of all the tests, comma-separated, in the order they run
extern const char kAllTests[];

// A generator: the kernels of the tests compiled with its source by hiprtc (or NVRTC) for the current device,
// and loaded as a module. Compiling takes a few seconds; the kernels are reused across runs.
class Generator {
  public:
    // the source (see above); name is that of the program in the messages of hiprtc or NVRTC
    explicit Generator(const std::string &source, const std::string &name = "generator");
    ~Generator();
    Generator(const Generator &) = delete;
    Generator &operator=(const Generator &) = delete;
    struct Impl;
    const Impl &impl() const { return *impl_; }

  private:
    std::unique_ptr<Impl> impl_;
};

// the kinds of statistics, and so of the lines of format()
enum class Stat {
    kChiSquare,        // stat: chi-square, df: its degrees of freedom
    kPoisson,          // stat: the number of collisions, mean: its mean
    kNormal,           // stat: a standard normal z
    kAndersonDarling,  // stat: A^2
    kClosePairs,       // stat: a statistic of ClosePairs (NP, mNP, ... in params)
    kDiscrete,         // a p-value only (the longest run of LongestHeadRun)
};

// a statistic of a test
struct Result {
    std::string test;    // the short name (e.g. "poker", "collover")
    std::string params;  // the parameters (e.g. "n=400000000 r=0 d=8 k=8")
    Stat kind;
    double stat = 0, mean = 0;
    long df = 0;
    double p = 0;        // the p-value (of the right tail, as TestU01)
    double z = 0;        // its z-score for the summary (0 if none)
    double seconds = 0;  // the time of the test
};

// the line of a result, as TestU01 prints it in spirit: a p-value outside [0.001, 0.999] is marked
// with *****
std::string format(const Result &r);

// the smallest and the largest p-values, the largest z-score, and the number of p-values outside
// [0.001, 0.999]
struct Summary {
    double min_p = 1, max_p = 0, max_z = 0;
    int flagged = 0;
    void add(double p, double z = 0) {
        min_p = p < min_p ? p : min_p;
        max_p = p > max_p ? p : max_p;
        max_z = z > max_z ? z : max_z;
        flagged += p < 0.001 || p > 0.999;
    }
    void add(const Summary &o) {
        min_p = o.min_p < min_p ? o.min_p : min_p;
        max_p = o.max_p > max_p ? o.max_p : max_p;
        max_z = o.max_z > max_z ? o.max_z : max_z;
        flagged += o.flagged;
    }
};

// the batteries of TestU01 whose instances of the tests (their parameters) run
enum class Suite { kSmallCrush, kCrush, kBigCrush };

struct Options {
    Suite suite = Suite::kBigCrush;  // the battery
    std::string tests = kAllTests;  // the tests to run (comma-separated names of kAllTests; those of the battery run)
    double scale = 1;               // multiplies the sample sizes n of the battery
    uint64_t first = 0;             // the first position of the sequence used by the tests
    double limit = HUGE_VAL;        // stop the tests once a z-score is above it (for searches)
    size_t memory = size_t(2) << 30;  // the device memory of the large tests (bytes): beyond it, they make their
                                      // points again by parts (the results are the same)
    bool fast = false;  // faster computations whose results differ from those of TestU01 by the order of the
                        // floating-point sums: the Anderson-Darling statistics of SampleProd and SampleMean summed
                        // on GPU (by a parallel reduction) instead of in order on the host
    std::function<void(const Result &)> report;  // called for every statistic (if set)
};

// A test failing completely, e.g. a generator so bad that the observations of a test take far more
// outputs than expected; the errors of HIP or CUDA are std::runtime_error.
struct TestFailure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// The device memory of the tests, allocated when a test needs it and kept across runs.
class Battery {
  public:
    Battery();
    ~Battery();
    Battery(const Battery &) = delete;
    Battery &operator=(const Battery &) = delete;
    // Run the tests on the sequence of the generator and the parameters. Every test (and replication)
    // takes the next range of consecutive positions from o.first, as TestU01 takes the outputs of a single
    // stream.
    Summary run(const Generator &gen, const Params &params, const Options &o);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace testu01_gpu

#endif
