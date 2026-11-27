// SPDX-License-Identifier: Apache-2.0 AND LicenseRef-TestU01
//
// testu01_gpu: the tests of BigCrush, implemented on GPU (LinearComp and LempelZiv on the host) (see README.md for the tests and their differences from TestU01).
//
// The tests are those of BigCrush, reimplemented after the source code of TestU01 1.2.3 (so this file
// is also subject to the license of TestU01; see NOTICE) with the same parameters as their instances in
// BigCrush: sknuth_SimpPoker, sknuth_CouponCollector, sknuth_Gap, sknuth_MaxOft, smarsa_CollisionOver,
// smarsa_BirthdaySpacings, snpair_ClosePairs, sknuth_Permutation, sknuth_Run, smarsa_SerialOver,
// sknuth_CollisionPermut, svaria_WeightDistrib, smarsa_Savir2, smarsa_GCD, sstring_HammingWeight2,
// sstring_HammingCorr, sstring_HammingIndep, sstring_Run, sstring_LongestHeadRun, sstring_AutoCor,
// svaria_SampleCorr, svaria_SampleProd, svaria_SampleMean, swalk_RandomWalk1, sstring_PeriodsInStrings and
// svaria_SumCollector, sspectral_Fourier3, scomp_LinearComp, smarsa_MatrixRank, svaria_AppearanceSpacings and scomp_LempelZiv (in the order they were added, so that the positions of the earlier ones stay as
// before). SerialOver, Run, the runs of bits, AutoCor and SampleCorr see the single sequence exactly
// (every thread from a few values before its share); the others make their observations from the
// consecutive positions of every thread (the tests of blocks: an observation does not span threads).
// Where TestU01 takes a statistic of a wrong variance, the right one is taken: the number of the runs of
// bits (sstring_Run, B / 16 for B bits) and SampleCorr (13/144 per term); the runs of bits are counted in a
// fixed number of values instead of until n runs. The other differences: the chi-squares merge the
// classes below 10 expected from the low end (gofs_MergeClasses of TestU01 from both); HammingWeight2
// counts the weights within 12 standard deviations; LongestHeadRun starts every block afresh. Every test
// (and replication) takes the next range of consecutive positions from the first one, i.e. a single
// stream as in TestU01, and every thread runs the test on its own consecutive piece of it; the counts of
// the categories are accumulated in histograms. (Pieces at positions far apart, e.g. differing only in
// the bits 28 and above, would compare outputs of related counters, which a single stream of TestU01
// doesn't, and the low bits of weak counter-based generators fail such comparisons.) The collisions of
// CollisionOver (overlapping t-tuples) and of the spacings of BirthdaySpacings (non-overlapping t-tuples,
// sorted on GPU by rocThrust or Thrust) are counted exactly by a hash table on GPU (in the cache of the GPU, in
// passes over the points), and compared with a Poisson distribution
// (fbar_Poisson1), as TestU01 does for these parameters. Beyond the memory budget, the points are made again
// by parts (see count_equal, sorted_parts and close_pairs), with the same results.
// ClosePairs finds the close pairs of the points in the torus (the maximum norm) in a grid of cells on
// GPU (see close_pairs), and computes the six statistics of TestU01 on the host (NP, mNP, mNP1, mNP2,
// NJumps and mNP2S). The streams differ from those of TestU01, so the p-values agree only statistically.
// As in TestU01, a value in [0, d) is floor(d * u) of u = (x << r) / 2^32 for the output x, i.e. the r
// most significant bits are dropped. The categories with an expected count below 10 are merged with their
// neighbours, and the p-value is that of the chi-square statistic (fbar_ChiSquare2 of TestU01's
// probdist); for MaxOft, that of the sum over the N replications. Fourier3 transforms the sequences of bits
// by rocFFT or cuFFT (in double, so that its statistics agree with those of TestU01 on the same outputs).
//
// The backend is HIP (ROCm: hiprtc, rocFFT, rocThrust), or CUDA if TESTU01_GPU_CUDA is defined (NVRTC, cuFFT,
// Thrust); see gpu.hpp.
#include "gpu.hpp"

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <format>
#include <map>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"

extern "C" {
#include <fbar.h>
#include <fdist.h>
#include <finv.h>
#include <fmass.h>
#include <gofs.h>
#include <gofw.h>
}

#include "device.hpp"
#include "testu01_gpu.hpp"

namespace testu01_gpu {

const char kAllTests[] = "poker,coupon,gap,maxoft,collision,kcollision,birthday,closepairs,cpbitmatch,permutation,run,serial,collpermut,weight,savir2,gcd,hamweight,hamcorr,"
                         "hamindep,bitrun,longhead,autocor,samplecorr,sampleprod,samplemean,walk,periods,sumcoll,fourier,linearcomp,"
                         "matrixrank,appearance,lempelziv";

namespace {

using detail::Coupon;
using detail::CpGrid;
using detail::Gap;
using detail::HashTable;
using detail::Sink;
using detail::kThreads;
using detail::MaxOft;
using detail::Permut;
using detail::Poker;
using detail::Tuples;

constexpr int kBlocks = 1024;       // blocks per launch
constexpr int kMaxLdsBins = 12288;  // histograms up to this size are accumulated in LDS first
constexpr double kMinExpected = 10;

// the sources of device.hpp and testu01_gpu_params.hpp, embedded by the build as raw string literals
constexpr char kDeviceSource[] =
#include "device.inc"
    ;
constexpr char kParamsSource[] =
#include "params.inc"
    ;

#ifdef TESTU01_GPU_CUDA
constexpr char kBackend[] = "CUDA";
#else
constexpr char kBackend[] = "HIP";
#endif

void check(gpuError_t e, const char *what, int line) {
    if (e != gpuSuccess) throw std::runtime_error(absl::StrCat(kBackend, " ", gpuGetErrorString(e), " at ", __FILE__, ":", line, " (", what, ")"));
}
#ifdef TESTU01_GPU_CUDA
// the errors of the driver API (the modules and the launches)
void check(CUresult e, const char *what, int line) {
    if (e == CUDA_SUCCESS) return;
    const char *s = nullptr;
    if (cuGetErrorString(e, &s) != CUDA_SUCCESS || !s) s = "unknown error";
    throw std::runtime_error(absl::StrCat("CUDA ", s, " at ", __FILE__, ":", line, " (", what, ")"));
}
#endif
#define CK(x) check((x), #x, __LINE__)

#ifdef TESTU01_GPU_CUDA
// NVRTC: the program of the source (with the header testu01_gpu_params.hpp) compiled for the current device, as
// a CUBIN for its architecture if NVRTC supports it, else as the PTX of the newest architecture below it (which
// the driver compiles when loading it)
std::vector<char> compile_program(const std::string &src) {
    const auto rtc = [](nvrtcResult r, const char *what) {
        if (r != NVRTC_SUCCESS) throw std::runtime_error(absl::StrCat(what, ": ", nvrtcGetErrorString(r)));
    };
    int dev, major, minor;
    CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
    CK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));
    int narch = 0;
    rtc(nvrtcGetNumSupportedArchs(&narch), "nvrtcGetNumSupportedArchs");
    std::vector<int> archs(narch);
    rtc(nvrtcGetSupportedArchs(archs.data()), "nvrtcGetSupportedArchs");
    const int want = 10 * major + minor;
    int use = 0;
    for (const int a : archs)
        if (a <= want && a > use) use = a;
    if (!use) throw std::runtime_error(absl::StrCat("NVRTC supports no architecture up to the compute capability ", major, ".", minor));
    const bool cubin = use == want;
    const std::string arch = absl::StrCat("--gpu-architecture=", cubin ? "sm_" : "compute_", use);
    const char *opts[] = {arch.c_str(), "--std=c++17"};
    const char *headers[] = {kParamsSource}, *include_names[] = {"testu01_gpu_params.hpp"};
    nvrtcProgram prog;
    rtc(nvrtcCreateProgram(&prog, src.c_str(), "device.hpp", 1, headers, include_names), "nvrtcCreateProgram");
    const nvrtcResult r = nvrtcCompileProgram(prog, 2, opts);
    if (r != NVRTC_SUCCESS) {
        size_t n = 0;
        nvrtcGetProgramLogSize(prog, &n);
        std::string log(n, '\0');
        nvrtcGetProgramLog(prog, log.data());
        nvrtcDestroyProgram(&prog);
        throw std::runtime_error(absl::StrCat("nvrtc: ", nvrtcGetErrorString(r), "\n", log));
    }
    size_t size = 0;
    std::vector<char> code;
    if (cubin) {
        rtc(nvrtcGetCUBINSize(prog, &size), "nvrtcGetCUBINSize");
        code.resize(size);
        rtc(nvrtcGetCUBIN(prog, code.data()), "nvrtcGetCUBIN");
    } else {
        rtc(nvrtcGetPTXSize(prog, &size), "nvrtcGetPTXSize");
        code.resize(size);
        rtc(nvrtcGetPTX(prog, code.data()), "nvrtcGetPTX");
    }
    nvrtcDestroyProgram(&prog);
    return code;
}
#else
// hiprtc: the program of the source (with the header testu01_gpu_params.hpp) compiled for the current device
std::vector<char> compile_program(const std::string &src) {
    const auto rtc = [](hiprtcResult r, const char *what) {
        if (r != HIPRTC_SUCCESS) throw std::runtime_error(absl::StrCat(what, ": ", hiprtcGetErrorString(r)));
    };
    hipDevice_t dev;
    CK(hipGetDevice(&dev));
    hipDeviceProp_t prop;
    CK(hipGetDeviceProperties(&prop, dev));
    const std::string arch = absl::StrCat("--offload-arch=", prop.gcnArchName);
    const char *opts[] = {arch.c_str(), "-O3", "-std=c++17"};
    const char *headers[] = {kParamsSource}, *include_names[] = {"testu01_gpu_params.hpp"};
    hiprtcProgram prog;
    rtc(hiprtcCreateProgram(&prog, src.c_str(), "device.hpp", 1, headers, include_names), "hiprtcCreateProgram");
    const hiprtcResult r = hiprtcCompileProgram(prog, 3, opts);
    if (r != HIPRTC_SUCCESS) {
        size_t n = 0;
        hiprtcGetProgramLogSize(prog, &n);
        std::string log(n, '\0');
        hiprtcGetProgramLog(prog, log.data());
        hiprtcDestroyProgram(&prog);
        throw std::runtime_error(absl::StrCat("hiprtc: ", hiprtcGetErrorString(r), "\n", log));
    }
    size_t size = 0;
    rtc(hiprtcGetCodeSize(prog, &size), "hiprtcGetCodeSize");
    std::vector<char> code(size);
    rtc(hiprtcGetCode(prog, code.data()), "hiprtcGetCode");
    hiprtcDestroyProgram(&prog);
    return code;
}
#endif

}  // namespace

// The kernels of device.hpp for a generator, compiled at run time by hiprtc (or NVRTC) with the source of the
// generator after the device code, so that the generator is inlined as straight-line code (which is
// several times faster than interpreting a generator on GPU), and loaded as a module.
struct Generator::Impl {
    Impl(const std::string &source, const std::string &name) {
        const std::string src = absl::StrCat(kDeviceSource, "\n#line 1 \"", name, "\"\n", source);
        const std::vector<char> code = compile_program(src);
#ifdef TESTU01_GPU_CUDA
        CK(cudaFree(nullptr));  // the primary context of the device, current for the driver API
        CK(cuModuleLoadData(&module_, code.data()));
#else
        CK(hipModuleLoadData(&module_, code.data()));
#endif
        for (auto [f, fname] : {std::pair{&poker, "ktest_poker"}, {&coupon, "ktest_coupon"}, {&gap, "ktest_gap"}, {&maxoft, "ktest_maxoft"}, {&permut, "ktest_permut"},
                                {&points, "kpoints"}, {&spacings, "kspacings"}, {&insert, "kinsert"}, {&fourier_bits, "kfourier_bits"}, {&fourier_acc, "kfourier_acc"},
                                {&cp_outputs, "kcp_outputs"}, {&cp_cells, "kcp_cells"}, {&cp_scatter, "kcp_scatter"}, {&cp_pairs, "kcp_pairs"}, {&krun, "krun"},
                                {&weight, "ktest_weight"}, {&savir2, "ktest_savir2"}, {&gcd, "ktest_gcd"}, {&serial, "kserial"}, {&perm_points, "kperm_points"}, {&cp_slab, "kcp_slab"},
                                {&hamming, "ktest_hamming"}, {&autocor, "kautocor"}, {&samplecorr, "ksamplecorr"}, {&bitrun, "kbitrun"}, {&longhead, "klonghead"},
                                {&uniform, "kuniform"}, {&strip, "ktest_strip"}, {&periods, "ktest_periods"}, {&walk, "kwalk"}, {&sumcoll, "ktest_sumcoll"}, {&matrank, "ktest_matrank"}, {&mat_fill, "kmat_fill"},
                                {&mat_rank, "kmat_rank"}, {&appear, "kappear"}})
#ifdef TESTU01_GPU_CUDA
            CK(cuModuleGetFunction(f, module_, fname));
#else
            CK(hipModuleGetFunction(f, module_, fname));
#endif
    }
#ifdef TESTU01_GPU_CUDA
    ~Impl() { (void)cuModuleUnload(module_); }
#else
    ~Impl() { (void)hipModuleUnload(module_); }
#endif
    Impl(const Impl &) = delete;
    Impl &operator=(const Impl &) = delete;

    // the kernel of a test
    gpuFunction_t of(const Poker &) const { return poker; }
    gpuFunction_t of(const Coupon &) const { return coupon; }
    gpuFunction_t of(const Gap &) const { return gap; }
    gpuFunction_t of(const MaxOft &) const { return maxoft; }
    gpuFunction_t of(const Permut &) const { return permut; }
    gpuFunction_t of(const detail::Weight &) const { return weight; }
    gpuFunction_t of(const detail::Savir2 &) const { return savir2; }
    gpuFunction_t of(const detail::Gcd &) const { return gcd; }
    gpuFunction_t of(const detail::Hamming &) const { return hamming; }
    gpuFunction_t of(const detail::Strip &) const { return strip; }
    gpuFunction_t of(const detail::Periods &) const { return periods; }
    gpuFunction_t of(const detail::SumColl &) const { return sumcoll; }
    gpuFunction_t of(const detail::MatRank &) const { return matrank; }

    gpuFunction_t poker, coupon, gap, maxoft, permut, points, spacings, insert, fourier_bits, fourier_acc, cp_outputs, cp_cells,
        cp_scatter, cp_pairs, cp_slab, krun, weight, savir2, gcd, serial, perm_points, hamming, autocor, samplecorr, bitrun,
        longhead, uniform, strip, periods, walk, sumcoll, matrank, mat_fill, mat_rank, appear;

  private:
    gpuModule_t module_ = nullptr;
};

Generator::Generator(const std::string &source, const std::string &name) : impl_(std::make_unique<Impl>(source, name)) {}
Generator::~Generator() = default;

namespace {

using Kernels = Generator::Impl;

// launch the kernel f on grid blocks of block threads with shmem bytes of dynamic LDS (shared memory)
template <class... A>
void launch(gpuFunction_t f, unsigned grid, unsigned block, unsigned shmem, A... a) {
    void *args[] = {static_cast<void *>(&a)...};
#ifdef TESTU01_GPU_CUDA
    CK(cuLaunchKernel(f, grid, 1, 1, block, 1, 1, shmem, nullptr, args, nullptr));
#else
    CK(hipModuleLaunchKernel(f, grid, 1, 1, block, 1, 1, shmem, nullptr, args, nullptr));
#endif
}

// The device memory of the tests on keys (the points of CollisionOver, BirthdaySpacings, CollisionPermut,
// SampleProd and SampleMean): two arrays of cap keys and a counter.
struct KeyBuffers {
    unsigned long long *a = nullptr, *b = nullptr, *count = nullptr;
    long long cap = 0;
    void ensure(long long n) {
        if (n <= cap) return;
        release();
        CK(gpuMalloc(reinterpret_cast<void **>(&a), sizeof(unsigned long long) * n));
        CK(gpuMalloc(reinterpret_cast<void **>(&b), sizeof(unsigned long long) * n));
        CK(gpuMalloc(reinterpret_cast<void **>(&count), sizeof(unsigned long long)));
        cap = n;
    }
    void release() {
        for (void *p : {static_cast<void *>(a), static_cast<void *>(b), static_cast<void *>(count)}) (void)gpuFree(p);
        a = b = count = nullptr;
        cap = 0;
    }
    KeyBuffers() = default;
    KeyBuffers(const KeyBuffers &) = delete;
    KeyBuffers &operator=(const KeyBuffers &) = delete;
    ~KeyBuffers() { release(); }
    long long read_count() const {
        unsigned long long c;
        CK(gpuMemcpy(&c, count, sizeof c, gpuMemcpyDeviceToHost));
        return static_cast<long long>(c);
    }
};

// The counters of the launches: every launch takes the next range of nthreads * stride consecutive
// counters from 0, and its threads take consecutive pieces of stride counters of it. So a test (or a
// replication) uses a single stream as in TestU01, and the counters stay far below 2^64 (about 2^40
// for all the tests).
class Counters {
  public:
    static constexpr uint64_t kThreadsPerLaunch = static_cast<uint64_t>(kBlocks) * kThreads;
    // The stride covering the values of the observations of a thread: the largest share of n, of
    // values of mean and variance var per observation, with a margin of 12 standard deviations (and
    // for the overlap of the first observation), rounded up to the generation by kLanes. For a share of
    // fewer than 100 observations, whose sum is far from normal for heavy tails (e.g. the geometric
    // lengths of Gap), 40 standard deviations of an observation more; the stride costs nothing beyond the
    // values used, so the margin is generous.
    static uint64_t stride(long long n, double mean, double var, int extra = 0) {
        const double q = static_cast<double>((n + kThreadsPerLaunch - 1) / kThreadsPerLaunch);
        const double need = q * mean + 12 * std::sqrt(q * var) + (q < 100 ? 40 * std::sqrt(var) : 0) + extra;
        return (static_cast<uint64_t>(std::ceil(need)) + kLanes - 1) / kLanes * kLanes;
    }
    explicit Counters(uint64_t first = 0) : next_(first) {}
    // the base of the next launch with the stride
    uint64_t take(uint64_t stride) {
        const uint64_t b = next_;
        next_ += stride * kThreadsPerLaunch;
        return b;
    }

  private:
    uint64_t next_;
};

// The real FFTs of Fourier3 by rocFFT or cuFFT (in double): batch sequences of n doubles in a, transformed to
// n / 2 + 1 complex coefficients each in f (unnormalized, as the FFT of TestU01), with the sums of their squares
// in b.
#ifdef TESTU01_GPU_CUDA
struct Fft {
    int n;
    long long batch;
    double *a = nullptr, *f = nullptr;
    double *b = nullptr;
    cufftHandle plan = 0;
    bool have_plan = false;
    Fft(int n_, long long batch_) : n(n_), batch(batch_) {
        CK(gpuMalloc(reinterpret_cast<void **>(&a), sizeof(double) * n * batch));
        CK(gpuMalloc(reinterpret_cast<void **>(&f), sizeof(double) * 2 * (n / 2 + 1) * batch));
        CK(gpuMalloc(reinterpret_cast<void **>(&b), sizeof(double) * (n / 4 + 1)));
        // the sequences and the coefficients contiguous, as the default layout of rocFFT
        int len[1] = {n};
        cf(cufftPlanMany(&plan, 1, len, nullptr, 1, n, nullptr, 1, n / 2 + 1, CUFFT_D2Z, static_cast<int>(batch)), "cufftPlanMany");
        have_plan = true;
    }
    ~Fft() {
        if (have_plan) cufftDestroy(plan);
        for (void *p : {static_cast<void *>(a), static_cast<void *>(f), static_cast<void *>(b)}) (void)gpuFree(p);
    }
    Fft(const Fft &) = delete;
    Fft &operator=(const Fft &) = delete;
    // transform the batch (on the null stream, after the kernels filling a)
    void execute() { cf(cufftExecD2Z(plan, a, reinterpret_cast<cufftDoubleComplex *>(f)), "cufftExecD2Z"); }
    static void cf(cufftResult st, const char *what) {
        if (st != CUFFT_SUCCESS) throw std::runtime_error(absl::StrCat(what, ": cufft status ", static_cast<int>(st)));
    }
};
#else
struct Fft {
    int n;
    long long batch;
    double *a = nullptr, *f = nullptr;
    double *b = nullptr;
    void *work = nullptr;
    rocfft_plan plan = nullptr;
    rocfft_execution_info info = nullptr;
    Fft(int n_, long long batch_) : n(n_), batch(batch_) {
        static std::once_flag setup;
        std::call_once(setup, [] { rf(rocfft_setup(), "rocfft_setup"); });
        CK(gpuMalloc(reinterpret_cast<void **>(&a), sizeof(double) * n * batch));
        CK(gpuMalloc(reinterpret_cast<void **>(&f), sizeof(double) * 2 * (n / 2 + 1) * batch));
        CK(gpuMalloc(reinterpret_cast<void **>(&b), sizeof(double) * (n / 4 + 1)));
        const size_t len[1] = {static_cast<size_t>(n)};
        rf(rocfft_plan_create(&plan, rocfft_placement_notinplace, rocfft_transform_type_real_forward, rocfft_precision_double, 1, len, batch, nullptr),
           "rocfft_plan_create");
        rf(rocfft_execution_info_create(&info), "rocfft_execution_info_create");
        size_t ws = 0;
        rf(rocfft_plan_get_work_buffer_size(plan, &ws), "rocfft_plan_get_work_buffer_size");
        if (ws) {
            CK(gpuMalloc(&work, ws));
            rf(rocfft_execution_info_set_work_buffer(info, work, ws), "rocfft_execution_info_set_work_buffer");
        }
    }
    ~Fft() {
        if (info) rocfft_execution_info_destroy(info);
        if (plan) rocfft_plan_destroy(plan);
        for (void *p : {static_cast<void *>(a), static_cast<void *>(f), static_cast<void *>(b), work}) (void)gpuFree(p);
    }
    Fft(const Fft &) = delete;
    Fft &operator=(const Fft &) = delete;
    // transform the batch (on the null stream, after the kernels filling a)
    void execute() {
        void *in[1] = {a}, *out[1] = {f};
        rf(rocfft_execute(plan, in, out, info), "rocfft_execute");
    }
    static void rf(rocfft_status st, const char *what) {
        if (st != rocfft_status_success) throw std::runtime_error(absl::StrCat(what, ": rocfft status ", static_cast<int>(st)));
    }
};
#endif

// a bijection of the 64-bit keys (SplitMix64's finalizer, as mix64 of the device code)
struct Mix64 {
    __host__ __device__ uint64_t operator()(uint64_t z) const {
        z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
        return z ^ (z >> 31);
    }
};

void sort_keys(unsigned long long *a, long long n) {
    const thrust::device_ptr<unsigned long long> p(a);
    thrust::sort(p, p + n);
}

// The bits of the parts of n keys when they don't fit in cap (2^pb parts, of about n / 2^pb keys, with
// room for their fluctuations).
int part_bits(long long n, long long cap) {
    int pb = 1;
    for (;; ++pb) {
        const double e = std::ldexp(static_cast<double>(n), -pb);
        if (e * 1.05 + 8 * std::sqrt(e) + 1000 <= static_cast<double>(cap)) return pb;
    }
}

// The terms of the Anderson-Darling statistic of TestU01 (gofs_AndersonDarling) of the sorted keys k[i] of
// the ranks off + i + 1 among N, as u = (k + 1/2) / 2^64 (for the fast mode, which sums them on GPU).
struct AdTerm {
    const unsigned long long *k;
    long long off, N;
    __host__ __device__ double operator()(long long i) const {
        const double eps = DBL_EPSILON / 2;  // gofs_EpsilonAD
        double u = (static_cast<double>(k[i]) + 0.5) * 0x1p-64, u1 = u;
        if (u <= eps) u1 = u = eps;
        else if (u >= 1 - eps) u1 = 1 - eps;
        const long long j = off + i + 1;
        return (2.0 * j - 1) * log(u) + (1.0 + 2.0 * (N - j)) * log1p(-u1);
    }
};

// a launch of a kernel putting its keys into the sink
using Launch = std::function<void(const Sink &)>;

// The device memory of the tests within the budget (bytes): the keys, ClosePairs and the FFTs.
struct Mem;

long long count_equal_part(const Kernels &kn, Mem &m, const Launch &f, int pb, uint64_t part);

struct CpBuffers;

// The device memory of a hash table of up to kTableSlots slots (64 MB), which fits in the cache of the GPU
// (the 128 MB Infinity Cache of RX 6800); more keys are inserted in several passes (see HashTable), as the
// random accesses to a table in the memory are slow.
struct HashBuffers {
    static constexpr uint64_t kTableSlots = UINT64_C(1) << 23;
    static constexpr double kMaxLoad = 0.5;  // the largest number of keys per slot in a pass
    unsigned long long *slots = nullptr, *counters = nullptr;
    uint64_t size = 0;  // the slots (a power of 2)
    HashBuffers() = default;
    HashBuffers(const HashBuffers &) = delete;
    HashBuffers &operator=(const HashBuffers &) = delete;
    ~HashBuffers() {
        (void)gpuFree(slots);
        (void)gpuFree(counters);
    }
    // the slots for n keys (a table of a pass for at most kMaxLoad keys per slot), and the bits of the passes
    int prepare(long long n) {
        uint64_t want = 1;
        while (static_cast<double>(want) * kMaxLoad < n && want < kTableSlots) want <<= 1;
        if (want > size) {
            (void)gpuFree(slots);
            CK(gpuMalloc(reinterpret_cast<void **>(&slots), sizeof(unsigned long long) * want));
            if (!counters) CK(gpuMalloc(reinterpret_cast<void **>(&counters), sizeof(unsigned long long) * 2));
            size = want;
        }
        int pb = 0;
        while (static_cast<double>(n >> pb) > kMaxLoad * size) ++pb;
        return pb;
    }
    // the empty table of the pass part of 2^pbits; the collisions are counted from the first pass
    HashTable table(int pbits, uint64_t part) {
        CK(gpuMemset(slots, 0xff, sizeof(unsigned long long) * size));
        if (part == 0) CK(gpuMemset(counters, 0, sizeof(unsigned long long) * 2));
        return {slots, counters, counters + 1, size - 1, pbits, part};
    }
    // the collisions in the table over the passes: the counted ones and those of the key kEmpty
    long long collisions() {
        unsigned long long c[2];
        CK(gpuMemcpy(c, counters, sizeof c, gpuMemcpyDeviceToHost));
        return static_cast<long long>(c[0] + (c[1] > 1 ? c[1] - 1 : 0));
    }
};

struct Mem {
    size_t budget = size_t(2) << 30;
    KeyBuffers keys;
    HashBuffers hash;
    std::unique_ptr<CpBuffers> cp;
    std::unique_ptr<Fft> fft;
    // the keys of a pass: two arrays and the temporary memory of the sort
    long long key_cap() const { return std::max<long long>(static_cast<long long>(budget / 24), 1 << 20); }
};

// The number of the keys a[0..n) equal to another one before them (the collisions): by the hash table, in
// passes over the keys (faster than sorting them).
long long count_equal_keys(const Kernels &kn, Mem &m, const unsigned long long *a, long long n) {
    if (n < 2) return 0;
    const int pb = m.hash.prepare(n);
    for (uint64_t part = 0; part < UINT64_C(1) << pb; ++part) launch(kn.insert, kBlocks, kThreads, 0, a, n, m.hash.table(pb, part));
    return m.hash.collisions();
}

// The number of the keys equal to an earlier one among the n keys of f (the collisions), by the hash table:
// the keys stored once if they fit in the memory (for keys costly to make, as those of CollisionPermut),
// else by the parts of their hashes (equal keys have equal hashes), the keys being made again for every
// part. Either way the count is the same.
long long count_equal(const Kernels &kn, Mem &m, long long n, const Launch &f) {
    const long long cap = m.key_cap();
    if (n <= cap) {
        m.keys.ensure(n);
        f(Sink{m.keys.a, m.keys.count, n, 0, 64, 0, 0});
        return count_equal_keys(kn, m, m.keys.a, n);
    }
    m.keys.ensure(cap);
    const int pb = part_bits(n, cap);
    long long c = 0;
    for (uint64_t part = 0; part < UINT64_C(1) << pb; ++part) c += count_equal_part(kn, m, f, pb, part);
    return c;
}

// (a part too large, of many equal keys, is split further)
long long count_equal_part(const Kernels &kn, Mem &m, const Launch &f, int pb, uint64_t part) {
    CK(gpuMemset(m.keys.count, 0, sizeof(unsigned long long)));
    f(Sink{m.keys.a, m.keys.count, m.keys.cap, pb, 64, 1, part});
    const long long c = m.keys.read_count();
    if (c <= m.keys.cap) return count_equal_keys(kn, m, m.keys.a, c);
    if (pb >= 40) throw TestFailure("too many equal keys for the memory budget");
    return count_equal_part(kn, m, f, pb + 1, 2 * part) + count_equal_part(kn, m, f, pb + 1, 2 * part + 1);
}

using PartFn = std::function<void(unsigned long long *, long long)>;

void sorted_part(const Kernels &kn, Mem &m, int kbits, const Launch &f, const PartFn &part_fn, int pb, uint64_t part) {
    CK(gpuMemset(m.keys.count, 0, sizeof(unsigned long long)));
    f(Sink{m.keys.a, m.keys.count, m.keys.cap, pb, kbits, 0, part});
    const long long c = m.keys.read_count();
    if (c <= m.keys.cap) {
        sort_keys(m.keys.a, c);
        part_fn(m.keys.a, c);
        return;
    }
    if (pb >= kbits) throw TestFailure("too many equal points for the memory budget");
    sorted_part(kn, m, kbits, f, part_fn, pb + 1, 2 * part);
    sorted_part(kn, m, kbits, f, part_fn, pb + 1, 2 * part + 1);
}

// The n keys of kbits bits of f, sorted, in increasing parts part_fn(keys, count) (the keys in m.keys.a):
// all at once if they fit in the memory, else by the parts of their top bits, the keys being made again
// for every part.
void sorted_parts(const Kernels &kn, Mem &m, long long n, int kbits, const Launch &f, const PartFn &part_fn) {
    const long long cap = m.key_cap();
    if (n <= cap) {
        m.keys.ensure(n);
        f(Sink{m.keys.a, m.keys.count, n, 0, kbits, 0, 0});
        sort_keys(m.keys.a, n);
        part_fn(m.keys.a, n);
        return;
    }
    m.keys.ensure(cap);
    const int pb = std::min(part_bits(n, cap), kbits);
    for (uint64_t part = 0; part < UINT64_C(1) << pb; ++part) sorted_part(kn, m, kbits, f, part_fn, pb, part);
}

// CollisionOver (and sknuth_Collision, with non-overlapping points): the number of collisions of n overlapping
// (or non-overlapping) points in 2^(b * t) cells (the points are made again for every pass of the hash table, which is
// cheaper than storing them; no memory grows with n)
long long collision_over(const Kernels &kn, Mem &m, const Tuples &tup, const Params &g, Counters &ctr, long long n) {
    const uint64_t stride = tup.overlap ? Counters::stride(n, 1, 0, tup.t - 1) : Counters::stride(n, tup.t, 0), base = ctr.take(stride);
    const int pb = m.hash.prepare(n);
    for (uint64_t part = 0; part < UINT64_C(1) << pb; ++part) {
        Sink sk{};
        sk.h = m.hash.table(pb, part);
        launch(kn.points, kBlocks, kThreads, 0, tup, g, base, stride, n, sk);
    }
    return m.hash.collisions();
}

// BirthdaySpacings: the number of collisions among the spacings of n sorted points of b * t bits. If the
// points don't fit in the memory, they are sorted by the parts of their top bits (made again for every
// part), and their spacings (hashed) go to buckets on the host by the top bits of their hashes, whose
// collisions are counted one bucket at a time.
long long birthday_spacings(const Kernels &kn, Mem &m, const Tuples &tup, const Params &g, Counters &ctr, long long n) {
    const uint64_t stride = Counters::stride(n, tup.t, 0), base = ctr.take(stride);
    const Launch pts = [&](const Sink &sk) { launch(kn.points, kBlocks, kThreads, 0, tup, g, base, stride, n, sk); };
    const long long cap = m.key_cap();
    if (n <= cap) {  // the points sorted in a, their spacings inserted into the hash table
        m.keys.ensure(n);
        pts(Sink{m.keys.a, m.keys.count, n, 0, 64, 0, 0});
        sort_keys(m.keys.a, n);
        const int pb = m.hash.prepare(n - 1);
        for (uint64_t part = 0; part < UINT64_C(1) << pb; ++part)
            launch(kn.spacings, kBlocks, kThreads, 0, static_cast<const unsigned long long *>(m.keys.a), n, m.hash.table(pb, part),
                   static_cast<unsigned long long *>(nullptr));
        return m.hash.collisions();
    }
    const int hb = part_bits(n, cap);
    std::vector<std::vector<unsigned long long>> bucket(size_t(1) << hb);
    thrust::device_vector<unsigned long long> bounds(bucket.size());
    for (size_t j = 0; j < bucket.size(); ++j) bounds[j] = static_cast<unsigned long long>(j) << (64 - hb);
    thrust::device_vector<long long> at(bucket.size());
    std::vector<long long> hat(bucket.size() + 1);
    bool have_last = false;
    unsigned long long last = 0;
    sorted_parts(kn, m, n, tup.b * tup.t, pts, [&](unsigned long long *a, long long c) {
        if (c == 0) return;
        unsigned long long first;
        CK(gpuMemcpy(&first, a, sizeof first, gpuMemcpyDeviceToHost));
        if (have_last) {  // the spacing across the parts
            const unsigned long long h = Mix64()(first - last);
            bucket[h >> (64 - hb)].push_back(h);
        }
        CK(gpuMemcpy(&last, a + c - 1, sizeof last, gpuMemcpyDeviceToHost));
        have_last = true;
        if (c < 2) return;
        launch(kn.spacings, kBlocks, kThreads, 0, static_cast<const unsigned long long *>(a), c, HashTable{}, m.keys.b);
        const thrust::device_ptr<unsigned long long> sp(m.keys.b);
        thrust::transform(sp, sp + c - 1, sp, Mix64());
        thrust::sort(sp, sp + c - 1);
        thrust::lower_bound(sp, sp + c - 1, bounds.begin(), bounds.end(), at.begin());
        thrust::copy(at.begin(), at.end(), hat.begin());
        hat.back() = c - 1;
        for (size_t j = 0; j < bucket.size(); ++j) {
            const size_t o = bucket[j].size(), k = static_cast<size_t>(hat[j + 1] - hat[j]);
            bucket[j].resize(o + k);
            if (k) CK(gpuMemcpy(bucket[j].data() + o, m.keys.b + hat[j], sizeof(unsigned long long) * k, gpuMemcpyDeviceToHost));
        }
    });
    long long eq = 0;
    for (auto &bk : bucket) {
        const long long k = static_cast<long long>(bk.size());
        if (k <= m.keys.cap) {
            if (k) CK(gpuMemcpy(m.keys.a, bk.data(), sizeof(unsigned long long) * k, gpuMemcpyHostToDevice));
            eq += count_equal_keys(kn, m, m.keys.a, k);
        } else {  // a bucket too large (of many equal spacings): on the host
            std::sort(bk.begin(), bk.end());
            for (long long i = 1; i < k; ++i) eq += bk[i] == bk[i - 1];
        }
        std::vector<unsigned long long>().swap(bk);
    }
    return eq;
}

// The device memory of ClosePairs for up to ncoords coordinates (of points of at least 2 coordinates).
struct CpBuffers {
    static constexpr uint32_t kMaxCells = UINT32_C(1) << 22;
    static constexpr unsigned kCap = 50000;  // the distances kept (NUM_JUMPS_LIM of TestU01)
    static constexpr size_t kFixed = (sizeof(unsigned) + 2 * sizeof(unsigned long long)) * kMaxCells;  // the memory of the cells
    long long ncoords;
    uint32_t *pts = nullptr, *sorted = nullptr, *cell = nullptr, *dist = nullptr;
    unsigned *count = nullptr, *ndist = nullptr;
    unsigned long long *start = nullptr, *cursor = nullptr, *nslab = nullptr;
    explicit CpBuffers(long long ncoords_) : ncoords(ncoords_) {
        CK(gpuMalloc(reinterpret_cast<void **>(&pts), sizeof(uint32_t) * ncoords));
        CK(gpuMalloc(reinterpret_cast<void **>(&sorted), sizeof(uint32_t) * ncoords));
        CK(gpuMalloc(reinterpret_cast<void **>(&cell), sizeof(uint32_t) * (ncoords / 2 + 1)));
        CK(gpuMalloc(reinterpret_cast<void **>(&dist), sizeof(uint32_t) * kCap));
        CK(gpuMalloc(reinterpret_cast<void **>(&count), sizeof(unsigned) * kMaxCells));
        CK(gpuMalloc(reinterpret_cast<void **>(&ndist), sizeof(unsigned)));
        CK(gpuMalloc(reinterpret_cast<void **>(&start), sizeof(unsigned long long) * kMaxCells));
        CK(gpuMalloc(reinterpret_cast<void **>(&cursor), sizeof(unsigned long long) * kMaxCells));
        CK(gpuMalloc(reinterpret_cast<void **>(&nslab), sizeof(unsigned long long)));
    }
    ~CpBuffers() {
        for (void *p : {static_cast<void *>(pts), static_cast<void *>(sorted), static_cast<void *>(cell), static_cast<void *>(dist),
                        static_cast<void *>(count), static_cast<void *>(ndist), static_cast<void *>(start), static_cast<void *>(cursor),
                        static_cast<void *>(nslab)})
            (void)gpuFree(p);
    }
    CpBuffers(const CpBuffers &) = delete;
    CpBuffers &operator=(const CpBuffers &) = delete;
    // the coordinates for the budget (10 bytes per coordinate, the cells aside), or those of the need if less
    static long long coords_for(size_t budget, long long need) {
        const long long fit = static_cast<long long>((budget > kFixed + (size_t(64) << 20) ? budget - kFixed : size_t(64) << 20) / 10);
        return std::min(fit, need);
    }
};

// The grid for the pairs of n points of t coordinates closer than dlim (in units of 2^-32): the number
// g of gridded coordinates and the cells c per coordinate (at most 2^32 / dlim, so that a cell is at
// least as wide as dlim, and c^g <= kMaxCells) of the least work, about the neighbouring cells visited
// (3^g / 2 per point) plus the pairs compared (n (3 / c)^g / 2 per point); g = 0 compares all pairs.
CpGrid cp_grid(long long n, int t, uint32_t dlim) {
    CpGrid best{t, 0, 1, dlim, 1};
    double best_cost = 0.5 * static_cast<double>(n);
    const double cmax = std::floor(std::ldexp(1.0, 32) / dlim);
    for (int g = 1; g <= std::min(t, detail::kCpMaxGrid); ++g) {
        double c = std::min(cmax, std::floor(std::pow(static_cast<double>(CpBuffers::kMaxCells), 1.0 / g) + 1e-9));
        while (std::pow(c, g) > CpBuffers::kMaxCells) --c;
        if (c < 3) break;
        const double cost = 0.5 * std::pow(3.0, g) + 0.5 * static_cast<double>(n) * std::pow(3 / c, g);
        if (cost < best_cost) {
            best_cost = cost;
            best = {t, g, static_cast<uint32_t>(c), dlim, static_cast<uint32_t>(std::pow(c, g))};
        }
    }
    return best;
}

// The distances below gr.dlim of the pairs of the np points in buf.pts but those of two points outside the
// core [core_lo, core_lo + core_w) of the first coordinate, added to buf.dist (counted by buf.ndist).
void cp_find(const Kernels &kn, CpBuffers &buf, const CpGrid &gr, long long np, int t, uint32_t core_lo, uint64_t core_w) {
    CK(gpuMemset(buf.count, 0, sizeof(unsigned) * gr.cells));
    launch(kn.cp_cells, kBlocks, kThreads, 0, static_cast<const uint32_t *>(buf.pts), np, gr, buf.cell, buf.count);
    // the start of every cell (the exclusive prefix sum of the counts), and the cursors of the scatter
    const thrust::device_ptr<unsigned> cnt(buf.count);
    const thrust::device_ptr<unsigned long long> st(buf.start), cur(buf.cursor);
    thrust::exclusive_scan(cnt, cnt + gr.cells, st, 0ULL);
    thrust::copy(st, st + gr.cells, cur);
    launch(kn.cp_scatter, kBlocks, kThreads, 0, static_cast<const uint32_t *>(buf.pts), np, t, static_cast<const uint32_t *>(buf.cell), buf.cursor,
           buf.sorted);
    launch(kn.cp_pairs, kBlocks, kThreads, 0, static_cast<const uint32_t *>(buf.sorted), np, gr, static_cast<const unsigned long long *>(buf.start),
           static_cast<const unsigned *>(buf.count), core_lo, core_w, buf.dist, buf.ndist, CpBuffers::kCap);
}

// The close pairs of the slab [lo, lo + w) of the first coordinate: its points and those of the margin
// [lo - dlim, lo) below it (made again from the positions), and the pairs of a point of the slab; a slab of
// too many points (a very uneven generator) is split.
void cp_slab(const Kernels &kn, CpBuffers &buf, const CpGrid &gr, const Params &g, uint64_t base, long long n, int t, uint64_t lo, uint64_t w) {
    const uint64_t span = w + gr.dlim;
    if (span > (UINT64_C(1) << 32) || w <= gr.dlim) throw TestFailure("ClosePairs: the memory budget is too small");
    const long long pcap = buf.ncoords / t;
    CK(gpuMemset(buf.nslab, 0, sizeof(unsigned long long)));
    launch(kn.cp_slab, kBlocks, kThreads, 0, g, base, n, t, static_cast<uint32_t>(lo - gr.dlim), span, buf.pts, buf.nslab, pcap);
    unsigned long long c;
    CK(gpuMemcpy(&c, buf.nslab, sizeof c, gpuMemcpyDeviceToHost));
    if (static_cast<long long>(c) > pcap) {
        cp_slab(kn, buf, gr, g, base, n, t, lo, w / 2);
        cp_slab(kn, buf, gr, g, base, n, t, lo + w / 2, w - w / 2);
        return;
    }
    cp_find(kn, buf, gr, static_cast<long long>(c), t, static_cast<uint32_t>(lo), w);
}

// ClosePairs: the sorted distances (in units of 2^-32) of the close pairs of n points of t coordinates
// (consecutive outputs from the next counters) in the torus, with the maximum norm. TestU01 collects
// the pairs whose jump W = n (n - 1) / 2 * 2^t * d^t (the volume of the ball of the distance d, times
// the pairs; an exponential of mean 1 for the closest pair) is at most max(m, the m-th smallest W).
// They are those among the pairs closer than the distance of W = B, from B = 2m (about 2m pairs),
// enlarged if fewer than m are found (or reduced if more than kCap are, for a very bad generator). If the
// points don't fit in the memory, they are taken by slabs of their first coordinate (see cp_slab); the
// pairs found are the same.
std::vector<uint32_t> close_pairs(const Kernels &kn, Mem &m, const Params &g, Counters &ctr, long long n, int t, int mm) {
    const long long nc = n * t;
    const uint64_t base = ctr.take(Counters::stride(nc, 1, 0));
    const long long want = CpBuffers::coords_for(m.budget, nc);
    if (!m.cp || m.cp->ncoords < want) {
        m.cp.reset();
        m.cp = std::make_unique<CpBuffers>(want);
    }
    CpBuffers &buf = *m.cp;
    const bool slabs = nc > buf.ncoords;
    if (slabs && t > kLanes) throw TestFailure("ClosePairs: too many coordinates for the slabs");
    if (!slabs) launch(kn.cp_outputs, kBlocks, kThreads, 0, g, base, nc, buf.pts);
    const double pairs_vol = 0.5 * static_cast<double>(n) * (n - 1.0) * std::ldexp(1.0, t);  // W = pairs_vol * d^t
    const uint32_t all = (UINT32_C(1) << 31) + 1;  // above any distance
    double B = 2.0 * mm;
    for (int iter = 0;; ++iter) {
        const double d = std::pow(B / pairs_vol, 1.0 / t);
        const uint32_t dlim = d >= 0.5 ? all : std::max(UINT32_C(1), static_cast<uint32_t>(std::ceil(std::ldexp(d, 32))));
        const CpGrid gr = cp_grid(n, t, dlim);
        CK(gpuMemset(buf.ndist, 0, sizeof(unsigned)));
        if (!slabs) {
            cp_find(kn, buf, gr, n, t, 0, UINT64_C(1) << 32);
        } else {  // 2^sb slabs of about n (2^-sb + dlim 2^-32) points each
            const long long pcap = buf.ncoords / t;
            int sb = 1;
            for (; sb < 24; ++sb) {
                const double e = static_cast<double>(n) * (std::ldexp(1.0, -sb) + std::ldexp(static_cast<double>(dlim), -32));
                if (e * 1.05 + 8 * std::sqrt(e) + 1000 <= static_cast<double>(pcap)) break;
            }
            for (uint64_t sl = 0; sl < UINT64_C(1) << sb; ++sl) cp_slab(kn, buf, gr, g, base, n, t, sl << (32 - sb), UINT64_C(1) << (32 - sb));
        }
        unsigned nd;
        CK(gpuMemcpy(&nd, buf.ndist, sizeof nd, gpuMemcpyDeviceToHost));
        if (iter < 40 && nd < static_cast<unsigned>(mm) && dlim < all) {
            B *= 4;
            continue;
        }
        if (iter < 40 && nd > CpBuffers::kCap && dlim > 1) {
            B /= 2;
            continue;
        }
        std::vector<uint32_t> dist(std::min(nd, CpBuffers::kCap));
        CK(gpuMemcpy(dist.data(), buf.dist, sizeof(uint32_t) * dist.size(), gpuMemcpyDeviceToHost));
        std::sort(dist.begin(), dist.end());
        return dist;
    }
}

// The Berlekamp-Massey algorithm on the bits s[0..n) (scomp_LinearComp of TestU01), on the host with the
// polynomials and the reversed sequence packed in 64-bit words: the linear complexity L, the number of the
// jumps of the complexity, and the numbers of the jumps of each size (cnt[1..jmax], those above jmax at jmax).
// The loops over the words have no branches inside, so that the compiler vectorizes them (with the SIMD of
// the host architecture, e.g. HOST_ARCH=-march=native of the Makefile).
void berlekamp_massey(const std::vector<uint8_t> &s, int jmax, long &L, long &numj, std::vector<long> &cnt) {
    const long n = static_cast<long>(s.size()), W = n / 64 + 4;
    std::vector<uint64_t> R(W + 1, 0), C(W, 0), B(W, 0), T(W, 0);
    for (long p = 0; p < n; ++p)  // R[p] = s[n - 1 - p], so that s[k - i] = R[n - 1 - k + i]
        if (s[n - 1 - p]) R[p / 64] |= UINT64_C(1) << (p % 64);
    cnt.assign(jmax + 1, 0);
    C[0] = B[0] = 1;
    L = numj = 0;
    long m = -1, lenb = 0;  // B has the degree at most lenb
    for (long k = 0; k < n; ++k) {
        // the discrepancy: the parity of sum_{i=0..L} c_i s[k - i] (c_0 = 1)
        const long off = n - 1 - k, q = off / 64, nw = L / 64 + 1;
        const int sh = static_cast<int>(off % 64);
        uint64_t acc = 0;
        const uint64_t *Rq = R.data() + q, *Cp = C.data();
        if (sh) {
            for (long w = 0; w < nw - 1; ++w) acc ^= Cp[w] & ((Rq[w] >> sh) | (Rq[w + 1] << (64 - sh)));
        } else {
            for (long w = 0; w < nw - 1; ++w) acc ^= Cp[w] & Rq[w];
        }
        {  // the last word, of the coefficients up to L
            uint64_t rw = Rq[nw - 1] >> sh;
            if (sh) rw |= Rq[nw] << (64 - sh);
            uint64_t cw = Cp[nw - 1];
            if (L % 64 != 63) cw &= (UINT64_C(1) << (L % 64 + 1)) - 1;
            acc ^= cw & rw;
        }
        if (!(__builtin_popcountll(acc) & 1)) continue;
        const bool grow = 2 * L <= k;
        if (grow) std::copy(C.begin(), C.begin() + nw, T.begin());
        const long sft = k - m, ws = sft / 64, nb = lenb / 64 + 1;
        const int bs = static_cast<int>(sft % 64);
        uint64_t *Cw = C.data() + ws;
        const uint64_t *Bp = B.data();
        if (bs) {  // C ^= B x^(k - m): B << bs word by word, with the carry of the word before
            Cw[0] ^= Bp[0] << bs;
            for (long w = 1; w < nb; ++w) Cw[w] ^= (Bp[w] << bs) | (Bp[w - 1] >> (64 - bs));
            Cw[nb] ^= Bp[nb - 1] >> (64 - bs);
        } else {
            for (long w = 0; w < nb; ++w) Cw[w] ^= Bp[w];
        }
        if (grow) {
            const long Ln = k + 1 - L;
            ++numj;
            ++cnt[std::min<long>(std::labs(k + 1 - 2 * Ln), jmax)];
            std::fill(B.begin(), B.begin() + nb, 0);
            std::copy(T.begin(), T.begin() + nw, B.begin());
            lenb = L;
            L = Ln;
            m = k;
        }
    }
}

// AppearanceSpacings: the term log(i - prev) of the key k[i] (block << ibits | index) of the sorted keys, prev
// being the index of the key before it if it has the same block (the most recent occurrence of the block),
// else 0 (TestU01 counts from 0 for a block not seen yet); 0 for the Q blocks of the initialization
struct AppearTerm {
    const unsigned long long *k;
    int ibits;
    long long Q;
    __host__ __device__ double operator()(long long i) const {
        const unsigned long long mask = (1ULL << ibits) - 1, idx = k[i] & mask;
        if (static_cast<long long>(idx) < Q) return 0;
        const unsigned long long prev = i > 0 && (k[i - 1] >> ibits) == (k[i] >> ibits) ? k[i - 1] & mask : 0;
        return log(static_cast<double>(idx - prev));
    }
};

// LempelZiv: the number of the words of the LZ78 parsing of the n bits of the values x (the s bits
// (x << r) >> (32 - s) of every value, the most significant first), as scomp_LempelZiv of TestU01 (a word
// left incomplete at the end counts if its node has children); the trie in arrays of the children
long lz78(const std::vector<uint32_t> &x, long long n, int r, int s) {
    std::vector<int32_t> ch(2, 0);  // the children (0 for none: the root is no child) of every node
    ch.reserve(static_cast<size_t>(2.4 * static_cast<double>(n) / std::log2(static_cast<double>(n))) + 64);
    long W = 0;
    long long i = 0;
    size_t vi = 0;
    int k = s - 1;
    uint32_t Y = (x[0] << r) >> (32 - s);
    while (i < n) {
        int32_t node = 0;
        for (;;) {
            const int bit = (Y >> k) & 1;
            const int32_t next = ch[2 * node + bit];
            if (next) {
                node = next;
                ++i;
            } else {  // a new word
                ++W;
                const int32_t nn = static_cast<int32_t>(ch.size() / 2);
                ch[2 * node + bit] = nn;
                ch.push_back(0);
                ch.push_back(0);
                node = nn;
                ++i;
                if (i < n && --k < 0) {
                    Y = (x[++vi] << r) >> (32 - s);
                    k = s - 1;
                }
                break;
            }
            if (i >= n) {
                if (ch[2 * node] || ch[2 * node + 1]) ++W;
                break;
            }
            if (--k < 0) {
                Y = (x[++vi] << r) >> (32 - s);
                k = s - 1;
            }
        }
    }
    return W;
}

// the Anderson-Darling statistic of the values (sorted here) and its p-value
std::pair<double, double> anderson_darling(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    v.insert(v.begin(), 0.0);  // the arrays of TestU01 start at 1
    const long n = static_cast<long>(v.size()) - 1;
    const double a2 = gofs_AndersonDarling(v.data(), n);
    return {a2, fbar_AndersonDarling(n, a2)};
}

// The statistics of ClosePairs over N replications, as in TestU01: NP, the AD test of the N values
// 1 - exp(-W_1) of the closest pairs; mNP, the AD test of the N p-values of the AD tests of the m
// values 1 - exp(-(W_i - W_{i-1})) of every replication (the spacings of the jumps, i.i.d. U(0, 1));
// mNP1, the AD test of these N m values; mNP2, the AD test of the jumps W <= m of all the replications
// divided by m (uniform given their number); NJumps, the number of these jumps against a Poisson
// distribution of the mean N m; mNP2S, the AD test of them after the spacings transform of TestU01.
struct CpStats {
    std::string_view name;
    double stat, p;
};
std::vector<CpStats> close_pairs_stats(const std::vector<std::vector<uint32_t>> &reps, long long n, int t, int m) {
    const double pairs_vol = 0.5 * static_cast<double>(n) * (n - 1.0) * std::ldexp(1.0, t);
    std::vector<double> wn, pad, wni, y;
    for (const auto &dist : reps) {
        std::vector<double> w(m, HUGE_VAL);  // missing pairs (fewer than m found) count as far
        for (size_t i = 0; i < dist.size(); ++i) {
            const double wi = pairs_vol * std::pow(std::ldexp(static_cast<double>(dist[i]), -32), t);
            if (static_cast<int>(i) < m) w[i] = wi;
            if (wi <= m) y.push_back(wi / m);
        }
        wn.push_back(-std::expm1(-w[0]));
        std::vector<double> u(m);
        double prev = 0;
        for (int i = 0; i < m; ++i) {
            u[i] = w[i] == HUGE_VAL ? 1.0 : -std::expm1(-(w[i] - prev));
            prev = w[i];
        }
        wni.insert(wni.end(), u.begin(), u.end());
        pad.push_back(anderson_darling(u).second);
    }
    std::vector<CpStats> s;
    const auto np = anderson_darling(wn), mnp = anderson_darling(pad), mnp1 = anderson_darling(wni);
    s.push_back({"NP", np.first, np.second});
    s.push_back({"mNP", mnp.first, mnp.second});
    s.push_back({"mNP1", mnp1.first, mnp1.second});
    const long N = static_cast<long>(reps.size());
    if (!y.empty()) {
        const auto mnp2 = anderson_darling(y);
        s.push_back({"mNP2", mnp2.first, mnp2.second});
    }
    fmass_INFO mass = fmass_CreatePoisson(static_cast<double>(N) * m);
    const long jumps = static_cast<long>(y.size());
    s.push_back({"NJumps", static_cast<double>(jumps), gofw_pDisc(fdist_Poisson2(mass, jumps), fbar_Poisson2(mass, jumps))});
    fmass_DeletePoisson(mass);
    if (!y.empty()) {
        std::vector<double> v = y, sp(y.size() + 1);
        std::sort(v.begin(), v.end());
        v.insert(v.begin(), 0.0);
        gofs_DiffD(v.data(), sp.data(), 1, jumps, 0.0, 1.0);
        gofs_IterateSpacings(v.data(), sp.data(), jumps);
        const auto mnp2s = anderson_darling(std::vector<double>(v.begin() + 1, v.end()));
        s.push_back({"mNP2S", mnp2s.first, mnp2s.second});
    }
    return s;
}

// The counts of n observations of the test in nbins categories, from the next counters; an observation
// takes values of the mean and the variance var (for the stride of the threads).
template <class T>
std::vector<unsigned long long> histogram(const Kernels &kn, const T &test, const Params &g, Counters &ctr, long long n, int nbins, double mean, double var,
                                    int extra = 0) {
    unsigned long long *d_hist;
    int *d_overrun;
    CK(gpuMalloc(reinterpret_cast<void **>(&d_hist), sizeof(unsigned long long) * nbins));
    CK(gpuMalloc(reinterpret_cast<void **>(&d_overrun), sizeof(int)));
    CK(gpuMemset(d_hist, 0, sizeof(unsigned long long) * nbins));
    CK(gpuMemset(d_overrun, 0, sizeof(int)));
    const int lds = nbins <= kMaxLdsBins;
    const uint64_t stride = Counters::stride(n, mean, var, extra);
    launch(kn.of(test), kBlocks, kThreads, lds ? sizeof(unsigned) * nbins : 0, test, g, ctr.take(stride), stride, n, d_hist, nbins, lds,
           d_overrun);
    std::vector<unsigned long long> h(nbins);
    int overrun;
    CK(gpuMemcpy(h.data(), d_hist, sizeof(unsigned long long) * nbins, gpuMemcpyDeviceToHost));
    CK(gpuMemcpy(&overrun, d_overrun, sizeof overrun, gpuMemcpyDeviceToHost));
    CK(gpuFree(d_hist));
    CK(gpuFree(d_overrun));
    // far beyond the margin for an ideal generator, so a structure taking too many values (e.g. a gap
    // test on outputs never in the interval) fails
    if (overrun) throw TestFailure("a thread ran out of its counters (the observations take far more values than expected)");
    return h;
}

// The chi-square statistic of the counts against the probabilities for n observations, after merging
// the categories whose expected counts are below kMinExpected into their neighbours (from both ends).
struct Chi2 {
    double x;
    long df;
};
Chi2 chi2(const std::vector<unsigned long long> &obs, const std::vector<double> &prob, long long n) {
    std::vector<double> o, e;
    double ao = 0, ae = 0;
    for (size_t i = 0; i < prob.size(); ++i) {  // merge from the low end, pushing small classes up
        ao += obs[i];
        ae += prob[i] * n;
        if (ae >= kMinExpected) {
            o.push_back(ao);
            e.push_back(ae);
            ao = ae = 0;
        }
    }
    if (ae > 0 || ao > 0) {  // the rest of the high end joins the last class
        if (e.empty()) throw TestFailure("too few observations");
        o.back() += ao;
        e.back() += ae;
    }
    double x = 0;
    for (size_t i = 0; i < o.size(); ++i) x += (o[i] - e[i]) * (o[i] - e[i]) / e[i];
    return {x, static_cast<long>(o.size()) - 1};
}

// the chi-square of the observed numbers obs against the expected numbers e of the classes, by the merging and the
// floating-point operations of TestU01 (gofs_MergeClasses: from the low end, a class of fewer than 10 expected
// joined with the next ones until 10, and a last class still below 10 joined with the previous one; then gofs_Chi2)
Chi2 chi2_gofs(const std::vector<unsigned long long> &obs, std::vector<double> e) {
    const long smax0 = static_cast<long>(e.size()) - 1;
    std::vector<long> loc(e.size());
    long ncl = 0, s = 0, s0 = 0, smin = 0, smax = smax0;
    while (s <= smax) {
        if (e[s] < kMinExpected) {
            s0 = s;
            double somme = e[s];
            while (somme < kMinExpected && s < smax) {
                e[s] = 0.0;
                ++s;
                somme += e[s];
            }
            e[s] = somme;
            for (long j = s0; j <= s; ++j) loc[j] = s;
        } else {
            loc[s] = s;
        }
        ++ncl;
        ++s;
    }
    smin = loc[smin];
    if (e[smax] < kMinExpected) {
        if (s0 > smin) --s0;
        e[s0] += e[smax];
        e[smax] = 0.0;
        --ncl;
        for (long j = s0 + 1; j <= smax; ++j) loc[j] = s0;
        smax = s0;
    }
    if (ncl < 2) throw TestFailure("fewer than two classes");
    std::vector<double> cnt(e.size(), 0.0);
    for (long j = 0; j <= smax0; ++j) cnt[loc[j]] += static_cast<double>(obs[j]);
    double x = 0;
    for (long j = smin; j <= smax; ++j)
        if (e[j] > 0.0) {
            const double d = cnt[j] - e[j];
            x += d * d / e[j];
        }
    return {x, ncl - 1};
}

// the sum of the values of the N replications as TestU01 takes it, for the same floating-point result: sorted
// (gofw_ActiveTests2 sorts them in place), summed in that order and divided by N (statcoll_Average), times N
double sum_stat(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    double sum = 0;
    for (const double x : v) sum += x;
    const double N = static_cast<double>(v.size());
    return N * (sum / N);
}

// the probabilities of the number of distinct values among k values in [0, d) (index 0..k)
// The correlations of the strings of s bits (s <= 32) and their populations (the numbers of strings of each), in the
// order of TestU01 (CreateCorr of sstring.c, after Guibas and Odlyzko): the correlations of length i are built from
// those of every length j <= i - 2 (as 10...0C for a correlation C of length j), and the correlation 11...1 last.
std::vector<std::pair<uint32_t, long long>> correlations(int s) {
    struct Corr {
        int nb;
        uint64_t c;
        long long pop;
        int ext = -1, ext0 = -1;  // the smallest extension, and the next one (indices in all)
    };
    std::vector<Corr> all{{0, 0, 1}, {1, 1, 2}};
    std::vector<std::vector<int>> lists(s + 1);
    lists[0] = {0};
    lists[1] = {1};
    const auto psi = [](uint64_t c, long j, long k) -> long long {
        if (k > j) return 0;
        if (k <= 0) return 1LL << -k;
        return static_cast<long long>(c >> (j - k) & 1);
    };
    for (int i = 2; i <= s; ++i) {
        for (int j = 0; j <= i - 2; ++j)
            for (size_t q = 0; q < lists[j].size(); ++q) {
                const int cj = lists[j][q];
                long long p = all[cj].pop * psi(all[cj].c, j, 2L * j - i);
                if (q + 1 == lists[j].size()) p -= 2 * psi(all[cj].c, j, 2L * j + 2 - i);  // 1C: only for C = 11...1, the last
                for (int e = all[cj].ext; e >= 0 && all[e].nb <= (i + j) / 2; e = all[e].ext0) p -= all[e].pop * psi(all[cj].c, j, 2L * all[e].nb - i);
                if (p <= 0) continue;
                uint64_t c = 1;
                for (int k = 0; k < j; ++k)
                    if (all[cj].c >> k & 1) c |= UINT64_C(1) << (k + i - j);
                all.push_back({i, c, p});
                const int ni = static_cast<int>(all.size()) - 1;
                if (all[cj].ext < 0) {
                    all[cj].ext = ni;
                } else {
                    int e = all[cj].ext;
                    while (all[e].ext0 >= 0) e = all[e].ext0;
                    all[e].ext0 = ni;
                }
                lists[i].push_back(ni);
            }
        all.push_back({i, i == 64 ? ~UINT64_C(0) : (UINT64_C(1) << i) - 1, 2});
        lists[i].push_back(static_cast<int>(all.size()) - 1);
    }
    std::vector<std::pair<uint32_t, long long>> out;
    for (const int i : lists[s]) out.push_back({static_cast<uint32_t>(all[i].c), all[i].pop});
    return out;
}

std::vector<double> poker_prob(int d, int k) {
    std::vector<double> p(k + 1, 0.0);
    p[0] = 1;
    for (int i = 0; i < k; ++i)
        for (int m = std::min(i + 1, d); m >= 0; --m)
            p[m] = p[m] * m / d + (m > 0 ? p[m - 1] * (d - m + 1) / d : 0.0);
    return p;
}

// the probabilities of the number of values until all of [0, d) appear (index 0..tmax, tmax lumped)
std::vector<double> coupon_prob(int d, int tmax) {
    std::vector<double> q(d + 1, 0.0), p(tmax + 1, 0.0);  // q[c]: c distinct values so far, not finished
    q[0] = 1;
    double sum = 0;
    for (int s = 1; s < tmax; ++s) {
        for (int c = d - 1; c >= 0; --c) {
            const double stay = q[c] * c / d, move = q[c] * (d - c) / d;
            q[c] = stay;
            if (c + 1 == d) p[s] = move;
            else q[c + 1] += move;
        }
        sum += p[s];
    }
    p[tmax] = std::max(0.0, 1 - sum);
    return p;
}

// the probabilities of the gap lengths for an interval of probability w (index 0..tmax, tmax lumped)
std::vector<double> gap_prob(double w, int tmax) {
    std::vector<double> p(tmax + 1);
    double tail = 1;
    for (int j = 0; j < tmax; ++j) {
        p[j] = w * tail;
        tail *= 1 - w;
    }
    p[tmax] = tail;
    return p;
}

// the smallest tmax whose lumped tail of the probabilities from f(t) has an expected count below
// kMinExpected for n observations (searched up to limit)
template <class F>
int lump_point(F tail_after, long long n, int lo, int limit) {
    int t = lo;
    while (t < limit && tail_after(t) * n >= kMinExpected) ++t;
    return t;
}

// the instances of a test in the batteries SmallCrush, Crush and BigCrush (the order of Suite)
template <class T>
using PerSuite = std::array<std::vector<T>, 3>;

bool wanted(const Options &o, const char *t) { return absl::StrContains(absl::StrCat(",", o.tests, ","), absl::StrCat(",", t, ",")); }

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

std::string format(const Result &r) {
    const char *mk = r.p < 0.001 || r.p > 0.999 ? "  *****" : "";
    const std::string &t = r.test, &q = r.params;
    switch (r.kind) {
    case Stat::kChiSquare: return std::format("{:<7} {:<34} chi2={:<12.6g} df={:<8} p={:<9.3g}{}  ({:.1f}s)", t, q, r.stat, r.df, r.p, mk, r.seconds);
    case Stat::kPoisson:
        return std::format("{:<7} {:<34} coll={:<11} mean={:<9.6g} p={:<9.3g}{}  ({:.1f}s)", t, q, static_cast<long long>(r.stat), r.mean, r.p, mk, r.seconds);
    case Stat::kNormal: return std::format("{:<7} {:<34} z={:<14.6g} p={:<9.3g}{}  ({:.1f}s)", t, q, r.stat, r.p, mk, r.seconds);
    case Stat::kAndersonDarling: return std::format("{:<7} {:<34} AD={:<13.6g} p={:<9.3g}{}  ({:.1f}s)", t, q, r.stat, r.p, mk, r.seconds);
    case Stat::kClosePairs: return std::format("{:<7} {:<34} stat={:<12.6g} p={:<9.3g}{}  ({:.1f}s)", t, q, r.stat, r.p, mk, r.seconds);
    case Stat::kDiscrete: return std::format("{:<7} {:<34} p={:<9.3g}{}", t, q, r.p, mk);
    }
    return {};
}

// the device memory of the tests on points and of ClosePairs, allocated for the largest scale so far
struct Battery::Impl {
    Mem mem;
};

Battery::Battery() : impl_(std::make_unique<Impl>()) {}
Battery::~Battery() = default;

// Run the tests on the sequence of the generator and the parameters g. Every test and replication takes
// the next range of positions.
Summary Battery::run(const Generator &gen, const Params &g, const Options &o) {
    const Kernels &kn = gen.impl();
    Mem &mem = impl_->mem;
    if (o.memory != mem.budget) {  // the memory of another budget
        mem.keys.release();
        mem.cp.reset();
        mem.fft.reset();
        mem.budget = o.memory;
    }
    const auto scaled = [&](double n) { return static_cast<long long>(n * o.scale); };
    Counters ctr(o.first);
    Summary sum;
    const auto emit = [&](const char *name, const std::string &params, Stat kind, double stat, double p, double z, double sec, long df = 0, double mean = 0) {
        sum.add(p, z);
        if (o.report) o.report(Result{name, params, kind, stat, mean, df, p, z, sec});
    };
    const auto report = [&](const char *name, const std::string &params, Chi2 c, double sec) {
        const double p = fbar_ChiSquare2(c.df, 12, c.x);
        emit(name, params, Stat::kChiSquare, c.x, p, (c.x - c.df) / std::sqrt(2.0 * c.df), sec, c.df);
    };
    std::string params;  // the parameters of the current statistic
    const size_t bat = static_cast<size_t>(o.suite);
    if (bat > 2) throw std::invalid_argument("testu01_gpu: unknown battery");

    // MaxOft, SampleProd and SampleMean: the Anderson-Darling test of n values transformed to U(0, 1) by their
    // distribution function (as keys of 64 bits, see kuniform), sorted on GPU by parts if they don't fit in
    // the memory, and collected on the host
    const auto ad_sorted = [&](long long n, const Launch &f) {
        if (o.fast) {  // the terms of the statistic summed on GPU, part by part
            double a2 = 0;
            long long off = 0;
            sorted_parts(kn, mem, n, 64, f, [&](unsigned long long *a, long long c) {
                a2 += thrust::transform_reduce(thrust::counting_iterator<long long>(0), thrust::counting_iterator<long long>(c), AdTerm{a, off, n}, 0.0,
                                               thrust::plus<double>());
                off += c;
            });
            a2 = -static_cast<double>(n) - a2 / static_cast<double>(n);
            return std::pair<double, double>{a2, fbar_AndersonDarling(static_cast<long>(n), a2)};
        }
        std::vector<unsigned long long> keys;
        keys.reserve(n);
        sorted_parts(kn, mem, n, 64, f, [&](unsigned long long *a, long long c) {
            const size_t at = keys.size();
            keys.resize(at + c);
            if (c) CK(gpuMemcpy(keys.data() + at, a, sizeof(unsigned long long) * c, gpuMemcpyDeviceToHost));
        });
        std::vector<double> v(n + 1);  // the arrays of TestU01 start at 1
        for (long long i = 0; i < n; ++i) v[i + 1] = (static_cast<double>(keys[i]) + 0.5) * 0x1p-64;
        const double a2 = gofs_AndersonDarling(v.data(), static_cast<long>(n));
        return std::pair<double, double>{a2, fbar_AndersonDarling(static_cast<long>(n), a2)};
    };
    const auto report_ad = [&](const char *name, const std::string &params, double a2, double p, double sec) {
        emit(name, params, Stat::kAndersonDarling, a2, p, p > 0 ? std::max(0.0, -finv_Normal1(std::max(p, 1e-300))) : HUGE_VAL, sec);
    };
    if (wanted(o, "poker")) {
        struct P { double n; int r, d, k; };  // N = 1
        static const PerSuite<P> sets{{{{4e5, 24, 64, 64}}, {{4e7, 0, 16, 16}, {4e7, 26, 16, 16}, {1e7, 0, 64, 64}, {1e7, 24, 64, 64}}, {{4e8, 0, 8, 8}, {4e8, 27, 8, 8}, {1e8, 0, 32, 32}, {1e8, 25, 32, 32}}}};
        for (const auto &s : sets[bat]) {
            if (s.d < 2 || s.d > 64) throw std::invalid_argument("poker: d must be in [2, 64]");
            const long long n = scaled(s.n);
            const auto t0 = std::chrono::steady_clock::now();
            const auto h = histogram(kn, Poker{s.r, s.d, s.k}, g, ctr, n, s.k + 1, s.k, 0);
            params = std::format("n={} r={} d={} k={}", n, s.r, s.d, s.k);
            report("poker", params, chi2(h, poker_prob(s.d, s.k), n), seconds_since(t0));
            if (sum.max_z > o.limit) return sum;
        }
    }
    if (wanted(o, "coupon")) {
        struct P { double n; int r, d; };  // N = 1
        static const PerSuite<P> sets{{{{5e5, 26, 16}}, {{4e7, 0, 4}, {4e7, 28, 4}, {1e7, 0, 16}, {1e7, 26, 16}}, {{2e8, 0, 8}, {2e8, 10, 8}, {2e8, 20, 8}, {2e8, 27, 8}}}};
        for (const auto &s : sets[bat]) {
            const int r = s.r, d = s.d;
            if (d < 2 || d > 64) throw std::invalid_argument("coupon: d must be in [2, 64]");
            const long long n = scaled(s.n);
            const int tmax = lump_point([&](int t) { const auto p = coupon_prob(d, t); return p[t]; }, n, d, 4000);
            // the values until all d appear: a sum of geometric numbers of the probabilities (d - c) / d
            double mean = 0, var = 0;
            for (int c = 0; c < d; ++c) {
                const double p = static_cast<double>(d - c) / d;
                mean += 1 / p;
                var += (1 - p) / (p * p);
            }
            const auto t0 = std::chrono::steady_clock::now();
            const auto h = histogram(kn, Coupon{r, d, tmax}, g, ctr, n, tmax + 1, mean, var);
            params = std::format("n={} r={} d={}", n, r, d);
            report("coupon", params, chi2(h, coupon_prob(d, tmax), n), seconds_since(t0));
            if (sum.max_z > o.limit) return sum;
        }
    }
    if (wanted(o, "gap")) {
        struct P { double n; int r; int lg; };  // N = 1, alpha = 0, beta = 2^-lg
        static const PerSuite<P> sets{{{{2e5, 22, 8}}, {{1e8, 0, 3}, {1e8, 27, 3}, {5e6, 0, 8}, {5e6, 22, 8}}, {{5e8, 0, 4}, {3e8, 25, 5}, {1e8, 0, 7}, {1e7, 20, 10}}}};
        for (const auto &s : sets[bat]) {
            const long long n = scaled(s.n);
            const double w = std::ldexp(1.0, -s.lg);
            const int tmax = lump_point([&](int t) { return std::pow(1 - w, t); }, n, 1, 1 << 20);
            const auto t0 = std::chrono::steady_clock::now();
            // the values per gap (including the one in the interval): geometric of the probability w
            const auto h = histogram(kn, Gap{s.r, tmax, 0, UINT64_C(1) << (32 - s.lg)}, g, ctr, n, tmax + 1, 1 / w, (1 - w) / (w * w));
            params = std::format("n={} r={} beta=2^-{}", n, s.r, s.lg);
            report("gap", params, chi2(h, gap_prob(w, tmax), n), seconds_since(t0));
            if (sum.max_z > o.limit) return sum;
        }
    }
    if (wanted(o, "maxoft")) {
        struct P { int N; double n; int d, t; };  // r = 0
        static const PerSuite<P> sets{{{{1, 2e6, 100000, 6}}, {{10, 1e7, 100000, 5}, {5, 1e7, 100000, 10}, {1, 1e7, 100000, 20}, {1, 1e7, 100000, 30}}, {{40, 1e7, 100000, 8}, {30, 1e7, 100000, 16}, {20, 1e7, 100000, 24}, {20, 1e7, 100000, 32}}}};
        for (const auto &s : sets[bat]) {
            const int d = s.d;
            const long long n = scaled(s.n);
            const auto t0 = std::chrono::steady_clock::now();
            Chi2 all{0, 0};
            std::vector<double> ps;                // the p-values of the AD tests of the replications
            std::pair<double, double> one{0, 0};  // the AD statistic and the p-value of the last one
            for (int rep = 0; rep < s.N; ++rep) {
                Counters again = ctr;  // the same counters, for the AD test of the same maxima
                const auto h = histogram(kn, MaxOft{d, s.t}, g, ctr, n, d, s.t, 0);
                const Chi2 c = chi2(h, std::vector<double>(d, 1.0 / d), n);
                all.x += c.x;
                all.df += c.df;
                const uint64_t stride = Counters::stride(n, s.t, 0), base = again.take(stride);
                one = ad_sorted(n, [&](const Sink &sk) {
                    launch(kn.uniform, kBlocks, kThreads, 0, 2, 0, s.t, g, base, stride, n, static_cast<const double *>(nullptr), 0, 0.0, sk);
                });
                ps.push_back(one.second);
            }
            params = std::format("N={} n={} d={} t={}", s.N, n, d, s.t);
            report("maxoft", params, all, seconds_since(t0));
            // the AD test of the maxima against x^t (Bas of TestU01): that of the N p-values, or for N = 1 of the replication
            const auto ad = s.N == 1 ? one : anderson_darling(ps);
            report_ad("maxoft", params + " AD", ad.first, ad.second, seconds_since(t0));
            if (sum.max_z > o.limit) return sum;
        }
    }
    // the tests on points: the total number of collisions over N replications, against a Poisson
    // distribution of the mean N * mu; the p-value is P[X >= observed]
    const auto report_poisson = [&](const char *name, const std::string &params, long long obs, double mean, double sec) {
        const double p = fbar_Poisson1(mean, obs);
        emit(name, params, Stat::kPoisson, static_cast<double>(obs), p, (obs - mean) / std::sqrt(mean), sec, 0, mean);
    };
    const auto tuples = [](int r, int b, int t, int overlap) {
        return Tuples{r, b, t, overlap, b * t == 64 ? ~UINT64_C(0) : (UINT64_C(1) << (b * t)) - 1};
    };
    {
        if (wanted(o, "collision")) {
            struct P { int N; double n; int r, b, t; };  // d = 2^b
            static const PerSuite<P> sets{{{}, {{10, 1e7, 0, 20, 2}, {10, 1e7, 10, 20, 2}, {10, 1e7, 0, 10, 4}, {10, 1e7, 20, 10, 4}, {10, 1e7, 0, 5, 8}, {10, 1e7, 25, 5, 8}, {10, 1e7, 0, 2, 20}, {10, 1e7, 28, 2, 20}}, {{30, 2e7, 0, 21, 2}, {30, 2e7, 9, 21, 2}, {30, 2e7, 0, 14, 3}, {30, 2e7, 16, 14, 3}, {30, 2e7, 0, 6, 7},
                                                     {30, 2e7, 24, 6, 7}, {30, 2e7, 0, 3, 14}, {30, 2e7, 27, 3, 14}, {30, 2e7, 0, 2, 21}, {30, 2e7, 28, 2, 21}}}};
            for (const auto &s : sets[bat]) {
                const int N = s.N;
                const long long n = scaled(s.n);
                const double k = std::ldexp(1.0, s.b * s.t);
                const double mu = n + k * std::expm1(n * std::log1p(-1 / k));  // expected collisions of n points in k cells
                const auto t0 = std::chrono::steady_clock::now();
                long long c = 0;
                for (int rep = 0; rep < N; ++rep) {
                    c += collision_over(kn, mem, tuples(s.r, s.b, s.t, 1), g, ctr, n);
                    if ((c - (rep + 1) * mu) / std::sqrt((rep + 1) * mu) > o.limit) {  // already far above its mean
                        sum.add(0, (c - (rep + 1) * mu) / std::sqrt((rep + 1) * mu));
                        return sum;
                    }
                }
                params = std::format("N={} n={} r={} d=2^{} t={}", N, n, s.r, s.b, s.t);
                report_poisson("collover", params, c, N * mu, seconds_since(t0));
                if (sum.max_z > o.limit) return sum;
            }
        }
        if (wanted(o, "kcollision")) {
            // sknuth_Collision: the collisions of n non-overlapping points of t values in d^t cells (d = 2^b), against
            // the Poisson distribution of TestU01 for n > 1e5 and n <= 0.1 k (smultin_CreateCollisions), of the mean
            // n - k + k (1 - 1/k)^n, summed over N
            struct P { int N; double n; int r, b, t; };
            static const PerSuite<P> sets{{{{1, 5e6, 0, 16, 2}}, {}, {}}};
            for (const auto &s : sets[bat]) {
                const long long n = scaled(s.n);
                const double k = std::ldexp(1.0, s.b * s.t);
                if (n <= 100000 || n > 0.1 * k) throw TestFailure("kcollision: n out of the range of the Poisson approximation");
                const double mu = n + k * std::expm1(n * std::log1p(-1 / k));
                const auto t0 = std::chrono::steady_clock::now();
                long long c = 0;
                for (int rep = 0; rep < s.N; ++rep) c += collision_over(kn, mem, tuples(s.r, s.b, s.t, 0), g, ctr, n);
                params = std::format("N={} n={} r={} d=2^{} t={}", s.N, n, s.r, s.b, s.t);
                report_poisson("kcoll", params, c, s.N * mu, seconds_since(t0));
                if (sum.max_z > o.limit) return sum;
            }
        }
        if (wanted(o, "birthday")) {
            struct P { int N; double n; int r, b, t; };
        static const PerSuite<P> sets{{{{1, 5e6, 0, 30, 2}}, {{5, 2e7, 0, 31, 2}, {5, 2e7, 0, 21, 3}, {5, 2e7, 0, 16, 4}, {3, 2e7, 0, 9, 7}, {3, 2e7, 7, 9, 7}, {3, 2e7, 14, 8, 8}, {3, 2e7, 22, 8, 8}}, {{100, 1e7, 0, 31, 2}, {20, 2e7, 0, 21, 3}, {20, 3e7, 14, 16, 4}, {20, 2e7, 0, 9, 7}, {20, 2e7, 7, 9, 7}, {20, 3e7, 14, 8, 8}, {20, 3e7, 22, 8, 8}, {20, 3e7, 0, 4, 16}, {20, 3e7, 26, 4, 16}}}};
            for (const auto &s : sets[bat]) {
                const long long n = scaled(s.n);
                const double k = std::ldexp(1.0, s.b * s.t);
                const double lambda = static_cast<double>(n) * n * n / (4 * k);  // expected collisions of the spacings
                const auto t0 = std::chrono::steady_clock::now();
                long long c = 0;
                for (int rep = 0; rep < s.N; ++rep) {
                    c += birthday_spacings(kn, mem, tuples(s.r, s.b, s.t, 0), g, ctr, n);
                    if ((c - (rep + 1) * lambda) / std::sqrt((rep + 1) * lambda) > o.limit) {
                        sum.add(0, (c - (rep + 1) * lambda) / std::sqrt((rep + 1) * lambda));
                        return sum;
                    }
                }
                params = std::format("N={} n={} r={} d=2^{} t={}", s.N, n, s.r, s.b, s.t);
                report_poisson("birthday", params, c, s.N * lambda, seconds_since(t0));
                if (sum.max_z > o.limit) return sum;
            }
        }
    }
    if (wanted(o, "closepairs")) {
        struct P { int N; double n; int t; bool s2 = true; };  // s2: mNP2S reported (not by Crush for t = 2, as bbattery.c)
        static const PerSuite<P> sets{{{}, {{10, 2e6, 2, false}, {10, 2e6, 3}, {5, 2e6, 7}}, {{30, 6e6, 3}, {20, 4e6, 5}, {10, 3e6, 9}, {5, 2e6, 16}}}};  // r = 0, p = 0, m = 30
        for (const auto &s : sets[bat]) {
            const int m = 30;
            const long long n = scaled(s.n);
            const auto t0 = std::chrono::steady_clock::now();
            std::vector<std::vector<uint32_t>> reps;
            for (int rep = 0; rep < s.N; ++rep) reps.push_back(close_pairs(kn, mem, g, ctr, n, s.t, m));
            const double sec = seconds_since(t0);
            for (const CpStats &c : close_pairs_stats(reps, n, s.t, m)) {
                if (!s.s2 && c.name == "mNP2S") continue;
                params = std::format("N={} n={} t={} m={} {}", s.N, n, s.t, m, c.name);
                emit("cpairs", params, Stat::kClosePairs, c.stat, c.p, 0, sec);
            }
        }
    }
    if (wanted(o, "cpbitmatch")) {
        // ClosePairsBitMatch (snpair_ClosePairsBitMatch): N times n points of k coordinates (consecutive values u =
        // (x << r) / 2^32, not on the torus); the distance of a pair is the least over the coordinates of the number of
        // their equal leading bits (2 * 64 for equal coordinates, as TestU01 with 64-bit longs), and MaxY the largest of
        // all the pairs, against Prob[max over N n (n - 1) / 2 pairs <= MaxY] of the distance of a pair, P[Y >= y] =
        // 2^(-k y). MaxY is the largest y for which two points have the same top y bits of every coordinate: found by a
        // binary search of the collisions of the keys of the top y bits of the k coordinates (k y <= 64 bits; beyond,
        // MaxY is the lower bound 64 / k, as the pairs so close are improbable for k > 2)
        struct P { int N; double n; int r, k; };
        static const PerSuite<P> sets{{{}, {{4, 4e6, 0, 2}, {2, 4e6, 0, 4}}, {}}};
        for (const auto &sv : sets[bat]) {
            const int k = sv.k;
            if (k < 1 || k > 64 || sv.r < 0 || sv.r > 31) throw std::invalid_argument("cpbitmatch: 1 <= k <= 64, 0 <= r <= 31");
            const long long n = scaled(sv.n);
            const int ymax = std::min(32 - sv.r, 64 / k);
            const auto t0 = std::chrono::steady_clock::now();
            int maxy = 0;
            for (int rep = 0; rep < sv.N; ++rep) {
                const uint64_t stride = Counters::stride(n, k, 0), base = ctr.take(stride);
                const auto collide = [&](int y) {
                    const Tuples tup = tuples(sv.r, y, k, 0);
                    return count_equal(kn, mem, n, [&](const Sink &sk) { launch(kn.points, kBlocks, kThreads, 0, tup, g, base, stride, n, sk); }) > 0;
                };
                int lo = 0, hi = ymax;  // collide(lo) (y = 0: every pair), the largest y in [lo, hi] with a collision
                while (lo < hi) {
                    const int mid = (lo + hi + 1) / 2;
                    if (collide(mid)) lo = mid;
                    else hi = mid - 1;
                }
                // equal points in all their bits (only for k <= 2, with 32 - r bits): equal coordinates, as TestU01
                int y = lo;
                if (y == 32 - sv.r && k * (32 - sv.r) <= 64) y = 128;
                maxy = std::max(maxy, y);
            }
            const double pairs = static_cast<double>(sv.N) * n * (n - 1) / 2;
            const double z1 = std::exp2(-static_cast<double>(k) * (maxy + 1));
            double pl, pr;
            if (z1 > DBL_EPSILON) {
                pl = std::exp(std::log1p(-z1) * pairs);
                pr = 1 - pl;
            } else {
                pr = z1 * pairs;
                pl = 1 - pr;
            }
            params = std::format("N={} n={} r={} k={} maxy={}", sv.N, n, sv.r, k, maxy);
            emit("cpbitm", params, Stat::kDiscrete, static_cast<double>(maxy), gofw_pDisc(pl, pr), 0, seconds_since(t0));
            if (sum.max_z > o.limit) return sum;
        }
    }
    if (wanted(o, "permutation")) {
        struct P { double n; int r, t; };
        static const PerSuite<P> sets{{{}, {{5e7, 0, 10}, {5e7, 15, 10}}, {{1e9, 5, 3}, {1e9, 5, 5}, {5e8, 5, 7}, {5e8, 10, 10}}}};  // N = 1
        for (const auto &s : sets[bat]) {
            const long long n = scaled(s.n);
            int cells = 1;
            for (int i = 2; i <= s.t; ++i) cells *= i;
            const auto t0 = std::chrono::steady_clock::now();
            const auto h = histogram(kn, Permut{s.r, s.t}, g, ctr, n, cells, s.t, 0);
            params = std::format("n={} r={} t={}", n, s.r, s.t);
            // the chi-square of TestU01 (Delta = 1), whose correction of the mean is negligible here
            report("permut", params, chi2(h, std::vector<double>(cells, 1.0 / cells), n), seconds_since(t0));
            if (sum.max_z > o.limit) return sum;
        }
    }
    if (wanted(o, "run")) {
        // the modified chi-square of Knuth (TestU01 sknuth_Run) over N sequences of n values, against
        // the chi-square of 6N degrees of freedom (as BigCrush takes the p-value of the sum)
        static const double A[6][6] = {{4529.35365, 9044.90208, 13567.9452, 18091.2672, 22614.7139, 27892.1588},
                                       {9044.90208, 18097.0254, 27139.4552, 36186.6493, 45233.8198, 55788.8311},
                                       {13567.9452, 27139.4552, 40721.3320, 54281.2656, 67852.0446, 83684.5705},
                                       {18091.2672, 36186.6493, 54281.2656, 72413.6082, 90470.0789, 111580.110},
                                       {22614.7139, 45233.8198, 67852.0446, 90470.0789, 113261.815, 139475.555},
                                       {27892.1588, 55788.8311, 83684.5705, 111580.110, 139475.555, 172860.170}};
        static const double B[6] = {1.0 / 6, 5.0 / 24, 11.0 / 120, 19.0 / 720, 29.0 / 5040, 1.0 / 840};
        struct P { int N; double n; int r, up; };
        static const PerSuite<P> sets{{{}, {{1, 5e8, 0, 1}, {1, 5e8, 15, 0}}, {{5, 1e9, 0, 0}, {10, 1e9, 15, 1}}}};
        unsigned long long *d_cnt;
        CK(gpuMalloc(reinterpret_cast<void **>(&d_cnt), sizeof(unsigned long long) * 6));
        for (const auto &s : sets[bat]) {
            const long long n = scaled(s.n);
            const auto t0 = std::chrono::steady_clock::now();
            std::vector<double> khi;
            // the N sequences of n values one after the other, as TestU01
            const uint64_t start0 = ctr.take((s.N * n + Counters::kThreadsPerLaunch - 1) / Counters::kThreadsPerLaunch);
            for (int rep = 0; rep < s.N; ++rep) {
                CK(gpuMemset(d_cnt, 0, sizeof(unsigned long long) * 6));
                const uint64_t start = start0 + static_cast<uint64_t>(rep) * n;
                launch(kn.krun, kBlocks, kThreads, 0, g, start, n, s.r, s.up, d_cnt);
                unsigned long long c[6];
                CK(gpuMemcpy(c, d_cnt, sizeof c, gpuMemcpyDeviceToHost));
                double k = 0;
                for (int i = 0; i < 6; ++i)
                    for (int j = 0; j < 6; ++j) k += A[i][j] * (c[i] - n * B[i]) * (c[j] - n * B[j]);
                khi.push_back(k / (n - 6.0));
            }
            params = std::format("N={} n={} r={} {}", s.N, n, s.r, s.up ? "up" : "down");
            report("run", params, Chi2{sum_stat(khi), 6L * s.N}, seconds_since(t0));
            if (sum.max_z > o.limit) {
                CK(gpuFree(d_cnt));
                return sum;
            }
        }
        CK(gpuFree(d_cnt));
    }
    if (wanted(o, "serial")) {
        // SerialOver (N = 1, d = 256, t = 3): the chi-square X of the overlapping 3-tuples minus that of the
        // 2-tuples (circular, as TestU01), against the chi-square of k - k1 degrees of freedom (Good)
        struct P { double n; int r, b, t; };  // N = 1, d = 2^b
        static const PerSuite<P> sets{{{}, {{5e8, 0, 12, 2}, {3e8, 0, 6, 4}}, {{1e9, 0, 8, 3}, {1e9, 22, 8, 3}}}};
        for (const auto &s : sets[bat]) {
            const int r = s.r, b = s.b, t = s.t;
            const long long n = scaled(s.n);
            const uint64_t k = UINT64_C(1) << (b * t), k1 = UINT64_C(1) << (b * (t - 1));
            const auto t0 = std::chrono::steady_clock::now();
            unsigned *d_cnt, *d_cnt1;
            CK(gpuMalloc(reinterpret_cast<void **>(&d_cnt), sizeof(unsigned) * k));
            CK(gpuMalloc(reinterpret_cast<void **>(&d_cnt1), sizeof(unsigned) * k1));
            CK(gpuMemset(d_cnt, 0, sizeof(unsigned) * k));
            CK(gpuMemset(d_cnt1, 0, sizeof(unsigned) * k1));
            const uint64_t start = ctr.take((n + Counters::kThreadsPerLaunch - 1) / Counters::kThreadsPerLaunch);
            launch(kn.serial, kBlocks, kThreads, 0, g, start, n, r, b, t, d_cnt, d_cnt1);
            std::vector<unsigned> c(k), c1(k1);
            CK(gpuMemcpy(c.data(), d_cnt, sizeof(unsigned) * k, gpuMemcpyDeviceToHost));
            CK(gpuMemcpy(c1.data(), d_cnt1, sizeof(unsigned) * k1, gpuMemcpyDeviceToHost));
            CK(gpuFree(d_cnt));
            CK(gpuFree(d_cnt1));
            // the chi-square of the counts of the cells, by the same floating-point operations as TestU01 (CalcPowDivEqual of
            // smultin.c, the cells in their order): for the t-tuples, the terms (c - e)^2 / e summed if 6 e <= k (its
            // precomputed table, flagTab), else the (c - e)^2 summed and divided by e; for the (t - 1)-tuples always the latter
            const auto pearson = [n](const std::vector<unsigned> &v, bool table) {
                const double e = static_cast<double>(n) / v.size();
                double x = 0;
                if (!table || 6.0 * e > static_cast<double>(v.size())) {
                    for (const unsigned ci : v) {
                        const double d = ci - e;
                        x += d * d;
                    }
                    return x / e;
                }
                for (const unsigned ci : v) {
                    const double d = ci - e;
                    x += d * d / e;
                }
                return x;
            };
            params = std::format("n={} r={} d={} t={}", n, r, 1 << b, t);
            // standardized by the corrected mean and standard deviation of Read and Cressie, as TestU01
            // (smultin_PowDivMomCorChi for Delta = 1, of k - k1 cells): x = (X - X1 - mu) / sigma
            const double kk = static_cast<double>(k - k1), nn = static_cast<double>(n);
            const double sigma = std::sqrt(1.0 + (2.0 - 2.0 * kk - kk * kk + kk * kk) / (2.0 * nn * (kk - 1.0)));
            const double mu = (kk - 1.0) * (1.0 - sigma);
            const double x = (pearson(c, true) - pearson(c1, false) - mu) / sigma;
            report("serial", params, Chi2{x, static_cast<long>(k - k1)}, seconds_since(t0));
            if (sum.max_z > o.limit) return sum;
        }
    }
    if (wanted(o, "collpermut")) {
        // CollisionPermut (N = 20, n = 2e7, t = 14): collisions of the orders of 14 values among 14! cells
        struct P { int N; double n; int r, t; };
        static const PerSuite<P> sets{{{}, {{5, 1e7, 0, 13}, {5, 1e7, 15, 13}}, {{20, 2e7, 0, 14}, {20, 2e7, 10, 14}}}};
        for (const auto &s : sets[bat]) {
            const int N = s.N, r = s.r, t = s.t;
            const long long n = scaled(s.n);
            double k = 1;
            for (int i = 2; i <= t; ++i) k *= i;
            const double mu = n + k * std::expm1(n * std::log1p(-1 / k));
            const auto t0 = std::chrono::steady_clock::now();
            long long c = 0;
            for (int rep = 0; rep < N; ++rep) {
                const uint64_t stride = Counters::stride(n, t, 0), base = ctr.take(stride);
                c += count_equal(kn, mem, n, [&](const Sink &sk) { launch(kn.perm_points, kBlocks, kThreads, 0, r, t, g, base, stride, n, sk); });
                if ((c - (rep + 1) * mu) / std::sqrt((rep + 1) * mu) > o.limit) {
                    sum.add(0, (c - (rep + 1) * mu) / std::sqrt((rep + 1) * mu));
                    return sum;
                }
            }
            params = std::format("N={} n={} r={} t={}", N, n, r, t);
            report_poisson("collperm", params, c, N * mu, seconds_since(t0));
            if (sum.max_z > o.limit) return sum;
        }
    }
    if (wanted(o, "weight")) {
        // WeightDistrib (N = 1, n = 2e7, k = 256, alpha = 0): the binomial number of values below beta
        struct P { double n; int r, k, lg; };  // N = 1, alpha = 0, beta = 2^-lg
        static const PerSuite<P> sets{{{{2e5, 27, 256, 3}}, {{2e6, 0, 256, 3}, {2e6, 8, 256, 3}, {2e6, 16, 256, 3}, {2e6, 24, 256, 3}}, {{2e7, 0, 256, 2}, {2e7, 20, 256, 2}, {2e7, 28, 256, 2}, {2e7, 0, 256, 4}, {2e7, 10, 256, 4}, {2e7, 26, 256, 4}}}};
        for (const auto &s : sets[bat]) {
            const int k = s.k;
            const long long n = scaled(s.n);
            const double p = std::ldexp(1.0, -s.lg);
            std::vector<double> prob(k + 1);
            for (int i = 0; i <= k; ++i) prob[i] = std::exp(std::lgamma(k + 1.0) - std::lgamma(i + 1.0) - std::lgamma(k - i + 1.0) + i * std::log(p) + (k - i) * std::log1p(-p));
            const auto t0 = std::chrono::steady_clock::now();
            const auto h = histogram(kn, detail::Weight{s.r, k, UINT64_C(1) << (32 - s.lg)}, g, ctr, n, k + 1, k, 0);
            params = std::format("n={} r={} k={} beta=2^-{}", n, s.r, k, s.lg);
            report("weight", params, chi2(h, prob, n), seconds_since(t0));
            if (sum.max_z > o.limit) return sum;
        }
    }
    if (wanted(o, "savir2")) {
        // Savir2 (N = 10, n = 1e7, r = 10, m = 2^20, t = 30): the sum of the chi-squares of the N replications
        struct P { int N; double n; int r, lgm, t; };  // m = 2^lgm
        static const PerSuite<P> sets{{{}, {{1, 2e7, 0, 20, 30}}, {{10, 1e7, 10, 20, 30}}}};
        for (const auto &sv : sets[bat]) {
        const int N = sv.N, r = sv.r, t = sv.t;
        const long long m = 1LL << sv.lgm, n = scaled(sv.n);
        std::vector<double> pr(m + 2, 0.0);
        for (long long j = 1; j <= m; ++j) pr[j] = 1.0 / m;
        for (int i = 2; i <= t; ++i)
            for (long long j = m; j >= 1; --j) pr[j] = pr[j + 1] + pr[j] / j;
        long long msup = 1;
        while (pr[msup] > 1e-15) ++msup;
        --msup;
        std::vector<double> prob(pr.begin(), pr.begin() + msup + 1);
        prob[0] = 0;
        double tail = 0;
        for (long long j = msup + 1; j <= m; ++j) tail += pr[j];
        prob[msup] += tail;  // the values above msup are counted at msup
        const auto t0 = std::chrono::steady_clock::now();
        Chi2 all{0, 0};
        for (int rep = 0; rep < N; ++rep) {
            const auto h = histogram(kn, detail::Savir2{r, t, m, msup}, g, ctr, n, static_cast<int>(msup + 1), t, 0);
            const Chi2 c = chi2(h, prob, n);
            all.x += c.x;
            all.df += c.df;
        }
        params = std::format("N={} n={} r={} m=2^{} t={}", N, n, r, sv.lgm, t);
        report("savir2", params, all, seconds_since(t0));
        if (sum.max_z > o.limit) return sum;
        }
    }
    if (wanted(o, "gcd")) {
        // GCD: the gcd of pairs of s-bit values against 6 / (pi^2 j^2), lumped at jmax, summed over N
        struct P { int N; double n; int r, s; };
        static const PerSuite<P> sets{{{}, {{1, 1e8, 0, 30}, {1, 4e7, 10, 20}}, {{10, 5e7, 0, 30}}}};
        for (const auto &sv : sets[bat]) {
        const int N = sv.N, r = sv.r, s = sv.s;
        const long long n = scaled(sv.n);
        const double c1 = 6 / (M_PI * M_PI);
        const int jmax = 1 + static_cast<int>(std::sqrt(c1 * n / kMinExpected));
        std::vector<double> prob(jmax + 1, 0.0);
        double acc = 0;
        for (int j = 1; j < jmax; ++j) acc += prob[j] = c1 / (static_cast<double>(j) * j);
        prob[jmax] = 1 - acc;
        const auto t0 = std::chrono::steady_clock::now();
        Chi2 all{0, 0};
        for (int rep = 0; rep < N; ++rep) {
            // a pair with a zero is drawn again: the pairs per observation are geometric of the probability q of no zero
            const double q = std::pow(1 - std::ldexp(1.0, -s), 2);
            const auto h = histogram(kn, detail::Gcd{r, s, jmax}, g, ctr, n, jmax + 1, 2 / q, 4 * (1 - q) / (q * q));
            const Chi2 c = chi2(h, prob, n);
            all.x += c.x;
            all.df += c.df;
        }
        params = std::format("N={} n={} r={} s={}", N, n, r, s);
        report("gcd", params, all, seconds_since(t0));
        if (sum.max_z > o.limit) return sum;
        }
    }
    // the tests on the bits of the Hamming tests (sstring): blocks of L bits of ldivs values of s bits and
    // the first lmods bits of one more
    const auto hamming = [](int r, int s, int L, int mode, int h) { return detail::Hamming{r, s, L / s, L % s, L, mode, h}; };
    const auto vals_per_block = [](int s, int L) { return L / s + (L % s > 0); };
    const auto report_normal = [&](const char *name, const std::string &params, double z, double sec) {
        const double p = fbar_Normal1(z);
        emit(name, params, Stat::kNormal, z, p, std::fabs(z), sec);
    };
    const auto binomial = [](int L, double p) {
        std::vector<double> b(L + 1);
        for (int i = 0; i <= L; ++i) b[i] = std::exp(std::lgamma(L + 1.0) - std::lgamma(i + 1.0) - std::lgamma(L - i + 1.0) + i * std::log(p) + (L - i) * std::log1p(-p));
        return b;
    };
    if (wanted(o, "hamweight")) {
        // HammingWeight2 (N = 10, n = 1e9 bits, s = 3, L = 1e6): sum over the K = n / L blocks of the N
        // replications of (weight - L/2)^2 * 4 / L, a chi-square of N K degrees of freedom; the weights are
        // counted within L/2 +- h (12 standard deviations), the farther ones at the ends
        struct P { int N; double n; int r, s, L; };  // n bits
        static const PerSuite<P> sets{{{}, {{100, 1e8, 0, 30, 1000000}, {30, 1e8, 20, 10, 1000000}}, {{10, 1e9, 0, 3, 1000000}, {10, 1e9, 27, 3, 1000000}}}};
        for (const auto &sv : sets[bat]) {
            const int N = sv.N, r = sv.r, s = sv.s, L = sv.L, h = static_cast<int>(std::ceil(6 * std::sqrt(static_cast<double>(L))));
            const long long K = static_cast<long long>(scaled(sv.n) / L);
            const auto t0 = std::chrono::steady_clock::now();
            const auto hist = histogram(kn, hamming(r, s, L, 2, h), g, ctr, N * K, 2 * h + 1, vals_per_block(s, L), 0);
            double x = 0;
            for (int c = 0; c <= 2 * h; ++c) x += hist[c] * static_cast<double>(c - h) * (c - h);
            params = std::format("N={} n={} r={} s={} L={}", N, K * L, r, s, L);
            report("hamweight", params, Chi2{x * 4.0 / L, N * K}, seconds_since(t0));
            if (sum.max_z > o.limit) return sum;
        }
    }
    if (wanted(o, "hamcorr")) {
        // HammingCorr (N = 1, r = 10, s = 10): the correlation of the weights of consecutive blocks,
        // sum (2 X_i - L)(2 X_{i+1} - L) / (L sqrt(pairs)), a standard normal
        struct P { double n; int r, s, L; };  // N = 1
        static const PerSuite<P> sets{{{}, {{5e8, 0, 30, 30}, {5e7, 0, 30, 300}, {1e7, 0, 30, 1200}}, {{1e9, 10, 10, 30}, {1e8, 10, 10, 300}, {1e8, 10, 10, 1200}}}};
        for (const auto &c : sets[bat]) {
            const int r = c.r, s = c.s, L = c.L;
            const long long n = scaled(c.n);
            const auto t0 = std::chrono::steady_clock::now();
            // (a thread's first block starts its pairs: one block more per thread)
            const auto hist = histogram(kn, hamming(r, s, L, 0, 0), g, ctr, n, (L + 1) * (L + 1), vals_per_block(s, L), 0, vals_per_block(s, L));
            double acc = 0, pairs = 0;
            for (int i = 0; i <= L; ++i)
                for (int j = 0; j <= L; ++j) {
                    const double m = static_cast<double>(hist[i * (L + 1) + j]);
                    acc += m * (2.0 * i - L) * (2.0 * j - L);
                    pairs += m;
                }
            params = std::format("n={} r={} s={} L={}", n, r, s, L);
            report_normal("hamcorr", params, acc / (L * std::sqrt(pairs)), seconds_since(t0));
            if (sum.max_z > o.limit) return sum;
        }
    }
    if (wanted(o, "hamindep")) {
        // HammingIndep (d = 0): the chi-square of the (L + 1)^2 pairs of the weights of non-overlapping
        // pairs of blocks against the binomial products, the cells of an expected count of at least 10 as
        // classes and the others merged into one (into the last class if below 10), summed over N
        struct P { int N; double n; int r, s, L; };
        static const PerSuite<P> sets{{{{1, 5e5, 20, 10, 300}}, {{1, 3e8, 0, 30, 30}, {1, 1e8, 20, 10, 30}, {1, 3e7, 0, 30, 300}, {1, 1e7, 20, 10, 300}, {1, 1e7, 0, 30, 1200}, {1, 1e6, 20, 10, 1200}}, {{10, 3e7, 0, 3, 30}, {10, 3e7, 27, 3, 30}, {1, 3e7, 0, 4, 300}, {1, 3e7, 26, 4, 300}, {1, 1e7, 0, 5, 1200}, {1, 1e7, 25, 5, 1200}}}};
        for (const auto &c : sets[bat]) {
            const long long n = scaled(c.n);
            const int L = c.L;
            const auto b = binomial(L, 0.5);
            const auto t0 = std::chrono::steady_clock::now();
            Chi2 all{0, 0};
            for (int rep = 0; rep < c.N; ++rep) {
                const auto hist = histogram(kn, hamming(c.r, c.s, L, 1, 0), g, ctr, n, (L + 1) * (L + 1), 2 * vals_per_block(c.s, L), 0);
                std::vector<double> eo, oo;
                double ze = 0, zo = 0;
                for (int i = 0; i <= L; ++i)
                    for (int j = 0; j <= L; ++j) {
                        const double ex = static_cast<double>(n) * b[i] * b[j], ob = static_cast<double>(hist[i * (L + 1) + j]);
                        if (ex >= kMinExpected) {
                            eo.push_back(ex);
                            oo.push_back(ob);
                        } else {
                            ze += ex;
                            zo += ob;
                        }
                    }
                if (ze >= kMinExpected || eo.empty()) {
                    eo.push_back(ze);
                    oo.push_back(zo);
                } else {
                    eo.back() += ze;
                    oo.back() += zo;
                }
                for (size_t i = 0; i < eo.size(); ++i) all.x += (oo[i] - eo[i]) * (oo[i] - eo[i]) / eo[i];
                all.df += static_cast<long>(eo.size()) - 1;
            }
            params = std::format("N={} n={} r={} s={} L={}", c.N, n, c.r, c.s, L);
            report("hamindep", params, all, seconds_since(t0));
            if (sum.max_z > o.limit) return sum;
        }
    }
    if (wanted(o, "bitrun")) {
        // the runs of bits (sstring_Run, N = 1, n = 2e9 runs, s = 3): the lengths of the runs of 0s and of 1s
        // (lumped at k) against 2^-j, a chi-square of 2(k - 1) degrees of freedom as TestU01 computes it, for
        // the bits of 4n / s values (TestU01 takes the bits until n runs of 1s); and the number of runs of 1s
        // against its mean B / 4 and variance B / 16 for B bits (the starts of the runs of 1s are pairwise
        // independent but for adjacent bits; TestU01 takes the bits for n runs, of the variance 8n instead of 4n)
        unsigned long long *d_cnt;
        CK(gpuMalloc(reinterpret_cast<void **>(&d_cnt), sizeof(unsigned long long) * 80));
        struct P { double n; int r, s; };  // N = 1, n runs
        static const PerSuite<P> sets{{{}, {{1e9, 0, 30}, {1e9, 20, 10}}, {{2e9, 0, 3}, {2e9, 27, 3}}}};
        for (const auto &sv : sets[bat]) {
            const int r = sv.r, s = sv.s;
            const long long nr = scaled(sv.n), nv = (4 * nr + s - 1) / s;
            const int k = 1 + static_cast<int>(std::log2(nr / kMinExpected));
            if (k > 40) throw TestFailure("bitrun: too many classes");
            const auto t0 = std::chrono::steady_clock::now();
            CK(gpuMemset(d_cnt, 0, sizeof(unsigned long long) * 80));
            const uint64_t start = ctr.take((nv + Counters::kThreadsPerLaunch - 1) / Counters::kThreadsPerLaunch);
            launch(kn.bitrun, kBlocks, kThreads, 0, g, start, nv, r, s, k, d_cnt);
            std::vector<unsigned long long> c(2 * k);
            CK(gpuMemcpy(c.data(), d_cnt, sizeof(unsigned long long) * 2 * k, gpuMemcpyDeviceToHost));
            double x = 0, runs1 = 0;
            for (int b = 0; b < 2; ++b) {
                double tot = 0;
                for (int j = 0; j < k; ++j) tot += c[b * k + j];
                if (b == 1) runs1 = tot;
                for (int j = 1; j <= k; ++j) {
                    const double pj = std::ldexp(1.0, -std::min(j, k - 1)), ex = tot * pj, d = c[b * k + j - 1] - ex;
                    x += d * d / (ex * (1 - pj));
                }
            }
            const double bits = static_cast<double>(nv) * s;
            params = std::format("n={} r={} s={} runs", nr, r, s);
            report("bitrun", params, Chi2{x, 2L * (k - 1)}, seconds_since(t0));
            params = std::format("n={} r={} s={} count", nr, r, s);
            report_normal("bitrun", params, (bits / 4 - runs1) / std::sqrt(bits / 16), seconds_since(t0));
            if (sum.max_z > o.limit) {
                CK(gpuFree(d_cnt));
                return sum;
            }
        }
        CK(gpuFree(d_cnt));
    }
    if (wanted(o, "longhead")) {
        // LongestHeadRun (N = 1, n = 1000 blocks, s = 3, L = 1e7 + 20 bits): the longest run of 1s in every
        // block against the asymptotic distribution of TestU01 (a chi-square), and the longest of all
        // against that of n L bits (a discrete p-value)
        const auto plhr = [](long j, double lnl) {
            double t = (j + 1) * M_LN2 - lnl;
            const double x = std::exp(-std::exp(-t));
            t += M_LN2;
            return std::exp(-std::exp(-t)) - x;
        };
        long long *d_long;
        struct P { double n; int r, s; long long L; };  // N = 1, n blocks of L bits
        static const PerSuite<P> sets{{{}, {{1000, 0, 30, 10000020}, {300, 20, 10, 10000020}}, {{1000, 0, 3, 10000020}, {1000, 27, 3, 10000020}}}};
        for (const auto &sv : sets[bat]) {
            const int r = sv.r, s = sv.s;
            const long long nb = scaled(sv.n), K = sv.L / s, L = K * s;
            std::vector<double> prob;
            double cdf = 0;
            for (long j = 0; j < 1000; ++j) {
                const double pj = plhr(j, std::log(static_cast<double>(L)));
                prob.push_back(pj);
                cdf += pj;
                if (pj <= DBL_EPSILON && cdf > 0.5) break;
            }
            prob.back() = 1 - (cdf - prob.back());
            const int jhigh = static_cast<int>(prob.size()) - 1;
            const auto t0 = std::chrono::steady_clock::now();
            CK(gpuMalloc(reinterpret_cast<void **>(&d_long), sizeof(long long) * nb));
            const uint64_t start = ctr.take((nb * K + Counters::kThreadsPerLaunch - 1) / Counters::kThreadsPerLaunch);
            launch(kn.longhead, static_cast<unsigned>(nb), kThreads, 0, g, start, K, r, s, d_long);
            std::vector<long long> lg(nb);
            CK(gpuMemcpy(lg.data(), d_long, sizeof(long long) * nb, gpuMemcpyDeviceToHost));
            CK(gpuFree(d_long));
            std::vector<unsigned long long> h(jhigh + 1, 0);
            long long top = 0;
            for (const long long x : lg) {
                ++h[std::min<long long>(x, jhigh)];
                top = std::max(top, x);
            }
            params = std::format("n={} r={} s={} L={}", nb, r, s, L);
            report("longhead", params, chi2(h, prob, nb), seconds_since(t0));
            const double lnall = std::log(static_cast<double>(nb) * L);
            double cl = 0, cprev = 0;
            for (long j = 0; j <= top; ++j) {
                cprev = cl;
                cl += plhr(j, lnall);
            }
            const double p = gofw_pDisc(cl, 1 - cprev);
            params = std::format("n={} r={} s={} longest={}", nb, r, s, top);
            emit("longhead", params, Stat::kDiscrete, static_cast<double>(top), p, 0, seconds_since(t0));
            if (sum.max_z > o.limit) return sum;
        }
    }
    if (wanted(o, "autocor")) {
        // AutoCor: the differing bits A at the distance d over K = (n - d) / s values of s bits, 2 (A - K s / 2) / sqrt(K s)
        // a standard normal, summed over N (divided by sqrt(N))
        unsigned long long *d_cnt;
        CK(gpuMalloc(reinterpret_cast<void **>(&d_cnt), sizeof(unsigned long long)));
        struct P { int N; double n; int r, s, d; };  // n bits
        static const PerSuite<P> sets{{{}, {{10, 1e9 + 30, 0, 30, 1}, {5, 1e9 + 1, 20, 10, 1}, {10, 1e9 + 31, 0, 30, 30}, {5, 1e9 + 11, 20, 10, 10}}, {{10, 1e9 + 30, 0, 3, 1}, {10, 1e9 + 30, 0, 3, 3}, {10, 1e9 + 30, 27, 3, 1}, {10, 1e9 + 30, 27, 3, 3}}}};
        for (const auto &c : sets[bat]) {
            const int N = c.N, s = c.s;
            if (s < 1 || s > 32 || c.r < 0 || c.r + s > 32 || c.d < 1 || c.d > s) throw std::invalid_argument("autocor: 1 <= d <= s <= 32, r + s <= 32");
            const long long nbits = scaled(c.n), K = (nbits - c.d) / s;
            // as TestU01: the replications one after the other, of M - 1 + K values each (M = d / s + 2); TestU01 cuts n
            // to d + K s, so its statistic takes the K s pairs of bits compared
            const long long per = K + c.d / s + 1;
            const uint64_t start0 = ctr.take((N * per + Counters::kThreadsPerLaunch - 1) / Counters::kThreadsPerLaunch);
            const auto t0 = std::chrono::steady_clock::now();
            std::vector<double> zs;
            for (int rep = 0; rep < N; ++rep) {
                CK(gpuMemset(d_cnt, 0, sizeof(unsigned long long)));
                const uint64_t start = start0 + static_cast<uint64_t>(rep) * per;
                launch(kn.autocor, kBlocks, kThreads, 0, g, start, K, c.r, s, c.d, d_cnt);
                unsigned long long A;
                CK(gpuMemcpy(&A, d_cnt, sizeof A, gpuMemcpyDeviceToHost));
                const double m = static_cast<double>(K) * s;
                zs.push_back(2.0 * (A - m / 2) / std::sqrt(m));
            }
            params = std::format("N={} n={} r={} s={} d={}", N, nbits, c.r, s, c.d);
            report_normal("autocor", params, sum_stat(zs) / std::sqrt(static_cast<double>(N)), seconds_since(t0));
            if (sum.max_z > o.limit) {
                CK(gpuFree(d_cnt));
                return sum;
            }
        }
        CK(gpuFree(d_cnt));
    }
    if (wanted(o, "samplecorr")) {
        // SampleCorr (N = 1, n = 2e9): the sum of u_{i-k} u_i - 1/4 over n - k terms, of the variance 13/144 per
        // term (7/144, and 1/48 with each of the two terms sharing a value; TestU01 takes 1/12, so that its
        // statistic has the variance 13/12)
        double *d_sum;
        CK(gpuMalloc(reinterpret_cast<void **>(&d_sum), sizeof(double)));
        struct P { double n; int k; };  // N = 1, r = 0
        static const PerSuite<P> sets{{{}, {{5e8, 1}}, {{2e9, 1}, {2e9, 2}}}};
        for (const auto &sv : sets[bat]) {
            const int k = sv.k;
            const long long n = scaled(sv.n);
            const auto t0 = std::chrono::steady_clock::now();
            CK(gpuMemset(d_sum, 0, sizeof(double)));
            const uint64_t start = ctr.take((n + Counters::kThreadsPerLaunch - 1) / Counters::kThreadsPerLaunch);
            launch(kn.samplecorr, kBlocks, kThreads, 0, g, start, n, 0, k, d_sum);
            double sm;
            CK(gpuMemcpy(&sm, d_sum, sizeof sm, gpuMemcpyDeviceToHost));
            params = std::format("n={} r=0 k={}", n, k);
            report_normal("samplecorr", params, sm * 12.0 / std::sqrt(13.0 * (n - k)), seconds_since(t0));
            if (sum.max_z > o.limit) {
                CK(gpuFree(d_sum));
                return sum;
            }
        }
        CK(gpuFree(d_sum));
    }
    if (wanted(o, "sampleprod")) {
        // SampleProd (r = 0): for each of N replications, the AD test of the n products of t values against FDistProd;
        // then the AD test of the N p-values, or for N = 1 that of the replication (as TestU01 reports it)
        struct P { int N; double n; int t; };  // r = 0
        static const PerSuite<P> sets{{{}, {{1, 1e7, 10}, {1, 1e7, 30}}, {{40, 1e7, 8}, {20, 1e7, 16}, {20, 1e7, 24}}}};
        for (const auto &sv : sets[bat]) {
            const int c[2] = {sv.N, sv.t};
            const long long n = scaled(sv.n);
            const auto t0 = std::chrono::steady_clock::now();
            std::vector<double> ps;
            std::pair<double, double> one{0, 0};  // the statistic and the p-value of the last replication
            for (int rep = 0; rep < c[0]; ++rep) {
                const uint64_t stride = Counters::stride(n, c[1], 0);
                const uint64_t base = ctr.take(stride);
                one = ad_sorted(n, [&](const Sink &sk) {
                    launch(kn.uniform, kBlocks, kThreads, 0, 0, 0, c[1], g, base, stride, n, static_cast<const double *>(nullptr), 0, 0.0, sk);
                });
                ps.push_back(one.second);
            }
            const auto ad = c[0] == 1 ? one : anderson_darling(ps);
            params = std::format("N={} n={} r=0 t={}", c[0], n, c[1]);
            report_ad("sprod", params, ad.first, ad.second, seconds_since(t0));
            if (sum.max_z > o.limit) return sum;
        }
    }
    if (wanted(o, "samplemean")) {
        // SampleMean (N = 2e7 sums of n = 30 values, r = 0 and 10): the AD test of the N sums against the
        // Irwin-Hall distribution (by a table of it and its density, see kuniform)
        struct P { double N; int m, r; };  // N sums of m values
        static const PerSuite<P> sets{{{}, {{1e7, 20, 0}}, {{2e7, 30, 0}, {2e7, 30, 10}}}};
        for (const auto &sv : sets[bat]) {
        const int m = sv.m, ntab = 1 << 16, r = sv.r;
        const double hstep = static_cast<double>(m) / (ntab - 1);
        std::vector<double> tab(2 * ntab);
        {
            const auto ih = [m](long double x, int deg) {  // sum_k (-1)^k C(m, k) (x - k)^deg / deg!, x <= m / 2
                long double s = 0, c = 1, f = 1;
                for (int i = 2; i <= deg; ++i) f *= i;
                for (int k = 0; k <= m && k <= x; ++k) {
                    s += (k & 1 ? -c : c) * std::pow(x - k, static_cast<long double>(deg));
                    c = c * (m - k) / (k + 1);
                }
                return s / f;
            };
            for (int i = 0; i < ntab; ++i) {
                const long double x = static_cast<long double>(i) * m / (ntab - 1);
                const bool low = x <= m / 2.0L;
                const long double y = low ? x : m - x;
                tab[2 * i] = static_cast<double>(low ? ih(y, m) : 1 - ih(y, m));
                tab[2 * i + 1] = static_cast<double>(ih(y, m - 1));
            }
        }
        double *d_tab;
        CK(gpuMalloc(reinterpret_cast<void **>(&d_tab), sizeof(double) * tab.size()));
        CK(gpuMemcpy(d_tab, tab.data(), sizeof(double) * tab.size(), gpuMemcpyHostToDevice));
        {
            const long long n = scaled(sv.N);
            const auto t0 = std::chrono::steady_clock::now();
            const uint64_t stride = Counters::stride(n, m, 0);
            const uint64_t base = ctr.take(stride);
            const auto ad = ad_sorted(n, [&](const Sink &sk) {
                launch(kn.uniform, kBlocks, kThreads, 0, 1, r, m, g, base, stride, n, static_cast<const double *>(d_tab), ntab, hstep, sk);
            });
            params = std::format("N={} n={} r={}", n, m, r);
            report_ad("smean", params, ad.first, ad.second, seconds_since(t0));
            if (sum.max_z > o.limit) {
                CK(gpuFree(d_tab));
                return sum;
            }
        }
        CK(gpuFree(d_tab));
        }
    }
    if (wanted(o, "walk")) {
        // RandomWalk1 (N = 1, L0 = L1 = L): the chi-squares of the statistics H, M, J, R and C of n walks
        // against the expected numbers of TestU01 (CalcNbExp of swalk.c)
        struct P { double n; int r, s, L; };
        static const PerSuite<P> sets{{{{1e6, 0, 30, 150}}, {{5e7, 0, 30, 90}, {1e7, 20, 10, 90}, {5e6, 0, 30, 1000}, {1e6, 20, 10, 1000}, {5e5, 0, 30, 10000}, {1e5, 20, 10, 10000}}, {{1e8, 0, 5, 50}, {1e8, 25, 5, 50}, {1e7, 0, 10, 1000}, {1e7, 20, 10, 1000}, {1e6, 0, 15, 10000}, {1e6, 15, 15, 10000}}}};
        for (const auto &c : sets[bat]) {
            const long long n = scaled(c.n);
            const long L = c.L, L2 = L / 2;
            const double nl = static_cast<double>(n), epsn = 1e-16 * nl;
            const auto bin = [](long m, long k) { return std::exp(std::lgamma(m + 1.0) - std::lgamma(k + 1.0) - std::lgamma(m - k + 1.0) - m * M_LN2); };
            std::vector<std::vector<double>> ex(5, std::vector<double>(L + 1, 0.0));
            auto &H = ex[0], &M = ex[1], &J = ex[2], &R = ex[3], &C = ex[4];
            H[L2] = nl * bin(L, L2);
            for (long i = L2; i > 0 && H[i] > epsn; --i) H[i - 1] = H[i] * i / (L - i + 1);
            for (long i = L2; i < L && H[i] > epsn; ++i) H[i + 1] = H[i] * (L - i) / (i + 1);
            M[0] = H[L2];
            for (long i = 0; i < L && M[i] > epsn; i += 2) M[i + 2] = M[i + 1] = M[i] * ((L - i) / 2) / ((L + i) / 2 + 1);
            J[0] = J[L] = M[0];
            for (long i = 0; i < L2 && J[i] > epsn; i += 2) J[L - i - 2] = J[i + 2] = J[i] * ((L - i) / 2) * (1 + i) / (static_cast<double>(i / 2 + 1) * (L - i - 1));
            R[0] = J[0];
            for (long i = 0; i < L2 && R[i] > epsn; ++i) R[i + 1] = R[i] * (L - 2 * i) / (L - i);
            C[0] = 2.0 * nl * bin(L - 1, L2);
            for (long i = 0; i < L2 - 1 && C[i] > epsn; ++i) C[i + 1] = C[i] * (L2 - i - 1) / (L2 + i + 1);
            const auto t0 = std::chrono::steady_clock::now();
            const int nb = 5 * (L + 1), lds = nb <= kMaxLdsBins;
            unsigned long long *d_hist;
            CK(gpuMalloc(reinterpret_cast<void **>(&d_hist), sizeof(unsigned long long) * nb));
            CK(gpuMemset(d_hist, 0, sizeof(unsigned long long) * nb));
            const int per = L / c.s + (L % c.s > 0);
            const uint64_t stride = Counters::stride(n, per, 0);
            launch(kn.walk, kBlocks, kThreads, lds ? sizeof(unsigned) * nb : 0, g, ctr.take(stride), stride, n, c.r, c.s, c.L, d_hist, lds);
            std::vector<unsigned long long> h(nb);
            CK(gpuMemcpy(h.data(), d_hist, sizeof(unsigned long long) * nb, gpuMemcpyDeviceToHost));
            CK(gpuFree(d_hist));
            const double sec = seconds_since(t0);
            constexpr std::string_view names = "HMJRC";
            for (int k = 0; k < 5; ++k) {
                std::vector<unsigned long long> obs(h.begin() + k * (L + 1), h.begin() + (k + 1) * (L + 1));
                std::vector<double> prob(L + 1);
                for (long i = 0; i <= L; ++i) prob[i] = ex[k][i] / nl;
                params = std::format("n={} r={} s={} L={} {}", n, c.r, c.s, c.L, names[k]);
                report("walk", params, chi2(obs, prob, n), sec);
            }
            if (sum.max_z > o.limit) return sum;
        }
    }
    if (wanted(o, "periods")) {
        // PeriodsInStrings (N = 10, n = 5e8, s = 10): the correlations (the periods) of strings of s bits
        // against their populations among the 2^s strings, summed over N
        struct P { int N; double n; int r, s; };
        static const PerSuite<P> sets{{{}, {{1, 3e8, 0, 30}, {1, 3e8, 15, 15}}, {{10, 5e8, 0, 10}, {10, 5e8, 20, 10}}}};
        for (const auto &sv : sets[bat]) {
        const int N = sv.N, sb = sv.s, r = sv.r;
        if (sb < 2 || sb > 32 || r < 0 || r + sb > 32) throw std::invalid_argument("periods: s must be in [2, 32], and r + s <= 32");
        if (sb > 16) {
            // the classes of the correlations of TestU01 (in its order), found on GPU by a binary search of the correlations
            const auto corr = correlations(sb);
            const int ncls = static_cast<int>(corr.size());
            std::vector<std::pair<uint32_t, int>> sorted;
            std::vector<double> prob;
            for (int i = 0; i < ncls; ++i) {
                sorted.push_back({corr[i].first, i});
                prob.push_back(std::ldexp(static_cast<double>(corr[i].second), -sb));
            }
            std::sort(sorted.begin(), sorted.end());
            std::vector<uint32_t> keys;
            std::vector<int> cl;
            for (const auto &[k, i] : sorted) {
                keys.push_back(k);
                cl.push_back(i);
            }
            uint32_t *d_keys;
            int *d_cls;
            CK(gpuMalloc(reinterpret_cast<void **>(&d_keys), sizeof(uint32_t) * ncls));
            CK(gpuMalloc(reinterpret_cast<void **>(&d_cls), sizeof(int) * ncls));
            CK(gpuMemcpy(d_keys, keys.data(), sizeof(uint32_t) * ncls, gpuMemcpyHostToDevice));
            CK(gpuMemcpy(d_cls, cl.data(), sizeof(int) * ncls, gpuMemcpyHostToDevice));
            const long long n = scaled(sv.n);
            const auto t0 = std::chrono::steady_clock::now();
            Chi2 all{0, 0};
            try {
                for (int rep = 0; rep < N; ++rep) {
                    const Chi2 c = chi2(histogram(kn, detail::Periods{r, sb, ncls, d_keys, d_cls}, g, ctr, n, ncls, 1, 0), prob, n);
                    all.x += c.x;
                    all.df += c.df;
                }
            } catch (...) {
                CK(gpuFree(d_keys));
                CK(gpuFree(d_cls));
                throw;
            }
            CK(gpuFree(d_keys));
            CK(gpuFree(d_cls));
            params = std::format("N={} n={} r={} s={}", N, n, r, sb);
            report("periods", params, all, seconds_since(t0));
            if (sum.max_z > o.limit) return sum;
            continue;
        }
        std::map<unsigned, int> cls;  // the correlation of every string -> its class
        std::vector<int> of(1 << sb);
        for (unsigned v = 0; v < (1u << sb); ++v) {
            unsigned gg = v, d = v, c = 1, lbit = sb - 1;
            for (int k = 1; k < sb; ++k) {
                gg &= ~(1u << lbit);
                d >>= 1;
                if (gg == d) c |= 1u << k;
                --lbit;
            }
            of[v] = static_cast<int>(cls.emplace(c, static_cast<int>(cls.size())).first->second);
        }
        std::vector<double> prob(cls.size(), 0.0);
        for (unsigned v = 0; v < (1u << sb); ++v) prob[of[v]] += 1.0 / (1 << sb);
        {
            const long long n = scaled(sv.n);
            const auto t0 = std::chrono::steady_clock::now();
            Chi2 all{0, 0};
            for (int rep = 0; rep < N; ++rep) {
                const auto hs = histogram(kn, detail::Strip{r, sb}, g, ctr, n, 1 << sb, 1, 0);
                std::vector<unsigned long long> h(cls.size(), 0);
                for (unsigned v = 0; v < (1u << sb); ++v) h[of[v]] += hs[v];
                const Chi2 c = chi2(h, prob, n);
                all.x += c.x;
                all.df += c.df;
            }
            params = std::format("N={} n={} r={} s={}", N, n, r, sb);
            report("periods", params, all, seconds_since(t0));
            if (sum.max_z > o.limit) return sum;
        }
        }
    }
    if (wanted(o, "sumcoll")) {
        // SumCollector (N = 1, n = 5e8, r = 0, g = 10): the number of values until their sum exceeds g, against
        // P(N = j + 1) = F_j(g) - F_{j+1}(g) of the Irwin-Hall distributions F (in long double), lumped at 50
        struct P { double n; int r, g; };  // N = 1
        static const PerSuite<P> sets{{{}, {{2e7, 0, 10}}, {{5e8, 0, 10}}}};
        for (const auto &sv : sets[bat]) {
        const int jmax = 50, g10 = sv.g;
        const auto ihcdf = [](int m, int x) {  // P(sum of m uniforms <= x) for an integer x <= m
            long double s = 0, c = 1, f = 1;
            for (int i = 2; i <= m; ++i) f *= i;
            for (int k = 0; k <= x; ++k) {
                s += (k & 1 ? -c : c) * std::pow(static_cast<long double>(x - k), static_cast<long double>(m));
                c = c * (m - k) / (k + 1);
            }
            return s / f;
        };
        std::vector<double> prob(jmax + 1, 0.0);
        double acc = 0;
        for (int j = g10; j < jmax; ++j) acc += prob[j] = static_cast<double>(ihcdf(j, g10) - ihcdf(j + 1, g10));
        prob[jmax] = std::max(0.0, 1 - acc);
        const long long n = scaled(sv.n);
        const auto t0 = std::chrono::steady_clock::now();
        const auto h = histogram(kn, detail::SumColl{sv.r, jmax, static_cast<double>(g10)}, g, ctr, n, jmax + 1, 21, 0.4 * 21);
        params = std::format("n={} r={} g={}", n, sv.r, g10);
        report("sumcoll", params, chi2(h, prob, n), seconds_since(t0));
        if (sum.max_z > o.limit) return sum;
        }
    }
    if (wanted(o, "fourier")) {
        // Fourier3 (N = 1e5, k = 14, s = 3, r = 0 and 27): for each of N sequences of n = 2^14 bits (1 and -1)
        // from n / s + 1 values, the squares |F_i|^2 of its Fourier coefficients for i = 1..n/4, summed over
        // the N sequences; the sums divided by n have the mean N and the variance N (1 - 2 / n) (Erdmann), and
        // the AD test of the n / 4 standardized sums against the standard normal distribution
        struct P { double N; int k, r, s; };  // N sequences of 2^k bits
        static const PerSuite<P> sets{{{}, {{5e4, 14, 0, 30}, {5e4, 14, 20, 10}}, {{1e5, 14, 0, 3}, {1e5, 14, 27, 3}}}};
        for (const auto &sv : sets[bat]) {
        const int k = sv.k, sb = sv.s, n = 1 << k, kall = n / sb + 1, r = sv.r;
        const long long N = scaled(sv.N);
        // the batch of the FFTs within a quarter of the budget (about 24 n bytes per sequence: its bits,
        // coefficients and the work buffer of rocFFT or cuFFT), a power of 2 in [64, 4096]
        long long batch = 64;
        while (batch < 4096 && static_cast<double>(batch) * 2 * 24 * n <= mem.budget / 4.0) batch *= 2;
        if (!mem.fft || mem.fft->n != n || mem.fft->batch != batch) {
            mem.fft.reset();
            mem.fft = std::make_unique<Fft>(n, batch);
        }
        Fft &ft = *mem.fft;
        {
            const auto t0 = std::chrono::steady_clock::now();
            CK(gpuMemset(ft.b, 0, sizeof(double) * (n / 4 + 1)));
            const uint64_t start = ctr.take((N * kall + Counters::kThreadsPerLaunch - 1) / Counters::kThreadsPerLaunch);
            for (long long done = 0; done < N; done += batch) {
                const long long nb = std::min(batch, N - done);
                launch(kn.fourier_bits, kBlocks, kThreads, 0, g, start + static_cast<uint64_t>(done) * kall, nb, kall, n, r, sb, ft.a);
                ft.execute();
                launch(kn.fourier_acc, (n / 4 + 255) / 256, 256, 0, static_cast<const double *>(ft.f), nb, n, ft.b);
            }
            std::vector<double> b(n / 4 + 1), u(n / 4);
            CK(gpuMemcpy(b.data(), ft.b, sizeof(double) * b.size(), gpuMemcpyDeviceToHost));
            for (int i = 1; i <= n / 4; ++i) u[i - 1] = fdist_Normal2((b[i] / n - N) / std::sqrt(N * (1.0 - 2.0 / n)));
            const auto ad = anderson_darling(u);
            params = std::format("N={} n=2^{} r={} s={}", N, k, r, sb);
            report_ad("fourier", params, ad.first, ad.second, seconds_since(t0));
            if (sum.max_z > o.limit) return sum;
        }
        }
    }
    if (wanted(o, "linearcomp")) {
        // LinearComp (N = 1, n = 400020 bits, s = 1, r = 0 and 29): the linear complexity profile of n bits
        // (the s bits (x << r) >> (32 - s) of n / s values, the most significant first) by the Berlekamp-
        // Massey algorithm on the host: the number of its jumps against the normal distribution of the mean mu
        // and the standard deviation sigma of TestU01, and the sizes of the jumps against 2^-k (k < M0, the
        // last class lumped), a chi-square of M0 - 1 degrees of freedom; summed over the N sequences. (The
        // chi-square of the linear complexity, for N > 20 in TestU01, is not computed.)
        struct P { double n; int s, r0, r1; };  // N = 1; the two instances r = r0, r1 of the battery
        static const PerSuite<P> sets{{{}, {{120000, 1, 0, 29}}, {{400020, 1, 0, 29}}}};
        for (const auto &sv : sets[bat]) {
        const int sb = sv.s;
        const long long N = 1;
        const long long K0 = scaled(sv.n) / sb, n = K0 * sb;
        const int par = static_cast<int>(n & 1);
        const double tmp = n >= DBL_MAX_EXP ? 0.0 : std::pow(2.0, -static_cast<double>(n));
        const double mu = n / 4.0 + (4 + par) / 12.0 - tmp / 3.0;
        const double sigma =
            std::sqrt(n / 8.0 - (2 - par) / (9.0 - par) + n * tmp / 6.0 + (6 + par) * tmp / 18.0 - tmp * tmp / 9.0);
        const int M0 = static_cast<int>(std::log2(mu / kMinExpected));
        if (M0 < 2) throw TestFailure("linearcomp: too few bits");
        // the bits of all the sequences from the GPU, then the Berlekamp-Massey algorithm of every sequence (its
        // time grows as n^2)
        const int rs[] = {sv.r0, sv.r1};
        struct Job {
            std::vector<uint8_t> bits;
            long L = 0, numj = 0;
            std::vector<long> cnt;
        };
        std::vector<Job> jobs(2 * N);
        uint32_t *d_out;
        CK(gpuMalloc(reinterpret_cast<void **>(&d_out), sizeof(uint32_t) * K0));
        std::vector<uint32_t> x(K0);
        const auto t0 = std::chrono::steady_clock::now();
        for (int ri = 0; ri < 2; ++ri) {
            const uint64_t start = ctr.take((N * K0 + Counters::kThreadsPerLaunch - 1) / Counters::kThreadsPerLaunch);
            for (long long q = 0; q < N; ++q) {
                launch(kn.cp_outputs, kBlocks, kThreads, 0, g, start + static_cast<uint64_t>(q * K0), K0, d_out);
                CK(gpuMemcpy(x.data(), d_out, sizeof(uint32_t) * K0, gpuMemcpyDeviceToHost));
                std::vector<uint8_t> &bits = jobs[ri * N + q].bits;
                bits.resize(n);
                for (long long i = 0; i < K0; ++i) {
                    const uint32_t v = (x[i] << rs[ri]) >> (32 - sb);
                    for (int j = 0; j < sb; ++j) bits[i * sb + j] = (v >> (sb - 1 - j)) & 1;
                }
            }
        }
        CK(gpuFree(d_out));
        for (Job &jb : jobs) berlekamp_massey(jb.bits, M0, jb.L, jb.numj, jb.cnt);
        const double sec = seconds_since(t0);
        for (int ri = 0; ri < 2; ++ri) {
            std::vector<double> zs, jx;
            Chi2 jl{0, 0};
            for (long long q = 0; q < N; ++q) {
                const Job &jb = jobs[ri * N + q];
                zs.push_back((jb.numj - mu) / sigma);
                double x = 0;
                for (int k = 1; k <= M0; ++k) {
                    const double e = jb.numj / std::ldexp(1.0, k < M0 ? k : M0 - 1);
                    x += (jb.cnt[k] - e) * (jb.cnt[k] - e) / e;
                }
                jx.push_back(x);
                jl.df += M0 - 1;
            }
            params = std::format("N={} n={} r={} s={} jumps", N, n, rs[ri], sb);
            report_normal("lincomp", params, sum_stat(zs) / std::sqrt(static_cast<double>(N)), sec);
            params = std::format("N={} n={} r={} s={} sizes", N, n, rs[ri], sb);
            jl.x = sum_stat(jx);
            report("lincomp", params, jl, sec);
            if (sum.max_z > o.limit) return sum;
        }
        }
    }
    if (wanted(o, "matrixrank")) {
        // MatrixRank (as BigCrush: N = 10, n = 1e6 matrices of 30 x 30 with s = 5, r = 0 and 25; N = 1, n = 5000 of
        // 1000 x 1000 with s = 4, r = 0 and 26; N = 1, n = 80 of 5000 x 5000, r = 15 with s = 15 and r = 0 with
        // s = 30): the ranks over GF(2) against their probabilities (TestU01, in log2), the chi-square summed
        // over N. The small matrices are made from the counters of every thread; the large ones in global memory
        // (a block each, batched within the memory budget) from consecutive values.
        struct P { int N; double n; int r, s, l, k; };
        static const PerSuite<P> sets{{{{1, 2e4, 20, 10, 60, 60}}, {{1, 1e6, 0, 30, 60, 60}, {1, 1e6, 20, 10, 60, 60}, {1, 5e4, 0, 30, 300, 300}, {1, 5e4, 20, 10, 300, 300}, {1, 2e3, 0, 30, 1200, 1200}, {1, 2e3, 20, 10, 1200, 1200}}, {{10, 1e6, 0, 5, 30, 30}, {10, 1e6, 25, 5, 30, 30}, {1, 5000, 0, 4, 1000, 1000}, {1, 5000, 26, 4, 1000, 1000}, {1, 80, 15, 15, 5000, 5000}, {1, 80, 0, 30, 5000, 5000}}}};
        for (const auto &c : sets[bat]) {
            const long long n = std::max<long long>(scaled(c.n), 1);
            const int mkl = std::min(c.l, c.k), vpr = c.k / c.s + (c.k % c.s > 0);
            // the expected numbers of the ranks, by the operations of TestU01 (num_Log2(x) = log(x) / ln 2, pow)
            const auto log2_ = [](double x) { return 1.44269504088896340737 * std::log(x); };
            std::vector<double> nbexp(mkl + 1);
            double t2 = log2_(static_cast<double>(n)) - c.l * c.k;
            nbexp[0] = std::pow(2.0, t2);
            for (int j = 1; j <= mkl; ++j) {
                t2 += c.l + c.k - 2 * j + 1 + log2_(1.0 - std::pow(2.0, -static_cast<double>(c.l - j + 1))) +
                      log2_(1.0 - std::pow(2.0, -static_cast<double>(c.k - j + 1))) - log2_(1.0 - std::pow(2.0, -static_cast<double>(j)));
                nbexp[j] = std::pow(2.0, t2);
            }
            const auto t0 = std::chrono::steady_clock::now();
            Chi2 all{0, 0};
            std::vector<double> xs;
            for (int rep = 0; rep < c.N; ++rep) {
                std::vector<unsigned long long> h(mkl + 1, 0);
                if (c.l <= 32 && c.k <= 32) {
                    h = histogram(kn, detail::MatRank{c.r, c.s, c.l, c.k}, g, ctr, n, mkl + 1, static_cast<double>(c.l) * vpr, 0);
                } else {
                    const int W = (c.k + 63) / 64;
                    const long long per = static_cast<long long>(c.l) * W;  // words per matrix
                    const long long batch = std::max<long long>(1, std::min<long long>(n, static_cast<long long>(mem.budget / 2 / 8 / per)));
                    const uint64_t start = ctr.take((n * c.l * vpr + Counters::kThreadsPerLaunch - 1) / Counters::kThreadsPerLaunch);
                    unsigned long long *d_m;
                    int *d_rank;
                    CK(gpuMalloc(reinterpret_cast<void **>(&d_m), sizeof(unsigned long long) * per * batch));
                    CK(gpuMalloc(reinterpret_cast<void **>(&d_rank), sizeof(int) * batch));
                    std::vector<int> rk(batch);
                    for (long long m0 = 0; m0 < n; m0 += batch) {
                        const long long nb = std::min(batch, n - m0);
                        launch(kn.mat_fill, kBlocks, kThreads, 0, g, start + static_cast<uint64_t>(m0 * c.l * vpr), nb * c.l, c.r, c.s, c.k, W, d_m);
                        launch(kn.mat_rank, static_cast<unsigned>(nb), kThreads, 0, d_m, c.l, c.k, W, d_rank);
                        CK(gpuMemcpy(rk.data(), d_rank, sizeof(int) * nb, gpuMemcpyDeviceToHost));
                        for (long long i = 0; i < nb; ++i) ++h[rk[i]];
                    }
                    CK(gpuFree(d_m));
                    CK(gpuFree(d_rank));
                }
                Chi2 x{0, 0};
                try {
                    x = chi2_gofs(h, nbexp);
                } catch (const TestFailure &) {  // too few matrices for a class (a small scale): no statistic, as TestU01
                }
                xs.push_back(x.x);
                all.df += x.df;
            }
            params = std::format("N={} n={} r={} s={} L={} k={}", c.N, n, c.r, c.s, c.l, c.k);
            all.x = sum_stat(xs);
            if (all.df >= 1) report("matrank", params, all, seconds_since(t0));  // (else too few matrices for two classes)
            if (sum.max_z > o.limit) return sum;
        }
    }
    if (wanted(o, "appearance")) {
        // AppearanceSpacings (N = 1, Q = 1e7, K = 1e9, s = 3, L = 15, r = 0 and 27): the blocks of L bits, and for each
        // of the K blocks after the Q of the initialization, log of the distance to the most recent occurrence of the
        // same block (from 0 for a block not seen yet); the sum against the mean and the variance of Maurer, with the
        // correction of Coron and Naccache (as TestU01). The keys (block, index) are sorted on GPU (by the parts of
        // their top bits, i.e. of the blocks, beyond the memory budget), so that the occurrences of a block are in
        // order; the terms are summed on GPU.
        static const double E[17] = {0, 0.73264948, 1.53743829, 2.40160681, 3.31122472, 4.25342659, 5.21770525, 6.19625065, 7.18366555,
                                     8.17642476, 9.17232431, 10.1700323, 11.1687649, 12.1680703, 13.1676926, 14.1674884, 15.1673788};
        static const double KV[17] = {0,       0.68977, 1.33774, 1.90133, 2.35774, 2.70455, 2.95403, 3.12539, 3.23866,
                                      3.31120, 3.35646, 3.38409, 3.40065, 3.41043, 3.41614, 3.41943, 3.42130};
        static const double dCor[17] = {0, 0, 0, 0.2732725, 0.3045101, 0.3296587, 0.3489769, 0.3631815, 0.3732189,
                                        0.3800637, 0.3845867, 0.3874942, 0.3893189, 0.3904405, 0.3911178, 0.3915202, 0.3917561};
        static const double eCor[17] = {0, 0, 0, 0.4890883, 0.4435381, 0.4137196, 0.3941338, 0.3813210, 0.3730195,
                                        0.3677118, 0.3643695, 0.3622979, 0.3610336, 0.3602731, 0.3598216, 0.3595571, 0.3594040};
        struct P { double Q, K; int r, s, L; };  // N = 1 (Q is not scaled)
        static const PerSuite<P> sets{{{}, {{1e7, 4e8, 0, 30, 15}, {1e7, 1e8, 20, 10, 15}}, {{1e7, 1e9, 0, 3, 15}, {1e7, 1e9, 27, 3, 15}}}};
        for (const auto &sv : sets[bat]) {
            const int sb = sv.s, L = sv.L, vpb = L / sb + (L % sb > 0), ibits = 64 - L, r = sv.r;
            const long long Q = static_cast<long long>(sv.Q), K = std::max<long long>(scaled(sv.K), 1), n = Q + K;
            // (as CalcSigma of TestU01: no correction for L < 3)
            const double sigma = L < 3 ? std::sqrt(KV[L] / K) : std::sqrt((dCor[L] + eCor[L] * std::ldexp(1.0, L) / K) * KV[L] / K);
            const auto t0 = std::chrono::steady_clock::now();
            // the values: vpb per block, or one per s / L blocks for s > L (as TestU01)
            const long long nvals = sb > L ? (n + sb / L - 1) / (sb / L) : n * vpb;
            const uint64_t base = ctr.take((nvals + Counters::kThreadsPerLaunch - 1) / Counters::kThreadsPerLaunch);
            double somme = 0;
            sorted_parts(kn, mem, n, 64, [&](const Sink &sk) { launch(kn.appear, kBlocks, kThreads, 0, g, base, n, r, sb, L, ibits, sk); },
                         [&](unsigned long long *a, long long c) {
                             somme += thrust::transform_reduce(thrust::counting_iterator<long long>(0), thrust::counting_iterator<long long>(c),
                                                               AppearTerm{a, ibits, Q}, 0.0, thrust::plus<double>());
                         });
            params = std::format("N=1 Q={} K={} r={} s={} L={}", Q, K, r, sb, L);
            report_normal("appear", params, (somme * (1.0 / (M_LN2 * K)) - E[L]) / sigma, seconds_since(t0));
            if (sum.max_z > o.limit) return sum;
        }
    }
    if (wanted(o, "lempelziv")) {
        // LempelZiv (N = 10, n = 2^27 bits, r = 0 with s = 30, and r = 15 with s = 15): the number of the words of
        // the LZ78 parsing of every sequence (on the host; the values made on GPU), against the empirical mean
        // and standard deviation of TestU01, summed over the N sequences. The tables of TestU01 end at 2^28 bits,
        // so the scale multiplies N.
        static const double LZMu[29] = {0, 0, 0, 4.44, 7.64, 12.5, 20.8, 34.8, 58.9, 101.1, 176.0, 310.0, 551.9, 992.3, 1799.,
                                        3286.2, 6041.5, 11171.5, 20761.8, 38760.4, 72654., 136677., 257949., 488257., 926658.,
                                        1762965., 3361490., 6422497., 12293930.};
        static const double LZSigma[29] = {0, 0, 0, 0.49, 0.51, 0.62, 0.75, 0.78, 0.86, 0.94, 1.03, 1.19, 1.43, 1.68, 2.09,
                                           2.46, 3.36, 4.2, 5.4, 6.8, 9.1, 10.9, 14.7, 19.1, 25.2, 33.5, 44.546, 58.194, 75.513};
        struct P { int N, k, r, s; };  // N sequences of 2^k bits (the scale multiplies N)
        static const PerSuite<P> sets{{{}, {{10, 25, 0, 30}}, {{10, 27, 0, 30}, {10, 27, 15, 15}}}};
        for (const auto &sv : sets[bat]) {
            const int t = sv.k, r = sv.r, sb = sv.s;
            const long long nbits = 1LL << t, N = std::max<long long>(1, std::llround(sv.N * o.scale));
            const long long nv = (nbits + sb - 1) / sb;
            const auto t0 = std::chrono::steady_clock::now();
            const uint64_t start = ctr.take((N * nv + Counters::kThreadsPerLaunch - 1) / Counters::kThreadsPerLaunch);
            uint32_t *d_out;
            CK(gpuMalloc(reinterpret_cast<void **>(&d_out), sizeof(uint32_t) * nv));
            std::vector<uint32_t> x(nv);
            std::vector<double> zs;
            for (long long q = 0; q < N; ++q) {
                launch(kn.cp_outputs, kBlocks, kThreads, 0, g, start + static_cast<uint64_t>(q * nv), nv, d_out);
                CK(gpuMemcpy(x.data(), d_out, sizeof(uint32_t) * nv, gpuMemcpyDeviceToHost));
                zs.push_back((lz78(x, nbits, r, sb) - LZMu[t]) / LZSigma[t]);
            }
            CK(gpuFree(d_out));
            params = std::format("N={} n=2^{} r={} s={}", N, t, r, sb);
            report_normal("lempelziv", params, sum_stat(zs) / std::sqrt(static_cast<double>(N)), seconds_since(t0));
            if (sum.max_z > o.limit) return sum;
        }
    }
    return sum;
}

}  // namespace testu01_gpu
