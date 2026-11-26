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
// sorted on GPU by rocThrust) are counted exactly by a hash table on GPU (in the cache of the GPU, in
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
// by rocFFT (in double, so that its statistics agree with those of TestU01 on the same outputs).
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <rocfft/rocfft.h>
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
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

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

const char kAllTests[] = "poker,coupon,gap,maxoft,collision,birthday,closepairs,permutation,run,serial,collpermut,weight,savir2,gcd,hamweight,hamcorr,"
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

void check(hipError_t e, const char *what, int line) {
    if (e != hipSuccess) throw std::runtime_error(std::string("HIP ") + hipGetErrorString(e) + " at " + __FILE__ + ":" + std::to_string(line) + " (" + what + ")");
}
#define CK(x) check((x), #x, __LINE__)

}  // namespace

// The kernels of device.hpp for a generator, compiled at run time by hiprtc with the source of the
// generator after the device code, so that the generator is inlined as straight-line code (which is
// several times faster than interpreting a generator on GPU), and loaded as a module.
struct Generator::Impl {
    Impl(const std::string &source, const std::string &name) {
        const std::string src = std::string(kDeviceSource) + "\n#line 1 \"" + name + "\"\n" + source;
        hipDevice_t dev;
        CK(hipGetDevice(&dev));
        hipDeviceProp_t prop;
        CK(hipGetDeviceProperties(&prop, dev));
        const std::string arch = std::string("--offload-arch=") + prop.gcnArchName;
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
            throw std::runtime_error("hiprtc: " + std::string(hiprtcGetErrorString(r)) + "\n" + log);
        }
        size_t size = 0;
        rtc(hiprtcGetCodeSize(prog, &size), "hiprtcGetCodeSize");
        std::vector<char> code(size);
        rtc(hiprtcGetCode(prog, code.data()), "hiprtcGetCode");
        hiprtcDestroyProgram(&prog);
        CK(hipModuleLoadData(&module_, code.data()));
        for (auto [f, fname] : {std::pair{&poker, "ktest_poker"}, {&coupon, "ktest_coupon"}, {&gap, "ktest_gap"}, {&maxoft, "ktest_maxoft"}, {&permut, "ktest_permut"},
                                {&points, "kpoints"}, {&spacings, "kspacings"}, {&insert, "kinsert"}, {&fourier_bits, "kfourier_bits"}, {&fourier_acc, "kfourier_acc"},
                                {&cp_outputs, "kcp_outputs"}, {&cp_cells, "kcp_cells"}, {&cp_scatter, "kcp_scatter"}, {&cp_pairs, "kcp_pairs"}, {&krun, "krun"},
                                {&weight, "ktest_weight"}, {&savir2, "ktest_savir2"}, {&gcd, "ktest_gcd"}, {&serial, "kserial"}, {&perm_points, "kperm_points"}, {&cp_slab, "kcp_slab"},
                                {&hamming, "ktest_hamming"}, {&autocor, "kautocor"}, {&samplecorr, "ksamplecorr"}, {&bitrun, "kbitrun"}, {&longhead, "klonghead"},
                                {&uniform, "kuniform"}, {&strip, "ktest_strip"}, {&walk, "kwalk"}, {&sumcoll, "ktest_sumcoll"}, {&matrank, "ktest_matrank"}, {&mat_fill, "kmat_fill"},
                                {&mat_rank, "kmat_rank"}, {&appear, "kappear"}})
            CK(hipModuleGetFunction(f, module_, fname));
    }
    ~Impl() { (void)hipModuleUnload(module_); }
    Impl(const Impl &) = delete;
    Impl &operator=(const Impl &) = delete;

    // the kernel of a test
    hipFunction_t of(const Poker &) const { return poker; }
    hipFunction_t of(const Coupon &) const { return coupon; }
    hipFunction_t of(const Gap &) const { return gap; }
    hipFunction_t of(const MaxOft &) const { return maxoft; }
    hipFunction_t of(const Permut &) const { return permut; }
    hipFunction_t of(const detail::Weight &) const { return weight; }
    hipFunction_t of(const detail::Savir2 &) const { return savir2; }
    hipFunction_t of(const detail::Gcd &) const { return gcd; }
    hipFunction_t of(const detail::Hamming &) const { return hamming; }
    hipFunction_t of(const detail::Strip &) const { return strip; }
    hipFunction_t of(const detail::SumColl &) const { return sumcoll; }
    hipFunction_t of(const detail::MatRank &) const { return matrank; }

    hipFunction_t poker, coupon, gap, maxoft, permut, points, spacings, insert, fourier_bits, fourier_acc, cp_outputs, cp_cells,
        cp_scatter, cp_pairs, cp_slab, krun, weight, savir2, gcd, serial, perm_points, hamming, autocor, samplecorr, bitrun,
        longhead, uniform, strip, walk, sumcoll, matrank, mat_fill, mat_rank, appear;

  private:
    static void rtc(hiprtcResult r, const char *what) {
        if (r != HIPRTC_SUCCESS) throw std::runtime_error(std::string(what) + ": " + hiprtcGetErrorString(r));
    }
    hipModule_t module_ = nullptr;
};

Generator::Generator(const std::string &source, const std::string &name) : impl_(std::make_unique<Impl>(source, name)) {}
Generator::~Generator() = default;

namespace {

using Kernels = Generator::Impl;

// launch the kernel f on grid blocks of block threads with shmem bytes of dynamic LDS
template <class... A>
void launch(hipFunction_t f, unsigned grid, unsigned block, unsigned shmem, A... a) {
    void *args[] = {static_cast<void *>(&a)...};
    CK(hipModuleLaunchKernel(f, grid, 1, 1, block, 1, 1, shmem, nullptr, args, nullptr));
}

// The device memory of the tests on keys (the points of CollisionOver, BirthdaySpacings, CollisionPermut,
// SampleProd and SampleMean): two arrays of cap keys and a counter.
struct KeyBuffers {
    unsigned long long *a = nullptr, *b = nullptr, *count = nullptr;
    long long cap = 0;
    void ensure(long long n) {
        if (n <= cap) return;
        release();
        CK(hipMalloc(reinterpret_cast<void **>(&a), sizeof(unsigned long long) * n));
        CK(hipMalloc(reinterpret_cast<void **>(&b), sizeof(unsigned long long) * n));
        CK(hipMalloc(reinterpret_cast<void **>(&count), sizeof(unsigned long long)));
        cap = n;
    }
    void release() {
        for (void *p : {static_cast<void *>(a), static_cast<void *>(b), static_cast<void *>(count)}) (void)hipFree(p);
        a = b = count = nullptr;
        cap = 0;
    }
    KeyBuffers() = default;
    KeyBuffers(const KeyBuffers &) = delete;
    KeyBuffers &operator=(const KeyBuffers &) = delete;
    ~KeyBuffers() { release(); }
    long long read_count() const {
        unsigned long long c;
        CK(hipMemcpy(&c, count, sizeof c, hipMemcpyDeviceToHost));
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
    // for the overlap of the first observation), rounded up to the generation by kLanes.
    static uint64_t stride(long long n, double mean, double var, int extra = 0) {
        const double q = static_cast<double>((n + kThreadsPerLaunch - 1) / kThreadsPerLaunch);
        const double need = q * mean + 12 * std::sqrt(q * var) + extra;
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

// The real FFTs of Fourier3 by rocFFT (in double): batch sequences of n doubles in a, transformed to n / 2 + 1 complex
// coefficients each in f (unnormalized, as the FFT of TestU01), with the sums of their squares in b.
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
        CK(hipMalloc(reinterpret_cast<void **>(&a), sizeof(double) * n * batch));
        CK(hipMalloc(reinterpret_cast<void **>(&f), sizeof(double) * 2 * (n / 2 + 1) * batch));
        CK(hipMalloc(reinterpret_cast<void **>(&b), sizeof(double) * (n / 4 + 1)));
        const size_t len[1] = {static_cast<size_t>(n)};
        rf(rocfft_plan_create(&plan, rocfft_placement_notinplace, rocfft_transform_type_real_forward, rocfft_precision_double, 1, len, batch, nullptr),
           "rocfft_plan_create");
        rf(rocfft_execution_info_create(&info), "rocfft_execution_info_create");
        size_t ws = 0;
        rf(rocfft_plan_get_work_buffer_size(plan, &ws), "rocfft_plan_get_work_buffer_size");
        if (ws) {
            CK(hipMalloc(&work, ws));
            rf(rocfft_execution_info_set_work_buffer(info, work, ws), "rocfft_execution_info_set_work_buffer");
        }
    }
    ~Fft() {
        if (info) rocfft_execution_info_destroy(info);
        if (plan) rocfft_plan_destroy(plan);
        for (void *p : {static_cast<void *>(a), static_cast<void *>(f), static_cast<void *>(b), work}) (void)hipFree(p);
    }
    Fft(const Fft &) = delete;
    Fft &operator=(const Fft &) = delete;
    // transform the batch (on the null stream, after the kernels filling a)
    void execute() {
        void *in[1] = {a}, *out[1] = {f};
        rf(rocfft_execute(plan, in, out, info), "rocfft_execute");
    }
    static void rf(rocfft_status st, const char *what) {
        if (st != rocfft_status_success) throw std::runtime_error(std::string(what) + ": rocfft status " + std::to_string(static_cast<int>(st)));
    }
};

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
        (void)hipFree(slots);
        (void)hipFree(counters);
    }
    // the slots for n keys (a table of a pass for at most kMaxLoad keys per slot), and the bits of the passes
    int prepare(long long n) {
        uint64_t want = 1;
        while (static_cast<double>(want) * kMaxLoad < n && want < kTableSlots) want <<= 1;
        if (want > size) {
            (void)hipFree(slots);
            CK(hipMalloc(reinterpret_cast<void **>(&slots), sizeof(unsigned long long) * want));
            if (!counters) CK(hipMalloc(reinterpret_cast<void **>(&counters), sizeof(unsigned long long) * 2));
            size = want;
        }
        int pb = 0;
        while (static_cast<double>(n >> pb) > kMaxLoad * size) ++pb;
        return pb;
    }
    // the empty table of the pass part of 2^pbits; the collisions are counted from the first pass
    HashTable table(int pbits, uint64_t part) {
        CK(hipMemset(slots, 0xff, sizeof(unsigned long long) * size));
        if (part == 0) CK(hipMemset(counters, 0, sizeof(unsigned long long) * 2));
        return {slots, counters, counters + 1, size - 1, pbits, part};
    }
    // the collisions in the table over the passes: the counted ones and those of the key kEmpty
    long long collisions() {
        unsigned long long c[2];
        CK(hipMemcpy(c, counters, sizeof c, hipMemcpyDeviceToHost));
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
    CK(hipMemset(m.keys.count, 0, sizeof(unsigned long long)));
    f(Sink{m.keys.a, m.keys.count, m.keys.cap, pb, 64, 1, part});
    const long long c = m.keys.read_count();
    if (c <= m.keys.cap) return count_equal_keys(kn, m, m.keys.a, c);
    if (pb >= 40) throw TestFailure("too many equal keys for the memory budget");
    return count_equal_part(kn, m, f, pb + 1, 2 * part) + count_equal_part(kn, m, f, pb + 1, 2 * part + 1);
}

using PartFn = std::function<void(unsigned long long *, long long)>;

void sorted_part(const Kernels &kn, Mem &m, int kbits, const Launch &f, const PartFn &part_fn, int pb, uint64_t part) {
    CK(hipMemset(m.keys.count, 0, sizeof(unsigned long long)));
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

// CollisionOver: the number of collisions of n overlapping points in 2^(b * t) cells
// (the points are made again for every pass of the hash table, which is cheaper than storing them; no memory
// grows with n)
long long collision_over(const Kernels &kn, Mem &m, const Tuples &tup, const Params &g, Counters &ctr, long long n) {
    const uint64_t stride = Counters::stride(n, 1, 0, tup.t - 1), base = ctr.take(stride);
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
        CK(hipMemcpy(&first, a, sizeof first, hipMemcpyDeviceToHost));
        if (have_last) {  // the spacing across the parts
            const unsigned long long h = Mix64()(first - last);
            bucket[h >> (64 - hb)].push_back(h);
        }
        CK(hipMemcpy(&last, a + c - 1, sizeof last, hipMemcpyDeviceToHost));
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
            if (k) CK(hipMemcpy(bucket[j].data() + o, m.keys.b + hat[j], sizeof(unsigned long long) * k, hipMemcpyDeviceToHost));
        }
    });
    long long eq = 0;
    for (auto &bk : bucket) {
        const long long k = static_cast<long long>(bk.size());
        if (k <= m.keys.cap) {
            if (k) CK(hipMemcpy(m.keys.a, bk.data(), sizeof(unsigned long long) * k, hipMemcpyHostToDevice));
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
        CK(hipMalloc(reinterpret_cast<void **>(&pts), sizeof(uint32_t) * ncoords));
        CK(hipMalloc(reinterpret_cast<void **>(&sorted), sizeof(uint32_t) * ncoords));
        CK(hipMalloc(reinterpret_cast<void **>(&cell), sizeof(uint32_t) * (ncoords / 2 + 1)));
        CK(hipMalloc(reinterpret_cast<void **>(&dist), sizeof(uint32_t) * kCap));
        CK(hipMalloc(reinterpret_cast<void **>(&count), sizeof(unsigned) * kMaxCells));
        CK(hipMalloc(reinterpret_cast<void **>(&ndist), sizeof(unsigned)));
        CK(hipMalloc(reinterpret_cast<void **>(&start), sizeof(unsigned long long) * kMaxCells));
        CK(hipMalloc(reinterpret_cast<void **>(&cursor), sizeof(unsigned long long) * kMaxCells));
        CK(hipMalloc(reinterpret_cast<void **>(&nslab), sizeof(unsigned long long)));
    }
    ~CpBuffers() {
        for (void *p : {static_cast<void *>(pts), static_cast<void *>(sorted), static_cast<void *>(cell), static_cast<void *>(dist),
                        static_cast<void *>(count), static_cast<void *>(ndist), static_cast<void *>(start), static_cast<void *>(cursor),
                        static_cast<void *>(nslab)})
            (void)hipFree(p);
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
    CK(hipMemset(buf.count, 0, sizeof(unsigned) * gr.cells));
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
    CK(hipMemset(buf.nslab, 0, sizeof(unsigned long long)));
    launch(kn.cp_slab, kBlocks, kThreads, 0, g, base, n, t, static_cast<uint32_t>(lo - gr.dlim), span, buf.pts, buf.nslab, pcap);
    unsigned long long c;
    CK(hipMemcpy(&c, buf.nslab, sizeof c, hipMemcpyDeviceToHost));
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
        CK(hipMemset(buf.ndist, 0, sizeof(unsigned)));
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
        CK(hipMemcpy(&nd, buf.ndist, sizeof nd, hipMemcpyDeviceToHost));
        if (iter < 40 && nd < static_cast<unsigned>(mm) && dlim < all) {
            B *= 4;
            continue;
        }
        if (iter < 40 && nd > CpBuffers::kCap && dlim > 1) {
            B /= 2;
            continue;
        }
        std::vector<uint32_t> dist(std::min(nd, CpBuffers::kCap));
        CK(hipMemcpy(dist.data(), buf.dist, sizeof(uint32_t) * dist.size(), hipMemcpyDeviceToHost));
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
    const char *name;
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
    CK(hipMalloc(reinterpret_cast<void **>(&d_hist), sizeof(unsigned long long) * nbins));
    CK(hipMalloc(reinterpret_cast<void **>(&d_overrun), sizeof(int)));
    CK(hipMemset(d_hist, 0, sizeof(unsigned long long) * nbins));
    CK(hipMemset(d_overrun, 0, sizeof(int)));
    const int lds = nbins <= kMaxLdsBins;
    const uint64_t stride = Counters::stride(n, mean, var, extra);
    launch(kn.of(test), kBlocks, kThreads, lds ? sizeof(unsigned) * nbins : 0, test, g, ctr.take(stride), stride, n, d_hist, nbins, lds,
           d_overrun);
    std::vector<unsigned long long> h(nbins);
    int overrun;
    CK(hipMemcpy(h.data(), d_hist, sizeof(unsigned long long) * nbins, hipMemcpyDeviceToHost));
    CK(hipMemcpy(&overrun, d_overrun, sizeof overrun, hipMemcpyDeviceToHost));
    CK(hipFree(d_hist));
    CK(hipFree(d_overrun));
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

// the probabilities of the number of distinct values among k values in [0, d) (index 0..k)
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

bool wanted(const Options &o, const char *t) { return ("," + o.tests + ",").find(std::string(",") + t + ",") != std::string::npos; }

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

std::string format(const Result &r) {
    char line[512];
    const char *mk = r.p < 0.001 || r.p > 0.999 ? "  *****" : "";
    const char *t = r.test.c_str(), *q = r.params.c_str();
    switch (r.kind) {
    case Stat::kChiSquare: std::snprintf(line, sizeof line, "%-7s %-34s chi2=%-12.6g df=%-8ld p=%-9.3g%s  (%.1fs)", t, q, r.stat, r.df, r.p, mk, r.seconds); break;
    case Stat::kPoisson:
        std::snprintf(line, sizeof line, "%-7s %-34s coll=%-11lld mean=%-9.6g p=%-9.3g%s  (%.1fs)", t, q, static_cast<long long>(r.stat), r.mean, r.p, mk, r.seconds);
        break;
    case Stat::kNormal: std::snprintf(line, sizeof line, "%-7s %-34s z=%-14.6g p=%-9.3g%s  (%.1fs)", t, q, r.stat, r.p, mk, r.seconds); break;
    case Stat::kAndersonDarling: std::snprintf(line, sizeof line, "%-7s %-34s AD=%-13.6g p=%-9.3g%s  (%.1fs)", t, q, r.stat, r.p, mk, r.seconds); break;
    case Stat::kClosePairs: std::snprintf(line, sizeof line, "%-7s %-34s stat=%-12.6g p=%-9.3g%s  (%.1fs)", t, q, r.stat, r.p, mk, r.seconds); break;
    case Stat::kDiscrete: std::snprintf(line, sizeof line, "%-7s %-34s p=%-9.3g%s", t, q, r.p, mk); break;
    }
    return line;
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
    const auto emit = [&](const char *name, const char *params, Stat kind, double stat, double p, double z, double sec, long df = 0, double mean = 0) {
        sum.add(p, z);
        if (o.report) o.report(Result{name, params, kind, stat, mean, df, p, z, sec});
    };
    const auto report = [&](const char *name, const char *params, Chi2 c, double sec) {
        const double p = fbar_ChiSquare2(c.df, 12, c.x);
        emit(name, params, Stat::kChiSquare, c.x, p, (c.x - c.df) / std::sqrt(2.0 * c.df), sec, c.df);
    };
    char params[128];

    if (wanted(o, "poker")) {
        const int set[][4] = {{0, 8, 8, 0}, {27, 8, 8, 0}, {0, 32, 32, 1}, {25, 32, 32, 1}};  // r, d, k, n = 4e8 or 1e8
        for (const auto &s : set) {
            const long long n = scaled(s[3] ? 1e8 : 4e8);
            const auto t0 = std::chrono::steady_clock::now();
            const auto h = histogram(kn, Poker{s[0], s[1], s[2]}, g, ctr, n, s[2] + 1, s[2], 0);
            std::snprintf(params, sizeof params, "n=%lld r=%d d=%d k=%d", n, s[0], s[1], s[2]);
            report("poker", params, chi2(h, poker_prob(s[1], s[2]), n), seconds_since(t0));
            if (sum.max_z > o.limit) return sum;
        }
    }
    if (wanted(o, "coupon")) {
        for (const int r : {0, 10, 20, 27}) {
            const int d = 8;
            const long long n = scaled(2e8);
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
            std::snprintf(params, sizeof params, "n=%lld r=%d d=%d", n, r, d);
            report("coupon", params, chi2(h, coupon_prob(d, tmax), n), seconds_since(t0));
            if (sum.max_z > o.limit) return sum;
        }
    }
    if (wanted(o, "gap")) {
        const struct { double n; int r; int lg; } set[] = {{5e8, 0, 4}, {3e8, 25, 5}, {1e8, 0, 7}, {1e7, 20, 10}};  // beta = 2^-lg
        for (const auto &s : set) {
            const long long n = scaled(s.n);
            const double w = std::ldexp(1.0, -s.lg);
            const int tmax = lump_point([&](int t) { return std::pow(1 - w, t); }, n, 1, 1 << 20);
            const auto t0 = std::chrono::steady_clock::now();
            // the values per gap (including the one in the interval): geometric of the probability w
            const auto h = histogram(kn, Gap{s.r, tmax, 0, UINT64_C(1) << (32 - s.lg)}, g, ctr, n, tmax + 1, 1 / w, (1 - w) / (w * w));
            std::snprintf(params, sizeof params, "n=%lld r=%d beta=2^-%d", n, s.r, s.lg);
            report("gap", params, chi2(h, gap_prob(w, tmax), n), seconds_since(t0));
            if (sum.max_z > o.limit) return sum;
        }
    }
    if (wanted(o, "maxoft")) {
        const int set[][2] = {{40, 8}, {30, 16}, {20, 24}, {20, 32}};  // N, t; n = 1e7, d = 1e5
        for (const auto &s : set) {
            const int d = 100000;
            const long long n = scaled(1e7);
            const auto t0 = std::chrono::steady_clock::now();
            Chi2 all{0, 0};
            for (int rep = 0; rep < s[0]; ++rep) {
                const auto h = histogram(kn, MaxOft{d, s[1]}, g, ctr, n, d, s[1], 0);
                const Chi2 c = chi2(h, std::vector<double>(d, 1.0 / d), n);
                all.x += c.x;
                all.df += c.df;
            }
            std::snprintf(params, sizeof params, "N=%d n=%lld d=%d t=%d", s[0], n, d, s[1]);
            report("maxoft", params, all, seconds_since(t0));
            if (sum.max_z > o.limit) return sum;
        }
    }
    // the tests on points: the total number of collisions over N replications, against a Poisson
    // distribution of the mean N * mu; the p-value is P[X >= observed]
    const auto report_poisson = [&](const char *name, const char *params, long long obs, double mean, double sec) {
        const double p = fbar_Poisson1(mean, obs);
        emit(name, params, Stat::kPoisson, static_cast<double>(obs), p, (obs - mean) / std::sqrt(mean), sec, 0, mean);
    };
    const auto tuples = [](int r, int b, int t, int overlap) {
        return Tuples{r, b, t, overlap, b * t == 64 ? ~UINT64_C(0) : (UINT64_C(1) << (b * t)) - 1};
    };
    {
        if (wanted(o, "collision")) {
            const int set[][3] = {{0, 21, 2}, {9, 21, 2}, {0, 14, 3}, {16, 14, 3}, {0, 6, 7}, {24, 6, 7}, {0, 3, 14}, {27, 3, 14}, {0, 2, 21}, {28, 2, 21}};  // r, b, t; N = 30, n = 2e7
            for (const auto &s : set) {
                const int N = 30;
                const long long n = scaled(2e7);
                const double k = std::ldexp(1.0, s[1] * s[2]);
                const double mu = n + k * std::expm1(n * std::log1p(-1 / k));  // expected collisions of n points in k cells
                const auto t0 = std::chrono::steady_clock::now();
                long long c = 0;
                for (int rep = 0; rep < N; ++rep) {
                    c += collision_over(kn, mem, tuples(s[0], s[1], s[2], 1), g, ctr, n);
                    if ((c - (rep + 1) * mu) / std::sqrt((rep + 1) * mu) > o.limit) {  // already far above its mean
                        sum.add(0, (c - (rep + 1) * mu) / std::sqrt((rep + 1) * mu));
                        return sum;
                    }
                }
                std::snprintf(params, sizeof params, "N=%d n=%lld r=%d d=2^%d t=%d", N, n, s[0], s[1], s[2]);
                report_poisson("collover", params, c, N * mu, seconds_since(t0));
                if (sum.max_z > o.limit) return sum;
            }
        }
        if (wanted(o, "birthday")) {
            const struct { int N; double n; int r, b, t; } set[] = {
                {100, 1e7, 0, 31, 2}, {20, 2e7, 0, 21, 3}, {20, 3e7, 14, 16, 4}, {20, 2e7, 0, 9, 7}, {20, 2e7, 7, 9, 7},
                {20, 3e7, 14, 8, 8}, {20, 3e7, 22, 8, 8}, {20, 3e7, 0, 4, 16}, {20, 3e7, 26, 4, 16}};
            for (const auto &s : set) {
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
                std::snprintf(params, sizeof params, "N=%d n=%lld r=%d d=2^%d t=%d", s.N, n, s.r, s.b, s.t);
                report_poisson("birthday", params, c, s.N * lambda, seconds_since(t0));
                if (sum.max_z > o.limit) return sum;
            }
        }
    }
    if (wanted(o, "closepairs")) {
        const struct { int N; double n; int t; } set[] = {{30, 6e6, 3}, {20, 4e6, 5}, {10, 3e6, 9}, {5, 2e6, 16}};  // r = 0, p = 0, m = 30
        for (const auto &s : set) {
            const int m = 30;
            const long long n = scaled(s.n);
            const auto t0 = std::chrono::steady_clock::now();
            std::vector<std::vector<uint32_t>> reps;
            for (int rep = 0; rep < s.N; ++rep) reps.push_back(close_pairs(kn, mem, g, ctr, n, s.t, m));
            const double sec = seconds_since(t0);
            for (const CpStats &c : close_pairs_stats(reps, n, s.t, m)) {
                std::snprintf(params, sizeof params, "N=%d n=%lld t=%d m=%d %s", s.N, n, s.t, m, c.name);
                emit("cpairs", params, Stat::kClosePairs, c.stat, c.p, 0, sec);
            }
        }
    }
    if (wanted(o, "permutation")) {
        const struct { double n; int r, t; } set[] = {{1e9, 5, 3}, {1e9, 5, 5}, {5e8, 5, 7}, {5e8, 10, 10}};  // N = 1
        for (const auto &s : set) {
            const long long n = scaled(s.n);
            int cells = 1;
            for (int i = 2; i <= s.t; ++i) cells *= i;
            const auto t0 = std::chrono::steady_clock::now();
            const auto h = histogram(kn, Permut{s.r, s.t}, g, ctr, n, cells, s.t, 0);
            std::snprintf(params, sizeof params, "n=%lld r=%d t=%d", n, s.r, s.t);
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
        const struct { int N; double n; int r, up; } set[] = {{5, 1e9, 0, 0}, {10, 1e9, 15, 1}};
        unsigned long long *d_cnt;
        CK(hipMalloc(reinterpret_cast<void **>(&d_cnt), sizeof(unsigned long long) * 6));
        for (const auto &s : set) {
            const long long n = scaled(s.n);
            const auto t0 = std::chrono::steady_clock::now();
            double khi = 0;
            for (int rep = 0; rep < s.N; ++rep) {
                CK(hipMemset(d_cnt, 0, sizeof(unsigned long long) * 6));
                const uint64_t start = ctr.take((n + Counters::kThreadsPerLaunch - 1) / Counters::kThreadsPerLaunch);
                launch(kn.krun, kBlocks, kThreads, 0, g, start, n, s.r, s.up, d_cnt);
                unsigned long long c[6];
                CK(hipMemcpy(c, d_cnt, sizeof c, hipMemcpyDeviceToHost));
                double k = 0;
                for (int i = 0; i < 6; ++i)
                    for (int j = 0; j < 6; ++j) k += A[i][j] * (c[i] - n * B[i]) * (c[j] - n * B[j]);
                khi += k / (n - 6.0);
            }
            std::snprintf(params, sizeof params, "N=%d n=%lld r=%d %s", s.N, n, s.r, s.up ? "up" : "down");
            report("run", params, Chi2{khi, 6L * s.N}, seconds_since(t0));
            if (sum.max_z > o.limit) {
                CK(hipFree(d_cnt));
                return sum;
            }
        }
        CK(hipFree(d_cnt));
    }
    if (wanted(o, "serial")) {
        // SerialOver (N = 1, d = 256, t = 3): the chi-square X of the overlapping 3-tuples minus that of the
        // 2-tuples (circular, as TestU01), against the chi-square of k - k1 degrees of freedom (Good)
        for (const int r : {0, 22}) {
            const int b = 8, t = 3;
            const long long n = scaled(1e9);
            const uint64_t k = UINT64_C(1) << (b * t), k1 = UINT64_C(1) << (b * (t - 1));
            const auto t0 = std::chrono::steady_clock::now();
            unsigned *d_cnt, *d_cnt1;
            CK(hipMalloc(reinterpret_cast<void **>(&d_cnt), sizeof(unsigned) * k));
            CK(hipMalloc(reinterpret_cast<void **>(&d_cnt1), sizeof(unsigned) * k1));
            CK(hipMemset(d_cnt, 0, sizeof(unsigned) * k));
            CK(hipMemset(d_cnt1, 0, sizeof(unsigned) * k1));
            const uint64_t start = ctr.take((n + Counters::kThreadsPerLaunch - 1) / Counters::kThreadsPerLaunch);
            launch(kn.serial, kBlocks, kThreads, 0, g, start, n, r, b, t, d_cnt, d_cnt1);
            std::vector<unsigned> c(k), c1(k1);
            CK(hipMemcpy(c.data(), d_cnt, sizeof(unsigned) * k, hipMemcpyDeviceToHost));
            CK(hipMemcpy(c1.data(), d_cnt1, sizeof(unsigned) * k1, hipMemcpyDeviceToHost));
            CK(hipFree(d_cnt));
            CK(hipFree(d_cnt1));
            const auto pearson = [n](const std::vector<unsigned> &v) {
                const double e = static_cast<double>(n) / v.size();
                double x = 0;
                for (const unsigned ci : v) x += (ci - e) * (ci - e);
                return x / e;
            };
            std::snprintf(params, sizeof params, "n=%lld r=%d d=%d t=%d", n, r, 1 << b, t);
            report("serial", params, Chi2{pearson(c) - pearson(c1), static_cast<long>(k - k1)}, seconds_since(t0));
            if (sum.max_z > o.limit) return sum;
        }
    }
    if (wanted(o, "collpermut")) {
        // CollisionPermut (N = 20, n = 2e7, t = 14): collisions of the orders of 14 values among 14! cells
        for (const int r : {0, 10}) {
            const int N = 20, t = 14;
            const long long n = scaled(2e7);
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
            std::snprintf(params, sizeof params, "N=%d n=%lld r=%d t=%d", N, n, r, t);
            report_poisson("collperm", params, c, N * mu, seconds_since(t0));
            if (sum.max_z > o.limit) return sum;
        }
    }
    if (wanted(o, "weight")) {
        // WeightDistrib (N = 1, n = 2e7, k = 256, alpha = 0): the binomial number of values below beta
        const struct { int r, lg; } set[] = {{0, 2}, {20, 2}, {28, 2}, {0, 4}, {10, 4}, {26, 4}};  // beta = 2^-lg
        for (const auto &s : set) {
            const int k = 256;
            const long long n = scaled(2e7);
            const double p = std::ldexp(1.0, -s.lg);
            std::vector<double> prob(k + 1);
            for (int i = 0; i <= k; ++i) prob[i] = std::exp(std::lgamma(k + 1.0) - std::lgamma(i + 1.0) - std::lgamma(k - i + 1.0) + i * std::log(p) + (k - i) * std::log1p(-p));
            const auto t0 = std::chrono::steady_clock::now();
            const auto h = histogram(kn, detail::Weight{s.r, k, UINT64_C(1) << (32 - s.lg)}, g, ctr, n, k + 1, k, 0);
            std::snprintf(params, sizeof params, "n=%lld r=%d k=%d beta=2^-%d", n, s.r, k, s.lg);
            report("weight", params, chi2(h, prob, n), seconds_since(t0));
            if (sum.max_z > o.limit) return sum;
        }
    }
    if (wanted(o, "savir2")) {
        // Savir2 (N = 10, n = 1e7, r = 10, m = 2^20, t = 30): the sum of the chi-squares of the N replications
        const int N = 10, r = 10, t = 30;
        const long long m = 1 << 20, n = scaled(1e7);
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
        std::snprintf(params, sizeof params, "N=%d n=%lld r=%d m=2^20 t=%d", N, n, r, t);
        report("savir2", params, all, seconds_since(t0));
        if (sum.max_z > o.limit) return sum;
    }
    if (wanted(o, "gcd")) {
        // GCD (N = 10, n = 5e7, r = 0, s = 30): the gcd of pairs against 6 / (pi^2 j^2), lumped at jmax
        const int N = 10, r = 0, s = 30;
        const long long n = scaled(5e7);
        const double c1 = 6 / (M_PI * M_PI);
        const int jmax = 1 + static_cast<int>(std::sqrt(c1 * n / kMinExpected));
        std::vector<double> prob(jmax + 1, 0.0);
        double acc = 0;
        for (int j = 1; j < jmax; ++j) acc += prob[j] = c1 / (static_cast<double>(j) * j);
        prob[jmax] = 1 - acc;
        const auto t0 = std::chrono::steady_clock::now();
        Chi2 all{0, 0};
        for (int rep = 0; rep < N; ++rep) {
            const auto h = histogram(kn, detail::Gcd{r, s, jmax}, g, ctr, n, jmax + 1, 2, 0);
            const Chi2 c = chi2(h, prob, n);
            all.x += c.x;
            all.df += c.df;
        }
        std::snprintf(params, sizeof params, "N=%d n=%lld r=%d s=%d", N, n, r, s);
        report("gcd", params, all, seconds_since(t0));
        if (sum.max_z > o.limit) return sum;
    }
    // the tests on the bits of the Hamming tests (sstring): blocks of L bits of ldivs values of s bits and
    // the first lmods bits of one more
    const auto hamming = [](int r, int s, int L, int mode, int h) { return detail::Hamming{r, s, L / s, L % s, L, mode, h}; };
    const auto vals_per_block = [](int s, int L) { return L / s + (L % s > 0); };
    const auto report_normal = [&](const char *name, const char *params, double z, double sec) {
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
        for (const int r : {0, 27}) {
            const int N = 10, s = 3, L = 1000000, h = 6000;
            const long long K = static_cast<long long>(scaled(1e9) / L);
            const auto t0 = std::chrono::steady_clock::now();
            const auto hist = histogram(kn, hamming(r, s, L, 2, h), g, ctr, N * K, 2 * h + 1, vals_per_block(s, L), 0);
            double x = 0;
            for (int c = 0; c <= 2 * h; ++c) x += hist[c] * static_cast<double>(c - h) * (c - h);
            std::snprintf(params, sizeof params, "N=%d n=%lld r=%d s=%d L=%d", N, K * L, r, s, L);
            report("hamweight", params, Chi2{x * 4.0 / L, N * K}, seconds_since(t0));
            if (sum.max_z > o.limit) return sum;
        }
    }
    if (wanted(o, "hamcorr")) {
        // HammingCorr (N = 1, r = 10, s = 10): the correlation of the weights of consecutive blocks,
        // sum (2 X_i - L)(2 X_{i+1} - L) / (L sqrt(pairs)), a standard normal
        const struct { double n; int L; } set[] = {{1e9, 30}, {1e8, 300}, {1e8, 1200}};
        for (const auto &c : set) {
            const int r = 10, s = 10, L = c.L;
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
            std::snprintf(params, sizeof params, "n=%lld r=%d s=%d L=%d", n, r, s, L);
            report_normal("hamcorr", params, acc / (L * std::sqrt(pairs)), seconds_since(t0));
            if (sum.max_z > o.limit) return sum;
        }
    }
    if (wanted(o, "hamindep")) {
        // HammingIndep (d = 0): the chi-square of the (L + 1)^2 pairs of the weights of non-overlapping
        // pairs of blocks against the binomial products, the cells of an expected count of at least 10 as
        // classes and the others merged into one (into the last class if below 10), summed over N
        const struct { int N; double n; int r, s, L; } set[] = {
            {10, 3e7, 0, 3, 30}, {10, 3e7, 27, 3, 30}, {1, 3e7, 0, 4, 300}, {1, 3e7, 26, 4, 300}, {1, 1e7, 0, 5, 1200}, {1, 1e7, 25, 5, 1200}};
        for (const auto &c : set) {
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
            std::snprintf(params, sizeof params, "N=%d n=%lld r=%d s=%d L=%d", c.N, n, c.r, c.s, L);
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
        CK(hipMalloc(reinterpret_cast<void **>(&d_cnt), sizeof(unsigned long long) * 80));
        for (const int r : {0, 27}) {
            const int s = 3;
            const long long nr = scaled(2e9), nv = (4 * nr + s - 1) / s;
            const int k = 1 + static_cast<int>(std::log2(nr / kMinExpected));
            if (k > 40) throw TestFailure("bitrun: too many classes");
            const auto t0 = std::chrono::steady_clock::now();
            CK(hipMemset(d_cnt, 0, sizeof(unsigned long long) * 80));
            const uint64_t start = ctr.take((nv + Counters::kThreadsPerLaunch - 1) / Counters::kThreadsPerLaunch);
            launch(kn.bitrun, kBlocks, kThreads, 0, g, start, nv, r, s, k, d_cnt);
            std::vector<unsigned long long> c(2 * k);
            CK(hipMemcpy(c.data(), d_cnt, sizeof(unsigned long long) * 2 * k, hipMemcpyDeviceToHost));
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
            std::snprintf(params, sizeof params, "n=%lld r=%d s=%d runs", nr, r, s);
            report("bitrun", params, Chi2{x, 2L * (k - 1)}, seconds_since(t0));
            std::snprintf(params, sizeof params, "n=%lld r=%d s=%d count", nr, r, s);
            report_normal("bitrun", params, (bits / 4 - runs1) / std::sqrt(bits / 16), seconds_since(t0));
            if (sum.max_z > o.limit) {
                CK(hipFree(d_cnt));
                return sum;
            }
        }
        CK(hipFree(d_cnt));
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
        for (const int r : {0, 27}) {
            const int s = 3;
            const long long nb = scaled(1000), K = 10000020 / s, L = K * s;
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
            CK(hipMalloc(reinterpret_cast<void **>(&d_long), sizeof(long long) * nb));
            const uint64_t start = ctr.take((nb * K + Counters::kThreadsPerLaunch - 1) / Counters::kThreadsPerLaunch);
            launch(kn.longhead, static_cast<unsigned>(nb), kThreads, 0, g, start, K, r, s, d_long);
            std::vector<long long> lg(nb);
            CK(hipMemcpy(lg.data(), d_long, sizeof(long long) * nb, hipMemcpyDeviceToHost));
            CK(hipFree(d_long));
            std::vector<unsigned long long> h(jhigh + 1, 0);
            long long top = 0;
            for (const long long x : lg) {
                ++h[std::min<long long>(x, jhigh)];
                top = std::max(top, x);
            }
            std::snprintf(params, sizeof params, "n=%lld r=%d s=%d L=%lld", nb, r, s, L);
            report("longhead", params, chi2(h, prob, nb), seconds_since(t0));
            const double lnall = std::log(static_cast<double>(nb) * L);
            double cl = 0, cprev = 0;
            for (long j = 0; j <= top; ++j) {
                cprev = cl;
                cl += plhr(j, lnall);
            }
            const double p = gofw_pDisc(cl, 1 - cprev);
            std::snprintf(params, sizeof params, "n=%lld r=%d s=%d longest=%lld", nb, r, s, top);
            emit("longhead", params, Stat::kDiscrete, static_cast<double>(top), p, 0, seconds_since(t0));
            if (sum.max_z > o.limit) return sum;
        }
    }
    if (wanted(o, "autocor")) {
        // AutoCor (N = 10, n = 1e9 + 30 bits, s = 3): the differing bits at the distance d over K = (n - d) / s
        // values, 2 (A - K s / 2) / sqrt(K s) a standard normal, summed over N (divided by sqrt(N))
        unsigned long long *d_cnt;
        CK(hipMalloc(reinterpret_cast<void **>(&d_cnt), sizeof(unsigned long long)));
        const struct { int r, d; } set[] = {{0, 1}, {0, 3}, {27, 1}, {27, 3}};
        for (const auto &c : set) {
            const int N = 10, s = 3;
            const long long nbits = scaled(1e9 + 30), K = (nbits - c.d) / s;
            const auto t0 = std::chrono::steady_clock::now();
            double zs = 0;
            for (int rep = 0; rep < N; ++rep) {
                CK(hipMemset(d_cnt, 0, sizeof(unsigned long long)));
                const uint64_t start = ctr.take((K + 1 + Counters::kThreadsPerLaunch - 1) / Counters::kThreadsPerLaunch);
                launch(kn.autocor, kBlocks, kThreads, 0, g, start, K, c.r, s, c.d, d_cnt);
                unsigned long long A;
                CK(hipMemcpy(&A, d_cnt, sizeof A, hipMemcpyDeviceToHost));
                const double m = static_cast<double>(K) * s;
                zs += 2.0 * (A - m / 2) / std::sqrt(m);
            }
            std::snprintf(params, sizeof params, "N=%d n=%lld r=%d s=%d d=%d", N, nbits, c.r, s, c.d);
            report_normal("autocor", params, zs / std::sqrt(static_cast<double>(N)), seconds_since(t0));
            if (sum.max_z > o.limit) {
                CK(hipFree(d_cnt));
                return sum;
            }
        }
        CK(hipFree(d_cnt));
    }
    if (wanted(o, "samplecorr")) {
        // SampleCorr (N = 1, n = 2e9): the sum of u_{i-k} u_i - 1/4 over n - k terms, of the variance 13/144 per
        // term (7/144, and 1/48 with each of the two terms sharing a value; TestU01 takes 1/12, so that its
        // statistic has the variance 13/12)
        double *d_sum;
        CK(hipMalloc(reinterpret_cast<void **>(&d_sum), sizeof(double)));
        for (const int k : {1, 2}) {
            const long long n = scaled(2e9);
            const auto t0 = std::chrono::steady_clock::now();
            CK(hipMemset(d_sum, 0, sizeof(double)));
            const uint64_t start = ctr.take((n + Counters::kThreadsPerLaunch - 1) / Counters::kThreadsPerLaunch);
            launch(kn.samplecorr, kBlocks, kThreads, 0, g, start, n, 0, k, d_sum);
            double sm;
            CK(hipMemcpy(&sm, d_sum, sizeof sm, hipMemcpyDeviceToHost));
            std::snprintf(params, sizeof params, "n=%lld r=0 k=%d", n, k);
            report_normal("samplecorr", params, sm * 12.0 / std::sqrt(13.0 * (n - k)), seconds_since(t0));
            if (sum.max_z > o.limit) {
                CK(hipFree(d_sum));
                return sum;
            }
        }
        CK(hipFree(d_sum));
    }
    // SampleProd and SampleMean: the Anderson-Darling test of n values transformed to U(0, 1) by their
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
            if (c) CK(hipMemcpy(keys.data() + at, a, sizeof(unsigned long long) * c, hipMemcpyDeviceToHost));
        });
        std::vector<double> v(n + 1);  // the arrays of TestU01 start at 1
        for (long long i = 0; i < n; ++i) v[i + 1] = (static_cast<double>(keys[i]) + 0.5) * 0x1p-64;
        const double a2 = gofs_AndersonDarling(v.data(), static_cast<long>(n));
        return std::pair<double, double>{a2, fbar_AndersonDarling(static_cast<long>(n), a2)};
    };
    const auto report_ad = [&](const char *name, const char *params, double a2, double p, double sec) {
        emit(name, params, Stat::kAndersonDarling, a2, p, p > 0 ? std::max(0.0, -finv_Normal1(std::max(p, 1e-300))) : HUGE_VAL, sec);
    };
    if (wanted(o, "sampleprod")) {
        // SampleProd (n = 1e7, r = 0): for each of N replications, the AD test of the products of t values
        // against FDistProd; then the AD test of the N p-values
        const int set[][2] = {{40, 8}, {20, 16}, {20, 24}};
        for (const auto &c : set) {
            const long long n = scaled(1e7);
            const auto t0 = std::chrono::steady_clock::now();
            std::vector<double> ps;
            for (int rep = 0; rep < c[0]; ++rep) {
                const uint64_t stride = Counters::stride(n, c[1], 0);
                const uint64_t base = ctr.take(stride);
                ps.push_back(ad_sorted(n, [&](const Sink &sk) {
                                 launch(kn.uniform, kBlocks, kThreads, 0, 0, 0, c[1], g, base, stride, n, static_cast<const double *>(nullptr), 0, 0.0, sk);
                             }).second);
            }
            const auto ad = anderson_darling(ps);
            std::snprintf(params, sizeof params, "N=%d n=%lld r=0 t=%d", c[0], n, c[1]);
            report_ad("sprod", params, ad.first, ad.second, seconds_since(t0));
            if (sum.max_z > o.limit) return sum;
        }
    }
    if (wanted(o, "samplemean")) {
        // SampleMean (N = 2e7 sums of n = 30 values, r = 0 and 10): the AD test of the N sums against the
        // Irwin-Hall distribution (by a table of it and its density, see kuniform)
        const int m = 30, ntab = 1 << 16;
        const double hstep = static_cast<double>(m) / (ntab - 1);
        std::vector<double> tab(2 * ntab);
        {
            const auto ih = [](long double x, int deg) {  // sum_k (-1)^k C(m, k) (x - k)^deg / deg!, x <= m / 2
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
        CK(hipMalloc(reinterpret_cast<void **>(&d_tab), sizeof(double) * tab.size()));
        CK(hipMemcpy(d_tab, tab.data(), sizeof(double) * tab.size(), hipMemcpyHostToDevice));
        for (const int r : {0, 10}) {
            const long long n = scaled(2e7);
            const auto t0 = std::chrono::steady_clock::now();
            const uint64_t stride = Counters::stride(n, m, 0);
            const uint64_t base = ctr.take(stride);
            const auto ad = ad_sorted(n, [&](const Sink &sk) {
                launch(kn.uniform, kBlocks, kThreads, 0, 1, r, m, g, base, stride, n, static_cast<const double *>(d_tab), ntab, hstep, sk);
            });
            std::snprintf(params, sizeof params, "N=%lld n=%d r=%d", n, m, r);
            report_ad("smean", params, ad.first, ad.second, seconds_since(t0));
            if (sum.max_z > o.limit) {
                CK(hipFree(d_tab));
                return sum;
            }
        }
        CK(hipFree(d_tab));
    }
    if (wanted(o, "walk")) {
        // RandomWalk1 (N = 1, L0 = L1 = L): the chi-squares of the statistics H, M, J, R and C of n walks
        // against the expected numbers of TestU01 (CalcNbExp of swalk.c)
        const struct { double n; int r, s, L; } set[] = {{1e8, 0, 5, 50}, {1e8, 25, 5, 50}, {1e7, 0, 10, 1000}, {1e7, 20, 10, 1000}, {1e6, 0, 15, 10000}, {1e6, 15, 15, 10000}};
        for (const auto &c : set) {
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
            CK(hipMalloc(reinterpret_cast<void **>(&d_hist), sizeof(unsigned long long) * nb));
            CK(hipMemset(d_hist, 0, sizeof(unsigned long long) * nb));
            const int per = L / c.s + (L % c.s > 0);
            const uint64_t stride = Counters::stride(n, per, 0);
            launch(kn.walk, kBlocks, kThreads, lds ? sizeof(unsigned) * nb : 0, g, ctr.take(stride), stride, n, c.r, c.s, c.L, d_hist, lds);
            std::vector<unsigned long long> h(nb);
            CK(hipMemcpy(h.data(), d_hist, sizeof(unsigned long long) * nb, hipMemcpyDeviceToHost));
            CK(hipFree(d_hist));
            const double sec = seconds_since(t0);
            const char *names = "HMJRC";
            for (int k = 0; k < 5; ++k) {
                std::vector<unsigned long long> obs(h.begin() + k * (L + 1), h.begin() + (k + 1) * (L + 1));
                std::vector<double> prob(L + 1);
                for (long i = 0; i <= L; ++i) prob[i] = ex[k][i] / nl;
                std::snprintf(params, sizeof params, "n=%lld r=%d s=%d L=%d %c", n, c.r, c.s, c.L, names[k]);
                report("walk", params, chi2(obs, prob, n), sec);
            }
            if (sum.max_z > o.limit) return sum;
        }
    }
    if (wanted(o, "periods")) {
        // PeriodsInStrings (N = 10, n = 5e8, s = 10): the correlations (the periods) of strings of s bits
        // against their populations among the 2^s strings, summed over N
        const int N = 10, sb = 10;
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
        for (const int r : {0, 20}) {
            const long long n = scaled(5e8);
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
            std::snprintf(params, sizeof params, "N=%d n=%lld r=%d s=%d", N, n, r, sb);
            report("periods", params, all, seconds_since(t0));
            if (sum.max_z > o.limit) return sum;
        }
    }
    if (wanted(o, "sumcoll")) {
        // SumCollector (N = 1, n = 5e8, r = 0, g = 10): the number of values until their sum exceeds g, against
        // P(N = j + 1) = F_j(g) - F_{j+1}(g) of the Irwin-Hall distributions F (in long double), lumped at 50
        const int jmax = 50, g10 = 10;
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
        const long long n = scaled(5e8);
        const auto t0 = std::chrono::steady_clock::now();
        const auto h = histogram(kn, detail::SumColl{0, jmax, static_cast<double>(g10)}, g, ctr, n, jmax + 1, 21, 0.4 * 21);
        std::snprintf(params, sizeof params, "n=%lld r=0 g=%d", n, g10);
        report("sumcoll", params, chi2(h, prob, n), seconds_since(t0));
        if (sum.max_z > o.limit) return sum;
    }
    if (wanted(o, "fourier")) {
        // Fourier3 (N = 1e5, k = 14, s = 3, r = 0 and 27): for each of N sequences of n = 2^14 bits (1 and -1)
        // from n / s + 1 values, the squares |F_i|^2 of its Fourier coefficients for i = 1..n/4, summed over
        // the N sequences; the sums divided by n have the mean N and the variance N (1 - 2 / n) (Erdmann), and
        // the AD test of the n / 4 standardized sums against the standard normal distribution
        const int k = 14, sb = 3, n = 1 << k, kall = n / sb + 1;
        const long long N = scaled(1e5);
        // the batch of the FFTs within a quarter of the budget (about 24 n bytes per sequence: its bits,
        // coefficients and the work buffer of rocFFT), a power of 2 in [64, 4096]
        long long batch = 64;
        while (batch < 4096 && static_cast<double>(batch) * 2 * 24 * n <= mem.budget / 4.0) batch *= 2;
        if (!mem.fft || mem.fft->n != n || mem.fft->batch != batch) {
            mem.fft.reset();
            mem.fft = std::make_unique<Fft>(n, batch);
        }
        Fft &ft = *mem.fft;
        for (const int r : {0, 27}) {
            const auto t0 = std::chrono::steady_clock::now();
            CK(hipMemset(ft.b, 0, sizeof(double) * (n / 4 + 1)));
            const uint64_t start = ctr.take((N * kall + Counters::kThreadsPerLaunch - 1) / Counters::kThreadsPerLaunch);
            for (long long done = 0; done < N; done += batch) {
                const long long nb = std::min(batch, N - done);
                launch(kn.fourier_bits, kBlocks, kThreads, 0, g, start + static_cast<uint64_t>(done) * kall, nb, kall, n, r, sb, ft.a);
                ft.execute();
                launch(kn.fourier_acc, (n / 4 + 255) / 256, 256, 0, static_cast<const double *>(ft.f), nb, n, ft.b);
            }
            std::vector<double> b(n / 4 + 1), u(n / 4);
            CK(hipMemcpy(b.data(), ft.b, sizeof(double) * b.size(), hipMemcpyDeviceToHost));
            for (int i = 1; i <= n / 4; ++i) u[i - 1] = fdist_Normal2((b[i] / n - N) / std::sqrt(N * (1.0 - 2.0 / n)));
            const auto ad = anderson_darling(u);
            std::snprintf(params, sizeof params, "N=%lld n=2^%d r=%d s=%d", N, k, r, sb);
            report_ad("fourier", params, ad.first, ad.second, seconds_since(t0));
            if (sum.max_z > o.limit) return sum;
        }
    }
    if (wanted(o, "linearcomp")) {
        // LinearComp (N = 1, n = 400020 bits, s = 1, r = 0 and 29): the linear complexity profile of n bits
        // (the s bits (x << r) >> (32 - s) of n / s values, the most significant first) by the Berlekamp-
        // Massey algorithm on the host: the number of its jumps against the normal distribution of the mean mu
        // and the standard deviation sigma of TestU01, and the sizes of the jumps against 2^-k (k < M0, the
        // last class lumped), a chi-square of M0 - 1 degrees of freedom; summed over the N sequences. (The
        // chi-square of the linear complexity, for N > 20 in TestU01, is not computed.)
        const int sb = 1;
        const long long N = 1;
        const long long K0 = scaled(400020) / sb, n = K0 * sb;
        const int par = static_cast<int>(n & 1);
        const double tmp = n >= DBL_MAX_EXP ? 0.0 : std::pow(2.0, -static_cast<double>(n));
        const double mu = n / 4.0 + (4 + par) / 12.0 - tmp / 3.0;
        const double sigma =
            std::sqrt(n / 8.0 - (2 - par) / (9.0 - par) + n * tmp / 6.0 + (6 + par) * tmp / 18.0 - tmp * tmp / 9.0);
        const int M0 = static_cast<int>(std::log2(mu / kMinExpected));
        if (M0 < 2) throw TestFailure("linearcomp: too few bits");
        // the bits of all the sequences from the GPU, then the Berlekamp-Massey algorithm of every sequence (its
        // time grows as n^2)
        const int rs[] = {0, 29};
        struct Job {
            std::vector<uint8_t> bits;
            long L = 0, numj = 0;
            std::vector<long> cnt;
        };
        std::vector<Job> jobs(2 * N);
        uint32_t *d_out;
        CK(hipMalloc(reinterpret_cast<void **>(&d_out), sizeof(uint32_t) * K0));
        std::vector<uint32_t> x(K0);
        const auto t0 = std::chrono::steady_clock::now();
        for (int ri = 0; ri < 2; ++ri) {
            const uint64_t start = ctr.take((N * K0 + Counters::kThreadsPerLaunch - 1) / Counters::kThreadsPerLaunch);
            for (long long q = 0; q < N; ++q) {
                launch(kn.cp_outputs, kBlocks, kThreads, 0, g, start + static_cast<uint64_t>(q * K0), K0, d_out);
                CK(hipMemcpy(x.data(), d_out, sizeof(uint32_t) * K0, hipMemcpyDeviceToHost));
                std::vector<uint8_t> &bits = jobs[ri * N + q].bits;
                bits.resize(n);
                for (long long i = 0; i < K0; ++i) {
                    const uint32_t v = (x[i] << rs[ri]) >> (32 - sb);
                    for (int j = 0; j < sb; ++j) bits[i * sb + j] = (v >> (sb - 1 - j)) & 1;
                }
            }
        }
        CK(hipFree(d_out));
        for (Job &jb : jobs) berlekamp_massey(jb.bits, M0, jb.L, jb.numj, jb.cnt);
        const double sec = seconds_since(t0);
        for (int ri = 0; ri < 2; ++ri) {
            double zs = 0;
            Chi2 jl{0, 0};
            for (long long q = 0; q < N; ++q) {
                const Job &jb = jobs[ri * N + q];
                zs += (jb.numj - mu) / sigma;
                for (int k = 1; k <= M0; ++k) {
                    const double e = jb.numj / std::ldexp(1.0, k < M0 ? k : M0 - 1);
                    jl.x += (jb.cnt[k] - e) * (jb.cnt[k] - e) / e;
                }
                jl.df += M0 - 1;
            }
            std::snprintf(params, sizeof params, "N=%lld n=%lld r=%d s=%d jumps", N, n, rs[ri], sb);
            report_normal("lincomp", params, zs / std::sqrt(static_cast<double>(N)), sec);
            std::snprintf(params, sizeof params, "N=%lld n=%lld r=%d s=%d sizes", N, n, rs[ri], sb);
            report("lincomp", params, jl, sec);
            if (sum.max_z > o.limit) return sum;
        }
    }
    if (wanted(o, "matrixrank")) {
        // MatrixRank (as BigCrush: N = 10, n = 1e6 matrices of 30 x 30 with s = 5, r = 0 and 25; N = 1, n = 5000 of
        // 1000 x 1000 with s = 4, r = 0 and 26; N = 1, n = 80 of 5000 x 5000, r = 15 with s = 15 and r = 0 with
        // s = 30): the ranks over GF(2) against their probabilities (TestU01, in log2), the chi-square summed
        // over N. The small matrices are made from the counters of every thread; the large ones in global memory
        // (a block each, batched within the memory budget) from consecutive values.
        const struct { int N; double n; int r, s, l, k; } set[] = {{10, 1e6, 0, 5, 30, 30}, {10, 1e6, 25, 5, 30, 30}, {1, 5000, 0, 4, 1000, 1000},
                                                                   {1, 5000, 26, 4, 1000, 1000}, {1, 80, 15, 15, 5000, 5000}, {1, 80, 0, 30, 5000, 5000}};
        for (const auto &c : set) {
            const long long n = std::max<long long>(scaled(c.n), 1);
            const int mkl = std::min(c.l, c.k), vpr = c.k / c.s + (c.k % c.s > 0);
            std::vector<double> prob(mkl + 1);
            double t2 = -static_cast<double>(c.l) * c.k;
            prob[0] = std::exp2(t2);
            for (int j = 1; j <= mkl; ++j) {
                t2 += c.l + c.k - 2 * j + 1 + std::log2(1 - std::exp2(-static_cast<double>(c.l - j + 1))) +
                      std::log2(1 - std::exp2(-static_cast<double>(c.k - j + 1))) - std::log2(1 - std::exp2(-static_cast<double>(j)));
                prob[j] = std::exp2(t2);
            }
            const auto t0 = std::chrono::steady_clock::now();
            Chi2 all{0, 0};
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
                    CK(hipMalloc(reinterpret_cast<void **>(&d_m), sizeof(unsigned long long) * per * batch));
                    CK(hipMalloc(reinterpret_cast<void **>(&d_rank), sizeof(int) * batch));
                    std::vector<int> rk(batch);
                    for (long long m0 = 0; m0 < n; m0 += batch) {
                        const long long nb = std::min(batch, n - m0);
                        launch(kn.mat_fill, kBlocks, kThreads, 0, g, start + static_cast<uint64_t>(m0 * c.l * vpr), nb * c.l, c.r, c.s, c.k, W, d_m);
                        launch(kn.mat_rank, static_cast<unsigned>(nb), kThreads, 0, d_m, c.l, c.k, W, d_rank);
                        CK(hipMemcpy(rk.data(), d_rank, sizeof(int) * nb, hipMemcpyDeviceToHost));
                        for (long long i = 0; i < nb; ++i) ++h[rk[i]];
                    }
                    CK(hipFree(d_m));
                    CK(hipFree(d_rank));
                }
                Chi2 x{0, 0};
                try {
                    x = chi2(h, prob, n);
                } catch (const TestFailure &) {  // too few matrices for a class (a small scale): no statistic, as TestU01
                }
                all.x += x.x;
                all.df += x.df;
            }
            std::snprintf(params, sizeof params, "N=%d n=%lld r=%d s=%d L=%d k=%d", c.N, n, c.r, c.s, c.l, c.k);
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
        const int sb = 3, L = 15, vpb = L / sb + (L % sb > 0), ibits = 64 - L;
        const long long Q = 10000000, K = std::max<long long>(scaled(1e9), 1), n = Q + K;
        const double sigma = std::sqrt((dCor[L] + eCor[L] * std::ldexp(1.0, L) / K) * KV[L] / K);
        for (const int r : {0, 27}) {
            const auto t0 = std::chrono::steady_clock::now();
            const uint64_t base = ctr.take((n * vpb + Counters::kThreadsPerLaunch - 1) / Counters::kThreadsPerLaunch);
            double somme = 0;
            sorted_parts(kn, mem, n, 64, [&](const Sink &sk) { launch(kn.appear, kBlocks, kThreads, 0, g, base, n, r, sb, L, ibits, sk); },
                         [&](unsigned long long *a, long long c) {
                             somme += thrust::transform_reduce(thrust::counting_iterator<long long>(0), thrust::counting_iterator<long long>(c),
                                                               AppearTerm{a, ibits, Q}, 0.0, thrust::plus<double>());
                         });
            std::snprintf(params, sizeof params, "N=1 Q=%lld K=%lld r=%d s=%d L=%d", Q, K, r, sb, L);
            report_normal("appear", params, (somme / (M_LN2 * K) - E[L]) / sigma, seconds_since(t0));
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
        const int t = 27;
        const long long nbits = 1LL << t, N = std::max<long long>(1, std::llround(10 * o.scale));
        for (const auto &c : {std::pair{0, 30}, std::pair{15, 15}}) {
            const int r = c.first, sb = c.second;
            const long long nv = (nbits + sb - 1) / sb;
            const auto t0 = std::chrono::steady_clock::now();
            const uint64_t start = ctr.take((N * nv + Counters::kThreadsPerLaunch - 1) / Counters::kThreadsPerLaunch);
            uint32_t *d_out;
            CK(hipMalloc(reinterpret_cast<void **>(&d_out), sizeof(uint32_t) * nv));
            std::vector<uint32_t> x(nv);
            double zs = 0;
            for (long long q = 0; q < N; ++q) {
                launch(kn.cp_outputs, kBlocks, kThreads, 0, g, start + static_cast<uint64_t>(q * nv), nv, d_out);
                CK(hipMemcpy(x.data(), d_out, sizeof(uint32_t) * nv, hipMemcpyDeviceToHost));
                zs += (lz78(x, nbits, r, sb) - LZMu[t]) / LZSigma[t];
            }
            CK(hipFree(d_out));
            std::snprintf(params, sizeof params, "N=%lld n=2^%d r=%d s=%d", N, t, r, sb);
            report_normal("lempelziv", params, zs / std::sqrt(static_cast<double>(N)), seconds_since(t0));
            if (sum.max_z > o.limit) return sum;
        }
    }
    return sum;
}

}  // namespace testu01_gpu
