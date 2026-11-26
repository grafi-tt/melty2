// SPDX-License-Identifier: Apache-2.0 AND LicenseRef-TestU01
//
// The device code of testu01_gpu: the tests as state machines over the outputs, and the kernels. The
// library compiles it at run time by hiprtc, followed by the source of the generator (see Generator in
// testu01_gpu.hpp), which defines testu01_gpu_generate; so the generator is inlined into the kernels as
// straight-line code. The library includes this file too, for the parameters of the kernels (the
// structs); the device functions and the kernels are compiled only by hiprtc (__HIPCC_RTC__).
#ifndef TESTU01_GPU_DEVICE_HPP_
#define TESTU01_GPU_DEVICE_HPP_

#ifndef __HIPCC_RTC__
#include <hip/hip_runtime.h>
#endif

#include "testu01_gpu_params.hpp"

#ifdef __HIPCC_RTC__
// the generator, defined by its source after this file: the outputs at the kLanes consecutive positions
// pos, pos + 1, ... of the sequence of the parameters p
__device__ inline void testu01_gpu_generate(const testu01_gpu::Params &p, uint64_t pos, uint32_t (&out)[testu01_gpu::kLanes]);
#endif

namespace testu01_gpu {
namespace detail {

constexpr int kThreads = 256;                  // threads per block

// floor(d * u) of u = (x << r) / 2^32
__device__ inline uint32_t value(uint32_t x, int r, uint32_t d) {
    return static_cast<uint32_t>((static_cast<uint64_t>(x << r) * d) >> 32);
}

// The tests as state machines over the outputs: step() returns the category of a completed
// observation, or -1.

// SimpPoker: the number of distinct values among k values in [0, d) (d <= 32)
struct Poker {
    int r, d, k;
    struct State {
        int cnt = 0;
        uint32_t mask = 0;
    };
    __device__ int step(State &s, uint32_t x) const {
        s.mask |= UINT32_C(1) << value(x, r, d);
        if (++s.cnt < k) return -1;
        const int m = __popc(s.mask);
        s.cnt = 0;
        s.mask = 0;
        return m;
    }
};

// CouponCollector: the number of values in [0, d) (d <= 32) until all d values appear, lumped at tmax
struct Coupon {
    int r, d, tmax;
    struct State {
        int len = 0;
        uint32_t mask = 0;
    };
    __device__ int step(State &s, uint32_t x) const {
        s.mask |= UINT32_C(1) << value(x, r, d);
        ++s.len;
        if (s.mask != (d == 32 ? ~UINT32_C(0) : (UINT32_C(1) << d) - 1)) return -1;
        const int c = min(s.len, tmax);
        s.len = 0;
        s.mask = 0;
        return c;
    }
};

// Gap: the number of values outside [alpha, beta) before one in it, lumped at tmax; the interval is
// [a, b) in units of 2^-32
struct Gap {
    int r, tmax;
    uint64_t a, b;
    struct State {
        int j = 0;
    };
    __device__ int step(State &s, uint32_t x) const {
        const uint64_t u = static_cast<uint32_t>(x << r);
        if (u < a || u >= b) {
            ++s.j;
            return -1;
        }
        const int c = min(s.j, tmax);
        s.j = 0;
        return c;
    }
};

// MaxOft: floor(d * max^t) of the maximum of t values u = x / 2^32 (r = 0), which is uniform
struct MaxOft {
    int d, t;
    struct State {
        int cnt = 0;
        uint32_t mx = 0;
    };
    __device__ int step(State &s, uint32_t x) const {
        s.mx = max(s.mx, x);
        if (++s.cnt < t) return -1;
        const double u = s.mx * 0x1p-32;
        double y = 1;
        for (int i = 0; i < t; ++i) y *= u;
        s.cnt = 0;
        s.mx = 0;
        return min(static_cast<int>(y * d), d - 1);
    }
};

// Permutation: the order of t non-overlapping values u = (x << r) / 2^32 (t <= 10), numbered as
// smultin_GenerCellPermut of TestU01 does (for i = t down to 2, the position of the first maximum of
// the first i values, which is then replaced by the i-th value), among the t! orders
struct Permut {
    int r, t;
    struct State {
        int cnt = 0;
        uint32_t u[10];
    };
    __device__ int step(State &s, uint32_t x) const {
        s.u[s.cnt] = x << r;
        if (++s.cnt < t) return -1;
        int cell = 0;
        for (int i = t; i >= 2; --i) {
            int m = 0;
            for (int j = 1; j < i; ++j)
                if (s.u[j] > s.u[m]) m = j;
            cell = cell * i + m;
            s.u[m] = s.u[i - 1];
        }
        s.cnt = 0;
        return cell;
    }
};

// WeightDistrib: the number of values u = (x << r) / 2^32 in [0, b) (in units of 2^-32; alpha = 0 in
// BigCrush) among k values
struct Weight {
    int r, k;
    uint64_t b;
    struct State {
        int cnt = 0, w = 0;
    };
    __device__ int step(State &s, uint32_t x) const {
        s.w += static_cast<uint32_t>(x << r) < b;
        if (++s.cnt < k) return -1;
        const int w = s.w;
        s.cnt = s.w = 0;
        return w;
    }
};

// Savir2: I = m, then I = 1 + floor(u * I) for t values u = (x << r) / 2^32, lumped at msup
struct Savir2 {
    int r, t;
    long long m, msup;
    struct State {
        int cnt = 0;
        long long i = 0;
    };
    __device__ int step(State &s, uint32_t x) const {
        if (s.cnt == 0) s.i = m;
        s.i = 1 + static_cast<long long>(static_cast<uint32_t>(x << r) * 0x1p-32 * static_cast<double>(s.i));
        if (++s.cnt < t) return -1;
        s.cnt = 0;
        return static_cast<int>(min(s.i, msup));
    }
};

// GCD: the gcd of pairs of s-bit values (x << r) >> (32 - s), a pair with a zero being drawn again,
// lumped at jmax
struct Gcd {
    int r, s, jmax;
    struct State {
        int have = 0;
        uint32_t u = 0;
    };
    __device__ int step(State &st, uint32_t x) const {
        const uint32_t v = (x << r) >> (32 - s);
        if (!st.have) {
            st.u = v;
            st.have = 1;
            return -1;
        }
        st.have = 0;
        uint32_t a = st.u, b = v;
        if (a == 0 || b == 0) return -1;
        while (b) {
            const uint32_t t = a % b;
            a = b;
            b = t;
        }
        return static_cast<int>(min(a, static_cast<uint32_t>(jmax)));
    }
};

// The blocks of L bits of the Hamming tests of sstring: ldivs values of s bits ((x << r) >> (32 - s))
// and, if lmods > 0, the first lmods bits of one more value; the weight of a block is its number of 1s.
// Modes: 0, the pairs of the weights of overlapping consecutive blocks (HammingCorr), as the category
// pre * (L + 1) + x; 1, of non-overlapping pairs (HammingIndep); 2, the weight of every block minus
// L / 2, plus h, clamped to [0, 2h] (HammingWeight2).
struct Hamming {
    int r, s, ldivs, lmods, L, mode, h;
    struct State {
        int cnt = 0, x = 0, pre = 0, have = 0;
    };
    __device__ int step(State &st, uint32_t v) const {
        const uint32_t y = v << r;
        st.x += __popc(st.cnt < ldivs ? y >> (32 - s) : y >> (32 - lmods));
        if (++st.cnt < ldivs + (lmods > 0)) return -1;
        const int x = st.x;
        st.cnt = st.x = 0;
        if (mode == 2) return min(max(x - L / 2 + h, 0), 2 * h);
        if (!st.have) {  // the first block of a pair (or of the thread, for overlapping pairs)
            st.pre = x;
            st.have = 1;
            return -1;
        }
        const int c = st.pre * (L + 1) + x;
        if (mode == 1) st.have = 0;
        else st.pre = x;
        return c;
    }
};

// SumCollector: the number of values u = (x << r) / 2^32 until their sum exceeds g, minus 1, lumped at jmax
struct SumColl {
    int r, jmax;
    double g;
    struct State {
        int j = 0;
        double x = 0;
    };
    __device__ int step(State &s, uint32_t v) const {
        s.x += static_cast<uint32_t>(v << r) * 0x1p-32;
        ++s.j;
        if (s.x <= g) return -1;
        const int c = min(s.j - 1, jmax);
        s.j = 0;
        s.x = 0;
        return c;
    }
};

// PeriodsInStrings: the string of s bits (x << r) >> (32 - s) of every value (the host maps the strings to
// their correlations)
struct Strip {
    int r, s;
    struct State {};
    __device__ int step(State &, uint32_t x) const { return static_cast<int>((x << r) >> (32 - s)); }
};

// MatrixRank of small matrices (l, k <= 32): the rank over GF(2) of l x k matrices, a row of every
// c = k / s values of s bits (x << r) >> (32 - s) and one more value of b = k % s bits if b > 0 (the order of
// the bits within a row doesn't change the rank, as long as it is the same for every row)
struct MatRank {
    int r, s, l, k;
    struct State {
        uint32_t row[32];
        int i = 0, j = 0;
        uint64_t cur = 0;
    };
    __device__ int step(State &st, uint32_t x) const {
        const int c = k / s, b = k % s, bits = st.j < c ? s : b;
        st.cur = st.cur << bits | (x << r) >> (32 - bits);
        if (++st.j < c + (b > 0)) return -1;
        st.row[st.i++] = static_cast<uint32_t>(st.cur);
        st.j = 0;
        st.cur = 0;
        if (st.i < l) return -1;
        st.i = 0;
        int rank = 0;
        for (int col = 0; col < k && rank < l; ++col) {  // Gaussian elimination
            const uint32_t m = UINT32_C(1) << col;
            int p = -1;
            for (int i = rank; i < l; ++i)
                if (st.row[i] & m) {
                    p = i;
                    break;
                }
            if (p < 0) continue;
            const uint32_t t = st.row[p];
            st.row[p] = st.row[rank];
            st.row[rank] = t;
            for (int i = rank + 1; i < l; ++i)
                if (st.row[i] & m) st.row[i] ^= t;
            ++rank;
        }
        return rank;
    }
};

// The points of CollisionOver and BirthdaySpacings: t values of b bits (d = 2^b) of the outputs (after
// dropping r bits), concatenated into a cell number of b * t <= 64 bits; overlapping (a point per value,
// from the last t values) or not (a point per t values).
struct Tuples {
    int r, b, t, overlap;
    uint64_t mask;  // 2^(b * t) - 1
    struct State {
        int cnt = 0;
        uint64_t cell = 0;
    };
    // whether a point is completed, in p
    __device__ bool step(State &s, uint32_t x, uint64_t &p) const {
        s.cell = (s.cell << b | (x << r) >> (32 - b)) & mask;
        if (++s.cnt < t) return false;
        if (!overlap) s.cnt = 0;
        p = s.cell;
        return true;
    }
};

// ClosePairs: n points of t coordinates (consecutive outputs x, i.e. u = x / 2^32) in the unit torus,
// and their pairs at a distance (the maximum over the coordinates of min(|x - y|, 2^32 - |x - y|))
// below dlim. The pairs are searched in a grid of c cells (c >= 3, of a width of at least dlim) along
// each of the first g coordinates (none if g = 0), so that a close pair is in the same cell or in
// neighbouring ones (with the wrap of the torus); the points are sorted by the cells.
struct CpGrid {
    int t, g;
    uint32_t c, dlim;
    uint32_t cells;  // c^g
};

constexpr int kCpMaxGrid = 8;  // the most gridded coordinates

// An open-addressing hash table of 64-bit keys on GPU, counting the insertions of keys already present
// (the collisions). The key kEmpty, which marks the empty slots, is counted separately. The keys are
// inserted in 2^pbits passes over the same keys, each taking those whose hash has the top pbits bits
// equal to part, so that the table of a pass is small enough for the cache of the GPU.
constexpr uint64_t kEmpty = ~UINT64_C(0);
struct HashTable {
    unsigned long long *slots, *collisions, *empties;
    uint64_t mask;  // the slots - 1 (a power of 2)
    int pbits;
    uint64_t part;
};

// Where the keys (points) made by a kernel go: into the hash table h if h.slots (then counted there, see
// insert, without being stored); with pbits = 0, all of them to out in their order; else only
// those of the part `part` of 2^pbits by the top pbits bits of the key (of kbits bits), or of its hash
// mix64(key) (64 bits; the hash, a bijection, is then stored instead of the key) if hash, appended at
// out[*count] (up to cap keys; count counts all of them, so that the host sees an overflow).
struct Sink {
    unsigned long long *out, *count;
    long long cap;
    int pbits, kbits, hash;
    uint64_t part;
    HashTable h;
};

#ifdef __HIPCC_RTC__

__device__ inline uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}


__device__ inline void insert(const HashTable &h, uint64_t key) {
    if (key == kEmpty) {
        if (h.part == 0) atomicAdd(h.empties, 1ULL);
        return;
    }
    const uint64_t z = mix64(key);
    if (h.pbits && z >> (64 - h.pbits) != h.part) return;  // another pass
    for (uint64_t i = z & h.mask;; i = (i + 1) & h.mask) {
        const unsigned long long prev = atomicCAS(&h.slots[i], kEmpty, key);
        if (prev == kEmpty) return;
        if (prev == key) {
            atomicAdd(h.collisions, 1ULL);
            return;
        }
    }
}

// put the key of the position pos (the index of the key in the order of the keys) into the sink
__device__ inline void put(const Sink &s, long long pos, uint64_t key) {
    if (s.h.slots) {
        insert(s.h, key);
        return;
    }
    if (s.pbits == 0) {
        s.out[pos] = key;
        return;
    }
    const uint64_t h = s.hash ? mix64(key) : key;
    if (h >> ((s.hash ? 64 : s.kbits) - s.pbits) != s.part) return;
    const unsigned long long j = atomicAdd(s.count, 1ULL);
    if (j < static_cast<unsigned long long>(s.cap)) s.out[j] = h;
}

// insert the keys a[0..n) into the hash table
extern "C" __global__ void kinsert(const unsigned long long *a, long long n, HashTable h) {
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += static_cast<long long>(gridDim.x) * blockDim.x) insert(h, a[i]);
}

// Every thread makes its share of the n observations of the test from its consecutive counters from
// base + tid * stride, and adds them to the histogram ghist of nbins categories (through LDS if lds).
// So the threads use consecutive pieces of a single stream from base, as TestU01 uses a single stream.
// A thread needing more than stride counters sets overrun, and stops.
template <class T>
__device__ void test_body(const T &test, const Params &g, uint64_t base, uint64_t stride, long long n, unsigned long long *ghist, int nbins, int lds, int *overrun) {
    extern __shared__ unsigned sh[];
    if (lds) {
        for (int i = threadIdx.x; i < nbins; i += kThreads) sh[i] = 0;
        __syncthreads();
    }
    const long long nthreads = static_cast<long long>(gridDim.x) * kThreads;
    const long long tid = static_cast<long long>(blockIdx.x) * kThreads + threadIdx.x;
    const long long quota = n / nthreads + (tid < n % nthreads);
    uint64_t ctr = base + tid * stride;
    const uint64_t end = ctr + stride;
    typename T::State s;
    long long done = 0;
    while (done < quota) {
        if (ctr >= end) {  // the counters of the next thread would be used twice
            atomicOr(overrun, 1);
            break;
        }
        uint32_t o[kLanes];
        testu01_gpu_generate(g, ctr, o);
        ctr += kLanes;
#pragma unroll
        for (int l = 0; l < kLanes; ++l) {
            if (done < quota) {
                const int c = test.step(s, o[l]);
                if (c >= 0) {
                    if (lds) atomicAdd(&sh[c], 1);
                    else atomicAdd(&ghist[c], 1);
                    ++done;
                }
            }
        }
    }
    if (lds) {
        __syncthreads();
        for (int i = threadIdx.x; i < nbins; i += kThreads)
            if (sh[i]) atomicAdd(&ghist[i], static_cast<unsigned long long>(sh[i]));
    }
}

#define TESTU01_GPU_TEST_KERNEL(name, T)                                                                                     \
    extern "C" __global__ void __launch_bounds__(kThreads) name(T test, Params g, uint64_t base, uint64_t stride, long long n, \
                                                                unsigned long long *ghist, int nbins, int lds, int *overrun) { \
        test_body(test, g, base, stride, n, ghist, nbins, lds, overrun);                                               \
    }
TESTU01_GPU_TEST_KERNEL(ktest_poker, Poker)
TESTU01_GPU_TEST_KERNEL(ktest_coupon, Coupon)
TESTU01_GPU_TEST_KERNEL(ktest_gap, Gap)
TESTU01_GPU_TEST_KERNEL(ktest_maxoft, MaxOft)
TESTU01_GPU_TEST_KERNEL(ktest_permut, Permut)
TESTU01_GPU_TEST_KERNEL(ktest_weight, Weight)
TESTU01_GPU_TEST_KERNEL(ktest_savir2, Savir2)
TESTU01_GPU_TEST_KERNEL(ktest_gcd, Gcd)
TESTU01_GPU_TEST_KERNEL(ktest_hamming, Hamming)
TESTU01_GPU_TEST_KERNEL(ktest_strip, Strip)
TESTU01_GPU_TEST_KERNEL(ktest_sumcoll, SumColl)
TESTU01_GPU_TEST_KERNEL(ktest_matrank, MatRank)
#undef TESTU01_GPU_TEST_KERNEL

// RandomWalk1 (swalk_RandomWalk1, L0 = L1 = L even): n random walks of L steps, the bits of L / s values
// (x << r) >> (32 - s) (most significant first) and of the first L % s bits of one more value, a step +1
// for a 1. The statistics of a walk, counted in hist (5 histograms of L + 1 classes, through LDS if lds):
// H, the number of steps +1, (L + S) / 2; M, the maximum of S; J, 2 times the number of odd steps with
// S > 0; R, the number of returns to 0; C, the number of sign changes (at the odd steps). Every thread
// makes its share of the walks from its counters from base + tid * stride.
extern "C" __global__ void __launch_bounds__(kThreads) kwalk(Params g, uint64_t base, uint64_t stride, long long n, int r, int s, int L,
                                                             unsigned long long *hist, int lds) {
    extern __shared__ unsigned sh[];
    const int nb = 5 * (L + 1);
    if (lds) {
        for (int i = threadIdx.x; i < nb; i += kThreads) sh[i] = 0;
        __syncthreads();
    }
    const long long nthreads = static_cast<long long>(gridDim.x) * kThreads;
    const long long tid = static_cast<long long>(blockIdx.x) * kThreads + threadIdx.x;
    const long long quota = n / nthreads + (tid < n % nthreads);
    uint64_t ctr = base + tid * stride;
    const int full = L / s, rest = L % s, per = full + (rest > 0);
    int S = 0, S2 = 0, M = 0, J = 0, R = 0, C = 0, step = 0, v = 0;
    long long done = 0;
    while (done < quota) {
        uint32_t o[kLanes];
        testu01_gpu_generate(g, ctr, o);
        ctr += kLanes;
#pragma unroll
        for (int l = 0; l < kLanes; ++l) {
            if (done >= quota) break;
            const int bits = v < full ? s : rest;
            const uint32_t z = (o[l] << r) >> (32 - bits);
            for (int i = bits - 1; i >= 0; --i) {
                ++step;
                S += (z >> i & 1) ? 1 : -1;
                M = max(M, S);
                R += S == 0;
                if (step & 1) {
                    J += S > 0;
                    C += S * S2 < 0;
                    S2 = S;
                }
            }
            if (++v < per) continue;
            const int idx[5] = {(L + S) / 2, M, 2 * J, R, C};
            for (int k = 0; k < 5; ++k) {
                const int b = k * (L + 1) + min(idx[k], L);
                if (lds) atomicAdd(&sh[b], 1u);
                else atomicAdd(&hist[b], 1ull);
            }
            S = S2 = M = J = R = C = step = v = 0;
            ++done;
        }
    }
    if (lds) {
        __syncthreads();
        for (int i = threadIdx.x; i < nb; i += kThreads)
            if (sh[i]) atomicAdd(&hist[i], static_cast<unsigned long long>(sh[i]));
    }
}

// The values of SampleProd and SampleMean, transformed by their distribution function F to U(0, 1) and
// written as keys floor(F 2^64) to out (in the order of the threads) for sorting. Every thread makes its
// share of the n values from its counters from base + tid * stride: SampleProd (mode 0), the product of t
// values u = (x << r) / 2^32, of F(x) = x sum_{j < t} (-ln x)^j / j!; SampleMean (mode 1), the sum of t
// values, of F the Irwin-Hall distribution by the cubic Hermite interpolation of the table tab of F and its
// density at the points i * hstep (i < ntab).
extern "C" __global__ void __launch_bounds__(kThreads) kuniform(int mode, int r, int t, Params g, uint64_t base, uint64_t stride, long long n, const double *tab,
                                                                int ntab, double hstep, Sink sk) {
    const long long nthreads = static_cast<long long>(gridDim.x) * kThreads;
    const long long tid = static_cast<long long>(blockIdx.x) * kThreads + threadIdx.x;
    const long long quota = n / nthreads + (tid < n % nthreads);
    long long pos = tid * (n / nthreads) + min(tid, n % nthreads);
    uint64_t ctr = base + tid * stride;
    double acc = mode ? 0.0 : 1.0;
    int cnt = 0;
    long long done = 0;
    while (done < quota) {
        uint32_t o[kLanes];
        testu01_gpu_generate(g, ctr, o);
        ctr += kLanes;
#pragma unroll
        for (int l = 0; l < kLanes; ++l) {
            if (done >= quota) break;
            const double u = static_cast<uint32_t>(o[l] << r) * 0x1p-32;
            acc = mode ? acc + u : acc * u;
            if (++cnt < t) continue;
            double f;
            if (mode == 0) {  // FDistProd of TestU01
                if (acc <= 0) f = 0;
                else {
                    const double lg = log(acc);
                    double sum = 1, vt = 1, jt = 1;
                    for (int j = 1; j < t; ++j) {
                        vt *= lg;
                        jt *= -j;
                        sum += vt / jt;
                    }
                    f = acc * sum;
                }
            } else {
                const double x = acc / hstep;
                const int i = min(max(static_cast<int>(x), 0), ntab - 2);
                const double s = x - i, F0 = tab[2 * i], d0 = tab[2 * i + 1] * hstep, F1 = tab[2 * i + 2], d1 = tab[2 * i + 3] * hstep;
                const double s2 = s * s, s3 = s2 * s;
                f = (2 * s3 - 3 * s2 + 1) * F0 + (s3 - 2 * s2 + s) * d0 + (-2 * s3 + 3 * s2) * F1 + (s3 - s2) * d1;
            }
            f = min(max(f, 0.0), 1.0);
            put(sk, pos++, f >= 1.0 ? ~UINT64_C(0) : static_cast<unsigned long long>(f * 0x1p64));
            acc = mode ? 0.0 : 1.0;
            cnt = 0;
            ++done;
        }
    }
}

// the share [a, e) of the positions [0, n) of a thread
#define TESTU01_GPU_SHARE(n)                                                                    \
    const long long nthreads = static_cast<long long>(gridDim.x) * kThreads;              \
    const long long tid = static_cast<long long>(blockIdx.x) * kThreads + threadIdx.x;    \
    const long long a = tid * ((n) / nthreads) + min(tid, (n) % nthreads), e = a + (n) / nthreads + (tid < (n) % nthreads)

// AutoCor (sstring_AutoCor): the number of differing bits at the distance d (1 <= d <= s) in the bits of
// the values (x << r) >> (32 - s) (most significant first) of the single sequence from start: the bits of
// the values [0, n) against those d later (the value n is read too); every thread adds that of its share.
extern "C" __global__ void __launch_bounds__(kThreads) kautocor(Params g, uint64_t start, long long n, int r, int s, int d, unsigned long long *cnt) {
    TESTU01_GPU_SHARE(n);
    if (a == e) return;
    unsigned long long c = 0;
    uint32_t cur = 0;
    for (long long q0 = a; q0 <= e; q0 += kLanes) {
        uint32_t o[kLanes];
        testu01_gpu_generate(g, start + q0, o);
#pragma unroll
        for (int l = 0; l < kLanes; ++l) {
            const long long q = q0 + l;
            const uint32_t v = (o[l] << r) >> (32 - s);
            if (q > a && q <= e) {  // the bits of the value q - 1 against those d later
                const uint32_t w = cur << s | v;
                c += __popc(((w ^ (w << d)) >> s) & ((1u << s) - 1));
            }
            cur = v;
        }
    }
    atomicAdd(cnt, c);
}

// SampleCorr (svaria_SampleCorr): the sum over i in [k, n) of u_{i-k} u_i - 1/4 of the values
// u = (x << r) / 2^32 of the single sequence from start; every thread adds that of its share of i.
extern "C" __global__ void __launch_bounds__(kThreads) ksamplecorr(Params g, uint64_t start, long long n, int r, int k, double *sum) {
    TESTU01_GPU_SHARE(n);
    const long long a0 = max(a, static_cast<long long>(k));
    if (a0 >= e) return;
    double buf[8], acc = 0;  // the last k values (k <= 8)
    long long q0 = a0 - k;
    int fill = 0;
    for (; q0 < e; q0 += kLanes) {
        uint32_t o[kLanes];
        testu01_gpu_generate(g, start + q0, o);
#pragma unroll
        for (int l = 0; l < kLanes; ++l) {
            const long long q = q0 + l;
            if (q >= e) break;
            const double u = static_cast<uint32_t>(o[l] << r) * 0x1p-32;
            if (fill >= k) acc += buf[q % k] * u - 0.25;
            else ++fill;
            buf[q % k] = u;
        }
    }
    atomicAdd(sum, acc);
}

// The runs of bits (sstring_Run): the bits of the values (x << r) >> (32 - s) (most significant first)
// of the single sequence from start; a run ends where the next bit differs. Every thread counts the runs
// ending in its share of the values (from k bits before it, to know their lengths up to k), by their
// length lumped at k: cnt[0..k-1] for the runs of 0s, cnt[k..2k-1] of 1s.
extern "C" __global__ void __launch_bounds__(kThreads) kbitrun(Params g, uint64_t start, long long n, int r, int s, int k, unsigned long long *cnt) {
    TESTU01_GPU_SHARE(n);
    if (a == e) return;
    const long long back = (k + s - 1) / s + 1;
    const long long p0 = a > back ? a - back : 0;
    int prev = -1, len = 0;
    unsigned c[2][40] = {};
    for (long long q0 = p0; q0 < e; q0 += kLanes) {
        uint32_t o[kLanes];
        testu01_gpu_generate(g, start + q0, o);
#pragma unroll
        for (int l = 0; l < kLanes; ++l) {
            const long long q = q0 + l;
            if (q >= e) break;
            const uint32_t v = (o[l] << r) >> (32 - s);
            for (int j = s - 1; j >= 0; --j) {
                const int bit = v >> j & 1;
                if (bit == prev) {
                    ++len;
                    continue;
                }
                // a run of prev of the length len ends before this bit, counted if this bit is in the
                // share (a run begun before the first value read is at least k long: lumped at k anyway)
                if (prev >= 0 && q >= a) ++c[prev][min(len, k) - 1];
                prev = bit;
                len = 1;
            }
        }
    }
    for (int b = 0; b < 2; ++b)
        for (int i = 0; i < k; ++i)
            if (c[b][i]) atomicAdd(&cnt[b * k + i], static_cast<unsigned long long>(c[b][i]));
}

// LongestHeadRun (sstring_LongestHeadRun): the longest run of 1s in each of nb blocks of K values of
// s bits ((x << r) >> (32 - s), most significant first) from start, a block per workgroup: every thread
// summarizes its consecutive values (the leading and trailing runs of 1s, the longest inside, all 1s),
// and the summaries are combined in order.
struct RunSummary {
    long long pre, suf, best;
    int full;
};
__device__ inline RunSummary combine(const RunSummary &x, const RunSummary &y) {
    RunSummary z;
    z.full = x.full && y.full;
    z.pre = x.full ? x.pre + y.pre : x.pre;
    z.suf = y.full ? x.suf + y.suf : y.suf;
    z.best = max(max(x.best, y.best), x.suf + y.pre);
    z.best = max(z.best, max(z.pre, z.suf));
    return z;
}
extern "C" __global__ void __launch_bounds__(kThreads) klonghead(Params g, uint64_t start, long long K, int r, int s, long long *longest) {
    __shared__ RunSummary sh[kThreads];
    const long long blk = blockIdx.x, t = threadIdx.x;
    const long long a = t * (K / kThreads) + min(t, K % kThreads), e = a + K / kThreads + (t < K % kThreads);
    RunSummary m{0, 0, 0, 1};
    for (long long q0 = a; q0 < e; q0 += kLanes) {
        uint32_t o[kLanes];
        testu01_gpu_generate(g, start + blk * K + q0, o);
#pragma unroll
        for (int l = 0; l < kLanes; ++l) {
            if (q0 + l >= e) break;
            const uint32_t v = (o[l] << r) >> (32 - s);
            for (int j = s - 1; j >= 0; --j) {
                if (v >> j & 1) {
                    if (m.full) ++m.pre;
                    ++m.suf;
                    m.best = max(m.best, m.suf);
                } else {
                    m.full = 0;
                    m.suf = 0;
                }
            }
        }
    }
    sh[t] = m;
    __syncthreads();
    for (int w = 1; w < kThreads; w <<= 1) {  // combine the summaries in order, pairwise
        if (t % (2 * w) == 0 && t + w < kThreads) sh[t] = combine(sh[t], sh[t + w]);
        __syncthreads();
    }
    if (t == 0) longest[blk] = sh[0].best;
}

// SerialOver (smultin_MultinomialOver, dense): the overlapping t-tuples of n values in [0, 2^b) (the top
// b bits of x << r) of the single sequence from the position start, made circular as in TestU01 (the
// last t - 1 tuples wrap to the first values), counted in cnt (2^(b t) cells) and the (t - 1)-tuples in
// cnt1. Every thread takes the tuples starting in its share of the positions.
extern "C" __global__ void __launch_bounds__(kThreads) kserial(Params g, uint64_t start, long long n, int r, int b, int t, unsigned *cnt, unsigned *cnt1) {
    const long long nthreads = static_cast<long long>(gridDim.x) * kThreads;
    const long long tid = static_cast<long long>(blockIdx.x) * kThreads + threadIdx.x;
    const long long a = tid * (n / nthreads) + min(tid, n % nthreads), e = a + n / nthreads + (tid < n % nthreads);
    if (a == e) return;
    const uint64_t mask = (UINT64_C(1) << (b * t)) - 1, mask1 = (UINT64_C(1) << (b * (t - 1))) - 1;
    uint64_t cell = 0;
    long long have = 0;  // the values in cell
    for (long long q0 = a; q0 < e + t - 1; q0 += kLanes) {
        // the values from q0 (wrapping past n to the first ones), by kLanes from the position q0 mod n
        uint32_t o[kLanes];
        const long long w0 = q0 % n;
        if (w0 + kLanes <= n) testu01_gpu_generate(g, start + w0, o);
        else
            for (int l = 0; l < kLanes; ++l) {
                uint32_t oo[kLanes];
                const long long w = (q0 + l) % n;
                testu01_gpu_generate(g, start + w, oo);
                o[l] = oo[0];
            }
#pragma unroll
        for (int l = 0; l < kLanes; ++l) {
            if (q0 + l >= e + t - 1) break;
            cell = (cell << b | (o[l] << r) >> (32 - b)) & mask;
            if (++have >= t) {  // the tuple starting at q0 + l - t + 1
                atomicAdd(&cnt[cell], 1u);
                atomicAdd(&cnt1[cell >> b & mask1], 1u);  // its first t - 1 values
            }
        }
    }
}

// CollisionOver of permutations (sknuth_CollisionPermut): the cell (in [0, t!), t <= 20) of the order of
// t non-overlapping values u = (x << r) / 2^32, numbered as smultin_GenerCellPermut; the points of n cells
// from the counters of every thread are put into the sink
extern "C" __global__ void __launch_bounds__(kThreads) kperm_points(int r, int t, Params g, uint64_t base, uint64_t stride, long long n, Sink sk) {
    const long long nthreads = static_cast<long long>(gridDim.x) * kThreads;
    const long long tid = static_cast<long long>(blockIdx.x) * kThreads + threadIdx.x;
    const long long quota = n / nthreads + (tid < n % nthreads);
    long long pos = tid * (n / nthreads) + min(tid, n % nthreads);
    uint64_t ctr = base + tid * stride;
    uint32_t u[20];
    int cnt = 0;
    long long done = 0;
    while (done < quota) {
        uint32_t o[kLanes];
        testu01_gpu_generate(g, ctr, o);
        ctr += kLanes;
#pragma unroll
        for (int l = 0; l < kLanes; ++l) {
            if (done >= quota) break;
            u[cnt] = o[l] << r;
            if (++cnt < t) continue;
            cnt = 0;
            uint64_t cell = 0;
            for (int i = t; i >= 2; --i) {
                int m = 0;
                for (int j = 1; j < i; ++j)
                    if (u[j] > u[m]) m = j;
                cell = cell * i + m;
                u[m] = u[i - 1];
            }
            put(sk, pos++, cell);
            ++done;
        }
    }
}

// Run (sknuth_Run): the runs up (or down) of n consecutive values u = (x << r) / 2^32 from the position
// start, counted by their length lumped at 6 as TestU01 does (a run ends at a value below the previous one
// for up, above it for down; equal values continue it). Every thread takes its share of the positions,
// and starts 7 values before it (where it can) to know the length of a run ending in it (lumped at 6), so
// the counts are exactly those of the single sequence. cnt[0..5]: the runs of length 1..6+.
extern "C" __global__ void __launch_bounds__(kThreads) krun(Params g, uint64_t start, long long n, int r, int up, unsigned long long *cnt) {
    const long long nthreads = static_cast<long long>(gridDim.x) * kThreads;
    const long long tid = static_cast<long long>(blockIdx.x) * kThreads + threadIdx.x;
    const uint64_t a = start + tid * (n / nthreads) + min(tid, n % nthreads), b = a + n / nthreads + (tid < n % nthreads);
    if (a == b) return;
    const uint64_t p0 = a - start > 7 ? a - 7 : start;
    uint64_t last = p0 == start ? start : p0 - 6;  // the start of the current run (or one 6 values before)
    unsigned c[6] = {0, 0, 0, 0, 0, 0};
    uint32_t prev = 0;
    for (uint64_t q0 = p0; q0 < b; q0 += kLanes) {
        uint32_t o[kLanes];
        testu01_gpu_generate(g, q0, o);
#pragma unroll
        for (int l = 0; l < kLanes; ++l) {
            const uint64_t q = q0 + l;
            const uint32_t u = o[l] << r;
            if (q < b && q > p0 && (up ? u < prev : u > prev)) {
                if (q >= a) ++c[min(q - last, UINT64_C(6)) - 1];
                last = q;
            }
            prev = u;
        }
    }
    if (b == start + n) ++c[min(b - last, UINT64_C(6)) - 1];  // the last run
    for (int i = 0; i < 6; ++i)
        if (c[i]) atomicAdd(&cnt[i], static_cast<unsigned long long>(c[i]));
}

// Every thread makes its share of the n points of the tuples from its consecutive counters from
// base + tid * stride (the stride covers the values of its points), and puts them into the sink (in the
// order of the threads).
extern "C" __global__ void __launch_bounds__(kThreads) kpoints(Tuples tup, Params g, uint64_t base, uint64_t stride, long long n, Sink sk) {
    const long long nthreads = static_cast<long long>(gridDim.x) * kThreads;
    const long long tid = static_cast<long long>(blockIdx.x) * kThreads + threadIdx.x;
    const long long quota = n / nthreads + (tid < n % nthreads);
    long long pos = tid * (n / nthreads) + min(tid, n % nthreads);
    uint64_t ctr = base + tid * stride;
    Tuples::State s;
    long long done = 0;
    while (done < quota) {
        uint32_t o[kLanes];
        testu01_gpu_generate(g, ctr, o);
        ctr += kLanes;
#pragma unroll
        for (int l = 0; l < kLanes; ++l) {
            uint64_t p;
            if (done < quota && tup.step(s, o[l], p)) {
                put(sk, pos++, p);
                ++done;
            }
        }
    }
}

// MatrixRank of large matrices: the rows of nmat l x k matrices of W 64-bit words per row (the row j of all
// of them in order takes vpr values from base + j vpr, c = k / s values of s bits (x << r) >> (32 - s) and
// one more of b = k % s bits if b > 0), a thread per row
extern "C" __global__ void __launch_bounds__(kThreads) kmat_fill(Params g, uint64_t base, long long rows, int r, int s, int k, int W,
                                                                 unsigned long long *out) {
    const int c = k / s, b = k % s, vpr = c + (b > 0);
    for (long long j = static_cast<long long>(blockIdx.x) * kThreads + threadIdx.x; j < rows; j += static_cast<long long>(gridDim.x) * kThreads) {
        unsigned long long *row = out + j * W;
        for (int w = 0; w < W; ++w) row[w] = 0;
        uint32_t o[kLanes];
        uint64_t pos = base + static_cast<uint64_t>(j) * vpr;
        int have = 0, at = 0, bit = 0;
        for (int v = 0; v < vpr; ++v) {
            if (at == have) {
                testu01_gpu_generate(g, pos, o);
                pos += kLanes;
                have = kLanes;
                at = 0;
            }
            const int bits = v < c ? s : b;
            const uint64_t y = (o[at++] << r) >> (32 - bits);
            row[bit >> 6] |= y << (bit & 63);
            if ((bit & 63) + bits > 64) row[(bit >> 6) + 1] |= y >> (64 - (bit & 63));
            bit += bits;
        }
    }
}

// MatrixRank of large matrices: the rank over GF(2) of every matrix (a block each) by the forward Gaussian
// elimination of its rows in place; the rows at and below the rank are zero in the columns before the
// current one, so that only their words from the column on are read and written.
extern "C" __global__ void __launch_bounds__(kThreads) kmat_rank(unsigned long long *mats, int l, int k, int W, int *ranks) {
    __shared__ int piv;
    __shared__ unsigned flag[(5000 + 31) / 32 + 1];  // the rows with the bit of the column (l <= 5000)
    __shared__ unsigned long long prow[128];           // the pivot row (W <= 128)
    unsigned long long *M = mats + static_cast<long long>(blockIdx.x) * l * W;
    int rank = 0;
    for (int col = 0; col < k && rank < l; ++col) {
        const int w = col >> 6;
        const unsigned long long m = 1ULL << (col & 63);
        if (threadIdx.x == 0) piv = l;
        for (int i = threadIdx.x; i < (l + 31) / 32; i += kThreads) flag[i] = 0;
        __syncthreads();
        for (int i = rank + threadIdx.x; i < l; i += kThreads)
            if (M[static_cast<long long>(i) * W + w] & m) {
                atomicMin(&piv, i);
                atomicOr(&flag[i >> 5], 1u << (i & 31));
            }
        __syncthreads();
        const int p = piv;
        __syncthreads();  // (every thread has read piv before it is reset)
        if (p == l) continue;
        // swap the rows p and rank (from the word w), keeping the pivot row in LDS
        for (int x = w + threadIdx.x; x < W; x += kThreads) {
            const unsigned long long a = M[static_cast<long long>(p) * W + x];
            if (p != rank) M[static_cast<long long>(p) * W + x] = M[static_cast<long long>(rank) * W + x];
            M[static_cast<long long>(rank) * W + x] = a;
            prow[x] = a;
        }
        if (threadIdx.x == 0 && p != rank) {  // the flag of the row moved to p
            const bool fr = (flag[rank >> 5] >> (rank & 31)) & 1;
            if (fr) atomicOr(&flag[p >> 5], 1u << (p & 31));
            else atomicAnd(&flag[p >> 5], ~(1u << (p & 31)));
        }
        __syncthreads();
        // eliminate the column from the rows below the pivot
        const int nw = W - w;
        const long long cells = static_cast<long long>(l - rank - 1) * nw;
        for (long long q = threadIdx.x; q < cells; q += kThreads) {
            const int i = rank + 1 + static_cast<int>(q / nw), x = w + static_cast<int>(q % nw);
            if ((flag[i >> 5] >> (i & 31)) & 1) M[static_cast<long long>(i) * W + x] ^= prow[x];
        }
        __syncthreads();
        ++rank;
    }
    if (threadIdx.x == 0) ranks[blockIdx.x] = rank;
}

// AppearanceSpacings: the keys (block << ibits | i) of the n blocks i of L bits from base (vpb values each: L / s
// values of s bits (x << r) >> (32 - s), and one of L % s bits if any), into the sink at their positions i; a
// thread takes 16 consecutive blocks
extern "C" __global__ void __launch_bounds__(kThreads) kappear(Params g, uint64_t base, long long n, int r, int s, int L, int ibits, Sink sk) {
    const int c = L / s, b = L % s, vpb = c + (b > 0);
    const long long groups = (n + 15) / 16;
    for (long long q = static_cast<long long>(blockIdx.x) * kThreads + threadIdx.x; q < groups; q += static_cast<long long>(gridDim.x) * kThreads) {
        uint32_t o[kLanes];
        uint64_t pos = base + static_cast<uint64_t>(q) * 16 * vpb;
        int have = 0, at = 0;
        for (int j = 0; j < 16; ++j) {
            const long long i = q * 16 + j;
            uint64_t block = 0;
            for (int v = 0; v < vpb; ++v) {
                if (at == have) {
                    testu01_gpu_generate(g, pos, o);
                    pos += kLanes;
                    have = kLanes;
                    at = 0;
                }
                const int bits = v < c ? s : b;
                block = block << bits | (o[at++] << r) >> (32 - bits);
            }
            if (i < n) put(sk, i, block << ibits | static_cast<uint64_t>(i));
        }
    }
}

// the spacings y[i + 1] - y[i] of the n sorted points y inserted into the hash table (h.slots), or written to
// d[i] (n - 1 of them)
extern "C" __global__ void kspacings(const unsigned long long *y, long long n, HashTable h, unsigned long long *d) {
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i + 1 < n; i += static_cast<long long>(gridDim.x) * blockDim.x) {
        if (h.slots) insert(h, y[i + 1] - y[i]);
        else d[i] = y[i + 1] - y[i];
    }
}

// Fourier3: the bits of nseq sequences of n bits, each from kall consecutive values from base + q * kall
// (the strings of s bits (x << r) >> (32 - s), the most significant bit first; the bits beyond n are
// dropped), as 1.0 for a 1 and -1.0 for a 0, to a[q * n + i]. A thread takes kLanes values.
extern "C" __global__ void __launch_bounds__(kThreads) kfourier_bits(Params g, uint64_t base, long long nseq, int kall, int n, int r, int s, double *a) {
    const int chunks = (kall + kLanes - 1) / kLanes;
    for (long long c = static_cast<long long>(blockIdx.x) * kThreads + threadIdx.x; c < nseq * chunks; c += static_cast<long long>(gridDim.x) * kThreads) {
        const long long q = c / chunks;
        const int k0 = static_cast<int>(c % chunks) * kLanes;
        uint32_t o[kLanes];
        testu01_gpu_generate(g, base + static_cast<uint64_t>(q) * kall + k0, o);
        double *row = a + q * n;
#pragma unroll
        for (int l = 0; l < kLanes; ++l) {
            const int k = k0 + l;
            if (k >= kall) break;
            const uint32_t z = (o[l] << r) >> (32 - s);
            for (int j = 0; j < s; ++j) {
                const int i = k * s + j;
                if (i < n) row[i] = (z >> (s - 1 - j)) & 1 ? 1.0 : -1.0;
            }
        }
    }
}

// Fourier3: b[i] += |F_i|^2 (in double) over the nseq transforms of n / 2 + 1 complex coefficients each, for
// i = 1..n/4 (a thread per i, so the sums are deterministic)
extern "C" __global__ void kfourier_acc(const double *f, long long nseq, int n, double *b) {
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    if (i > n / 4) return;
    const long long nc = n / 2 + 1;
    double acc = 0;
    for (long long q = 0; q < nseq; ++q) {
        const double re = f[2 * (q * nc + i)], im = f[2 * (q * nc + i) + 1];
        acc += re * re + im * im;
    }
    b[i] += acc;
}

// the count consecutive outputs from the position base of the sequence, to out
extern "C" __global__ void __launch_bounds__(kThreads) kcp_outputs(Params g, uint64_t base, long long count, uint32_t *out) {
    const long long chunks = (count + kLanes - 1) / kLanes;
    for (long long q = static_cast<long long>(blockIdx.x) * kThreads + threadIdx.x; q < chunks; q += static_cast<long long>(gridDim.x) * kThreads) {
        uint32_t o[kLanes];
        testu01_gpu_generate(g, base + static_cast<uint64_t>(q) * kLanes, o);
#pragma unroll
        for (int l = 0; l < kLanes; ++l)
            if (q * kLanes + l < count) out[q * kLanes + l] = o[l];
    }
}

// The points i < n of t <= kLanes coordinates (the outputs from the position base + i t) whose first
// coordinate is in [lo, lo + span) of the torus, appended to out (up to cap points; count counts all).
// ClosePairs takes the points by slabs of the first coordinate this way, when they don't fit in memory.
extern "C" __global__ void __launch_bounds__(kThreads) kcp_slab(Params g, uint64_t base, long long n, int t, uint32_t lo, uint64_t span, uint32_t *out,
                                                                unsigned long long *count, long long cap) {
    for (long long i = static_cast<long long>(blockIdx.x) * kThreads + threadIdx.x; i < n; i += static_cast<long long>(gridDim.x) * kThreads) {
        uint32_t o[kLanes];
        testu01_gpu_generate(g, base + static_cast<uint64_t>(i) * t, o);
        if (static_cast<uint32_t>(o[0] - lo) >= span) continue;
        const unsigned long long j = atomicAdd(count, 1ULL);
        if (j < static_cast<unsigned long long>(cap))
            for (int d = 0; d < t; ++d) out[j * t + d] = o[d];
    }
}

// the cell of a coordinate x along a gridded coordinate, and the cell of a point (along the first g)
__device__ inline uint32_t cp_coord_cell(uint32_t x, const CpGrid &gr) { return static_cast<uint32_t>((static_cast<uint64_t>(x) * gr.c) >> 32); }
__device__ inline uint32_t cp_cell(const uint32_t *x, const CpGrid &gr) {
    uint32_t id = 0;
    for (int d = gr.g - 1; d >= 0; --d) id = id * gr.c + cp_coord_cell(x[d], gr);
    return id;
}

// the cell of every point, and the number of points per cell
extern "C" __global__ void kcp_cells(const uint32_t *pts, long long n, CpGrid gr, uint32_t *cell, unsigned *count) {
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const uint32_t id = cp_cell(pts + i * gr.t, gr);
        cell[i] = id;
        atomicAdd(&count[id], 1);
    }
}

// the points sorted by the cells (in any order within a cell), by the cursors of the cells
extern "C" __global__ void kcp_scatter(const uint32_t *pts, long long n, int t, const uint32_t *cell, unsigned long long *cursor, uint32_t *out) {
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const unsigned long long j = atomicAdd(&cursor[cell[i]], 1);
        for (int d = 0; d < t; ++d) out[j * t + d] = pts[i * t + d];
    }
}

// The distances below dlim of the pairs of the points sorted by the cells, to dist (the first cap of
// them; ndist counts all), but those of two points outside the core [core_lo, core_lo + core_w) of the
// first coordinate (the points of a slab and of its margin below; core_w = 2^32 for all). Every point i looks at the points j > i of its cell and all the points of
// the neighbouring cells of the offsets after the zero offset (in the order of the offsets in
// {-1, 0, 1}^g), so that every pair is seen once (c >= 3 keeps the neighbours distinct). The
// coordinates outside the grid are compared first, as the gridded ones are close already.
extern "C" __global__ void __launch_bounds__(kThreads) kcp_pairs(const uint32_t *pts, long long n, CpGrid gr, const unsigned long long *start, const unsigned *count,
                                                                 uint32_t core_lo, uint64_t core_w, uint32_t *dist, unsigned *ndist, unsigned cap) {
    for (long long i = static_cast<long long>(blockIdx.x) * kThreads + threadIdx.x; i < n; i += static_cast<long long>(gridDim.x) * kThreads) {
        const uint32_t *p = pts + i * gr.t;
        const bool pcore = static_cast<uint32_t>(p[0] - core_lo) < core_w;
        uint32_t cc[kCpMaxGrid];  // the cell along every gridded coordinate
        for (int d = 0; d < gr.g; ++d) cc[d] = cp_coord_cell(p[d], gr);
        int noff = 1;
        for (int d = 0; d < gr.g; ++d) noff *= 3;
        for (int k = (noff - 1) / 2; k < noff; ++k) {
            uint32_t id = 0;
            int kk = k, digit[kCpMaxGrid];
            for (int d = 0; d < gr.g; ++d) {
                digit[d] = kk % 3;
                kk /= 3;
            }
            for (int d = gr.g - 1; d >= 0; --d) id = id * gr.c + (cc[d] + gr.c + static_cast<uint32_t>(digit[d]) - 1) % gr.c;
            const long long b = static_cast<long long>(start[id]), e = b + count[id];
            for (long long j = k == (noff - 1) / 2 ? i + 1 : b; j < e; ++j) {
                const uint32_t *q = pts + j * gr.t;
                if (!pcore && static_cast<uint32_t>(q[0] - core_lo) >= core_w) continue;  // a pair of the slab below
                uint32_t m = 0;
                for (int d = gr.g; d < gr.t + gr.g && m < gr.dlim; ++d) {  // the coordinates outside the grid first
                    const int dd = d < gr.t ? d : d - gr.t;
                    const uint32_t a = p[dd] - q[dd];
                    m = max(m, min(a, 0u - a));
                }
                if (m >= gr.dlim) continue;
                const unsigned s = atomicAdd(ndist, 1);
                if (s < cap) dist[s] = m;
            }
        }
    }
}

#endif  // __HIPCC_RTC__

}  // namespace detail
}  // namespace testu01_gpu

#endif
