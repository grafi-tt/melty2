// SPDX-License-Identifier: Apache-2.0
// GPU implementation (HIP or CUDA, see gpu_runtime.hpp) of metric::bias_per_key() in metric.hpp.
//
// The structure is interpreted on the device: ops are packed into constant memory, and each thread
// traces kPairs counter pairs. Histograms of up to kMaxGroup depths at a time are accumulated in shared memory,
// and then in global memory; the chi-square statistics are reduced on the device.
//
// The transfers are asynchronous on a stream, through pinned host buffers, and the host waits only
// for the results at the end of bias_per_key(); it waits by blocking, not by spinning.
#include <algorithm>
#include <cstring>

#include "gpu_runtime.hpp"
#include "logging.hpp"
#include "metric.hpp"

namespace metric {

namespace {

constexpr int kPairs = 8;                  // counter pairs per thread
constexpr int kTraces = 2 * kPairs;        // traces per thread
constexpr int kThreads = 256;              // threads per block
constexpr int kBins = 2 * kWindows * 256;  // bins of the histograms of a depth (addition and xor)
// maximum depths per launch: the histograms of the depths in the shared memory of a block (4 in the 64 KiB of the LDS
// of HIP, 3 in the 48 KiB of CUDA); the depths are independent, so the results don't depend on it
constexpr int kMaxGroup = kGpuStaticSharedBytes / static_cast<int>(kBins * sizeof(unsigned));
static_assert(kMaxGroup >= 3);

}  // namespace

__constant__ int c_op[arx::kMaxOps + 1];  // (t * 36 + d * 6 + s) | rot << 8

namespace {

// SplitMix64 finalizer
__device__ uint64_t smix(uint64_t z) {
    z += UINT64_C(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}
// r in [1, 31]
__device__ uint32_t rotl_u(uint32_t x, int r) { return x << r | x >> (32 - r); }

// histogram the differences of the kPairs pairs of traces o (o[p] and o[kPairs + p]) into H
__device__ void hist_add(unsigned *H, const uint32_t (&o)[kTraces]) {
#pragma unroll
    for (int p = 0; p < kPairs; ++p) {
        const uint32_t ad = o[kPairs + p] - o[p], xd = o[kPairs + p] ^ o[p];
#pragma unroll
        for (int w = 0; w < kWindows; ++w) {
            const uint32_t a2 = w ? rotl_u(ad, 4 * w) : ad, x2 = w ? rotl_u(xd, 4 * w) : xd;
            atomicAdd(&H[w * 256 + (a2 >> 24)], 1);
            atomicAdd(&H[(kWindows + w) * 256 + (x2 >> 24)], 1);
        }
    }
}

// The cases of the switch over the packed op codes: every (type, d, s) has its own case, so that the
// words are indexed by constants and stay in registers.
#define CASE_A(d, s) case 0 * 36 + d * 6 + s: _Pragma("unroll") for (int l = 0; l < kTraces; ++l) v[d][l] += v[s][l]; break;
#define CASE_X(d, s) case 1 * 36 + d * 6 + s: _Pragma("unroll") for (int l = 0; l < kTraces; ++l) v[d][l] ^= v[s][l]; break;
#define CASE_R(d, s) case 2 * 36 + d * 6 + s: _Pragma("unroll") for (int l = 0; l < kTraces; ++l) v[d][l] = rotl_u(v[d][l] ^ v[s][l], r); break;
#define CASE_S(d, s) case 3 * 36 + d * 6 + s: _Pragma("unroll") for (int l = 0; l < kTraces; ++l) v[d][l] -= v[s][l]; break;
#define CASES_DS(M, d) M(d, 0) M(d, 1) M(d, 2) M(d, 3) M(d, 4) M(d, 5)
#define CASES(M) CASES_DS(M, 0) CASES_DS(M, 1) CASES_DS(M, 2) CASES_DS(M, 3) CASES_DS(M, 4) CASES_DS(M, 5)

// Histogram the output differences for the G depths dlo..dlo + G - 1 (G <= kMaxGroup, as the shared memory holds
// kMaxGroup histogram sets), running the first nops ops (nops = dlo + G - 1, the last depth of the
// launch). Every G has its own kernel, so that a launch for fewer depths uses less shared memory. A thread traces
// kPairs pairs of counters c and c + delta (kTraces traces) iters times, with c from a SplitMix64
// stream of the seed, the key and delta, and the thread. The histograms are accumulated in shared memory
// and then added to ghist.
// grid: x = blocks per (key, delta), y = key * kDeltas + delta
template <int G>
__global__ void __launch_bounds__(kThreads) kdm(int nops, int dlo, const uint32_t *keys, uint64_t seed, int iters, unsigned *ghist, int nd, int mlo) {
    __shared__ unsigned hist[G * kBins];
    const int dhi = dlo + G - 1;
    for (int i = threadIdx.x; i < G * kBins; i += kThreads) hist[i] = 0;
    __syncthreads();
    const int kd = blockIdx.y, key = kd / kDeltas, di = kd % kDeltas;
    const int j = di < 8 ? di : 32 + di - 8;
    const uint64_t delta = UINT64_C(1) << j;
    const uint32_t *k = keys + 6 * key;
    const uint32_t k0 = k[0], k1 = k[1], k2 = k[2], k3 = k[3], k4 = k[4], k5 = k[5];
    const uint64_t tid = static_cast<uint64_t>(blockIdx.x) * kThreads + threadIdx.x;
    uint64_t st = smix(seed ^ smix((static_cast<uint64_t>(kd) << 40) ^ tid));
    for (int it = 0; it < iters; ++it) {
        uint32_t v[6][kTraces];
#pragma unroll
        for (int p = 0; p < kPairs; ++p) {
            st = smix(st);
            const uint64_t c0 = st, c1 = c0 + delta;
            const uint32_t lo0 = static_cast<uint32_t>(c0), hi0 = static_cast<uint32_t>(c0 >> 32);
            const uint32_t lo1 = static_cast<uint32_t>(c1), hi1 = static_cast<uint32_t>(c1 >> 32);
            v[0][p] = k0 ^ lo0; v[1][p] = k1 + hi0; v[2][p] = k2 ^ hi0; v[3][p] = k3 + lo0; v[4][p] = k4; v[5][p] = k5;
            v[0][kPairs + p] = k0 ^ lo1; v[1][kPairs + p] = k1 + hi1; v[2][kPairs + p] = k2 ^ hi1; v[3][kPairs + p] = k3 + lo1;
            v[4][kPairs + p] = k4; v[5][kPairs + p] = k5;
        }
        int cur = c_op[0];
        for (int i = 0; i < nops; ++i) {
            const int nxt = c_op[i + 1];  // prefetch the next op
            const int code = cur & 255, r = cur >> 8;
            cur = nxt;
            const int depth = i + 1, slot = (depth >= dlo && depth <= dhi) ? depth - dlo : -1;
            switch (code) { CASES(CASE_A) CASES(CASE_X) CASES(CASE_R) CASES(CASE_S) default: break; }
            if (slot >= 0) {
                uint32_t o[kTraces];
                switch ((code % 36) / 6) {
                case 0: _Pragma("unroll") for (int l = 0; l < kTraces; ++l) o[l] = v[0][l]; break;
                case 1: _Pragma("unroll") for (int l = 0; l < kTraces; ++l) o[l] = v[1][l]; break;
                case 2: _Pragma("unroll") for (int l = 0; l < kTraces; ++l) o[l] = v[2][l]; break;
                case 3: _Pragma("unroll") for (int l = 0; l < kTraces; ++l) o[l] = v[3][l]; break;
                case 4: _Pragma("unroll") for (int l = 0; l < kTraces; ++l) o[l] = v[4][l]; break;
                default: _Pragma("unroll") for (int l = 0; l < kTraces; ++l) o[l] = v[5][l]; break;
                }
                hist_add(hist + slot * kBins, o);
            }
        }
    }
    __syncthreads();
    // accumulate into global memory: ghist[(kd * nd + depth_index) * kBins + bin]
    for (int sidx = 0; sidx <= dhi - dlo; ++sidx)
        for (int i = threadIdx.x; i < kBins; i += kThreads) {
            const unsigned c = hist[sidx * kBins + i];
            if (c) atomicAdd(&ghist[(static_cast<size_t>(kd) * nd + (dlo + sidx - mlo)) * kBins + i], c);
        }
}

#undef CASE_A
#undef CASE_X
#undef CASE_R
#undef CASE_S
#undef CASES_DS
#undef CASES

// one block per histogram: chi[h] = sum_b (c_b - e)^2 / e - 255
__global__ void kchi(const unsigned *ghist, double e, double *chi) {
    __shared__ double part[256];
    const size_t h = blockIdx.x;
    const double d = ghist[h * 256 + threadIdx.x] - e;
    part[threadIdx.x] = d * d;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) part[threadIdx.x] += part[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) chi[h] = part[0] / e - 255;
}

// Launch kdm<g> for g depths from dlo.
void launch_kdm(int g, dim3 grid, gpuStream_t st, int nops, int dlo, const uint32_t *keys, uint64_t seed, int iters, unsigned *ghist, int nd, int mlo) {
    switch (g) {
    case 1: kdm<1><<<grid, dim3(kThreads), 0, st>>>(nops, dlo, keys, seed, iters, ghist, nd, mlo); break;
    case 2: kdm<2><<<grid, dim3(kThreads), 0, st>>>(nops, dlo, keys, seed, iters, ghist, nd, mlo); break;
    case 3: kdm<3><<<grid, dim3(kThreads), 0, st>>>(nops, dlo, keys, seed, iters, ghist, nd, mlo); break;
    default: kdm<kMaxGroup><<<grid, dim3(kThreads), 0, st>>>(nops, dlo, keys, seed, iters, ghist, nd, mlo); break;
    }
}

// A device buffer and a pinned host buffer of n elements of T (either may be absent), grown as needed;
// the old contents are dropped.
template <class T>
struct Buffer {
    T *dev = nullptr, *host = nullptr;
    size_t cap = 0;
    void reserve(size_t n, bool with_dev, bool with_host) {
        if (n <= cap) return;
        if (dev) CHECK_GPU(gpuFree(dev));
        if (host) CHECK_GPU(gpuHostFree(host));
        dev = host = nullptr;
        if (with_dev) CHECK_GPU(gpuMalloc(reinterpret_cast<void **>(&dev), n * sizeof(T)));
        if (with_host) CHECK_GPU(gpuHostMalloc(reinterpret_cast<void **>(&host), n * sizeof(T)));
        cap = n;
    }
};

// The stream and the buffers, kept across calls. The host buffers are reused only after the wait at
// the end of the previous call.
struct Device {
    gpuStream_t stream = nullptr;
    Buffer<int> ops;        // host only: the packed ops
    Buffer<uint32_t> keys;  // the keys
    Buffer<unsigned> hist;  // device only: the histograms
    Buffer<double> chi;     // the chi-square statistics
    Device() {
        // wait for the device by blocking the thread, instead of spinning on a CPU core
        CHECK_GPU(gpuSetDeviceFlags(gpuDeviceScheduleBlockingSync));
        CHECK_GPU(gpuStreamCreateWithFlags(&stream, gpuStreamNonBlocking));
        ops.reserve(arx::kMaxOps + 1, false, true);
    }
};

}  // namespace

static_assert(sizeof(Key) == 6 * sizeof(uint32_t), "the keys are copied as arrays of words");

// See metric.hpp. The ops are packed into constant memory, and kdm is launched for every group of up
// to kMaxGroup depths. n is rounded up to a multiple of the pairs of a launch.
std::vector<double> bias_per_key(const arx::Arx &a, int mlo, int mhi, const std::vector<Key> &keys, int n, uint64_t seed) {
    static Device dev;
    const gpuStream_t st = dev.stream;
    const int nk = static_cast<int>(keys.size());
    int *packed = dev.ops.host;
    std::fill(packed, packed + arx::kMaxOps + 1, 0);
    for (int i = 0; i < mhi; ++i) {
        const arx::Op &o = a[i];
        int code = o.t * 36 + o.d * 6 + o.s;
        const int rot = o.t == arx::kRot ? o.r : 0;
        if (o.t == arx::kRot && rot == 0) code = 36 + o.d * 6 + o.s;  // rotation by 0 is an xor
        packed[i] = code | rot << 8;
    }
    CHECK_GPU(gpuMemcpyToSymbolAsync(c_op, packed, sizeof(int) * (mhi + 1), 0, gpuMemcpyHostToDevice, st));
    dev.keys.reserve(6 * static_cast<size_t>(nk), true, true);
    std::memcpy(dev.keys.host, keys.data(), sizeof(Key) * nk);
    CHECK_GPU(gpuMemcpyAsync(dev.keys.dev, dev.keys.host, sizeof(Key) * nk, gpuMemcpyHostToDevice, st));
    const int nd = mhi - mlo + 1;
    const size_t nh = static_cast<size_t>(nk) * kDeltas * nd * 2 * kWindows;  // number of 256-bin histograms
    dev.hist.reserve(nh * 256, true, false);
    dev.chi.reserve(nh, true, true);
    CHECK_GPU(gpuMemsetAsync(dev.hist.dev, 0, nh * 256 * sizeof(unsigned), st));
    const int per_block_iter = kThreads * kPairs;
    const int total = (n + per_block_iter - 1) / per_block_iter;
    const int bx = std::min(total, 16);
    const int iters = (total + bx - 1) / bx;
    const int nreal = bx * iters * per_block_iter;  // n rounded up
    for (int dlo = mlo; dlo <= mhi; dlo += kMaxGroup) {
        const int g = std::min(mhi - dlo + 1, kMaxGroup);
        // the ops after the last depth of the launch don't affect its histograms
        launch_kdm(g, dim3(bx, nk * kDeltas), st, dlo + g - 1, dlo, dev.keys.dev, seed, iters, dev.hist.dev, nd, mlo);
    }
    const double e = nreal / 256.0;
    kchi<<<dim3(nh), dim3(256), 0, st>>>(dev.hist.dev, e, dev.chi.dev);
    CHECK_GPU(gpuGetLastError());
    CHECK_GPU(gpuMemcpyAsync(dev.chi.host, dev.chi.dev, nh * sizeof(double), gpuMemcpyDeviceToHost, st));
    CHECK_GPU(gpuStreamSynchronize(st));  // the only wait: the results are needed below
    std::vector<double> per_key(static_cast<size_t>(nk) * nd);
    for (int key = 0; key < nk; ++key)
        for (int m = 0; m < nd; ++m) {
            double kb = 0;
            for (int di = 0; di < kDeltas; ++di)
                for (int q = 0; q < 2 * kWindows; ++q)
                    kb += dev.chi.host[(static_cast<size_t>(key * kDeltas + di) * nd + m) * 2 * kWindows + q] / nreal;
            per_key[static_cast<size_t>(key) * nd + m] = kb;
        }
    return per_key;
}

}  // namespace metric
