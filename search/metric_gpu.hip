// SPDX-License-Identifier: MIT
// GPU (HIP) implementation of metric_keys() in metric.h.
//
// The structure is interpreted on the device: ops are packed into constant memory, and each thread
// traces NP counter pairs. Histograms of DG depths at a time are accumulated in LDS, and then in
// global memory; the chi-square statistics are reduced on the device.
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "metric.h"

#define NDELTA 16
#define NWIN 8
#define NP 8         // counter pairs per thread
#define NT (2 * NP)  // traces per thread
#define DG 3         // depths per LDS group
#define TPB 256

__constant__ int c_op[MAXOPS + 1];  // (t * 36 + d * 6 + s) | rot << 8

// SplitMix64 finalizer
__device__ __forceinline__ uint64_t smix(uint64_t z) {
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
// r in [1, 31]
__device__ __forceinline__ uint32_t rotl_u(uint32_t x, int r) { return x << r | x >> (32 - r); }

// histogram one depth slot for all NP pairs; o is the array of the traces of the written word
#define HIST(o) do { if (slot >= 0) { \
    unsigned *H = hist + slot * (2 * NWIN * 256); \
    _Pragma("unroll") for (int p = 0; p < NP; ++p) { \
        uint32_t ad = (o)[NP + p] - (o)[p], xd = (o)[NP + p] ^ (o)[p]; \
        _Pragma("unroll") for (int w = 0; w < NWIN; ++w) { \
            uint32_t a2 = w ? rotl_u(ad, 4 * w) : ad, x2 = w ? rotl_u(xd, 4 * w) : xd; \
            atomicAdd(&H[w * 256 + (a2 >> 24)], 1u); \
            atomicAdd(&H[(NWIN + w) * 256 + (x2 >> 24)], 1u); \
        } } } } while (0)

#define CASE_A(d, s) case 0 * 36 + d * 6 + s: _Pragma("unroll") for (int l = 0; l < NT; ++l) v[d][l] += v[s][l]; break;
#define CASE_X(d, s) case 1 * 36 + d * 6 + s: _Pragma("unroll") for (int l = 0; l < NT; ++l) v[d][l] ^= v[s][l]; break;
#define CASE_R(d, s) case 2 * 36 + d * 6 + s: _Pragma("unroll") for (int l = 0; l < NT; ++l) v[d][l] = rotl_u(v[d][l] ^ v[s][l], r); break;
#define CASES_DS(M, d) M(d, 0) M(d, 1) M(d, 2) M(d, 3) M(d, 4) M(d, 5)
#define CASES(M) CASES_DS(M, 0) CASES_DS(M, 1) CASES_DS(M, 2) CASES_DS(M, 3) CASES_DS(M, 4) CASES_DS(M, 5)

// grid: x = blocks per (key, delta), y = key * NDELTA + delta
__global__ void __launch_bounds__(TPB) kdm(int nops, int dlo, int dhi, const uint32_t *keys, uint64_t seed, int iters, unsigned *ghist, int nd, int mlo) {
    __shared__ unsigned hist[DG * 2 * NWIN * 256];
    for (int i = threadIdx.x; i < DG * 2 * NWIN * 256; i += TPB) hist[i] = 0;
    __syncthreads();
    int kd = blockIdx.y, key = kd / NDELTA, di = kd % NDELTA;
    int j = di < 8 ? di : 32 + di - 8;
    uint64_t delta = 1ULL << j;
    const uint32_t *k = keys + 6 * key;
    uint32_t k0 = k[0], k1 = k[1], k2 = k[2], k3 = k[3], k4 = k[4], k5 = k[5];
    uint64_t tid = (uint64_t)blockIdx.x * TPB + threadIdx.x;
    uint64_t st = smix(seed ^ smix(((uint64_t)kd << 40) ^ tid));
    for (int it = 0; it < iters; ++it) {
        uint32_t v[6][NT];
        _Pragma("unroll") for (int p = 0; p < NP; ++p) {
            st = smix(st);
            uint64_t c0 = st, c1 = c0 + delta;
            uint32_t lo0 = (uint32_t)c0, hi0 = (uint32_t)(c0 >> 32), lo1 = (uint32_t)c1, hi1 = (uint32_t)(c1 >> 32);
            v[0][p] = k0 ^ lo0; v[1][p] = k1 + hi0; v[2][p] = k2 ^ hi0; v[3][p] = k3 + lo0; v[4][p] = k4; v[5][p] = k5;
            v[0][NP + p] = k0 ^ lo1; v[1][NP + p] = k1 + hi1; v[2][NP + p] = k2 ^ hi1; v[3][NP + p] = k3 + lo1; v[4][NP + p] = k4; v[5][NP + p] = k5;
        }
        int cur = c_op[0];
        for (int i = 0; i < nops; ++i) {
            int nxt = c_op[i + 1];  // prefetch the next op
            int code = cur & 255, r = cur >> 8;
            cur = nxt;
            int depth = i + 1, slot = (depth >= dlo && depth <= dhi) ? depth - dlo : -1;
            switch (code) { CASES(CASE_A) CASES(CASE_X) CASES(CASE_R) default: break; }
            if (slot >= 0) {
                uint32_t o[NT];
                switch ((code % 36) / 6) {
                case 0: _Pragma("unroll") for (int l = 0; l < NT; ++l) o[l] = v[0][l]; break;
                case 1: _Pragma("unroll") for (int l = 0; l < NT; ++l) o[l] = v[1][l]; break;
                case 2: _Pragma("unroll") for (int l = 0; l < NT; ++l) o[l] = v[2][l]; break;
                case 3: _Pragma("unroll") for (int l = 0; l < NT; ++l) o[l] = v[3][l]; break;
                case 4: _Pragma("unroll") for (int l = 0; l < NT; ++l) o[l] = v[4][l]; break;
                default: _Pragma("unroll") for (int l = 0; l < NT; ++l) o[l] = v[5][l]; break;
                }
                HIST(o);
            }
        }
    }
    __syncthreads();
    // accumulate into global memory: ghist[(kd * nd + depth_index) * 4096 + bin]
    for (int sidx = 0; sidx <= dhi - dlo; ++sidx)
        for (int i = threadIdx.x; i < 2 * NWIN * 256; i += TPB) {
            unsigned c = hist[sidx * 2 * NWIN * 256 + i];
            if (c) atomicAdd(&ghist[((size_t)kd * nd + (dlo + sidx - mlo)) * (2 * NWIN * 256) + i], c);
        }
}

// one block per histogram: chi[h] = sum_b (c_b - e)^2 / e - 255
__global__ void kchi(const unsigned *ghist, double e, double *chi) {
    __shared__ double part[256];
    size_t h = blockIdx.x;
    double d = ghist[h * 256 + threadIdx.x] - e;
    part[threadIdx.x] = d * d;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) part[threadIdx.x] += part[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) chi[h] = part[0] / e - 255;
}

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

extern "C" void metric_keys(const arx_t *a, int mlo, int mhi, const uint32_t *keys, int nk, int n, uint64_t seed, double *per_key) {
    static uint32_t *d_keys = 0;
    static unsigned *d_hist = 0;
    static double *d_chi = 0, *h_chi = 0;
    static size_t hist_cap = 0;
    static int keys_cap = 0;
    int packed[MAXOPS + 1] = {0};
    for (int i = 0; i < mhi; ++i) {
        int code = a->t[i] * 36 + a->d[i] * 6 + a->s[i], rot = a->t[i] == 2 ? a->r[i] : 0;
        if (a->t[i] == 2 && rot == 0) code = 36 + a->d[i] * 6 + a->s[i];  // rotation by 0 is an xor
        packed[i] = code | rot << 8;
    }
    CK(hipMemcpyToSymbol(HIP_SYMBOL(c_op), packed, sizeof(int) * (mhi + 1)));
    if (nk > keys_cap) {
        if (d_keys) CK(hipFree(d_keys));
        CK(hipMalloc(&d_keys, sizeof(uint32_t) * 6 * nk));
        keys_cap = nk;
    }
    CK(hipMemcpy(d_keys, keys, sizeof(uint32_t) * 6 * nk, hipMemcpyHostToDevice));
    int nd = mhi - mlo + 1;
    size_t nh = (size_t)nk * NDELTA * nd * 2 * NWIN;  // number of 256-bin histograms
    size_t hsz = nh * 256;
    if (hsz > hist_cap) {
        if (d_hist) {
            CK(hipFree(d_hist));
            CK(hipFree(d_chi));
            free(h_chi);
        }
        CK(hipMalloc(&d_hist, hsz * sizeof(unsigned)));
        CK(hipMalloc(&d_chi, nh * sizeof(double)));
        h_chi = (double *)malloc(nh * sizeof(double));
        hist_cap = hsz;
    }
    CK(hipMemsetAsync(d_hist, 0, hsz * sizeof(unsigned), 0));
    int per_block_iter = TPB * NP;
    int total = (n + per_block_iter - 1) / per_block_iter;
    int bx = total < 16 ? total : 16;
    int iters = (total + bx - 1) / bx;
    int nreal = bx * iters * per_block_iter;  // n rounded up
    for (int dlo = mlo; dlo <= mhi; dlo += DG) {
        int dhi = dlo + DG - 1 < mhi ? dlo + DG - 1 : mhi;
        hipLaunchKernelGGL(kdm, dim3(bx, nk * NDELTA), dim3(TPB), 0, 0, mhi, dlo, dhi, d_keys, seed, iters, d_hist, nd, mlo);
    }
    double e = nreal / 256.0;
    hipLaunchKernelGGL(kchi, dim3(nh), dim3(256), 0, 0, d_hist, e, d_chi);
    CK(hipGetLastError());
    CK(hipMemcpy(h_chi, d_chi, nh * sizeof(double), hipMemcpyDeviceToHost));
    for (int key = 0; key < nk; ++key)
        for (int m = 0; m < nd; ++m) {
            double kb = 0;
            for (int di = 0; di < NDELTA; ++di)
                for (int q = 0; q < 2 * NWIN; ++q) kb += h_chi[((size_t)(key * NDELTA + di) * nd + m) * 2 * NWIN + q] / nreal;
            per_key[key * nd + m] = kb;
        }
}

extern "C" double metric_noise(int n) { return sqrt(2.0 * 255 * NDELTA * 2 * NWIN) / n; }
