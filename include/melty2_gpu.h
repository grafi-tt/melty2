#ifndef MELTY2_MELTY2_GPU_H_
#define MELTY2_MELTY2_GPU_H_

/*
 * Header-only GPU implementation of melty2 for CUDA and HIP.
 *
 * The outputs are identical to those of the CPU library: for the same seed
 * and counter, melty2::gpu::gen returns the same value as melty2_gen.
 */

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#elif !defined(__CUDACC__)
#error "melty2_gpu.h must be compiled by a CUDA or HIP compiler"
#endif

#include <stddef.h>
#include <stdint.h>

#include "melty2.h"

#ifndef MELTY2_INTERNAL_FUNC
#define MELTY2_INTERNAL_FUNC static inline __host__ __device__
#endif

#include "melty2_internal_derive.h"
#include "melty2_internal_kernel.h"

namespace melty2::gpu {

#if defined(__HIPCC__)
using stream_t = hipStream_t;
using error_t = hipError_t;
#else
using stream_t = cudaStream_t;
using error_t = cudaError_t;
#endif

/**
 * @brief Derive a key from a 128bit seed. Same as melty2_init, but callable from device code.
 */
__host__ __device__ inline melty2_key init(uint32_t seed0, uint32_t seed1, uint32_t seed2, uint32_t seed3) {
    melty2_key key;
    melty2_internal_derive_key(key.v_, seed0, seed1, seed2, seed3);
    return key;
}

/**
 * @brief Derive a key from a 128bit seed given by 64bit integers. Same as melty2_init_u64.
 */
__host__ __device__ inline melty2_key init_u64(uint64_t seed0, uint64_t seed1) {
    return init(static_cast<uint32_t>(seed0), static_cast<uint32_t>(seed0 >> 32),
                static_cast<uint32_t>(seed1), static_cast<uint32_t>(seed1 >> 32));
}

/**
 * @brief Return the random number for the counter.
 */
__host__ __device__ inline uint32_t gen(const melty2_key &key, uint64_t ctr) {
    uint32_t v[6] = {key.v_[0], key.v_[1], key.v_[2], key.v_[3], key.v_[4], key.v_[5]};
    return melty2_kernel(v, static_cast<uint32_t>(ctr), static_cast<uint32_t>(ctr >> 32));
}

/**
 * @brief Per-thread generator holding a key and a counter.
 * @note  It has no buffer, as computing one number is cheap.
 */
class generator {
public:
    /* GPU generator accepts explicit melty2_key argument */
    __host__ __device__ generator(const melty2_key &key, uint64_t ctr = 0) : key_(key), ctr_(ctr) {}
    /* GPU generator uses 4 uint32_t arguments instead of std::array<uint32_t, 4> */
    __host__ __device__ generator(uint32_t seed0, uint32_t seed1, uint32_t seed2, uint32_t seed3, uint64_t ctr = 0)
        : key_(init(seed0, seed1, seed2, seed3)), ctr_(ctr) {}
    /* the 128bit seed is the zero extension of seed, as melty2::generator */
    __host__ __device__ explicit generator(uint64_t seed, uint64_t ctr = 0) : key_(init_u64(seed, 0)), ctr_(ctr) {}

    __host__ __device__ uint32_t operator()() { return gen(key_, ctr_++); }

    using result_type = uint32_t;
    __host__ __device__ static constexpr uint32_t min() { return 0; }
    __host__ __device__ static constexpr uint32_t max() { return UINT32_MAX; }

    __host__ __device__ uint64_t ctr() const { return ctr_; }
    __host__ __device__ generator &ctr(uint64_t new_ctr) { ctr_ = new_ctr; return *this; }
    __host__ __device__ const melty2_key &key() const { return key_; }

private:
    melty2_key key_;
    uint64_t ctr_;
};

/* 4 numbers written by one 128bit store */
struct alignas(16) u32x4_ {
    uint32_t v[4];
};

/**
 * @brief Kernel storing the random numbers for counters ctr, ctr + 1, ..., ctr + len - 1 to out.
 * @note  The 16 bytes aligned part of out is written by 128bit stores, each of which has 4 numbers.
 *        The unaligned head and the tail (at most 3 numbers each) are written by 32bit stores.
 *        It is a template, so that the kernel defined in this header may be in several translation units
 *        (as an inline function; inline is ignored for a __global__ function by nvcc and hipcc).
 */
template <typename Dummy = void>
__global__ void fill_kernel(melty2_key key, uint64_t ctr, size_t len, uint32_t *out) {
    size_t head = ((16 - reinterpret_cast<uintptr_t>(out) % 16) % 16) / 4;
    if (head > len) {
        head = len;
    }
    const size_t n4 = (len - head) / 4;
    const size_t body_end = head + 4 * n4;

    const size_t id = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id < head) {
        out[id] = gen(key, ctr + id);
    }
    if (id < len - body_end) {
        out[body_end + id] = gen(key, ctr + body_end + id);
    }

    u32x4_ *const out4 = reinterpret_cast<u32x4_ *>(out + head);
    const uint64_t ctr4 = ctr + head;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = id; i < n4; i += stride) {
        const uint64_t c = ctr4 + 4 * static_cast<uint64_t>(i);
        const u32x4_ r{{gen(key, c), gen(key, c + 1), gen(key, c + 2), gen(key, c + 3)}};
        out4[i] = r;
    }
}

/**
 * @brief Store the random numbers for counters ctr, ctr + 1, ..., ctr + len - 1 to device memory out.
 * @return The error of the kernel launch. The kernel runs asynchronously on stream.
 */
inline error_t fill(const melty2_key &key, uint64_t ctr, size_t len, uint32_t *out, stream_t stream = 0) {
    if (len == 0) {
#if defined(__HIPCC__)
        return hipSuccess;
#else
        return cudaSuccess;
#endif
    }
    const unsigned int block = 256;
    const unsigned int max_grid = 65536;
    /* one thread for each 4 numbers, rounded up (so at least one thread for len < 4) */
    size_t grid = ((len + 3) / 4 + (block - 1)) / block;
    if (grid > max_grid) {
        grid = max_grid;
    }
    fill_kernel<><<<static_cast<unsigned int>(grid), block, 0, stream>>>(key, ctr, len, out);
#if defined(__HIPCC__)
    return hipGetLastError();
#else
    return cudaGetLastError();
#endif
}

}  // namespace melty2::gpu

#endif  // MELTY2_MELTY2_GPU_H_
