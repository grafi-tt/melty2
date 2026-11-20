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

#include "melty2_kernel.h"

#ifndef MELTY2_INTERNAL_FUNC
#define MELTY2_INTERNAL_FUNC static inline __host__ __device__
#endif
#include "melty2_derive.h"

namespace melty2 {
namespace gpu {

#if defined(__HIPCC__)
typedef hipStream_t stream_t;
typedef hipError_t error_t;
#else
typedef cudaStream_t stream_t;
typedef cudaError_t error_t;
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
 * @brief Return the random number for the counter.
 */
__host__ __device__ inline uint32_t gen(const melty2_key &key, uint64_t ctr) {
    uint32_t v[6] = {key.v_[0], key.v_[1], key.v_[2], key.v_[3], key.v_[4], key.v_[5]};
    return melty2_kernel(v, (uint32_t)ctr, (uint32_t)(ctr >> 32));
}

/**
 * @brief Per-thread generator holding a key and a counter.
 * @note  It has no buffer, as computing one number is cheap.
 */
class generator {
public:
    __host__ __device__ generator(const melty2_key &key, uint64_t ctr = 0) : key_(key), ctr_(ctr) {}
    __host__ __device__ generator(uint32_t seed0, uint32_t seed1, uint32_t seed2, uint32_t seed3, uint64_t ctr = 0)
        : key_(init(seed0, seed1, seed2, seed3)), ctr_(ctr) {}

    __host__ __device__ uint32_t operator()() { return gen(key_, ctr_++); }

    typedef uint32_t result_type;
    __host__ __device__ static constexpr uint32_t min() { return 0; }
    __host__ __device__ static constexpr uint32_t max() { return UINT32_MAX; }

    __host__ __device__ uint64_t ctr() const { return ctr_; }
    __host__ __device__ generator &ctr(uint64_t new_ctr) { ctr_ = new_ctr; return *this; }
    __host__ __device__ const melty2_key &key() const { return key_; }

private:
    melty2_key key_;
    uint64_t ctr_;
};

/**
 * @brief Kernel storing the random numbers for counters ctr, ctr + 1, ..., ctr + len - 1 to out.
 */
template <typename Dummy = void>
__global__ void fill_kernel(melty2_key key, uint64_t ctr, size_t len, uint32_t *out) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < len; i += stride) {
        out[i] = gen(key, ctr + i);
    }
}

/**
 * @brief Store the random numbers for counters ctr, ctr + 1, ..., ctr + len - 1 to device memory out.
 * @return The error of the kernel launch. The kernel runs asynchronously on stream.
 */
inline error_t fill(const melty2_key &key, uint64_t ctr, size_t len, uint32_t *out, stream_t stream = 0) {
    const unsigned block = 256;
    const size_t max_grid = 65536;
    size_t grid = (len + block - 1) / block;
    if (grid > max_grid) {
        grid = max_grid;
    }
    if (grid == 0) {
#if defined(__HIPCC__)
        return hipSuccess;
#else
        return cudaSuccess;
#endif
    }
    fill_kernel<><<<(unsigned)grid, block, 0, stream>>>(key, ctr, len, out);
#if defined(__HIPCC__)
    return hipGetLastError();
#else
    return cudaGetLastError();
#endif
}

}  // namespace gpu
}  // namespace melty2

#endif  // MELTY2_MELTY2_GPU_H_
