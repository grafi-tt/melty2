// SPDX-License-Identifier: Apache-2.0
//
// The GPU runtime of the search tools: HIP (ROCm), or CUDA if MELTY2_SEARCH_CUDA is defined (by the build, for
// MELTY2_SEARCH_GPU_BACKEND=CUDA). The GPU code (metric_gpu.hip, testu01_stest.hip) is written once, by the names gpu*
// mapped to those of the backend, and kernels launched by <<<...>>> (as both compile them); CHECK_GPU(call) checks a
// call (fatal, with the error string and the call).
#pragma once

#ifdef MELTY2_SEARCH_CUDA

#include <cuda_runtime.h>

#define gpuError_t cudaError_t
#define gpuSuccess cudaSuccess
#define gpuGetErrorString cudaGetErrorString
#define gpuGetLastError cudaGetLastError
#define gpuSetDeviceFlags cudaSetDeviceFlags
#define gpuDeviceScheduleBlockingSync cudaDeviceScheduleBlockingSync
#define gpuStream_t cudaStream_t
#define gpuStreamCreateWithFlags cudaStreamCreateWithFlags
#define gpuStreamNonBlocking cudaStreamNonBlocking
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuHostMalloc(p, n) cudaMallocHost(p, n)
#define gpuHostFree cudaFreeHost
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuMemcpyToSymbolAsync(sym, src, n, off, kind, st) cudaMemcpyToSymbolAsync(sym, src, n, off, kind, st)
#define gpuMemsetAsync cudaMemsetAsync
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
// the static shared memory of a block: 48 KiB in CUDA (more only as dynamic shared memory)
inline constexpr int kGpuStaticSharedBytes = 48 * 1024;

#else

#include <hip/hip_runtime.h>

#define gpuError_t hipError_t
#define gpuSuccess hipSuccess
#define gpuGetErrorString hipGetErrorString
#define gpuGetLastError hipGetLastError
#define gpuSetDeviceFlags hipSetDeviceFlags
#define gpuDeviceScheduleBlockingSync hipDeviceScheduleBlockingSync
#define gpuStream_t hipStream_t
#define gpuStreamCreateWithFlags hipStreamCreateWithFlags
#define gpuStreamNonBlocking hipStreamNonBlocking
#define gpuStreamSynchronize hipStreamSynchronize
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuHostMalloc(p, n) hipHostMalloc(p, n, hipHostMallocDefault)
#define gpuHostFree hipHostFree
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuMemcpyToSymbolAsync(sym, src, n, off, kind, st) hipMemcpyToSymbolAsync(HIP_SYMBOL(sym), src, n, off, kind, st)
#define gpuMemsetAsync hipMemsetAsync
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
// the LDS of a block (64 KiB on the AMD GPUs)
inline constexpr int kGpuStaticSharedBytes = 64 * 1024;

#endif

#include "logging.hpp"

#define CHECK_GPU(x)                                                                                 \
    do {                                                                                             \
        const gpuError_t check_gpu_e = (x);                                                          \
        CHECK(check_gpu_e == gpuSuccess) << "GPU " << gpuGetErrorString(check_gpu_e) << ": " #x;     \
    } while (0)
