// SPDX-License-Identifier: Apache-2.0
//
// The GPU runtime of the host code of testu01_gpu: HIP (ROCm), or CUDA if TESTU01_GPU_CUDA is defined (by the
// build, for TESTU01_GPU_BACKEND=CUDA). The memory and the errors go by the names gpu*, mapped to those of the
// backend; the run-time compilation (hiprtc or NVRTC), the modules and the FFTs (rocFFT or cuFFT) differ more,
// and are written for each backend where they are used.
#ifndef TESTU01_GPU_GPU_HPP_
#define TESTU01_GPU_GPU_HPP_

#ifdef TESTU01_GPU_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <nvrtc.h>

#define gpuError_t cudaError_t
#define gpuSuccess cudaSuccess
#define gpuGetErrorString cudaGetErrorString
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuMemset cudaMemset
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
using gpuModule_t = CUmodule;
using gpuFunction_t = CUfunction;

#else

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <rocfft/rocfft.h>

#define gpuError_t hipError_t
#define gpuSuccess hipSuccess
#define gpuGetErrorString hipGetErrorString
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuMemset hipMemset
#define gpuMemcpy hipMemcpy
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
using gpuModule_t = hipModule_t;
using gpuFunction_t = hipFunction_t;

#endif

#endif
