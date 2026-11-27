// SPDX-License-Identifier: Apache-2.0
//
// The parameters of a generator for testu01_gpu, shared by the host and the device code (the library
// passes this file to hiprtc or NVRTC as a header, so it must stay self-contained).
#ifndef TESTU01_GPU_PARAMS_HPP_
#define TESTU01_GPU_PARAMS_HPP_

#if defined(__HIPCC_RTC__) || defined(__CUDACC_RTC__)
#ifdef __HIPCC_RTC__
// hiprtc has no <cstdint>, and defines the fixed-width types in __hip_internal
using __hip_internal::uint32_t;
using __hip_internal::uint64_t;
#else
// NVRTC has no <cstdint> (without the headers of libcu++)
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
#endif
#ifndef UINT32_C
#define UINT32_C(c) c##U
#endif
#ifndef UINT64_C
#define UINT64_C(c) c##ULL
#endif
#else
#include <cstdint>
#endif

namespace testu01_gpu {

// the outputs generated at once by a thread: testu01_gpu_generate writes those of kLanes consecutive
// positions
constexpr int kLanes = 16;

// the words of the parameters of a generator (e.g. its key, or the selection of its streams)
constexpr int kParamWords = 32;

// The parameters of a generator, passed to every kernel by value: the generator source interprets the
// words as it likes. A sequence of outputs is defined by the generator source and its parameters.
struct Params {
    uint32_t w[kParamWords];
};

}  // namespace testu01_gpu

#endif
