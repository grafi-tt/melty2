// SPDX-License-Identifier: Apache-2.0
//
// splitmix_bigcrush [seed] [scale] [tests] [fast] [battery] : run the tests of testu01_gpu on the upper 32 bits of
// SplitMix64, as a counter-based generator: the output at the position i is that of the state
// seed + (i + 1) * 0x9e3779b97f4a7c15. An example of the library, and a check of the tests on a good
// generator (the p-values are uniform). tests "all" runs all of them; battery is small, crush or big (default, BigCrush).
// The generator source is the same for HIP and CUDA (TESTU01_GPU_CUDA).
#ifdef TESTU01_GPU_CUDA
#include <cuda_runtime.h>
#else
#include <hip/hip_runtime.h>
#endif

#include <charconv>
#include <cstdint>
#include <exception>
#include <format>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <vector>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "testu01_gpu.hpp"

namespace {

// the generator for the device: testu01_gpu_generate writes the outputs at kLanes consecutive positions
const char kSource[] = R"(
__device__ inline void testu01_gpu_generate(const testu01_gpu::Params &p, uint64_t pos, uint32_t (&out)[testu01_gpu::kLanes]) {
    const uint64_t seed = static_cast<uint64_t>(p.w[1]) << 32 | p.w[0];
#pragma unroll
    for (int l = 0; l < testu01_gpu::kLanes; ++l) {
        uint64_t z = seed + (pos + l + 1) * UINT64_C(0x9e3779b97f4a7c15);
        z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
        out[l] = static_cast<uint32_t>((z ^ (z >> 31)) >> 32);
    }
}
)";

// a number of the command line, all of it (an integer in hexadecimal with 0x)
template <typename T> T number(std::string_view a) {
    int base = 10;
    if constexpr (std::is_integral_v<T>) {
        if (a.starts_with("0x") || a.starts_with("0X")) a.remove_prefix(2), base = 16;
    }
    T v{};
    const auto [end, ec] = [&] {
        if constexpr (std::is_integral_v<T>) return std::from_chars(a.data(), a.data() + a.size(), v, base);
        else return std::from_chars(a.data(), a.data() + a.size(), v);
    }();
    if (ec != std::errc() || end != a.data() + a.size()) throw std::invalid_argument(std::format("not a number: {}", a));
    return v;
}

}  // namespace

int main(int argc, char **argv) {
    absl::InitializeLog();
    absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);  // the warnings to stderr as well
    try {
        const std::vector<std::string_view> args(argv + 1, argv + argc);
        const uint64_t seed = args.size() > 0 ? number<uint64_t>(args[0]) : 1;
        testu01_gpu::Options o;
        if (args.size() > 1) o.scale = number<double>(args[1]);
        if (args.size() > 2 && args[2] != "all") o.tests = args[2];
        if (args.size() > 3) o.fast = number<int>(args[3]) != 0;
        if (args.size() > 4) {
            const std::string_view b = args[4];
            if (b == "small") o.suite = testu01_gpu::Suite::kSmallCrush;
            else if (b == "crush") o.suite = testu01_gpu::Suite::kCrush;
            else if (b == "big") o.suite = testu01_gpu::Suite::kBigCrush;
            else throw std::invalid_argument(std::format("battery: small, crush or big, not {}", b));
        }
        o.report = [](const testu01_gpu::Result &r) {
            std::cout << testu01_gpu::format(r) << std::endl;
        };
        // wait for the device by blocking the thread, instead of spinning on a CPU core
#ifdef TESTU01_GPU_CUDA
        if (cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync) != cudaSuccess) LOG(WARNING) << "cudaSetDeviceFlags failed";
#else
        if (hipSetDeviceFlags(hipDeviceScheduleBlockingSync) != hipSuccess) LOG(WARNING) << "hipSetDeviceFlags failed";
#endif
        const testu01_gpu::Generator gen(kSource, "splitmix");
        testu01_gpu::Params params{};
        params.w[0] = static_cast<uint32_t>(seed);
        params.w[1] = static_cast<uint32_t>(seed >> 32);
        testu01_gpu::Battery battery;
        const testu01_gpu::Summary s = battery.run(gen, params, o);
        std::cout << std::format("summary\tmin_p={:.3g}\tmax_p={:.6g}\tflagged={}\n", s.min_p, s.max_p, s.flagged);
        return 0;
    } catch (const std::exception &e) {
        LOG(ERROR) << e.what();
        return 1;
    }
}
