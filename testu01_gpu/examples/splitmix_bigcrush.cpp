// SPDX-License-Identifier: Apache-2.0
//
// splitmix_bigcrush [seed] [scale] [tests] : run the tests of testu01_gpu on the upper 32 bits of
// SplitMix64, as a counter-based generator: the output at the position i is that of the state
// seed + (i + 1) * 0x9e3779b97f4a7c15. An example of the library, and a check of the tests on a good
// generator (the p-values are uniform).
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <exception>

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

}  // namespace

int main(int argc, char **argv) {
    try {
        const uint64_t seed = argc > 1 ? std::strtoull(argv[1], nullptr, 0) : 1;
        testu01_gpu::Options o;
        if (argc > 2) o.scale = std::atof(argv[2]);
        if (argc > 3) o.tests = argv[3];
        o.report = [](const testu01_gpu::Result &r) {
            std::printf("%s\n", testu01_gpu::format(r).c_str());
            std::fflush(stdout);
        };
        // wait for the device by blocking the thread, instead of spinning on a CPU core
        if (hipSetDeviceFlags(hipDeviceScheduleBlockingSync) != hipSuccess) std::fprintf(stderr, "hipSetDeviceFlags failed\n");
        const testu01_gpu::Generator gen(kSource, "splitmix");
        testu01_gpu::Params params{};
        params.w[0] = static_cast<uint32_t>(seed);
        params.w[1] = static_cast<uint32_t>(seed >> 32);
        testu01_gpu::Battery battery;
        const testu01_gpu::Summary s = battery.run(gen, params, o);
        std::printf("summary\tmin_p=%.3g\tmax_p=%.6g\tflagged=%d\n", s.min_p, s.max_p, s.flagged);
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
