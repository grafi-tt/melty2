// SPDX-License-Identifier: Apache-2.0
// check : arx::kMelty2 in arx.hpp computes the same outputs as the kernel of the library (melty2_internal_kernel.h)
#include <format>
#include <iostream>

#include "arx.hpp"
#include "melty2_internal_kernel.h"

int main() {
    const arx::Arx a = arx::parse(arx::kMelty2);
    uint64_t x = 1;
    const auto next = [&x] {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        return x;
    };
    long bad = 0;
    const long n = 1000000;
    for (long i = 0; i < n; ++i) {
        arx::Key k;
        uint32_t v[6];
        for (int j = 0; j < 6; ++j) k[j] = v[j] = static_cast<uint32_t>(next());
        const uint64_t ctr = next();
        const uint32_t expected = melty2_kernel(v, static_cast<uint32_t>(ctr), static_cast<uint32_t>(ctr >> 32));
        bad += arx::eval(a, static_cast<int>(a.size()), k, ctr) != expected;
    }
    std::cout << std::format("arx::kMelty2 vs melty2_internal_kernel.h: {} mismatches of {}\n", bad, n);
    return bad != 0;
}
