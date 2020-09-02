#include "melty2.hpp"

#include <array>
#include <iostream>

static int err = 0;

#define test_assert(cond) if (!(cond)) { std::cerr << __func__ << ":" << __LINE__ << " " << #cond << "failed" << std::endl; ++err; }

void test_empty_seed() {
    constexpr size_t N = 123;

    melty2::key k{};
    melty2::generator<> gen(k);
    std::array<uint32_t, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = gen();
    }

    melty2_seeder seeder;
    melty2_initseeder(&seeder);
    melty2_key key;
    melty2_initkey(&key, &seeder);
    std::array<uint32_t, N> expected;
    melty2_gen(&key, 0, N, expected.data());

    test_assert(result == expected);
}

int main() {
    test_empty_seed();
    return !!err;
}
