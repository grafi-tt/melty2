#include "melty2.hpp"

#include <array>
#include <iostream>

static int err = 0;

#define test_assert(cond) if (!(cond)) { std::cerr << __func__ << ":" << __LINE__ << " " << #cond << "failed" << std::endl; ++err; }

void test_empty_seed() {
    constexpr size_t N = 123;

    melty2::generator gen;
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

void test_ctr() {
    constexpr size_t N = 123;

    melty2::generator gen;
    gen.set_ctr(123);
    std::array<uint32_t, 2 * N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = gen();
    }
    test_assert(gen.current_ctr() == 2 * N);
    gen.set_ctr(10 * N);
    for (size_t i = N; i < 2 * N; ++i) {
        result[i] = gen();
    }
    test_assert(gen.current_ctr() == 11 * N);

    melty2_seeder seeder;
    melty2_initseeder(&seeder);
    melty2_key key;
    melty2_initkey(&key, &seeder);
    std::array<uint32_t, 2 * N> expected;
    melty2_gen(&key, N, N, expected.data());
    melty2_gen(&key, 10 * N, N, expected.data() + N);

    test_assert(result == expected);
}

int main() {
    test_empty_seed();
    test_ctr();
    return !!err;
}
