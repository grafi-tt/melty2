#include "melty2.hpp"

#include <array>
#include <iostream>
#include <random>

static int err = 0;

#define test_assert(cond) \
    for (bool test_assert_failed_ = !(cond); test_assert_failed_; std::cerr << std::endl, ++err, test_assert_failed_ = false) \
        std::cerr << __func__ << ":" << __LINE__ << " " << #cond << " failed"

void test_default() {
    constexpr size_t N = 123;

    melty2::generator gen;
    std::array<uint32_t, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = gen();
    }

    melty2_key key;
    melty2_init(&key, 0, 0, 0, 0);
    std::array<uint32_t, N> expected;
    melty2_gen(&key, 0, N, expected.data());

    test_assert(result == expected);
}

void test_seed() {
    melty2::generator gen{{3141592653u, 2718281828u, 123456789, 42}};
    uint32_t result = gen();

    melty2_key key;
    melty2_init(&key, 3141592653u, 2718281828u, 123456789, 42);
    uint32_t expected;
    melty2_gen(&key, 0, 1, &expected);

    test_assert(result == expected);
}

void test_ctr() {
    constexpr size_t N = 123;

    melty2::generator gen;
    gen.ctr(123);
    std::array<uint32_t, 2 * N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = gen();
    }
    test_assert(gen.ctr() == 2 * N);
    gen.ctr(10 * N);
    for (size_t i = N; i < 2 * N; ++i) {
        result[i] = gen();
    }
    test_assert(gen.ctr() == 11 * N);

    melty2_key key;
    melty2_init(&key, 0, 0, 0, 0);
    std::array<uint32_t, 2 * N> expected;
    melty2_gen(&key, N, N, expected.data());
    melty2_gen(&key, 10 * N, N, expected.data() + N);

    test_assert(result == expected);
}

void test_std_random() {
    melty2::generator gen;
    std::uniform_int_distribution<uint32_t> dist(gen.min(), gen.max());
    uint32_t result = dist(gen);
    uint32_t expected = gen.ctr(0)();

    test_assert(result == expected);
}

int main() {
    test_default();
    test_seed();
    test_ctr();
    test_std_random();
    return !!err;
}
