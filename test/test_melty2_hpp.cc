#include "melty2.hpp"

#include <algorithm>
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

/* generators constructed with a counter near 2^32 (not aligned to the block) */
void test_ctor_ctr() {
    constexpr size_t N = 100;
    const uint64_t c0 = UINT64_C(0xfffffff0);

    melty2_key key;
    melty2_init(&key, 1, 2, 3, 4);
    std::array<uint32_t, N> expected;
    melty2_gen(&key, c0, N, expected.data());

    melty2::generator gen{{1, 2, 3, 4}, c0};
    test_assert(gen.ctr() == c0);
    std::array<uint32_t, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = gen();
    }
    test_assert(result == expected);
    test_assert(gen.ctr() == c0 + N);

    auto gen_u64 = melty2::generator::from_u64_seeds(UINT64_C(2) << 32 | 1, UINT64_C(4) << 32 | 3, c0);
    test_assert(gen_u64.ctr() == c0);
    for (size_t i = 0; i < N; ++i) {
        result[i] = gen_u64();
    }
    test_assert(result == expected);
}

void test_u64_seed() {
    constexpr size_t N = 50;
    const uint64_t seed = UINT64_C(0x0123456789abcdef);

    melty2::generator gen{seed, 7};
    melty2::generator gen_u32{{0x89abcdefu, 0x01234567u, 0, 0}, 7};
    std::array<uint32_t, N> result, expected;
    for (size_t i = 0; i < N; ++i) {
        result[i] = gen();
        expected[i] = gen_u32();
    }
    test_assert(result == expected);

    /* an int seed selects the uint64_t constructor, which gives the same key as the array */
    melty2::generator gen_int{42};
    melty2::generator gen_arr{{42, 0, 0, 0}};
    test_assert(gen_int() == gen_arr());
}

void test_init_u64() {
    const uint64_t s0 = UINT64_C(0xfedcba9876543210), s1 = UINT64_C(0x0f1e2d3c4b5a6978);
    melty2_key key, key_u32;
    melty2_init_u64(&key, s0, s1);
    melty2_init(&key_u32, 0x76543210u, 0xfedcba98u, 0x4b5a6978u, 0x0f1e2d3cu);
    test_assert(std::equal(key.v_, key.v_ + 6, key_u32.v_));

    auto gen = melty2::generator::from_u64_seeds(s0, s1, 3);
    melty2::generator gen_u32{{0x76543210u, 0xfedcba98u, 0x4b5a6978u, 0x0f1e2d3cu}, 3};
    test_assert(gen() == gen_u32());
}

int main() {
    test_default();
    test_seed();
    test_ctr();
    test_std_random();
    test_ctor_ctr();
    test_u64_seed();
    test_init_u64();
    return !!err;
}
