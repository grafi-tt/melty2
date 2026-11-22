// CPU benchmark of melty2 and other counter-based generators, single thread. The results are written
// to stdout in YAML.
//
// Philox4x32, Threefry4x32 and ARS are implemented here following the paper of Salmon et
// al. (SC'11), and checked by the known-answer vectors of Random123 at startup. The numbers of rounds are
// template parameters; the benchmark uses those proposed in the paper (kPhiloxRounds and so on).
//
// melty2 is measured by melty2_gen of the library, with the SIMD implementation dispatched at runtime.
// Like melty2_rawblkgen, the other generators compute a chunk of kChunk counters by a plain loop over the
// counters, which the compiler vectorizes. Their kernels are force-inlined into the loop, as the kernel of
// melty2 is expanded there as a macro; otherwise the compiler may keep a call in the loop by its inlining
// cost and give up vectorization. ARS uses AES-NI for each block, without vectorization.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#if defined(__AES__)
#include <immintrin.h>
#define BENCH_HAVE_ARS 1
#endif

#include "melty2.h"

#if defined(__GNUC__)
#define BENCH_FORCE_INLINE inline __attribute__((__always_inline__))
#elif defined(_MSC_VER)
#define BENCH_FORCE_INLINE __forceinline
#else
#define BENCH_FORCE_INLINE inline
#endif

namespace {

//
// PRNG kernels
//

// Philox4x32
template <int kRounds, int r = 0>
BENCH_FORCE_INLINE void philox4x32(uint32_t x[4], uint32_t k0, uint32_t k1) {
    if constexpr (r != kRounds) {
        uint64_t p0 = UINT64_C(0xD2511F53) * x[0];
        uint64_t p1 = UINT64_C(0xCD9E8D57) * x[2];
        x[0] = static_cast<uint32_t>(p1 >> 32) ^ x[1] ^ k0;
        x[2] = static_cast<uint32_t>(p0 >> 32) ^ x[3] ^ k1;
        x[1] = static_cast<uint32_t>(p1);
        x[3] = static_cast<uint32_t>(p0);
        k0 += UINT32_C(0x9E3779B9);
        k1 += UINT32_C(0xBB67AE85);
        philox4x32<kRounds, r + 1>(x, k0, k1);
    }
}

// Threefry
constexpr int kThreefryRot[8][2] = {{10, 26}, {11, 21}, {13, 27}, {23, 5}, {6, 20}, {17, 11}, {25, 10}, {18, 20}};

template <int kRounds, int r = 0>
BENCH_FORCE_INLINE void threefry4x32_loop(uint32_t x[4], const uint32_t k[5]) {
    if constexpr (r != kRounds) {
        constexpr int a = r % 2 == 0 ? 1 : 3;
        constexpr int b = r % 2 == 0 ? 3 : 1;
        constexpr int m = kThreefryRot[r % 8][0];
        constexpr int n = kThreefryRot[r % 8][1];
        x[0] += x[a];
        x[a] = x[a] << m | x[a] >> (32 - m);
        x[a] ^= x[0];
        x[2] += x[b];
        x[b] = x[b] << n | x[b] >> (32 - n);
        x[b] ^= x[2];

        if (r % 4 == 3) {
            constexpr uint32_t s = r / 4 + 1;
            for (int i = 0; i < 4; ++i) {
                x[i] += k[(s + i) % 5];
            }
            x[3] += s;
        }
        threefry4x32_loop<kRounds, r + 1>(x, k);
    }
}

template <int kRounds>
BENCH_FORCE_INLINE void threefry4x32(uint32_t x[4], const uint32_t key[4]) {
    const uint32_t k[5] = {key[0], key[1], key[2], key[3], UINT32_C(0x1BD11BDA) ^ key[0] ^ key[1] ^ key[2] ^ key[3]};
    for (int i = 0; i < 4; ++i) {
        x[i] += k[i];
    }
    threefry4x32_loop<kRounds>(x, k);
}

// ARS (AES-NI)
#ifdef BENCH_HAVE_ARS
template <int kRounds, int r = 1>
BENCH_FORCE_INLINE __m128i ars4x32_loop(__m128i v, __m128i k, __m128i weyl) {
    k = _mm_add_epi64(k, weyl);
    if constexpr (r != kRounds) {
        return ars4x32_loop<kRounds, r + 1>(_mm_aesenc_si128(v, k), k, weyl);
    } else {
        return _mm_aesenclast_si128(v, k);
    }
}

template <int kRounds>
BENCH_FORCE_INLINE void ars4x32(uint32_t x[4], const uint32_t key[4]) {
    const __m128i weyl = _mm_set_epi64x(0xBB67AE8584CAA73Bull, 0x9E3779B97F4A7C15ull);
    const __m128i k = _mm_loadu_si128(reinterpret_cast<const __m128i *>(key));
    const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i *>(x));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(x), ars4x32_loop<kRounds>(_mm_xor_si128(v, k), k, weyl));
}
#endif

//
// Constants and tests
//

// the numbers of rounds proposed by Salmon et al.
constexpr int kPhiloxRounds = 10;
constexpr int kThreefryRounds = 20;
constexpr int kArsRounds = 7;

// known-answer tests (vectors of Random123, tests/kat_vectors)
bool random123_known_answer_test() {
    bool ok = true;
    auto check = [&](const char *name, const uint32_t *got, const uint32_t *want, int n) {
        for (int i = 0; i < n; ++i)
            if (got[i] != want[i]) {
                std::fprintf(stderr, "%s: known-answer test failed\n", name);
                ok = false;
                return;
            }
    };
    const uint32_t pi[8] = {0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344, 0xa4093822, 0x299f31d0, 0x082efa98, 0xec4e6c89};
    {
        const uint32_t want[4] = {0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1};
        const uint32_t want7[4] = {0x4dfccaba, 0x190a87f0, 0xc47362ba, 0xb6b5242a};
        uint32_t x[4] = {pi[0], pi[1], pi[2], pi[3]}, y[4] = {pi[0], pi[1], pi[2], pi[3]};
        philox4x32<10>(x, pi[4], pi[5]);
        philox4x32<7>(y, pi[4], pi[5]);
        check("Philox4x32-10", x, want, 4);
        check("Philox4x32-7", y, want7, 4);
    }

    {
        const uint32_t want[4] = {0x59cd1dbb, 0xb8879579, 0x86b5d00c, 0xac8b6d84};
        const uint32_t want13[4] = {0x4aa71d8f, 0x734738c2, 0x431fc6a8, 0xae6debf1};
        uint32_t x[4] = {pi[0], pi[1], pi[2], pi[3]}, y[4] = {pi[0], pi[1], pi[2], pi[3]};
        threefry4x32<20>(x, pi + 4);
        threefry4x32<13>(y, pi + 4);
        check("Threefry4x32-20", x, want, 4);
        check("Threefry4x32-13", y, want13, 4);
    }

#ifdef BENCH_HAVE_ARS
    {
        const uint32_t want[4] = {0xa516e7d6, 0x8357ad74, 0x5b59b3ec, 0x8763fff3};
        uint32_t x[4] = {pi[0], pi[1], pi[2], pi[3]};
        ars4x32<10>(x, pi + 4);
        check("ARS4x32-10", x, want, 4);
    }
#endif
    return ok;
}

//
// Benchmark
//

// counters per chunk, as MELTY2_RAWBLKLEN
constexpr uint32_t kChunk = 32;

// Generate a chunk of RNGs for kChunk counters (should be fully inlined for benchmark purpose). Each block
// is generated into a local array and then copied to out; if gen writes to out directly, the compiler
// fails to vectorize the loop.
template <int BlockSize, typename BlockGen>
void fill_chunk(uint64_t& ctr, uint32_t *out, BlockGen gen) {
    for (uint32_t k = 0; k < kChunk; ++k) {
        uint32_t x[BlockSize];
        gen(ctr++, x);
        for (int w = 0; w < BlockSize; ++w) out[BlockSize * k + w] = x[w];
    }
}

// allocate RNG buffer
constexpr size_t kBuf = 1 << 16;
static std::vector<uint32_t> BulkBuf(kBuf);

// A function filling a buffer of kBuf numbers with the blocks of gen, by chunks.
template <int BlockSize, typename BlockGen>
auto block_buf_gen(BlockGen gen) {
    return [gen](uint64_t& ctr, uint32_t* out) {
        for (uint32_t* end = out + kBuf; out < end; out += BlockSize * kChunk) {
            fill_chunk<BlockSize, BlockGen>(ctr, out, gen);
        }
    };
}

// numbers of uint32_t items generated by bulk generation
constexpr size_t kBulk = 1 << 28;

// bulk generation into a buffer by repeated fill; buf_gen(ctr, out) fills out with kBuf numbers for the
// counters from ctr, and advances ctr
template <typename BufGen>
uint32_t fill_bulk(uint64_t ctr, BufGen buf_gen) {
    uint32_t checksum = 0;
    for (size_t i = 0; i < kBulk / kBuf; ++i) {
        buf_gen(ctr, BulkBuf.data());
        checksum ^= BulkBuf.front() ^ BulkBuf.back();
    }
    return checksum;
}

// The best time of kRepeat runs of `bulk`. The results of `bulk` are accumulated by xor into checksum,
// which is printed at the end so that the compiler does not omit the generation.
template <int BlockSize, typename BufGen>
std::pair<double, double> bench_generator(uint64_t ctr, BufGen buf_gen, uint32_t &checksum) {
    constexpr int kRepeat = 5;

    uint64_t best_ns = std::numeric_limits<uint64_t>::max();
    for (int r = 0; r < kRepeat; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        checksum ^= fill_bulk(ctr, buf_gen);
        const auto t1 = std::chrono::steady_clock::now();
        const uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        best_ns = std::min(best_ns, ns);
    }

    double best_bulk_ns = static_cast<double>(best_ns) / kBulk;
    double best_one_ns = best_bulk_ns * BlockSize;
    return {best_bulk_ns, best_one_ns};
}

struct Lib {
    std::string name;
    double bulk_ns = 0;
    double one_ns = 0;
    uint32_t checksum = 0;
};

// the initial hash values of SHA-2, from the square roots of the first primes
static const uint32_t IV[4] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a};

}  // namespace

int main(int argc, char **argv) {
    if (!random123_known_answer_test()) return 1;

    uint64_t ctr = 0;
    if (argc > 2) {
        std::fprintf(stderr, "usage: %s [FIRST_COUNTER]\n", argv[0]);
        return 2;
    }
    if (argc == 2) {
        char *end;
        ctr = std::strtoull(argv[1], &end, 0);
        if (end == argv[1] || *end != '\0') {
            std::fprintf(stderr, "usage: %s [FIRST_COUNTER]\n", argv[0]);
            return 2;
        }
    }

    std::vector<Lib> libs;

    {
        Lib l{"melty2"};
        melty2_key mkey;
        melty2_init(&mkey, IV[0], IV[1], IV[2], IV[3]);
        auto buf_gen = [&mkey](uint64_t& ctr, uint32_t* out) {
            melty2_gen(&mkey, ctr, kBuf, out);
            ctr += kBuf;
        };
        std::tie(l.bulk_ns, l.one_ns) = bench_generator<1>(ctr, buf_gen, l.checksum);
        libs.push_back(l);
    }

    {
        Lib l{"Philox4x32-" + std::to_string(kPhiloxRounds)};
        auto gen = [](uint64_t ctr, uint32_t* out) {
            uint32_t x[4] = {static_cast<uint32_t>(ctr), static_cast<uint32_t>(ctr >> 32)};
            philox4x32<kPhiloxRounds>(x, IV[0], IV[1]);
            std::memcpy(out, x, sizeof(x));
        };
        std::tie(l.bulk_ns, l.one_ns) = bench_generator<4>(ctr, block_buf_gen<4>(gen), l.checksum);
        libs.push_back(l);
    }

    {
        Lib l{"Threefry4x32-" + std::to_string(kThreefryRounds)};
        auto gen = [](uint64_t ctr, uint32_t* out) {
            uint32_t x[4] = {static_cast<uint32_t>(ctr), static_cast<uint32_t>(ctr >> 32)};
            threefry4x32<kThreefryRounds>(x, IV);
            std::memcpy(out, x, sizeof(x));
        };
        std::tie(l.bulk_ns, l.one_ns) = bench_generator<4>(ctr, block_buf_gen<4>(gen), l.checksum);
        libs.push_back(l);
    }

#ifdef BENCH_HAVE_ARS
    {
        Lib l{"ARS-" + std::to_string(kArsRounds) + " (AES-NI)"};
        auto gen = [](uint64_t ctr, uint32_t* out) {
            uint32_t x[4] = {static_cast<uint32_t>(ctr), static_cast<uint32_t>(ctr >> 32)};
            ars4x32<kArsRounds>(x, IV);
            std::memcpy(out, x, sizeof(x));
        };
        std::tie(l.bulk_ns, l.one_ns) = bench_generator<4>(ctr, block_buf_gen<4>(gen), l.checksum);
        libs.push_back(l);
    }
#endif

    const struct {
        const char *title;
        double Lib::*field;
    } benches[] = {
        {"Bulk generation into a buffer", &Lib::bulk_ns},
        {"One 32bit number per counter", &Lib::one_ns},
    };
    for (const auto &b : benches) {
        std::printf("%s:\n", b.title);
        for (const auto &l : libs) std::printf("  %s:\n    ns_per_number: %.4g\n", l.name.c_str(), l.*(b.field));
    }
    for (const auto &l : libs) std::fprintf(stderr, "checksum of %s: %08x\n", l.name.c_str(), l.checksum);
    return 0;
}
