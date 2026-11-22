// GPU benchmark of melty2 and other counter-based generators, for CUDA and HIP. The results are written to
// stdout in YAML.
//
// Philox4x32 and Threefry4x32 are implemented here following the paper of Salmon et al. (SC'11), as in
// bench_cpu.cc, and checked on the device by the known-answer vectors of Random123 at startup. The numbers
// of rounds are template parameters; the benchmark uses those proposed in the paper.
//
// The benchmarks are the following, all for kBulk numbers.
// - Fill device memory: melty2 is measured by melty2::gpu::fill of the header, as it is the path actually
//   used. The others write a block of 128bit to each counter by a grid-stride loop, with the same launch
//   configuration as melty2::gpu::fill.
// - Generate in kernel: each of kThreads threads generates the numbers for its own range of counters and
//   xors them, so the memory access is only one store per thread. All outputs of each block are used, or
//   only the first 32bit word of each block (one 32bit number per counter, e.g. random access).
// - Key derivation of melty2: as the generation in kernel, each thread derives the keys from the seeds for
//   its own range, and xors their words. This is compared with the generation of numbers, for the case
//   that each thread derives its own key from a seed.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#include "melty2.h"
#include "melty2_gpu.h"

#if defined(__HIPCC__)
#define gpuError_t hipError_t
#define gpuEvent_t hipEvent_t
#define gpuDeviceProp hipDeviceProp_t
#define gpuGetDeviceProperties hipGetDeviceProperties
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuMemset hipMemset
#define gpuMemcpy hipMemcpy
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuEventCreate hipEventCreate
#define gpuEventDestroy hipEventDestroy
#define gpuEventRecord hipEventRecord
#define gpuEventSynchronize hipEventSynchronize
#define gpuEventElapsedTime hipEventElapsedTime
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuGetLastError hipGetLastError
#define gpuGetErrorString hipGetErrorString
#define gpuSuccess hipSuccess
#else
#define gpuError_t cudaError_t
#define gpuEvent_t cudaEvent_t
#define gpuDeviceProp cudaDeviceProp
#define gpuGetDeviceProperties cudaGetDeviceProperties
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuMemset cudaMemset
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuEventCreate cudaEventCreate
#define gpuEventDestroy cudaEventDestroy
#define gpuEventRecord cudaEventRecord
#define gpuEventSynchronize cudaEventSynchronize
#define gpuEventElapsedTime cudaEventElapsedTime
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuGetLastError cudaGetLastError
#define gpuGetErrorString cudaGetErrorString
#define gpuSuccess cudaSuccess
#endif

#define BENCH_FORCE_INLINE inline __attribute__((__always_inline__))

// exit on an error of the GPU runtime
#define CHECK_GPU(expr) do { \
    const gpuError_t err = (expr); \
    if (err != gpuSuccess) { \
        std::fprintf(stderr, "%s:%d %s failed: %s\n", __FILE__, __LINE__, #expr, gpuGetErrorString(err)); \
        std::exit(1); \
    } \
} while (0)

namespace {

//
// PRNG kernels
//

// Philox4x32
template <int kRounds, int r = 0>
__device__ BENCH_FORCE_INLINE void philox4x32(uint32_t x[4], uint32_t k0, uint32_t k1) {
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
__device__ BENCH_FORCE_INLINE void threefry4x32_loop(uint32_t x[4], const uint32_t k[5]) {
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
__device__ BENCH_FORCE_INLINE void threefry4x32(uint32_t x[4], const uint32_t key[4]) {
    const uint32_t k[5] = {key[0], key[1], key[2], key[3], UINT32_C(0x1BD11BDA) ^ key[0] ^ key[1] ^ key[2] ^ key[3]};
    for (int i = 0; i < 4; ++i) {
        x[i] += k[i];
    }
    threefry4x32_loop<kRounds>(x, k);
}

//
// Constants and tests
//

// the numbers of rounds proposed by Salmon et al.
constexpr int kPhiloxRounds = 10;
constexpr int kThreefryRounds = 20;

// the initial hash values of SHA-2, from the square roots of the first primes
constexpr uint32_t IV[4] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a};

// generators of a block for a counter, passed to the kernels by value

struct Melty2 {
    static constexpr int kWords = 1;
    melty2_key key;
    __device__ BENCH_FORCE_INLINE void operator()(uint64_t ctr, uint32_t x[1]) const {
        x[0] = melty2::gpu::gen(key, ctr);
    }
};

template <int kRounds>
struct Philox4x32 {
    static constexpr int kWords = 4;
    uint32_t key[2];
    __device__ BENCH_FORCE_INLINE void operator()(uint64_t ctr, uint32_t x[4]) const {
        x[0] = static_cast<uint32_t>(ctr);
        x[1] = static_cast<uint32_t>(ctr >> 32);
        x[2] = 0;
        x[3] = 0;
        philox4x32<kRounds>(x, key[0], key[1]);
    }
};

template <int kRounds>
struct Threefry4x32 {
    static constexpr int kWords = 4;
    uint32_t key[4];
    __device__ BENCH_FORCE_INLINE void operator()(uint64_t ctr, uint32_t x[4]) const {
        x[0] = static_cast<uint32_t>(ctr);
        x[1] = static_cast<uint32_t>(ctr >> 32);
        x[2] = 0;
        x[3] = 0;
        threefry4x32<kRounds>(x, key);
    }
};

// Compute the known-answer tests into out[4 * i], ..., out[4 * i + 3] for the i-th test.
__global__ void known_answer_kernel(uint32_t *out) {
    const uint32_t pi[8] = {0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344, 0xa4093822, 0x299f31d0, 0x082efa98, 0xec4e6c89};
    for (int i = 0; i < 4; ++i) {
        for (int w = 0; w < 4; ++w) out[4 * i + w] = pi[w];
    }
    philox4x32<10>(out, pi[4], pi[5]);
    philox4x32<7>(out + 4, pi[4], pi[5]);
    threefry4x32<20>(out + 8, pi + 4);
    threefry4x32<13>(out + 12, pi + 4);
}

// known-answer tests on the device (vectors of Random123, tests/kat_vectors)
bool random123_known_answer_test() {
    struct {
        const char *name;
        uint32_t want[4];
    } const tests[4] = {
        {"Philox4x32-10", {0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1}},
        {"Philox4x32-7", {0x4dfccaba, 0x190a87f0, 0xc47362ba, 0xb6b5242a}},
        {"Threefry4x32-20", {0x59cd1dbb, 0xb8879579, 0x86b5d00c, 0xac8b6d84}},
        {"Threefry4x32-13", {0x4aa71d8f, 0x734738c2, 0x431fc6a8, 0xae6debf1}},
    };
    uint32_t *out;
    CHECK_GPU(gpuMalloc(&out, 16 * sizeof(uint32_t)));
    known_answer_kernel<<<1, 1>>>(out);
    CHECK_GPU(gpuGetLastError());
    uint32_t got[16];
    CHECK_GPU(gpuMemcpy(got, out, sizeof(got), gpuMemcpyDeviceToHost));
    CHECK_GPU(gpuFree(out));

    bool ok = true;
    for (int i = 0; i < 4; ++i) {
        if (!std::equal(tests[i].want, tests[i].want + 4, got + 4 * i)) {
            std::fprintf(stderr, "%s: known-answer test failed\n", tests[i].name);
            ok = false;
        }
    }
    return ok;
}

//
// Kernels
//

// numbers of uint32_t items generated by each benchmark
constexpr size_t kBulk = 1 << 28;

// launch configuration, same as melty2::gpu::fill
constexpr unsigned kBlockDim = 256;
constexpr unsigned kFillGridDim = 65536;

// threads and numbers per thread for generation in kernel
constexpr uint32_t kThreads = 1 << 20;
constexpr uint32_t kPerThread = kBulk / kThreads;

struct alignas(16) U32x4 {
    uint32_t v[4];
};

// Store the blocks for counters ctr, ctr + 1, ..., ctr + n - 1 to out, by 128bit stores.
template <typename BlockGen>
__global__ void fill_kernel(BlockGen gen, uint64_t ctr, size_t n, U32x4 *out) {
    static_assert(BlockGen::kWords == 4, "a block must be 128bit");
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
        U32x4 x;
        gen(ctr + i, x.v);
        out[i] = x;
    }
}

// Each thread xors kPerThread numbers, using the first kUsedWords words of each block, and stores it to
// out[thread]. The threads use consecutive ranges of counters from ctr.
template <int kUsedWords, typename BlockGen>
__global__ void generate_kernel(BlockGen gen, uint64_t ctr, uint32_t *out) {
    constexpr uint32_t kBlocksPerThread = kPerThread / kUsedWords;
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t acc = 0;
    for (uint32_t j = 0; j < kBlocksPerThread; ++j) {
        uint32_t x[BlockGen::kWords];
        gen(ctr + static_cast<uint64_t>(t) * kBlocksPerThread + j, x);
        for (int w = 0; w < kUsedWords; ++w) acc ^= x[w];
    }
    out[t] = acc;
}

// Each thread xors the words of kPerThread keys of melty2, derived from the seeds (seed0, seed1, i) for
// 64bit i, and stores it to out[thread]. The threads use consecutive ranges of i from first.
__global__ void derive_key_kernel(uint32_t seed0, uint32_t seed1, uint64_t first, uint32_t *out) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t acc = 0;
    for (uint32_t j = 0; j < kPerThread; ++j) {
        const uint64_t i = first + static_cast<uint64_t>(t) * kPerThread + j;
        const melty2_key key = melty2::gpu::init(seed0, seed1, static_cast<uint32_t>(i), static_cast<uint32_t>(i >> 32));
        for (int w = 0; w < 6; ++w) acc ^= key.v_[w];
    }
    out[t] = acc;
}

//
// Benchmark
//

// The best time of kRepeat runs of `launch` in nanoseconds, after a warm-up run.
template <typename Launch>
uint64_t best_time_ns(Launch launch) {
    constexpr int kRepeat = 5;

    gpuEvent_t start, stop;
    CHECK_GPU(gpuEventCreate(&start));
    CHECK_GPU(gpuEventCreate(&stop));
    launch();
    CHECK_GPU(gpuDeviceSynchronize());

    uint64_t best_ns = std::numeric_limits<uint64_t>::max();
    for (int r = 0; r < kRepeat; ++r) {
        CHECK_GPU(gpuEventRecord(start));
        launch();
        CHECK_GPU(gpuEventRecord(stop));
        CHECK_GPU(gpuEventSynchronize(stop));
        float ms;
        CHECK_GPU(gpuEventElapsedTime(&ms, start, stop));
        best_ns = std::min(best_ns, static_cast<uint64_t>(ms * 1e6));
    }
    CHECK_GPU(gpuEventDestroy(start));
    CHECK_GPU(gpuEventDestroy(stop));
    return best_ns;
}

// xor of out[0], ..., out[n - 1] in device memory
uint32_t xor_all(const uint32_t *out, size_t n) {
    std::vector<uint32_t> host(n);
    CHECK_GPU(gpuMemcpy(host.data(), out, n * sizeof(uint32_t), gpuMemcpyDeviceToHost));
    uint32_t x = 0;
    for (uint32_t v : host) x ^= v;
    return x;
}

// xor of the first and the last elements of buf in device memory, filled with kBulk numbers
uint32_t xor_front_back(const uint32_t *buf) {
    uint32_t front, back;
    CHECK_GPU(gpuMemcpy(&front, buf, sizeof(uint32_t), gpuMemcpyDeviceToHost));
    CHECK_GPU(gpuMemcpy(&back, buf + kBulk - 1, sizeof(uint32_t), gpuMemcpyDeviceToHost));
    return front ^ back;
}

struct Lib {
    const char *name;
    double fill_ns = 0;
    double all_ns = 0;
    double one_ns = 0;
    uint32_t fill_checksum = 0;
    uint32_t all_checksum = 0;
    uint32_t one_checksum = 0;
};

// Measure the generation in kernel for a generator, using all words of each block and the first one.
template <typename BlockGen>
void bench_generate(Lib &l, BlockGen gen, uint64_t ctr, uint32_t *out) {
    const unsigned grid = kThreads / kBlockDim;
    l.all_ns = static_cast<double>(best_time_ns([&] {
        generate_kernel<BlockGen::kWords><<<grid, kBlockDim>>>(gen, ctr, out);
        CHECK_GPU(gpuGetLastError());
    })) / kBulk;
    l.all_checksum = xor_all(out, kThreads);
    if constexpr (BlockGen::kWords == 1) {
        l.one_ns = l.all_ns;
        l.one_checksum = l.all_checksum;
        return;
    }
    l.one_ns = static_cast<double>(best_time_ns([&] {
        generate_kernel<1><<<grid, kBlockDim>>>(gen, ctr, out);
        CHECK_GPU(gpuGetLastError());
    })) / kBulk;
    l.one_checksum = xor_all(out, kThreads);
}

// Measure a generator of 128bit blocks.
template <typename BlockGen>
Lib bench_block_generator(const char *name, BlockGen gen, uint64_t ctr, uint32_t *buf) {
    Lib l{name};
    l.fill_ns = static_cast<double>(best_time_ns([&] {
        fill_kernel<<<kFillGridDim, kBlockDim>>>(gen, ctr, kBulk / 4, reinterpret_cast<U32x4 *>(buf));
        CHECK_GPU(gpuGetLastError());
    })) / kBulk;
    l.fill_checksum = xor_front_back(buf);
    bench_generate(l, gen, ctr, buf);
    return l;
}

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

    gpuDeviceProp prop;
    CHECK_GPU(gpuGetDeviceProperties(&prop, 0));
    std::fprintf(stderr, "device: %s\n", prop.name);

    uint32_t *buf;
    CHECK_GPU(gpuMalloc(&buf, kBulk * sizeof(uint32_t)));

    // keep the GPU busy for a while before the measurements, so that its clock rises
    for (int i = 0; i < 200; ++i) {
        CHECK_GPU(gpuMemset(buf, 0, kBulk * sizeof(uint32_t)));
    }
    CHECK_GPU(gpuDeviceSynchronize());

    // filling memory by memset, as the reference of the memory bandwidth
    const double memset_ns = static_cast<double>(best_time_ns([&] {
        CHECK_GPU(gpuMemset(buf, 0, kBulk * sizeof(uint32_t)));
    })) / kBulk;

    std::vector<Lib> libs;

    {
        Lib l{"melty2"};
        const Melty2 gen{melty2::gpu::init(IV[0], IV[1], IV[2], IV[3])};
        l.fill_ns = static_cast<double>(best_time_ns([&] {
            CHECK_GPU(melty2::gpu::fill(gen.key, ctr, kBulk, buf));
        })) / kBulk;
        l.fill_checksum = xor_front_back(buf);

        // the first and the last numbers should be the same as those of the CPU library
        melty2_key key;
        melty2_init(&key, IV[0], IV[1], IV[2], IV[3]);
        uint32_t front, back;
        melty2_gen(&key, ctr, 1, &front);
        melty2_gen(&key, ctr + kBulk - 1, 1, &back);
        if (l.fill_checksum != (front ^ back)) {
            std::fprintf(stderr, "melty2: different from the CPU library\n");
            return 1;
        }

        bench_generate(l, gen, ctr, buf);
        libs.push_back(l);
    }

    // key derivation of melty2, for kBulk keys
    const double derive_ns = static_cast<double>(best_time_ns([&] {
        derive_key_kernel<<<kThreads / kBlockDim, kBlockDim>>>(IV[0], IV[1], ctr, buf);
        CHECK_GPU(gpuGetLastError());
    })) / kBulk;
    const uint32_t derive_checksum = xor_all(buf, kThreads);

    libs.push_back(bench_block_generator("Philox4x32-10", Philox4x32<kPhiloxRounds>{{IV[0], IV[1]}}, ctr, buf));
    libs.push_back(bench_block_generator("Threefry4x32-20", Threefry4x32<kThreefryRounds>{{IV[0], IV[1], IV[2], IV[3]}}, ctr, buf));

    CHECK_GPU(gpuFree(buf));

    const struct {
        const char *title;
        double Lib::*field;
    } benches[] = {
        {"Fill device memory", &Lib::fill_ns},
        {"Generate in kernel, using all outputs", &Lib::all_ns},
        {"Generate in kernel, one 32bit number per counter", &Lib::one_ns},
    };
    for (const auto &b : benches) {
        std::printf("%s:\n", b.title);
        for (const auto &l : libs) std::printf("  %s:\n    ns_per_number: %.4g\n", l.name, l.*(b.field));
    }
    std::printf("Memset of device memory (reference):\n  ns_per_number: %.4g\n", memset_ns);
    std::printf("Key derivation of melty2 (reference):\n  ns_per_key: %.4g\n", derive_ns);
    for (const auto &l : libs) {
        std::fprintf(stderr, "checksums of %s: fill %08x, all outputs %08x, one per counter %08x\n", l.name,
                     l.fill_checksum, l.all_checksum, l.one_checksum);
    }
    std::fprintf(stderr, "checksum of key derivation of melty2: %08x\n", derive_checksum);
    return 0;
}
