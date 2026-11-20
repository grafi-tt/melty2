#include "melty2_gpu.h"

#include <cinttypes>
#include <cstdio>
#include <vector>

#if defined(__HIPCC__)
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuMemcpy hipMemcpy
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuGetLastError hipGetLastError
#define gpuGetErrorString hipGetErrorString
#define gpuSuccess hipSuccess
#else
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuGetLastError cudaGetLastError
#define gpuGetErrorString cudaGetErrorString
#define gpuSuccess cudaSuccess
#endif

#define CHECK_GPU(expr) do { \
    if ((expr) != gpuSuccess) { \
        std::fprintf(stderr, "%s:%d %s failed: %s\n", __FILE__, __LINE__, #expr, gpuGetErrorString(gpuGetLastError())); \
        return 1; \
    } \
} while (0)

static int err = 0;

static void check_equal(const char *title, const std::vector<uint32_t> &result, const std::vector<uint32_t> &expected) {
    size_t n_diff = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (result[i] != expected[i]) {
            if (n_diff++ == 0) {
                std::fprintf(stderr, "%s -- i=%zu Result: 0x%08" PRIx32 ", Expected: 0x%08" PRIx32 "\n", title, i, result[i], expected[i]);
            }
        }
    }
    if (n_diff) {
        std::fprintf(stderr, "%s -- %zu mismatches\n", title, n_diff);
        ++err;
    }
}

struct Seed {
    uint32_t s[4];
};

/* Each thread derives the key from its own seed on device, and generates per_thread numbers by the generator. */
__global__ void per_thread_kernel(const Seed *seeds, uint64_t ctr, unsigned per_thread, uint32_t *out) {
    const unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    const Seed sd = seeds[t];
    melty2::gpu::generator g(sd.s[0], sd.s[1], sd.s[2], sd.s[3], ctr);
    for (unsigned k = 0; k < per_thread; ++k) {
        out[(size_t)t * per_thread + k] = g();
    }
}

static int test_fill(const char *title, uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3, uint64_t ctr, size_t len) {
    melty2_key key;
    melty2_init(&key, s0, s1, s2, s3);
    std::vector<uint32_t> expected(len);
    melty2_gen(&key, ctr, len, expected.data());

    /* the key derived by the host side of the GPU header must be identical */
    const melty2_key key_gpu = melty2::gpu::init(s0, s1, s2, s3);
    for (int i = 0; i < 6; ++i) {
        if (key_gpu.v_[i] != key.v_[i]) {
            std::fprintf(stderr, "%s -- key mismatch at %d\n", title, i);
            ++err;
        }
    }

    uint32_t *d_out;
    CHECK_GPU(gpuMalloc(&d_out, len * sizeof(uint32_t)));
    CHECK_GPU(melty2::gpu::fill(key_gpu, ctr, len, d_out));
    CHECK_GPU(gpuDeviceSynchronize());
    std::vector<uint32_t> result(len);
    CHECK_GPU(gpuMemcpy(result.data(), d_out, len * sizeof(uint32_t), gpuMemcpyDeviceToHost));
    CHECK_GPU(gpuFree(d_out));
    check_equal(title, result, expected);
    return 0;
}

static int test_per_thread() {
    const unsigned blocks = 64, threads = 128, n = blocks * threads, per_thread = 37;
    const uint64_t ctr = UINT64_C(0xfffffff0);  /* crosses the 2^32 boundary */

    std::vector<Seed> seeds(n);
    for (unsigned t = 0; t < n; ++t) {
        seeds[t] = Seed{{t, t * 2654435761u, 0, t % 7 == 0 ? 0xffffffffu : 12345u}};
    }
    std::vector<uint32_t> expected((size_t)n * per_thread);
    for (unsigned t = 0; t < n; ++t) {
        melty2_key key;
        melty2_init(&key, seeds[t].s[0], seeds[t].s[1], seeds[t].s[2], seeds[t].s[3]);
        melty2_gen(&key, ctr, per_thread, expected.data() + (size_t)t * per_thread);
    }

    Seed *d_seeds;
    uint32_t *d_out;
    CHECK_GPU(gpuMalloc(&d_seeds, n * sizeof(Seed)));
    CHECK_GPU(gpuMalloc(&d_out, expected.size() * sizeof(uint32_t)));
    CHECK_GPU(gpuMemcpy(d_seeds, seeds.data(), n * sizeof(Seed), gpuMemcpyHostToDevice));
    per_thread_kernel<<<blocks, threads>>>(d_seeds, ctr, per_thread, d_out);
    CHECK_GPU(gpuGetLastError());
    CHECK_GPU(gpuDeviceSynchronize());
    std::vector<uint32_t> result(expected.size());
    CHECK_GPU(gpuMemcpy(result.data(), d_out, result.size() * sizeof(uint32_t), gpuMemcpyDeviceToHost));
    CHECK_GPU(gpuFree(d_seeds));
    CHECK_GPU(gpuFree(d_out));
    check_equal("per-thread seeds", result, expected);
    return 0;
}

int main() {
    int fail = 0;
    fail |= test_fill("seed 0, ctr 0", 0, 0, 0, 0, 0, 1000);
    fail |= test_fill("2^32 boundary", 3141592653u, 2718281828u, 123456789u, 42u, UINT64_C(0xffffff00), 100003);
    fail |= test_fill("2^64 wrap", 1, 2, 3, 4, UINT64_C(0xffffffffffffff00), 4096);
    fail |= test_fill("large", 5, 0, 0, 0, UINT64_C(0x123456789), (size_t)1 << 26);
    fail |= test_fill("empty", 0, 0, 0, 0, 0, 0);
    fail |= test_per_thread();
    return fail || err;
}
