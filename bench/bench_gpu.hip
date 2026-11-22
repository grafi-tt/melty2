/* GPU benchmark: melty2 vs Philox4x32-10 (written from the specification of Salmon et al., SC'11). */
#include "melty2_gpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(1); } } while (0)

/* ---------- Philox4x32-10 ---------- */
struct u32x4 { uint32_t x, y, z, w; };

__host__ __device__ inline uint32_t mulhi32(uint32_t a, uint32_t b) {
#ifdef __HIP_DEVICE_COMPILE__
    return __umulhi(a, b);
#else
    return (uint32_t)(((uint64_t)a * b) >> 32);
#endif
}

__host__ __device__ inline u32x4 philox4x32_10(u32x4 c, uint32_t k0, uint32_t k1) {
    for (int r = 0; r < 10; ++r) {
        const uint32_t lo0 = 0xD2511F53u * c.x, hi0 = mulhi32(0xD2511F53u, c.x);
        const uint32_t lo1 = 0xCD9E8D57u * c.z, hi1 = mulhi32(0xCD9E8D57u, c.z);
        c = u32x4{hi1 ^ c.y ^ k0, lo1, hi0 ^ c.w ^ k1, lo0};
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    return c;
}

/* ---------- A: fill device memory ---------- */
__global__ void philox_fill(uint32_t k0, uint32_t k1, uint64_t n4, u32x4 *out) {
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        out[i] = philox4x32_10(u32x4{(uint32_t)i, (uint32_t)(i >> 32), 0, 0}, k0, k1);
    }
}

/* melty2 fill with 128bit stores: each thread computes 4 consecutive counters */
__global__ void melty2_fill4(melty2_key key, uint64_t ctr, uint64_t n4, uint4 *out) {
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        const uint64_t c = ctr + 4 * i;
        out[i] = make_uint4(melty2::gpu::gen(key, c), melty2::gpu::gen(key, c + 1), melty2::gpu::gen(key, c + 2), melty2::gpu::gen(key, c + 3));
    }
}

/* ---------- B: consume in kernel, all outputs used ---------- */
__global__ void melty2_consume(melty2_key key, int K, uint32_t *out) {
    const uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t acc = 0;
    for (int j = 0; j < K; ++j) acc ^= melty2::gpu::gen(key, t * K + j);
    out[t] = acc;
}
__global__ void philox_consume(uint32_t k0, uint32_t k1, int K, uint32_t *out) {
    const uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t acc = 0;
    for (int j = 0; j < K / 4; ++j) {
        const uint64_t c = t * (K / 4) + j;
        const u32x4 r = philox4x32_10(u32x4{(uint32_t)c, (uint32_t)(c >> 32), 0, 0}, k0, k1);
        acc ^= r.x ^ r.y ^ r.z ^ r.w;
    }
    out[t] = acc;
}

/* ---------- C: one 32bit number per call (random access) ---------- */
__global__ void philox_single(uint32_t k0, uint32_t k1, int K, uint32_t *out) {
    const uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t acc = 0;
    for (int j = 0; j < K; ++j) {
        const uint64_t c = t * K + j;
        acc ^= philox4x32_10(u32x4{(uint32_t)c, (uint32_t)(c >> 32), 0, 0}, k0, k1).x;
    }
    out[t] = acc;
}

/* ---------- E: per-thread key derivation from a seed, then K numbers ---------- */
__global__ void melty2_seeded(uint32_t seed, int K, uint32_t *out) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    melty2::gpu::generator g(seed, t, 0, 0);
    uint32_t acc = 0;
    for (int j = 0; j < K; ++j) acc ^= g();
    out[t] = acc;
}

/* ---------- D: latency of a dependent chain (one thread) ---------- */
__global__ void melty2_chain(melty2_key key, int n, uint32_t *out, long long *cycles) {
    uint32_t x = 0;
    const long long t0 = clock64();
    for (int i = 0; i < n; ++i) x = melty2::gpu::gen(key, x);
    const long long t1 = clock64();
    *out = x; *cycles = t1 - t0;
}
__global__ void philox_chain(uint32_t k0, uint32_t k1, int n, uint32_t *out, long long *cycles) {
    uint32_t x = 0;
    const long long t0 = clock64();
    for (int i = 0; i < n; ++i) x = philox4x32_10(u32x4{x, 0, 0, 0}, k0, k1).x;
    const long long t1 = clock64();
    *out = x; *cycles = t1 - t0;
}

/* ---------- timing helpers ---------- */
template <typename F>
static float time_ms(F f, int reps) {
    f();  /* warm up */
    CK(hipDeviceSynchronize());
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    CK(hipEventRecord(a));
    for (int i = 0; i < reps; ++i) f();
    CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
    float ms; CK(hipEventElapsedTime(&ms, a, b));
    return ms / reps;
}

static int kat() {
    struct { u32x4 c; uint32_t k0, k1; u32x4 e; } v[] = {
        {{0, 0, 0, 0}, 0, 0, {0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8}},
        {{0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff}, 0xffffffff, 0xffffffff, {0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd}},
        {{0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344}, 0xa4093822, 0x299f31d0, {0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1}},
    };
    int bad = 0;
    for (auto &t : v) {
        u32x4 r = philox4x32_10(t.c, t.k0, t.k1);
        if (r.x != t.e.x || r.y != t.e.y || r.z != t.e.z || r.w != t.e.w) {
            std::printf("Philox KAT mismatch: %08x %08x %08x %08x\n", r.x, r.y, r.z, r.w); bad = 1;
        }
    }
    std::printf("Philox known-answer test: %s\n", bad ? "FAILED" : "ok");
    return bad;
}

int main() {
    if (kat()) return 1;
    const melty2_key key = melty2::gpu::init(1, 2, 3, 4);
    const uint32_t k0 = 0x01234567, k1 = 0x89abcdef;

    /* A: fill 2^28 numbers (1 GiB) */
    const uint64_t N = (uint64_t)1 << 28;
    uint32_t *buf; CK(hipMalloc(&buf, N * 4));
    const int reps = 20;
    time_ms([&] { CK(hipMemsetD32((hipDeviceptr_t)buf, 0, N)); }, 200);  /* ramp up the clocks */
    float ms_p = time_ms([&] { philox_fill<<<65536, 256>>>(k0, k1, N / 4, (u32x4 *)buf); CK(hipGetLastError()); }, reps);
    float ms_m4 = time_ms([&] { melty2_fill4<<<65536, 256>>>(key, 0, N / 4, (uint4 *)buf); CK(hipGetLastError()); }, reps);
    {   /* check that fill4 gives the same numbers as fill */
        uint32_t *ref; CK(hipMalloc(&ref, (1 << 20) * 4));
        CK(melty2::gpu::fill(key, 0, 1 << 20, ref));
        melty2_fill4<<<1024, 256>>>(key, 0, (1 << 20) / 4, (uint4 *)buf);
        static uint32_t h1[1 << 20], h2[1 << 20];
        CK(hipMemcpy(h1, ref, sizeof h1, hipMemcpyDeviceToHost)); CK(hipMemcpy(h2, buf, sizeof h2, hipMemcpyDeviceToHost));
        std::printf("fill4 == fill: %s\n", std::memcmp(h1, h2, sizeof h1) ? "NO" : "yes");
        CK(hipFree(ref));
    }
    float ms_m = time_ms([&] { CK(melty2::gpu::fill(key, 0, N, buf)); }, reps);
    float ms_copy = time_ms([&] { CK(hipMemsetD32((hipDeviceptr_t)buf, 0, N)); }, reps);
    std::printf("\n[A] fill device memory, 2^28 x 32bit (1 GiB)\n");
    std::printf("  melty2 fill : %7.3f ms  %6.1f Gnum/s  %6.1f GB/s\n", ms_m, N / ms_m * 1e-6, N * 4 / ms_m * 1e-6);
    std::printf("  melty2 fill4: %7.3f ms  %6.1f Gnum/s  %6.1f GB/s  (128bit stores)\n", ms_m4, N / ms_m4 * 1e-6, N * 4 / ms_m4 * 1e-6);
    std::printf("  Philox fill : %7.3f ms  %6.1f Gnum/s  %6.1f GB/s\n", ms_p, N / ms_p * 1e-6, N * 4 / ms_p * 1e-6);
    std::printf("  (memset     : %7.3f ms                  %6.1f GB/s)\n", ms_copy, N * 4 / ms_copy * 1e-6);

    /* B, C, E: in-kernel generation, 2^20 threads */
    const int T = 1 << 20, blk = 256;
    const int K = 1024;
    const double total = (double)T * K;
    float b_m = time_ms([&] { melty2_consume<<<T / blk, blk>>>(key, K, buf); CK(hipGetLastError()); }, 5);
    float b_p = time_ms([&] { philox_consume<<<T / blk, blk>>>(k0, k1, K, buf); CK(hipGetLastError()); }, 5);
    float c_p = time_ms([&] { philox_single<<<T / blk, blk>>>(k0, k1, K, buf); CK(hipGetLastError()); }, 5);
    std::printf("\n[B] generate in kernel, all outputs used (2^20 threads x %d numbers)\n", K);
    std::printf("  melty2      : %7.3f ms  %6.1f Gnum/s\n", b_m, total / b_m * 1e-6);
    std::printf("  Philox (x4) : %7.3f ms  %6.1f Gnum/s\n", b_p, total / b_p * 1e-6);
    std::printf("\n[C] one 32bit number per call (random access)\n");
    std::printf("  melty2      : %7.3f ms  %6.1f Gnum/s  (same as B)\n", b_m, total / b_m * 1e-6);
    std::printf("  Philox (x1) : %7.3f ms  %6.1f Gnum/s\n", c_p, total / c_p * 1e-6);

    std::printf("\n[E] melty2 with per-thread key derivation from a seed (2^20 threads)\n");
    for (int k : {1, 16, 256}) {
        float e = time_ms([&] { melty2_seeded<<<T / blk, blk>>>(7, k, buf); CK(hipGetLastError()); }, 5);
        float b = time_ms([&] { melty2_consume<<<T / blk, blk>>>(key, k, buf); CK(hipGetLastError()); }, 5);
        std::printf("  K=%4d : seeded %7.3f ms, shared key %7.3f ms  (derivation %.2f ns/thread)\n", k, e, b, (e - b) * 1e6 / T);
    }

    /* D: latency */
    long long *d_cyc; uint32_t *d_x; CK(hipMalloc(&d_cyc, 8)); CK(hipMalloc(&d_x, 4));
    const int n = 1 << 16;
    long long cm, cp;
    melty2_chain<<<1, 1>>>(key, n, d_x, d_cyc); CK(hipMemcpy(&cm, d_cyc, 8, hipMemcpyDeviceToHost));
    philox_chain<<<1, 1>>>(k0, k1, n, d_x, d_cyc); CK(hipMemcpy(&cp, d_cyc, 8, hipMemcpyDeviceToHost));
    std::printf("\n[D] latency of a dependent chain (1 thread, %d calls), clock64 cycles per call\n", n);
    std::printf("  melty2      : %6.1f\n", (double)cm / n);
    std::printf("  Philox      : %6.1f\n", (double)cp / n);
    return 0;
}
