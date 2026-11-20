# Melty2

**Fast, portable and easy-to-use counter-based pseudo-random number generator** for accelerators and CPUs, written in C99 with a C++ wrapper. Each 32bit output is a pure function of a 192bit key and a 64bit counter: `out = f(key, ctr)`.

**Being counter-based, it needs no mutable state, so it is much easier to use in parallel than common generators.**

* **Parallel streams without coordination.** Give each thread its own seed, or its own counter range under a shared seed. No jump-ahead tables nor stream splitting is required.
* **Random access.** Any position of the stream is computed directly, so skipping, rewinding and partitioning work into chunks are trivial.
* **Reproducibility independent of parallelism.** The value for a given (seed, counter) never depends on the number of threads, the scheduling, or the vector width, so a simulation gives bitwise identical results on different machines, e.g. on a CPU and on a GPU running the same kernel.
* **Tiny footprint.** Only the 192bit key and the counter have to be held, and the counter is often derived from indices already at hand (e.g. particle ID and time step). Taking a snapshot or a checkpoint of the generator is just saving the counter.

**It uses only 32bit additions, xors and rotations (ARX), and no multiplication.** One output costs only 20 ARX steps, so it runs efficiently on GPUs and on processors without an integer multiplier. The C library dispatches SIMD implementations (AVX-512, AVX2, SSE2, SVE and NEON) at runtime, and a header-only implementation for CUDA and HIP is also provided. The statistical quality is checked by PractRand, TestU01 BigCrush and the Hamming-weight dependency test, including keys with structure (see [Algorithm detail](#algorithm-detail)).

**Melty2 is not a cryptographically secure generator.** It is designed for simulations and tests, where statistical quality, speed and reproducibility matter, and its outputs may be predictable to an adversary. Don't use it for keys, nonces, tokens, passwords or anything security-sensitive; use the random source of the OS (e.g. `getrandom`) or a cryptographically secure generator instead.

Its name derives from *melting* in Isaac Asimov's novel *The Gods Themselves*, where three beings of the para-universe merge into one. Melty2 merges six 32bit words, twice as many as the three beings.

## Usage

### C

Build and install the library by CMake, then include `melty2.h`.

```c
#include <stdio.h>
#include "melty2.h"

int main(void) {
    melty2_key key;
    melty2_init(&key, 0, 0, 0, 0);  /* 128bit seed */
    uint32_t r[8];
    melty2_gen(&key, 0, 8, r);      /* outputs for counters 0, 1, ..., 7 */
    printf("%08x\n", r[0]);         /* prints 008ae4d0 */
    return 0;
}
```

It provides the following functions.

```c
void melty2_init(melty2_key *key, uint32_t seed0, uint32_t seed1, uint32_t seed2, uint32_t seed3);
void melty2_init_u64(melty2_key *key, uint64_t seed0, uint64_t seed1);
void melty2_gen(const melty2_key *key, uint64_t ctr, size_t len, uint32_t *out);
```

`melty2_init` takes a 128bit seed as four 32bit integers, lower ones first, and stretches it to a key by a cryptographic algorithm, so even similar seeds (e.g. thread IDs) yield unrelated keys. `melty2_init_u64` takes the same seed as two 64bit integers; for example, `melty2_init_u64(&key, 42, 0)` is the same as `melty2_init(&key, 42, 0, 0, 0)`.

`melty2_gen` stores `len` numbers for the counters `ctr`, `ctr + 1`, ... to `out`. The counter wraps around after 2^64 - 1.

For advanced users, the low-level function generating a block of `MELTY2_RAWBLKLEN` (32) numbers is also provided.

```c
void melty2_rawblkgen(const melty2_key *key, uint32_t ctr_lo, uint32_t ctr_hi, uint32_t *out);
```

The counter is given by its lower and upper 32bit. As the carry from the lower to the upper is not handled, `ctr_lo` must not exceed `2^32 - MELTY2_RAWBLKLEN`. `melty2_gen` is built on this function, and handles any counter range.

### C++

`melty2.hpp` provides `melty2::generator`, which satisfies the UniformRandomBitGenerator requirements, so it can be used with the distributions of `<random>`.

```cpp
#include <iostream>
#include <random>
#include "melty2.hpp"

int main() {
    melty2::generator gen{{3141592653u, 2718281828u, 123456789u, 42u}};
    std::uniform_real_distribution<double> dist{0.0, 1.0};
    std::cout << dist(gen) << std::endl;
}
```

A generator can also be constructed from a 64bit seed as `melty2::generator gen{seed}`, which is the same as the 128bit seed `{(uint32_t)seed, seed >> 32, 0, 0}`, or from two 64bit integers as `melty2::generator::from_u64_seeds(seed0, seed1)`, which is the same as `melty2_init_u64`. The counter is given as the optional last argument of them, and can be read and set by `gen.ctr()` and `gen.ctr(new_ctr)`. Jumping ahead costs nothing more than generating one block.

### Independent generators for tests

A counter-based generator makes it easy to give every test its own independent random numbers while keeping them reproducible. Hash the test name into a part of the 128bit seed, and put a global seed into the rest.

```cpp
#include <cstdint>
#include <string_view>
#include "melty2.hpp"

/* FNV-1a 64bit hash; a simple hash suffices, as melty2_init decorrelates similar seeds */
constexpr uint64_t fnv1a64(std::string_view s) {
    uint64_t h = 0xcbf29ce484222325;
    for (char c : s) {
        h ^= static_cast<unsigned char>(c);
        h *= 0x00000100000001b3;
    }
    return h;
}

melty2::generator test_rng(uint64_t global_seed, std::string_view test_name) {
    return melty2::generator::from_u64_seeds(global_seed, fnv1a64(test_name));
}

/* in each test */
auto rng = test_rng(global_seed, "sort/random_ints");
```

* Each test gets numbers unrelated to those of the other tests. It avoids the common pitfall of initializing a generator with the same seed at the beginning of every test, where all tests silently consume the same random sequence.
* The numbers of a test don't depend on which tests run before it, or on the order or the parallelism of the test runner. So a failed test can be reproduced alone.
* Changing the global seed (e.g. by an environment variable or a command line option) runs all the tests with different random inputs, and the seed printed on failure reproduces it.

### GPU (CUDA and HIP)

`melty2_gpu.h` is a header-only implementation for CUDA and HIP. It doesn't require linking the library, and its outputs are identical to those of the CPU library for the same seed and counter.

The simplest way is that each thread derives its own key from a seed containing the thread ID. The key derivation costs one ChaCha8 block once per thread, and then each number costs only 20 ARX steps.

```cpp
#include "melty2_gpu.h"

__global__ void simulate(uint32_t seed, float *out) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    melty2::gpu::generator gen(seed, tid, 0, 0);  /* 128bit seed (seed, tid, 0, 0) */
    float sum = 0.0f;
    for (int i = 0; i < 100; ++i) {
        sum += (gen() >> 8) * (1.0f / 16777216.0f);  /* uniform in [0, 1) */
    }
    out[tid] = sum;
}
```

Alternatively, threads can share a key and use disjoint counters, e.g. `melty2::gpu::gen(key, (uint64_t)item << 32 | i)` for the `i`-th number of the work item `item`. This avoids the key derivation in each thread, and the result doesn't depend on how the work items are assigned to threads.

It provides the following functions and class in the namespace `melty2::gpu`, all of which are callable from both host and device code except `fill`.

```cpp
melty2_key init(uint32_t seed0, uint32_t seed1, uint32_t seed2, uint32_t seed3);  /* same as melty2_init */
melty2_key init_u64(uint64_t seed0, uint64_t seed1);  /* same as melty2_init_u64 */
uint32_t gen(const melty2_key &key, uint64_t ctr);  /* the number for the counter */
class generator;  /* holds a key and a counter; constructed from a key, four 32bit seeds or a 64bit seed */
error_t fill(const melty2_key &key, uint64_t ctr, size_t len, uint32_t *out, stream_t stream = 0);
```

`fill` launches a kernel that stores the numbers for counters `ctr`, ..., `ctr + len - 1` to the device memory `out`, which is the GPU counterpart of `melty2_gen`. `error_t` and `stream_t` are `cudaError_t` and `cudaStream_t` for CUDA, or `hipError_t` and `hipStream_t` for HIP.

In a CMake project, link the `melty2_gpu` target, which only adds the include directory.

### Build

```sh
git clone --recursive https://github.com/grafi-tt/melty2.git
cd melty2
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
cmake --install build --prefix /usr/local
```

Then use it from another CMake project by `find_package(melty2)` and `target_link_libraries(your_target PRIVATE melty2)` (or `melty2_gpu` for the GPU header), or add this repository by `add_subdirectory`.

The following options are available.

* `BUILD_SHARED_LIBS` (default `OFF`): build a shared library.
* `MELTY2_ENABLE_CPU_FEATURES` (default `ON`): select the SIMD implementation at runtime by [cpu_features](https://github.com/google/cpu_features). If disabled, only the generic implementation is built.
* `MELTY2_ENABLE_TEST` (default `ON`): build tests.
* `MELTY2_ENABLE_BIGCRUSH` (default `OFF`): build the BigCrush runner, which requires TestU01.
* `MELTY2_GPU_TEST` (default empty): build the GPU test by `CUDA` or `HIP`, which requires CMake 3.21. Set `CMAKE_CUDA_ARCHITECTURES` or `CMAKE_HIP_ARCHITECTURES` as needed.

## Benchmark

### GPU

The following results compare melty2 with Philox4x32-10 on AMD Radeon RX 6800 (RDNA2, gfx1030), compiled by HIP 7.16 (Clang 23.1.4) with `-O3`. Philox is implemented for the benchmark following its specification, and verified by the known-answer test vectors of Random123.

| Benchmark | melty2 | Philox4x32-10 |
|---|---|---|
| Fill device memory (2^28 numbers) | 110 Gnum/s (440 GB/s) | 110 Gnum/s (440 GB/s) |
| Generate in kernel, using all outputs | 242 Gnum/s | 181 Gnum/s |
| Generate in kernel, one 32bit number per call | 242 Gnum/s | 46.5 Gnum/s |
| Latency of a dependent chain | 98 cycles | 248 cycles |

* Filling memory is limited by the memory bandwidth for both (`memset` achieves 490 GB/s).
* When generating numbers in a kernel and consuming them without memory access, melty2 is 1.3 times faster than Philox, even though Philox yields four 32bit numbers per call. This is because 32bit multiplications (especially for the upper half) have a lower throughput than additions, xors and rotations on this GPU, while melty2 uses only ARX operations.
* When only one 32bit number is needed for each counter (e.g. random access, or a single draw per particle), the other three outputs of Philox are wasted, and melty2 is 5.2 times faster.
* The latency to obtain a 32bit number, measured by a single thread whose next counter depends on the previous output, is 2.5 times shorter.
* Deriving a key from a seed in each thread costs about as much as generating 10 numbers, so it is negligible when a thread uses hundreds of numbers or more.

<!-- TODO: CPU benchmark, comparison with Threefry and ARS -->

## Similar projects

### Counter-based generators

* [Random123](https://dl.acm.org/doi/10.1145/2063384.2063405) (Philox, Threefry, ARS)

  Philox is excellent, but it relies on 32x32→64bit multiplications, and produces 128bit per call while the latency to obtain a 32bit output is long. Threefry is ARX-based, derived from the Threefish block cipher used in the Skein hash function, but has a 256bit or 128bit state and more rounds. ARS uses 5 rounds of AES with a simplified key schedule, which is very fast with AES instructions (AES-NI, ARMv8 Crypto Extension), but they are unavailable on GPUs and many accelerators.

  Philox and Threefry are widely used, e.g. Philox in cuRAND, PyTorch and NumPy, and Threefry as the default of JAX.

* Weyl sequence based generators: [SplitMix64](https://dl.acm.org/doi/10.1145/2660193.2660195), [rapidrand](https://docs.rs/rapidrand/latest/rapidrand/)

  They compute the `n`-th number as `mix(seed + n * gamma)`, so they allow random access like counter-based generators, and they are very fast on CPUs. However, the mixing relies on 64bit (or 64x64→128bit for rapidrand) multiplications. Moreover, when streams differ only by the seed, they are offsets on a single cycle of length 2^64, so the streams of many threads may overlap.

* [Squares](https://arxiv.org/abs/2004.06278)

  A counter-based generator based on the middle-square method, which relies on 64bit multiplications.

* [ChaCha](https://cr.yp.to/chacha.html) (as used by e.g. [rand_chacha](https://crates.io/crates/rand_chacha))

  ChaCha with 8, 12 or 20 rounds is a cryptographically strong counter-based generator. As one block yields 512bit, it is efficient for bulk generation, but costly when only a single 32bit number is needed.

### Other fast generators

* [xoshiro / xoroshiro](https://prng.di.unimi.it/)

  Very good generators, while assigning streams to many threads requires jump-ahead to be used correctly.

## Algorithm detail

### What is a counter-based generator

Most generators, such as Mersenne Twister, xoshiro and PCG, hold a state, and produce each number by updating the state and applying an output function to it. The update is cheap, but the numbers must be produced in sequence. To use them in parallel, the stream has to be split, for example by jump-ahead functions, and each thread has to hold its own state.

A counter-based generator (Salmon et al., [*Parallel random numbers: as easy as 1, 2, 3*](https://dl.acm.org/doi/10.1145/2063384.2063405), SC'11) instead computes the `n`-th number directly as `f(key, n)`, where `f` is a fixed mixing function like a reduced-round block cipher. The key selects a stream, and the counter selects a position in it. Nothing is updated, so an arbitrary number of threads can compute any of the numbers independently.

The price is that every number has to go through the whole mixing function, while a state-based generator mixes its state only a little per number. So the mixing function must be designed to be both cheap and strong enough: the outputs for counters close to each other (differing in a few bits, or by a small integer) must look unrelated. Melty2 achieves this with 20 ARX steps for each 32bit number, and the counter-differential metric described below measures exactly this property.

### Key derivation

The 192bit melty2 key consists of the words 4 to 9 of the [ChaCha8](https://cr.yp.to/chacha.html) block, where the ChaCha key is zero and the 128bit seed is given as the counter and nonce. ChaCha8 uses only 32bit ARX operations, so its implementation is straightforward on GPUs and processors without 64bit arithmetic. As ChaCha8 is a well-trusted cipher, its outputs can be regarded as uniform. Thus keys of distinct seeds are independent, and keys related to each other appear only with a negligible probability.

### Kernel

The key consists of six 32bit words `v[0..5]`. The 64bit counter `(hi, lo)` is injected as

```c
v[0] ^= lo; v[1] += hi; v[2] ^= hi; v[3] += lo;
```

and then 20 ARX steps (7 additions, 3 xors and 10 xor-rotations) are applied, where the last written word is the output. See [`src/melty2_kernel.h`](src/melty2_kernel.h) for the exact definition.

The kernel consists of two phases.

* In the first half (steps 1 to 11), all the six words are mixed together. The words `v[0..3]`, into which the counter is injected, are mixed first. Then `v[5]` and `v[4]` are absorbed by additions, so that every key word affects the remaining state.
* In the second half (steps 12 to 20), the state is narrowed down to the four words `v[0]`, `v[1]`, `v[3]` and `v[5]` (`v[2]` is only read once, and `v[4]` is no longer used). As each step updates one of the four words, every word is updated more frequently than in a six-word state, which yields stronger diffusion per step toward the output.

These steps are almost sequential: with bitwise rotation instructions, the kernel including the counter injection consists of 34 instructions, and its critical path spans 31 of them. This is intended, as a sequential dependency mixes the state more thoroughly than independent operations do. Instruction-level parallelism is not essential, since the outputs for multiple counters can be computed in parallel by SIMD lanes or GPU threads. This also suits accelerators whose cores are not superscalar.

The original kernel designed by the author had 22 steps. It was shortened to 20 steps by Claude Opus 5.5 (Anthropic) through a structure search, which minimizes a counter-differential bias metric at reduced depths, under the constraints that every operation affects the output and every key word passes through an addition. The metric compares outputs for counters `c` and `c + 2^j` at each intermediate step; the bias vanishes 4 steps before the output for random keys, while it does 2 to 3 steps before the output for the original 22-step kernel.

### Weak keys

When each of a huge number of threads uses its own seed, a key with some structure, such as a zero word, appears with a non-negligible probability. Such keys were first tested by the differential metric on GPU, and then by PractRand on the least significant bits on CPU. After that, heavier tests were performed: PractRand on the full outputs, and a subset of BigCrush.

The weak keys considered are as follows:

* a zero, all-one or Hamming weight 1 word,
* equal, negated, MSB-flipped or rotated word pairs,
* zero half-words in multiple words, low Hamming weight word pairs, and near-equal word pairs.

The most sensitive case is that `v[4]` and `v[5]`, which the counter is not injected to and which are absorbed only by a few steps of the first half, are sparse at the same time. Even then the bias vanishes 3 steps before the output. Relations between words (equal, rotated, negated or near-equal pairs) and sparse `v[0..2]` are as harmless as random keys.

### Statistical tests

<!-- TODO: link to the result files -->

* PractRand: passed up to 64GB for multiple seeds, both counter orders (lower or upper 32bit of the counter incrementing first) and weak keys.
* TestU01 BigCrush: passed for multiple seeds and both counter orders, with and without bit reversal.
* [Hamming-weight dependency test](https://prng.di.unimi.it/hwd.php): <!-- TODO -->

## License

Melty2 is dedicated to the public domain under [CC0 1.0](LICENSE). [cpu_features](third_party/cpu_features) is distributed under the Apache License 2.0 ([LICENSE.cpu_features](LICENSE.cpu_features)).
