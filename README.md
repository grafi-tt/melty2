# Melty2

**Fast, portable and easy-to-use counter-based pseudo-random number generator** for accelerators and CPUs, written in C99 with a C++ wrapper. Each 32bit output is a pure function of a 192bit key and a 64bit counter: `out = f(key, ctr)`.

**Being counter-based, it needs no mutable state.** This makes it far easier to use in parallel and on accelerators than generators with a sequential state.

* **Parallel streams without coordination.** Give each thread its own seed, or its own counter range under a shared seed. No jump-ahead tables nor stream splitting is required.
* **Random access.** Any position of the stream is computed directly, so skipping, rewinding and partitioning work into chunks are trivial.
* **Reproducibility independent of parallelism.** The value for a given (seed, counter) never depends on the number of threads, the scheduling, or the vector width, so a simulation gives bitwise identical results on different machines, e.g. on a CPU and on a GPU running the same kernel.
* **Tiny footprint.** Only the 192bit key and the counter have to be held, and the counter is often derived from indices already at hand (e.g. particle ID and time step). Taking a snapshot or a checkpoint of the generator is just saving the counter.
* **No warm-up.** A fresh key yields good outputs from the first counter.

**It uses only 32bit additions, xors and rotations (ARX), and no multiplication.** One output costs 20 ARX steps with a short critical path, so it runs efficiently on GPUs and on processors without an integer multiplier. The C library dispatches SIMD implementations (AVX-512, AVX2, SSE2, SVE and NEON) at runtime. The statistical quality is checked by PractRand, TestU01 BigCrush and the Hamming-weight dependency test, including keys with structure (see [Algorithm detail](#algorithm-detail)).

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
    printf("%08x\n", r[0]);         /* prints 40247540 */
    return 0;
}
```

It provides the following functions.

```c
void melty2_init(melty2_key *key, uint32_t seed0, uint32_t seed1, uint32_t seed2, uint32_t seed3);
void melty2_gen(const melty2_key *key, uint64_t ctr, size_t len, uint32_t *out);
void melty2_rawblkgen(const melty2_key *key, uint32_t ctr_lo, uint32_t ctr_hi, uint32_t *out);
```

`melty2_init` derives a 192bit key from a 128bit seed by SipHash-2-4, so even similar seeds (e.g. thread IDs) yield unrelated keys. `melty2_gen` stores the outputs for counters `ctr`, `ctr + 1`, ..., `ctr + len - 1` to `out`; the counter wraps around modulo 2^64. `melty2_rawblkgen` is the low-level function generating a block of `MELTY2_RAWBLKLEN` (32) outputs, where `ctr_lo` must not exceed `2^32 - MELTY2_RAWBLKLEN`.

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

The counter can be read and set by `gen.ctr()` and `gen.ctr(new_ctr)`. Jumping ahead costs nothing more than generating one block.

### Build

```sh
git clone --recursive https://github.com/grafi-tt/melty2.git
cd melty2
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
cmake --install build --prefix /usr/local
```

Then use it from another CMake project by `find_package(melty2)` and `target_link_libraries(your_target PRIVATE melty2)`, or add this repository by `add_subdirectory`.

The following options are available.

* `BUILD_SHARED_LIBS` (default `OFF`): build a shared library.
* `MELTY2_ENABLE_CPU_FEATURES` (default `ON`): select the SIMD implementation at runtime by [cpu_features](https://github.com/google/cpu_features). If disabled, only the generic implementation is built.
* `MELTY2_ENABLE_TEST` (default `ON`): build tests.
* `MELTY2_ENABLE_BIGCRUSH` (default `OFF`): build the BigCrush runner, which requires TestU01.

## Benchmark

<!-- TODO: comparison with Philox and Threefry (Random123), on CPU and GPU -->

## Similar projects

### Counter-based generators

* [Random123](https://www.deshawresearch.com/resources_random123.html) (Philox, Threefry, ARS)

  Philox is excellent, but it relies on 32x32→64bit multiplications, and produces 128bit per call while the latency to obtain a 32bit output is long. Threefry is ARX-based, derived from the Threefish block cipher used in the Skein hash function, but has a 256bit or 128bit state and more rounds. ARS uses 5 rounds of AES with a simplified key schedule, which is very fast with AES instructions (AES-NI, ARMv8 Crypto Extension), but they are unavailable on GPUs and many accelerators.

* [Squares](https://arxiv.org/abs/2004.06278)
* ChaCha (as used by e.g. [rand_chacha](https://crates.io/crates/rand_chacha))

<!-- TODO: more -->

### Other fast generators

* [xoshiro / xoroshiro](https://prng.di.unimi.it/)

  Very good generators, while assigning streams to many threads requires jump-ahead to be used correctly.

## Algorithm detail

<!-- TODO: elaborate -->

### Key derivation

The 128bit seed is used as the SipHash-2-4 key, and the 192bit melty2 key is the concatenation of SipHash-2-4 outputs for the messages 0, 1 and 2. As the outputs are uniform, keys of distinct seeds are independent, and keys related to each other (whose difference cancels with that of the counter) appear only with a negligible probability.

### Kernel

The key consists of six 32bit words `v[0..5]`. The 64bit counter `(hi, lo)` is injected as

```c
v[0] ^= lo; v[1] += hi; v[2] ^= hi; v[3] += lo;
```

and then 20 ARX steps (7 additions, 3 xors and 10 xor-rotations) are applied, where the last written word is the output. See [`src/melty2_kernel.h`](src/melty2_kernel.h) for the exact definition.

The kernel consists of two phases.

* In the first half (steps 1 to 11), all the six words are mixed together. The counter-injected words `v[0..3]` are mixed first, and then `v[5]` and `v[4]` are absorbed by additions, so that every key word affects the remaining state.
* In the second half (steps 12 to 20), the state is narrowed down to the four words `v[0]`, `v[1]`, `v[3]` and `v[5]` (`v[2]` is only read once, and `v[4]` is no longer used). As each step updates one of the four words, every word is updated more frequently than in a six-word state, which yields stronger diffusion per step toward the output.

The original kernel designed by the author had 22 steps. It was shortened to 20 steps by Claude Opus 5.5 (Anthropic) through a structure search, which minimizes a counter-differential bias metric at reduced depths, under the constraints that every operation affects the output and every key word passes through an addition. The metric compares outputs for counters `c` and `c + 2^j` at each intermediate step; the bias vanishes 4 steps before the output for random keys, while it does 2 to 3 steps before the output for the original 22-step kernel.

### Weak keys

When each of a huge number of threads uses its own seed, a key with some structure, such as a zero word, appears with a probability that is not negligible. The following kinds of keys were tested by the differential metric, PractRand of the least significant bits, PractRand and BigCrush subsets.

* a zero, all-one or Hamming weight 1 word
* equal, negated, MSB-flipped or rotated word pairs
* zero half-words in multiple words, low Hamming weight word pairs, and near-equal word pairs

The most sensitive case is that `v[4]` and `v[5]`, which the counter is not injected to and which are absorbed only by a few steps of the first half, are sparse at the same time. Even then the bias vanishes 3 steps before the output. Relations between words (equal, rotated, negated or near-equal pairs) and sparse `v[0..2]` are as harmless as random keys.

### Statistical tests

<!-- TODO: link to the result files -->

* PractRand: passed up to 64GB for multiple seeds, both counter orders (lower or upper 32bit of the counter incrementing first) and weak keys.
* TestU01 BigCrush: passed for multiple seeds and both counter orders, with and without bit reversal.
* [Hamming-weight dependency test](https://prng.di.unimi.it/hwd.php): <!-- TODO -->

## License

Melty2 is dedicated to the public domain under [CC0 1.0](LICENSE). [cpu_features](third_party/cpu_features) is distributed under the Apache License 2.0 ([LICENSE.cpu_features](LICENSE.cpu_features)).
