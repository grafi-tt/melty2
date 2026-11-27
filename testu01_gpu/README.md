# testu01_gpu

A library running the tests of [TestU01](http://simul.iro.umontreal.ca/testu01/tu01.html) on a GPU, by HIP (ROCm) or CUDA. The tests are reimplemented after the source code of TestU01 1.2.3: the test functions used by its batteries SmallCrush, Crush and BigCrush (33 of them, see [Tests](#tests)), with the parameters of their instances in these batteries and sample sizes that can be scaled.

On a current GPU, BigCrush takes tens of seconds instead of hours on a CPU core, Crush about ten seconds, and SmallCrush less than one. Most tests make their observations from other outputs than TestU01 does (see [Tests](#tests)), so their p-values agree with those of TestU01 statistically; the tests on the single sequence give the same statistics as TestU01 on the same outputs, or statistics differing only by the order of their floating-point operations (Fourier3, AppearanceSpacings). They screen generators quickly, and sample sizes far beyond those of BigCrush are practical.

Currently only counter-based generators are supported: generators whose output at any position of their sequence can be computed directly.

## Build

Requirements: ROCm (HIP, hiprtc, rocFFT and rocThrust) or the CUDA Toolkit (NVRTC, cuFFT and Thrust; a GPU of compute capability 6.0 or later), Abseil (tested with 20260107.1), and an installation of TestU01. The host code is written in C++20, while the header requires only C++17 of its users.

Note that the Abseil package of Ubuntu 24.04 (20220623) is too old: you will probably need to build Abseil from source, as the CI does.

With CMake (3.21 or later):

```sh
cmake -B build                                       # Release, the static library, HIP (for gfx1030) if found, else CUDA
cmake -B build -DCMAKE_HIP_ARCHITECTURES="gfx1030;gfx1100" -DTestU01_ROOT=/opt/testu01
cmake -B build -DTESTU01_GPU_BACKEND=CUDA -DCMAKE_CUDA_ARCHITECTURES="80;89"  # CUDA (default: the GPUs of the build machine)
cmake -B build -DTESTU01_GPU_HOST_FLAGS=-march=native  # the SIMD of the host for the tests on the host (LinearComp), not portable
cmake --build build                                  # build/libtestu01_gpu.a and the example build/splitmix_bigcrush
cmake --install build --prefix /opt/testu01_gpu      # the headers, the library and the CMake package
```

Other options: `-DBUILD_SHARED_LIBS=ON` (a shared library), `-DTESTU01_GPU_EXAMPLES=OFF`. A CMake project uses the installed library by

```cmake
find_package(testu01_gpu REQUIRED)  # (CMAKE_PREFIX_PATH including the prefix of the installation)
target_link_libraries(app PRIVATE testu01_gpu::testu01_gpu)
```

or the source tree by `add_subdirectory(testu01_gpu)` and the same target. `TESTU01_GPU_BACKEND` chooses HIP or CUDA (by default HIP if CMake finds a HIP compiler, else CUDA); the installed package records it (`testu01_gpu_BACKEND`). The kernels of the tests are compiled at run time for the current device (by hiprtc or NVRTC), but those of rocThrust or Thrust (the sorts and prefix sums) are compiled by the build: for HIP, for `CMAKE_HIP_ARCHITECTURES` (default gfx1030), which must include the GPU used; for CUDA, for `CMAKE_CUDA_ARCHITECTURES` (default `native`, the GPUs of the build machine; a GPU newer than these runs their PTX, if any, compiled by the driver). `TESTU01_GPU_HOST_FLAGS` adds flags to the host code only (a list): the default is portable, and `-march=native` (passed to the host compiler by `-Xarch_host` for HIP and `-Xcompiler` for CUDA) lets the compiler vectorize the Berlekamp-Massey algorithm of LinearComp with the SIMD of the host (e.g. with AVX-512).

A program built without CMake includes `include/testu01_gpu.hpp` and links `build/libtestu01_gpu.a -lprobdist -lmylib -lhiprtc -lrocfft` with `hipcc`, or for CUDA `build/libtestu01_gpu.a -lprobdist -lmylib -lnvrtc -lcufft -lcuda -lcudart` with `nvcc`.

## Usage

The generator is given as device source (HIP or CUDA: the same source serves both, `__device__` functions of C++17), defining in the global namespace

```cpp
__device__ inline void testu01_gpu_generate(const testu01_gpu::Params &p, uint64_t pos,
                                            uint32_t (&out)[testu01_gpu::kLanes]);
```

which writes the 32-bit outputs at the positions `pos, pos + 1, ..., pos + kLanes - 1` (kLanes = 16) of the sequence of the parameters `p` (32 words, interpreted by the source as it likes: a key, a seed, a selection of streams, ...). The library compiles its kernels with the source by hiprtc (HIP) or NVRTC (CUDA), so the generator is inlined into them as straight-line code; they have no standard headers, but `uint32_t`, `uint64_t`, `UINT32_C` and `UINT64_C` are available. A source needing the intrinsics of a backend tells them by `__HIPCC_RTC__` and `__CUDACC_RTC__`. For CUDA, the kernels are compiled to a CUBIN for the GPU, or to the PTX of the newest architecture that NVRTC supports below it (then compiled by the driver).

```cpp
#include "testu01_gpu.hpp"

const testu01_gpu::Generator gen(source);  // compiles the kernels (a few seconds), reused across runs
testu01_gpu::Params params{};              // e.g. the key in params.w[0..]
testu01_gpu::Battery battery;              // the device memory, allocated as needed and kept
testu01_gpu::Options o;                    // tests (default all), scale, first position, limit
o.report = [](const testu01_gpu::Result &r) { std::puts(testu01_gpu::format(r).c_str()); };
const testu01_gpu::Summary s = battery.run(gen, params, o);  // min_p, max_p, max_z, flagged
```

* `Options::suite`: the battery whose instances of the tests run: `Suite::kSmallCrush`, `Suite::kCrush` or `Suite::kBigCrush` (the default).
* `Options::tests`: the comma-separated names of the tests to run (`testu01_gpu::kAllTests` by default, in the order below); a test without instances in the battery runs nothing.
* `Options::scale`: multiplies the sample sizes `n` of the battery (e.g. 16 for 16 times the samples).
* `Options::first`: the first position used. Every test and replication takes the next range of consecutive positions, as TestU01 takes the outputs of a single stream in turn (so the positions of a test depend on the tests run before it).
* `Options::limit`: stops the tests once a z-score is above it, for searches rejecting such a generator anyway.
* `Options::memory`: the device memory of the large tests (2 GiB by default). The tests whose memory would grow with the samples make their points again by parts beyond it, or batch their work within it (the column memory of [Tests](#tests); see [The memory budget](#the-memory-budget)), so any scale runs in a fixed device memory; the results are the same for any budget.
* `Options::fast`: faster computations whose results differ from those of TestU01 by the order of the floating-point sums (off by default): the Anderson-Darling statistics of SampleProd and SampleMean are summed on GPU by a parallel reduction instead of in order by `gofs_AndersonDarling` on the host.
* `Options::report`: called for every statistic with a `Result` (the name, the parameters, the kind of statistic, its value, the p-value and a z-score); `format` gives the line printed above, marking a p-value outside [0.001, 0.999] with `*****` as TestU01 does.

`run` throws `testu01_gpu::TestFailure` for a test failing completely (e.g. the observations of a test taking far more outputs than expected, for a very bad generator), `std::invalid_argument` for parameters outside the range a test supports (e.g. d > 64 of SimpPoker and CouponCollector, s > 32 or r + s > 32 of the tests taking s bits of an output), and `std::runtime_error` for the errors of HIP and hiprtc (or CUDA and NVRTC).

[examples/splitmix_bigcrush.cpp](examples/splitmix_bigcrush.cpp) tests SplitMix64 as a counter-based generator: `build/splitmix_bigcrush [seed] [scale] [tests|all] [fast] [small|crush|big]`. [../search/testu01_stest.hip](../search/testu01_stest.hip) tests the ARX structures of melty2, with the streams of counters or keys interleaved by the parameters.

## Tests

Only the test functions used by SmallCrush, Crush and BigCrush are ported, with the parameters of their instances in these batteries. Porting the rest of TestU01 is not planned:
* the other test functions, e.g. sknuth_Serial, sknuth_SerialSparse, sknuth_RunIndep, smarsa_Opso, smarsa_CAT, snpair_BickelBreiman, sspectral_Fourier1 and Fourier2, sstring_HammingWeight, svaria_SumLogs, svaria_CollisionArgMax, swalk_RandomWalk1a, swalk_VarGeoN and VarGeoP, and the modules sentrop and sspacings;
* the other batteries: Rabbit, Alphabit, BlockAlphabit, FIPS_140_2 and pseudoDIEHARD;
* the generators of TestU01 (its modules u*).

The tests and the instances of the batteries (the numbers of instances below: the calls of the test functions in `bbattery.c` of TestU01 1.2.3, for a 64-bit `long`) are those of SmallCrush (10, with 15 statistics), Crush (96, 144 statistics) and BigCrush (106, 160 statistics). The column CPU names the work a test does on the host, besides the final statistic and p-value that every test computes there (see [Tests using the CPU](#tests-using-the-cpu)). The column memory tells how a test keeps within the memory budget (see [The memory budget](#the-memory-budget)); the other tests use a memory that doesn't grow with the samples. The last column tells how a statistic compares with that of TestU01 on the same outputs of the generator; its links lead to [Differences from TestU01](#differences-from-testu01).

| name | test of TestU01 | SmallCrush | Crush | BigCrush | CPU | memory | compared with TestU01 on the same outputs |
|---|---|---|---|---|---|---|---|
| `poker` | sknuth_SimpPoker | 1 | 4 | 4 |  |  | agrees statistically: [observations from per-thread pieces](#observations-from-per-thread-pieces) |
| `coupon` | sknuth_CouponCollector | 1 | 4 | 4 |  |  | agrees statistically: [observations from per-thread pieces](#observations-from-per-thread-pieces) |
| `gap` | sknuth_Gap | 1 | 4 | 4 |  |  | agrees statistically: [observations from per-thread pieces](#observations-from-per-thread-pieces) |
| `maxoft` | sknuth_MaxOft (the chi-square and the AD statistic) | 1 | 4 | 4 | [AD statistic](#the-anderson-darling-statistics) | sorted by parts (AD) | agrees statistically: [observations from per-thread pieces](#observations-from-per-thread-pieces) |
| `collision` | smarsa_CollisionOver | | 8 | 10 |  |  | agrees statistically: [observations from per-thread pieces](#observations-from-per-thread-pieces) |
| `kcollision` | sknuth_Collision | 1 | | |  |  | agrees statistically: [observations from per-thread pieces](#observations-from-per-thread-pieces) |
| `birthday` | smarsa_BirthdaySpacings | 1 | 7 | 9 | [spacings by buckets](#birthdayspacings), beyond the budget | sorted by parts | agrees statistically: [observations from per-thread pieces](#observations-from-per-thread-pieces) |
| `closepairs` | snpair_ClosePairs (NP, mNP, mNP1, mNP2, NJumps, mNP2S) | | 3 | 4 | [the six statistics](#closepairs) | by slabs | agrees statistically: [observations from per-thread pieces](#observations-from-per-thread-pieces) |
| `cpbitmatch` | snpair_ClosePairsBitMatch | | 2 |  | hashed by parts | | agrees statistically: [observations from per-thread pieces](#observations-from-per-thread-pieces) |
| `permutation` | sknuth_Permutation | | 2 | 4 |  |  | agrees statistically: [observations from per-thread pieces](#observations-from-per-thread-pieces) |
| `run` | sknuth_Run | | 2 | 2 |  |  | identical |
| `serial` | smarsa_SerialOver | | 2 | 2 |  |  | identical statistic; [rounding difference](#rounding-the-p-value-of-the-upper-tail) of the p-value (1–2 ulp) |
| `collpermut` | sknuth_CollisionPermut | | 2 | 2 |  | hashed by parts | agrees statistically: [observations from per-thread pieces](#observations-from-per-thread-pieces) |
| `weight` | svaria_WeightDistrib | 1 | 4 | 6 |  |  | agrees statistically: [observations from per-thread pieces](#observations-from-per-thread-pieces) |
| `savir2` | smarsa_Savir2 | | 1 | 1 |  |  | agrees statistically: [observations from per-thread pieces](#observations-from-per-thread-pieces) |
| `gcd` | smarsa_GCD | | 2 | 1 |  |  | agrees statistically: [observations from per-thread pieces](#observations-from-per-thread-pieces) |
| `hamweight` | sstring_HammingWeight2 | | 2 | 2 |  |  | agrees statistically: [observations from per-thread pieces](#observations-from-per-thread-pieces), and [a variant of the statistic](#variants-of-the-statistic) |
| `hamcorr` | sstring_HammingCorr | | 3 | 3 |  |  | agrees statistically: [observations from per-thread pieces](#observations-from-per-thread-pieces) |
| `hamindep` | sstring_HammingIndep | 1 | 6 | 6 |  |  | agrees statistically: [observations from per-thread pieces](#observations-from-per-thread-pieces) |
| `bitrun` | sstring_Run | | 2 | 2 |  |  | same sequence, [corrected variance of the statistic](#corrected-variance-of-the-statistic) |
| `longhead` | sstring_LongestHeadRun | | 2 | 2 |  |  | agrees statistically: [observations from per-thread pieces](#observations-from-per-thread-pieces), and [a variant of the statistic](#variants-of-the-statistic) |
| `autocor` | sstring_AutoCor | | 4 | 4 |  |  | identical |
| `samplecorr` | svaria_SampleCorr | | 1 | 2 |  |  | same sequence, [corrected variance of the statistic](#corrected-variance-of-the-statistic) |
| `sampleprod` | svaria_SampleProd | | 2 | 3 | [AD statistic](#the-anderson-darling-statistics) | sorted by parts | agrees statistically: [observations from per-thread pieces](#observations-from-per-thread-pieces) |
| `samplemean` | svaria_SampleMean | | 1 | 2 | [AD statistic](#the-anderson-darling-statistics) | sorted by parts | agrees statistically: [observations from per-thread pieces](#observations-from-per-thread-pieces) |
| `walk` | swalk_RandomWalk1 (H, M, J, R, C) | 1 | 6 | 6 |  |  | agrees statistically: [observations from per-thread pieces](#observations-from-per-thread-pieces) |
| `periods` | sstring_PeriodsInStrings | | 2 | 2 |  |  | agrees statistically: [observations from per-thread pieces](#observations-from-per-thread-pieces) |
| `sumcoll` | svaria_SumCollector | | 1 | 1 |  |  | agrees statistically: [observations from per-thread pieces](#observations-from-per-thread-pieces) |
| `fourier` | sspectral_Fourier3 | | 2 | 2 |  | batched | [rounding difference](#rounding-the-fft-of-fourier3) of about 1e-9 (relative); the p-values are [not uniform, as in TestU01](#the-null-distribution-of-fourier3-as-in-testu01) |
| `linearcomp` | scomp_LinearComp (the number and the sizes of the jumps) | | 2 | 2 | [Berlekamp-Massey](#linearcomp) |  | identical statistics; [rounding difference](#rounding-the-p-value-of-the-upper-tail) of the p-values (1–2 ulp) |
| `matrixrank` | smarsa_MatrixRank | 1 | 6 | 6 |  | batched (larger than 32 x 32) | matrices larger than 32 x 32: identical statistic, [rounding difference](#rounding-the-p-value-of-the-upper-tail) of the p-value (1–2 ulp); 30 x 30 (BigCrush): agrees statistically, [observations from per-thread pieces](#observations-from-per-thread-pieces) |
| `appearance` | svaria_AppearanceSpacings | | 2 | 2 |  | sorted by parts | [rounding difference](#rounding-the-order-of-the-sum-of-appearancespacings) of about 1e-10 (relative) in the sum, about 2e-5 in z |
| `lempelziv` | scomp_LempelZiv | | 1 | 2 | [LZ78 parsing](#lempelziv) |  | identical |

"Identical" means the same statistic and p-value to all 17 significant digits. This was checked for every instance of Crush and BigCrush, except the large matrices of MatrixRank in BigCrush. The check ran TestU01 on the same outputs of a SplitMix64 with one round of its finalizer, from the positions where testu01_gpu starts each instance.

## How the tests are computed

### The sequence and the threads

Every test, and every replication of it, takes the next range of consecutive positions of the sequence: a single stream, as in TestU01. Within that range, every thread runs the test on its own consecutive piece. Pieces at positions far apart, e.g. differing only in the bits 28 and above, would compare outputs of related counters, which a single stream of TestU01 never does, and weak counter-based generators fail such comparisons in their low bits. The pieces are generous in length (12 standard deviations of the values used, and more for a few observations with heavy tails), so a thread rarely runs out of its piece; a thread that does makes the test fail completely (`TestFailure`).

Most tests make each observation from the consecutive positions of one thread (an observation never spans threads), and count the observations in histograms on GPU. Their statistics come from other outputs than those of TestU01, and agree with it only statistically.

The other tests see the single sequence as TestU01 does. SerialOver, Run, the runs of bits, AutoCor and SampleCorr do so with every thread starting a few values before its share. Fourier3, LinearComp, LempelZiv, AppearanceSpacings and MatrixRank of matrices larger than 32 x 32 read the consecutive outputs directly. Run and AutoCor take their replications one after the other, as TestU01 does.

### The statistics on the host

Every test finishes on the host: the counts made on GPU are copied back, and the statistic and its p-value are computed by the probdist library of TestU01. For the tests on the single sequence, these computations follow the floating-point operations of TestU01:

* SerialOver sums the terms of its chi-square in the order of the cells, as CalcPowDivEqual does: the precomputed terms (c − e)²/e when 6e ≤ k, and otherwise the sum of the (c − e)² divided by e at the end. It is then standardized by the corrected mean and standard deviation of Read and Cressie (smultin_PowDivMomCorChi).
* A statistic summed over N replications sorts the values first, as gofw_ActiveTests2 does in place, and takes N times their mean (statcoll_Average).
* MatrixRank computes its expected numbers in log2 as TestU01 does, and merges them by the operations of gofs_MergeClasses, adding the expected numbers of a merged class in the same order.

The few remaining differences are explained in [Differences from TestU01](#differences-from-testu01).

### Collisions and sorting

The collisions of CollisionOver, Collision, CollisionPermut and of the spacings of BirthdaySpacings are counted exactly by a hash table on GPU. The table is small enough for the cache of the GPU, and is filled in passes over the points. CollisionOver and Collision make their points again for every pass, which is cheaper than storing them. CollisionPermut stores its points once, as they are costly to make. As TestU01 does for these parameters, the counts are tested against a Poisson distribution (for CollisionOver and Collision: n > 1e5 and n / k ≤ 0.1). ClosePairsBitMatch finds the largest number of equal leading bits of two points by a binary search over the collisions of the keys of the points.

The points of BirthdaySpacings and AppearanceSpacings, and the values of the Anderson-Darling statistics, are sorted on GPU by rocThrust (or Thrust for CUDA).

### The memory budget

A test whose memory would grow with the samples keeps within `Options::memory`, so any scale runs in a fixed device memory with the same results:

* Hashed by parts (CollisionPermut, ClosePairsBitMatch): beyond the budget, the points are made again for every part of their hashes, so that equal points fall in the same part.
* Sorted by parts (BirthdaySpacings, AppearanceSpacings, and the AD statistics of MaxOft, SampleProd and SampleMean): beyond the budget, the points are made again for every part of their top bits, and the parts are taken in increasing order.
* By slabs (ClosePairs): the points are taken by slabs of their first coordinate, each with a margin of the searched distance below it.
* Batched (Fourier3, and MatrixRank of matrices larger than 32 x 32): the transforms and the matrices are made in batches that fit in the budget.

CollisionOver and Collision need no memory growing with the samples.

### Other tests on GPU

PeriodsInStrings classifies the strings of s bits by their correlations (their periods). For s ≤ 16 it takes a histogram of the strings. For larger s, it builds the classes of the correlations as CreateCorr of TestU01 does (after Guibas and Odlyzko), and finds each string in them by a binary search on GPU.

MatrixRank computes the ranks over GF(2) on GPU. A matrix of at most 32 x 32 is made per thread, from the consecutive outputs of the thread. A larger one is made in global memory from the consecutive outputs of the sequence, one block each.

AppearanceSpacings sorts the keys (block, index), so that the occurrences of every block are in order, and sums the logarithms of their distances on GPU. For s > L, an output gives s / L blocks, the low bits first, as in TestU01.

Fourier3 transforms the sequences of bits by rocFFT (or cuFFT for CUDA) in double precision. In single precision, its statistic differed from that of TestU01 on the same outputs in the fifth digit.

The scale multiplies n, the number of matrices of MatrixRank, and K of AppearanceSpacings. LempelZiv is the exception: the mean and the standard deviation of TestU01 are tabulated only up to 2^28 bits, so its scale multiplies the number of sequences N instead of their length.

## Tests using the CPU

Some tests do a part of their work on the host: an algorithm that is sequential by nature, or a step on data too small to be worth the GPU. The GPU makes the outputs, or the values derived from them, and the host takes them from there.

### LinearComp

The GPU makes the bits of the sequences, packed in 64-bit words, and the host runs the Berlekamp-Massey algorithm on them. The algorithm is sequential in the bits, and its time grows as the square of their number, so it is the slowest part of the large scales. Its loops on the words are written for the compiler to vectorize: `TESTU01_GPU_HOST_FLAGS` (e.g. `-march=native`) lets it use the SIMD of the host. The host then counts the jumps of the linear complexity and their sizes. TestU01 also computes a chi-square of the linear complexity, but only for more than 20 sequences, which no battery uses, so it is not computed.

### LempelZiv

The GPU makes the outputs of each sequence, and the host parses its bits by LZ78, counting the words of the parsing, with the trie in arrays. The parsing is sequential: each word depends on all the earlier ones.

### ClosePairs

The GPU finds the close pairs of the points in the torus (with the maximum norm) in a grid of cells. The host takes their distances and computes the six statistics of TestU01: NP, mNP, mNP1, mNP2, NJumps and mNP2S. As in bbattery.c, Crush reports no mNP2S for t = 2. Where TestU01 has no value for mNP2 (no jumps, which it reports as −1), that statistic is not reported.

### BirthdaySpacings

Within the budget, the GPU sorts the points and counts the collisions of their spacings by its hash table. Beyond the budget, the points are sorted by parts of their top bits on GPU, and the host takes over the counting. The spacings, hashed, go to buckets on the host by the top bits of their hashes, and their collisions are counted one bucket at a time.

### The Anderson-Darling statistics

MaxOft (its AD statistic), SampleProd and SampleMean transform their values to U(0, 1) on GPU and sort them there. The host collects the sorted values and computes the statistic by gofs_AndersonDarling of TestU01, in its order of the terms. With `Options::fast`, the terms are summed on GPU instead, part by part, which is faster but sums them in another order (see [Usage](#usage)).

## Differences from TestU01

### Observations from per-thread pieces

Every thread makes the observations from its own piece of the sequence (see [The sequence and the threads](#the-sequence-and-the-threads)), so they come from other outputs than those of TestU01. The p-values agree with those of TestU01 only statistically. For a generator with weak low bits, the verdicts of a few tests can differ (see [Weak generators](#weak-generators)).

### Rounding: the p-value of the upper tail

For N = 1, TestU01 takes the p-value as 1 − F(x) (gofw_ActiveTests0), where F is the distribution function. testu01_gpu takes the upper tail F̄(x) (fbar_ChiSquare2, fbar_Normal1) directly. The two differ by one or two units in the last place: for example, 0.43621255682282989 against 0.43621255682282994. Computing 1 − F from the statistic of testu01_gpu gives the p-value of TestU01 exactly. The upper tail is kept because it stays precise below 1e-16, where 1 − F becomes 0.

### Corrected variance of the statistic

Where TestU01 takes a statistic of a wrong variance, the right one is taken, on the same sequence as TestU01. For the number of runs of B bits of sstring_Run, the variance is B / 16. For SampleCorr, it is 13/144 per term. The runs of bits are also counted in a fixed number of values, instead of until n runs.

### Variants of the statistic

Besides their other outputs, two statistics are computed in a variant of those of TestU01. HammingWeight2 counts the weights within 12 standard deviations of the mean, and the farther ones at the ends. LongestHeadRun starts every block afresh.

### Rounding: the FFT of Fourier3

TestU01 transforms each sequence with the split-radix real FFT of Sorensen (rsrfft in fftc.c). testu01_gpu uses rocFFT or cuFFT in double precision. The two algorithms compute each coefficient by different sequences of floating-point operations, so the squares |F_i|² differ in their last bits. The AD statistic then differed by about 1e-9 relative: for example, p = 0.228234762973 against 0.228234762203.

### Rounding: the order of the sum of AppearanceSpacings

TestU01 adds the K logarithms (K = 1e8 to 1e9) one after the other to a running sum. testu01_gpu adds them by a parallel reduction on GPU, in a tree order. The two sums differ by about 1e-10 relative. Since the standard deviation of the mean is small (about 6e-5 for K = 4e8), the normal statistic then differs by about 2e-5: z = −1.23425663 against −1.23428288. On the CPU, the same sum done in order reproduces the statistic of TestU01 exactly, and a pairwise sum reproduces that of testu01_gpu to 17 digits. A sum in long double gives −1.23428288, so the sum in order of TestU01 is the one off, by the rounding errors accumulated at the scale of a running sum of about 4e9.

### The null distribution of Fourier3 (as in TestU01)

The p-values of Fourier3 are not uniform for good generators, with too few small p-values, in TestU01 itself as in testu01_gpu. Over 40 seeds, Fourier3 of Crush gave a KS p-value of 0.001 by testu01_gpu (see [The null distributions](#the-null-distributions)). TestU01 on the CPU, on a SplitMix64 with one round of its finalizer and 40 seeds, gives a KS p-value of 0.007, with 3 of 80 p-values below 0.1. This comes from the approximation of the test, not from the port: Fourier3 tests its n/4 standardized sums B_i as n/4 independent normal variables, but their mean varies only as much as that of n/8 independent ones.

Fourier3 transforms a sequence a_0, …, a_{n−1} of ±1 (its bits) to the frequency domain F_0, …, F_{n−1} by the unnormalized DFT:

```math
F_k = \sum_j a_j ω^{−jk},\quad\text{with } ω = e^{2πi/n} .
```

By Parseval's theorem, the total of the frequency domain is the same for every sequence, as

```math
\sum_{k=0}^{n−1} |F_k|^2 = n \sum_j a_j^2 = n^2 .
```

Since a_0, …, a_{n−1} is a real sequence, |F_{n−k}| = |F_k| holds. so

```math
n^2 = F_0^2 + F_{n/2}² + 2 \sum_{k=1}^{n/2−1} |F_k|^2 .
```

The n/2 − 1 free coefficients thus have a nearly fixed total, and so do the B_i of all of them, summed over the N sequences.

Fourier3 takes half of the free coefficients, k = 1, …, n/4. For M exchangeable variables of variance σ² with a fixed total, every pair has the covariance −σ²/(M − 1), since the variance of the total is 0. A sum of m of them then has the variance m σ² (M − m)/(M − 1) instead of m σ², which is half of it for m = M/2: the same as m/2 independent variables. The fluctuation of the mean of the n/4 values B_i is that of n/8 independent ones. The AD statistic, comparing the B_i with n/4 independent normal variables, comes out small too often, and the small p-values are too few.

A simulation of Fourier3 on random bits confirms it, with n = 2^14, N = 50 and 4000 runs:

| quantity | measured | for independent B_i |
|---|---|---|
| variance of a single B_i | 1.0000 | 1 (the standardization holds) |
| variance of the mean of the B_i for k = 1, …, n/4, times n/4 | 0.48 | 1 |
| the same for all the free k = 1, …, n/2 − 1, times n/2 − 1 | 0.0001 | 1 (the nearly fixed total) |

### Values, classes and limits

As in TestU01, a value in [0, d) is floor(d u) of u = (x << r) / 2^32 for the output x: the r most significant bits are dropped. The limits of TestU01 that come from its 32-bit words are lifted where they are not needed, e.g. d up to 64 for SimpPoker and CouponCollector, and s up to 32 for PeriodsInStrings and AutoCor. Parameters beyond the limits of the implementation are rejected (`std::invalid_argument`) rather than computed wrongly. The chi-squares merge the classes with an expected count below 10 by the rule of gofs_MergeClasses: from the low end until 10, with a remainder at the high end joining the last class.

## Validation against TestU01

### Same outputs

The tests that see the single sequence were run by TestU01 (on the CPU) and by testu01_gpu on the same outputs. The generator was a SplitMix64 with one round of its finalizer, and each instance of Crush and BigCrush started from the position testu01_gpu gives it. The results are those in the last column of the table above.

### Weak generators

SmallCrush and Crush were run by both on three generators:
* a 32-bit LCG (69069 x + 1 mod 2^32);
* the upper 32 bits of a 64-bit LCG (with the constants of PCG);
* the SplitMix64 above.

With a failure defined as a p-value outside [1e-10, 1 − 1e-10], the results agree:

| generator | SmallCrush | Crush |
|---|---|---|
| SplitMix64, one round | no failures on either | no failures on either |
| 64-bit LCG, upper 32 bits | no failures on either | the same 3 failures |
| 32-bit LCG | the same 13 of 15 statistics fail | the same 105 statistics fail |

For the 32-bit LCG, the verdicts differ in a few statistics on its low bits:

| test | TestU01 | testu01_gpu |
|---|---|---|
| PeriodsInStrings, r = 0 (Crush) | 0.82 | 1e-141 |
| RandomWalk1 J (SmallCrush) | 0.37 | 0 |
| Gap, r = 0, β = 2^-8 (Crush) | 0.63 | 2e-10 |
| LongestHeadRun, r = 0 (Crush) | 0.99 | 7e-5 |
| LempelZiv (Crush) | 1e-6 | 7e-12 |
| CollisionOver, t = 20, r = 0 (Crush) | 1 − 9e-10 | 1 |

Bit k of this LCG has a period of 2^(k + 1). The single stream of TestU01 (for PeriodsInStrings, 3e8 consecutive outputs) passes through many whole periods of the bits up to about 27. The pieces of the threads are shorter: for PeriodsInStrings, about 1150 consecutive outputs at a stride of 1152 = 2^7 · 9. A piece sees only a part of the periods of the bits 10 and above, and the phases of the pieces don't cancel, so those bits are unbalanced. The last two rows are failing or suspect on both sides.

### The null distributions

The tests on per-thread pieces can't be compared with TestU01 value by value. An error in their null distributions, e.g. a wrong expected count, a wrong merging of classes or a piece cut short, would still show as p-values that are not uniform for a good generator. So the p-values of testu01_gpu were checked for uniformity on SplitMix64 with 40 seeds. The check covered SmallCrush and Crush, and MaxOft, Run, AutoCor, SerialOver and MatrixRank of BigCrush.

Leaving out the discrete statistics, each of the 150 statistics was tested by a Kolmogorov-Smirnov test, and none had a KS p-value below 0.001. Pooled per test and battery, the KS p-values are above 0.02, except for Fourier3 of Crush (0.001, with too few small p-values). TestU01 shows the same for Fourier3 on the CPU: see [The null distribution of Fourier3](#the-null-distribution-of-fourier3-as-in-testu01).

## License

Apache License 2.0 ([LICENSE](LICENSE)). The tests are reimplemented after the source code of TestU01, and the library links its probdist and mylib libraries, so it is also subject to the license of TestU01 ([LICENSE.TestU01](LICENSE.TestU01)), which permits only non-commercial use; see [NOTICE](NOTICE).
