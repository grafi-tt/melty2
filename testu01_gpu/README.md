# testu01_gpu

A library running the tests of [TestU01](http://simul.iro.umontreal.ca/testu01/tu01.html) on a GPU, by HIP (ROCm) or CUDA. The tests are reimplemented after the source code of TestU01 1.2.3: the test functions used by its batteries SmallCrush, Crush and BigCrush (33 of them, see [Tests](#tests)), with the parameters of their instances in these batteries and sample sizes that can be scaled.

The tests of BigCrush take tens of seconds on a Radeon RX 6800 instead of hours on a CPU core, those of Crush about ten seconds, and those of SmallCrush less than one. Most tests make their observations from other outputs than TestU01 does (see [Tests](#tests)), so their p-values agree with those of TestU01 statistically; the tests on the single sequence give the same statistics as TestU01 on the same outputs, or statistics differing only by the order of their floating-point operations (Fourier3, AppearanceSpacings). They screen generators quickly, and sample sizes far beyond those of BigCrush are practical.

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

or the source tree by `add_subdirectory(testu01_gpu)` and the same target. `TESTU01_GPU_BACKEND` chooses HIP or CUDA (by default HIP if CMake finds a HIP compiler, else CUDA); the installed package records it (`testu01_gpu_BACKEND`). The kernels of the tests are compiled at run time for the current device (by hiprtc or NVRTC), but those of rocThrust or Thrust (the sorts and prefix sums) are compiled by the build: for HIP, for `CMAKE_HIP_ARCHITECTURES` (default gfx1030), which must include the GPU used; for CUDA, for `CMAKE_CUDA_ARCHITECTURES` (default `native`, the GPUs of the build machine; a GPU newer than these runs their PTX, if any, compiled by the driver). `TESTU01_GPU_HOST_FLAGS` adds flags to the host code only (a list): the default is portable, and `-march=native` (passed to the host compiler by `-Xarch_host` for HIP and `-Xcompiler` for CUDA) lets the compiler vectorize the Berlekamp-Massey algorithm of LinearComp with the SIMD of the host (AVX-512 halved its time on a Ryzen 9 9950X).

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
* `Options::memory`: the device memory of the large tests (2 GiB by default). The tests whose memory grows with the samples (CollisionOver, BirthdaySpacings, CollisionPermut, ClosePairs, SampleProd, SampleMean) make their points again by parts beyond it (see [Collisions, sorting and the memory budget](#collisions-sorting-and-the-memory-budget)), so any scale runs in a fixed memory; the results are the same for any budget.
* `Options::fast`: faster computations whose results differ from those of TestU01 by the order of the floating-point sums (off by default): the Anderson-Darling statistics of SampleProd and SampleMean are summed on GPU by a parallel reduction instead of in order by `gofs_AndersonDarling` on the host.
* `Options::report`: called for every statistic with a `Result` (the name, the parameters, the kind of statistic, its value, the p-value and a z-score); `format` gives the line printed above, marking a p-value outside [0.001, 0.999] with `*****` as TestU01 does.

`run` throws `testu01_gpu::TestFailure` for a test failing completely (e.g. the observations of a test taking far more outputs than expected, for a very bad generator), `std::invalid_argument` for parameters outside the range a test supports (e.g. d > 64 of SimpPoker and CouponCollector, s > 32 or r + s > 32 of the tests taking s bits of an output), and `std::runtime_error` for the errors of HIP and hiprtc (or CUDA and NVRTC).

[examples/splitmix_bigcrush.cpp](examples/splitmix_bigcrush.cpp) tests SplitMix64 as a counter-based generator: `build/splitmix_bigcrush [seed] [scale] [tests|all] [fast] [small|crush|big]`. [../search/testu01_stest.hip](../search/testu01_stest.hip) tests the ARX structures of melty2, with the streams of counters or keys interleaved by the parameters.

## Tests

Only the test functions used by SmallCrush, Crush and BigCrush are ported, with the parameters of their instances in these batteries. Porting the rest of TestU01 is not planned:
* the other test functions, e.g. sknuth_Serial, sknuth_SerialSparse, sknuth_RunIndep, smarsa_Opso, smarsa_CAT, snpair_BickelBreiman, sspectral_Fourier1 and Fourier2, sstring_HammingWeight, svaria_SumLogs, svaria_CollisionArgMax, swalk_RandomWalk1a, swalk_VarGeoN and VarGeoP, and the modules sentrop and sspacings;
* the other batteries: Rabbit, Alphabit, BlockAlphabit, FIPS_140_2 and pseudoDIEHARD;
* the generators of TestU01 (its modules u*).

The tests and the instances of the batteries (the numbers of instances below: the calls of the test functions in `bbattery.c` of TestU01 1.2.3, for a 64-bit `long`) are those of SmallCrush (10, with 15 statistics), Crush (96, 144 statistics) and BigCrush (106, 160 statistics). The last column tells how a statistic compares with that of TestU01 on the same outputs of the generator.

| name | test of TestU01 | SmallCrush | Crush | BigCrush | the same outputs as TestU01: the statistic and the p-value |
|---|---|---|---|---|---|
| `poker` | sknuth_SimpPoker | 1 | 4 | 4 | other outputs [^pieces] |
| `coupon` | sknuth_CouponCollector | 1 | 4 | 4 | other outputs |
| `gap` | sknuth_Gap | 1 | 4 | 4 | other outputs |
| `maxoft` | sknuth_MaxOft (the chi-square and the AD statistic) | 1 | 4 | 4 | other outputs |
| `collision` | smarsa_CollisionOver | | 8 | 10 | other outputs |
| `kcollision` | sknuth_Collision | 1 | | | other outputs |
| `birthday` | smarsa_BirthdaySpacings | 1 | 7 | 9 | other outputs |
| `closepairs` | snpair_ClosePairs (NP, mNP, mNP1, mNP2, NJumps, mNP2S) | | 3 | 4 | other outputs |
| `cpbitmatch` | snpair_ClosePairsBitMatch | | 2 | | other outputs |
| `permutation` | sknuth_Permutation | | 2 | 4 | other outputs |
| `run` | sknuth_Run | | 2 | 2 | identical |
| `serial` | smarsa_SerialOver | | 2 | 2 | identical statistic; the p-value differs in the last bits [^tail] |
| `collpermut` | sknuth_CollisionPermut | | 2 | 2 | other outputs |
| `weight` | svaria_WeightDistrib | 1 | 4 | 6 | other outputs |
| `savir2` | smarsa_Savir2 | | 1 | 1 | other outputs |
| `gcd` | smarsa_GCD | | 2 | 1 | other outputs |
| `hamweight` | sstring_HammingWeight2 | | 2 | 2 | other outputs |
| `hamcorr` | sstring_HammingCorr | | 3 | 3 | other outputs |
| `hamindep` | sstring_HammingIndep | 1 | 6 | 6 | other outputs |
| `bitrun` | sstring_Run | | 2 | 2 | a corrected statistic [^corrected] |
| `longhead` | sstring_LongestHeadRun | | 2 | 2 | other outputs |
| `autocor` | sstring_AutoCor | | 4 | 4 | identical |
| `samplecorr` | svaria_SampleCorr | | 1 | 2 | a corrected statistic [^corrected] |
| `sampleprod` | svaria_SampleProd | | 2 | 3 | other outputs |
| `samplemean` | svaria_SampleMean | | 1 | 2 | other outputs |
| `walk` | swalk_RandomWalk1 (H, M, J, R, C) | 1 | 6 | 6 | other outputs |
| `periods` | sstring_PeriodsInStrings | | 2 | 2 | other outputs |
| `sumcoll` | svaria_SumCollector | | 1 | 1 | other outputs |
| `fourier` | sspectral_Fourier3 | | 2 | 2 | differs by about 1e-9 (relative): another FFT algorithm [^fft]; the p-values are not uniform, as in TestU01 (see [below](#the-null-distribution-of-fourier3)) |
| `linearcomp` | scomp_LinearComp (the number and the sizes of the jumps) | | 2 | 2 | identical statistics; the p-values differ in the last bits [^tail] |
| `matrixrank` | smarsa_MatrixRank | 1 | 6 | 6 | matrices larger than 32 x 32: identical statistic, the p-value differs in the last bits [^tail]; 30 x 30 (BigCrush): other outputs |
| `appearance` | svaria_AppearanceSpacings | | 2 | 2 | differs by about 1e-10 (relative) in the sum of the logarithms: another order of the floating-point sum [^sum] |
| `lempelziv` | scomp_LempelZiv | | 1 | 2 | identical |

"Identical" means the same statistic and p-value to all 17 significant digits. This was checked for every instance of Crush and BigCrush, except the large matrices of MatrixRank in BigCrush. The check ran TestU01 on the same outputs of a SplitMix64 with one round of its finalizer, from the positions where testu01_gpu starts each instance.

[^pieces]: Every thread makes the observations from its own piece of the sequence (see [The sequence and the threads](#the-sequence-and-the-threads)), so they come from other outputs than those of TestU01. The p-values agree with those of TestU01 only statistically.

[^tail]: For N = 1, TestU01 takes the p-value as 1 − F(x) (gofw_ActiveTests0), where F is the distribution function. testu01_gpu takes the upper tail F̄(x) (fbar_ChiSquare2, fbar_Normal1) directly. The two differ by one or two units in the last place: for example, 0.43621255682282989 against 0.43621255682282994. Computing 1 − F from the statistic of testu01_gpu gives the p-value of TestU01 exactly. The upper tail is kept because it stays precise below 1e-16, where 1 − F becomes 0.

[^corrected]: Same sequence as TestU01, but a statistic of the right variance (see [Differences from TestU01](#differences-from-testu01)).

[^fft]: TestU01 transforms each sequence with the split-radix real FFT of Sorensen (rsrfft in fftc.c). testu01_gpu uses rocFFT or cuFFT in double precision. The two algorithms compute each coefficient by different sequences of floating-point operations, so the squares |F_i|² differ in their last bits. The AD statistic then differed by about 1e-9 relative: for example, p = 0.228234762973 against 0.228234762203.

[^sum]: TestU01 adds the K logarithms (K = 1e8 to 1e9) one after the other to a running sum. testu01_gpu adds them by a parallel reduction on GPU, in a tree order. The two sums differ by about 1e-10 relative. Since the standard deviation of the mean is small (about 6e-5 for K = 4e8), the normal statistic then differs by about 2e-5: z = −1.23425663 against −1.23428288. On the CPU, the same sum done in order reproduces the statistic of TestU01 exactly, and a pairwise sum reproduces that of testu01_gpu to 17 digits. A sum in long double gives −1.23428288, so the sum in order of TestU01 is the one off, by the rounding errors accumulated at the scale of a running sum of about 4e9.

### The null distribution of Fourier3

The p-values of Fourier3 are not uniform for good generators, with too few small p-values, in TestU01 itself as in testu01_gpu. Over 40 seeds, Fourier3 of Crush gave a KS p-value of 0.001 by testu01_gpu (see [Good generators](#good-generators)). TestU01 on the CPU, on a SplitMix64 with one round of its finalizer and 40 seeds, gives a KS p-value of 0.007, with 3 of 80 p-values below 0.1. This comes from the approximation of the test, not from the port: Fourier3 tests its n/4 standardized sums B_i as n/4 independent normal variables, but their mean varies only as much as that of n/8 independent ones.

The total of the |F_k|² is the same for every sequence. Fourier3 transforms a sequence a_0, …, a_{n−1} of ±1 (its bits) by the unnormalized DFT F_k = Σ_j a_j ω^{−jk}, with ω = e^{2πi/n}. By Parseval's theorem, Σ_{k=0}^{n−1} |F_k|² = n Σ_j a_j² = n². To see it, expand |F_k|² = Σ_j Σ_l a_j a_l ω^{(j−l)k} and sum over k. The sum Σ_k ω^{(j−l)k} is n for j = l and 0 otherwise, which leaves n Σ_j a_j², and a_j² = 1 for every j. For a real sequence |F_{n−k}| = |F_k|, so n² = F_0² + F_{n/2}² + 2 Σ_{k=1}^{n/2−1} |F_k|². The n/2 − 1 free coefficients thus have a nearly fixed total, and so do the B_i of all of them, summed over the N sequences.

Fourier3 takes half of the free coefficients, k = 1, …, n/4. For M exchangeable variables of variance σ² with a fixed total, every pair has the covariance −σ²/(M − 1), since the variance of the total is 0. A sum of m of them then has the variance m σ² (M − m)/(M − 1) instead of m σ², which is half of it for m = M/2: the same as m/2 independent variables. The fluctuation of the mean of the n/4 values B_i is that of n/8 independent ones. The AD statistic, comparing the B_i with n/4 independent normal variables, comes out small too often, and the small p-values are too few.

A simulation of Fourier3 on random bits confirms it, with n = 2^14, N = 50 and 4000 runs:

| quantity | measured | for independent B_i |
|---|---|---|
| variance of a single B_i | 1.0000 | 1 (the standardization holds) |
| variance of the mean of the B_i for k = 1, …, n/4, times n/4 | 0.48 | 1 |
| the same for all the free k = 1, …, n/2 − 1, times n/2 − 1 | 0.0001 | 1 (the nearly fixed total) |

## How the tests are computed

### The sequence and the threads

Every test, and every replication of it, takes the next range of consecutive positions of the sequence: a single stream, as in TestU01. Within that range, every thread runs the test on its own consecutive piece. Pieces at positions far apart, e.g. differing only in the bits 28 and above, would compare outputs of related counters, which a single stream of TestU01 never does, and weak counter-based generators fail such comparisons in their low bits. The pieces are generous in length (12 standard deviations of the values used, and more for a few observations with heavy tails), so a thread rarely runs out of its piece; a thread that does makes the test fail completely (`TestFailure`).

Most tests make each observation from the consecutive positions of one thread (an observation never spans threads) and count the observations in histograms. Their statistics therefore come from other outputs than those of TestU01, and agree with it only statistically.

### Tests on the single sequence

Other tests see the single sequence exactly:

* SerialOver, Run, the runs of bits, AutoCor and SampleCorr, where every thread starts a few values before its share.
* Fourier3, LinearComp, LempelZiv, AppearanceSpacings and MatrixRank of matrices larger than 32 x 32, which read the consecutive outputs directly.

Run and AutoCor take their replications one after the other, as TestU01 does.

On these outputs, the computations on the host follow the floating-point operations of TestU01:

* SerialOver sums the terms of its chi-square in the order of the cells, as CalcPowDivEqual does: the precomputed terms (c − e)²/e when 6e ≤ k, and otherwise the sum of the (c − e)² divided by e at the end. It is then standardized by the corrected mean and standard deviation of Read and Cressie (smultin_PowDivMomCorChi).
* A statistic summed over N replications sorts the values first, as gofw_ActiveTests2 does in place, and takes N times their mean (statcoll_Average).
* MatrixRank computes its expected numbers in log2 and merges its classes as gofs_MergeClasses does.

The remaining differences are those of the table above: the p-value of the upper tail, the FFT algorithm of Fourier3, and the order of the sum of AppearanceSpacings.

### Collisions, sorting and the memory budget

The collisions of CollisionOver, Collision, CollisionPermut and of the spacings of BirthdaySpacings are counted exactly by a hash table on GPU. The table is small enough for the cache of the GPU and is filled in passes over the points: those of CollisionOver and Collision are made again for every pass, those of CollisionPermut are stored once. As TestU01 does for these parameters, the counts are tested against a Poisson distribution (for CollisionOver and Collision: n > 1e5 and n / k ≤ 0.1). The points of BirthdaySpacings, SampleProd and SampleMean are sorted on GPU by rocThrust (or Thrust for CUDA).

The tests whose memory grows with the samples make their points again by parts beyond the budget (`Options::memory`), so any scale runs in a fixed memory with the same results. Each test splits its points in its own way:

* CollisionPermut splits by the hashes of the points, so that equal points fall in the same part.
* The sorted points are taken by the parts of their top bits, in increasing order. The spacings of BirthdaySpacings then go to buckets on the host by their hashes, and their collisions are counted bucket by bucket.
* ClosePairs takes its points by slabs of the first coordinate, each with a margin of the searched distance below it.
* The Fourier transforms are batched within the budget.

CollisionOver needs no memory that grows with the samples.

### Notes on single tests

ClosePairs finds the close pairs of the points in the torus (with the maximum norm) in a grid of cells on GPU, and computes the six statistics of TestU01 on the host. As in bbattery.c, Crush reports no mNP2S for t = 2. Where TestU01 has no value for mNP2 (no jumps, which it reports as −1), the statistic is not reported. ClosePairsBitMatch finds the largest number of equal leading bits of two points by a binary search over the collisions of the keys of the points on GPU.

PeriodsInStrings classifies the strings of s bits by their correlations (their periods). For s ≤ 16 it takes a histogram of the strings. For larger s, it builds the classes of the correlations as CreateCorr of TestU01 does (after Guibas and Odlyzko), and finds each string in them by a binary search on GPU.

LinearComp runs the Berlekamp-Massey algorithm on the host, on the bits made on GPU, with 64-bit words in loops vectorized by the compiler. Its time grows as the square of the bits: 0.4 s at the scale 1 of BigCrush, and about 53 s at 16 with `-march=native` (`TESTU01_GPU_HOST_FLAGS`) on a Ryzen 9 9950X. TestU01 computes the chi-square of the linear complexity only for more than 20 sequences, which no battery uses, so it is not computed. MatrixRank computes the ranks over GF(2) on GPU. A matrix of at most 32 x 32 is made per thread, from the consecutive outputs of the thread. The larger ones are made in global memory from the consecutive outputs of the sequence, one block each, batched within the memory budget.

AppearanceSpacings sorts the keys (block, index) on GPU, by parts of the blocks beyond the memory budget, so that the occurrences of every block are in order, and sums the logarithms of their distances on GPU. For s > L, an output gives s / L blocks, the low bits first, as in TestU01. LempelZiv parses its sequences by LZ78 on the host, on the outputs made on GPU, with the trie in arrays. Fourier3 transforms the sequences of bits by rocFFT (or cuFFT for CUDA) in double precision, in batches of up to 4096. In single precision, its statistic differed from that of TestU01 on the same outputs in the fifth digit.

The scale multiplies n, the number of matrices of MatrixRank, and K of AppearanceSpacings. LempelZiv is the exception: the mean and the standard deviation of TestU01 are tabulated only up to 2^28 bits, so its scale multiplies the number of sequences N instead of their length.

## Differences from TestU01

Where TestU01 takes a statistic of a wrong variance, the right one is taken: B / 16 for the number of runs of B bits of sstring_Run, and 13/144 per term for SampleCorr. The runs of bits are also counted in a fixed number of values, instead of until n runs. HammingWeight2 counts the weights within 12 standard deviations of the mean, and the farther ones at the ends. LongestHeadRun starts every block afresh. Except in MatrixRank, the chi-squares merge the classes with an expected count below 10 from the low end, while gofs_MergeClasses of TestU01 merges from both ends.

As in TestU01, a value in [0, d) is floor(d u) of u = (x << r) / 2^32 for the output x: the r most significant bits are dropped. The limits of TestU01 that come from its 32-bit words are lifted where they are not needed, e.g. d up to 64 for SimpPoker and CouponCollector, and s up to 32 for PeriodsInStrings and AutoCor. Parameters beyond the limits of the implementation are rejected (`std::invalid_argument`) rather than computed wrongly.

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

### Good generators

The p-values of SmallCrush and Crush, and of MaxOft, Run, AutoCor, SerialOver and MatrixRank of BigCrush, were taken for 40 seeds of SplitMix64. Leaving out the discrete statistics, each of the 150 statistics was tested for uniformity by a Kolmogorov-Smirnov test, and none had a KS p-value below 0.001. Pooled per test and battery, the KS p-values are above 0.02, except for Fourier3 of Crush: its 80 p-values give a KS p-value of 0.001, with too few small p-values (none below 0.1) for their AD statistics.

TestU01 itself shows the same on the CPU (see [The null distribution of Fourier3](#the-null-distribution-of-fourier3)).

## License

Apache License 2.0 ([LICENSE](LICENSE)). The tests are reimplemented after the source code of TestU01, and the library links its probdist and mylib libraries, so it is also subject to the license of TestU01 ([LICENSE.TestU01](LICENSE.TestU01)), which permits only non-commercial use; see [NOTICE](NOTICE).
