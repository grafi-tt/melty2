# testu01_gpu

A library running the tests of the BigCrush battery of [TestU01](http://simul.iro.umontreal.ca/testu01/tu01.html) on a GPU, by HIP (ROCm) or CUDA, for counter-based generators: a generator whose output at any position of its sequence can be computed directly. The tests are reimplemented after the source code of TestU01 1.2.3 with the parameters of their instances in BigCrush, and take tens of seconds of a Radeon RX 6800 instead of hours of a CPU core. Their p-values agree with those of TestU01 statistically (the outputs used differ), so they screen generators quickly, and sample sizes far beyond BigCrush are practical; the original BigCrush still judges the final generators.

All the tests of BigCrush are implemented.

## Build

Requirements: ROCm (HIP, hiprtc, rocFFT and rocThrust) or the CUDA Toolkit (NVRTC, cuFFT and Thrust; a GPU of the compute capability 6.0 or later), and TestU01 installed (its headers and the `probdist` and `mylib` libraries, for the distributions and the goodness-of-fit tests). The host code is C++20 (the header needs only C++17 of its users) and uses [abseil](https://abseil.io/) (`find_package(absl)`, an LTS release from 20230125: its strings in the library, its logging in the example), which is built in C++20 on current systems; the package of Ubuntu 24.04 (20220623) is too old, so the CI builds abseil 20260107.1 from its source.

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

* `Options::tests`: the comma-separated names of the tests to run (`testu01_gpu::kAllTests` by default, in the order below).
* `Options::scale`: multiplies the sample sizes `n` of BigCrush (e.g. 16 for 16 times the samples).
* `Options::first`: the first position used. Every test and replication takes the next range of consecutive positions, as TestU01 takes the outputs of a single stream in turn (so the positions of a test depend on the tests run before it).
* `Options::limit`: stops the tests once a z-score is above it, for searches rejecting such a generator anyway.
* `Options::memory`: the device memory of the large tests (2 GiB by default). The tests whose memory grows with the samples (CollisionOver, BirthdaySpacings, CollisionPermut, ClosePairs, SampleProd, SampleMean) make their points again by parts beyond it (see below), so any scale runs in a fixed memory; the results are the same for any budget.
* `Options::fast`: faster computations whose results may differ from those of TestU01 in rounding (off by default): the Anderson-Darling statistics of SampleProd and SampleMean are summed on GPU instead of by `gofs_AndersonDarling` on the host.
* `Options::report`: called for every statistic with a `Result` (the name, the parameters, the kind of statistic, its value, the p-value and a z-score); `format` gives the line printed above, marking a p-value outside [0.001, 0.999] with `*****` as TestU01 does.

`run` throws `testu01_gpu::TestFailure` for a test failing completely (e.g. the observations of a test taking far more outputs than expected, for a very bad generator), and `std::runtime_error` for the errors of HIP and hiprtc (or CUDA and NVRTC).

[examples/splitmix_bigcrush.cpp](examples/splitmix_bigcrush.cpp) tests SplitMix64 as a counter-based generator: `build/splitmix_bigcrush [seed] [scale] [tests]`. [../search/testu01_stest.hip](../search/testu01_stest.hip) tests the ARX structures of melty2, with the streams of counters or keys interleaved by the parameters.

## Tests

| name | test of TestU01 | instances of BigCrush |
|---|---|---|
| `poker` | sknuth_SimpPoker | r = 0, 27 (d = k = 8); r = 0, 25 (d = k = 32) |
| `coupon` | sknuth_CouponCollector | r = 0, 10, 20, 27 (d = 8) |
| `gap` | sknuth_Gap | 4 instances |
| `maxoft` | sknuth_MaxOft | t = 8, 16, 24, 32 |
| `collision` | smarsa_CollisionOver | 10 instances |
| `birthday` | smarsa_BirthdaySpacings | 9 instances |
| `closepairs` | snpair_ClosePairs | t = 3, 5, 9, 16 (NP, mNP, mNP1, mNP2, NJumps, mNP2S) |
| `permutation` | sknuth_Permutation | t = 3, 5, 7, 10 |
| `run` | sknuth_Run | up and down |
| `serial` | smarsa_SerialOver | r = 0, 22 |
| `collpermut` | sknuth_CollisionPermut | r = 0, 10 |
| `weight` | svaria_WeightDistrib | 6 instances |
| `savir2` | smarsa_Savir2 | 1 instance |
| `gcd` | smarsa_GCD | 1 instance |
| `hamweight` | sstring_HammingWeight2 | r = 0, 27 |
| `hamcorr` | sstring_HammingCorr | L = 30, 300, 1200 |
| `hamindep` | sstring_HammingIndep | 6 instances |
| `bitrun` | sstring_Run | r = 0, 27 |
| `longhead` | sstring_LongestHeadRun | r = 0, 27 |
| `autocor` | sstring_AutoCor | 4 instances |
| `samplecorr` | svaria_SampleCorr | k = 1, 2 |
| `sampleprod` | svaria_SampleProd | t = 8, 16, 24 |
| `samplemean` | svaria_SampleMean | r = 0, 10 |
| `walk` | swalk_RandomWalk1 | 6 instances (H, M, J, R, C) |
| `periods` | sstring_PeriodsInStrings | r = 0, 20 |
| `sumcoll` | svaria_SumCollector | 1 instance |
| `fourier` | sspectral_Fourier3 | r = 0, 27 |
| `linearcomp` | scomp_LinearComp | r = 0, 29 (the number and the sizes of the jumps) |
| `matrixrank` | smarsa_MatrixRank | 30 x 30 (r = 0, 25), 1000 x 1000 (r = 0, 26), 5000 x 5000 (r = 15, 0) |
| `appearance` | svaria_AppearanceSpacings | r = 0, 27 |
| `lempelziv` | scomp_LempelZiv | r = 0 (s = 30), r = 15 (s = 15) |

How they are computed on GPU:

* Every test (and replication) takes the next range of consecutive positions, i.e. a single stream as in TestU01, and every thread runs the test on its own consecutive piece of it. (Pieces at positions far apart, e.g. differing only in the bits 28 and above, would compare outputs of related counters, which a single stream of TestU01 doesn't, and weak counter-based generators fail such comparisons in their low bits.)
* SerialOver, Run, the runs of bits, AutoCor and SampleCorr see the single sequence exactly (every thread from a few values before its share). The others make their observations from the consecutive positions of every thread (an observation does not span threads), and count them in histograms.
* The collisions of CollisionOver, CollisionPermut and of the spacings of BirthdaySpacings are counted exactly by a hash table on GPU (small enough for the cache of the GPU, in passes over the points: those of CollisionOver are made again for every pass, those of CollisionPermut stored once), against a Poisson distribution as TestU01 does for these parameters. The points of BirthdaySpacings, SampleProd and SampleMean are sorted on GPU by rocThrust (or Thrust for CUDA).
* Beyond the memory budget, the points are made again for every part: the collisions of CollisionPermut are counted by the parts of the hashes of the points (equal points fall in the same part; CollisionOver needs no memory growing with the samples), and the sorted points are taken by the parts of their top bits in increasing order; the spacings of BirthdaySpacings then go to buckets on the host by their hashes, and their collisions are counted bucket by bucket. ClosePairs takes the points by slabs of their first coordinate, each with a margin of the distance searched below it, and counts the pairs of a point of the slab. The Fourier transforms are batched within the budget.
* ClosePairs finds the close pairs of the points in the torus (the maximum norm) in a grid of cells on GPU, and computes the six statistics of TestU01 on the host.
* LinearComp runs the Berlekamp-Massey algorithm on the host (the bits made on GPU), on 64-bit words in loops vectorized by the compiler; its time grows as the square of the bits (0.4 s at the scale 1, about 53 s at 16 with `HOST_ARCH=-march=native` on a Ryzen 9 9950X). The chi-square of the linear complexity, which TestU01 computes only for more than 20 sequences (not in BigCrush), is not computed.
* MatrixRank computes the ranks over GF(2) on GPU: a matrix of 30 x 30 per thread (from the consecutive outputs of the thread), the larger ones in global memory by a block each (batched within the memory budget).
* AppearanceSpacings sorts the keys (block, index) on GPU (by the parts of the blocks beyond the memory budget), so that the occurrences of a block are in order, and sums the logarithms of their distances on GPU.
* LempelZiv parses the sequences by LZ78 on the host (the outputs made on GPU), with the trie in arrays; as the mean and the standard deviation of TestU01 are tabulated up to 2^28 bits, the scale multiplies the number of sequences N instead of their length (for the other tests, it multiplies n, the number of the matrices of MatrixRank and K of AppearanceSpacings).
* Fourier3 transforms the sequences of bits by rocFFT (or cuFFT for CUDA), in batches of 4096, in double precision (in single precision, its statistic differed from that of TestU01 on the same outputs in the fifth digit).

The differences from TestU01:

* Where TestU01 takes a statistic of a wrong variance, the right one is taken: the number of the runs of bits of sstring_Run (B / 16 for B bits), and SampleCorr (13/144 per term). The runs of bits are counted in a fixed number of values instead of until n runs.
* The chi-squares merge the classes of an expected count below 10 from the low end (gofs_MergeClasses of TestU01 merges from both ends).
* HammingWeight2 counts the weights within 12 standard deviations of the mean (the farther ones at the ends); LongestHeadRun starts every block afresh.
* As in TestU01, a value in [0, d) is floor(d u) of u = (x << r) / 2^32 for the output x, i.e. the r most significant bits are dropped.

The tests were checked against TestU01: SerialOver, Fourier3, LinearComp, MatrixRank (of the large matrices), AppearanceSpacings and LempelZiv give the same statistics as TestU01 on the same outputs, the p-values are uniform for good generators (Kolmogorov-Smirnov tests over seeds), and for weak generators the same tests and instances fail as in BigCrush, with failures of a similar strength.

## License

Apache License 2.0 ([LICENSE](LICENSE)). The tests are reimplemented after the source code of TestU01, and the library links its probdist and mylib libraries, so it is also subject to the license of TestU01 ([LICENSE.TestU01](LICENSE.TestU01)), which permits only non-commercial use; see [NOTICE](NOTICE).
