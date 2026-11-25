# testu01_gpu

A library running the tests of the BigCrush battery of [TestU01](http://simul.iro.umontreal.ca/testu01/tu01.html) on an AMD GPU (HIP), for counter-based generators: a generator whose output at any position of its sequence can be computed directly. The tests are reimplemented after the source code of TestU01 1.2.3 with the parameters of their instances in BigCrush, and take tens of seconds of a Radeon RX 6800 instead of hours of a CPU core. Their p-values agree with those of TestU01 statistically (the outputs used differ), so they screen generators quickly, and sample sizes far beyond BigCrush are practical; the original BigCrush still judges the final generators.

All the tests of BigCrush are implemented but MatrixRank, AppearanceSpacings, LinearComp and LempelZiv.

## Build

Requirements: ROCm (HIP, hiprtc, rocFFT and rocThrust), and TestU01 installed (its headers and the `probdist` and `mylib` libraries, for the distributions and the goodness-of-fit tests).

```sh
make                          # build/libtestu01_gpu.a and the example build/splitmix_bigcrush
make TESTU01=/opt/testu01 HIP_ARCH=gfx1100
```

A program includes `include/testu01_gpu.hpp` and links `build/libtestu01_gpu.a -lprobdist -lmylib -lhiprtc -lrocfft` with `hipcc`. The kernels of the tests are compiled at run time for the current device, but those of rocThrust (the sorts and prefix sums) are compiled by the build for `HIP_ARCH` (default gfx1030), which must include the GPU used.

## Usage

The generator is given as HIP device source, defining in the global namespace

```cpp
__device__ inline void testu01_gpu_generate(const testu01_gpu::Params &p, uint64_t pos,
                                            uint32_t (&out)[testu01_gpu::kLanes]);
```

which writes the 32-bit outputs at the positions `pos, pos + 1, ..., pos + kLanes - 1` (kLanes = 16) of the sequence of the parameters `p` (32 words, interpreted by the source as it likes: a key, a seed, a selection of streams, ...). The library compiles its kernels with the source by hiprtc, so the generator is inlined into them as straight-line code; hiprtc has no standard headers, but `uint32_t`, `uint64_t`, `UINT32_C` and `UINT64_C` are available.

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
* `Options::report`: called for every statistic with a `Result` (the name, the parameters, the kind of statistic, its value, the p-value and a z-score); `format` gives the line printed above, marking a p-value outside [0.001, 0.999] with `*****` as TestU01 does.

`run` throws `testu01_gpu::TestFailure` for a test failing completely (e.g. the observations of a test taking far more outputs than expected, for a very bad generator), and `std::runtime_error` for the errors of HIP and hiprtc.

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

How they are computed on GPU:

* Every test (and replication) takes the next range of consecutive positions, i.e. a single stream as in TestU01, and every thread runs the test on its own consecutive piece of it. (Pieces at positions far apart, e.g. differing only in the bits 28 and above, would compare outputs of related counters, which a single stream of TestU01 doesn't, and weak counter-based generators fail such comparisons in their low bits.)
* SerialOver, Run, the runs of bits, AutoCor and SampleCorr see the single sequence exactly (every thread from a few values before its share). The others make their observations from the consecutive positions of every thread (an observation does not span threads), and count them in histograms.
* The collisions of CollisionOver, CollisionPermut and of the spacings of BirthdaySpacings are counted exactly by a hash table on GPU (in passes small enough for the cache of the GPU), against a Poisson distribution as TestU01 does for these parameters. The points of BirthdaySpacings, SampleProd and SampleMean are sorted on GPU by rocThrust.
* ClosePairs finds the close pairs of the points in the torus (the maximum norm) in a grid of cells on GPU, and computes the six statistics of TestU01 on the host.
* Fourier3 transforms the sequences of bits by rocFFT, in batches of 4096, in double precision (in single precision, its statistic differed from that of TestU01 on the same outputs in the fifth digit).

The differences from TestU01:

* Where TestU01 takes a statistic of a wrong variance, the right one is taken: the number of the runs of bits of sstring_Run (B / 16 for B bits), and SampleCorr (13/144 per term). The runs of bits are counted in a fixed number of values instead of until n runs.
* The chi-squares merge the classes of an expected count below 10 from the low end (gofs_MergeClasses of TestU01 merges from both ends).
* HammingWeight2 counts the weights within 12 standard deviations of the mean (the farther ones at the ends); LongestHeadRun starts every block afresh.
* As in TestU01, a value in [0, d) is floor(d u) of u = (x << r) / 2^32 for the output x, i.e. the r most significant bits are dropped.

The tests were checked against TestU01: SerialOver and Fourier3 give the same statistics as TestU01 on the same outputs, the p-values are uniform for good generators (Kolmogorov-Smirnov tests over seeds), and for weak generators the same tests and instances fail as in BigCrush, with failures of a similar strength.

## License

Apache License 2.0 ([LICENSE](LICENSE)). The tests are reimplemented after the source code of TestU01, and the library links its probdist and mylib libraries, so it is also subject to the license of TestU01 ([LICENSE.TestU01](LICENSE.TestU01)), which permits only non-commercial use; see [NOTICE](NOTICE).
