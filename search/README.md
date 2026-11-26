# Kernel search tools

Tools used to design the melty2 kernel: the counter-differential bias metric, the simulated annealing search of ARX structures, the evaluation of structured ("weak") keys, and generators of output streams for statistical test suites. See [doc/kernel_search.md](../doc/kernel_search.md) for the method and the results.

The code in this directory is distributed under the [Apache License 2.0](LICENSE), unlike the rest of melty2 (CC0). `testu01_stest` uses the library [testu01_gpu](../testu01_gpu), which reimplements tests of [TestU01](http://simul.iro.umontreal.ca/testu01/tu01.html) after its source code, so it is also subject to the license of TestU01 ([LICENSE.TestU01](LICENSE.TestU01)), which permits only non-commercial use; see [NOTICE](NOTICE).

## Build

```sh
make cpu          # build/{prof,sa,weakkeys}_cpu, build/check and build/sactl
make gpu          # build/{prof,sa,weakkeys}_gpu, with the HIP metric (HIPCC=hipcc HIP_ARCH=gfx1030 by default), and build/sactl
make stest        # build/testu01_stest, with the library ../testu01_gpu and the probdist library of TestU01 (TESTU01=/usr/local by default;
                  # HOST_ARCH=-march=native for the SIMD of the host in LinearComp, see ../testu01_gpu)
```

The tools are written in C++17, and the GPU metric in HIP (C++17 as well); `melty2_init.c` of the library is compiled as C, and so is `hwd/prngs_hwd.c`, which is included into `hwd.c`. The GPU metric is about 50 times faster than the single-threaded CPU one (Radeon RX 6800 vs a core of Ryzen 9 9950X). The CPU and GPU tools compute the same statistic, but use different random counters, so they agree only statistically.

## Structures

A structure is written as ops separated by `;`, after the counter injection `v0 ^= lo; v1 += hi; v2 ^= hi; v3 += lo`:

* `A d s`: `v[d] += v[s]`
* `X d s`: `v[d] ^= v[s]`
* `R d s r`: `v[d] = rotl(v[d] ^ v[s], r)`
* `S d s`: `v[d] -= v[s]` (used by the search only with `sa -S`)

The kernel of melty2 1.0 is `arx::kMelty2` in [arx.hpp](arx.hpp), and `build/check` confirms that it computes the same outputs as `include/melty2_kernel.h`. [candidates.txt](candidates.txt) lists the structures compared in the search.

## Tools

The tools take the required arguments as positional arguments, and the optional ones as options (`getopt_long` for C++ and `argparse` for Python), which are given before the positional arguments.

* `prof [-s seed] [-w weak_words] [-p pidfile] structure mlo mhi nkeys n`

  The bias profile at the depths `mlo` to `mhi`, as the mean and the maximum over random keys, with `n` counter pairs per key and delta. The keys are from `seed` (default 1), and with `weak_words` > 0, some of them have a zero word or a rotation relation among the first `weak_words` words (see [metric.hpp](metric.hpp)).

* `sa [-k nkeys] [-n pairs] [-w weak_words] [-K keyset] [-O objective] [-c checkpoint] [-p pidfile] [-S] [-R] [-C] [-P] [-F freeze] [-D deadline] [-T] L seed iters init`

  Simulated annealing of an `L`-op structure starting from the first `L` ops of `init` (`ORIG22` for the original 22-step kernel). The objective is evaluated on `nkeys` keys (default 16) with `pairs` counter pairs (default 2^20) and `weak_words` (default 6); with `keyset` 1, a quarter of the keys have sparse `v[4]` and `v[5]`. With `objective` 1, structures are compared by the last clearly biased depth first, and then by the default objective. With `checkpoint`, the state is saved every 100 iterations and on SIGUSR1 (and SIGTERM or SIGINT, then exiting), and an existing checkpoint is resumed exactly. With `-S`, subtractions are also used, and with `-R`, the structures repeating an addition, a subtraction or an xor without any write to its words in between (which merge into one op or cancel) are rejected. With `-C`, a candidate is compared with the current structure at the same seed. With `-P`, the structures where a key word doesn't propagate into the other words (e.g. it is only added to the output at the last op) are rejected. With `-F`, the first `freeze` ops of `init` are kept, e.g. to extend a core found with a smaller `L`. With `-D`, the structures where a key word doesn't propagate into the other words by the first `deadline` ops (e.g. one only added near the output, which the metric can't see as the key is fixed) are rejected. With `-T`, only the rotation amounts are changed, e.g. to tune a structure designed by hand. The kernel of melty2 1.0 is the result of `build/sa_gpu 20 504 3000 ORIG22`.

* `rank [-k nkeys] [-n pairs] [-s seed] [-w weak_words] [-K keyset] [-p pidfile] list`

  The margin of every structure in the list (a structure per line, optionally after a name and a tab), at a high resolution (32 keys including sparse `v[4]` and `v[5]`, and 2^24 pairs by default): the length minus the last depth whose maximum bias over the keys is above 5 times the noise level. As the objective of `sa` is evaluated at a resolution too low to see small biases near the output, `sa` generates candidates, and `rank` judges them. `collect.py FILES...` collects the distinct structures from the outputs of `sa` (e.g. the logs of `sactl`) into such a list.

* `weakkeys [-b nbase] [-d dump] [-p pidfile] structure mlo mhi n list`

  The bias of structured keys given by a list of key modifications (see [keymod.hpp](keymod.hpp)), applied to `nbase` base keys (default 2), compared with 64 control keys. With `dump`, the bias of every key is written to the file. The lists used are in [weak/](weak/), and `weak/gen_weak.py [--classes] {weak32|weak48|sweep}` generates them.

* `testu01_stest [-s seed] [-w] [-t tests] [-f scale] [-l keylist] [-m streams] [-o offset] [-M MB] [-F] [-p pidfile] structure M`
* `testu01_stest -S M[,M...] [-s seed] [-t tests] [-f scale] [-l keylist] [-m streams] [-o offset] [-D deadline] [-v] [-L] [-M MB] [-F] [-p pidfile] < structures`
* `testu01_stest -P plan [-s seed] [-m streams] [-D deadline] [-v] [-L] [-M MB] [-F] [-p pidfile] < structures`

  The tests of BigCrush, reimplemented on GPU after the source code of TestU01 with the parameters of their instances in BigCrush: SimpPoker, CouponCollector, Gap, MaxOft, CollisionOver, BirthdaySpacings, ClosePairs, Permutation and Run (`-t poker,coupon,gap,maxoft,collision,birthday,closepairs,permutation,run`), which the ARX structures of melty2 with too few steps fail systematically, and SerialOver, CollisionPermut, WeightDistrib, Savir2, GCD, HammingWeight2, HammingCorr, HammingIndep, the runs of bits, LongestHeadRun, AutoCor, SampleCorr, SampleProd, SampleMean, RandomWalk1, PeriodsInStrings, SumCollector, Fourier3, LinearComp, MatrixRank, AppearanceSpacings and LempelZiv (`serial,collpermut,weight,savir2,gcd,hamweight,hamcorr,hamindep,bitrun,longhead,autocor,samplecorr,sampleprod,samplemean,walk,periods,sumcoll,fourier,linearcomp,matrixrank,appearance,lempelziv`; see [testu01_gpu](../testu01_gpu) for the differences from TestU01). The tests are those of the library testu01_gpu, for the structure as a counter-based generator; the structure is compiled into the kernels at run time by hiprtc, and the tests take about 10 seconds of a Radeon RX 6800 for the outputs of the first `M` ops and a key, instead of about 28 minutes of a CPU core for the same tests of TestU01, so they screen structures, steps and keys quickly; the original BigCrush judges the final ones. They fail where TestU01 fails for structures near the boundary, and their p-values are uniform for a structure passing BigCrush. The key is that of `melty2_init(seed, 0, 0, 0)`, modified by `MELTY2_KEYMOD`, or by every spec of the file `keylist` (one per line, `-` for none; `keymod@streams` for a key with its own streams as `-m`, and `...#seed` for the key of another seed) in turn in one process with `-l`; `-w` swaps the halves of the counter, and `-f` scales the sample sizes; `-M` sets the device memory of the large tests (2048 MiB by default), beyond which they make their points again by parts, with the same results. `-F` takes faster computations whose results may differ from those of TestU01 in rounding (the Anderson-Darling statistics of SampleProd and SampleMean summed on GPU). With `-m ctr:b:c`, the tests see the interleaved outputs of 2^b streams whose counters differ in the bits c and above (as with a stream number in the high bits of the counter; c must be above the bits of the stream indices, about 44), and with `-m key:b:w`, those of the keys differing in the low bits of the word w (related keys, which don't arise from `melty2_init`). After the tests of a key, a line starting with `summary` gives the smallest and the largest p-values and the number of them outside [0.001, 0.999]. `-o` starts the counters of the tests at `offset` instead of 0 (a fresh sample, or the counters of the later tests of BigCrush). The second form is the score mode for searches: for every structure of stdin, the tests run quietly for the first `M` ops (or each of a list such as `15,16`), every key and both counter orders, and a line starting with `score` gives the largest z-score of the statistics (not for ClosePairs), the smallest p-value, the number of flagged ones, and the validity, the key propagation by `deadline` steps and the repetition of the structure; a line `structure<TAB>limit` stops the tests once the z-score is above `limit` (for a search rejecting the structure anyway). The third form scores the stages of the file `plan` in one process, a line `M[,M...] tests scale keylist [offset]` per stage, in their order. With `-v`, both print a line per key, step count and counter order before the `score` line. With `-L`, the score modes also give `logsum`, the sum of log(1 + z) of the largest z-scores of every key, step count and counter order, a test failing completely counting as z = 1e6 instead of stopping the structure, and the limit applies to this sum: it ranks structures failing badly, e.g. for tuning rotations far from passing. Every test (and replication) uses a single stream of consecutive counters as in TestU01, so the p-values agree with those of TestU01 statistically, though the streams differ (see [testu01_gpu](../testu01_gpu)).

* `sa_stat.py [--verify plan] [--verify-below Z] [--log] [--sub] [--max-sub K] [--alt N] [--pidfile F] structure plan log`

  Simulated annealing of a structure against the statistical tests of `testu01_stest` (as its score mode), for the weaknesses that the differential metric of `sa` does not see (e.g. those of sparse keys or of counter streams): the objective is the largest z-score of the tests of the stages of `plan` (as `testu01_stest -P`), and a candidate about to become the best is also scored on those of `--verify` (e.g. larger samples, more keys and streams). The Metropolis threshold is drawn first and passed to `testu01_stest`, so a failing candidate costs only its first failing tests. The moves are those of `sa` among the ops up to the smallest `M` (with `--sub`, also S, at most `K` with `--max-sub K`; with `--alt N`, the first `N` ops alternate R and A/S). For a structure failing grossly, `--log` anneals log(1 + z), and `--verify-below` skips the verification of candidates far from passing.

* `margin.py [--scan] [--groups F] [--only names] structure`

  The statistical margins of a structure by `testu01_stest`, per group of keys with a target margin: by default the normal keys (3), the weak-key classes of probability 2^-32 (3), about 2^-48 (2, a sample of the kinds of `weak/gen_weak.py weak48`) and 2^-64 (1), and counter streams of several keys (3). A key fails at `M` steps if a p-value is below 1e-6 or 3 are outside [0.001, 0.999]; the margin is the number of steps passing beyond the largest failing `M`. By default each group is tested only at `L - target` (a verdict); `--scan` finds the exact margins. A group file (lines `name target spec spec ...`) can take e.g. all 570 classes of `weak/gen_weak.py weak48`. The tests are by default SimpPoker, CouponCollector, Gap, MaxOft, CollisionOver, BirthdaySpacings and Fourier3 at the samples of BigCrush, and Permutation and Run at 4 times the samples, as their biases depend on the counter range. A test left out of the objective tends to get worse, so the objective should take every test, step count and stream setting judged. See `sa_stat.py --help`.

* `sactl {run [-s sa] [-i minutes] | snapshot | stop | status} DIR`

  Runs the `sa` jobs listed in `DIR/jobs.tsv` (a name and the arguments of `sa` per line, separated by tabs) in order, resuming interrupted ones from their checkpoints. `run` takes a snapshot of the directory into `DIR/snapshots/` every `minutes` (default 10), and `snapshot` and `stop` let the running `run` take a snapshot now, or save the state and exit. See the comment of [sactl.cc](sactl.cc) for the layout of the directory.

* `gen/run.sh TOOL structure M args...`

  Builds `TOOL` with the first `M` ops of the structure compiled in (cached in `build/gen/`, and rebuilt when the source is updated), and runs it with `args...`. The key is that of `melty2_init(seed, 0, 0, 0)`, modified by the environment variable `MELTY2_KEYMOD`, and `swap = 1` swaps the halves of the counter.

  * `gen seed swap`: the output stream, e.g. for `| RNG_test stdin32` of PractRand.
  * `genlow [-s shift] seed swap nbits`: the lower `nbits` bits of the outputs (after `>> shift`), packed.
  * `genpair seed swap mask mode`: two streams of keys sharing the words in `mask`, combined by xor, interleaving or subtraction.
  * `fperm seed swap N r t`: the permutation test of `t`-tuples.
  * `bcsub [-b small|crush|big] seed swap [tests...]`: selected tests of BigCrush, or the whole SmallCrush, Crush or BigCrush with `-b` (needs TestU01; set `CXXFLAGS` and `LDFLAGS` for its location).

* `hwd/prngs_hwd.c`

  The input of the [Hamming-weight dependency test](https://prng.di.unimi.it/hwd.php) from stdin. `hwd.c` is distributed under the GPL and isn't included; see the comment in the file.

## Example

```sh
make cpu gpu
build/check
build/prof_gpu "$(sed -n 's/^gr10_4: //p' candidates.txt)" 10 20 16 4194304
build/weakkeys_gpu -b 8 "$(sed -n 's/^gr10_4: //p' candidates.txt)" 15 20 4194304 weak/weak32.lst
MELTY2_KEYMOD='4=z,5=z' gen/run.sh gen "$(sed -n 's/^gr10_4: //p' candidates.txt)" 20 1 0 | RNG_test stdin32 -tlmax 64GB
```

A queue of searches:

```sh
mkdir -p jobs
printf 's1\t-K\t1\t-O\t1\t20\t601\t3000\tORIG22\n' > jobs/jobs.tsv
build/sactl run jobs &      # runs build/sa_gpu (copied to jobs/bin/sa)
build/sactl status jobs
build/sactl stop jobs       # later, "build/sactl run jobs" resumes
```
