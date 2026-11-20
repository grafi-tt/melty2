# Kernel search tools

Tools used to design the melty2 kernel: the counter-differential bias metric, the simulated annealing search of ARX structures, the evaluation of structured ("weak") keys, and generators of output streams for statistical test suites. See [doc/kernel_search.md](../doc/kernel_search.md) for the method and the results.

The code in this directory is distributed under the [MIT License](LICENSE), unlike the rest of melty2 (CC0).

## Build

```sh
make cpu          # build/{prof,sa,weakkeys}_cpu and build/check
make gpu          # build/{prof,sa,weakkeys}_gpu, with the HIP metric (HIPCC=hipcc HIP_ARCH=gfx1030 by default)
```

The GPU metric is written in HIP, and is about 50 times faster than the single-threaded CPU one (Radeon RX 6800 vs a core of Ryzen 9 9950X). The CPU and GPU tools compute the same statistic, but use different random counters, so they agree only statistically.

## Structures

A structure is written as ops separated by `;`, after the counter injection `v0 ^= lo; v1 += hi; v2 ^= hi; v3 += lo`:

* `A d s`: `v[d] += v[s]`
* `X d s`: `v[d] ^= v[s]`
* `R d s r`: `v[d] = rotl(v[d] ^ v[s], r)`

The kernel of melty2 1.0 is `ARX_MELTY2` in [arx.h](arx.h), and `build/check` confirms that it computes the same outputs as `include/melty2_kernel.h`. [candidates.txt](candidates.txt) lists the structures compared in the search.

## Tools

* `prof structure mlo mhi nkeys n [seed [weak_words]]`

  The bias profile at the depths `mlo` to `mhi`, as the mean and the maximum over random keys, with `n` counter pairs per key and delta.

* `sa L seed iters init [nkeys n weak_words]`

  Simulated annealing of an `L`-op structure starting from the first `L` ops of `init` (`ORIG22` for the original 22-step kernel). The kernel of melty2 1.0 is the result of `build/sa_gpu 20 504 3000 ORIG22`.

* `weakkeys structure mlo mhi n list [nbase [dump]]`

  The bias of structured keys given by a list of key modifications (see [keymod.h](keymod.h)), compared with 64 control keys. The lists used are in [weak/](weak/), and [weak/gen_weak.py](weak/gen_weak.py) generates them.

* `gen/run.sh TOOL structure M args...`

  Builds `TOOL` with the first `M` ops of the structure compiled in (cached in `build/gen/`), and runs it. The key is that of `melty2_init(seed, 0, 0, 0)`, modified by the environment variable `MELTY2_KEYMOD`, and `swap = 1` swaps the halves of the counter.

  * `gen structure M seed swap`: the output stream, e.g. for `| RNG_test stdin32` of PractRand.
  * `genlow structure M seed swap nbits [shift]`: the lower `nbits` bits of the outputs, packed.
  * `genpair structure M seed swap mask mode`: two streams of keys sharing the words in `mask`, combined by xor, interleaving or subtraction.
  * `fperm structure M seed swap N r t`: the permutation test of `t`-tuples.
  * `bcsub structure M seed swap [tests...]`: selected tests of BigCrush (needs TestU01; set `CFLAGS` and `LDFLAGS` for its location).

* `hwd/prngs_hwd.c`

  The input of the [Hamming-weight dependency test](https://prng.di.unimi.it/hwd.php) from stdin. `hwd.c` is distributed under the GPL and isn't included; see the comment in the file.

## Example

```sh
make cpu gpu
build/check
build/prof_gpu "$(sed -n 's/^gr10_4: //p' candidates.txt)" 10 20 16 4194304
build/weakkeys_gpu "$(sed -n 's/^gr10_4: //p' candidates.txt)" 15 20 4194304 weak/weak32.lst 8
MELTY2_KEYMOD='4=z,5=z' gen/run.sh gen "$(sed -n 's/^gr10_4: //p' candidates.txt)" 20 1 0 | RNG_test stdin32 -tlmax 64GB
```
