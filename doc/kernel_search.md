# Search of the melty2 kernel

This document describes how the 20-step kernel of melty2 1.0 was derived from the original 22-step kernel, and how it was evaluated. The tools are in [search/](../search/) (MIT License).

## Goals and constraints

* The output is one 32bit word per counter, and the key is six 32bit words `v[0..5]` with the counter injection `v0 ^= lo; v1 += hi; v2 ^= hi; v3 += lo`.
* Only 32bit additions, xors and rotations are used. Multiplications are avoided, as some accelerators (e.g. MN-Core) have no integer multiplier, and optimizations that only help x86 are avoided.
* Fewer steps than the original 22, without losing the safety margin, including the margin for structured keys. When a huge number of threads each use their own seed, keys with a structure appearing with a probability around 2^-32 (e.g. a zero word) do appear, so they have to be as safe as random keys.

Structures are written as `A d s` (`v[d] += v[s]`), `X d s` (`v[d] ^= v[s]`) and `R d s r` (`v[d] = rotl(v[d] ^ v[s], r)`) separated by `;`, and the output of a prefix of `m` ops is the word written by its last op. The original kernel is

```
R1 0 7;A2 1;R3 2 9;A4 3;R5 4 13;A0 5;R2 0 18;A4 2;R1 4 16;A3 1;R3 5 12;A0 3;R1 0 8;A2 1;R3 2 7;A3 4;R0 3 9;A1 0;R2 1 13;A3 2;R0 3 18;A0 2
```

## Counter-differential bias metric

A counter-based generator must make the outputs for nearby counters look unrelated. The metric ([search/metric.h](../search/metric.h)) measures how far this property holds at every intermediate depth of a structure.

* For a key and random base counters `c`, the outputs at the depth `m` are compared between `c` and `c + delta`, where `delta = 2^j` for `j` in 0..7 (the lower word) and 32..39 (the upper word).
* The top 8 bits of `rotl(diff, 4w)` are histogrammed, for the addition difference and the xor difference, and for the 8 windows `w = 0..7`, so every bit position of the difference is observed.
* The bias of the key at the depth `m` is the sum over (delta, kind, window) of `(chi2 - 255) / n`, where `n` is the number of counter pairs. For an ideal generator, it is 0 with the standard deviation `sqrt(2 * 255 * 256) / n`.

The **margin** of a structure is its length minus the last depth whose maximum bias over keys is clearly above the noise. The bias decreases roughly by one or two orders of magnitude per step, so a resolution of 2^22 to 2^28 pairs per delta identifies the depth within about one step.

The metric has a blind spot: the least significant bits of ARX outputs mix slowly (an addition has no carry into bit 0), which a bias of the top 8 bits of differences doesn't capture well. So the stream of the lower bits of the outputs is tested separately by PractRand (`search/gen/genlow.c`).

## Valid structures

Early searches found structures with good metric values but with ops not affecting the output, or with a key word only xored into the output. Such structures are useless or weak, so the search accepts only structures satisfying the following (`arx_valid` in [search/arx.h](../search/arx.h)):

* every op affects the output, and all the six key words reach the output;
* every key word passes through an addition (as either operand) before the output, so that no word only whitens the output by xor.

## Simulated annealing

[search/sa.c](../search/sa.c) searches structures of a fixed length `L` and a fixed number of rotations.

* A move changes the destination or the source of an op, the rotation amount (or flips an addition and an xor), swaps two ops, or moves a rotation to another op. One or two moves are applied per iteration, and invalid structures are rejected.
* The objective is the sum of `log10(max(bias_max(m), T))` over the last 11 depths, where `T` is twice the noise level. Minimizing it pushes the biased depths earlier, rather than only reducing the bias of the output.
* The temperature decreases linearly from 0.52 to 0.02.
* The objective is evaluated on 16 keys with 2^20 pairs per delta. A quarter of the keys have a zero word and another quarter have a word equal to a rotation of another word, so that the search doesn't overfit random keys.

The search for 20 steps started from the first 20 ops of the original kernel, with 10 rotations, 3000 iterations and the seeds 501 to 508. The objective at the verification ranged from -11.3 to -19.3, and the structures of the seeds 503, 504, 507 and 508 reached -17.9 to -19.3. At 2^26 pairs over 16 random keys, the last depth clearly biased (above 5 sigma) was 16 for the seeds 504 (8 sigma) and 507 (39 sigma), 17 for 503 (21 sigma), and 18 for 508 (86 sigma). So `gr10_4` (seed 504) has the largest margin and the weakest remaining bias, and it was evaluated in detail together with `gr10_3` (seed 503). The GPU implementation of the metric ([search/metric_gpu.hip](../search/metric_gpu.hip)) made this feasible: an iteration takes about 0.15 seconds on Radeon RX 6800 (about 50 times faster than a core of Ryzen 9 9950X), and the run of a seed takes about 8 minutes. Rerunning the command below reproduces `gr10_4`, with the same objective values.

```sh
make -C search gpu
search/build/sa_gpu 20 504 3000 ORIG22   # prints gr10_4
```

A variant with a 160bit key (five words) was also searched: a 17-step core was found by the same kind of annealing, and then extended by 3 ops with an objective of the robustness against truncating the core (`e3_r1_1` and `e3_r1_s102`). Its smaller state mixes faster, but structured keys affect it more, as discussed below.

## Evaluation

The finalists were evaluated in the following order, as each stage is more expensive than the previous one.

1. **Metric at a high resolution**: the margin over random keys at 2^26 pairs.
2. **Lower bits**: PractRand on the stream of the least significant bit (and the lower bits) of the outputs, for both counter orders, to measure the margin of the blind spot of the metric.
3. **Structured keys**: the metric per key ([search/weakkeys.c](../search/weakkeys.c)) for lists of key modifications ([search/weak/gen_weak.py](../search/weak/gen_weak.py)), compared with 64 control keys:
   * `weak32` (714 specs): a zero, all-one or Hamming weight 1 word, and equal, negated, MSB-flipped or rotated word pairs, i.e. structures with probabilities around 2^-32;
   * `weak48` (570 specs): zero runs over words, zero half-words in three words, low weight word pairs, near-equal pairs and small offsets, with probabilities around 2^-48 to 2^-38;
   * `sweep` (834 specs): parameterized classes of a single word (weight, zero top or bottom bits, periodicity, runs) and of word pairs (xor with low weight masks, rotations, offsets, negation, complement), to see which features matter.
4. **Pairs of keys sharing words**: PractRand on the xor, the interleaving and the difference of two streams whose keys share some words (`genpair`), as related keys appear when parts of seeds are shared.
5. **Output tests**: PractRand up to 64GB, including structured keys, a subset of BigCrush for structured keys (`bcsub`), full BigCrush, and the Hamming-weight dependency test.

Both counter orders are tested everywhere, as the lower and the upper word of the counter are injected into different words. This matters: an earlier 20-step candidate failed the permutation test of BigCrush only with the swapped counter order, for some keys (`search/gen/fperm.c` reproduces the statistic quickly).

## Results

| Structure | Steps | Key | Instructions (with / without rotation) | Metric margin | Lower bits margin | Notes |
|---|---|---|---|---|---|---|
| original | 22 | 192bit | 37 / 59 | 2 to 3 | 3 | |
| `gr10_4` | 20 | 192bit | 34 / 54 | 4 | 3 | adopted |
| `gr10_3` | 20 | 192bit | 34 / 54 | 3 | 3 | sensitive to special values of `v[4]` |
| `e3_r1_1` | 20 | 160bit | 34 / 54 | 2 | 3 | slight excess for structured keys at depth 20 |
| `e3_r1_s102` | 20 | 160bit | 34 / 54 | 2 | 3 | |

`gr10_3` was rejected because keys with an all-one `v[4]` showed a reproducible bias in CollisionOver (t = 21) of BigCrush, in addition to its sensitivity in the metric. The 160bit variants have smaller margins, so the 192bit `gr10_4` was adopted.

For `gr10_4`:

* **Structured keys**: for the keys of `weak32`, a bias remains only up to depth 16 (by weight-1 values of `v[4]`), and the depths 19 and 20 are indistinguishable from random keys at 2^28 pairs. The sweep showed that sparse `v[4]`, `v[5]` and `v[3]` (in this order) are the sensitive features, while sparse `v[0..2]` and relations between words (equal, rotated, negated, near-equal or offset pairs) are as harmless as random keys. The worst keys of `weak48`, with both `v[4]` and `v[5]` sparse, have a bias up to depth 17 (7.7 sigma at 2^27 pairs), and the depths 18 to 20 are clean. So the margin for structured keys is 3 steps, not less than the margin of the original kernel for random keys. As `v[4]` and `v[5]` are not injected with the counter and are absorbed only by a few steps of the first half of the kernel, this ordering is expected.
* **Lower bits**: PractRand on the least significant bit passed 1GB for all the 1428 runs of `weak32` and the 1140 runs of `weak48` (both counter orders).
* **Pairs of keys**: all the runs passed.
* **PractRand**: passed 64GB for 2 seeds and 2 structured keys (a zero `v[0]` and an all-one `v[4]`) in both counter orders.
* **BigCrush**: the subset passed for the 13 structured keys in both counter orders, and full BigCrush passed in 8 runs (2 seeds, both counter orders, with and without the bit reversal of the outputs); the 4 p-values outside [0.001, 0.999] were in different tests, and didn't reproduce with other seeds.
* **Hamming-weight dependency test**: 14 and 15 steps fail at about 1.5e11 bytes, and 16 steps are detected from 3e12 to 1e13 bytes, while 17, 18 and 20 steps, including structured keys such as `v[4] = v[5] = 0`, pass 1e13 bytes. So the margin seen by this test is about 4 steps.

The kernel is almost sequential: the critical path is 31 of its 34 instructions. This is intended, as the outputs for many counters are computed in parallel by SIMD lanes or GPU threads, and a sequential dependency mixes the state more per step.

## Notes

* **Rejection of weak keys**: rejecting keys with both `v[4]` and `v[5]` sparse (each with a popcount at most 8 or at least 24, or with a zero or all-one half-word; a rate of about 2^-14.6) would remove the worst class. It isn't adopted, as the margin is already 3 steps for such keys, and the rejection would make the key derivation more complex and the key distribution non-uniform. If it were adopted in the future, it would change the outputs, so it would be a new major version.
* **A wider counter**: a third 32bit counter word can only be injected into `v[4]` and `v[5]` without letting differences cancel with the lower word. With the current kernel, a difference in such a word is clearly biased at the output (z = 12 to 37000 depending on the injection), as these words are mixed the slowest. A 96bit counter would need a search with the three-word injection, and likely 22 to 25 steps.
