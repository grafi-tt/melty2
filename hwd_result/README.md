# Hamming-weight dependency test results

Outputs of the [Hamming-weight dependency test](https://prng.di.unimi.it/hwd.php) (32bit words, up to 10^14 bytes) for the key of `melty2_init(1, 0, 0, 0)`.

| File | Kernel | Counter order | Final p-value |
|---|---|---|---|
| [default.txt](default.txt) | melty2 (20 steps) | default | 0.863 |
| [swap32_ctr.txt](swap32_ctr.txt) | melty2 (20 steps) | halves swapped | 0.0393 |
| [17steps_default.txt](17steps_default.txt) | first 17 steps | default | 1.42e-20 (fail) |
| [17steps_swap32_ctr.txt](17steps_swap32_ctr.txt) | first 17 steps | halves swapped | 0.381 |

The 17-step kernel is detected by the weight-1 signatures in the default counter order; its p-value goes below 10^-4 at about 3 × 10^13 bytes. The tools are in [search/](../search) (see [search/hwd/prngs_hwd.c](../search/hwd/prngs_hwd.c) to build `hwd32`). The outputs were made by

```sh
S="R1 0 5;A2 1;R3 2 9;A1 3;R5 1 13;A3 5;R2 3 30;A4 2;R1 4 17;A4 1;R3 4 25;A0 3;R1 0 5;R0 1 17;A5 0;R3 5 14;A3 2;R0 3 12;X3 0;A3 5"
search/gen/run.sh gen "$S" M 1 SWAP | search/hwd/hwd32 1e14 --progress --low-pv=1e-20
```

with M = 20 or 17, and SWAP = 0 (default counter order) or 1 (halves swapped).
