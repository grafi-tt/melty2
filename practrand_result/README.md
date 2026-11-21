# PractRand results

Outputs of [PractRand](https://pracrand.sourceforge.net/) 0.94 (`RNG_test stdin32`, core test set, up to 64GB) for the full 32bit outputs of melty2. The keys are those of `melty2_init(seed, 0, 0, 0)`, and two of them are modified to have structure.

| File | Seed | Key | Counter order |
|---|---|---|---|
| [seed1_default.txt](seed1_default.txt) | 1 | as derived | default |
| [seed1_swap32_ctr.txt](seed1_swap32_ctr.txt) | 1 | as derived | halves swapped |
| [seed2_default.txt](seed2_default.txt) | 2 | as derived | default |
| [seed2_swap32_ctr.txt](seed2_swap32_ctr.txt) | 2 | as derived | halves swapped |
| [seed1_k0zero_default.txt](seed1_k0zero_default.txt) | 1 | `v[0]` = 0 | default |
| [seed1_k0zero_swap32_ctr.txt](seed1_k0zero_swap32_ctr.txt) | 1 | `v[0]` = 0 | halves swapped |
| [seed1_k4ones_default.txt](seed1_k4ones_default.txt) | 1 | `v[4]` = 0xffffffff | default |
| [seed1_k4ones_swap32_ctr.txt](seed1_k4ones_swap32_ctr.txt) | 1 | `v[4]` = 0xffffffff | halves swapped |

All of them pass up to 64GB. There are 5 "unusual" results in total, the mildest grade, each at a single length only.

The tools are in [search/](../search). The outputs were made by

```sh
S="R1 0 5;A2 1;R3 2 9;A1 3;R5 1 13;A3 5;R2 3 30;A4 2;R1 4 17;A4 1;R3 4 25;A0 3;R1 0 5;R0 1 17;A5 0;R3 5 14;A3 2;R0 3 12;X3 0;A3 5"
MELTY2_KEYMOD=MOD search/gen/run.sh gen "$S" 20 SEED SWAP | RNG_test stdin32 -tlmax 64GB -tlfail
```

with MOD = `''`, `0=z` or `4=o` (see [search/keymod.h](../search/keymod.h)), and SWAP = 0 (default counter order) or 1 (halves swapped).
