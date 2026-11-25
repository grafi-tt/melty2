/* SPDX-License-Identifier: Apache-2.0 */
/* next() for the Hamming-weight dependency test of Blackman and Vigna, reading 32bit little-endian
 * words from stdin. hwd.c itself is distributed under the GPL at https://prng.di.unimi.it/hwd.php and
 * isn't included here; download it next to this file, and build as
 *   cc -O3 -march=native -DHWD_BITS=32 -I. hwd.c -o hwd32 -lm
 * Then, e.g. ../gen/run.sh gen "structure" 20 1 0 | ./hwd32 1e14 --progress */
#define HWD_BUFW (1 << 16)
static uint32_t hwd_buf[HWD_BUFW];
static size_t hwd_n = 0, hwd_i = 0;
static uint32_t next(void) {
    if (hwd_i == hwd_n) {
        hwd_n = fread(hwd_buf, 4, HWD_BUFW, stdin);
        hwd_i = 0;
        if (hwd_n == 0) {
            fprintf(stderr, "input ended\n");
            exit(1);
        }
    }
    return hwd_buf[hwd_i++];
}
