#ifndef MELTY2_MELTY2_INTERNAL_KERNEL_H_
#define MELTY2_MELTY2_INTERNAL_KERNEL_H_

#define melty2_rotl(x, s) ((x) << (s) | (x) >> (32 - (s)))

#ifdef MELTY2_INTERNAL_SWAP32_CTR
#define MELTY2_INTERNAL_FST(a, b) (b)
#define MELTY2_INTERNAL_SND(a, b) (a)
#else
#define MELTY2_INTERNAL_FST(a, b) (a)
#define MELTY2_INTERNAL_SND(a, b) (b)
#endif

#define melty2_kernel(v, a, b) ( \
    (v)[0] ^= MELTY2_INTERNAL_FST(a, b), \
    (v)[1] += MELTY2_INTERNAL_SND(a, b), \
    (v)[2] ^= MELTY2_INTERNAL_SND(a, b), \
    (v)[3] += MELTY2_INTERNAL_FST(a, b), \
    (v)[1] ^= (v)[0], \
    (v)[1] = melty2_rotl((v)[1], 5), \
    (v)[2] += (v)[1], \
    (v)[3] ^= (v)[2], \
    (v)[3] = melty2_rotl((v)[3], 9), \
    (v)[1] += (v)[3], \
    (v)[5] ^= (v)[1], \
    (v)[5] = melty2_rotl((v)[5], 13), \
    (v)[3] += (v)[5], \
    (v)[2] ^= (v)[3], \
    (v)[2] = melty2_rotl((v)[2], 30), \
    (v)[4] += (v)[2], \
    (v)[1] ^= (v)[4], \
    (v)[1] = melty2_rotl((v)[1], 17), \
    (v)[4] += (v)[1], \
    (v)[3] ^= (v)[4], \
    (v)[3] = melty2_rotl((v)[3], 25), \
    (v)[0] += (v)[3], \
    (v)[1] ^= (v)[0], \
    (v)[1] = melty2_rotl((v)[1], 5), \
    (v)[0] ^= (v)[1], \
    (v)[0] = melty2_rotl((v)[0], 17), \
    (v)[5] += (v)[0], \
    (v)[3] ^= (v)[5], \
    (v)[3] = melty2_rotl((v)[3], 14), \
    (v)[3] += (v)[2], \
    (v)[0] ^= (v)[3], \
    (v)[0] = melty2_rotl((v)[0], 12), \
    (v)[3] ^= (v)[0], \
    (v)[3] += (v)[5] \
)

#endif  /* MELTY2_MELTY2_INTERNAL_KERNEL_H_ */
