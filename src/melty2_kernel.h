#ifndef MELTY2_MELTY2_KERNEL_H_
#define MELTY2_MELTY2_KERNEL_H_

#define melty2_rotl(x, s) ((x) << (s) | (x) >> (32 - (s)))

#define melty2_kernel(v, a, b) ( \
    (v)[0] ^= (a), \
    (v)[1] += (b), \
    (v)[2] ^= (b), \
    (v)[3] += (a), \
    (v)[1] ^= (v)[0], \
    (v)[1] = melty2_rotl((v)[1], 7), \
    (v)[2] += (v)[1], \
    (v)[3] ^= (v)[2], \
    (v)[3] = melty2_rotl((v)[3], 9), \
    (v)[4] += (v)[3], \
    (v)[5] ^= (v)[4], \
    (v)[5] = melty2_rotl((v)[5], 13), \
    (v)[0] += (v)[5], \
    (v)[2] ^= (v)[0], \
    (v)[2] = melty2_rotl((v)[2], 18), \
    (v)[4] += (v)[2], \
    (v)[1] ^= (v)[4], \
    (v)[1] = melty2_rotl((v)[1], 16), \
    (v)[3] += (v)[1], \
    (v)[3] ^= (v)[5], \
    (v)[3] = melty2_rotl((v)[3], 12), \
    (v)[0] += (v)[3], \
    (v)[1] ^= (v)[0], \
    (v)[1] = melty2_rotl((v)[1], 8), \
    (v)[2] += (v)[1], \
    (v)[3] ^= (v)[2], \
    (v)[3] = melty2_rotl((v)[3], 7), \
    (v)[3] += (v)[4], \
    (v)[0] ^= (v)[3], \
    (v)[0] = melty2_rotl((v)[0], 9), \
    (v)[1] += (v)[0], \
    (v)[2] ^= (v)[1], \
    (v)[2] = melty2_rotl((v)[2], 13), \
    (v)[3] += (v)[2], \
    (v)[0] ^= (v)[3], \
    (v)[0] = melty2_rotl((v)[0], 18), \
    (v)[0] += (v)[2] \
)

#endif  /* MELTY2_MELTY2_KERNEL_H_ */
