#include "melty2.h"

#include <stdatomic.h>

#include <cpu_features_macros.h>

#ifdef _MSC_VER

#ifdef CPU_FEATURES_ARCH_X86_64
#define MELTY2_RAWBLKGEN_DISPATCH_X86
void melty2_rawblkgen_avx512(const melty2_key *key, uint32_t ctr_lo, uint32_t ctr_hi, uint32_t *out);
void melty2_rawblkgen_avx2(const melty2_key *key, uint32_t ctr_lo, uint32_t ctr_hi, uint32_t *out);
#endif

#else

#ifdef CPU_FEATURES_ARCH_X86_64
#define MELTY2_RAWBLKGEN_DISPATCH_X86
void melty2_rawblkgen_avx512(const melty2_key *key, uint32_t ctr_lo, uint32_t ctr_hi, uint32_t *out);
void melty2_rawblkgen_avx2(const melty2_key *key, uint32_t ctr_lo, uint32_t ctr_hi, uint32_t *out);
#endif
#ifdef CPU_FEATURES_ARCH_X86_32
#define MELTY2_RAWBLKGEN_DISPATCH_X86
void melty2_rawblkgen_avx2(const melty2_key *key, uint32_t ctr_lo, uint32_t ctr_hi, uint32_t *out);
void melty2_rawblkgen_sse2(const melty2_key *key, uint32_t ctr_lo, uint32_t ctr_hi, uint32_t *out);
#endif
#ifdef CPU_FEATURES_ARCH_AARCH64
#define MELTY2_RAWBLKGEN_DISPATCH_AARCH64
#include <cpuinfo_aarch64.h>
void melty2_rawblkgen_sve(const melty2_key *key, uint32_t ctr_lo, uint32_t ctr_hi, uint32_t *out);
#endif
#ifdef CPU_FEATURES_ARCH_ARM
#define MELTY2_RAWBLKGEN_DISPATCH_ARM
#include <cpuinfo_arm.h>
void melty2_rawblkgen_neon(const melty2_key *key, uint32_t ctr_lo, uint32_t ctr_hi, uint32_t *out);
#endif

#endif

#ifndef __GNUC__
#define __builtin_expect(c, v) c
#endif

void melty2_rawblkgen_generic(const melty2_key *key, uint32_t ctr_lo, uint32_t ctr_hi, uint32_t *out);

#ifdef MELTY2_RAWBLKGEN_DISPATCH_X86
#include <cpuinfo_x86.h>
#define Features X86Features
#define GetInfo() GetX86Info()
#endif
#ifdef MELTY2_RAWBLKGEN_DISPATCH_AARCH64
#include <cpuinfo_aarch64.h>
#define Features Aarch64Features
#define GetInfo() GetAarch64Info()
#endif
#ifdef MELTY2_RAWBLKGEN_DISPATCH_ARM
#include <cpuinfo_arm.h>
#define Features ArmFeatures
#define GetInfo() GetArmInfo()
#endif

void melty2_rawblkgen(const melty2_key *key, uint32_t ctr_lo, uint32_t ctr_hi, uint32_t *out) {
#ifdef GetInfo
    static atomic_int tri_flag = 0;  /* uninitialized: 0, locked: -1, initialized: 1 */
    static Features features;

    int v = atomic_load_explicit(&tri_flag, memory_order_acquire);
    if (__builtin_expect(v <= 0, 0)) {
        int locked = 1;
        if (__builtin_expect(v == 0, 1)) {
            if (__builtin_expect(atomic_compare_exchange_strong_explicit(
                    &tri_flag, &v, -1, memory_order_acq_rel, memory_order_acquire), 1)) {
                features = GetInfo().features;
                atomic_store_explicit(&tri_flag, 1, memory_order_release);
                locked = 0;
            } else if (v > 0) {
                locked = 0;
            }
        }
        if (locked) {
            do {
                v = atomic_load_explicit(&tri_flag, memory_order_acquire);
            } while (v < 0);
        }
    }

#ifdef CPU_FEATURES_ARCH_X86_64
    if (features.avx512f && features.avx512dq) {
        melty2_rawblkgen_avx512(key, ctr_lo, ctr_hi, out);
        return;
    }
    if (features.avx2) {
        melty2_rawblkgen_avx2(key, ctr_lo, ctr_hi, out);
        return;
    }
#endif
#ifdef CPU_FEATURES_ARCH_X86_32
    if (features.avx2) {
        melty2_rawblkgen_avx2(key, ctr_lo, ctr_hi, out);
        return;
    }
    if (features.sse2) {
        melty2_rawblkgen_sse2(key, ctr_lo, ctr_hi, out);
        return;
    }
#endif
#ifdef CPU_FEATURES_ARCH_AARCH64
    if (features.sve) {
        melty2_rawblkgen_sve(key, ctr_lo, ctr_hi, out);
        return;
    }
#endif
#ifdef CPU_FEATURES_ARCH_ARM
    if (features.neon) {
        melty2_rawblkgen_neon(key, ctr_lo, ctr_hi, out);
        return;
    }
#endif
#endif
    melty2_rawblkgen_generic(key, ctr_lo, ctr_hi, out);
}
