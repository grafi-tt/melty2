#include "melty2.h"

#include <stdatomic.h>

#include <cpu_features_macros.h>

#ifdef CPU_FEATURES_ARCH_X86
#include <cpuinfo_x86.h>
#define GetInfo() GetX86Info()
#define Features X86Features
void melty2_rawblkgen_avx512(const melty2_key *key, uint32_t ctr_lo, uint32_t ctr_hi, uint32_t *out);
void melty2_rawblkgen_avx2(const melty2_key *key, uint32_t ctr_lo, uint32_t ctr_hi, uint32_t *out);
#ifndef _MSC_VER
void melty2_rawblkgen_sse2(const melty2_key *key, uint32_t ctr_lo, uint32_t ctr_hi, uint32_t *out);
#endif
#endif

#if defined(CPU_FEATURES_ARCH_ARM) && !defined(_MSC_VER)
#include <cpuinfo_arm.h>
#define GetInfo() GetArmInfo()
#define Features ArmFeatures
void melty2_rawblkgen_neon(const melty2_key *key, uint32_t ctr_lo, uint32_t ctr_hi, uint32_t *out);
#endif

void melty2_rawblkgen_generic(const melty2_key *key, uint32_t ctr_lo, uint32_t ctr_hi, uint32_t *out);

void melty2_rawblkgen(const melty2_key *key, uint32_t ctr_lo, uint32_t ctr_hi, uint32_t *out) {
    static atomic_int tri_flag = 0;  /* uninitialized: 0, locked: -1, initialized: 1 */
    static Features features;

    int v = atomic_load_explicit(&tri_flag, memory_order_acquire);
    if (v == 0 && atomic_compare_exchange_strong_explicit(&tri_flag, &v, -1, memory_order_acq_rel, memory_order_acquire)) {
        features = GetInfo().features;
        atomic_store_explicit(&tri_flag, 1, memory_order_release);
    } else {
        while (v == -1) {
            v = atomic_load_explicit(&tri_flag, memory_order_acquire);
        }
    }

#ifdef CPU_FEATURES_ARCH_X86
    if (features.avx512f && features.avx512dq) {
        melty2_rawblkgen_avx512(key, ctr_lo, ctr_hi, out);
        return;
    }
    if (features.avx2) {
        melty2_rawblkgen_avx2(key, ctr_lo, ctr_hi, out);
        return;
    }
#ifndef _MSC_VER
    if (features.sse2) {
        melty2_rawblkgen_sse2(key, ctr_lo, ctr_hi, out);
        return;
    }
#endif
#endif

#if defined(CPU_FEATURES_ARCH_ARM) && !defined(_MSC_VER)
    if (features.neon) {
        melty2_rawblkgen_neon(key, ctr_lo, ctr_hi, out);
        return;
    }
#endif

    melty2_rawblkgen_generic(key, ctr_lo, ctr_hi, out);
}
