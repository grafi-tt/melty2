#include "melty2.h"

#include <cpu_features_macros.h>

#ifdef CPU_FEATURES_ARCH_X86
#include <cpuinfo_x86.h>
#define GetInfo() cpu_features::GetX86Info()
extern "C" void melty2_gen32_avx512(const melty2_key *key, uint32_t ctr_lo, uint32_t ctr_hi, uint32_t *out);
extern "C" void melty2_gen32_avx2(const melty2_key *key, uint32_t ctr_lo, uint32_t ctr_hi, uint32_t *out);
#ifndef _MSC_VER
extern "C" void melty2_gen32_sse2(const melty2_key *key, uint32_t ctr_lo, uint32_t ctr_hi, uint32_t *out);
#endif
#endif

#if defined(CPU_FEATURES_ARCH_ARM) && !defined(_MSC_VER)
#include <cpuinfo_arm.h>
#define GetInfo() cpu_features::GetArmInfo()
extern "C" void melty2_gen32_neon(const melty2_key *key, uint32_t ctr_lo, uint32_t ctr_hi, uint32_t *out);
#endif

extern "C" void melty2_gen32_generic(const melty2_key *key, uint32_t ctr_lo, uint32_t ctr_hi, uint32_t *out);

extern "C" void melty2_gen32(const melty2_key *key, uint32_t ctr_lo, uint32_t ctr_hi, uint32_t *out) {
    static const auto features = GetInfo().features;

#ifdef CPU_FEATURES_ARCH_X86
    if (features.avx512f && features.avx512dq) return melty2_gen32_avx512(key, ctr_lo, ctr_hi, out);
    if (features.avx2) return melty2_gen32_avx2(key, ctr_lo, ctr_hi, out);
#ifndef _MSC_VER
    if (features.sse2) return melty2_gen32_sse2(key, ctr_lo, ctr_hi, out);
#endif
#endif

#if defined(CPU_FEATURES_ARCH_ARM) && !defined(_MSC_VER)
    if (features.neon) return melty2_gen32_neon(key, ctr_lo, ctr_hi, out);
#endif

    return melty2_gen32_generic(key, ctr_lo, ctr_hi, out);
}
