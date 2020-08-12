#include "melty2.h"

#include <cpu_features_macros.h>

#ifdef CPU_FEATURES_ARCH_X86
#include <cpuinfo_x86.h>
#define GetInfo() cpu_features::GetX86Info()
extern "C" void melty2_gen_avx512(const melty2_key *key, uint64_t idx, uint64_t len, uint32_t *out);
extern "C" void melty2_gen_avx2(const melty2_key *key, uint64_t idx, uint64_t len, uint32_t *out);
#ifndef _MSC_VER
extern "C" void melty2_gen_sse2(const melty2_key *key, uint64_t idx, uint64_t len, uint32_t *out);
#endif
#endif

#if defined(CPU_FEATURES_ARCH_ARM) && !defined(_MSC_VER)
#include <cpuinfo_arm.h>
#define GetInfo() cpu_features::GetArmInfo()
extern "C" void melty2_gen_neon(const melty2_key *key, uint64_t idx, uint64_t len, uint32_t *out);
#endif

extern "C" void melty2_gen_generic(const melty2_key *key, uint64_t idx, uint64_t len, uint32_t *out);

extern "C" void melty2_gen(const melty2_key *key, uint64_t idx, uint64_t len, uint32_t *out) {
    static const auto features = GetInfo().features;

#ifdef CPU_FEATURES_ARCH_X86
    if (features.avx512f && features.avx512dq) return melty2_gen_avx512(key, idx, len, out);
    if (features.avx2) return melty2_gen_avx2(key, idx, len, out);
#ifndef _MSC_VER
    if (features.sse2) return melty2_gen_sse2(key, idx, len, out);
#endif
#endif

#if defined(CPU_FEATURES_ARCH_ARM) && !defined(_MSC_VER)
    if (features.neon) return melty2_gen_neon(key, idx, len, out);
#endif

    return melty2_gen_generic(key, idx, len, out);
}
