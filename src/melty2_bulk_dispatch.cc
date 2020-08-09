#include "melty2.h"

#include <cpu_features_macros.h>

#ifdef CPU_FEATURES_ARCH_X86
#include <cpuinfo_x86.h>
#define GetInfo() cpu_features::GetX86Info()
extern "C" void melty2_bulkgen_avx512(const uint32_t* key_v, uint64_t idx, uint64_t len, uint32_t* out);
extern "C" void melty2_bulkgen_avx2(const uint32_t* key_v, uint64_t idx, uint64_t len, uint32_t* out);
#ifndef _MSC_VER
extern "C" void melty2_bulkgen_sse2(const uint32_t* key_v, uint64_t idx, uint64_t len, uint32_t* out);
#endif
#endif

#if defined(CPU_FEATURES_ARCH_ARM) && !defined(_MSC_VER)
#include <cpuinfo_arm.h>
#define GetInfo() cpu_features::GetArmInfo()
extern "C" void melty2_bulkgen_neon(const uint32_t* key_v, uint64_t idx, uint64_t len, uint32_t* out);
#endif

extern "C" void melty2_bulkgen_generic(const uint32_t* key_v, uint64_t idx, uint64_t len, uint32_t* out);

extern "C" void melty2_bulkgen(const melty2_key *key, uint64_t idx, uint64_t len, uint32_t *out) {
    static const auto features = GetInfo().features;

#ifdef CPU_FEATURES_ARCH_X86
    if (features.avx512f && features.avx512dq) return melty2_bulkgen_avx512(key->v_, idx, len, out);
    if (features.avx2) return melty2_bulkgen_avx2(key->v_, idx, len, out);
#ifndef _MSC_VER
    if (features.sse2) return melty2_bulkgen_sse2(key->v_, idx, len, out);
#endif
#endif

#if defined(CPU_FEATURES_ARCH_ARM) && !defined(_MSC_VER)
    if (features.neon) return melty2_bulkgen_neon(key->v_, idx, len, out);
#endif

    return melty2_bulkgen_generic(key->v_, idx, len, out);
}
