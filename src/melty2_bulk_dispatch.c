#include "melty2.h"

#include <stdbool.h>

#ifdef MELTY2_ENABLE_CPU_FEATURES
#include <cpu_features_macros.h>

#ifdef CPU_FEATURES_ARCH_X86
#include <cpuinfo_x86.h>
static X86Features features;
#define GetInfo() GetX86Info()
void melty2_bulkgen_avx512(const uint32_t * restrict key_v, uint64_t idx, uint64_t len, uint32_t * restrict out);
void melty2_bulkgen_avx2(const uint32_t * restrict key_v, uint64_t idx, uint64_t len, uint32_t * restrict out);
void melty2_bulkgen_sse2(const uint32_t * restrict key_v, uint64_t idx, uint64_t len, uint32_t * restrict out);
#endif

#ifdef CPU_FEATURES_ARCH_ARM
#include <cpuinfo_arm.h>
static ArmFeatures features;
#define GetInfo() GetArmInfo()
void melty2_bulkgen_neon(const uint32_t * restrict key_v, uint64_t idx, uint64_t len, uint32_t * restrict out);
#endif

static int init_state;
#endif

void melty2_bulkgen_generic(const uint32_t * restrict key_v, uint64_t idx, uint64_t len, uint32_t * restrict out);

void melty2_bulkgen(const melty2_key *key, uint64_t idx, uint64_t len, uint32_t *out) {
#ifdef MELTY2_ENABLE_CPU_FEATURES
    int tmp = __atomic_load_n(&init_state, __ATOMIC_ACQUIRE);
    if (tmp < 1) {
        if (tmp == 0 && __atomic_compare_exchange_n(&init_state, &tmp, -1, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
            features = GetInfo().features;
            __atomic_store_n(&init_state, 1, __ATOMIC_RELEASE);
        } else {
            while (__atomic_load_n(&init_state, __ATOMIC_ACQUIRE) != 1);
        }
    }
#endif
#ifdef CPU_FEATURES_ARCH_X86
    if (features.avx512f && features.avx512dq) return melty2_bulkgen_avx512(key->v_, idx, len, out);
    if (features.avx2) return melty2_bulkgen_avx2(key->v_, idx, len, out);
    if (features.sse2) return melty2_bulkgen_sse2(key->v_, idx, len, out);
#endif
#ifdef CPU_FEATURES_ARCH_ARM
    static ArmFeatures *p_features;
    if (features.neon) return melty2_bulkgen_neon(key->v_, idx, len, out);
#endif
    return melty2_bulkgen_generic(key->v_, idx, len, out);
}
