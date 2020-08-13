#ifndef MELTY2_MELTY2_HPP_
#define MELTY2_MELTY2_HPP_

#include <limits>

#include "melty2.h"

namespace melty2 {

class seeder {
public:
    seeder() {
        melty2_initseeder(&impl_);
    }

    friend seeder& operator<<(seeder& seeder, nullptr_t) {
        melty2_seed_null(&seeder.impl_);
        return seeder;
    }
    friend seeder& operator<<(seeder& seeder, bool s) {
        melty2_seed_bool(&seeder.impl_, s);
        return seeder;
    }
    friend seeder& operator<<(seeder& seeder, uint64_t s) {
        melty2_seed_uint(&seeder.impl_, s);
        return seeder;
    }
    friend seeder& operator<<(seeder& seeder, int64_t s) {
        melty2_seed_int(&seeder.impl_, s);
        return seeder;
    }

private:
    melty2_seeder impl_;
};

class key {
public:

private:
    melty2_key impl_;
};

class generator {
    static constexpr uint32_t min() { std::numeric_limits<uint32_t>::min(); }
    static constexpr uint32_t max() { std::numeric_limits<uint32_t>::max(); }
};

}  // namespace melty2

#endif
