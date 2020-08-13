#ifndef MELTY2_MELTY2_HPP_
#define MELTY2_MELTY2_HPP_

#include <limits>
#include <utility>

#include "melty2.h"

namespace melty2 {

class seeder {
public:
    seeder() { melty2_initseeder(&impl_); }

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
    friend class key;
};

class key {
public:
    key(seeder&& seeder) { melty2_initkey(&impl_, &seeder.impl_); }

    template <typename... Args>
    key(Args&&... args) {
        seeder seeder;
        seed_args(seeder, std::forward<Args>(args)...);
        melty2_initkey(&impl_, &seeder.impl_);
    }

    void gen(uint64_t ctr, size_t len, uint32_t* out) {
        melty2_gen(&impl_, ctr, len, out);
    }

private:
    template <typename Head, typename... Tail>
    static seeder& seed_args(seeder& seeder, Head&& head, Tail&&... tail) {
        return seed_args(seeder << head, std::forward<Tail>(tail)...);
    }
    static seeder& seed_args(seeder& seeder) { return seeder; }

    melty2_key impl_;
};

template <size_t buflen = 32>
class generator {
public:
    uint32_t operator()() {
        if (idx_ == buflen) {
            idx_ = 0;
            key_.gen(ctr_, buflen, buf_);
        }
        return buf_[idx_++];
    }

    static constexpr uint32_t min() { return std::numeric_limits<uint32_t>::min(); }
    static constexpr uint32_t max() { return std::numeric_limits<uint32_t>::max(); }

private:
    key key_;
    uint64_t ctr_;
    size_t idx_;
    uint32_t buf_[buflen];
};

}  // namespace melty2

#endif
