#ifndef MELTY2_MELTY2_HPP_
#define MELTY2_MELTY2_HPP_

#include <limits>
#include <string>
#include <utility>

#if __cplusplus >= 201703L
#include <string_view>
#endif

#include "melty2.h"

namespace melty2 {

namespace detail {

struct manip_str_t {};
struct manip_bin_t {};

}  // namespace detail

static auto str = detail::manip_str_t{};
static auto bin = detail::manip_bin_t{};

class seeder {
public:
    seeder() : use_bin_(false) {
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

    friend seeder& operator<<(seeder& seeder, unsigned short s) {
        melty2_seed_uint(&seeder.impl_, s);
        return seeder;
    }

    friend seeder& operator<<(seeder& seeder, unsigned int s) {
        melty2_seed_uint(&seeder.impl_, s);
        return seeder;
    }

    friend seeder& operator<<(seeder& seeder, unsigned long s) {
        melty2_seed_uint(&seeder.impl_, s);
        return seeder;
    }

    friend seeder& operator<<(seeder& seeder, unsigned long long s) {
        melty2_seed_uint(&seeder.impl_, s);
        return seeder;
    }

    friend seeder& operator<<(seeder& seeder, short s) {
        melty2_seed_int(&seeder.impl_, s);
        return seeder;
    }

    friend seeder& operator<<(seeder& seeder, int s) {
        melty2_seed_int(&seeder.impl_, s);
        return seeder;
    }

    friend seeder& operator<<(seeder& seeder, long s) {
        melty2_seed_int(&seeder.impl_, s);
        return seeder;
    }

    friend seeder& operator<<(seeder& seeder, long long s) {
        melty2_seed_int(&seeder.impl_, s);
        return seeder;
    }

    friend seeder& operator<<(seeder& seeder, float s) {
        melty2_seed_float(&seeder.impl_, s);
        return seeder;
    }

    friend seeder& operator<<(seeder& seeder, double s) {
        melty2_seed_double(&seeder.impl_, s);
        return seeder;
    }

    friend seeder& operator<<(seeder& seeder, const char* s) {
        melty2_seed_str(&seeder.impl_, s);
        return seeder;
    }

    friend seeder& operator<<(seeder& seeder, detail::manip_str_t) {
        seeder.use_bin_ = false;
        return seeder;
    }

    friend seeder& operator<<(seeder& seeder, detail::manip_bin_t) {
        seeder.use_bin_ = true;
        return seeder;
    }

    friend seeder& operator<<(seeder& seeder, const std::string& s) {
        if (seeder.use_bin_) {
            melty2_seed_bin(&seeder.impl_, s.data(), s.size());
        } else {
            melty2_seed_strwithlen(&seeder.impl_, s.data(), s.size());
        }
        return seeder;
    }

#if __cplusplus >= 201703L
    friend seeder& operator<<(seeder& seeder, std::string_view s) {
        if (seeder.use_bin_) {
            melty2_seed_bin(&seeder.impl_, s.data(), s.size());
        } else {
            melty2_seed_strwithlen(&seeder.impl_, s.data(), s.size());
        }
        return seeder;
    }
#endif

private:
    melty2_seeder impl_;
    bool use_bin_;
    friend class key;
};

class key {
public:
    key(const key&) = default;
    key(key&&) = default;

    explicit key(seeder&& seeder) { melty2_initkey(&impl_, &seeder.impl_); }

    template <typename... Args>
    explicit key(Args&&... args) {
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
    static_assert(buflen != 0, "buflen must be non-zero");

public:
    generator(const key& key, uint64_t ctr = 0) : key_(key), ctr_(ctr), idx_(buflen) {}

    uint32_t operator()() {
        if (idx_ == buflen) {
            idx_ = 0;
            key_.gen(ctr_, buflen, buf_);
            ctr_ += static_cast<uint64_t>(buflen);
        }
        return buf_[idx_++];
    }

    uint64_t current_ctr() const { return ctr_ - static_cast<uint64_t>(buflen - idx_); }
    void set_ctr(uint64_t ctr) { ctr_ = ctr, idx_ = buflen; }

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
