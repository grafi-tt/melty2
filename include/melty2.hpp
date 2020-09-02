#ifndef MELTY2_MELTY2_HPP_
#define MELTY2_MELTY2_HPP_

#include <limits>
#include <string>
#include <type_traits>
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
    explicit key(seeder&& seeder) { melty2_initkey(&impl_, &seeder.impl_); }

    explicit key() : key(seeder()) {}

    template <typename Head, typename... Tail, typename std::enable_if<
        !std::is_same<typename std::decay<Head>::type, key>::value,
        std::nullptr_t>::type = nullptr>
    explicit key(Head&& head, Tail&&... tail) : key(seed_args(seeder(), std::forward(head), std::forward(tail)...)) {}

    void gen(uint64_t ctr, size_t len, uint32_t* out) {
        melty2_gen(&impl_, ctr, len, out);
    }

private:
    template <typename Head, typename... Tail>
    static seeder&& seed_args(seeder&& seeder, Head&& head, Tail&&... tail) {
        return seed_args(std::move(seeder << head), std::forward<Tail>(tail)...);
    }
    static seeder&& seed_args(seeder&& seeder) { return std::move(seeder); }

    melty2_key impl_;
};

template <size_t buflen>
class basic_generator {
    static_assert(buflen != 0, "buflen must be non-zero");

public:
    explicit basic_generator(const key& key, uint64_t ctr) : key_(key), ctr_(ctr), idx_(buflen) {}

    explicit basic_generator() : basic_generator(key(), 0) {}

    template <typename Head, typename... Tail, typename std::enable_if<
        !std::is_same<typename std::decay<Head>::type, key>::value &&
        !std::is_same<typename std::decay<Head>::type, basic_generator<buflen>>::value,
        std::nullptr_t>::type = nullptr>
    explicit basic_generator(Head&& head, Tail&&... tail) : basic_generator(key(std::forward<Head>(head), std::forward<Tail>(tail)...), 0) {}

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

using generator = basic_generator<32>;

}  // namespace melty2

#endif
