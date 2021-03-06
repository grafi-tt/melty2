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
    seeder() noexcept : use_bin_(false) {
        melty2_initseeder(&impl_);
    }

    friend seeder& operator<<(seeder& seeder, std::nullptr_t) noexcept {
        melty2_seed_null(&seeder.impl_);
        return seeder;
    }

    friend seeder& operator<<(seeder& seeder, bool s) noexcept {
        melty2_seed_bool(&seeder.impl_, s);
        return seeder;
    }

    friend seeder& operator<<(seeder& seeder, unsigned short s) noexcept {
        melty2_seed_uint(&seeder.impl_, s);
        return seeder;
    }

    friend seeder& operator<<(seeder& seeder, unsigned int s) noexcept {
        melty2_seed_uint(&seeder.impl_, s);
        return seeder;
    }

    friend seeder& operator<<(seeder& seeder, unsigned long s) noexcept {
        melty2_seed_uint(&seeder.impl_, s);
        return seeder;
    }

    friend seeder& operator<<(seeder& seeder, unsigned long long s) noexcept {
        melty2_seed_uint(&seeder.impl_, s);
        return seeder;
    }

    friend seeder& operator<<(seeder& seeder, short s) noexcept {
        melty2_seed_int(&seeder.impl_, s);
        return seeder;
    }

    friend seeder& operator<<(seeder& seeder, int s) noexcept {
        melty2_seed_int(&seeder.impl_, s);
        return seeder;
    }

    friend seeder& operator<<(seeder& seeder, long s) noexcept {
        melty2_seed_int(&seeder.impl_, s);
        return seeder;
    }

    friend seeder& operator<<(seeder& seeder, long long s) noexcept {
        melty2_seed_int(&seeder.impl_, s);
        return seeder;
    }

    friend seeder& operator<<(seeder& seeder, float s) noexcept {
        melty2_seed_float(&seeder.impl_, s);
        return seeder;
    }

    friend seeder& operator<<(seeder& seeder, double s) noexcept {
        melty2_seed_double(&seeder.impl_, s);
        return seeder;
    }

    friend seeder& operator<<(seeder& seeder, detail::manip_str_t) noexcept {
        seeder.use_bin_ = false;
        return seeder;
    }

    friend seeder& operator<<(seeder& seeder, detail::manip_bin_t) noexcept {
        seeder.use_bin_ = true;
        return seeder;
    }

    friend seeder& operator<<(seeder& seeder, const std::string& s) noexcept {
        if (seeder.use_bin_) {
            melty2_seed_bin(&seeder.impl_, s.data(), s.size());
        } else {
            melty2_seed_strwithlen(&seeder.impl_, s.data(), s.size());
        }
        return seeder;
    }

    friend seeder& operator<<(seeder& seeder, const char* s) {
        if (seeder.use_bin_) {
#if __cplusplus >= 201703L
            std::string_view v(s);
#else
            std::string v(s);
#endif
            melty2_seed_bin(&seeder.impl_, v.data(), v.size());
        } else {
            melty2_seed_str(&seeder.impl_, s);
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
    explicit key() noexcept : key(seeder()) {}
    explicit key(seeder&& seeder) noexcept { melty2_initkey(&impl_, &seeder.impl_); }

    template <typename Head, typename... Tail, typename std::enable_if<
        !std::is_same<typename std::decay<Head>::type, key>::value &&
        !std::is_same<typename std::decay<Head>::type, seeder>::value &&
        !std::is_same<typename std::decay<Head>::type, melty2_key>::value,
        std::nullptr_t>::type = nullptr>
    explicit key(Head&& head, Tail&&... tail) :
        key(seed_args(seeder(), std::forward<Head>(head), std::forward<Tail>(tail)...)) {}

    key split() noexcept {
        melty2_key newimpl;
        melty2_splitkey(&impl_, &newimpl);
        return key(newimpl);
    }

    void gen(uint64_t ctr, size_t len, uint32_t* out) const noexcept {
        melty2_gen(&impl_, ctr, len, out);
    }

private:
    explicit key(melty2_key impl) noexcept : impl_(impl) {}

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
    explicit basic_generator() noexcept : basic_generator(key(), 0) {}
    explicit basic_generator(const key& key, uint64_t ctr = 0) noexcept : key_(key), ctr_(ctr), idx_(buflen) {}
    explicit basic_generator(seeder&& seeder, uint64_t ctr = 0) noexcept : key_(std::move(seeder)), ctr_(ctr), idx_(buflen) {}

    template <typename Head, typename... Tail, typename std::enable_if<
        !std::is_same<typename std::decay<Head>::type, key>::value &&
        !std::is_same<typename std::decay<Head>::type, seeder>::value &&
        !std::is_same<typename std::decay<Head>::type, basic_generator<buflen>>::value,
        std::nullptr_t>::type = nullptr>
    explicit basic_generator(Head&& head, Tail&&... tail) :
        basic_generator(key(std::forward<Head>(head), std::forward<Tail>(tail)...), 0) {}

    uint32_t operator()() noexcept {
        if (idx_ == buflen) {
            idx_ = 0;
            key_.gen(ctr_, buflen, buf_);
            ctr_ += static_cast<uint64_t>(buflen);
        }
        return buf_[idx_++];
    }

    using result_type = uint32_t;
    static constexpr uint32_t min() noexcept { return std::numeric_limits<uint32_t>::min(); }
    static constexpr uint32_t max() noexcept { return std::numeric_limits<uint32_t>::max(); }

    uint64_t ctr() const noexcept {
        return ctr_ - static_cast<uint64_t>(buflen - idx_);
    }
    basic_generator<buflen>& ctr(uint64_t new_ctr) noexcept {
        ctr_ = new_ctr;
        idx_ = buflen;
        return *this;
    }

    basic_generator<buflen> split() noexcept {
        key newkey = key_.split();
        return basic_generator<buflen>(newkey, ctr());
    }

private:
    key key_;
    uint64_t ctr_;
    size_t idx_;
    uint32_t buf_[buflen];
};

using generator = basic_generator<32>;

}  // namespace melty2

#endif
