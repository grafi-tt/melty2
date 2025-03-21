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

class seeder {
public:
    seeder() noexcept {
        melty2_initseeder(&impl_);
    }

    seeder&  update() noexcept {
        return *this;
    }
    template <typename Head, typename... Tail>
    seeder& update(const Head& head, const Tail&... tail) noexcept {
        return (*this << head).update(tail...);
    }

    seeder& write(const char *data, uint32_t len) {
        melty2_seed_strwithlen(&impl_, data, len);
        return *this;
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
    friend seeder& operator<<(seeder& seeder, const std::string& s) noexcept {
        melty2_seed_strwithlen(&seeder.impl_, s.data(), truncate_size(s.size()));
        return seeder;
    }
#if __cplusplus >= 201703L
    friend seeder& operator<<(seeder& seeder, std::string_view s) noexcept {
        melty2_seed_strwithlen(&seeder.impl_, s.data(), truncate_size(s.size()));
        return seeder;
    }
#endif
    friend seeder& operator<<(seeder& seeder, const char* s) noexcept {
        melty2_seed_str(&seeder.impl_, s);
        return seeder;
    }

private:
    template <size_t buflen>
    friend class basic_generator;

    static uint32_t truncate_size(size_t sz) {
        constexpr uint32_t max_sz = std::numeric_limits<uint32_t>::max();
        return sz <= max_sz ? static_cast<uint32_t>(sz) : max_sz;
    }

    melty2_seeder impl_;
};

template <size_t buflen>
class basic_generator {
    static_assert(buflen != 0, "buflen must be non-zero");

public:
    explicit basic_generator(seeder&& seeder, uint64_t ctr = 0) noexcept : ctr_(ctr), idx_(buflen) {
        melty2_initkey(&key_, &seeder.impl_);
    }

    basic_generator() noexcept : basic_generator(seeder(), 0) {}

    template <typename Head, typename... Tail, typename std::enable_if<
        !std::is_same<typename std::decay<Head>::type, seeder>::value &&
        !std::is_same<typename std::decay<Head>::type, basic_generator<buflen>>::value,
        std::nullptr_t>::type = nullptr>
    explicit basic_generator(const Head& head, const Tail&... tail) noexcept :
        basic_generator(make_seeder(head, tail...), 0) {}

    uint32_t operator()() noexcept {
        if (idx_ == buflen) {
            idx_ = 0;
            melty2_gen(&key_, ctr_, buflen, buf_);
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
        basic_generator newgen(uninit_tag{});
        newgen.ctr_ = 0;
        newgen.idx_ = buflen;
        melty2_splitkey(&key_, &newgen.key);
        return newgen;
    }

private:
    struct uninit_tag {};

    explicit basic_generator(uninit_tag) noexcept {}

    template <typename... Args>
    static seeder make_seeder(const Args&... args) noexcept {
        seeder s;
        s.update(args...);
        return s;
    }

    melty2_key key_;
    uint64_t ctr_;
    size_t idx_;
    uint32_t buf_[buflen];
};

using generator = basic_generator<32>;

}  // namespace melty2

#endif
