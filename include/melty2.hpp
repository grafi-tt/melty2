#ifndef MELTY2_MELTY2_HPP_
#define MELTY2_MELTY2_HPP_

#include <array>
#include <limits>

#include "melty2.h"

namespace melty2 {

template <size_t buflen>
class basic_generator {
    static_assert(buflen != 0, "buflen must be non-zero");

public:
    explicit basic_generator(std::array<uint32_t, 4> seed, uint64_t ctr = 0) noexcept : ctr_(ctr), idx_(buflen) {
        melty2_init(&key_, seed[0], seed[1], seed[2], seed[3]);
    }
    basic_generator() noexcept : basic_generator({}, 0) {}

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

private:
    melty2_key key_;
    uint64_t ctr_;
    size_t idx_;
    uint32_t buf_[buflen];
};

using generator = basic_generator<32>;

}  // namespace melty2

#endif
