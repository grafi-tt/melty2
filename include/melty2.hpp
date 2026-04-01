#ifndef MELTY2_MELTY2_HPP_
#define MELTY2_MELTY2_HPP_

#include <array>
#include <limits>

#include "melty2.h"

namespace melty2 {

class generator {
public:
    explicit generator(std::array<uint32_t, 4> seed, uint64_t ctr = 0) noexcept : ctr_{ctr}, idx_{MELTY2_RAWBLKLEN} {
        melty2_init(&key_, seed[0], seed[1], seed[2], seed[3]);
    }
    generator() noexcept : generator{{}, 0} {}

    uint32_t operator()() noexcept {
        if (idx_ >= MELTY2_RAWBLKLEN) {
            idx_ -= MELTY2_RAWBLKLEN;
            melty2_rawblkgen(&key_, static_cast<uint32_t>(ctr_), static_cast<uint32_t>(ctr_ >> 32), buf_);
            ctr_ += uint32_t{MELTY2_RAWBLKLEN};
        }
        return buf_[idx_++];
    }

    using result_type = uint32_t;
    static constexpr uint32_t min() noexcept { return std::numeric_limits<uint32_t>::min(); }
    static constexpr uint32_t max() noexcept { return std::numeric_limits<uint32_t>::max(); }

    uint64_t ctr() const noexcept {
        return ctr_ + idx_ - uint32_t{MELTY2_RAWBLKLEN};
    }
    generator& ctr(uint64_t new_ctr) noexcept {
        ctr_ = new_ctr - new_ctr % MELTY2_RAWBLKLEN;
        idx_ = MELTY2_RAWBLKLEN + new_ctr % MELTY2_RAWBLKLEN;
        return *this;
    }

private:
    melty2_key key_;
    uint64_t ctr_;
    uint32_t idx_;
    uint32_t buf_[MELTY2_RAWBLKLEN];
};

}  // namespace melty2

#endif  // MELTY2_MELTY2_HPP_
