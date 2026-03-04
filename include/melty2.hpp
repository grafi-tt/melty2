#ifndef MELTY2_MELTY2_HPP_
#define MELTY2_MELTY2_HPP_

#include <array>
#include <limits>

#include "melty2.h"

namespace melty2 {

class generator {
public:
    explicit generator(std::array<uint32_t, 4> seed, uint64_t ctr = 0) noexcept : ctr_(ctr), idx_(32) {
        melty2_init(&key_, seed[0], seed[1], seed[2], seed[3]);
    }
    generator() noexcept : generator({}, 0) {}

    uint32_t operator()() noexcept {
        if (idx_ >= 32) {
            idx_ -= 32;
            melty2_rawblkgen(&key_, ctr_, ctr_ >> 32, buf_);
            ctr_ += 32u;
        }
        return buf_[idx_++];
    }

    using result_type = uint32_t;
    static constexpr uint32_t min() noexcept { return std::numeric_limits<uint32_t>::min(); }
    static constexpr uint32_t max() noexcept { return std::numeric_limits<uint32_t>::max(); }

    uint64_t ctr() const noexcept {
        return ctr_ - 32u + idx_;
    }
    generator& ctr(uint64_t new_ctr) noexcept {
        ctr_ = new_ctr - new_ctr % 32;
        idx_ = 32 + new_ctr % 32;
        return *this;
    }

private:
    melty2_key key_;
    uint64_t ctr_;
    uint32_t idx_;
    uint32_t buf_[32];
};

}  // namespace melty2

#endif
