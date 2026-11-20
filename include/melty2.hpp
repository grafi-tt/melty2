#ifndef MELTY2_MELTY2_HPP_
#define MELTY2_MELTY2_HPP_

#include <array>
#include <limits>

#include "melty2.h"

namespace melty2 {

class generator {
public:
    /* the 128bit seed is seed[0] | seed[1] << 32 | seed[2] << 64 | seed[3] << 96 */
    explicit generator(std::array<uint32_t, 4> seed, uint64_t ctr = 0) noexcept {
        melty2_init(&key_, seed[0], seed[1], seed[2], seed[3]);
        this->ctr(ctr);
    }
    /* the 128bit seed is the zero extension of seed; same as generator{{lower 32bit, upper 32bit, 0, 0}, ctr} */
    explicit generator(uint64_t seed, uint64_t ctr = 0) noexcept {
        melty2_init_u64(&key_, seed, 0);
        this->ctr(ctr);
    }
    generator() noexcept : generator{uint64_t{0}, 0} {}

    /* the 128bit seed is seed0 | seed1 << 64, as melty2_init_u64 */
    static generator from_u64_seeds(uint64_t seed0, uint64_t seed1, uint64_t ctr = 0) noexcept {
        return generator{{static_cast<uint32_t>(seed0), static_cast<uint32_t>(seed0 >> 32),
                          static_cast<uint32_t>(seed1), static_cast<uint32_t>(seed1 >> 32)}, ctr};
    }

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
