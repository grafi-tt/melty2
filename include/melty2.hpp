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

    /* generator from a derived key (e.g. by melty2_init_u64), which avoids deriving it again */
    static generator from_key(const melty2_key &key, uint64_t ctr = 0) noexcept {
        return generator{from_key_tag_{}, key, ctr};
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
    /* a tag, so that the constructor doesn't make generator{{...}} ambiguous */
    struct from_key_tag_ {};
    generator(from_key_tag_, const melty2_key &key, uint64_t ctr) noexcept : key_(key) {
        this->ctr(ctr);
    }

    melty2_key key_;
    uint64_t ctr_;
    uint32_t idx_;
    uint32_t buf_[MELTY2_RAWBLKLEN];
};

}  // namespace melty2

#endif  // MELTY2_MELTY2_HPP_
