// SPDX-License-Identifier: Apache-2.0
//
// Structured ("weak") key modifications, applied to a key derived from a seed.
// A spec is a comma-separated list of "i=OP" applied in order, where i is the word index (0..5):
//   z: k_i = 0          o: k_i = ~0          bN: k_i = 1 << N
//   eJ: k_i = k_J       nJ: k_i = -k_J       mJ: k_i = k_J ^ 0x80000000     rJ:R: k_i = rotl(k_J, R)
//   hV: k_i = V         aV: k_i &= V         qV: k_i |= V                   xV: k_i ^= V   (V: 0x... etc., see number.hpp)
//   dJ:C: k_i = k_J + C (C: signed decimal)
// e.g. "4=z,5=a0x0000ffff" makes k_4 zero and the upper half of k_5 zero.
// Programs take the spec from the environment variable MELTY2_KEYMOD by from_env().
#ifndef KEYMOD_HPP_
#define KEYMOD_HPP_

#include <algorithm>
#include <cstdlib>
#include <format>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include "arx.hpp"
#include "melty2.h"
#include "number.hpp"

namespace keymod {

// Apply the spec to the key k in place; an empty one does nothing. Throws std::invalid_argument for an
// invalid spec.
inline void apply(arx::Key &k, std::string_view spec) {
    const auto bad = [&](std::string_view what) { throw std::invalid_argument(std::format("{}: {}", what, spec)); };
    std::string_view p = spec;
    const auto integer = [&]() -> long {
        const std::optional<long> v = number::take<long>(p);
        if (!v) bad("a number is expected in the key modification spec");
        return *v;
    };
    // the next character, or '\0' at the end
    const auto next = [&p] {
        if (p.empty()) return '\0';
        const char c = p.front();
        p.remove_prefix(1);
        return c;
    };
    const auto word = [&](long j) {
        if (j < 0 || j > 5) bad("invalid word index in the key modification spec");
        return static_cast<int>(j);
    };
    while (!p.empty()) {
        const long i = integer();
        if (next() != '=' || i < 0 || i > 5) bad("invalid key modification spec");
        const char c = next();
        uint32_t &ki = k[i];
        if (c == 'h' || c == 'a' || c == 'q' || c == 'x') {
            const auto v = static_cast<uint32_t>(integer());
            if (c == 'h') ki = v;
            else if (c == 'a') ki &= v;
            else if (c == 'q') ki |= v;
            else ki ^= v;
        } else if (c == 'd') {
            const int j = word(integer());
            if (p.starts_with(':')) p.remove_prefix(1);
            ki = k[j] + static_cast<uint32_t>(integer());
        } else {
            long j = 0, r = 0;
            if (c != 'z' && c != 'o') j = integer();
            if (c == 'r' && p.starts_with(':')) {
                p.remove_prefix(1);
                r = integer();
            }
            switch (c) {
            case 'z': ki = 0; break;
            case 'o': ki = ~UINT32_C(0); break;
            case 'b':
                if (j < 0 || j > 31) bad("invalid bit index in the key modification spec");
                ki = UINT32_C(1) << j;
                break;
            case 'e': ki = k[word(j)]; break;
            case 'n': ki = -k[word(j)]; break;
            case 'm': ki = k[word(j)] ^ UINT32_C(0x80000000); break;
            case 'r':
                if (r < 0 || r > 31) bad("invalid rotation in the key modification spec");
                ki = arx::rotl(k[word(j)], static_cast<int>(r));
                break;
            default: bad(std::format("invalid key modification '{}'", c));
            }
        }
        if (p.starts_with(',')) p.remove_prefix(1);
    }
}

// Apply the spec in the environment variable MELTY2_KEYMOD, if set, to the key k.
inline void from_env(arx::Key &k) {
    if (const char *spec = std::getenv("MELTY2_KEYMOD")) keymod::apply(k, spec);
}

// the key of melty2_init(seed, 0, 0, 0)
inline arx::Key seed_key(uint32_t seed) {
    melty2_key k;
    melty2_init(&k, seed, 0, 0, 0);
    arx::Key key;
    std::copy(std::begin(k.v_), std::end(k.v_), key.begin());
    return key;
}

}  // namespace keymod

#endif
