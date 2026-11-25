// SPDX-License-Identifier: Apache-2.0
//
// Structured ("weak") key modifications, applied to a key derived from a seed.
// A spec is a comma-separated list of "i=OP" applied in order, where i is the word index (0..5):
//   z: k_i = 0          o: k_i = ~0          bN: k_i = 1 << N
//   eJ: k_i = k_J       nJ: k_i = -k_J       mJ: k_i = k_J ^ 0x80000000     rJ:R: k_i = rotl(k_J, R)
//   hV: k_i = V         aV: k_i &= V         qV: k_i |= V                   xV: k_i ^= V   (V: C integer literal)
//   dJ:C: k_i = k_J + C (C: signed decimal)
// e.g. "4=z,5=a0x0000ffff" makes k_4 zero and the upper half of k_5 zero.
// Programs take the spec from the environment variable MELTY2_KEYMOD by from_env().
#ifndef KEYMOD_HPP_
#define KEYMOD_HPP_

#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <stdexcept>
#include <string>

#include "arx.hpp"
#include "melty2.h"

namespace keymod {

// Apply the spec to the key k in place; nullptr does nothing. Throws std::invalid_argument for an
// invalid spec.
inline void apply(arx::Key &k, const char *spec) {
    if (!spec) return;
    const auto bad = [&](const std::string &what) { throw std::invalid_argument(what + ": " + spec); };
    const char *p = spec;
    const auto integer = [&p] { return std::strtol(p, const_cast<char **>(&p), 10); };
    const auto word = [&](long j) {
        if (j < 0 || j > 5) bad("invalid word index in the key modification spec");
        return static_cast<int>(j);
    };
    while (*p) {
        const long i = integer();
        if (*p++ != '=' || i < 0 || i > 5) bad("invalid key modification spec");
        const char c = *p++;
        uint32_t &ki = k[i];
        if (c == 'h' || c == 'a' || c == 'q' || c == 'x') {
            const auto v = static_cast<uint32_t>(std::strtoul(p, const_cast<char **>(&p), 0));
            if (c == 'h') ki = v;
            else if (c == 'a') ki &= v;
            else if (c == 'q') ki |= v;
            else ki ^= v;
        } else if (c == 'd') {
            const int j = word(integer());
            if (*p == ':') ++p;
            ki = k[j] + static_cast<uint32_t>(integer());
        } else {
            long j = 0, r = 0;
            if (c != 'z' && c != 'o') j = integer();
            if (c == 'r' && *p == ':') {
                ++p;
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
            default: bad(std::string("invalid key modification '") + c + "'");
            }
        }
        if (*p == ',') ++p;
    }
}

// Apply the spec in the environment variable MELTY2_KEYMOD, if set, to the key k.
inline void from_env(arx::Key &k) { keymod::apply(k, std::getenv("MELTY2_KEYMOD")); }

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
