// SPDX-License-Identifier: Apache-2.0
//
// Numbers in text, by std::from_chars. number::take reads a number at the start of a string_view and removes it
// (as strtol does, without skipping spaces); number::parse reads all of a string_view (e.g. an argument of a tool),
// throwing std::invalid_argument otherwise. With the base 0 (the default), an integer is read by its prefix: 0x
// hexadecimal, 0d decimal, 0o octal, 0b binary (in either case, and without a sign after it), else decimal, also
// with a leading 0 (so 010 is 10) and a sign for a signed type.
#pragma once

#include <charconv>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>

namespace number {

template <typename T>
std::optional<T> take(std::string_view &s, int base = 0) {
    std::string_view d = s;
    T v{};
    std::from_chars_result r;
    if constexpr (std::is_integral_v<T>) {
        if (base == 0) {
            if (d.size() >= 2 && d[0] == '0') {
                switch (d[1]) {
                case 'x': case 'X': base = 16; break;
                case 'd': case 'D': base = 10; break;
                case 'o': case 'O': base = 8; break;
                case 'b': case 'B': base = 2; break;
                default: break;
                }
                if (base) {
                    d.remove_prefix(2);
                    if (d.starts_with('-')) return std::nullopt;
                }
            }
            if (!base) base = 10;
        }
        r = std::from_chars(d.data(), d.data() + d.size(), v, base);
    } else {
        r = std::from_chars(d.data(), d.data() + d.size(), v);
    }
    if (r.ec != std::errc()) return std::nullopt;
    s.remove_prefix(static_cast<size_t>(r.ptr - s.data()));
    return v;
}

template <typename T>
T parse(std::string_view s, int base = 0) {
    std::string_view rest = s;
    const std::optional<T> v = take<T>(rest, base);
    if (!v || !rest.empty()) throw std::invalid_argument("not a number: " + std::string(s));
    return *v;
}

}  // namespace number
