#include <charconv>
#include <iostream>
#include <optional>
#include <string_view>

extern "C" {
#include <bbattery.h>
#include <unif01.h>
//#include <sknuth.h>
}

#include "melty2.hpp"

namespace {

void show_help() {
    std::cerr << "usage: bigcrush [--bitrev] seeds...\n"
                 "    seeds consists of 4 unsigned 32bit integers, with default value 0\n";
}

std::optional<uint32_t> parse_seed(std::string_view arg) {
    uint32_t v;
    auto [ptr, ec] = ([&]() {
        if (arg.starts_with("0x")) {
            return std::from_chars(arg.data() + 2, arg.data() + arg.size(), v, 16);
        }
        if (arg.starts_with("0o")) {
            return std::from_chars(arg.data() + 2, arg.data() + arg.size(), v, 8);
        }
        if (arg.starts_with("0b")) {
            return std::from_chars(arg.data() + 2, arg.data() + arg.size(), v, 2);
        }
        if (arg.starts_with("0d")) {
            return std::from_chars(arg.data() + 2, arg.data() + arg.size(), v, 10);
        }
        return std::from_chars(arg.data(), arg.data() + arg.size(), v, 10);
    })();
    if (ec != std::errc{} || ptr != arg.data() + arg.size()) {
        return std::nullopt;
    }
    return v;
}

}  // namespace

int main(int argc, const char **argv) {
    static bool bitrev = false;
    static std::array<uint32_t, 4> seed{};

    int argi = 1;
    int num_seed = 0;
    while (argi < argc) {
        std::string_view arg{argv[argi++]};
        if (num_seed == 0 && arg.starts_with("--")) {
            if (arg == "--bitrev") {
                bitrev = true;
                continue;
            }
        } else if (num_seed < 4) {
            if (std::optional<uint32_t> v = parse_seed(arg)) {
                seed[num_seed++] = *v;
            }
            continue;
        }
        show_help();
        return 1;
    }

    static melty2::generator gen(seed);
    auto next = []() {
        return gen();
    };
    auto next_bitrev = []() {
        uint32_t r = gen();
        uint32_t t;
        t = r & UINT32_C(0x00FF00FF);
        r = (t << 16) | (t >> 16) | (t ^ r);
        t = r & UINT32_C(0x0F0F0F0F);
        r = (t << 8) | (t >> 24) | (t ^ r);
        t = r & UINT32_C(0x33333333);
        r = (t << 4) | (t >> 28) | (t ^ r);
        t = r & UINT32_C(0x55555555);
        r = (t << 2) | (t >> 30) | (t ^ r);
        r = (r << 1) | (r >> 31);
        return r;
    };

    char name[] = "melty2";
    unif01_Gen *ugen = unif01_CreateExternGenBits(name, bitrev ? next_bitrev : next);
    bbattery_BigCrush(ugen);
    unif01_DeleteExternGenBits(ugen);

    return 0;
}
