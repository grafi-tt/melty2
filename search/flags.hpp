// SPDX-License-Identifier: Apache-2.0
//
// The command lines of the search tools: abseil's flags (ABSL_FLAG in the tool, "--name=value" or "--name value",
// a bool flag also "--name" and "--noname"; --help lists them), and the positional arguments.
//
// parse_flags() parses the flags of argv (an unknown or malformed flag is an error: abseil prints it and exits),
// and returns the positional arguments, without the program name; flags and arguments may come in any order, and
// "--" ends the flags. usage() prints the usage message and exits with 1, for wrong arguments.
#ifndef FLAGS_HPP_
#define FLAGS_HPP_

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/flags/usage_config.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"

// the positional arguments of the command line, after its flags are parsed; usage is "tool [flags] args...: what"
inline std::vector<char *> parse_flags(int argc, char **argv, const char *usage) {
    absl::SetProgramUsageMessage(usage);
    // --help lists the flags of the tools (by default only those of a file named as the program, e.g. sa_cpu.cc),
    // not those of abseil (--helpfull)
    absl::FlagsUsageConfig config;
    config.contains_help_flags = config.contains_helpshort_flags = [](absl::string_view file) { return !absl::StrContains(file, "absl/"); };
    absl::SetFlagsUsageConfig(config);
    std::vector<char *> args = absl::ParseCommandLine(argc, argv);
    args.erase(args.begin());
    return args;
}

[[noreturn]] inline void usage() {
    std::fprintf(stderr, "usage: %s\n(--help for the flags)\n", std::string(absl::ProgramUsageMessage()).c_str());
    std::exit(1);
}

#endif  // FLAGS_HPP_
