// SPDX-License-Identifier: Apache-2.0
//
// pid files of the search tools, for managing them (e.g. by a job queue).
//
// pidfile::write(path) writes the process ID to path, and removes the file at exit (by std::exit() or
// the return from main). A file left by a killed process is detected by kill(pid, 0) of its reader.
#pragma once

#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>

namespace pidfile {

inline std::string &path_() {
    static std::string path;
    return path;
}

inline void write(const std::string &path) {
    std::ofstream f(path);
    f << getpid() << '\n';
    f.close();
    if (!f) throw std::runtime_error("cannot write " + path);
    // the static path is constructed before the handler is registered, so it is destroyed after the
    // handler runs
    path_() = path;
    std::atexit([] { unlink(path_().c_str()); });
}

}  // namespace pidfile
