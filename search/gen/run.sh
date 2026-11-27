#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# run.sh TOOL "structure" M args... : build TOOL (gen, genlow, genpair, fperm or bcsub) with the first M
# ops of the structure compiled in, and run it with the arguments args...
# Builds are cached in ../build/gen/, and rebuilt when the source of the tool, the headers it uses or
# mkkern.py is newer. The tools are C++20 (CXX, CXXFLAGS), as the abseil of their logging and flags (found by pkg-config,
# or ABSL_CFLAGS and ABSL_LIBS), and melty2_init.c of the library is compiled as C (CC, CFLAGS); LDFLAGS is
# added to the link, e.g. for the location of TestU01.
# Concurrent runs may build the same structure: every run writes its own temporary files and renames
# them, so that no run sees a partial file.
# The environment variable MELTY2_KEYMOD modifies the key (see keymod.hpp).
set -e
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
tool=$1; shift
st=$1; M=$2; shift 2
h=$(printf '%s|%s' "$st" "$M" | sha1sum | cut -c1-16)
d=$here/../build/gen/$h
if [ ! -f "$d/kern.hpp" ] || [ "$here/mkkern.py" -nt "$d/kern.hpp" ]; then
    mkdir -p "$d"
    if ! python3 "$here/mkkern.py" "$st" "$M" > "$d/kern.hpp.tmp.$$"; then
        rm -f "$d/kern.hpp.tmp.$$"
        exit 1
    fi
    mv "$d/kern.hpp.tmp.$$" "$d/kern.hpp"
fi
stale=0
for f in "$here/$tool.cc" "$here/../keymod.hpp" "$here/../arx.hpp" "$here/../logging.hpp" "$here/../flags.hpp" "$d/kern.hpp"; do
    [ "$f" -nt "$d/$tool" ] && stale=1
done
if [ ! -x "$d/$tool" ] || [ $stale = 1 ]; then
    libs=""
    [ "$tool" = bcsub ] && libs="-ltestu01 -lprobdist -lmylib"
    absl="absl_log absl_check absl_log_initialize absl_log_globals absl_flags absl_flags_parse absl_flags_usage"
    absl_cflags=${ABSL_CFLAGS-$(pkg-config --cflags $absl)}
    absl_libs=${ABSL_LIBS-$(pkg-config --libs $absl)}
    ${CC:-cc} -std=c99 -O3 -march=native ${CFLAGS} -I"$root/include" -c "$root/src/melty2_init.c" -o "$d/melty2_init.o.$$"
    if ! ${CXX:-c++} -std=c++20 -O3 -march=native ${CXXFLAGS} $absl_cflags -I"$d" -I"$here/.." -I"$root/include" \
        -o "$d/$tool.tmp.$$" "$here/$tool.cc" -x none "$d/melty2_init.o.$$" ${LDFLAGS} $libs $absl_libs; then
        rm -f "$d/melty2_init.o.$$"
        exit 1
    fi
    rm -f "$d/melty2_init.o.$$"
    mv "$d/$tool.tmp.$$" "$d/$tool"
fi
exec "$d/$tool" "$@"
