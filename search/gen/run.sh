#!/bin/sh
# SPDX-License-Identifier: MIT
# run.sh TOOL "structure" M args... : build TOOL (gen, genlow, genpair, fperm or bcsub) with the first M
# ops of the structure compiled in, and run it with the arguments "structure" M args...
# Builds are cached in ../build/gen/. The environment variable MELTY2_KEYMOD modifies the key (see keymod.h).
set -e
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
tool=$1; shift
st=$1; M=$2
h=$(printf '%s|%s' "$st" "$M" | sha1sum | cut -c1-16)
d=$here/../build/gen/$h
if [ ! -x "$d/$tool" ]; then
    mkdir -p "$d"
    [ -f "$d/kern.h" ] || { python3 "$here/mkkern.py" "$st" "$M" > "$d/kern.h.tmp" && mv "$d/kern.h.tmp" "$d/kern.h"; }
    libs=""
    [ "$tool" = bcsub ] && libs="-ltestu01 -lprobdist -lmylib"
    ${CC:-cc} -std=c99 -O3 -march=native ${CFLAGS} -I"$d" -I"$here/.." -I"$root/include" -o "$d/$tool.tmp" \
        "$here/$tool.c" "$root/src/melty2_init.c" ${LDFLAGS} $libs -lm
    mv "$d/$tool.tmp" "$d/$tool"
fi
exec "$d/$tool" "$@"
