# SPDX-License-Identifier: Apache-2.0
"""Collect the candidate structures from the logs of sa, for rank.

sa prints the best structure so far every 100 iterations to stderr ("it=N cur=F best=F structure"),
and the final best structure on the second line of stdout. This script reads the given files (e.g.
logs/*.err and logs/*.out of sactl), and prints every distinct structure once as "name<TAB>structure",
where the name is the file name without the extension and the iteration where the structure first
appeared ("final" for stdout), e.g. "L20_gr_s601@1200".
"""
import argparse
import os

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("files", nargs="+", help="stderr (.err) and stdout (.out) files of sa")
args = parser.parse_args()

seen = set()
for path in args.files:
    job = os.path.splitext(os.path.basename(path))[0]
    with open(path) as f:
        lines = f.read().splitlines()
    found = []
    if lines and lines[0].startswith("valid="):
        if len(lines) > 1:
            found.append(("final", lines[1].strip()))
    else:
        for line in lines:
            if line.startswith("it="):
                fields = line.split(" ", 3)
                if len(fields) == 4:
                    found.append((fields[0][3:], fields[3].strip()))
    for where, structure in found:
        structure = structure.rstrip(";")
        if structure and structure not in seen:
            seen.add(structure)
            print(f"{job}@{where}\t{structure}")
