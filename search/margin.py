# SPDX-License-Identifier: Apache-2.0
"""The statistical margins of a structure by the GPU tests of testu01_stest, per group of keys.

A group is a set of key specs (as testu01_stest --keylist: "keymod[@streams][#seed]") with a target margin,
e.g. the normal keys (3), the weak-key classes of probability 2^-32 (3), about 2^-48 (2) or 2^-64 (1),
and the counter streams (3). The default 2^-48 classes are a sample of the kinds of constraints of 33 to
48 bits: three half-words zero, a word fixed or related to another and a half-word zero, two words of
weight 2, and a word of weight 1 and a half-word zero. A key fails at M steps in a counter order if a p-value of its
tests is below 1e-6, or 3 or more are outside [0.001, 0.999] (as statjudge.sh), or a test fails completely (e.g. runs
out of its counters: min_p=0), or it has no result; a group fails at M if any of its keys fails in either order. Its
margin is L - 1 - M for the largest failing M of the structure of L ops (the steps passing beyond it: e.g. 3 for an
18-op kernel passing at 15 and failing at 14).

By default only M = L - target is tested for each group: the verdict is "pass" (margin >= target) or the
failure. With --scan, the steps from L - 1 down to L - DOWN are tested until the group fails, for its
exact margin (or "> DOWN").

The tests at every M are the stages of a testu01_stest --plan: TESTS at the scale SCALE, and LONG_TESTS
(Permutation and Run, whose biases depend on the counter range) at LONG_SCALE (4 reproduces BigCrush). The groups
are those of --groups (a file of lines "name target spec spec ...") or the defaults below.
"""
import argparse
import collections
import os
import subprocess
import sys
import tempfile

DEFAULT_STEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build", "testu01_stest")

DEFAULT_GROUPS = [
    ("normal", 3, ["-", "-#2"]),
    ("weak32", 3, ["3=z", "3=n0", "3=b31", "3=r0:1", "1=z", "1=n2", "0=z", "2=z", "4=z", "5=z"]),
    ("weak48", 2, ["4=z,5=a0x0000ffff", "4=z,5=a0xffff0000", "5=z,4=a0x0000ffff", "3=z,1=a0x0000ffff",
                   "3=z,2=a0xffff0000", "3=o,4=a0x0000ffff", "1=e0,3=a0x0000ffff", "3=n0,1=a0xffff0000",
                   "3=a0x0000ffff,4=a0x0000ffff,5=a0x0000ffff", "0=a0x0000ffff,3=a0x0000ffff,4=a0xffff0000",
                   "4=h0x00010001,5=h0x80000001", "3=b31,4=a0x0000ffff", "4=b0,5=a0xffff0000"]),
    ("weak64", 1, ["4=z,5=z", "3=z,1=z"]),
    ("streams", 3, [f"-@ctr:{c}#{s}" for c in ("8:48", "4:56", "4:48", "1:63") for s in (1, 2, 3, 4)]),
]


def read_groups(path):
    groups = []
    with open(path) as f:
        for line in f:
            w = line.split()
            if w and not w[0].startswith("#"):
                groups.append((w[0], int(w[1]), w[2:]))
    return groups


def count_ops(st):
    return sum(1 for o in st.split(";") if o.strip())


def norm(spec, seed=1):
    """a key spec as (keymod, streams, seed)"""
    spec, _, sd = spec.partition("#")
    km, _, streams = spec.partition("@")
    return km, streams or "-", int(sd) if sd else seed


def run_step(a, st, m, groups, tmp):
    """the results of the keys of the groups at m steps: {(group, spec, swap): (min_p, flagged)}"""
    lines = []
    for name, _, specs in groups:
        keys = os.path.join(tmp, f"{name}.lst")
        with open(keys, "w") as f:
            f.write("\n".join(specs) + "\n")
        if a.tests:
            lines.append(f"{m} {a.tests} {a.scale} {keys} {a.offset}")
        if a.long_scale > 0:
            lines.append(f"{m} {a.long_tests} {a.long_scale} {keys} {a.offset}")
    plan = os.path.join(tmp, "plan")
    with open(plan, "w") as f:
        f.write("\n".join(lines) + "\n")
    # --log_sum (without a limit): a key whose test fails completely gives min_p=0, and the other keys are still tested
    # (without it, testu01_stest stops the structure at such a key, printing no line for it and the keys after it)
    out = subprocess.run([a.stest, "--verbose", "--log_sum", f"--plan={plan}"], input=st + "\n", capture_output=True, text=True,
                         check=True).stdout
    res = {}
    for line in out.splitlines():
        if not line.startswith("key\t"):
            continue
        f = dict(x.split("=", 1) for x in line.split("\t")[1:])
        k = (norm(f["key"]), int(f["swap"]))
        p, fl = res.get(k, (1.0, 0))
        res[k] = (min(p, float(f["min_p"])), fl + int(f["flagged"]))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, help="L, the ops of the kernel (default those of the structure)")
    ap.add_argument("--scan", action="store_true", help="find the exact margins (see above)")
    ap.add_argument("--down", type=int, default=5, help="the deepest step count of --scan, L - DOWN")
    ap.add_argument("--groups", help="a file of groups (see above)")
    ap.add_argument("--only", help="only these groups (comma-separated names)")
    ap.add_argument("--tests", default="poker,coupon,gap,maxoft,collision,birthday,fourier", help="the tests at SCALE")
    ap.add_argument("--scale", type=float, default=1)
    ap.add_argument("--long-tests", default="permutation,run", help="the tests at LONG_SCALE")
    ap.add_argument("--long-scale", type=float, default=4, help="the scale of LONG_TESTS (0: none)")
    ap.add_argument("--offset", default="0", help="the first counter of the tests (testu01_stest --offset)")
    ap.add_argument("--stest", default=DEFAULT_STEST, help="testu01_stest (default build/testu01_stest)")
    ap.add_argument("structure")
    a = ap.parse_args()
    st = a.structure.strip().rstrip(";")
    L = a.steps or count_ops(st)
    groups = read_groups(a.groups) if a.groups else DEFAULT_GROUPS
    if a.only:
        groups = [g for g in groups if g[0] in a.only.split(",")]
    result = {}  # group -> (margin text, failing M, worst key text)
    pending = {g[0]: g for g in groups}
    steps = collections.defaultdict(list)  # M -> the groups tested at M
    if a.scan:
        order = list(range(L - 1, L - a.down - 1, -1))
    else:
        for g in groups:
            steps[L - g[1]].append(g)
        order = sorted(steps, reverse=True)
    with tempfile.TemporaryDirectory() as tmp:
        for m in order:
            gs = [pending[n] for n in pending] if a.scan else steps[m]
            gs = [g for g in gs if g[0] in pending]
            if not gs or m < 1:
                continue
            res = run_step(a, st, m, gs, tmp)
            for name, target, specs in gs:
                keys = {norm(x) for x in specs}
                mine = [(p, fl, f"{k[0]}@{k[1]}#{k[2]}", sw) for (k, sw), (p, fl) in res.items() if k in keys]
                bad = [x for x in mine if x[0] < 1e-6 or x[1] >= 3]
                # a key (and counter order) without a result fails: its tests stopped
                bad += [(0.0, 0, f"{k[0]}@{k[1]}#{k[2]}", sw) for k in sorted(keys) for sw in (0, 1) if (k, sw) not in res]
                if bad:
                    p, fl, spec, sw = min(bad)
                    what = "no result" if not any(x[2] == spec and x[3] == sw for x in mine) else f"min_p={p:.3g} flagged={fl}"
                    result[name] = (L - 1 - m, m, f"{spec} swap={sw} {what}")
                    del pending[name]
                elif not a.scan:
                    p, fl, spec, sw = min(mine)
                    result[name] = (None, m, f"{spec} swap={sw} min_p={p:.3g} flagged={fl}")
                    del pending[name]
            print(f"M={m} done: {', '.join(g[0] for g in gs)}", file=sys.stderr, flush=True)
    print(f"structure\t{st}\nsteps\t{L}")
    for name, target, _ in groups:
        if name in result:
            margin, m, text = result[name]
            if margin is None:
                print(f"{name}\ttarget={target}\tmargin>={target}\tpass at {m}\tworst {text}")
            else:
                verdict = "ok" if margin >= target else "BELOW TARGET"
                print(f"{name}\ttarget={target}\tmargin={margin}\t{verdict}\tfails at {m}: {text}")
        else:
            print(f"{name}\ttarget={target}\tmargin>={a.down}\tok\tno failure down to {L - a.down}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
