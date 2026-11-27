# SPDX-License-Identifier: Apache-2.0
"""Simulated annealing of a structure against the statistical tests of testu01_stest.

The objective is the largest z-score of the tests of the stages of the file PLAN (a stage per line,
"M[,M...] tests scale keylist [offset]", as testu01_stest --plan: the tests at the steps M for the keys of
the list, in both counter orders; a key spec "keymod@streams#seed" takes its own streams and seed). With
--verify, a candidate about to become the best is also scored on the stages of VERIFY (e.g. larger
samples, or more keys and streams), and its z-score is the larger of both: so the best structure always
passes both, while the walk is scored by the cheaper PLAN. Each plan is one testu01_stest --plan process.

The moves are those of sa: a new rotation amount, a new op type (A, X, R), a new destination or source
word, or a swap of two adjacent ops, among the ops from FREEZE to MOVES (default the smallest M of PLAN;
the later ops are kept); with --sub, a new op type may also be S (a subtraction, as sa -S). With --alt N,
the first N ops alternate R and A (or S with --sub), from an R: their moves keep their kind. With
--max-sub K, a candidate has at most K S ops (S is for sparse keys; many make a structure obscure). A candidate
must be arx::valid, propagate every key word by DEADLINE ops and repeat no op (as sa -R -P -D; checked
here as in arx.hpp).

The Metropolis test draws the threshold first and passes it to testu01_stest, which stops the tests of a
candidate once its z-score is above it: so a candidate failing badly costs only its first failing tests.
Order the stages (and the keys of their lists) to fail early: the cheap and discriminating ones first.

Every scored structure is logged to LOG ("plan<TAB>z<TAB>stopped<TAB>structure"); the progress and the best
structures are printed to stdout ("it N best Z<TAB>structure", "end Z<TAB>structure").
"""
import argparse
import math
import os
import random
import subprocess
import sys

DEFAULT_STEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build", "testu01_stest")


def parse(st):
    ops = []
    for s in st.split(";"):
        f = s.split()
        if not f:
            continue
        ops.append([f[0][0], int(f[0][1:]), int(f[1]), int(f[2]) if len(f) > 2 else 0])
    return ops


def fmt(ops):
    return ";".join(f"{t}{d} {s} {r}" if t == "R" else f"{t}{d} {s}" for t, d, s, r in ops)


def valid(ops):
    """arx::valid: every word reaches the output, and the output mixes every word by an addition."""
    need = [False] * 6
    need[ops[-1][1]] = True
    for t, d, s, r in reversed(ops):
        if not need[d]:
            return False
        need[s] = True
    if not all(need):
        return False
    dep = [1 << j for j in range(6)]
    mixed = [0] * 6
    for t, d, s, r in ops:
        nd = dep[d] | dep[s]
        nm = mixed[d] | mixed[s]
        if t in "AS":
            nm |= nd
        dep[d], mixed[d] = nd, nm
    return mixed[ops[-1][1]] == 63


def propagated_by(ops, k):
    """arx::keys_propagated_by: every word is read (directly or through others) by the first k ops."""
    dep = [1 << j for j in range(6)]
    read = 0
    for t, d, s, r in ops[:k]:
        read |= dep[s]
        dep[d] |= dep[s]
    return read == 63


def repeated(ops):
    """arx::repeated: an additive (A, S) or X op repeats the last op on its destination (A and S being of
    a kind), its source unchanged since."""
    def kind(t):
        return "A" if t == "S" else t

    for j in range(1, len(ops)):
        tb, db, sb, _ = ops[j]
        if tb not in "AXS":
            continue
        for i in range(j - 1, -1, -1):
            ti, di, si, _ = ops[i]
            if di == db:
                if kind(ti) == kind(tb) and si == sb:
                    return True
                break
            if di == sb:
                break
    return False


def alternates(ops, n):
    """the first n ops alternate R and an additive op (A or S), from an R"""
    return all((t == "R") == (i % 2 == 0) and t in "RAS" for i, (t, d, s, r) in enumerate(ops[:n]))


def mutate(ops, lo, hi, rnd, types="AXR", alt=0):
    c = [o[:] for o in ops]
    i = rnd.randrange(lo, hi)
    k = rnd.random()
    if k < 0.4 and c[i][0] == "R":
        c[i][3] = rnd.randrange(1, 32)
    elif k < 0.6:
        if i < alt:  # keep the kind: R stays R, an additive op may become the other one
            if c[i][0] == "R" or "S" not in types:
                return None
            c[i][0] = "S" if c[i][0] == "A" else "A"
        else:
            t = rnd.choice(types)
            c[i][0], c[i][3] = t, (rnd.randrange(1, 32) if t == "R" else 0)
    elif k < 0.75:
        c[i][1] = rnd.randrange(6)
    elif k < 0.9:
        c[i][2] = rnd.randrange(6)
    elif i + 1 < hi and i + 1 >= alt:
        c[i], c[i + 1] = c[i + 1], c[i]
    if c[i][1] == c[i][2]:
        return None
    return c


def plan_ms(plan):
    """the step counts of the stages of a plan"""
    ms = set()
    with open(plan) as f:
        for line in f:
            w = line.split()
            if w and not w[0].startswith("#"):
                ms.update(int(x) for x in w[0].split(","))
    return ms


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--t0", type=float, default=2.0, help="the initial temperature (in z-score)")
    ap.add_argument("--t1", type=float, default=0.2, help="the final temperature")
    ap.add_argument("--freeze", type=int, default=0, help="the first ops kept")
    ap.add_argument("--moves", type=int, help="the ops moved are those before this one (default the smallest M of PLAN)")
    ap.add_argument("--sub", action="store_true", help="allow S (subtraction) ops in the moves")
    ap.add_argument("--max-sub", type=int, metavar="K", help="at most K S ops in a candidate (see above)")
    ap.add_argument("--alt", type=int, default=0, metavar="N", help="the first N ops alternate R and A/S (see above)")
    ap.add_argument("--deadline", type=int, default=12)
    ap.add_argument("--verify", help="the plan scored for a candidate about to become the best (see above)")
    ap.add_argument("--verify-below", type=float, default=math.inf, metavar="Z",
                    help="verify only a candidate whose z-score is below Z (far failing ones need no verification)")
    ap.add_argument("--log", dest="logz", action="store_true", help="anneal log(1 + z) instead of z (for structures failing grossly,\n"
                    "whose z-scores span orders of magnitude; the temperatures are then in that unit)")
    ap.add_argument("--stest", default=DEFAULT_STEST, help="testu01_stest (default build/testu01_stest)")
    ap.add_argument("--pidfile", help="write the process ID to this file (removed at the end)")
    ap.add_argument("structure", help="the initial structure")
    ap.add_argument("plan", help="the stages of the objective (see above)")
    ap.add_argument("log")
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    m0 = a.moves or min(plan_ms(a.plan))
    cur = parse(a.structure)
    if not alternates(cur, a.alt):
        ap.error(f"the first {a.alt} ops of the structure do not alternate R and A/S")
    nsub = lambda ops: sum(o[0] == "S" for o in ops)
    if a.max_sub is not None and nsub(cur) > a.max_sub:
        ap.error(f"the structure has more than {a.max_sub} S ops")
    if a.pidfile:
        with open(a.pidfile, "w") as f:
            print(os.getpid(), file=f)

    def scorer(plan):
        return subprocess.Popen([a.stest, f"--plan={plan}", f"--deadline={a.deadline}"],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)

    procs = {"plan": scorer(a.plan)}
    if a.verify:
        procs["verify"] = scorer(a.verify)
    log = open(a.log, "a")
    cache = {}  # (plan, structure) -> (z, stopped): a stopped z is only a lower bound

    def run(kind, st, limit):
        z, stopped = cache.get((kind, st), (None, True))
        if z is None or (stopped and z <= limit):
            p = procs[kind]
            p.stdin.write(f"{st}\t{limit}\n")
            p.stdin.flush()
            line = p.stdout.readline()
            if not line:
                raise RuntimeError(f"the scorer {kind} exited")
            f = line.rstrip("\n").split("\t")
            z, stopped = float(f[1].split("=")[1]), f[7] == "stopped=1"
            cache[kind, st] = z, stopped
            print(f"{kind}\t{z:.3f}\t{int(stopped)}\t{st}", file=log, flush=True)
        return z

    def score(ops, limit, best):
        # the objective (z, or log(1 + z) with --log), or a lower bound of it above limit
        st = fmt(ops)
        zl = math.expm1(limit) if a.logz and limit < 700 else (math.inf if a.logz else limit)
        z = run("plan", st, zl)
        if a.verify and z <= zl and z < a.verify_below and (z < math.expm1(best) if a.logz and best < 700 else z < best):
            z = max(z, run("verify", st, zl))
        return math.log1p(z) if a.logz else z

    try:
        cz = score(cur, math.inf, math.inf)
        best, bz = cur, cz
        print(f"start {cz:.2f}", flush=True)
        for it in range(a.iters):
            temp = a.t0 * (a.t1 / a.t0) ** (it / max(1, a.iters - 1))
            cand = None
            while cand is None:
                c = mutate(cur, a.freeze, m0, rnd, "AXRS" if a.sub else "AXR", a.alt)
                if c and valid(c) and propagated_by(c, a.deadline) and not repeated(c) and alternates(c, a.alt) and \
                        (a.max_sub is None or nsub(c) <= a.max_sub):
                    cand = c
            limit = cz - temp * math.log(1 - rnd.random())  # accepted if z <= limit (Metropolis)
            z = score(cand, limit, bz)
            if z <= limit:
                cur, cz = cand, z
                if z < bz:
                    best, bz = cand, z
                    print(f"it {it} best {bz:.2f}\t{fmt(best)}", flush=True)
            if it % 100 == 0:
                print(f"it {it} T={temp:.2f} cur {cz:.2f} best {bz:.2f}", flush=True)
        print(f"end {bz:.2f}\t{fmt(best)}", flush=True)
    finally:
        for p in procs.values():
            p.stdin.close()
            p.wait()
        if a.pidfile:
            os.remove(a.pidfile)


if __name__ == "__main__":
    sys.exit(main())
