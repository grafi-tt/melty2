# SPDX-License-Identifier: MIT
# gen_weak.py {weak32|weak48|sweep} : print a list of key modification specs (see keymod.h).
#   weak32: structures appearing with a probability around 2^-32 (a zero, all-one or weight-1 word,
#           equal, negated, MSB-flipped or rotated word pairs).
#   weak48: structures with probabilities around 2^-48 to 2^-38 (zero runs over words, zero
#           half-words in three words, low weight word pairs, near-equal pairs, small offsets, ...).
#   sweep:  parameterized classes of a single word (weight, zero top or bottom bits, periodicity,
#           runs) and of word pairs (xor with a low weight mask, rotation, offsets, negation,
#           complement, both of low weight or with a zero half-word); with "sweep classes", the class
#           of every spec is printed as tab-separated values.
import itertools
import random
import sys


def weak32():
    out = []
    for i in range(6):
        out += [f"{i}=z", f"{i}=o"] + [f"{i}=b{b}" for b in range(32)]
    for i, j in itertools.combinations(range(6), 2):
        out += [f"{i}=e{j}", f"{i}=n{j}", f"{i}=m{j}"] + [f"{i}=r{j}:{r}" for r in range(1, 32)]
    return out


def weak48():
    R = random.Random(48)
    out = []
    H = ['0xffff0000', '0x0000ffff']  # and-masks: zero low or high half
    # 48bit zero / all-one runs (a word and a half of another)
    for i in range(6):
        for j in range(6):
            if i == j:
                continue
            for m in H:
                out.append(f"{i}=z,{j}=a{m}")
            for m in ['0x0000ffff', '0xffff0000']:
                out.append(f"{i}=o,{j}=q{m}")
    # three zero half-words in three distinct words
    for _ in range(60):
        a, b, c = R.sample(range(6), 3)
        out.append(",".join(f"{w}=a{R.choice(H)}" for w in (a, b, c)))

    # two words of Hamming weight 2
    def w2():
        x, y = R.sample(range(32), 2)
        return hex(1 << x | 1 << y)
    for _ in range(60):
        i, j = R.sample(range(6), 2)
        out.append(f"{i}=h{w2()},{j}=h{w2()}")
    # a weight-1 word and a zero half-word in another
    for _ in range(60):
        i, j = R.sample(range(6), 2)
        out.append(f"{i}=b{R.randrange(32)},{j}=a{R.choice(H)}")
    # a 32bit relation and a zero half-word
    for _ in range(90):
        i, j, l = R.sample(range(6), 3)
        rel = R.choice(['e', 'n', 'm', 'r'])
        s = f"{i}=r{j}:{R.randrange(1, 32)}" if rel == 'r' else f"{i}={rel}{j}"
        out.append(f"{s},{l}=a{R.choice(H)}")
    # two near-equalities (differing in one bit, possibly rotated)
    for _ in range(60):
        i, j, l, m = R.sample(range(6), 4)
        r1 = f"{i}=r{j}:{R.randrange(0, 32)}" if R.random() < .5 else f"{i}=e{j}"
        r2 = f"{l}=r{m}:{R.randrange(0, 32)}" if R.random() < .5 else f"{l}=e{m}"
        out.append(f"{r1},{i}=x{hex(1 << R.randrange(32))},{r2},{l}=x{hex(1 << R.randrange(32))}")
    # two small additive offsets (|C| < 256), including the counter-coupled pairs (0, 3) and (1, 2)
    for _ in range(60):
        if R.random() < .5:
            p1, p2 = (0, 3), (1, 2)
        else:
            p1 = tuple(R.sample(range(6), 2))
            p2 = tuple(R.sample([w for w in range(6) if w not in p1], 2))
        c1, c2 = R.randrange(-255, 256), R.randrange(-255, 256)
        out.append(f"{p1[0]}=d{p1[1]}:{c1},{p2[0]}=d{p2[1]}:{c2}")
    # two words with three zero bytes
    for _ in range(60):
        i, j = R.sample(range(6), 2)
        out.append(f"{i}=h{hex(R.randrange(1, 256) << 8 * R.randrange(4))},{j}=h{hex(R.randrange(1, 256) << 8 * R.randrange(4))}")
    return out


def sweep():
    R = random.Random(2026)
    out = []

    def wv(w):
        v = 0
        for b in R.sample(range(32), w):
            v |= 1 << b
        return v

    def H(v):
        return hex(v & 0xffffffff)
    for i in range(6):
        for w in [0, 1, 2, 3, 4, 6, 8]:
            for _ in range(3):
                out.append((f"{i}=h{H(wv(w))}", f"pop{w}@{i}"))
        for w in [1, 2, 4]:
            for _ in range(2):
                out.append((f"{i}=h{H(~wv(w))}", f"pop{32 - w}@{i}"))
        for b in [4, 8, 12, 16, 20, 24]:
            for _ in range(2):
                out.append((f"{i}=a{H((1 << b) - 1)}", f"small{b}@{i}"))  # top 32 - b bits zero
            for _ in range(2):
                out.append((f"{i}=a{H(~((1 << (32 - b)) - 1))}", f"lowz{32 - b}@{i}"))  # low 32 - b bits zero
        for per in [4, 8, 16]:
            for _ in range(2):
                u = R.getrandbits(per)
                v = 0
                for s in range(0, 32, per):
                    v |= u << s
                out.append((f"{i}=h{H(v)}", f"period{per}@{i}"))
        for _ in range(4):
            L = R.randrange(4, 29)
            s = R.randrange(32)
            v = (1 << L) - 1
            v = ((v << s) | (v >> (32 - s))) & 0xffffffff
            out.append((f"{i}=h{H(v)}", f"run@{i}"))
    for i, j in [(i, j) for i in range(6) for j in range(6) if i != j]:
        for w in [0, 1, 2, 4]:
            out.append((f"{i}=e{j},{i}=x{H(wv(w))}", f"xorw{w}@{i}{j}"))
        for w in [0, 1, 2]:
            r = R.randrange(1, 32)
            out.append((f"{i}=r{j}:{r},{i}=x{H(wv(w))}", f"rotw{w}@{i}{j}"))
        for b in [0, 4, 8, 16]:
            c = R.randrange(1 << b, 2 << b) * R.choice([-1, 1])
            out.append((f"{i}=d{j}:{c}", f"add{b}@{i}{j}"))
        for b in [0, 8]:
            c = R.randrange(1 << b, 2 << b) * R.choice([-1, 1])
            out.append((f"{i}=n{j},{i}=d{i}:{c}", f"neg{b}@{i}{j}"))
        for w in [0, 2]:
            out.append((f"{i}=e{j},{i}=x{H(~wv(w))}", f"cmpl{w}@{i}{j}"))
    for i, j in itertools.combinations(range(6), 2):
        for w in [2, 4]:
            for _ in range(2):
                out.append((f"{i}=h{H(wv(w))},{j}=h{H(wv(w))}", f"dpop{w}@{i}{j}"))
        for _ in range(4):
            out.append((f"{i}=a{R.choice(['0xffff0000', '0x0000ffff'])},{j}=a{R.choice(['0xffff0000', '0x0000ffff'])}", f"dhalf@{i}{j}"))
    return out


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else ""
    if kind == "weak32":
        print("\n".join(weak32()))
    elif kind == "weak48":
        print("\n".join(weak48()))
    elif kind == "sweep":
        seen = set()
        classes = len(sys.argv) > 2 and sys.argv[2] == "classes"
        for s, c in sweep():
            if classes:
                print(f"{s}\t{c}")
            elif s not in seen:  # duplicates (e.g. weight 0 words) are evaluated once
                seen.add(s)
                print(s)
    else:
        sys.exit("usage: gen_weak.py {weak32|weak48|sweep [classes]}")
