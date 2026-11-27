// SPDX-License-Identifier: Apache-2.0
//
// sa [flags] L seed iters "init" : simulated annealing of an L-op structure.
//   --nkeys N           number of keys (default 16)
//   --pairs N           counter pairs per key and delta (default 2^20)
//   --weak_words N      words given a zero or a rotation relation in the weak keys (default 6, 0 for none)
//   --keyset N          0: the keys above (default); 1: also keys with sparse v[4] and v[5]
//   --objective N       0: the sum below (default); 1: the last biased depth first (see below)
//   --checkpoint F      save the state to the file F, and resume from it if it exists (see below)
//   --pidfile F         write the process ID to the file F (see pidfile.hpp)
//   --subtract          also use subtractions (S ops): a move changes an addition, an xor or a
//                       subtraction into one of the other two
//   --reject_repeats      reject the structures repeating an addition, a subtraction or an xor (see
//                       arx::repeated in arx.hpp)
//   --common_seeds      compare a candidate with the current structure at the same seed (see below)
//   --propagate_keys      reject the structures where a key word doesn't propagate into the other words
//                       (see arx::keys_propagated in arx.hpp)
//   --freeze K          keep the first K ops of init, and move only the later ones (default 0), e.g. to
//                       extend a core found with a smaller L
//   --key_deadline K      reject the structures where a key word doesn't propagate into the other words
//                       by the first K ops, e.g. one only added near the output (see
//                       arx::keys_propagated_by in arx.hpp); 1 <= K < L (default none)
//   --rotations_only      change only the rotation amounts, keeping the ops otherwise, e.g. to tune a
//                       structure designed by hand
//
// The first L ops of init are the initial structure, and the number of rotations is kept. A move
// changes the destination or the source of an op, the rotation amount (or add <-> xor), swaps two
// ops, or moves a rotation to another op; one or two moves are applied per iteration. Invalid
// structures (see arx::valid) are rejected.
//
// The objective is the sum of log10(max(bias_max(m), T)) over the last W + 1 depths m, where T is
// twice the noise level (lower is better). It is evaluated on nkeys random keys, a quarter of which
// have a zero word and another quarter a rotation relation (weak_words > 0).
//
// With keyset = 1, the keys of index 1 mod 4 are replaced by keys whose words v[4] and v[5] are both
// sparse (zero, weight 1 or low weight), which are the most sensitive keys of the melty2 kernel.
//
// The sum may improve by reducing the bias at early depths even if the bias near the output gets
// worse, i.e. the margin gets smaller. With objective = 1, structures are compared by the last depth
// whose bias_max is clearly biased (above 5 times the noise level) first, and then by the sum: the
// objective is 100 * (the number of depths from L - W to the last biased depth) + the sum.
//
// The candidates are evaluated with the key seeds 1 to 4 in turn. By default, a candidate is compared
// with the objective of the current structure at the seed where it was accepted, so that the
// difference includes the noise of the two seeds. With --common_seeds, it is compared with the objective of the
// current structure at the same seed (common random numbers), which is computed when first needed and
// cached per seed with the bias at every depth. As the bias at a depth is determined by the ops before
// it and the seed, the depths before the first op differing from the current structure are taken from
// the cache, and only the later depths are measured; the cache isn't saved to the checkpoint, as it
// is measured again identically.
//
// With a checkpoint file, the state of the search (the iteration, the random state, and the current
// and the best structures with their objectives) is saved every 100 iterations and on SIGUSR1, by
// writing a temporary file and renaming it. If the file exists at the start, the search resumes from
// it, and continues exactly as without the interruption; the other arguments must be the same. On
// SIGTERM or SIGINT (see signal.hpp), the search stops at the end of the iteration: the best structure so far is
// logged, the state is saved (with a checkpoint file), and sa exits with the status 3; a second one quits at once.
//
// The defaults (16 keys, 2^20 pairs, weak_words = 6, keyset = 0) are those used to find the melty2
// kernel with the GPU metric; e.g. the kernel is the result of "sa 20 504 3000 ORIG22".
//
// The best structure so far is logged (LOG(INFO), to stderr) every 100 iterations.
#include <algorithm>
#include <array>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "flags.hpp"
#include "logging.hpp"
#include "metric.hpp"
#include "number.hpp"
#include "pidfile.hpp"
#include "signal.hpp"

ABSL_FLAG(int, nkeys, 16, "number of keys");
ABSL_FLAG(int, pairs, 1 << 20, "counter pairs per key and delta");
ABSL_FLAG(int, weak_words, 6, "words given a zero or a rotation relation in the weak keys (0: none)");
ABSL_FLAG(int, keyset, 0, "0: the keys of metric::random_keys; 1: also keys with sparse v[4] and v[5]");
ABSL_FLAG(int, objective, 0, "0: the sum of the biases; 1: the last biased depth first");
ABSL_FLAG(std::string, checkpoint, "", "save the state to this file, and resume from it if it exists");
ABSL_FLAG(std::string, pidfile, "", "write the process ID to this file (see pidfile.hpp)");
ABSL_FLAG(bool, subtract, false, "also use subtractions (S ops)");
ABSL_FLAG(bool, reject_repeats, false, "reject the structures repeating an addition, a subtraction or an xor");
ABSL_FLAG(bool, common_seeds, false, "compare a candidate with the current structure at the same seed");
ABSL_FLAG(bool, propagate_keys, false, "reject the structures where a key word doesn't propagate into the other words");
ABSL_FLAG(int, freeze, 0, "keep the first K ops of init, and move only the later ones");
ABSL_FLAG(int, key_deadline, 0, "reject the structures where a key word doesn't propagate into the other words by the first K ops (0: none)");
ABSL_FLAG(bool, rotations_only, false, "change only the rotation amounts");

namespace {

// the objective covers the last kW + 1 depths, L - kW to L
constexpr int kW = 10;

// bias_max at the depths L - kW to L
using Prof = std::array<double, kW + 1>;

// xorshift64: the random moves and acceptance
class XorShift64 {
  public:
    explicit XorShift64(uint64_t state = 1) : state_(state) {}
    uint64_t operator()() {
        state_ ^= state_ << 13;
        state_ ^= state_ >> 7;
        state_ ^= state_ << 17;
        return state_;
    }
    uint64_t state() const { return state_; }

  private:
    uint64_t state_;
};

// options (see the top of the file)
struct Options {
    int nk = 16, n = 1 << 20, weak = 6, keyset = 0, objmode = 0, freeze = 0, deadline = 0;
    bool subtract = false, reject_repeats = false, common = false, propagate = false, rotations_only = false;
    std::string ckpt, pidfile;  // (empty: none)
};

// the state of the search after the iteration it - 1
struct State {
    int it;
    uint64_t rng;
    arx::Arx cur, best;
    double fc, fb, f0;
};


class Search {
  public:
    explicit Search(const Options &o, uint64_t seed) : o_(o), rr_(seed), T_(2 * metric::noise(o.n)) {}

    XorShift64 &rng() { return rr_; }

    // bx[m - (L - kW)] = bias_max(m), the maximum bias over the nkeys keys of the seed with npairs
    // counter pairs, for the depths m from mfrom to L of the structure a (L - kW <= mfrom; nothing if
    // mfrom > L)
    void bias_profile(const arx::Arx &a, int nkeys, int npairs, uint64_t seed, int mfrom, Prof &bx) const {
        const int L = static_cast<int>(a.size()), nd = L - mfrom + 1;
        if (nd <= 0) return;
        const std::vector<metric::Key> keys = o_.keyset == 0 ? metric::random_keys(nkeys, seed, o_.weak)
                                                             : metric::sparse45_keys(nkeys, seed, o_.weak);
        const std::vector<double> pk = metric::bias_per_key(a, mfrom, L, keys, npairs, seed);
        for (int m = 0; m < nd; ++m) {
            double v = 0;
            for (int key = 0; key < nkeys; ++key) v = std::fmax(v, pk[key * nd + m]);
            bx[mfrom - (L - kW) + m] = v;
        }
    }

    // The objective of the bias profile bx (see bias_profile) measured with npairs counter pairs: the
    // sum of log10(max(bias_max(m), T)) over the depths m from L - kW to L, and with objective = 1, 100
    // times the number of depths up to the last biased one in addition. Lower is better; pushing the
    // biased depths earlier lowers it more than reducing the bias at the output alone.
    double score(const Prof &bx, int npairs) const {
        double f = 0;
        int last = -1;  // the last biased depth, relative to L - kW
        for (int i = 0; i <= kW; ++i) {
            f += std::log10(std::fmax(bx[i], T_));
            if (bx[i] > 5 * metric::noise(npairs)) last = i;
        }
        return o_.objmode == 1 ? 100.0 * (last + 1) + f : f;
    }

    // The objective of the structure a (see score). If prof is given, bias_max(m) is stored to
    // (*prof)[m - (L - kW)].
    double objective(const arx::Arx &a, int nkeys, int npairs, uint64_t seed, Prof *prof = nullptr) const {
        Prof bx{};
        bias_profile(a, nkeys, npairs, seed, static_cast<int>(a.size()) - kW, bx);
        if (prof) *prof = bx;
        return score(bx, npairs);
    }

    // Apply one random move to a (see the top of the file). The result may be invalid; the caller checks
    // it by arx::valid.
    void mutate(arx::Arx &a) {
        const int freeze = o_.freeze;
        if (o_.rotations_only) {
            // a random rotation after freeze gets a new amount (none if there is no such rotation)
            std::vector<int> rots;
            for (int i = freeze; i < static_cast<int>(a.size()); ++i)
                if (a[i].t == arx::kRot) rots.push_back(i);
            if (!rots.empty()) a[rots[rr_() % rots.size()]].r = static_cast<int>(1 + rr_() % 31);
            return;
        }
        const auto movable = static_cast<uint64_t>(a.size() - freeze);
        // the ops before freeze are kept
        const int kind = static_cast<int>(rr_() % 5), i = static_cast<int>(freeze + rr_() % movable);
        arx::Op &x = a[i];
        switch (kind) {
        case 0:  // change the destination
            do x.d = static_cast<int>(rr_() % 6);
            while (x.d == x.s);
            break;
        case 1:  // change the source
            do x.s = static_cast<int>(rr_() % 6);
            while (x.d == x.s);
            break;
        case 2:  // change the rotation amount, or flip add <-> xor (or change into a subtraction)
            if (x.t == arx::kRot) {
                x.r = static_cast<int>(1 + rr_() % 31);
            } else if (!o_.subtract) {
                x.t ^= 1;
            } else {
                // the types other than t
                static constexpr int kOthers[4][2] = {{arx::kXor, arx::kSub}, {arx::kAdd, arx::kSub}, {arx::kAdd, arx::kXor}, {arx::kAdd, arx::kXor}};
                x.t = kOthers[x.t][rr_() % 2];
            }
            break;
        case 3: {  // swap two ops
            const int j = static_cast<int>(freeze + rr_() % movable);
            std::swap(a[i], a[j]);
            break;
        }
        case 4: {  // move a rotation to a non-rotation op, keeping the number of rotations
            const int j = static_cast<int>(freeze + rr_() % movable);
            arx::Op &y = a[j];
            if (x.t == arx::kRot && y.t != arx::kRot) {
                y.t = arx::kRot;
                y.r = x.r;
                static constexpr int kNonRot[3] = {arx::kAdd, arx::kXor, arx::kSub};
                x.t = o_.subtract ? kNonRot[rr_() % 3] : static_cast<int>(rr_() % 2);
                x.r = 0;
            }
            break;
        }
        }
    }

  private:
    const Options &o_;
    XorShift64 rr_;
    // floor of the bias in the objective: twice the noise level, below which the bias is not resolved
    const double T_;
};

// the index of the first op where a and b differ, or their length if they are the same
int first_difference(const arx::Arx &a, const arx::Arx &b) {
    const auto p = std::mismatch(a.begin(), a.end(), b.begin(), b.end());
    return static_cast<int>(p.first - a.begin());
}

// Save the state with the arguments to path, atomically. A failure is reported, and the search goes on.
void save_checkpoint(const std::string &path, const std::string &args, const State &st) {
    const std::string tmp = path + ".tmp";
    std::ofstream f(tmp);
    if (!f) {
        PLOG(ERROR) << tmp;
        return;
    }
    f << std::format("sa-checkpoint 1\n{}\n{} {} {:.17g} {:.17g} {:.17g}\n{}\n{}\n", args, st.it, st.rng, st.fc, st.fb, st.f0,
                     arx::to_string(st.cur), arx::to_string(st.best));
    f.close();
    if (!f) {
        LOG(ERROR) << tmp << ": cannot write";
        return;
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) PLOG(ERROR) << path;
}

// Load the state from path if it exists. The arguments must match.
std::optional<State> load_checkpoint(const std::string &path, const std::string &args) {
    std::ifstream f(path);
    if (!f) return std::nullopt;
    std::string magic, saved, nums, cur, best;
    if (!std::getline(f, magic) || magic != "sa-checkpoint 1" || !std::getline(f, saved))
        throw std::runtime_error(path + ": not a checkpoint");
    if (saved != args) throw std::runtime_error(path + ": the arguments differ: " + saved);
    State st;
    // "it rng fc fb f0", as save_checkpoint writes them
    try {
        if (!std::getline(f, nums)) throw std::invalid_argument("no numbers");
        std::istringstream in(nums);
        std::string it, rng, fc, fb, f0, extra;
        if (!(in >> it >> rng >> fc >> fb >> f0) || in >> extra) throw std::invalid_argument("not 5 numbers");
        st.it = number::parse<int>(it);
        st.rng = number::parse<uint64_t>(rng);
        st.fc = number::parse<double>(fc);
        st.fb = number::parse<double>(fb);
        st.f0 = number::parse<double>(f0);
    } catch (const std::invalid_argument &) {
        throw std::runtime_error(path + ": broken checkpoint");
    }
    if (!std::getline(f, cur) || !std::getline(f, best)) throw std::runtime_error("truncated checkpoint");
    st.cur = arx::parse(cur);
    st.best = arx::parse(best);
    return st;
}

int run(int argc, char **argv) {
    const std::vector<std::string_view> arg = parse_flags(argc, argv,
                                                "sa [flags] L seed iters init: simulated annealing of an L-op structure (init may be ORIG22 for the "
                                                "original 22-step kernel)");
    if (arg.size() != 4) usage();
    Options o;
    o.nk = absl::GetFlag(FLAGS_nkeys);
    o.n = absl::GetFlag(FLAGS_pairs);
    o.weak = absl::GetFlag(FLAGS_weak_words);
    o.keyset = absl::GetFlag(FLAGS_keyset);
    o.objmode = absl::GetFlag(FLAGS_objective);
    o.ckpt = absl::GetFlag(FLAGS_checkpoint);
    o.pidfile = absl::GetFlag(FLAGS_pidfile);
    o.subtract = absl::GetFlag(FLAGS_subtract);
    o.reject_repeats = absl::GetFlag(FLAGS_reject_repeats);
    o.common = absl::GetFlag(FLAGS_common_seeds);
    o.propagate = absl::GetFlag(FLAGS_propagate_keys);
    o.freeze = absl::GetFlag(FLAGS_freeze);
    o.deadline = absl::GetFlag(FLAGS_key_deadline);
    o.rotations_only = absl::GetFlag(FLAGS_rotations_only);
    const int L = number::parse<int>(arg[0]);
    const int iters = number::parse<int>(arg[2]);
    arx::Arx cur = arx::parse(arg[3] == "ORIG22" ? arx::kMelty2Orig22 : arg[3]);
    if (L <= kW || L > static_cast<int>(cur.size())) {
        LOG(ERROR) << "L must satisfy " << kW << " < L <= (ops of init)";
        return 1;
    }
    if (o.freeze < 0 || o.freeze >= L) {
        LOG(ERROR) << "freeze must satisfy 0 <= freeze < L";
        return 1;
    }
    if (o.deadline < 0 || o.deadline >= L) {
        LOG(ERROR) << "the key deadline must satisfy 1 <= deadline < L";
        return 1;
    }
    cur.resize(L);
    Search search(o, number::parse<uint64_t>(arg[1]) * UINT32_C(2654435761) + 1);
    XorShift64 &rr = search.rng();
    if (!o.pidfile.empty()) pidfile::write(o.pidfile);

    // cur: the current structure and its objective fc; best: the best one so far and fb. The
    // candidates are evaluated with the key seeds 1 to 4 in turn, so that the search doesn't overfit
    // a fixed set of keys. A worse candidate is accepted with the probability exp(-(f - fc) / temp),
    // where the temperature decreases linearly from 0.52 to 0.02.
    // args: the arguments which a checkpoint must match
    std::string args = "L=" + std::to_string(L) + " seed=" + std::string(arg[1]) + " iters=" + std::to_string(iters) +
                       " k=" + std::to_string(o.nk) + " n=" + std::to_string(o.n) + " w=" + std::to_string(o.weak) +
                       " K=" + std::to_string(o.keyset) + " O=" + std::to_string(o.objmode);
    if (o.subtract) args += " S=1";
    if (o.reject_repeats) args += " R=1";
    if (o.common) args += " C=1";
    if (o.propagate) args += " P=1";
    if (o.freeze) args += " F=" + std::to_string(o.freeze);
    if (o.deadline) args += " D=" + std::to_string(o.deadline);
    if (o.rotations_only) args += " T=1";
    double fc, fb, f0;
    arx::Arx best;
    int it0 = 0;
    if (const std::optional<State> st = !o.ckpt.empty() ? load_checkpoint(o.ckpt, args) : std::nullopt) {
        it0 = st->it;
        rr = XorShift64(st->rng);
        cur = st->cur;
        best = st->best;
        fc = st->fc;
        fb = st->fb;
        f0 = st->f0;
        LOG(INFO) << "resumed from " << o.ckpt << " at it=" << it0;
    } else {
        fc = f0 = fb = search.objective(cur, o.nk, o.n, 1);
        best = cur;
    }
    // with -C: the bias profiles of cur at the seeds 1 to 4, and whether each is measured
    std::array<Prof, 4> cache{};
    std::array<bool, 4> cached{};
    for (int it = it0; it < iters; ++it) {
        const double temp = 0.5 * (1.0 - static_cast<double>(it) / iters) + 0.02;
        arx::Arx cand = cur;
        const int nm = static_cast<int>(1 + rr() % 2);
        for (int q = 0; q < nm; ++q) search.mutate(cand);
        const bool ok = arx::valid(cand) && !(o.reject_repeats && arx::repeated(cand)) &&
                        !(o.propagate && !arx::keys_propagated(cand)) &&
                        !(o.deadline && !arx::keys_propagated_by(cand, o.deadline));
        if (ok) {
            const int si = it % 4;
            double f;
            Prof bx{};
            if (o.common) {
                if (!cached[si]) {
                    search.bias_profile(cur, o.nk, o.n, 1 + si, L - kW, cache[si]);
                    cached[si] = true;
                }
                // the depths up to the first differing op are the same as those of cur
                const int k = first_difference(cand, cur), mfrom = std::max(k + 1, L - kW);
                bx = cache[si];
                search.bias_profile(cand, o.nk, o.n, 1 + si, mfrom, bx);
                f = search.score(bx, o.n);
                fc = search.score(cache[si], o.n);
            } else {
                f = search.objective(cand, o.nk, o.n, 1 + si);
            }
            if (f < fc || std::exp((fc - f) / temp) > static_cast<double>(rr() % 1000000) / 1e6) {
                cur = cand;
                fc = f;
                if (o.common) {
                    cache[si] = bx;
                    for (int i = 0; i < 4; ++i) cached[i] = i == si;
                }
            }
            if (fc < fb) {
                best = cur;
                fb = fc;
            }
        }
        // SIGUSR1: save the checkpoint; SIGINT or SIGTERM: save it and exit (see signal.hpp)
        const bool usr1 = signals::take(SIGUSR1), stop = signals::stop_requested();
        if ((it + 1) % 100 == 0 || usr1 || stop) {
            LOG(INFO) << std::format("it={} cur={:.2f} best={:.2f} {}", it + 1, fc, fb, arx::to_string(best));
            if (!o.ckpt.empty()) save_checkpoint(o.ckpt, args, State{it + 1, rr.state(), cur, best, fc, fb, f0});
            if (stop) {
                LOG(WARNING) << "interrupted at it=" << it + 1;
                std::exit(signals::kInterrupted);
            }
        }
    }

    // verify the best structure with more keys and pairs
    Prof prof{};
    const double fv = search.objective(best, 2 * o.nk, 4 * o.n, 12345, &prof);
    const arx::Cost c = arx::cost(best);
    std::cout << std::format("valid={:d} L={} R={} cost={}/{} init={:.2f} best={:.2f} verify={:.2f} prof(d{}..{}):",
                             arx::valid(best), best.size(), c.nrot, c.rot1, c.rot3, f0, fb, fv, L - kW, L);
    for (const double v : prof) std::cout << std::format(" {:.1e}", v);
    std::cout << std::format("\n  {}\n", arx::to_string(best));
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    init_logging();
    signals::watch({SIGINT, SIGTERM, SIGUSR1});
    try {
        return run(argc, argv);
    } catch (const std::exception &e) {
        LOG(ERROR) << e.what();
        return 1;
    }
}
