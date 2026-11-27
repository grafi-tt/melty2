// SPDX-License-Identifier: Apache-2.0
//
// The watcher thread of the signals (see signal.hpp).
#include "signal.hpp"

#include <pthread.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <thread>
#include <utility>

#include "logging.hpp"

namespace signals {

namespace {

constexpr int kMaxSignal = 64;

bool is_stop(int sig) { return sig == SIGINT || sig == SIGTERM; }

// the flags set by the watcher thread, polled by the tool
std::atomic<bool> stop{false};
std::array<std::atomic<bool>, kMaxSignal + 1> taken{};

void default_callback(int sig, int count) {
    if (!is_stop(sig)) LOG(INFO) << name(sig) << " taken";
    else if (count == 1) LOG(WARNING) << name(sig) << ": stopping at the next safe point (once more to quit at once)";
    else LOG(WARNING) << name(sig) << " again: quitting at once";
}

// terminate the process by the default action of the stop signal sig, from the watcher thread (where it is blocked)
[[noreturn]] void terminate_by(int sig) {
    std::signal(sig, SIG_DFL);
    sigset_t one;
    sigemptyset(&one);
    sigaddset(&one, sig);
    pthread_sigmask(SIG_UNBLOCK, &one, nullptr);
    raise(sig);  // (to this thread, where it is no longer blocked)
    std::_Exit(128 + sig);
}

class Watcher {
  public:
    Watcher(const sigset_t &set, Callback callback) : set_(set), callback_(std::move(callback)), thread_([this] { loop(); }) {}
    // joined at exit (by the static destruction), before the callback is destroyed
    ~Watcher() {
        quit_ = true;
        thread_.join();
    }

  private:
    void loop() {
        std::array<int, kMaxSignal + 1> count{};
        while (!quit_) {
            // a short timeout, to see quit_
            const timespec timeout{0, 100'000'000};
            const int sig = sigtimedwait(&set_, nullptr, &timeout);
            if (sig < 0) {
                if (errno != EAGAIN && errno != EINTR) PLOG(ERROR) << "sigtimedwait";
                continue;
            }
            const int c = ++count[sig];
            if (is_stop(sig)) stop = true;
            else taken[sig] = true;
            callback_(sig, c);
            if (is_stop(sig) && count[SIGINT] + count[SIGTERM] >= 2) terminate_by(sig);
        }
    }

    const sigset_t set_;
    const Callback callback_;
    std::atomic<bool> quit_{false};
    std::thread thread_;  // (last: started once the members above are made)
};

}  // namespace

void watch(std::initializer_list<int> sigs, Callback callback) {
    static std::atomic<bool> started{false};
    CHECK(!started.exchange(true)) << "signals::watch() is called once per process";
    // the other threads would not block the signals
    const auto threads = std::distance(std::filesystem::directory_iterator("/proc/self/task"), std::filesystem::directory_iterator());
    CHECK_EQ(threads, 1) << "signals::watch() must be called before any thread is created";
    sigset_t set;
    sigemptyset(&set);
    for (const int sig : sigs) {
        CHECK(sig > 0 && sig <= kMaxSignal) << "signal " << sig;
        sigaddset(&set, sig);
    }
    const int e = pthread_sigmask(SIG_BLOCK, &set, nullptr);
    CHECK_EQ(e, 0) << "pthread_sigmask: " << std::strerror(e);
    static Watcher watcher(set, callback ? std::move(callback) : Callback(default_callback));
}

bool stop_requested() { return stop; }

void throw_if_stopped() {
    if (stop) throw Interrupted{};
}

bool take(int sig) { return sig > 0 && sig <= kMaxSignal && taken[sig].exchange(false); }

const char *name(int sig) {
    switch (sig) {
        case SIGINT: return "SIGINT";
        case SIGTERM: return "SIGTERM";
        case SIGHUP: return "SIGHUP";
        case SIGQUIT: return "SIGQUIT";
        case SIGUSR1: return "SIGUSR1";
        case SIGUSR2: return "SIGUSR2";
        default: return "a signal";
    }
}

}  // namespace signals
