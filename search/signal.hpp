// SPDX-License-Identifier: Apache-2.0
//
// The signals of the search tools from the user (e.g. Ctrl-C, kill, a job queue): a thread waits for them, and the
// tools poll flags at their safe points, so that the handling needs no async-signal-safe code, and works with the
// threads of the tools (and those of the HIP runtime) alike.
//
// signals::watch({SIGINT, SIGTERM, ...}, callback) blocks the signals and starts the watcher thread, which takes
// them by sigtimedwait. It must be called first in main (after init_logging), before any thread is created: the
// threads created later (by the tool or the HIP runtime) inherit the mask, and a thread not blocking a signal would
// take it by its default action. The mask is also inherited by child processes, so a tool starting one must unblock
// the signals in it (none does).
//
// A stop signal (SIGINT, SIGTERM) sets stop_requested(), and the tool stops at its next safe point, exiting with
// kInterrupted (e.g. by throw_if_stopped(), and main catching Interrupted); a second one terminates the process at
// once by its default action (for a tool slow to reach a safe point). Another signal sets a flag, which take(sig) reads and clears (e.g. SIGUSR1: save a checkpoint).
// The callback is called on the watcher thread for every signal taken, with the signal and how many times it has
// been taken (it must be thread-safe, e.g. LOG); by default it logs that the tool stops or quits. The watcher thread
// is joined at exit.
#ifndef SIGNAL_HPP_
#define SIGNAL_HPP_

#include <functional>
#include <initializer_list>

namespace signals {

// the exit status of a tool stopped by a stop signal
constexpr int kInterrupted = 3;

// the callback of the watcher thread: the signal and how many times it has been taken (1 for the first)
using Callback = std::function<void(int sig, int count)>;

// block the signals sigs and start the watcher thread (once per process); the callback by default logs what the
// signal does
void watch(std::initializer_list<int> sigs, Callback callback = {});

// whether a stop signal (SIGINT, SIGTERM) has been taken
bool stop_requested();

// thrown by throw_if_stopped() (not a std::exception, so that only main catches it)
struct Interrupted {};

// throw Interrupted if a stop signal has been taken
void throw_if_stopped();

// whether the signal sig has been taken since the last take(sig), clearing it
bool take(int sig);

// the name of the signal, e.g. "SIGINT"
const char *name(int sig);

}  // namespace signals

#endif  // SIGNAL_HPP_
