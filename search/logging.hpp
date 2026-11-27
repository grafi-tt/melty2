// SPDX-License-Identifier: Apache-2.0
//
// The logging of the search tools: abseil's LOG, PLOG and CHECK, to stderr.
//
// init_logging() is called first in main: abseil writes only the errors to stderr by default (it has no log
// files), so the threshold is lowered for the informational messages (the progress of sa, the compile time of
// testu01_stest) as well. A line of LOG starts with its severity, time, thread and source location, e.g.
// "I1127 15:03:05.568861   94958 sa.cc:425] it=100 ...". The results of the tools go to stdout as before.
#pragma once

#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"

inline void init_logging() {
    absl::InitializeLog();
    absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
}
