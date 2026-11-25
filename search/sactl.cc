// SPDX-License-Identifier: Apache-2.0
/*
 * sactl : manage a queue of sa jobs in a directory.
 *
 *   sactl run [-s sa] [-i minutes] DIR   run the jobs of DIR/jobs.tsv in order, in the foreground
 *   sactl snapshot DIR                   let the running "sactl run" take a snapshot now
 *   sactl stop DIR                       let the running "sactl run" save the state and exit
 *   sactl status DIR                     print the state of every job
 *
 * Layout of DIR:
 *   jobs.tsv                 a job per line: the name, then the arguments of sa, separated by tabs
 *                            (empty lines and lines starting with # are skipped)
 *   bin/sa                   the copy of sa used for the jobs (see below)
 *   sactl.pid                the process ID of "sactl run", while it runs
 *   sactl.log                the log of "sactl run"
 *   logs/NAME.out            the result of sa (stdout); a job is done when it has a line "valid=..."
 *   logs/NAME.err            the progress of sa (stderr), appended across resumptions
 *   logs/NAME.ckpt           the checkpoint of sa (sa -c)
 *   logs/NAME.pid            the process ID of sa (sa -p), while it runs
 *   snapshots/STAMP.tar.gz   the snapshots of jobs.tsv, sactl.log and logs/
 *
 * "sactl run" copies sa (-s, default sa_gpu next to sactl) to DIR/bin/sa unless it exists, so that
 * rebuilding the tools doesn't change the running search; a resumed job continues exactly only with
 * the same binary. It runs sa for the first job not done, with -c and -p, so that an interrupted job
 * resumes from its checkpoint. A job whose sa fails is skipped until the next "sactl run".
 *
 * Every interval minutes (-i, default 10) and on SIGUSR1 ("sactl snapshot"), it lets sa save the
 * checkpoint by SIGUSR1, waits for the checkpoint to be updated, and archives the directory into
 * snapshots/. On SIGTERM or SIGINT ("sactl stop"), it lets sa save the checkpoint and exit by SIGTERM,
 * takes a snapshot, and exits; "sactl run" again resumes the queue.
 */
#include <fcntl.h>
#include <getopt.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Job {
    std::string name;
    std::vector<std::string> args;  // the arguments of sa
};

// paths of the files of a directory and its jobs
struct Layout {
    fs::path dir;
    fs::path jobs() const { return dir / "jobs.tsv"; }
    fs::path sa() const { return dir / "bin" / "sa"; }
    fs::path pid() const { return dir / "sactl.pid"; }
    fs::path log() const { return dir / "sactl.log"; }
    fs::path logs() const { return dir / "logs"; }
    fs::path snapshots() const { return dir / "snapshots"; }
    fs::path out(const Job &j) const { return logs() / (j.name + ".out"); }
    fs::path err(const Job &j) const { return logs() / (j.name + ".err"); }
    fs::path ckpt(const Job &j) const { return logs() / (j.name + ".ckpt"); }
    fs::path jobpid(const Job &j) const { return logs() / (j.name + ".pid"); }
};

std::vector<std::string> split_tabs(const std::string &line) {
    std::vector<std::string> v;
    std::string field;
    std::istringstream is(line);
    while (std::getline(is, field, '\t')) v.push_back(field);
    return v;
}

std::vector<Job> read_jobs(const Layout &l) {
    std::ifstream f(l.jobs());
    if (!f) {
        std::cerr << l.jobs().string() << ": cannot open\n";
        std::exit(1);
    }
    std::vector<Job> jobs;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::vector<std::string> v = split_tabs(line);
        if (v.size() < 2) {
            std::cerr << l.jobs().string() << ": a job needs a name and arguments: " << line << "\n";
            std::exit(1);
        }
        jobs.push_back({v[0], std::vector<std::string>(v.begin() + 1, v.end())});
    }
    return jobs;
}

// the first line of the file starting with prefix, or "" if none
std::string find_line(const fs::path &p, const std::string &prefix) {
    std::ifstream f(p);
    std::string line;
    while (std::getline(f, line))
        if (line.compare(0, prefix.size(), prefix) == 0) return line;
    return "";
}

std::string last_line(const fs::path &p) {
    std::ifstream f(p);
    std::string line, last;
    while (std::getline(f, line))
        if (!line.empty()) last = line;
    return last;
}

bool job_done(const Layout &l, const Job &j) { return !find_line(l.out(j), "valid=").empty(); }

// the process ID in the pid file if the process is alive, or 0
pid_t alive_pid(const fs::path &p) {
    std::ifstream f(p);
    long pid = 0;
    if (!(f >> pid) || pid <= 0) return 0;
    return kill(static_cast<pid_t>(pid), 0) == 0 || errno == EPERM ? static_cast<pid_t>(pid) : 0;
}

std::string stamp(const char *format) {
    std::time_t t = std::time(nullptr);
    char buf[64];
    std::strftime(buf, sizeof buf, format, std::localtime(&t));
    return buf;
}

void log(const Layout &l, const std::string &msg) {
    std::ofstream f(l.log(), std::ios::app);
    f << stamp("%F %T") << " " << msg << "\n";
    std::cerr << msg << "\n";
}

// Run the command and wait for it; return its exit status, or -1.
int run_command(const std::vector<std::string> &cmd) {
    pid_t pid = fork();
    if (pid == 0) {
        std::vector<char *> argv;
        for (const std::string &s : cmd) argv.push_back(const_cast<char *>(s.c_str()));
        argv.push_back(nullptr);
        execvp(argv[0], argv.data());
        std::perror(argv[0]);
        _exit(127);
    }
    int status;
    if (pid < 0 || waitpid(pid, &status, 0) < 0) return -1;
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

//
// sactl run
//

volatile sig_atomic_t snapshot_requested = 0, stop_requested = 0;

void on_signal(int sig) {
    if (sig == SIGUSR1) snapshot_requested = 1;
    else stop_requested = 1;
}

std::chrono::nanoseconds mtime(const fs::path &p) {
    struct stat st;
    if (stat(p.c_str(), &st) != 0) return std::chrono::nanoseconds(0);
    return std::chrono::seconds(st.st_mtim.tv_sec) + std::chrono::nanoseconds(st.st_mtim.tv_nsec);
}

// Let the running sa (if any) save the checkpoint, and archive the directory.
void take_snapshot(const Layout &l, const Job *job, pid_t child) {
    if (child > 0) {
        const auto before = mtime(l.ckpt(*job));
        kill(child, SIGUSR1);
        // an iteration takes at most a few seconds; give up after a minute
        for (int i = 0; i < 600 && mtime(l.ckpt(*job)) == before && waitpid(child, nullptr, WNOHANG) == 0; ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (mtime(l.ckpt(*job)) == before) log(l, "snapshot: the checkpoint of " + job->name + " was not updated");
    }
    fs::create_directories(l.snapshots());
    const std::string name = stamp("%Y%m%d-%H%M%S") + ".tar.gz";
    const int st = run_command({"tar", "czf", (l.snapshots() / name).string(), "-C", l.dir.string(), "jobs.tsv",
                                "sactl.log", "logs"});
    log(l, st == 0 ? "snapshot " + name : "snapshot " + name + " failed");
}

int cmd_run(const Layout &l, const fs::path &sa, int interval_min) {
    if (pid_t p = alive_pid(l.pid())) {
        std::cerr << "sactl run is already running for " << l.dir.string() << " (pid " << p << ")\n";
        return 1;
    }
    fs::create_directories(l.logs());
    if (!fs::exists(l.sa())) {
        fs::create_directories(l.sa().parent_path());
        fs::copy_file(sa, l.sa());
        log(l, "copied " + sa.string() + " to " + l.sa().string());
    }
    {
        std::ofstream f(l.pid());
        f << getpid() << "\n";
    }
    struct sigaction act;
    std::memset(&act, 0, sizeof act);
    act.sa_handler = on_signal;
    sigemptyset(&act.sa_mask);
    sigaction(SIGUSR1, &act, nullptr);
    sigaction(SIGTERM, &act, nullptr);
    sigaction(SIGINT, &act, nullptr);

    const auto interval = std::chrono::minutes(interval_min);
    auto next_snapshot = std::chrono::steady_clock::now() + interval;
    std::set<std::string> failed;
    log(l, "run started");
    take_snapshot(l, nullptr, 0);
    while (!stop_requested) {
        // the job list is read again for every job, so that jobs can be added while running
        const std::vector<Job> jobs = read_jobs(l);
        const Job *job = nullptr;
        for (const Job &j : jobs)
            if (!job_done(l, j) && !failed.count(j.name)) {
                job = &j;
                break;
            }
        if (!job) break;

        std::vector<std::string> cmd = {l.sa().string(), "-c", l.ckpt(*job).string(), "-p", l.jobpid(*job).string()};
        cmd.insert(cmd.end(), job->args.begin(), job->args.end());
        log(l, "start " + job->name + (fs::exists(l.ckpt(*job)) ? " (resuming)" : ""));
        pid_t child = fork();
        if (child == 0) {
            const int out = open(l.out(*job).c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            const int err = open(l.err(*job).c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
            if (out < 0 || err < 0) _exit(127);
            dup2(out, 1);
            dup2(err, 2);
            std::vector<char *> argv;
            for (const std::string &s : cmd) argv.push_back(const_cast<char *>(s.c_str()));
            argv.push_back(nullptr);
            execv(argv[0], argv.data());
            std::perror(argv[0]);
            _exit(127);
        }
        int status = 0;
        for (;;) {
            if (waitpid(child, &status, WNOHANG) == child) break;
            if (stop_requested) {
                kill(child, SIGTERM);  // sa saves the checkpoint and exits with the status 3
                waitpid(child, &status, 0);
                break;
            }
            if (snapshot_requested || std::chrono::steady_clock::now() >= next_snapshot) {
                snapshot_requested = 0;
                take_snapshot(l, job, child);
                next_snapshot = std::chrono::steady_clock::now() + interval;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        const int code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
        if (code == 0 && job_done(l, *job)) {
            log(l, "done " + job->name + ": " + find_line(l.out(*job), "valid=").substr(0, 100));
        } else if (code == 3 && stop_requested) {
            log(l, "interrupted " + job->name);
        } else {
            log(l, "failed " + job->name + " (status " + std::to_string(code) + "), skipped");
            failed.insert(job->name);
        }
    }
    log(l, stop_requested ? "run stopped" : "run finished");
    take_snapshot(l, nullptr, 0);
    fs::remove(l.pid());
    return 0;
}

//
// sactl snapshot / stop / status
//

int cmd_signal(const Layout &l, int sig, bool wait_exit) {
    const pid_t pid = alive_pid(l.pid());
    if (!pid) {
        std::cerr << "sactl run is not running for " << l.dir.string() << "\n";
        return 1;
    }
    kill(pid, sig);
    if (!wait_exit) {
        std::cout << "requested a snapshot (see " << l.log().string() << ")\n";
        return 0;
    }
    std::cout << "stopping sactl run (pid " << pid << ") ..." << std::flush;
    while (kill(pid, 0) == 0) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    std::cout << " stopped\n";
    return 0;
}

int cmd_status(const Layout &l) {
    if (pid_t p = alive_pid(l.pid())) std::cout << "sactl run: running (pid " << p << ")\n";
    else std::cout << "sactl run: not running\n";
    for (const Job &j : read_jobs(l)) {
        std::cout << j.name << ": ";
        if (job_done(l, j)) {
            std::cout << "done  " << find_line(l.out(j), "valid=").substr(0, 100) << "\n";
        } else if (pid_t p = alive_pid(l.jobpid(j))) {
            std::cout << "running (pid " << p << ")  " << last_line(l.err(j)).substr(0, 60) << "\n";
        } else if (fs::exists(l.ckpt(j))) {
            std::ifstream f(l.ckpt(j));
            std::string line;
            for (int i = 0; i < 3; ++i) std::getline(f, line);
            std::cout << "paused at it=" << line.substr(0, line.find(' ')) << "\n";
        } else {
            std::cout << "pending\n";
        }
    }
    return 0;
}

[[noreturn]] void usage() {
    std::cerr << "usage: sactl run [-s sa] [-i minutes] DIR\n"
                 "       sactl snapshot DIR\n"
                 "       sactl stop DIR\n"
                 "       sactl status DIR\n";
    std::exit(1);
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) usage();
    const std::string command = argv[1];
    fs::path sa = fs::path(argv[0]).parent_path() / "sa_gpu";
    int interval = 10;
    static const struct option longopts[] = {
        {"sa", required_argument, nullptr, 's'},
        {"interval", required_argument, nullptr, 'i'},
        {nullptr, 0, nullptr, 0},
    };
    optind = 2;  // the options follow the command
    int opt;
    while ((opt = getopt_long(argc, argv, "s:i:", longopts, nullptr)) != -1) {
        switch (opt) {
        case 's': sa = optarg; break;
        case 'i': interval = std::atoi(optarg); break;
        default: usage();
        }
    }
    if (argc - optind != 1 || interval < 1) usage();
    const Layout l{fs::absolute(argv[optind])};
    if (command == "run") return cmd_run(l, fs::absolute(sa), interval);
    if (command == "snapshot") return cmd_signal(l, SIGUSR1, false);
    if (command == "stop") return cmd_signal(l, SIGTERM, true);
    if (command == "status") return cmd_status(l);
    usage();
}
