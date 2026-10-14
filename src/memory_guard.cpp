#include "aero/memory_guard.hpp"

#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/task_info.h>
#include <unistd.h>

#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace aero {

namespace {

// ---- small helpers -------------------------------------------------------

std::string lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

bool all_digits(const std::string& s) {
    if (s.empty()) return false;
    for (char c : s)
        if (!std::isdigit(static_cast<unsigned char>(c))) return false;
    return true;
}

std::string human(size_t bytes) {
    static const char* u[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double v = static_cast<double>(bytes);
    int i = 0;
    while (v >= 1024.0 && i < 4) { v /= 1024.0; ++i; }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.2f %s", v, u[i]);
    return buf;
}

}  // namespace

const char* mem_class_name(MemClass c) {
    switch (c) {
        case MemClass::WeightsResident: return "weights-resident";
        case MemClass::WeightsCache:    return "weights-cache";
        case MemClass::KVCache:         return "kv-cache";
        case MemClass::Scratch:         return "scratch";
        case MemClass::Runtime:         return "runtime";
        default:                        return "?";
    }
}

bool parse_max_ram(const std::string& text, size_t* out_bytes, std::string* error) {
    auto fail = [&](const char* msg) {
        if (error) *error = msg;
        return false;
    };
    if (!out_bytes) return fail("null out_bytes");

    std::string t;
    for (char c : text) if (!std::isspace(static_cast<unsigned char>(c))) t += c;
    if (t.empty()) return fail("empty size");

    // split leading number (digits, optional single '.') from the suffix
    size_t i = 0;
    bool seen_dot = false;
    while (i < t.size()) {
        char c = t[i];
        if (std::isdigit(static_cast<unsigned char>(c))) { ++i; continue; }
        if (c == '.' && !seen_dot) { seen_dot = true; ++i; continue; }
        break;
    }
    if (i == 0) return fail("no leading number");
    const std::string num = t.substr(0, i);
    const std::string suf = lower(t.substr(i));

    double value = 0.0;
    try {
        value = std::stod(num);
    } catch (...) {
        return fail("unparseable number");
    }
    if (value < 0) return fail("negative size");

    // Suffix table. Bare numbers mean GiB -- the colloquial "8GB of RAM".
    // K/M/G/T are powers of 1024; KB/MB/GB/TB are powers of 1000.
    double mult = 0.0;
    if (suf.empty())          mult = 1024.0 * 1024.0 * 1024.0;      // bare -> GiB
    else if (suf == "b" || suf == "bytes") mult = 1.0;
    else if (suf == "k" || suf == "kib")   mult = 1024.0;
    else if (suf == "m" || suf == "mib")   mult = 1024.0 * 1024.0;
    else if (suf == "g" || suf == "gib")   mult = 1024.0 * 1024.0 * 1024.0;
    else if (suf == "t" || suf == "tib")   mult = 1024.0 * 1024.0 * 1024.0 * 1024.0;
    else if (suf == "kb")                  mult = 1000.0;
    else if (suf == "mb")                  mult = 1000.0 * 1000.0;
    else if (suf == "gb")                  mult = 1000.0 * 1000.0 * 1000.0;
    else if (suf == "tb")                  mult = 1000.0 * 1000.0 * 1000.0 * 1000.0;
    else return fail("unknown suffix");

    // A completely bare integer (no dot, no suffix) is ambiguous. Resolve it by
    // magnitude: values below 2^20 are read as GiB (nobody caps RAM at 2048
    // BYTES), and anything larger is read as raw bytes (nobody caps RAM at
    // 8589934592 GiB). The resolved byte count is always logged, so the choice
    // is never hidden in practice.
    if (suf.empty() && !seen_dot && all_digits(num) && value >= 1048576.0) mult = 1.0;

    const double bytes = value * mult;
    if (bytes > static_cast<double>(SIZE_MAX)) return fail("size overflows size_t");
    *out_bytes = static_cast<size_t>(bytes);
    if (error) error->clear();
    return true;
}

// ---- MemoryGuard ---------------------------------------------------------

MemoryGuard& MemoryGuard::instance() {
    static MemoryGuard g;  // intentionally leaked at exit; guard outlives users
    return g;
}

MemoryGuard::~MemoryGuard() { stop_monitor(); }

void MemoryGuard::set_budget(size_t bytes) {
    std::lock_guard<std::mutex> lk(mu_);
    budget_ = bytes;
}

size_t MemoryGuard::committed() const {
    std::lock_guard<std::mutex> lk(mu_);
    size_t total = 0;
    for (size_t v : committed_) total += v;
    return total;
}

size_t MemoryGuard::committed(MemClass c) const {
    std::lock_guard<std::mutex> lk(mu_);
    return committed_[static_cast<int>(c)];
}

size_t MemoryGuard::headroom() const {
    std::lock_guard<std::mutex> lk(mu_);
    size_t total = 0;
    for (size_t v : committed_) total += v;
    return budget_ > total ? budget_ - total : 0;
}

void MemoryGuard::set_evictor(EvictFn fn) {
    std::lock_guard<std::mutex> lk(mu_);
    evictor_ = std::move(fn);
}

bool MemoryGuard::try_reserve(size_t bytes, MemClass c) {
    EvictFn evictor;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (budget_ == 0) {  // no cap configured -> unbounded
            committed_[static_cast<int>(c)] += bytes;
            return true;
        }
        size_t total = 0;
        for (size_t v : committed_) total += v;
        if (total + bytes <= budget_) {
            committed_[static_cast<int>(c)] += bytes;
            return true;
        }
        evictor = evictor_;  // try to make room outside the lock
    }

    if (evictor) {
        const size_t need = bytes > 0 ? bytes : 0;
        evictor(need);
        std::lock_guard<std::mutex> lk(mu_);
        size_t total = 0;
        for (size_t v : committed_) total += v;
        if (total + bytes <= budget_) {
            committed_[static_cast<int>(c)] += bytes;
            return true;
        }
    }
    return false;  // refuse: this allocation would exceed the cap
}

void MemoryGuard::release(size_t bytes, MemClass c) {
    std::lock_guard<std::mutex> lk(mu_);
    size_t& slot = committed_[static_cast<int>(c)];
    slot = (bytes > slot) ? 0 : slot - bytes;
}

bool MemoryGuard::admit_plan(size_t planned_bytes, std::string* why) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (budget_ == 0) return true;
    if (planned_bytes > budget_) {
        if (why) {
            *why = "planned footprint " + human(planned_bytes) +
                   " exceeds --max-ram " + human(budget_);
        }
        return false;
    }
    size_t total = 0;
    for (size_t v : committed_) total += v;
    if (total + planned_bytes > budget_) {
        if (why) {
            *why = "already committed " + human(total) + "; plan needs " +
                   human(planned_bytes) + " but budget is " + human(budget_);
        }
        return false;
    }
    if (why) why->clear();
    return true;
}

size_t MemoryGuard::own_footprint() {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    const kern_return_t kr = task_info(mach_task_self(), TASK_VM_INFO,
                                       reinterpret_cast<task_info_t>(&info), &count);
    if (kr != KERN_SUCCESS) return 0;
    return static_cast<size_t>(info.phys_footprint);
}

size_t MemoryGuard::system_available() {
    vm_statistics64_data_t vmstat;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                          reinterpret_cast<host_info64_t>(&vmstat),
                          &count) != KERN_SUCCESS) {
        return 0;
    }
    vm_size_t page = 0;
    host_page_size(mach_host_self(), &page);
    return (static_cast<size_t>(vmstat.free_count) +
            static_cast<size_t>(vmstat.inactive_count)) * static_cast<size_t>(page);
}

void MemoryGuard::default_breach(size_t footprint, size_t budget) {
    std::fprintf(stderr,
                 "\n[aero] FATAL: memory cap exceeded — footprint %s > --max-ram %s. "
                 "Aborting immediately so the host cannot be pushed into swap/OOM.\n",
                 human(footprint).c_str(), human(budget).c_str());
    std::fflush(stderr);
    _exit(87);
}

void MemoryGuard::set_on_breach(BreachFn fn) {
    std::lock_guard<std::mutex> lk(mu_);
    on_breach_ = std::move(fn);
}

void MemoryGuard::start_monitor(unsigned interval_ms) {
    if (monitor_.joinable()) return;  // already running
    stop_.store(false);
    monitor_ = std::thread(&MemoryGuard::monitor_loop, this, interval_ms);
}

void MemoryGuard::stop_monitor() {
    stop_.store(true);
    if (monitor_.joinable()) monitor_.join();
}

void MemoryGuard::monitor_loop(unsigned interval_ms) {
    if (interval_ms == 0) interval_ms = 50;
    while (!stop_.load()) {
        const size_t fp = own_footprint();
        last_.store(fp);
        size_t prev = peak_.load();
        while (fp > prev && !peak_.compare_exchange_weak(prev, fp)) {}

        size_t budget;
        BreachFn fn;
        {
            std::lock_guard<std::mutex> lk(mu_);
            budget = budget_;
            fn = on_breach_;
        }
        if (budget > 0 && fp > budget) {
            exceeded_.store(true);
            if (fn) fn(fp, budget);
            else default_breach(fp, budget);
            return;  // breach is terminal by default
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }
}

bool MemoryGuard::preflight(size_t headroom_bytes, std::string* why) const {
    size_t budget;
    {
        std::lock_guard<std::mutex> lk(mu_);
        budget = budget_;
    }
    if (budget == 0) return true;
    const size_t avail = system_available();
    if (avail == 0) {
        if (why) *why = "could not read system memory statistics";
        return false;
    }
    if (avail < budget + headroom_bytes) {
        if (why) {
            *why = "system has only " + human(avail) + " reclaimable but the cap is " +
                   human(budget) + " (+" + human(headroom_bytes) +
                   " headroom) — refusing to start";
        }
        return false;
    }
    if (why) why->clear();
    return true;
}

// ---- LoadGate ------------------------------------------------------------

LoadGate::LoadGate(MemoryGuard& guard, size_t planned_total)
    : guard_(guard), planned_(planned_total) {
    std::string why;
    if (!guard_.admit_plan(planned_total, &why)) {
        ok_ = false;
        reason_ = why;
    }
}

bool LoadGate::next_chunk(size_t chunk_bytes, MemClass c) {
    if (!ok_) return false;   // already stopped by the guard
    if (done_) return false;  // plan already complete
    if (taken_ >= planned_) { // nothing left to load
        done_ = true;
        return false;
    }
    if (taken_ + chunk_bytes > planned_) {
        // The remaining plan is smaller than one chunk. That is COMPLETION, not
        // a violation: the admitted plan has been satisfied. Reporting it as a
        // failure would make every successful load look like a breach.
        done_ = true;
        return false;
    }
    if (!guard_.try_reserve(chunk_bytes, c)) {
        ok_ = false;
        reason_ = "next chunk of " + human(chunk_bytes) +
                  " would exceed --max-ram; load stopped";
        return false;
    }
    taken_ += chunk_bytes;
    reserved_ += chunk_bytes;

    // Pace the load so the monitor can sample the REAL footprint while the
    // memory is actually held. This is the window in which a breach that the
    // accounting cannot see (baseline process memory, allocator slack, GPU
    // driver allocations) becomes visible.
    if (pause_ms_ > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(pause_ms_));
        if (!check_measured()) return false;
    }
    return true;
}

bool LoadGate::pause_and_check() {
    if (!ok_) return false;
    if (pause_ms_ > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(pause_ms_));
    } else {
        std::this_thread::yield();  // at minimum, give the monitor a slot
    }
    return check_measured();
}

bool LoadGate::check_measured() {
    if (!ok_) return false;
    const size_t fp = MemoryGuard::own_footprint();
    const size_t budget = guard_.budget();
    if (budget > 0 && fp > budget) {
        ok_ = false;
        reason_ = "measured footprint " + human(fp) +
                  " exceeded --max-ram mid-load; load stopped";
        return false;
    }
    return true;
}

}  // namespace aero

// --------------------------------------------------------------------- C ABI
// Flat entry points so Python (ctypes) can arm the SAME guard the C++ tests use,
// rather than a reimplementation that could drift from it. The guard is a
// process-wide singleton; these forward to it.
extern "C" {

void aero_guard_set_budget(size_t bytes) {
    aero::MemoryGuard::instance().set_budget(bytes);
}
size_t aero_guard_budget(void) {
    return aero::MemoryGuard::instance().budget();
}
void aero_guard_start_monitor(unsigned interval_ms) {
    aero::MemoryGuard::instance().start_monitor(interval_ms);
}
void aero_guard_stop_monitor(void) {
    aero::MemoryGuard::instance().stop_monitor();
}
int aero_guard_exceeded(void) {
    return aero::MemoryGuard::instance().exceeded() ? 1 : 0;
}
size_t aero_guard_own_footprint(void) {
    return aero::MemoryGuard::own_footprint();
}
size_t aero_guard_peak_footprint(void) {
    return aero::MemoryGuard::instance().peak_footprint();
}
size_t aero_guard_system_available(void) {
    return aero::MemoryGuard::system_available();
}
int aero_guard_preflight(size_t headroom_bytes) {
    std::string why;
    return aero::MemoryGuard::instance().preflight(headroom_bytes, &why) ? 1 : 0;
}

}  // extern "C"
