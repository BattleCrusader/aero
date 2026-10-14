// Aero memory guard — hard, continuously-enforced (V)RAM cap.
//
// Design notes (macOS-specific, learned the hard way):
//   * `ulimit -v` is a NO-OP for malloc on macOS. It is never used here.
//   * `ps -o rss` does NOT include Metal/GPU allocations, so a process with a
//     large GPU mapping can look small and still OOM the host. The measured
//     quantity is therefore `phys_footprint` (task_info TASK_VM_INFO), which is
//     the same number Activity Monitor and /usr/bin/footprint report, and which
//     DOES include GPU memory.
//   * Enforcement is two-layer by construction:
//       1. EXACT accounting  — every byte is reserved through this guard before
//          it is allocated, so an over-budget allocation is refused *before* it
//          happens. Continuous, because it runs on every reservation.
//       2. MEASURED monitor  — a thread polls phys_footprint and fires a breach
//          if reality exceeds the cap for any reason the accounting missed
//          (allocator slack, Metal driver allocations, fragmentation).
#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

namespace aero {

// Why the model is consuming memory. Attribution only — every class is charged
// against the same single budget.
enum class MemClass {
    WeightsResident,  // always-read weights: pinned, never evicted
    WeightsCache,     // pageable expert blocks: evictable under pressure
    KVCache,          // attention state
    Scratch,          // transient compute buffers
    Runtime,          // context, scheduler, allocator slack
    Count
};

const char* mem_class_name(MemClass c);

// Parse a --max-ram style size. Accepted forms:
//   "8"      -> 8 GiB   (bare numbers mean the colloquial "8GB of RAM")
//   "8G"     -> 8 GiB   (K/M/G/T are powers of 1024)
//   "8GB"    -> 8.0e9   (KB/MB/GB/TB are powers of 1000)
//   "8589934592" -> bytes
// The resolved byte count is always logged, so there is never ambiguity in
// practice about which convention was applied.
bool parse_max_ram(const std::string& text, size_t* out_bytes, std::string* error);

class MemoryGuard {
public:
    static MemoryGuard& instance();

    // ---------- configuration ----------
    // Hard cap on total process footprint (weights + KV + scratch + runtime).
    void set_budget(size_t bytes);
    size_t budget() const { return budget_; }
    bool has_budget() const { return budget_ > 0; }

    // ---------- layer 1: exact accounting ----------
    // Reserve `bytes` before allocating them. Returns false if doing so would
    // exceed the budget AND nothing could be evicted. Never allocates itself.
    bool try_reserve(size_t bytes, MemClass c);
    void release(size_t bytes, MemClass c);

    size_t committed() const;
    size_t committed(MemClass c) const;
    // Bytes the engine may still hand out right now.
    size_t headroom() const;

    // Eviction hook: the engine registers how to free cache bytes. Called when a
    // reservation would otherwise be refused. Return the number of bytes freed.
    using EvictFn = std::function<size_t(size_t need_bytes)>;
    void set_evictor(EvictFn fn);

    // ---------- predictive gate ----------
    // Refuse an entire load before any bytes are read from disk. This is what
    // makes "stop loading immediately if it WILL exceed" true up front.
    bool admit_plan(size_t planned_bytes, std::string* why) const;

    // ---------- layer 2: measured footprint ----------
    static size_t own_footprint();       // phys_footprint, includes GPU
    static size_t system_available();    // free + inactive bytes

    // ---------- continuous monitor ----------
    void start_monitor(unsigned interval_ms = 50);
    void stop_monitor();
    bool exceeded() const { return exceeded_.load(); }
    size_t peak_footprint() const { return peak_.load(); }
    size_t last_footprint() const { return last_.load(); }

    // Action on breach. Default: report and hard-exit with code 87.
    using BreachFn = std::function<void(size_t footprint, size_t budget)>;
    void set_on_breach(BreachFn fn);
    static void default_breach(size_t footprint, size_t budget);

    // ---------- pre-flight ----------
    // Refuse to start at all if the host lacks the headroom.
    bool preflight(size_t headroom_bytes, std::string* why) const;

private:
    MemoryGuard() = default;
    ~MemoryGuard();
    void monitor_loop(unsigned interval_ms);

    mutable std::mutex mu_;
    size_t budget_ = 0;
    size_t committed_[static_cast<int>(MemClass::Count)] = {};
    EvictFn evictor_;

    std::atomic<bool> stop_{false};
    std::atomic<bool> exceeded_{false};
    std::atomic<size_t> peak_{0};
    std::atomic<size_t> last_{0};
    std::thread monitor_;
    BreachFn on_breach_;
};

// Chunked-load gate. Long loads must be interruptible, so the loader calls
// next_chunk() between steps; the moment the cap is at risk the gate closes and
// the load stops. This is the mechanism that stops a load already in flight.
//
// The gate can also PACE the load: with a chunk pause set, the loader sleeps
// between chunks, which gives the monitor thread a window to sample the real
// phys_footprint while the load is in flight. That catches breaches the exact
// accounting cannot see (baseline process memory, allocator slack, Metal driver
// allocations). Pacing is the difference between "checked at reservations" and
// "checked continuously while loading".
class LoadGate {
public:
    LoadGate(MemoryGuard& guard, size_t planned_total);

    bool ok() const { return ok_; }
    const std::string& reason() const { return reason_; }
    size_t taken() const { return taken_; }
    size_t planned() const { return planned_; }

    // Milliseconds to sleep between chunks. 0 = as fast as possible.
    void set_chunk_pause_ms(unsigned ms) { pause_ms_ = ms; }
    unsigned chunk_pause_ms() const { return pause_ms_; }

    // Reserve the next chunk, pause (if configured), then verify the MEASURED
    // footprint. Returns false when there is nothing more to load.
    //
    // The two ways it returns false are DIFFERENT and must not be conflated:
    //   ok() == true  && done()  -> the plan was fully loaded. SUCCESS.
    //   ok() == false            -> the guard stopped the load. FAILURE.
    bool next_chunk(size_t chunk_bytes, MemClass c = MemClass::WeightsCache);

    // True once the admitted plan has been fully consumed.
    bool done() const { return done_; }
    // Bytes still to load under the plan.
    size_t remaining() const { return planned_ > taken_ ? planned_ - taken_ : 0; }

    // Explicit yield point for a loader mid-chunk: sleeps for the configured
    // pause and re-checks the measured footprint.
    bool pause_and_check();

    // Verify the measured footprint is still inside the cap mid-load.
    bool check_measured();

    // Bytes reserved so far — the caller must release exactly this much if it
    // aborts on a closed gate.
    size_t reserved() const { return reserved_; }

private:
    MemoryGuard& guard_;
    size_t planned_;
    size_t taken_ = 0;
    size_t reserved_ = 0;
    unsigned pause_ms_ = 0;
    bool ok_ = true;
    bool done_ = false;
    std::string reason_;
};

}  // namespace aero
