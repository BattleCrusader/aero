// aero load demo — exercises the --max-ram guard end to end.
//
// This does NOT load a real model. It simulates the load plan a real loader
// would submit (resident set + paged expert blocks + KV + runtime) so the
// guard's behaviour is observable and testable without risking the host.

#include "aero/memory_guard.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace aero;

static const size_t MiB = 1024ull * 1024ull;


static std::string human(size_t bytes) {
    static const char* u[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double v = static_cast<double>(bytes);
    int i = 0;
    while (v >= 1024.0 && i < 4) { v /= 1024.0; ++i; }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.2f %s", v, u[i]);
    return buf;
}

static void usage() {
    std::printf(
        "usage: aero-load-demo --max-ram <size> [options]\n\n"
        "  --max-ram <size>   hard cap on process footprint (e.g. 8, 8G, 8GB)\n"
        "  --plan <size>      simulated model footprint (default 8.466G)\n"
        "  --resident <size>  simulated always-read weights (default 1.3G)\n"
        "  --kv <size>        simulated KV cache           (default 0.671G)\n"
        "  --runtime <size>   simulated runtime overhead   (default 0.2G)\n"
        "  --chunk <size>     expert block size            (default 64M)\n"
        "  --pause-ms <n>     sleep between blocks so the monitor can sample\n"
        "  --headroom <size>  pre-flight headroom          (default 1G)\n"
        "  --simulate         accounting only; never allocate (safe for big plans)\n");
}

int main(int argc, char** argv) {
    std::string max_ram, plan_s = "8.466G", resident_s = "1.3G", kv_s = "0.671G",
                runtime_s = "0.2G", chunk_s = "64M", headroom_s = "1G";
    unsigned pause_ms = 0;
    bool simulate = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](std::string& dst) {
            if (i + 1 < argc) dst = argv[++i];
        };
        if (a == "--max-ram") next(max_ram);
        else if (a == "--plan") next(plan_s);
        else if (a == "--resident") next(resident_s);
        else if (a == "--kv") next(kv_s);
        else if (a == "--runtime") next(runtime_s);
        else if (a == "--chunk") next(chunk_s);
        else if (a == "--headroom") next(headroom_s);
        else if (a == "--simulate") simulate = true;
        else if (a == "--pause-ms") pause_ms = static_cast<unsigned>(std::atoi(argv[++i]));
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { std::fprintf(stderr, "unknown argument: %s\n", a.c_str()); usage(); return 2; }
    }

    if (max_ram.empty()) { usage(); return 2; }

    auto parse = [](const std::string& s, size_t dflt) -> size_t {
        size_t b = dflt;
        std::string err;
        if (!parse_max_ram(s, &b, &err)) {
            std::fprintf(stderr, "bad size '%s': %s\n", s.c_str(), err.c_str());
            std::exit(2);
        }
        return b;
    };

    const size_t budget   = parse(max_ram, 0);
    const size_t plan     = parse(plan_s, 0);
    const size_t resident = parse(resident_s, 0);
    const size_t kv       = parse(kv_s, 0);
    const size_t runtime  = parse(runtime_s, 0);
    const size_t chunk    = parse(chunk_s, 0);
    const size_t headroom = parse(headroom_s, 0);

    std::printf("aero load demo\n");
    std::printf("  --max-ram %-8s -> %zu bytes (%s)\n", max_ram.c_str(), budget,
                human(budget).c_str());
    std::printf("  plan      %-8s -> %s   (resident %s + kv %s + runtime %s + paged experts)\n",
                plan_s.c_str(), human(plan).c_str(), human(resident).c_str(),
                human(kv).c_str(), human(runtime).c_str());
    std::printf("  chunk %s, pause %u ms\n\n", human(chunk).c_str(), pause_ms);

    MemoryGuard& g = MemoryGuard::instance();
    g.set_budget(budget);

    // ---- 1. predictive gate: refuse before touching the disk ---------------
    std::string why;
    if (!g.admit_plan(plan, &why)) {
        std::printf("  REFUSED before loading: %s\n", why.c_str());
        return 3;
    }
    std::printf("  plan admitted by the predictive gate\n");

    // ---- 2. pre-flight: refuse if the host lacks headroom ------------------
    if (!g.preflight(headroom, &why)) {
        std::printf("  REFUSED by pre-flight: %s\n", why.c_str());
        return 4;
    }
    std::printf("  pre-flight ok (system reclaimable %s)\n", human(MemoryGuard::system_available()).c_str());

    // ---- 3. the load itself, paced and interruptible -----------------------
    g.start_monitor(pause_ms > 0 ? 20u : 50u);

    if (!g.try_reserve(resident, MemClass::WeightsResident)) {
        std::printf("  REFUSED: resident trunk of %s does not fit\n", human(resident).c_str());
        return 5;
    }
    if (!g.try_reserve(kv, MemClass::KVCache)) {
        std::printf("  REFUSED: KV cache of %s does not fit\n", human(kv).c_str());
        return 5;
    }
    if (!g.try_reserve(runtime, MemClass::Runtime)) {
        std::printf("  REFUSED: runtime overhead of %s does not fit\n", human(runtime).c_str());
        return 5;
    }

    const size_t paged_budget = plan - resident - kv - runtime;

    // SAFETY: this demo allocates for real, so cap how much it will ever touch.
    // Without this, `--plan 8.466G` would try to malloc ~8 GB and could OOM the
    // host -- i.e. the demo itself would do the thing the guard exists to
    // prevent. Use --simulate to exercise the accounting without allocating.
    const size_t demo_alloc_cap = 256 * MiB;
    const size_t allocatable = simulate ? 0 : ((paged_budget < demo_alloc_cap)
                                                   ? paged_budget
                                                   : demo_alloc_cap);
    if (!simulate && paged_budget > demo_alloc_cap) {
        std::printf("  note: paging %s, but this demo only allocates the first %s\n"
                    "        (use --simulate for accounting only). The guard logic is\n"
                    "        identical either way; only the malloc is capped.\n",
                    human(paged_budget).c_str(), human(allocatable).c_str());
    }

    LoadGate gate(g, paged_budget);
    if (!gate.ok()) {
        std::printf("  REFUSED: paged plan of %s rejected: %s\n",
                    human(paged_budget).c_str(), gate.reason().c_str());
        return 5;
    }
    gate.set_chunk_pause_ms(pause_ms);

    std::vector<void*> blocks;
    size_t n = 0;
    size_t sim_charged = 0;
    std::printf("  loading %s of expert blocks...\n", human(paged_budget).c_str());
    while (gate.next_chunk(chunk)) {
        if (simulate) {
            // accounting only: charge the guard, touch nothing
            sim_charged += chunk;
            if (sim_charged > allocatable && allocatable == 0) { /* no-op */ }
        } else if (gate.taken() <= allocatable) {
            blocks.push_back(std::malloc(chunk));  // touch it, so the footprint is real
            if (blocks.back()) std::memset(blocks.back(), 1, chunk);
        }
        ++n;
        if (n % 8 == 0) {
            std::printf("    %4zu blocks  %8s paged  footprint %s / %s\n", n,
                        human(gate.taken()).c_str(),
                        human(MemoryGuard::own_footprint()).c_str(), human(budget).c_str());
        }
        if (n > 1000000) break;  // safety
    }

    g.stop_monitor();

    std::printf("\n  ledger:\n");
    size_t total = 0;
    for (int i = 0; i < static_cast<int>(MemClass::Count); ++i) {
        const MemClass c = static_cast<MemClass>(i);
        const size_t v = g.committed(c);
        total += v;
        std::printf("    %-18s %12s\n", mem_class_name(c), human(v).c_str());
    }
    std::printf("    %-18s %12s / %s cap\n", "TOTAL", human(total).c_str(), human(budget).c_str());
    std::printf("    %-18s %12s\n", "peak footprint", human(g.peak_footprint()).c_str());

    for (void* p : blocks) std::free(p);

    if (!gate.ok()) {
        std::printf("\n  LOAD STOPPED: %s\n", gate.reason().c_str());
        std::printf("  (the guard held the cap; the host was never at risk)\n");
        return 6;
    }
    std::printf("\n  load complete within --max-ram (%s paged, %s charged to the cap)\n",
                human(gate.taken()).c_str(), human(g.committed()).c_str());
    return 0;
}
