// Aero memory guard — self-test.
//
// SAFETY: this test never loads a model and never allocates more than a few MiB.
// It proves the guard's refusals by exercising the accounting, not by trying to
// exhaust the host. It is safe to run on the target machine at any time.

#include "aero/memory_guard.hpp"

#include <sys/types.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace aero;

static int g_checks = 0;
static int g_failures = 0;

static void check(bool cond, const char* what) {
    ++g_checks;
    if (cond) {
        std::printf("  ok   %s\n", what);
    } else {
        ++g_failures;
        std::printf("  FAIL %s\n", what);
    }
}

// Run the reference tool and return its reported footprint in bytes.
static size_t footprint_reference() {
    char cmd[128];
    std::snprintf(cmd, sizeof(cmd), "/usr/bin/footprint -p %d 2>/dev/null",
                  static_cast<int>(getpid()));
    FILE* p = popen(cmd, "r");
    if (!p) return 0;
    size_t bytes = 0;
    char line[4096];
    while (std::fgets(line, sizeof(line), p)) {
        const char* at = std::strstr(line, "Footprint:");
        if (!at) continue;
        unsigned long long value = 0;
        char unit[16] = {0};
        if (std::sscanf(at, "Footprint: %llu %15s", &value, unit) == 2) {
            double mult = 1.0;
            if (!std::strcmp(unit, "KB")) mult = 1024.0;
            else if (!std::strcmp(unit, "MB")) mult = 1024.0 * 1024.0;
            else if (!std::strcmp(unit, "GB")) mult = 1024.0 * 1024.0 * 1024.0;
            bytes = static_cast<size_t>(static_cast<double>(value) * mult);
        }
        break;
    }
    pclose(p);
    return bytes;
}

static const size_t KiB = 1024ull;
static const size_t MiB = 1024ull * 1024ull;
static const size_t GiB = 1024ull * MiB;

int main() {
    std::printf("=== aero memory guard self-test ===\n\n");

    // ---------------------------------------------------------------- parsing
    std::printf("[1] --max-ram parsing\n");
    {
        size_t b = 0;
        std::string err;

        check(parse_max_ram("8", &b, &err) && b == 8 * GiB,
              "\"8\" resolves to 8 GiB (colloquial RAM convention)");
        check(parse_max_ram("8G", &b, &err) && b == 8 * GiB, "\"8G\" resolves to 8 GiB");
        check(parse_max_ram("8GB", &b, &err) && b == 8000000000ull,
              "\"8GB\" resolves to 8.0e9 bytes (decimal)");
        check(parse_max_ram("512M", &b, &err) && b == 512 * MiB, "\"512M\" resolves to MiB");
        check(parse_max_ram("512MB", &b, &err) && b == 512000000ull,
              "\"512MB\" resolves to 5.12e8 bytes");
        check(parse_max_ram("7.5", &b, &err) && b == static_cast<size_t>(7.5 * GiB),
              "\"7.5\" resolves to 7.5 GiB");
        check(parse_max_ram("8589934592", &b, &err) && b == 8589934592ull,
              "a bare number >= 2^20 is read as raw bytes");
        check(parse_max_ram("2048", &b, &err) && b == 2048 * GiB,
              "a small bare number is read as GiB, not bytes");
        check(!parse_max_ram("", &b, &err), "empty string is rejected");
        check(!parse_max_ram("abc", &b, &err), "garbage is rejected");
        check(!parse_max_ram("8XB", &b, &err), "unknown suffix is rejected");
    }

    MemoryGuard& g = MemoryGuard::instance();

    // ------------------------------------------------------------- accounting
    std::printf("\n[2] exact accounting and refusal at the cap\n");
    {
        g.set_budget(64 * MiB);
        check(g.budget() == 64 * MiB, "budget set to 64 MiB");
        check(g.committed() == 0, "nothing committed initially");

        check(g.try_reserve(32 * MiB, MemClass::WeightsResident), "reserve 32 MiB fits");
        check(g.committed() == 32 * MiB, "committed is 32 MiB");
        check(g.headroom() == 32 * MiB, "headroom is 32 MiB");

        check(!g.try_reserve(40 * MiB, MemClass::WeightsCache),
              "reserve 40 MiB is REFUSED (32+40 > 64)");
        check(g.committed() == 32 * MiB, "refusal did not change committed bytes");

        check(g.try_reserve(32 * MiB, MemClass::KVCache), "reserve the remaining 32 MiB fits");
        check(g.headroom() == 0, "headroom is now 0");
        check(!g.try_reserve(1, MemClass::Scratch), "even 1 byte is refused at the cap");

        g.release(32 * MiB, MemClass::KVCache);
        check(g.headroom() == 32 * MiB, "release restores headroom");
        check(g.try_reserve(32 * MiB, MemClass::Scratch), "re-reserve succeeds");
        g.release(32 * MiB, MemClass::Scratch);
        g.release(32 * MiB, MemClass::WeightsResident);
        check(g.committed() == 0, "all released");
    }

    // -------------------------------------------------------------- evictor
    std::printf("\n[3] eviction hook is consulted before refusal\n");
    {
        static size_t cache_bytes = 0;
        g.set_budget(64 * MiB);
        cache_bytes = 48 * MiB;
        check(g.try_reserve(cache_bytes, MemClass::WeightsCache), "48 MiB of cache admitted");
        g.set_evictor([&](size_t need) -> size_t {
            const size_t freed = (need < cache_bytes) ? need : cache_bytes;
            g.release(freed, MemClass::WeightsCache);
            cache_bytes -= freed;
            return freed;
        });
        check(g.try_reserve(32 * MiB, MemClass::WeightsResident),
              "32 MiB admitted by evicting the cache first");
        check(cache_bytes == 16 * MiB, "evictor freed exactly the shortfall");
        g.set_evictor(nullptr);
        g.release(32 * MiB, MemClass::WeightsResident);
        g.release(cache_bytes, MemClass::WeightsCache);
        check(g.committed() == 0, "cleaned up");
    }

    // -------------------------------------------------- predictive plan gate
    std::printf("\n[4] predictive gate: refuse a load BEFORE reading the disk\n");
    {
        g.set_budget(8 * GiB);
        std::string why;

        check(g.admit_plan(7 * GiB, &why), "a 7 GiB plan is admitted under an 8 GiB cap");
        check(!g.admit_plan(9 * GiB, &why), "a 9 GiB plan is REFUSED against an 8 GiB cap");
        check(why.find("exceeds --max-ram") != std::string::npos,
              "refusal explains the reason");

        // The real-world boundary. The model file is 8,465,553,920 bytes =
        // 7.884 GiB, which just fits inside an 8 GiB cap — leaving only ~124 MiB
        // for KV, scratch and runtime. This is exactly the case the engine has to
        // handle by paging experts, and exactly what the guard must refuse if the
        // loader asks for the naive full-load plan.
        why.clear();
        check(g.admit_plan(8465553920ull, &why),
              "the 8.466 GB model file ALONE just fits an 8 GiB cap (~124 MiB spare)");
        why.clear();
        check(!g.admit_plan(8465553920ull + 256 * MiB, &why),
              "the naive full load (file + KV + runtime) is REFUSED against --max-ram 8");
        std::printf("       -> %s\n", why.c_str());
    }

    // ------------------------------------------------------ chunked loading
    std::printf("\n[5] chunked load: completes cleanly, and stops if the budget is taken mid-load\n");
    {
        g.set_budget(64 * MiB);

        // (a) a plan that fits must COMPLETE and report done(), not a failure
        {
            LoadGate gate(g, 24 * MiB);
            check(gate.ok(), "gate opened for a 24 MiB plan");
            std::vector<void*> blocks;
            size_t chunks = 0;
            while (gate.next_chunk(8 * MiB)) {
                blocks.push_back(std::malloc(8 * MiB));
                ++chunks;
                if (chunks > 16) break;  // safety
            }
            check(gate.ok() && gate.done(), "plan completed: ok()==true and done()==true");
            check(gate.taken() == 24 * MiB, "exactly the planned bytes were taken");
            check(gate.remaining() == 0, "nothing left under the plan");
            for (void* p : blocks) std::free(p);
            g.release(gate.reserved(), MemClass::WeightsCache);
        }

        // (b) the remaining budget is consumed by another component mid-load,
        //     so the next chunk must be REFUSED -- a failure, not completion
        {
            LoadGate gate(g, 48 * MiB);
            check(gate.ok(), "gate opened for a 48 MiB plan");
            check(gate.next_chunk(8 * MiB), "chunk 1 ok");
            check(gate.next_chunk(8 * MiB), "chunk 2 ok");
            check(gate.next_chunk(8 * MiB), "chunk 3 ok");
            check(gate.next_chunk(8 * MiB), "chunk 4 ok (32 MiB paged)");
            check(g.try_reserve(32 * MiB, MemClass::KVCache),
                  "32 MiB of KV taken mid-load, consuming the rest of the cap");
            check(!gate.next_chunk(8 * MiB), "next chunk REFUSED: cap reached");
            check(!gate.ok(), "gate closed as a FAILURE");
            check(!gate.done(), "done()==false: the plan was NOT completed");
            check(g.committed() <= g.budget(), "committed never exceeded the budget");
            std::printf("       -> %s\n", gate.reason().c_str());
            g.release(gate.reserved(), MemClass::WeightsCache);
            g.release(32 * MiB, MemClass::KVCache);
        }

        g.set_budget(0);
        check(g.committed() == 0, "cleaned up");
    }

    // ------------------------- paced load: the measured check closes the gate
    std::printf("\n[5b] paced load: the MEASURED footprint closes a gate the accounting let through\n");
    {
        // This is the real-world case the exact accounting cannot see: the plan
        // is legal by accounting, but the process ALREADY sits above the cap
        // because of baseline memory the accounting never tracked. Only
        // measurement can catch it. Budget is derived from the live footprint so
        // the test is deterministic rather than tuned to one binary.
        const size_t fp = MemoryGuard::own_footprint();
        const size_t budget = (fp > 256 * KiB) ? fp - 64 * KiB : 64 * KiB;
        const size_t planned = budget / 2;
        const size_t chunk = (budget / 8 > 4096) ? budget / 8 : 4096;

        g.set_budget(budget);
        std::string why;
        check(g.admit_plan(planned, &why),
              "accounting admits a plan comfortably inside the cap");

        LoadGate gate(g, planned);
        gate.set_chunk_pause_ms(10);  // pace the load so the monitor can sample
        check(gate.ok(), "gate opened");

        size_t chunks = 0;
        while (gate.next_chunk(chunk)) {
            ++chunks;
            if (chunks > 64) break;  // safety: test must terminate
        }
        check(!gate.ok(), "gate CLOSED — by measurement, not by accounting");
        check(gate.reason().find("measured footprint") != std::string::npos,
              "the reason names the measured footprint");
        std::printf("       -> stopped after %zu chunks: %s\n", chunks, gate.reason().c_str());

        g.release(gate.reserved(), MemClass::WeightsCache);
        g.set_budget(0);
        check(g.committed() == 0, "cleaned up");
    }

    // ------------------------------------------- measured footprint + monitor
    std::printf("\n[6] measured footprint agrees with /usr/bin/footprint\n");
    {
        const size_t mine = MemoryGuard::own_footprint();
        const size_t ref = footprint_reference();
        std::printf("       mach phys_footprint = %zu bytes, reference tool = %zu bytes\n",
                    mine, ref);
        check(mine > 0, "mach phys_footprint is readable");
        check(ref > 0, "reference tool produced a value");
        // The reference tool reports in KB and is sampled at a different instant,
        // so allow 10% or 1 MiB, whichever is larger.
        const size_t slack = (ref / 10 > MiB) ? ref / 10 : MiB;
        const size_t diff = (mine > ref) ? mine - ref : ref - mine;
        check(diff <= slack, "the two agree within tolerance (so the fast path is valid)");
    }

    std::printf("\n[7] continuous monitor fires on breach\n");
    {
        static bool breached = false;
        static size_t seen_fp = 0;
        g.set_budget(1024);  // 1 KiB — guaranteed below any real process footprint
        g.set_on_breach([](size_t fp, size_t) {
            breached = true;
            seen_fp = fp;
        });
        g.start_monitor(20);
        for (int i = 0; i < 200 && !breached; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        g.stop_monitor();
        check(breached, "monitor detected the footprint exceeding the cap");
        check(g.exceeded(), "exceeded() reports true");
        check(seen_fp > 1024, "breach reported the measured footprint");
        check(g.peak_footprint() > 0, "peak footprint was tracked");
        g.set_on_breach(nullptr);
        g.set_budget(0);
    }

    std::printf("\n[8] pre-flight refuses when the host lacks headroom\n");
    {
        std::string why;
        g.set_budget(8 * GiB);
        const size_t avail = MemoryGuard::system_available();
        std::printf("       system reclaimable = %zu bytes\n", avail);
        check(avail > 0, "system_available is readable");
        // An absurd cap must be refused.
        g.set_budget(4096ull * GiB);
        check(!g.preflight(0, &why), "absurd cap is refused on pre-flight");
        std::printf("       -> %s\n", why.c_str());
        g.set_budget(0);
    }

    std::printf("\n=== %d checks, %d failures ===\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
