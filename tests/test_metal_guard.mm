// Metal + memory guard test — the gap that matters.
//
// PROVES the claim the whole guard is built on:
//   * a GPU-only (MTLStorageModePrivate) allocation is counted by
//     phys_footprint  -- so the guard can see it
//   * the same allocation is NOT counted by resident_size (classic RSS)
//     -- so a guard built on rss/`ps -o rss` would miss it entirely, which is
//        exactly how a process can look small and still OOM the host
//
// Then proves the guard ACTS on it: with a cap below the post-allocation
// footprint, the continuous monitor fires and the load gate closes.
//
// SAFETY: allocates one 512 MB private Metal buffer, fills it to force real
// commitment, then releases it. Bounded and small.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mach/mach.h>
#include <mach/task_info.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <chrono>

#include "aero/memory_guard.hpp"

using namespace aero;

static int g_checks = 0;
static int g_fails = 0;

static void check(bool cond, const char* what) {
    ++g_checks;
    if (cond) std::printf("  ok   %s\n", what);
    else { ++g_fails; std::printf("  FAIL %s\n", what); }
}

// Classic RSS -- the metric that DOES miss GPU memory.
static size_t resident_size() {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO,
                  reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS) {
        return 0;
    }
    return static_cast<size_t>(info.resident_size);
}

static const char* kShader = R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void fill(device float *out [[buffer(0)]],
                 constant uint &n [[buffer(1)]],
                 uint gid [[thread_position_in_grid]])
{
    if (gid < n) out[gid] = 1.0f;
}
)METAL";

static std::string mb(size_t b) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.1f MiB", b / 1048576.0);
    return buf;
}

static long delta(size_t after, size_t before) {
    return static_cast<long>(after) - static_cast<long>(before);
}

int main() {
    std::printf("=== aero Metal guard test ===\n\n");

    const size_t SZ = 512ull * 1024 * 1024;   // 512 MiB

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { std::printf("no Metal device\n"); return 2; }
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kShader]
                                           options:nil error:&err];
    if (!lib) { std::printf("shader failed: %s\n",
                            err ? [[err localizedDescription] UTF8String] : "?"); return 2; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"fill"];
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) { std::printf("pipeline failed\n"); return 2; }

    // ---- baseline --------------------------------------------------------
    const size_t fp0 = MemoryGuard::own_footprint();
    const size_t rs0 = resident_size();
    std::printf("[1] baseline\n");
    std::printf("       phys_footprint = %s\n", mb(fp0).c_str());
    std::printf("       resident_size  = %s\n", mb(rs0).c_str());
    check(fp0 > 0, "phys_footprint readable");
    check(rs0 > 0, "resident_size readable");

    // ---- allocate GPU-ONLY memory and force commitment -------------------
    id<MTLBuffer> gpuBuf = [dev newBufferWithLength:SZ
                                            options:MTLResourceStorageModePrivate];
    if (!gpuBuf) { std::printf("private buffer alloc failed\n"); return 2; }

    id<MTLCommandQueue> q = [dev newCommandQueue];
    if (!q) { std::printf("no command queue\n"); return 2; }

    const uint32_t n = static_cast<uint32_t>(SZ / sizeof(float));
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:gpuBuf offset:0 atIndex:0];
    [enc setBytes:&n length:sizeof(n) atIndex:1];
    const NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if ([cb error]) {
        std::printf("fill kernel failed: %s\n",
                    [[[cb error] localizedDescription] UTF8String]);
        return 2;
    }

    const size_t fp1 = MemoryGuard::own_footprint();
    const size_t rs1 = resident_size();
    const long dfp = delta(fp1, fp0);
    const long drs = delta(rs1, rs0);

    // ---- the central claim ----------------------------------------------
    std::printf("\n[2] after allocating and filling %s of GPU-ONLY memory\n",
                mb(SZ).c_str());
    std::printf("       phys_footprint  %s -> %s   (delta %+ld MiB)\n",
                mb(fp0).c_str(), mb(fp1).c_str(), dfp / 1048576);
    std::printf("       resident_size   %s -> %s   (delta %+ld MiB)\n",
                mb(rs0).c_str(), mb(rs1).c_str(), drs / 1048576);

    // footprint must reflect most of the allocation
    check(dfp > static_cast<long>(SZ) * 8 / 10,
          "phys_footprint GREW by ~the full GPU allocation -> it does see GPU memory");

    // rss must NOT -- this is the whole reason the guard uses footprint
    check(drs < static_cast<long>(SZ) / 2,
          "resident_size did NOT grow by the allocation -> rss/`ps -o rss` MISSES it");

    std::printf("       => a guard built on rss would have missed %ld MiB of real GPU memory\n",
                dfp / 1048576);

    // ---- the guard acts on it -------------------------------------------
    MemoryGuard& g = MemoryGuard::instance();

    std::printf("\n[3] the guard ACTS on the GPU memory it can now see\n");
    {
        static bool breached = false;
        static size_t seen = 0;
        g.set_budget(fp1 - 32 * 1024 * 1024);   // cap 32 MiB BELOW current footprint
        g.set_on_breach([](size_t fp, size_t) { breached = true; seen = fp; });
        g.start_monitor(20);
        for (int i = 0; i < 200 && !breached; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        g.stop_monitor();
        check(breached, "continuous monitor fired on a GPU-memory breach");
        check(g.exceeded(), "exceeded() reports true");
        check(seen > fp1 - 32 * 1024 * 1024, "breach reported the measured footprint");
        g.set_on_breach(nullptr);
    }

    // ---- admission refuses a plan that needs GPU memory ------------------
    std::printf("\n[4] admission refuses a plan that would exceed the cap\n");
    {
        std::string why;
        check(!g.admit_plan(SZ, &why), "a plan needing another 512 MiB is REFUSED");
        std::printf("       -> %s\n", why.c_str());
    }

    // ---- the paced load gate sees it too --------------------------------
    std::printf("\n[5] the load gate closes on measured GPU footprint\n");
    {
        const size_t budget = g.budget();
        LoadGate gate(g, budget / 4);
        gate.set_chunk_pause_ms(5);
        size_t chunks = 0;
        while (gate.next_chunk(budget / 16)) {
            ++chunks;
            if (chunks > 64) break;
        }
        check(!gate.ok(), "gate CLOSED because the measured footprint (incl. GPU) was over cap");
        std::printf("       -> %s\n", gate.reason().c_str());
        g.release(gate.reserved(), MemClass::WeightsCache);
    }

    g.set_budget(0);

    // ---- release and confirm the guard stays safe ------------------------
    // NOTE: Metal does not necessarily return GPU memory to the system
    // footprint the instant the last reference drops -- the driver may hold a
    // cache. That is fine for the guard: a reading that lags HIGH is
    // conservative (it can only make us refuse more), never unsafe. What must
    // hold is that the footprint is BOUNDED -- release must not increase it.
    gpuBuf = nil;
    @autoreleasepool { }
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    const size_t fp2 = MemoryGuard::own_footprint();
    std::printf("\n[6] after releasing the GPU buffer\n");
    std::printf("       phys_footprint = %s (peak was %s)\n",
                mb(fp2).c_str(), mb(fp1).c_str());
    if (fp2 < fp1) {
        std::printf("       -> memory was returned to the system\n");
    } else {
        std::printf("       -> NOT returned yet: Metal retains a cache. Recorded as a\n"
                    "          property, not a failure -- the guard's reading lags HIGH,\n"
                    "          which is the safe direction (it refuses more, never less).\n");
    }
    check(fp2 <= fp1 + 8 * 1024 * 1024,
          "footprint is BOUNDED after release (did not grow) -> the guard stays safe");
    check(fp1 > fp0, "peak footprint did include the GPU allocation");

    // admit_plan() is the ACCOUNTING gate: it reasons only about bytes the
    // engine declared, so it is blind to memory the accounting never tracked --
    // which is exactly what the GPU allocation above is. Admitting a 1-byte
    // plan is therefore CORRECT behaviour, and it is precisely why the measured
    // layer exists. Assert both halves of that contract.
    //
    // The cap is derived from the LIVE footprint rather than a fixed number:
    // whether Metal returned the buffer above is driver-dependent (it sometimes
    // does), so a hard-coded cap made this assertion flaky.
    MemoryGuard& mg = MemoryGuard::instance();
    const size_t now_fp = MemoryGuard::own_footprint();
    const size_t tiny_cap = now_fp / 2;      // guaranteed below the current footprint
    mg.set_budget(tiny_cap);
    std::printf("\n[7] the two layers have different scopes\n");
    std::printf("       live footprint %s -> cap set to %s\n",
                mb(now_fp).c_str(), mb(tiny_cap).c_str());

    std::string why2;
    check(mg.admit_plan(1, &why2),
          "admit_plan is accounting-only: admits a 1-byte plan (documented limitation)");

    LoadGate probe(mg, 1);
    probe.set_chunk_pause_ms(1);
    check(!probe.check_measured(),
          "check_measured() REFUSES -> the MEASURED layer catches untracked memory");
    std::printf("       -> accounting: blind | measured: catches it. Both layers needed.\n");
    mg.release(probe.reserved(), MemClass::Scratch);
    mg.set_budget(0);

    std::printf("\n=== %d checks, %d failures ===\n", g_checks, g_fails);
    return g_fails == 0 ? 0 : 1;
}
