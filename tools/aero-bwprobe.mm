// aero bandwidth probe — can ONE unit saturate the memory bus, or does
// concurrency across units buy aggregate bandwidth?
//
// This decides whether "use every lever at once" is real or not.
// It measures pure streaming read bandwidth (the shape of a weight matvec):
//   CPU only, GPU only, and CPU+GPU concurrently.
// If concurrent ~= max(single), multi-unit is pointless.
// If concurrent > single, there is real aggregate-bandwidth headroom.
//
// SAFETY: allocates one buffer (default 256 MB) and runs under the memory
// guard. No model, no large allocation, safe to run on the target machine.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "aero/memory_guard.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

using namespace aero;

static const char* kShader = R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void bw_read(device const float4 *a      [[buffer(0)]],
                    device float        *out    [[buffer(1)]],
                    constant uint       &n4     [[buffer(2)]],
                    uint gid  [[thread_position_in_grid]],
                    uint gsz  [[threads_per_grid]])
{
    float acc = 0.0f;
    for (uint i = gid; i < n4; i += gsz) {
        float4 v = a[i];
        acc += v.x + v.y + v.z + v.w;
    }
    out[gid & 1023u] = acc;
}
)METAL";

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// CPU streaming pass over the same buffer shape.
static void cpu_stream(const float* data, size_t floats, std::atomic<double>* sink) {
    float acc = 0.0f;
    for (size_t i = 0; i < floats; i += 4) {
        const float* p = data + i;
        acc += p[0] + p[1] + p[2] + p[3];
    }
    *sink = acc;
}

int main(int argc, char** argv) {
    size_t mb = 256;
    int iters = 5;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--mb" && i + 1 < argc) mb = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
    }

    const size_t bytes = mb * 1024ull * 1024ull;

    // Run under the guard so this probe can never push the host.
    MemoryGuard& g = MemoryGuard::instance();
    g.set_budget(bytes * 2 + 256ull * 1024 * 1024);
    g.start_monitor(50);

    std::printf("aero bandwidth probe\n");
    std::printf("  buffer %zu MB, %d iterations, guard cap %zu MB\n\n",
                mb, iters, g.budget() / (1024 * 1024));

    std::vector<float> host(bytes / sizeof(float));
    for (size_t i = 0; i < host.size(); ++i) host[i] = static_cast<float>(i % 97) * 0.5f;
    if (!g.try_reserve(bytes, MemClass::Scratch)) {
        std::printf("  refused: buffer would exceed the guard\n");
        return 3;
    }
    std::printf("  buffer allocated (%zu MB)\n", bytes / (1024 * 1024));

    // ---------------- GPU setup ----------------
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { std::printf("  no Metal device\n"); return 4; }
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kShader]
                                           options:nil
                                             error:&err];
    if (!lib) {
        std::printf("  shader compile failed: %s\n",
                    err ? [[err localizedDescription] UTF8String] : "?");
        return 4;
    }
    id<MTLFunction> fn = [lib newFunctionWithName:@"bw_read"];
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) { std::printf("  pipeline failed\n"); return 4; }

    id<MTLBuffer> buf = [dev newBufferWithBytes:host.data()
                                         length:bytes
                                        options:MTLResourceStorageModeShared];
    id<MTLBuffer> out = [dev newBufferWithLength:4096
                                         options:MTLResourceStorageModeShared];
    if (!buf || !out) { std::printf("  buffer alloc failed\n"); return 4; }

    const uint32_t n4 = static_cast<uint32_t>(bytes / 16);
    id<MTLCommandQueue> q = [dev newCommandQueue];

    auto gpu_pass = [&]() -> double {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:buf offset:0 atIndex:0];
        [enc setBuffer:out offset:0 atIndex:1];
        [enc setBytes:&n4 length:sizeof(n4) atIndex:2];
        const NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        [enc dispatchThreads:MTLSizeMake(65536, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        return (double)bytes / 1e9;  // GB moved
    };

    // ---------------- CPU only ----------------
    std::atomic<double> sink{0.0};
    auto cpu_pass = [&]() -> double {
        cpu_stream(host.data(), host.size(), &sink);
        return (double)bytes / 1e9;
    };

    // warm up
    gpu_pass(); cpu_pass();

    auto bench = [&](const char* name, std::function<double()> f) {
        double best = 0.0, total = 0.0;
        for (int i = 0; i < iters; ++i) {
            const double t0 = now_s();
            const double gb = f();
            const double dt = now_s() - t0;
            const double bw = gb / dt;
            best = std::max(best, bw);
            total += bw;
        }
        std::printf("  %-22s best %8.2f GB/s   mean %8.2f GB/s\n", name, best, total / iters);
        return best;
    };

    std::printf("\n=== single-unit streaming read bandwidth ===\n");
    const double cpu_only = bench("CPU (P-cores)", cpu_pass);
    const double gpu_only = bench("GPU (Metal)", gpu_pass);

    // ---------------- concurrent ----------------
    std::printf("\n=== CPU + GPU concurrently, on the same buffer ===\n");
    double best_conc = 0.0, sum = 0.0;
    for (int i = 0; i < iters; ++i) {
        const double t0 = now_s();
        double gpu_gb = 0.0;
        std::thread t([&]() { cpu_pass(); });
        gpu_gb = gpu_pass();
        t.join();
        const double dt = now_s() - t0;
        const double bw = (gpu_gb + (double)bytes / 1e9) / dt;
        best_conc = std::max(best_conc, bw);
        sum += bw;
    }
    const double conc_mean = sum / iters;
    std::printf("  %-22s best %8.2f GB/s   mean %8.2f GB/s\n", "CPU+GPU together", best_conc, conc_mean);

    // ---------------- verdict ----------------
    const double baseline = std::max(cpu_only, gpu_only);
    const double gain = best_conc / baseline;
    std::printf("\n=== verdict ===\n");
    std::printf("  best single unit      : %6.2f GB/s\n", baseline);
    std::printf("  concurrent CPU+GPU    : %6.2f GB/s\n", best_conc);
    std::printf("  aggregate gain        : %6.2fx\n", gain);
    std::printf("  (M1 spec peak         :  68.25 GB/s)\n\n");
    if (gain < 1.05) {
        std::printf("  -> ONE unit already saturates the bus. Multi-unit buys nothing.\n");
        std::printf("     The only lever is reading FEWER BYTES.\n");
    } else {
        std::printf("  -> Concurrency DOES add aggregate bandwidth. Multi-unit is a real lever:\n");
        std::printf("     split independent work (routed experts, parallel projections) across units.\n");
    }

    g.stop_monitor();
    return 0;
}
