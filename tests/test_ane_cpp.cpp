// The ANE runtime, exercised in C++ against a real compiled model.
//
// There is no public ANE API: CoreML is the route, and CPU_AND_NE is what dispatches to
// the unit. This loads an mlmodelc that already exists on the machine, reads its feature
// names FROM THE MODEL (never assumed), runs a prediction and checks the result is finite.
// That proves the runtime half of the ANE path before the authoring half (our own matmul
// model, compiled by coremlcompiler) exists -- and it does it with no Python anywhere.
#include "aero/ane.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static int g_fail = 0, g_checks = 0;
static void check(bool ok, const char* what) {
    g_checks++;
    std::printf(ok ? "  ok   %s\n" : "  FAIL %s\n", what);
    if (!ok) g_fail++;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("(skipped: give an .mlmodelc path)\n\n=== 0 checks, 0 failures ===\n");
        return 0;
    }
    for (int units : {AERO_ANE_NEURAL, AERO_ANE_ALL}) {
        const char* u = units == AERO_ANE_NEURAL ? "CPU_AND_NE (ANE)" : "ALL";
        std::printf("[%s] %s\n", u, argv[1]);
        void* h = aero_ane_open(argv[1], units);
        if (!h) {
            std::printf("  (model refused by CoreML: %s)\n", aero_ane_last_error());
            continue;
        }
        check(aero_ane_compute_units(h) == units, "loaded with the requested compute units");
        check(std::string(aero_ane_input_name(h)).size() > 0,
              (std::string("read the input feature name from the model: ") +
               aero_ane_input_name(h)).c_str());
        const int64_t ni = aero_ane_input_size(h), no = aero_ane_output_size(h);
        std::printf("       in=%s (%lld)  out=%s (%lld)\n", aero_ane_input_name(h),
                    (long long)ni, aero_ane_output_name(h), (long long)no);
        check(ni > 0 && no > 0, "input and output shapes are multi-arrays");

        std::vector<float> x((size_t)ni, 0.0f), y((size_t)no, 0.0f);
        if (aero_ane_predict(h, x.data(), y.data()) == 1) {
            check(true, "a prediction on the ANE path returned success");
            bool finite = true;
            for (float v : y) if (!std::isfinite(v)) finite = false;
            check(finite, "every output element is finite");
            std::printf("       y[0]=%+.6f y[1]=%+.6f\n", y[0], no > 1 ? y[1] : 0.0f);
        } else {
            check(false, (std::string("prediction failed: ") + aero_ane_last_error()).c_str());
        }
        aero_ane_close(h);
        std::printf("\n");
    }
    std::printf("=== %d checks, %d failures ===\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
