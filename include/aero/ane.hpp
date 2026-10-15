// The ANE, reached from the engine. Objective-C++ over the CoreML runtime.
//
// WHY COREML AND NOT SOMETHING ELSE. There is no public API to the Neural Engine. CoreML
// is the only sanctioned route, and `MLComputeUnitsCPUAndNeuralEngine` is what dispatches
// an op to it. This is the engine-side half: load a compiled model and run it. Everything
// here is Obj-C++/C -- no Python anywhere in the chain.
//
// The model itself is authored in C++ as a protobuf and compiled with Apple's own
// coremlcompiler (present at
//   /Volumes/Backup/Applications/Xcode.app/.../usr/bin/coremlcompiler
// which consumes .mlmodel / .mlpackage and emits .mlmodelc).
//
// WHAT THIS DELIBERATELY DOES NOT DO: guess the feature names. A model's input/output
// feature names are read from the model description at load, so this works with a model
// authored by us AND with any .mlmodelc on the machine -- which is how it gets tested
// before the authoring half exists.
#pragma once

#include <cstdint>

extern "C" {

// Compute-unit selection, mirroring the engine's Device enum.
enum {
    AERO_ANE_CPU      = 0,
    AERO_ANE_GPU      = 1,
    AERO_ANE_NEURAL   = 2,   // CPU_AND_NE: this is the ANE path
    AERO_ANE_ALL      = 3,
};

// Open a compiled model (.mlmodelc). Returns NULL on failure; the reason is available
// from aero_ane_last_error(). Never throws across the boundary.
void* aero_ane_open(const char* mlmodelc_path, int compute_units);
void  aero_ane_close(void* h);

// Feature names, read from the model description (not assumed).
const char* aero_ane_input_name(void* h);
const char* aero_ane_output_name(void* h);
int64_t     aero_ane_input_size(void* h);
int64_t     aero_ane_output_size(void* h);

// One prediction. x has aero_ane_input_size() floats, y receives
// aero_ane_output_size(). Returns 1 on success, 0 on refusal.
int aero_ane_predict(void* h, const float* x, float* y);

// Which compute units the loaded model is actually configured for.
int aero_ane_compute_units(void* h);

const char* aero_ane_last_error(void);

}
