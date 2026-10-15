// Obj-C++ implementation of the ANE entry points. See include/aero/ane.hpp.
//
// Build: clang++ -std=c++17 -ObjC++ -fobjc-arc -framework CoreML -framework Foundation
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include "aero/ane.hpp"

#include <string>
#include <vector>

namespace {

std::string g_last_error;

struct AneModel {
    MLModel* model = nil;
    MLModelConfiguration* cfg = nil;
    NSString* in_name = nil;
    NSString* out_name = nil;
    int64_t in_size = 0;
    int64_t out_size = 0;
    int units = AERO_ANE_NEURAL;
};

// Element count of a multi-array description, from its shape.
int64_t shape_size(MLFeatureDescription* d) {
    if (!d || !d.multiArrayConstraint) return 0;
    NSArray<NSNumber*>* shape = d.multiArrayConstraint.shape;
    int64_t n = 1;
    for (NSNumber* v in shape) n *= (int64_t)v.integerValue;
    return n;
}

}  // namespace

extern "C" {

void* aero_ane_open(const char* path, int compute_units) {
    g_last_error.clear();
    if (!path) { g_last_error = "null path"; return nullptr; }
    @autoreleasepool {
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
        MLModelConfiguration* cfg = [[MLModelConfiguration alloc] init];
        switch (compute_units) {
            case AERO_ANE_CPU:    cfg.computeUnits = MLComputeUnitsCPUOnly; break;
            case AERO_ANE_GPU:    cfg.computeUnits = MLComputeUnitsCPUAndGPU; break;
            case AERO_ANE_ALL:    cfg.computeUnits = MLComputeUnitsAll; break;
            case AERO_ANE_NEURAL:
            default:              cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine; break;
        }
        NSError* err = nil;
        MLModel* model = [MLModel modelWithContentsOfURL:url configuration:cfg error:&err];
        if (!model) {
            g_last_error = err ? std::string(err.localizedDescription.UTF8String)
                               : std::string("model failed to load (no error reported)");
            return nullptr;
        }
        auto* h = new AneModel();
        h->model = model;
        h->cfg = cfg;
        h->units = compute_units;
        // Feature names come FROM THE MODEL, never from an assumption.
        MLModelDescription* d = model.modelDescription;
        for (NSString* k in d.inputDescriptionsByName) { h->in_name = k; break; }
        for (NSString* k in d.outputDescriptionsByName) { h->out_name = k; break; }
        if (!h->in_name || !h->out_name) {
            g_last_error = "model has no single input/output pair";
            delete h;
            return nullptr;
        }
        h->in_size = shape_size(d.inputDescriptionsByName[h->in_name]);
        h->out_size = shape_size(d.outputDescriptionsByName[h->out_name]);
        if (h->in_size <= 0 || h->out_size <= 0) {
            g_last_error = "model input/output shape is not a multi-array";
            delete h;
            return nullptr;
        }
        return h;
    }
}

void aero_ane_close(void* p) {
    if (!p) return;
    @autoreleasepool {
        auto* h = static_cast<AneModel*>(p);
        h->model = nil;
        h->cfg = nil;
        delete h;
    }
}

const char* aero_ane_input_name(void* p) {
    auto* h = static_cast<AneModel*>(p);
    return (h && h->in_name) ? h->in_name.UTF8String : "";
}

const char* aero_ane_output_name(void* p) {
    auto* h = static_cast<AneModel*>(p);
    return (h && h->out_name) ? h->out_name.UTF8String : "";
}

int64_t aero_ane_input_size(void* p) {
    auto* h = static_cast<AneModel*>(p);
    return h ? h->in_size : 0;
}

int64_t aero_ane_output_size(void* p) {
    auto* h = static_cast<AneModel*>(p);
    return h ? h->out_size : 0;
}

int aero_ane_compute_units(void* p) {
    auto* h = static_cast<AneModel*>(p);
    return h ? h->units : -1;
}

int aero_ane_predict(void* p, const float* x, float* y) {
    auto* h = static_cast<AneModel*>(p);
    if (!h || !x || !y) { g_last_error = "predict: null handle or buffer"; return 0; }
    @autoreleasepool {
        NSError* err = nil;
        NSArray<NSNumber*>* shape =
            h->model.modelDescription.inputDescriptionsByName[h->in_name].multiArrayConstraint.shape;
        MLMultiArray* in = [[MLMultiArray alloc] initWithShape:shape
                                                      dataType:MLMultiArrayDataTypeFloat32
                                                         error:&err];
        if (!in) {
            g_last_error = err ? std::string(err.localizedDescription.UTF8String)
                               : std::string("input array allocation failed");
            return 0;
        }
        float* dst = (float*)in.dataPointer;
        for (int64_t i = 0; i < h->in_size; ++i) dst[i] = x[i];

        id<MLFeatureProvider> prov =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{h->in_name: in}
                                                              error:&err];
        if (!prov) {
            g_last_error = err ? std::string(err.localizedDescription.UTF8String)
                               : std::string("feature provider failed");
            return 0;
        }
        id<MLFeatureProvider> out = [h->model predictionFromFeatures:prov error:&err];
        if (!out) {
            g_last_error = err ? std::string(err.localizedDescription.UTF8String)
                               : std::string("prediction failed (no error reported)");
            return 0;
        }
        MLMultiArray* res = (MLMultiArray*)[out featureValueForName:h->out_name].multiArrayValue;
        if (!res) { g_last_error = "output feature is not a multi-array"; return 0; }
        const float* src = (const float*)res.dataPointer;
        for (int64_t i = 0; i < h->out_size; ++i) y[i] = src[i];
        return 1;
    }
}

const char* aero_ane_last_error(void) { return g_last_error.c_str(); }

}  // extern "C"
