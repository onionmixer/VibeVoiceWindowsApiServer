#pragma once
#include "utils/cuda_buffer.h"
#include "inference/model_config.h"
#include <vector>

class KVCache {
public:
    KVCache() = default;
    ~KVCache() = default;
    KVCache(KVCache&&) = default;
    KVCache& operator=(KVCache&&) = default;

    // Allocate double-buffered GPU memory for all layers
    // useFp16=false: allocate fp32 buffers (default for 1.5B FP32 and 0.5B engines)
    // useFp16=true: allocate fp16 buffers (legacy, not used for 1.5B)
    bool init(int numLayers, int numKVHeads, int headDim, int maxSeqLen, bool useFp16 = false);

    // Load pre-computed KV-cache from VoicePresetGroup
    bool loadFromPreset(const VoicePresetGroup& group, int hiddenSize, int headDim);

    // Get buffer pointers for TRT binding (step-dependent which is past/present)
    void* pastKeyPtr(int layer);
    void* pastValuePtr(int layer);
    void* presentKeyPtr(int layer);
    void* presentValuePtr(int layer);

    // After TRT execution: seqLen++, swap past<->present
    void advance();

    // After prefill: swap once and set seqLen to prefilled length.
    // Prefill writes to present buffers; one swap makes them become past.
    void advanceAfterPrefill(int prefillLen);

    // Reset to empty state
    void reset();

    int seqLen() const { return seqLen_; }
    int numLayers() const { return numLayers_; }
    int numKVHeads() const { return numKVHeads_; }
    int headDim() const { return headDim_; }
    int maxSeqLen() const { return maxSeqLen_; }
    bool isFp16() const { return fp16_; }

private:
    int numLayers_ = 0, numKVHeads_ = 0, headDim_ = 0;
    int seqLen_ = 0, maxSeqLen_ = 0;
    int bufIdx_ = 0; // 0 or 1
    bool fp16_ = false;

    // keys_[layer * 2 + bufIdx]: CudaBuffer sized [1, numKVHeads, maxSeqLen, headDim]
    // Precision depends on useFp16 flag: fp32 for 1.5B (default), fp16 for legacy
    std::vector<CudaBuffer> keys_;   // [numLayers * 2]
    std::vector<CudaBuffer> values_; // [numLayers * 2]
};
