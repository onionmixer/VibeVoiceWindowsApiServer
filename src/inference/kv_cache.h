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
    bool init(int numLayers, int numKVHeads, int headDim, int maxSeqLen);

    // Load pre-computed KV-cache from VoicePresetGroup
    bool loadFromPreset(const VoicePresetGroup& group, int hiddenSize, int headDim);

    // Get buffer pointers for TRT binding (step-dependent which is past/present)
    void* pastKeyPtr(int layer);
    void* pastValuePtr(int layer);
    void* presentKeyPtr(int layer);
    void* presentValuePtr(int layer);

    // After TRT execution: seqLen++, swap past<->present
    void advance();

    // Reset to empty state
    void reset();

    int seqLen() const { return seqLen_; }
    int numLayers() const { return numLayers_; }
    int numKVHeads() const { return numKVHeads_; }
    int headDim() const { return headDim_; }
    int maxSeqLen() const { return maxSeqLen_; }

private:
    int numLayers_ = 0, numKVHeads_ = 0, headDim_ = 0;
    int seqLen_ = 0, maxSeqLen_ = 0;
    int bufIdx_ = 0; // 0 or 1

    // keys_[layer * 2 + bufIdx]: CudaBuffer sized [1, numKVHeads, maxSeqLen, headDim] fp16
    std::vector<CudaBuffer> keys_;   // [numLayers * 2]
    std::vector<CudaBuffer> values_; // [numLayers * 2]
};
