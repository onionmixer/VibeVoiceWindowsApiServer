#include "inference/kv_cache.h"
#include "utils/cuda_check.h"
#include <cstdio>
#include <cstring>

bool KVCache::init(int numLayers, int numKVHeads, int headDim, int maxSeqLen) {
    numLayers_ = numLayers;
    numKVHeads_ = numKVHeads;
    headDim_ = headDim;
    maxSeqLen_ = maxSeqLen;
    seqLen_ = 0;
    bufIdx_ = 0;

    // Each buffer: [1, numKVHeads, maxSeqLen, headDim] fp16
    size_t bufSize = (size_t)1 * numKVHeads * maxSeqLen * headDim * sizeof(uint16_t);

    keys_.resize(numLayers * 2);
    values_.resize(numLayers * 2);

    for (int i = 0; i < numLayers * 2; ++i) {
        if (!keys_[i].resize(bufSize)) {
            fprintf(stderr, "KVCache: failed to allocate key buffer %d (%zu bytes)\n", i, bufSize);
            return false;
        }
        if (!values_[i].resize(bufSize)) {
            fprintf(stderr, "KVCache: failed to allocate value buffer %d (%zu bytes)\n", i, bufSize);
            return false;
        }
        // Zero-initialize
        cudaMemset(keys_[i].data(), 0, bufSize);
        cudaMemset(values_[i].data(), 0, bufSize);
    }

    fprintf(stderr, "KVCache: allocated %d layers, %d kv_heads, head_dim=%d, max_seq=%d (%.1f MB)\n",
            numLayers, numKVHeads, headDim, maxSeqLen,
            (float)(numLayers * 2 * 2 * bufSize) / (1024.0f * 1024.0f));
    return true;
}

bool KVCache::loadFromPreset(const VoicePresetGroup& group, int /*hiddenSize*/, int headDim) {
    // Initialize with the group's parameters
    if (!init(group.num_layers, group.num_kv_heads, headDim, 4096)) {
        return false;
    }

    seqLen_ = group.seq_len;

    // Upload pre-computed KV-cache for each layer
    // Source shape: [num_kv_heads, seq_len, head_dim] fp16
    // Dest shape: [1, num_kv_heads, maxSeqLen, head_dim] fp16
    for (int layer = 0; layer < (int)group.num_layers; ++layer) {
        // Upload to bufIdx_=0 (the "past" buffer)
        int idx = layer * 2 + bufIdx_;
        uint8_t* keyDst = (uint8_t*)keys_[idx].data();
        uint8_t* valDst = (uint8_t*)values_[idx].data();

        // Source is [num_kv_heads, seq_len, head_dim] contiguous
        // Dest is [1, num_kv_heads, maxSeqLen, head_dim] contiguous
        // For each head: copy seq_len rows of head_dim to the start of maxSeqLen rows
        for (int h = 0; h < (int)group.num_kv_heads; ++h) {
            size_t dstOffset = (size_t)h * maxSeqLen_ * headDim * sizeof(uint16_t);
            size_t copyBytes = (size_t)group.seq_len * headDim * sizeof(uint16_t);

            cudaError_t err = cudaMemcpy(keyDst + dstOffset,
                                          group.key_cache[layer].data() + h * group.seq_len * headDim,
                                          copyBytes, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                fprintf(stderr, "KVCache: failed to upload key cache layer %d head %d: %s\n",
                        layer, h, cudaGetErrorString(err));
                return false;
            }

            err = cudaMemcpy(valDst + dstOffset,
                              group.value_cache[layer].data() + h * group.seq_len * headDim,
                              copyBytes, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                fprintf(stderr, "KVCache: failed to upload value cache layer %d head %d: %s\n",
                        layer, h, cudaGetErrorString(err));
                return false;
            }
        }
    }

    return true;
}

void* KVCache::pastKeyPtr(int layer) {
    return keys_[layer * 2 + bufIdx_].data();
}

void* KVCache::pastValuePtr(int layer) {
    return values_[layer * 2 + bufIdx_].data();
}

void* KVCache::presentKeyPtr(int layer) {
    return keys_[layer * 2 + (1 - bufIdx_)].data();
}

void* KVCache::presentValuePtr(int layer) {
    return values_[layer * 2 + (1 - bufIdx_)].data();
}

void KVCache::advance() {
    seqLen_++;
    bufIdx_ = 1 - bufIdx_;
}

void KVCache::reset() {
    seqLen_ = 0;
    bufIdx_ = 0;
}
