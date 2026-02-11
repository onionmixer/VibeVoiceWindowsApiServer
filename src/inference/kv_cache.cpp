#include "inference/kv_cache.h"
#include "utils/cuda_check.h"
#include "utils/logger.h"
#include <cstring>

// CPU fp16 -> fp32 conversion (IEEE 754 half to single precision)
static float fp16BitsToFloat(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t man  = h & 0x3ffu;
    uint32_t result;

    if (exp == 0) {
        if (man == 0) {
            result = sign; // +/- zero
        } else {
            // Subnormal: normalize
            exp = 1;
            while (!(man & 0x400u)) { man <<= 1; exp--; }
            man &= 0x3ffu;
            result = sign | ((exp + 112) << 23) | (man << 13);
        }
    } else if (exp == 31) {
        result = sign | 0x7f800000u | (man << 13); // Inf/NaN
    } else {
        result = sign | ((exp + 112) << 23) | (man << 13); // Normal
    }

    float f;
    std::memcpy(&f, &result, sizeof(float));
    return f;
}

bool KVCache::init(int numLayers, int numKVHeads, int headDim, int maxSeqLen, bool useFp16) {
    numLayers_ = numLayers;
    numKVHeads_ = numKVHeads;
    headDim_ = headDim;
    maxSeqLen_ = maxSeqLen;
    seqLen_ = 0;
    bufIdx_ = 0;
    fp16_ = useFp16;

    // Each buffer: [1, numKVHeads, maxSeqLen, headDim]
    // fp16 for 1.5B (ONNX exported in fp16), fp32 for 0.5B (ONNX exported in fp32)
    size_t elemSize = useFp16 ? sizeof(uint16_t) : sizeof(float);
    size_t bufSize = (size_t)1 * numKVHeads * maxSeqLen * headDim * elemSize;

    keys_.resize(numLayers * 2);
    values_.resize(numLayers * 2);

    for (int i = 0; i < numLayers * 2; ++i) {
        if (!keys_[i].resize(bufSize)) {
            LOG_ERROR("KV", "failed to allocate key buffer %d (%zu bytes)", i, bufSize);
            return false;
        }
        if (!values_[i].resize(bufSize)) {
            LOG_ERROR("KV", "failed to allocate value buffer %d (%zu bytes)", i, bufSize);
            return false;
        }
        // Zero-initialize
        cudaMemset(keys_[i].data(), 0, bufSize);
        cudaMemset(values_[i].data(), 0, bufSize);
    }

    LOG_INFO("KV", "allocated %d layers, %d kv_heads, head_dim=%d, max_seq=%d (%.1f MB %s)",
             numLayers, numKVHeads, headDim, maxSeqLen,
             (float)(numLayers * 2 * 2 * bufSize) / (1024.0f * 1024.0f),
             useFp16 ? "fp16" : "fp32");
    return true;
}

bool KVCache::loadFromPreset(const VoicePresetGroup& group, int /*hiddenSize*/, int headDim) {
    // Initialize with the group's parameters
    if (!init(group.num_layers, group.num_kv_heads, headDim, 4096)) {
        return false;
    }

    seqLen_ = group.seq_len;

    // Convert preset fp16 KV data to fp32 on CPU, then upload to GPU
    // Source shape: [num_kv_heads, seq_len, head_dim] fp16
    // Dest shape: [1, num_kv_heads, maxSeqLen, head_dim] fp32
    size_t headSlice = (size_t)group.seq_len * headDim;
    std::vector<float> fp32Buf(headSlice);  // reusable temp buffer per head

    for (int layer = 0; layer < (int)group.num_layers; ++layer) {
        int idx = layer * 2 + bufIdx_;
        uint8_t* keyDst = (uint8_t*)keys_[idx].data();
        uint8_t* valDst = (uint8_t*)values_[idx].data();

        for (int h = 0; h < (int)group.num_kv_heads; ++h) {
            size_t srcOff = (size_t)h * headSlice;
            size_t dstOffset = (size_t)h * maxSeqLen_ * headDim * sizeof(float);
            size_t copyBytes = headSlice * sizeof(float);

            // Convert key fp16 -> fp32 on CPU
            const uint16_t* keySrc = group.key_cache[layer].data() + srcOff;
            for (size_t i = 0; i < headSlice; ++i)
                fp32Buf[i] = fp16BitsToFloat(keySrc[i]);

            cudaError_t err = cudaMemcpy(keyDst + dstOffset, fp32Buf.data(),
                                          copyBytes, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                LOG_ERROR("KV", "failed to upload key cache layer %d head %d: %s",
                         layer, h, cudaGetErrorString(err));
                return false;
            }

            // Convert value fp16 -> fp32 on CPU
            const uint16_t* valSrc = group.value_cache[layer].data() + srcOff;
            for (size_t i = 0; i < headSlice; ++i)
                fp32Buf[i] = fp16BitsToFloat(valSrc[i]);

            err = cudaMemcpy(valDst + dstOffset, fp32Buf.data(),
                              copyBytes, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                LOG_ERROR("KV", "failed to upload value cache layer %d head %d: %s",
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

void KVCache::advanceAfterPrefill(int prefillLen) {
    // Prefill writes all positions to present buffers in one shot.
    // One swap makes present become past (so decode reads the prefill data).
    bufIdx_ = 1 - bufIdx_;
    seqLen_ = prefillLen;
}

void KVCache::reset() {
    seqLen_ = 0;
    bufIdx_ = 0;
}
