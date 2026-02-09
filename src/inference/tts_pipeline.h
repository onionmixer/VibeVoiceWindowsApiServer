#pragma once
#include "inference/trt_engine.h"
#include "inference/model_config.h"
#include "inference/kv_cache.h"
#include "inference/dpm_solver.h"
#include "inference/gpu_ops.h"
#include "tokenizer/bpe_tokenizer.h"
#include "tokenizer/special_tokens.h"
#include "audio/audio_io.h"
#include "utils/cuda_buffer.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <map>
#include <string>
#include <vector>

class TTSPipeline {
public:
    enum class ModelType { STREAMING_0_5B, FULL_1_5B };

    struct Config {
        ModelType type;
        std::string engineDir;         // "engines/tts_0.5b"
        std::string weightsDir;        // "onnx/tts_0.5b/weights"
        std::string voicesDir;         // "voices/streaming_model"
        std::string metadataPath;      // "onnx/tts_0.5b/model_metadata.json"
        std::string tokenizerPath;     // "tokenizer/qwen2.5-0.5b/tokenizer.json"
        std::string specialTokensPath; // "tokenizer/qwen2.5-0.5b/special_tokens.json"
        float cfgScale = 1.5f;
        int inferenceSteps = 5;        // DPM solver steps (configurable: 5-20)
    };

    struct Request {
        std::string text;
        std::string voice;   // voice name (e.g. "en-Carter_man")
        float speed = 1.0f;
    };

    struct Result {
        std::vector<float> audio;
        int sampleRate = 24000;
        bool ok = false;
        std::string error;
    };

    TTSPipeline();
    ~TTSPipeline();

    bool load(const Config& cfg);
    Result synthesize(const Request& req);
    std::vector<std::string> availableVoices() const;
    bool isLoaded() const { return loaded_; }

private:
    Result synth05B(const Request& req);
    Result synth15B(const Request& req);

    // Diffusion: returns latent [64] fp32 on CPU
    std::vector<float> sampleSpeechTokens(
        const __half* posCondGpu, const __half* negCondGpu,
        int hiddenSize, float cfgScale);

    // SpeechConnector: latent[64] -> embed[hidden_size] on GPU
    void runConnector(const __half* latentGpu, __half* outputGpu,
                      const CudaBuffer& fc1W, const CudaBuffer& fc1B,
                      const CudaBuffer& normW,
                      const CudaBuffer& fc2W, const CudaBuffer& fc2B,
                      int inputDim, int outputDim);

    // EOS classifier: hidden[hidden_size] -> sigmoid score
    float runEosClassifier(const __half* hiddenGpu);

    // Bind KV-cache buffers to a TRT engine for decode
    void bindKVCache(TRTEngine& engine, KVCache& kv);

    // Load all voice presets from directory
    bool loadVoicePresets();

    // Load GPU weights from binary files
    bool loadWeights();

    Config cfg_;
    ModelMetadata meta_;
    ModelType type_;
    bool loaded_ = false;

    // TRT engines
    TRTEngine baseLmPrefill_, baseLmDecode_;     // 0.5B only
    TRTEngine ttsLmPrefill_, ttsLmDecode_;       // 0.5B only
    TRTEngine lmPrefill_, lmDecode_;             // 1.5B only
    TRTEngine acousticEncoder_;                   // 1.5B only
    TRTEngine diffusionHead_;
    TRTEngine acousticDecoder_;

    // KV-cache (0.5B: 4 caches for CFG, 1.5B: 1 cache)
    KVCache baseLmKV_, ttsLmKV_;
    KVCache negBaseLmKV_, negTtsLmKV_;
    KVCache lmKV_;

    // GPU weight buffers
    CudaBuffer embedTokensGpu_;                   // [vocab_size, hidden_size] fp16
    CudaBuffer ttsInputTypesGpu_;                  // 0.5B: [2, hidden_size] fp16
    CudaBuffer connFc1W_, connFc1B_, connNormW_;   // connector weights
    CudaBuffer connFc2W_, connFc2B_;
    CudaBuffer eosFc1W_, eosFc1B_;                 // 0.5B: EOS classifier
    CudaBuffer eosFc2W_, eosFc2B_;
    CudaBuffer semanticConnFc1W_, semanticConnFc1B_; // 1.5B: semantic connector
    CudaBuffer semanticConnNormW_;
    CudaBuffer semanticConnFc2W_, semanticConnFc2B_;

    float speechScalingFactor_ = 0.0f;
    float speechBiasFactor_ = 0.0f;

    // Connector dimensions
    int connInputDim_ = 0;
    int connOutputDim_ = 0;

    // EOS classifier hidden size
    int eosHiddenSize_ = 0;

    // Solver
    DPMSolver dpm_;

    // Tokenizer
    BPETokenizer tokenizer_;
    TTSSpecialTokens specialTokens_;

    // Voice presets (0.5B: binary presets, 1.5B: WAV paths)
    std::map<std::string, VoicePreset> voicePresets_;
    std::map<std::string, std::string> voiceFiles_;

    // CUDA resources
    cudaStream_t stream_ = nullptr;
    cublasHandle_t cublas_ = nullptr;

    // Scratch buffers
    CudaBuffer scratchEmbed_;      // [1, hidden_size] fp16
    CudaBuffer scratchHidden_;     // [1, hidden_size] fp16
    CudaBuffer scratchHidden2_;    // [1, hidden_size] fp16 (for neg path)
    CudaBuffer scratchLatent_;     // [64] fp16
    CudaBuffer scratchLatentF32_;  // [64] fp32
    CudaBuffer scratchAudio_;      // [1, 1, max_audio_len] fp16
    CudaBuffer scratchCondition_;  // [2, hidden_size] fp16 (for CFG batched diffusion)
    CudaBuffer scratchNoisy_;      // [2, 64] fp16 (for CFG batched diffusion)
    CudaBuffer scratchVpred_;      // [2, 64] fp16 (diffusion head output)
    CudaBuffer scratchTimesteps_;  // [2] int64
    CudaBuffer positionIdsBuf_;    // [1, 1] int64
    CudaBuffer inputIdBuf_;        // [1] int32
    CudaBuffer connScratch_;       // [1, outputDim] fp16 (connector intermediate)
};
