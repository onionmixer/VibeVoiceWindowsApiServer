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
    // 4-stage fallback voice resolution (exact -> alias -> substring -> first available)
    std::string resolveVoice(const std::string& voice) const;
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

    // Bind KV-cache buffers to a TRT engine
    // prefillMode=true: only bind present outputs (prefill engine has no past inputs)
    // prefillMode=false: bind both past inputs and present outputs (decode engine)
    void bindKVCache(TRTEngine& engine, KVCache& kv, bool prefillMode = false);

    // Run one step of LM decode with proper fp16<->fp32 conversion
    // inputGpu: fp16 embeddings, outputGpu: fp16 hidden states
    void runLmDecode(TRTEngine& engine, KVCache& kv,
                     const __half* inputGpu, __half* outputGpu,
                     int64_t positionId);

    // 1.5B LM helpers: prefill and decode with logits output (FP16 ONNX engines)
    // runLmPrefill: runs LM prefill with embedding sequence, fills KV cache
    // embedsFp16Gpu: [1, S, H] fp16, logitsF32Gpu: [1, S, vocab] fp32, hiddenFp16Gpu: [1, S, H] fp16
    void runLmPrefill(TRTEngine& engine, KVCache& kv,
                      __half* embedsFp16Gpu, int seqLen,
                      float* logitsF32Gpu, __half* hiddenFp16Gpu);

    // runLmDecodeWithLogits: single-token decode returning logits + hidden (for 1.5B unified LM)
    // All tensor I/O is fp16 except logits (fp32) and attention_mask (fp32)
    // inputGpu: fp16 embedding, hiddenOutGpu: fp16 hidden state, logitsF32Gpu: fp32 logits [vocab]
    void runLmDecodeWithLogits(TRTEngine& engine, KVCache& kv,
                               const __half* inputGpu, __half* hiddenOutGpu,
                               float* logitsF32Gpu, int64_t positionId);

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

    // KV-cache (0.5B: 4 caches for CFG, 1.5B: 2 caches pos+neg)
    KVCache baseLmKV_, ttsLmKV_;
    KVCache negBaseLmKV_, negTtsLmKV_;
    KVCache lmKV_;
    KVCache negLmKV_;                                  // 1.5B: negative path for CFG

    // Semantic encoder (1.5B only)
    TRTEngine semanticEncoder_;

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

    // Connector dimensions (acoustic)
    int connInputDim_ = 0;
    int connOutputDim_ = 0;

    // Semantic connector dimensions (1.5B)
    int semConnInputDim_ = 0;
    int semConnOutputDim_ = 0;

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

    // Scratch buffers (fp16 — used for GPU-side operations: embedding, vectorAdd, connector, etc.)
    CudaBuffer scratchEmbed_;      // [1, hidden_size] fp16
    CudaBuffer scratchHidden_;     // [1, hidden_size] fp16
    CudaBuffer scratchHidden2_;    // [1, hidden_size] fp16 (for neg path)
    CudaBuffer scratchLatent_;     // [64] fp16
    CudaBuffer scratchLatentF32_;  // [64] fp32
    CudaBuffer scratchAudio_;      // [1, 1, max_audio_len] fp16
    CudaBuffer scratchTimesteps_;  // [2] int32
    CudaBuffer positionIdsBuf_;    // [1, 1] int32
    CudaBuffer inputIdBuf_;        // [1] int32
    CudaBuffer connScratch_;       // [1, outputDim] fp16 (connector intermediate)

    // fp32 staging buffers for TRT engine I/O
    // (0.5B engines have fp32 I/O; 1.5B uses fp16 embeds/hidden but fp32 logits/mask)
    CudaBuffer stagingEmbedsF32_;       // [H] fp32 — 0.5B LM decode input (fp16→fp32 conversion)
    CudaBuffer stagingHiddenF32_;       // [H] fp32 — 0.5B LM decode output (fp32→fp16 conversion)
    CudaBuffer stagingCondF32_;         // [2*H] fp32 — diffusion condition
    CudaBuffer stagingDecoderInF32_;    // [64] fp32 — acoustic decoder input

    // Per-request scratch buffers (promoted from local to avoid cudaMalloc/Free per request)
    CudaBuffer scratchDenormLatent_;    // synth05B: denormLatent [64] fp16
    CudaBuffer scratchAudioF32_;        // synth05B: audioF32Gpu [maxHop] fp32 (also decoder output)
    CudaBuffer scratchSampleF32_;       // sampleSpeechTokens: sampleF32Gpu [2*64] fp32 (also diffusion noisy input)
    CudaBuffer scratchVpredF32_;        // sampleSpeechTokens: vpredF32Gpu [2*64] fp32 (also diffusion output)
    CudaBuffer scratchEosHidden_;       // runEosClassifier: fc1Out [hidden] fp16
    CudaBuffer scratchEosOut_;          // runEosClassifier: fc2Out [1] fp16
    CudaBuffer scratchEosSigmoid_;      // runEosClassifier: sigmoidOut [1] fp32

    // 1.5B-specific scratch buffers
    CudaBuffer stagingLogitsF32_;       // [vocab_size] fp32 — logits from LM decode
    CudaBuffer stagingPrefillEmbedsF32_;// [maxPrefillSeq, H] fp32 — prefill input
    CudaBuffer stagingPrefillLogitsF32_;// [maxPrefillSeq, vocab] fp32 — prefill logits output
    CudaBuffer stagingPrefillHiddenF32_;// [maxPrefillSeq, H] fp32 — prefill hidden output
    CudaBuffer scratchSemanticLatent_;  // [semantic_vae_dim] fp16 — semantic encoder output
    CudaBuffer scratchSemanticEmbed_;   // [H] fp16 — semantic connector output
    CudaBuffer scratchVoiceEmbedF32_;   // [T, H] fp32 — voice embeddings for prefill

    // Persistent decode mask buffer (avoids freeing while TRT still reads async)
    CudaBuffer decodeMaskBuf_;
};
