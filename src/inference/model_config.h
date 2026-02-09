#pragma once
#include <cstdint>
#include <string>
#include <vector>

// ── Model metadata (parsed from model_metadata.json) ──

struct DiffusionConfig {
    int hidden_size = 0;
    int latent_size = 0;
    int head_layers = 0;
    int num_train_timesteps = 0;
    int num_inference_steps = 0;
    std::string beta_schedule;
    std::string prediction_type;
};

struct ModelMetadata {
    std::string model_type;             // "tts_0.5b", "tts_1.5b", "asr"
    int hidden_size = 0;
    int num_hidden_layers = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int intermediate_size = 0;
    int vocab_size = 0;
    int head_dim = 0;
    double rope_theta = 0.0;
    double rms_norm_eps = 0.0;
    int acoustic_vae_dim = 0;
    int sample_rate = 0;
    int hop_length = 0;

    // Optional fields (0.5b split model)
    int tts_backbone_num_hidden_layers = 0;
    int base_lm_layers = 0;
    int tts_lm_layers = 0;

    // Diffusion config (TTS models only)
    bool has_diffusion = false;
    DiffusionConfig diffusion;
};

bool loadModelMetadata(const std::string& jsonPath, ModelMetadata& out);

// ── Binary weight structures ──

// SpeechConnector: fc1 + RMSNorm + fc2
struct ConnectorWeights {
    uint32_t input_dim = 0;
    uint32_t output_dim = 0;
    std::vector<uint16_t> fc1_weight;   // [output_dim, input_dim] fp16
    std::vector<uint16_t> fc1_bias;     // [output_dim] fp16
    std::vector<uint16_t> norm_weight;  // [output_dim] fp16
    std::vector<uint16_t> fc2_weight;   // [output_dim, output_dim] fp16
    std::vector<uint16_t> fc2_bias;     // [output_dim] fp16
};

bool loadConnectorWeights(const std::string& binPath, ConnectorWeights& out);

// Embedding table
struct EmbeddingWeights {
    uint32_t num_embeddings = 0;
    uint32_t embedding_dim = 0;
    std::vector<uint16_t> data;         // [num_embeddings, embedding_dim] fp16
};

bool loadEmbeddingWeights(const std::string& binPath, EmbeddingWeights& out);

// Binary classifier (fc1 + fc2)
struct BinaryClassifierWeights {
    uint32_t hidden_size = 0;
    std::vector<uint16_t> fc1_weight;   // [hidden_size, hidden_size] fp16
    std::vector<uint16_t> fc1_bias;     // [hidden_size] fp16
    std::vector<uint16_t> fc2_weight;   // [1, hidden_size] fp16
    std::vector<uint16_t> fc2_bias;     // [1] fp16
};

bool loadBinaryClassifierWeights(const std::string& binPath, BinaryClassifierWeights& out);

// Voice preset (streaming model binary format)
struct VoicePresetGroup {
    uint32_t num_layers = 0;
    uint32_t num_kv_heads = 0;
    uint32_t seq_len = 0;
    std::vector<uint16_t> last_hidden_state;    // [seq_len, hidden_size] fp16
    // key_cache[layer]: [num_kv_heads, seq_len, head_dim] fp16
    // value_cache[layer]: same shape
    std::vector<std::vector<uint16_t>> key_cache;
    std::vector<std::vector<uint16_t>> value_cache;
};

struct VoicePreset {
    uint32_t hidden_size = 0;
    uint32_t head_dim = 0;
    uint32_t num_groups = 0;
    std::vector<VoicePresetGroup> groups; // "lm", "tts_lm"
};

bool loadVoicePreset(const std::string& binPath, VoicePreset& out);

// Scalar (single fp32 value, 4 bytes)
bool loadScalar(const std::string& binPath, float& out);
