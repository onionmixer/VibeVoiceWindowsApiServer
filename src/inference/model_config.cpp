#include "inference/model_config.h"
#include "utils/logger.h"
#include <json.hpp>
#include <fstream>
#include <cstring>

using json = nlohmann::json;

// ── Helpers ──

static bool readFile(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        LOG_ERROR("CFG", "Cannot open file: %s", path.c_str());
        return false;
    }
    size_t sz = (size_t)f.tellg();
    f.seekg(0);
    out.resize(sz);
    f.read(reinterpret_cast<char*>(out.data()), sz);
    return f.good();
}

template <typename T>
static bool readVec(const uint8_t*& ptr, const uint8_t* end, std::vector<T>& v, size_t count) {
    size_t bytes = count * sizeof(T);
    if (ptr + bytes > end) return false;
    v.resize(count);
    memcpy(v.data(), ptr, bytes);
    ptr += bytes;
    return true;
}

template <typename T>
static bool readVal(const uint8_t*& ptr, const uint8_t* end, T& val) {
    if (ptr + sizeof(T) > end) return false;
    memcpy(&val, ptr, sizeof(T));
    ptr += sizeof(T);
    return true;
}

// ── ModelMetadata ──

bool loadModelMetadata(const std::string& jsonPath, ModelMetadata& out) {
    std::ifstream f(jsonPath);
    if (!f.is_open()) {
        LOG_ERROR("CFG", "Cannot open model metadata: %s", jsonPath.c_str());
        return false;
    }

    json j;
    try {
        f >> j;
    } catch (const json::parse_error& e) {
        LOG_ERROR("CFG", "JSON parse error in %s: %s", jsonPath.c_str(), e.what());
        return false;
    }

    out.model_type = j.value("model_type", "");
    out.hidden_size = j.value("hidden_size", 0);
    out.num_hidden_layers = j.value("num_hidden_layers", 0);
    out.num_attention_heads = j.value("num_attention_heads", 0);
    out.num_key_value_heads = j.value("num_key_value_heads", 0);
    out.intermediate_size = j.value("intermediate_size", 0);
    out.vocab_size = j.value("vocab_size", 0);
    out.head_dim = j.value("head_dim", 0);
    out.rope_theta = j.value("rope_theta", 0.0);
    out.rms_norm_eps = j.value("rms_norm_eps", 0.0);
    out.acoustic_vae_dim = j.value("acoustic_vae_dim", 0);
    out.semantic_vae_dim = j.value("semantic_vae_dim", 0);
    out.sample_rate = j.value("sample_rate", 0);
    out.hop_length = j.value("hop_length", 0);

    out.tts_backbone_num_hidden_layers = j.value("tts_backbone_num_hidden_layers", 0);
    out.base_lm_layers = j.value("base_lm_layers", 0);
    out.tts_lm_layers = j.value("tts_lm_layers", 0);

    if (j.contains("diffusion")) {
        out.has_diffusion = true;
        auto& d = j["diffusion"];
        out.diffusion.hidden_size = d.value("hidden_size", 0);
        out.diffusion.latent_size = d.value("latent_size", 0);
        out.diffusion.head_layers = d.value("head_layers", 0);
        out.diffusion.num_train_timesteps = d.value("num_train_timesteps", 0);
        out.diffusion.num_inference_steps = d.value("num_inference_steps", 0);
        out.diffusion.beta_schedule = d.value("beta_schedule", "");
        out.diffusion.prediction_type = d.value("prediction_type", "");
    }

    return true;
}

// ── ConnectorWeights ──

bool loadConnectorWeights(const std::string& binPath, ConnectorWeights& out) {
    std::vector<uint8_t> buf;
    if (!readFile(binPath, buf)) return false;

    const uint8_t* ptr = buf.data();
    const uint8_t* end = ptr + buf.size();

    if (!readVal(ptr, end, out.input_dim)) return false;
    if (!readVal(ptr, end, out.output_dim)) return false;

    size_t od = out.output_dim;
    size_t id = out.input_dim;

    if (!readVec(ptr, end, out.fc1_weight, od * id)) return false;
    if (!readVec(ptr, end, out.fc1_bias, od)) return false;
    if (!readVec(ptr, end, out.norm_weight, od)) return false;
    if (!readVec(ptr, end, out.fc2_weight, od * od)) return false;
    if (!readVec(ptr, end, out.fc2_bias, od)) return false;

    return true;
}

// ── EmbeddingWeights ──

bool loadEmbeddingWeights(const std::string& binPath, EmbeddingWeights& out) {
    std::vector<uint8_t> buf;
    if (!readFile(binPath, buf)) return false;

    const uint8_t* ptr = buf.data();
    const uint8_t* end = ptr + buf.size();

    if (!readVal(ptr, end, out.num_embeddings)) return false;
    if (!readVal(ptr, end, out.embedding_dim)) return false;

    size_t count = (size_t)out.num_embeddings * out.embedding_dim;
    if (!readVec(ptr, end, out.data, count)) return false;

    return true;
}

// ── BinaryClassifierWeights ──

bool loadBinaryClassifierWeights(const std::string& binPath, BinaryClassifierWeights& out) {
    std::vector<uint8_t> buf;
    if (!readFile(binPath, buf)) return false;

    const uint8_t* ptr = buf.data();
    const uint8_t* end = ptr + buf.size();

    if (!readVal(ptr, end, out.hidden_size)) return false;

    size_t h = out.hidden_size;
    if (!readVec(ptr, end, out.fc1_weight, h * h)) return false;
    if (!readVec(ptr, end, out.fc1_bias, h)) return false;
    if (!readVec(ptr, end, out.fc2_weight, 1 * h)) return false;
    if (!readVec(ptr, end, out.fc2_bias, 1)) return false;

    return true;
}

// ── VoicePreset ──

bool loadVoicePreset(const std::string& binPath, VoicePreset& out) {
    std::vector<uint8_t> buf;
    if (!readFile(binPath, buf)) return false;

    const uint8_t* ptr = buf.data();
    const uint8_t* end = ptr + buf.size();

    // File header (32 bytes)
    uint32_t magic = 0, version = 0;
    if (!readVal(ptr, end, magic)) return false;
    if (magic != 0x56425643) { // "VBVC"
        LOG_ERROR("CFG", "Invalid voice preset magic: 0x%08X", magic);
        return false;
    }
    if (!readVal(ptr, end, version)) return false;
    if (version != 1) {
        LOG_ERROR("CFG", "Unsupported voice preset version: %u", version);
        return false;
    }
    if (!readVal(ptr, end, out.num_groups)) return false;
    if (!readVal(ptr, end, out.hidden_size)) return false;
    if (!readVal(ptr, end, out.head_dim)) return false;

    uint32_t reserved[3];
    for (int i = 0; i < 3; ++i) {
        if (!readVal(ptr, end, reserved[i])) return false;
    }

    out.groups.resize(out.num_groups);
    for (uint32_t g = 0; g < out.num_groups; ++g) {
        auto& grp = out.groups[g];

        // Group header (16 bytes)
        if (!readVal(ptr, end, grp.num_layers)) return false;
        if (!readVal(ptr, end, grp.num_kv_heads)) return false;
        if (!readVal(ptr, end, grp.seq_len)) return false;
        uint32_t grp_reserved;
        if (!readVal(ptr, end, grp_reserved)) return false;

        // last_hidden_state: fp16 [seq_len, hidden_size]
        size_t lhsCount = (size_t)grp.seq_len * out.hidden_size;
        if (!readVec(ptr, end, grp.last_hidden_state, lhsCount)) return false;

        // KV cache: for each layer, key then value
        // Each: [num_kv_heads, seq_len, head_dim] fp16
        size_t kvSize = (size_t)grp.num_kv_heads * grp.seq_len * out.head_dim;
        grp.key_cache.resize(grp.num_layers);
        grp.value_cache.resize(grp.num_layers);
        for (uint32_t l = 0; l < grp.num_layers; ++l) {
            if (!readVec(ptr, end, grp.key_cache[l], kvSize)) return false;
            if (!readVec(ptr, end, grp.value_cache[l], kvSize)) return false;
        }
    }

    return true;
}

// ── TTSInputTypesWeights ──

bool loadTTSInputTypesWeights(const std::string& binPath, TTSInputTypesWeights& out) {
    // Reuses EmbeddingWeights binary format
    EmbeddingWeights emb;
    if (!loadEmbeddingWeights(binPath, emb)) return false;
    out.num_types = emb.num_embeddings;
    out.embedding_dim = emb.embedding_dim;
    out.data = std::move(emb.data);
    return true;
}

// ── Scalar ──

bool loadScalar(const std::string& binPath, float& out) {
    std::vector<uint8_t> buf;
    if (!readFile(binPath, buf)) return false;
    if (buf.size() < 4) {
        LOG_ERROR("CFG", "Scalar file too small: %s (%zu bytes)", binPath.c_str(), buf.size());
        return false;
    }
    memcpy(&out, buf.data(), sizeof(float));
    return true;
}
