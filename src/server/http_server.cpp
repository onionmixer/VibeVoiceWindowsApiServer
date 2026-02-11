#include "server/http_server.h"
#include "audio/audio_io.h"
#include "audio/audio_convert.h"

#include "utils/logger.h"
#include <json.hpp>
#include <fstream>
#include <sstream>
#include <ctime>

using json = nlohmann::json;

// ── Config Loading ──

bool loadServerConfig(const std::string& jsonPath, ServerConfig& out) {
    std::ifstream f(jsonPath);
    if (!f.is_open()) {
        LOG_ERROR("HTTP", "Cannot open config: %s", jsonPath.c_str());
        return false;
    }

    json j;
    try {
        f >> j;
    } catch (const json::parse_error& e) {
        LOG_ERROR("HTTP", "Config JSON parse error: %s", e.what());
        return false;
    }

    if (j.contains("server")) {
        auto& s = j["server"];
        out.host = s.value("host", out.host);
        out.port = s.value("port", out.port);
    }

    if (j.contains("models")) {
        auto& m = j["models"];

        if (m.contains("tts_0.5b")) {
            auto& t = m["tts_0.5b"];
            out.tts05b.enabled = t.value("enabled", false);
            out.tts05b.engineDir = t.value("engine_dir", "");
            out.tts05b.weightsDir = t.value("weights_dir", "");
            out.tts05b.metadataPath = t.value("metadata_path", "");
            out.tts05b.voicesDir = t.value("voices_dir", "");
            out.tts05b.tokenizerPath = t.value("tokenizer_path", "");
            out.tts05b.specialTokensPath = t.value("special_tokens_path", "");
            out.tts05b.cfgScale = t.value("cfg_scale", 1.5f);
            out.tts05b.inferenceSteps = t.value("inference_steps", 5);
        }

        if (m.contains("tts_1.5b")) {
            auto& t = m["tts_1.5b"];
            out.tts15b.enabled = t.value("enabled", false);
            out.tts15b.engineDir = t.value("engine_dir", "");
            out.tts15b.weightsDir = t.value("weights_dir", "");
            out.tts15b.metadataPath = t.value("metadata_path", "");
            out.tts15b.voicesDir = t.value("voices_dir", "");
            out.tts15b.tokenizerPath = t.value("tokenizer_path", "");
            out.tts15b.specialTokensPath = t.value("special_tokens_path", "");
            out.tts15b.cfgScale = t.value("cfg_scale", 1.5f);
            out.tts15b.inferenceSteps = t.value("inference_steps", 5);
        }

        if (m.contains("asr")) {
            auto& a = m["asr"];
            out.asr.enabled = a.value("enabled", false);
            out.asr.engineDir = a.value("engine_dir", "");
            out.asr.weightsDir = a.value("weights_dir", "");
            out.asr.metadataPath = a.value("metadata_path", "");
            out.asr.tokenizerPath = a.value("tokenizer_path", "");
            out.asr.specialTokensPath = a.value("special_tokens_path", "");
            out.asr.maxNewTokens = a.value("max_new_tokens", 32768);
            out.asr.temperature = a.value("temperature", 0.0f);
        }
    }

    if (j.contains("defaults")) {
        auto& d = j["defaults"];
        out.defaultTTSVoice = d.value("tts_voice", out.defaultTTSVoice);
        out.defaultTTSModel = d.value("tts_model", out.defaultTTSModel);
        out.maxAudioDuration = d.value("max_audio_duration", 600);
    }

    return true;
}

// ── HttpServer ──

HttpServer::HttpServer() {}
HttpServer::~HttpServer() { stop(); }

bool HttpServer::init(const ServerConfig& cfg) {
    cfg_ = cfg;

    // Load TTS 0.5B
    if (cfg_.tts05b.enabled) {
        tts05b_ = std::make_unique<TTSPipeline>();
        TTSPipeline::Config tcfg;
        tcfg.type = TTSPipeline::ModelType::STREAMING_0_5B;
        tcfg.engineDir = cfg_.tts05b.engineDir;
        tcfg.weightsDir = cfg_.tts05b.weightsDir;
        tcfg.metadataPath = cfg_.tts05b.metadataPath;
        tcfg.voicesDir = cfg_.tts05b.voicesDir;
        tcfg.tokenizerPath = cfg_.tts05b.tokenizerPath;
        tcfg.specialTokensPath = cfg_.tts05b.specialTokensPath;
        tcfg.cfgScale = cfg_.tts05b.cfgScale;
        tcfg.inferenceSteps = cfg_.tts05b.inferenceSteps;
        if (!tts05b_->load(tcfg)) {
            LOG_ERROR("HTTP", "failed to load TTS 0.5B");
            tts05b_.reset();
        }
    }

    // Load TTS 1.5B
    if (cfg_.tts15b.enabled) {
        tts15b_ = std::make_unique<TTSPipeline>();
        TTSPipeline::Config tcfg;
        tcfg.type = TTSPipeline::ModelType::FULL_1_5B;
        tcfg.engineDir = cfg_.tts15b.engineDir;
        tcfg.weightsDir = cfg_.tts15b.weightsDir;
        tcfg.metadataPath = cfg_.tts15b.metadataPath;
        tcfg.voicesDir = cfg_.tts15b.voicesDir;
        tcfg.tokenizerPath = cfg_.tts15b.tokenizerPath;
        tcfg.specialTokensPath = cfg_.tts15b.specialTokensPath;
        tcfg.cfgScale = cfg_.tts15b.cfgScale;
        tcfg.inferenceSteps = cfg_.tts15b.inferenceSteps;
        if (!tts15b_->load(tcfg)) {
            LOG_ERROR("HTTP", "failed to load TTS 1.5B");
            tts15b_.reset();
        }
    }

    // Load STT
    if (cfg_.asr.enabled) {
        stt_ = std::make_unique<STTPipeline>();
        STTPipeline::Config scfg;
        scfg.engineDir = cfg_.asr.engineDir;
        scfg.weightsDir = cfg_.asr.weightsDir;
        scfg.metadataPath = cfg_.asr.metadataPath;
        scfg.tokenizerPath = cfg_.asr.tokenizerPath;
        scfg.specialTokensPath = cfg_.asr.specialTokensPath;
        scfg.maxNewTokens = cfg_.asr.maxNewTokens;
        scfg.temperature = cfg_.asr.temperature;
        if (!stt_->load(scfg)) {
            LOG_ERROR("HTTP", "failed to load STT");
            stt_.reset();
        }
    }

    // CORS
    svr_.set_default_headers({
        {"Access-Control-Allow-Origin", "*"},
        {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
        {"Access-Control-Allow-Headers", "Content-Type, Authorization"},
    });
    svr_.Options(".*", [](const httplib::Request&, httplib::Response& res) {
        res.status = 204;
    });

    // Routes
    svr_.Get("/health", [this](const httplib::Request& req, httplib::Response& res) {
        handleHealth(req, res);
    });
    svr_.Get("/v1/models", [this](const httplib::Request& req, httplib::Response& res) {
        handleModels(req, res);
    });
    svr_.Get("/v1/audio/voices", [this](const httplib::Request& req, httplib::Response& res) {
        handleVoices(req, res);
    });
    svr_.Post("/v1/audio/speech", [this](const httplib::Request& req, httplib::Response& res) {
        handleSpeech(req, res);
    });
    svr_.Post("/v1/audio/transcriptions", [this](const httplib::Request& req, httplib::Response& res) {
        handleTranscriptions(req, res);
    });
    svr_.Post("/v1/audio/translations", [this](const httplib::Request& req, httplib::Response& res) {
        handleTranslations(req, res);
    });

    LOG_INFO("HTTP", "initialized (TTS_0.5B=%s, TTS_1.5B=%s, STT=%s)",
             tts05b_ ? "loaded" : "off",
             tts15b_ ? "loaded" : "off",
             stt_ ? "loaded" : "off");
    return true;
}

void HttpServer::run() {
    LOG_INFO("HTTP", "listening on %s:%d", cfg_.host.c_str(), cfg_.port);
    svr_.listen(cfg_.host, cfg_.port);
}

void HttpServer::stop() {
    shuttingDown_ = true;
    svr_.stop();
}

// ── Helpers ──

void HttpServer::sendError(httplib::Response& res, int status,
                            const std::string& message, const std::string& type,
                            const std::string& code) {
    json err;
    err["error"]["message"] = message;
    err["error"]["type"] = type;
    if (!code.empty()) err["error"]["code"] = code;
    res.status = status;
    res.set_content(err.dump(), "application/json");
}

std::string HttpServer::mapModelName(const std::string& name) const {
    if (name == "tts-1" || name == "tts_0.5b" || name == "vibevoice-0.5b" || name == "tts-1-1106") return "tts_0.5b";
    if (name == "tts-1-hd" || name == "tts_1.5b" || name == "vibevoice-1.5b" || name == "tts-1-hd-1106") return "tts_1.5b";
    if (name == "whisper-1" || name == "asr" || name == "vibevoice-asr") return "asr";
    return name;
}

// ── GET /health ──

void HttpServer::handleHealth(const httplib::Request&, httplib::Response& res) {
    json j;
    j["status"] = "ok";
    j["models"]["tts_0.5b"] = tts05b_ && tts05b_->isLoaded();
    j["models"]["tts_1.5b"] = tts15b_ && tts15b_->isLoaded();
    j["models"]["asr"] = stt_ && stt_->isLoaded();
    res.set_content(j.dump(), "application/json");
}

// ── GET /v1/models ──

void HttpServer::handleModels(const httplib::Request&, httplib::Response& res) {
    json models = json::array();

    if (tts05b_ && tts05b_->isLoaded()) {
        json m;
        m["id"] = "tts-1";
        m["object"] = "model";
        m["created"] = 1699000000;
        m["owned_by"] = "vibevoice";
        models.push_back(m);
    }
    if (tts15b_ && tts15b_->isLoaded()) {
        json m;
        m["id"] = "tts-1-hd";
        m["object"] = "model";
        m["created"] = 1699000000;
        m["owned_by"] = "vibevoice";
        models.push_back(m);
    }
    if (stt_ && stt_->isLoaded()) {
        json m;
        m["id"] = "whisper-1";
        m["object"] = "model";
        m["created"] = 1699000000;
        m["owned_by"] = "vibevoice";
        models.push_back(m);
    }

    json resp;
    resp["object"] = "list";
    resp["data"] = models;
    res.set_content(resp.dump(), "application/json");
}

// ── GET /v1/audio/voices ──

void HttpServer::handleVoices(const httplib::Request&, httplib::Response& res) {
    json voices = json::array();

    auto addVoices = [&](TTSPipeline* p) {
        if (!p || !p->isLoaded()) return;
        for (auto& v : p->availableVoices()) {
            json vj;
            vj["name"] = v;
            voices.push_back(vj);
        }
    };

    addVoices(tts05b_.get());
    addVoices(tts15b_.get());

    json resp;
    resp["voices"] = voices;
    res.set_content(resp.dump(), "application/json");
}

// ── POST /v1/audio/speech ──

void HttpServer::handleSpeech(const httplib::Request& req, httplib::Response& res) {
    static std::atomic<uint64_t> counter{0};
    Logger::setRequestId("tts-" + std::to_string(counter.fetch_add(1)));

    // Parse JSON body
    json body;
    try {
        body = json::parse(req.body);
    } catch (const json::parse_error&) {
        sendError(res, 400, "Invalid JSON body");
        Logger::clearRequestId();
        return;
    }

    std::string modelName = body.value("model", cfg_.defaultTTSModel);
    std::string input = body.value("input", "");
    std::string voice = body.value("voice", cfg_.defaultTTSVoice);
    std::string responseFormat = body.value("response_format", "wav");
    float speed = body.value("speed", 1.0f);

    if (input.empty()) {
        sendError(res, 400, "Missing 'input' field");
        Logger::clearRequestId();
        return;
    }

    LOG_INFO("HTTP", "POST /v1/audio/speech model=%s voice=%s format=%s",
             modelName.c_str(), voice.c_str(), responseFormat.c_str());

    // Map model name
    std::string mapped = mapModelName(modelName);

    // Select pipeline
    TTSPipeline* pipeline = nullptr;
    bool use05b = false;
    if (mapped == "tts_0.5b" && tts05b_ && tts05b_->isLoaded()) {
        pipeline = tts05b_.get();
        use05b = true;
    } else if (mapped == "tts_1.5b" && tts15b_ && tts15b_->isLoaded()) {
        pipeline = tts15b_.get();
    } else if (tts05b_ && tts05b_->isLoaded()) {
        pipeline = tts05b_.get();
        use05b = true;
    } else if (tts15b_ && tts15b_->isLoaded()) {
        pipeline = tts15b_.get();
    }

    if (!pipeline) {
        sendError(res, 503, "No TTS model available", "server_error");
        Logger::clearRequestId();
        return;
    }

    // Resolve voice via pipeline's 4-stage fallback
    std::string resolvedVoice = pipeline->resolveVoice(voice);
    if (resolvedVoice.empty()) {
        auto voices = pipeline->availableVoices();
        std::string voiceList;
        for (size_t i = 0; i < voices.size(); ++i) {
            if (i > 0) voiceList += ", ";
            voiceList += voices[i];
        }
        sendError(res, 400,
            "Voice '" + voice + "' not found. Available: " + voiceList,
            "invalid_request_error");
        Logger::clearRequestId();
        return;
    }
    if (resolvedVoice != voice) {
        LOG_INFO("HTTP", "voice resolved: '%s' -> '%s'", voice.c_str(), resolvedVoice.c_str());
    }

    // Synthesize with per-pipeline mutex + timeout
    auto& mtx = use05b ? tts05bMutex_ : tts15bMutex_;
    std::unique_lock<std::timed_mutex> lock(mtx, std::defer_lock);
    if (!lock.try_lock_for(std::chrono::seconds(30))) {
        sendError(res, 503, "Model busy, try again later", "server_error");
        Logger::clearRequestId();
        return;
    }

    TTSPipeline::Result result;
    {
        LOG_TIMER("HTTP", "speech synthesis");
        result = pipeline->synthesize({input, resolvedVoice, speed});
    }

    if (!result.ok) {
        sendError(res, 500, result.error, "server_error");
        Logger::clearRequestId();
        return;
    }

    // Format response
    if (responseFormat == "wav" || responseFormat == "pcm") {
        if (responseFormat == "pcm") {
            // Raw PCM int16 LE
            std::vector<int16_t> pcm(result.audio.size());
            for (size_t i = 0; i < result.audio.size(); ++i) {
                float s = std::max(-1.0f, std::min(1.0f, result.audio[i]));
                pcm[i] = (int16_t)(s * 32767.0f);
            }
            res.set_content(
                std::string(reinterpret_cast<const char*>(pcm.data()), pcm.size() * sizeof(int16_t)),
                "audio/pcm");
        } else {
            // Write WAV to memory
            // Simple WAV header + int16 data
            uint32_t sr = (uint32_t)result.sampleRate;
            uint32_t numSamples = (uint32_t)result.audio.size();
            uint32_t dataBytes = numSamples * 2;

            std::vector<uint8_t> wav(44 + dataBytes);
            uint8_t* p = wav.data();

            // RIFF header
            memcpy(p, "RIFF", 4); p += 4;
            uint32_t fileSize = 36 + dataBytes;
            memcpy(p, &fileSize, 4); p += 4;
            memcpy(p, "WAVE", 4); p += 4;

            // fmt chunk
            memcpy(p, "fmt ", 4); p += 4;
            uint32_t fmtSize = 16;     memcpy(p, &fmtSize, 4); p += 4;
            uint16_t audioFmt = 1;     memcpy(p, &audioFmt, 2); p += 2;
            uint16_t channels = 1;     memcpy(p, &channels, 2); p += 2;
            memcpy(p, &sr, 4);         p += 4;
            uint32_t byteRate = sr * 2; memcpy(p, &byteRate, 4); p += 4;
            uint16_t blockAlign = 2;   memcpy(p, &blockAlign, 2); p += 2;
            uint16_t bitsPerSample = 16; memcpy(p, &bitsPerSample, 2); p += 2;

            // data chunk
            memcpy(p, "data", 4); p += 4;
            memcpy(p, &dataBytes, 4); p += 4;

            // int16 samples
            int16_t* samples = reinterpret_cast<int16_t*>(p);
            for (uint32_t i = 0; i < numSamples; ++i) {
                float s = std::max(-1.0f, std::min(1.0f, result.audio[i]));
                samples[i] = (int16_t)(s * 32767.0f);
            }

            res.set_content(
                std::string(reinterpret_cast<const char*>(wav.data()), wav.size()),
                "audio/wav");
        }
    } else {
        // Need ffmpeg for mp3/opus/aac/flac
        // First write WAV to temp file, convert, read back
        std::string tmpWav = "temp_tts_" + std::to_string(std::time(nullptr)) + ".wav";
        if (!AudioIO::writeWav(tmpWav, result.audio, (uint32_t)result.sampleRate)) {
            sendError(res, 500, "Failed to create temp WAV", "server_error");
            Logger::clearRequestId();
            return;
        }

        AudioConvert::OutputFormat fmt;
        std::string contentType;
        if (responseFormat == "mp3") {
            fmt = AudioConvert::OutputFormat::MP3;
            contentType = "audio/mpeg";
        } else if (responseFormat == "opus") {
            fmt = AudioConvert::OutputFormat::OPUS;
            contentType = "audio/opus";
        } else if (responseFormat == "aac") {
            fmt = AudioConvert::OutputFormat::AAC;
            contentType = "audio/aac";
        } else if (responseFormat == "flac") {
            fmt = AudioConvert::OutputFormat::FLAC;
            contentType = "audio/flac";
        } else {
            remove(tmpWav.c_str());
            sendError(res, 400, "Unsupported response_format: " + responseFormat);
            Logger::clearRequestId();
            return;
        }

        std::string tmpOut = "temp_tts_out_" + std::to_string(std::time(nullptr)) + "." + AudioConvert::formatExtension(fmt);
        bool convOk = AudioConvert::convertFile(tmpWav, tmpOut, fmt);
        remove(tmpWav.c_str());

        if (!convOk) {
            sendError(res, 500, "Audio format conversion failed (ffmpeg required)", "server_error");
            Logger::clearRequestId();
            return;
        }

        // Read output file
        std::ifstream outFile(tmpOut, std::ios::binary | std::ios::ate);
        if (!outFile.is_open()) {
            remove(tmpOut.c_str());
            sendError(res, 500, "Failed to read converted audio", "server_error");
            Logger::clearRequestId();
            return;
        }
        size_t outSize = (size_t)outFile.tellg();
        outFile.seekg(0);
        std::string outData(outSize, '\0');
        outFile.read(outData.data(), outSize);
        outFile.close();
        remove(tmpOut.c_str());

        res.set_content(outData, contentType);
    }
    Logger::clearRequestId();
}

// ── POST /v1/audio/transcriptions ──

void HttpServer::handleTranscriptions(const httplib::Request& req, httplib::Response& res) {
    static std::atomic<uint64_t> counter{0};
    Logger::setRequestId("stt-" + std::to_string(counter.fetch_add(1)));

    if (!stt_ || !stt_->isLoaded()) {
        sendError(res, 503, "STT model not available", "server_error");
        Logger::clearRequestId();
        return;
    }

    if (!req.is_multipart_form_data()) {
        sendError(res, 400, "Expected multipart/form-data");
        Logger::clearRequestId();
        return;
    }

    // Get file
    if (!req.form.has_file("file")) {
        sendError(res, 400, "Missing 'file' field");
        Logger::clearRequestId();
        return;
    }
    auto file = req.form.get_file("file");

    // Get optional fields
    std::string responseFormat = "json";
    std::string language;
    std::string prompt;
    float temperature = 0.0f;

    std::string rfmt = req.form.get_field("response_format");
    if (!rfmt.empty()) responseFormat = rfmt;
    language = req.form.get_field("language");
    prompt = req.form.get_field("prompt");
    std::string tempStr = req.form.get_field("temperature");
    if (!tempStr.empty()) temperature = std::stof(tempStr);

    LOG_INFO("HTTP", "POST /v1/audio/transcriptions format=%s lang=%s",
             responseFormat.c_str(), language.c_str());

    // Decode audio from memory
    std::vector<float> audio;
    if (!AudioIO::loadAndPrepareFromMemory(
            reinterpret_cast<const uint8_t*>(file.content.data()),
            file.content.size(), audio, 24000, -25.0f)) {
        sendError(res, 400, "Failed to decode audio file");
        Logger::clearRequestId();
        return;
    }

    // Transcribe with per-pipeline mutex + timeout
    std::unique_lock<std::timed_mutex> lock(sttMutex_, std::defer_lock);
    if (!lock.try_lock_for(std::chrono::seconds(60))) {
        sendError(res, 503, "STT model busy, try again later", "server_error");
        Logger::clearRequestId();
        return;
    }

    STTPipeline::Result result;
    {
        LOG_TIMER("HTTP", "transcription");
        STTPipeline::Request sreq;
        sreq.audio = std::move(audio);
        sreq.sampleRate = 24000;
        sreq.language = language;
        sreq.prompt = prompt;
        sreq.temperature = temperature;
        sreq.translate = false;
        result = stt_->transcribe(sreq);
    }

    if (!result.ok) {
        sendError(res, 500, result.error, "server_error");
        Logger::clearRequestId();
        return;
    }

    // Format response
    if (responseFormat == "text") {
        res.set_content(result.text, "text/plain");
    } else if (responseFormat == "srt") {
        res.set_content(STTPipeline::formatSRT(result.segments), "text/plain");
    } else if (responseFormat == "vtt") {
        res.set_content(STTPipeline::formatVTT(result.segments), "text/plain");
    } else if (responseFormat == "verbose_json") {
        json j;
        j["task"] = "transcribe";
        j["language"] = result.language;
        j["duration"] = result.duration;
        j["text"] = result.text;

        json segs = json::array();
        for (size_t i = 0; i < result.segments.size(); ++i) {
            json seg;
            seg["id"] = (int)i;
            seg["start"] = result.segments[i].startTime;
            seg["end"] = result.segments[i].endTime;
            seg["text"] = result.segments[i].text;
            if (!result.segments[i].speakerId.empty()) {
                seg["speaker"] = result.segments[i].speakerId;
            }
            segs.push_back(seg);
        }
        j["segments"] = segs;

        res.set_content(j.dump(), "application/json");
    } else {
        // "json" (default)
        json j;
        j["text"] = result.text;
        res.set_content(j.dump(), "application/json");
    }
    Logger::clearRequestId();
}

// ── POST /v1/audio/translations ──

void HttpServer::handleTranslations(const httplib::Request& req, httplib::Response& res) {
    static std::atomic<uint64_t> counter{0};
    Logger::setRequestId("translate-" + std::to_string(counter.fetch_add(1)));

    if (!stt_ || !stt_->isLoaded()) {
        sendError(res, 503, "STT model not available", "server_error");
        Logger::clearRequestId();
        return;
    }

    if (!req.is_multipart_form_data()) {
        sendError(res, 400, "Expected multipart/form-data");
        Logger::clearRequestId();
        return;
    }

    if (!req.form.has_file("file")) {
        sendError(res, 400, "Missing 'file' field");
        Logger::clearRequestId();
        return;
    }
    auto file = req.form.get_file("file");

    std::string responseFormat = "json";
    std::string prompt;
    float temperature = 0.0f;

    std::string rfmt = req.form.get_field("response_format");
    if (!rfmt.empty()) responseFormat = rfmt;
    prompt = req.form.get_field("prompt");
    std::string tempStr = req.form.get_field("temperature");
    if (!tempStr.empty()) temperature = std::stof(tempStr);

    LOG_INFO("HTTP", "POST /v1/audio/translations format=%s",
             responseFormat.c_str());

    std::vector<float> audio;
    if (!AudioIO::loadAndPrepareFromMemory(
            reinterpret_cast<const uint8_t*>(file.content.data()),
            file.content.size(), audio, 24000, -25.0f)) {
        sendError(res, 400, "Failed to decode audio file");
        Logger::clearRequestId();
        return;
    }

    // Translate with per-pipeline mutex + timeout
    std::unique_lock<std::timed_mutex> lock(sttMutex_, std::defer_lock);
    if (!lock.try_lock_for(std::chrono::seconds(60))) {
        sendError(res, 503, "STT model busy, try again later", "server_error");
        Logger::clearRequestId();
        return;
    }

    STTPipeline::Result result;
    {
        LOG_TIMER("HTTP", "translation");
        STTPipeline::Request sreq;
        sreq.audio = std::move(audio);
        sreq.sampleRate = 24000;
        sreq.language = "en";
        sreq.prompt = prompt;
        sreq.temperature = temperature;
        sreq.translate = true;
        result = stt_->transcribe(sreq);
    }

    if (!result.ok) {
        sendError(res, 500, result.error, "server_error");
        Logger::clearRequestId();
        return;
    }

    if (responseFormat == "text") {
        res.set_content(result.text, "text/plain");
    } else if (responseFormat == "srt") {
        res.set_content(STTPipeline::formatSRT(result.segments), "text/plain");
    } else if (responseFormat == "vtt") {
        res.set_content(STTPipeline::formatVTT(result.segments), "text/plain");
    } else if (responseFormat == "verbose_json") {
        json j;
        j["task"] = "translate";
        j["language"] = result.language;
        j["duration"] = result.duration;
        j["text"] = result.text;

        json segs = json::array();
        for (size_t i = 0; i < result.segments.size(); ++i) {
            json seg;
            seg["id"] = (int)i;
            seg["start"] = result.segments[i].startTime;
            seg["end"] = result.segments[i].endTime;
            seg["text"] = result.segments[i].text;
            segs.push_back(seg);
        }
        j["segments"] = segs;

        res.set_content(j.dump(), "application/json");
    } else {
        json j;
        j["text"] = result.text;
        res.set_content(j.dump(), "application/json");
    }
    Logger::clearRequestId();
}
