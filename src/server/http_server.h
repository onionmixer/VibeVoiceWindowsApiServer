#pragma once
#include "inference/tts_pipeline.h"
#include "inference/stt_pipeline.h"

#include <httplib.h>
#include <json.hpp>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct ServerConfig {
    std::string host = "0.0.0.0";
    int port = 8899;

    // TTS models
    struct TTSModelConfig {
        bool enabled = false;
        std::string engineDir;
        std::string weightsDir;
        std::string metadataPath;
        std::string voicesDir;
        std::string tokenizerPath;
        std::string specialTokensPath;
        float cfgScale = 1.5f;
        int inferenceSteps = 5;
    };
    TTSModelConfig tts05b;
    TTSModelConfig tts15b;

    // ASR model
    struct ASRModelConfig {
        bool enabled = false;
        std::string engineDir;
        std::string weightsDir;
        std::string metadataPath;
        std::string tokenizerPath;
        std::string specialTokensPath;
        int maxNewTokens = 32768;
        float temperature = 0.0f;
    };
    ASRModelConfig asr;

    // Defaults
    std::string defaultTTSVoice = "en-Carter_man";
    std::string defaultTTSModel = "tts_0.5b";
    int maxAudioDuration = 600;
};

bool loadServerConfig(const std::string& jsonPath, ServerConfig& out);

class HttpServer {
public:
    HttpServer();
    ~HttpServer();

    bool init(const ServerConfig& cfg);
    void run();   // blocking
    void stop();

private:
    // Endpoint handlers
    void handleHealth(const httplib::Request& req, httplib::Response& res);
    void handleModels(const httplib::Request& req, httplib::Response& res);
    void handleVoices(const httplib::Request& req, httplib::Response& res);
    void handleSpeech(const httplib::Request& req, httplib::Response& res);
    void handleTranscriptions(const httplib::Request& req, httplib::Response& res);
    void handleTranslations(const httplib::Request& req, httplib::Response& res);

    // Helpers
    void sendError(httplib::Response& res, int status,
                   const std::string& message, const std::string& type = "invalid_request_error",
                   const std::string& code = "");
    std::string mapVoiceName(const std::string& name) const;
    std::string mapModelName(const std::string& name) const;

    ServerConfig cfg_;
    httplib::Server svr_;

    std::unique_ptr<TTSPipeline> tts05b_;
    std::unique_ptr<TTSPipeline> tts15b_;
    std::unique_ptr<STTPipeline> stt_;

    std::timed_mutex tts05bMutex_;
    std::timed_mutex tts15bMutex_;
    std::timed_mutex sttMutex_;
    std::atomic<bool> shuttingDown_{false};
};
