#include "tests/test_common.h"

#include <iostream>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <thread>
#include <fstream>

#include <httplib.h>
#include <json.hpp>

#include "server/http_server.h"
#include "inference/tts_pipeline.h"
#include "inference/stt_pipeline.h"
#include "audio/audio_io.h"

using json = nlohmann::json;

// ── Helpers ──

static bool fileExists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

static bool directoryHasTrtFiles(const std::string& dir) {
    // Check if at least one .trt file exists in the directory
    // Simple check: try common engine names
    for (auto& name : {"language_model.trt", "acoustic_decoder.trt", "diffusion_head.trt"}) {
        if (fileExists(dir + "/" + name)) return true;
    }
    return false;
}

// ── Test 1: HTTP Endpoints (no model required) ──

static void testHttpEndpoints() {
    fprintf(stderr, "\n--- E2E: HTTP Endpoints ---\n");

    // Start server with all models disabled
    ServerConfig cfg;
    cfg.host = "127.0.0.1";
    cfg.port = 19876;
    cfg.tts05b.enabled = false;
    cfg.tts15b.enabled = false;
    cfg.asr.enabled = false;

    HttpServer server;
    bool initOk = server.init(cfg);
    check(initOk, "E2E: server init (no models)");
    if (!initOk) return;

    // Run server in background thread
    std::thread serverThread([&server]() {
        server.run();
    });

    // Give server time to start
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    httplib::Client cli("127.0.0.1", 19876);
    cli.set_connection_timeout(5);
    cli.set_read_timeout(5);

    // GET /health
    {
        auto res = cli.Get("/health");
        check(res != nullptr, "E2E: GET /health responds");
        if (res) {
            check(res->status == 200, "E2E: GET /health -> 200");
            auto j = json::parse(res->body, nullptr, false);
            check(!j.is_discarded(), "E2E: /health returns valid JSON");
            if (!j.is_discarded()) {
                check(j.value("status", "") == "ok", "E2E: /health status==ok");
            }
        }
    }

    // GET /v1/models
    {
        auto res = cli.Get("/v1/models");
        check(res != nullptr, "E2E: GET /v1/models responds");
        if (res) {
            check(res->status == 200, "E2E: GET /v1/models -> 200");
            auto j = json::parse(res->body, nullptr, false);
            check(!j.is_discarded(), "E2E: /v1/models returns valid JSON");
            if (!j.is_discarded()) {
                check(j.contains("data"), "E2E: /v1/models has data field");
                check(j["data"].is_array(), "E2E: /v1/models data is array");
                check(j["data"].empty(), "E2E: /v1/models data empty (no models loaded)");
            }
        }
    }

    // POST /v1/audio/speech -> 503 (no model)
    {
        json body;
        body["model"] = "tts-1";
        body["input"] = "Hello world.";
        body["voice"] = "alloy";
        auto res = cli.Post("/v1/audio/speech", body.dump(), "application/json");
        check(res != nullptr, "E2E: POST /v1/audio/speech responds");
        if (res) {
            check(res->status == 503, "E2E: POST /v1/audio/speech -> 503 (no model)");
        }
    }

    // POST /v1/audio/speech with bad JSON -> 400
    {
        auto res = cli.Post("/v1/audio/speech", "not json{{{", "application/json");
        check(res != nullptr, "E2E: POST bad JSON responds");
        if (res) {
            check(res->status == 400, "E2E: POST bad JSON -> 400");
        }
    }

    // POST /v1/audio/speech with missing input -> 400
    {
        json body;
        body["model"] = "tts-1";
        body["voice"] = "alloy";
        auto res = cli.Post("/v1/audio/speech", body.dump(), "application/json");
        check(res != nullptr, "E2E: POST missing input responds");
        if (res) {
            check(res->status == 400, "E2E: POST missing input -> 400");
        }
    }

    // Stop server
    server.stop();
    serverThread.join();
    fprintf(stderr, "  HTTP endpoint tests done\n");
}

// ── Test 2: TTS E2E (requires TRT engines) ──

static void testTTSE2E() {
    fprintf(stderr, "\n--- E2E: TTS Pipeline ---\n");

    std::string engineDir = "engines/tts_0.5b";
    if (!directoryHasTrtFiles(engineDir)) {
        fprintf(stderr, "  [SKIP] TRT engines not found in %s\n", engineDir.c_str());
        return;
    }

    TTSPipeline pipeline;
    TTSPipeline::Config tcfg;
    tcfg.type = TTSPipeline::ModelType::STREAMING_0_5B;
    tcfg.engineDir = engineDir;
    tcfg.weightsDir = "onnx/tts_0.5b/weights";
    tcfg.metadataPath = "onnx/tts_0.5b/model_metadata.json";
    tcfg.voicesDir = "voices/streaming_model";
    tcfg.tokenizerPath = "tokenizer/qwen2.5-0.5b/tokenizer.json";
    tcfg.specialTokensPath = "tokenizer/qwen2.5-0.5b/special_tokens.json";
    tcfg.cfgScale = 1.5f;
    tcfg.inferenceSteps = 5;

    bool loadOk = pipeline.load(tcfg);
    check(loadOk, "E2E TTS: load pipeline");
    if (!loadOk) return;

    TTSPipeline::Request req;
    req.text = "Hello world.";
    req.voice = "en-Carter_man";
    req.speed = 1.0f;

    auto result = pipeline.synthesize(req);
    check(result.ok, "E2E TTS: synthesize succeeds");
    if (!result.ok) {
        fprintf(stderr, "  Error: %s\n", result.error.c_str());
        return;
    }

    check(result.sampleRate == 24000, "E2E TTS: sampleRate == 24000");
    check(result.audio.size() > 3200, "E2E TTS: audio.size() > 3200");

    // Check not silence
    float maxAbs = 0.0f;
    for (auto s : result.audio) {
        float a = fabsf(s);
        if (a > maxAbs) maxAbs = a;
    }
    check(maxAbs > 0.001f, "E2E TTS: audio not silence");

    // Check all finite
    bool allFinite = true;
    for (auto s : result.audio) {
        if (!std::isfinite(s)) { allFinite = false; break; }
    }
    check(allFinite, "E2E TTS: all samples finite");

    fprintf(stderr, "    audio: %zu samples (%.2f sec), max=%.4f\n",
            result.audio.size(), (float)result.audio.size() / result.sampleRate, maxAbs);
}

// ── Test 3: STT E2E (requires TRT engines) ──

static void testSTTE2E() {
    fprintf(stderr, "\n--- E2E: STT Pipeline ---\n");

    std::string engineDir = "engines/asr";
    // Check for ASR-specific engine files
    bool hasEngines = false;
    for (auto& name : {"acoustic_encoder.trt", "semantic_encoder.trt",
                        "language_model_prefill.trt", "language_model_decode.trt"}) {
        if (fileExists(engineDir + "/" + name)) { hasEngines = true; break; }
    }

    if (!hasEngines) {
        fprintf(stderr, "  [SKIP] TRT engines not found in %s\n", engineDir.c_str());
        return;
    }

    STTPipeline pipeline;
    STTPipeline::Config scfg;
    scfg.engineDir = engineDir;
    scfg.weightsDir = "onnx/asr/weights";
    scfg.metadataPath = "onnx/asr/model_metadata.json";
    scfg.tokenizerPath = "tokenizer/qwen2.5-7b/tokenizer.json";
    scfg.specialTokensPath = "tokenizer/qwen2.5-7b/special_tokens.json";
    scfg.maxNewTokens = 32768;
    scfg.temperature = 0.0f;

    bool loadOk = pipeline.load(scfg);
    check(loadOk, "E2E STT: load pipeline");
    if (!loadOk) return;

    // Generate 5 seconds of sine wave audio at 24kHz
    int numSamples = 24000 * 5;
    std::vector<float> audio(numSamples);
    for (int i = 0; i < numSamples; ++i) {
        audio[i] = 0.3f * sinf(2.0f * 3.14159265f * 440.0f * (float)i / 24000.0f);
    }

    STTPipeline::Request req;
    req.audio = std::move(audio);
    req.sampleRate = 24000;
    req.temperature = 0.0f;
    req.translate = false;

    auto result = pipeline.transcribe(req);
    check(result.ok, "E2E STT: transcribe succeeds");
    if (!result.ok) {
        fprintf(stderr, "  Error: %s\n", result.error.c_str());
        return;
    }

    // Basic checks - sine wave may produce empty text, but duration should be >0
    check(result.duration > 0.0f, "E2E STT: duration > 0");
    fprintf(stderr, "    text: '%s'\n", result.text.c_str());
    fprintf(stderr, "    duration: %.2f sec, segments: %zu\n",
            result.duration, result.segments.size());
}

// ── Test 4: HTTP with Model (requires TRT engines) ──

static void testHttpWithModel() {
    fprintf(stderr, "\n--- E2E: HTTP + TTS Model ---\n");

    std::string engineDir = "engines/tts_0.5b";
    if (!directoryHasTrtFiles(engineDir)) {
        fprintf(stderr, "  [SKIP] TRT engines not found in %s\n", engineDir.c_str());
        return;
    }

    ServerConfig cfg;
    cfg.host = "127.0.0.1";
    cfg.port = 19877;
    cfg.tts05b.enabled = true;
    cfg.tts05b.engineDir = engineDir;
    cfg.tts05b.weightsDir = "onnx/tts_0.5b/weights";
    cfg.tts05b.metadataPath = "onnx/tts_0.5b/model_metadata.json";
    cfg.tts05b.voicesDir = "voices/streaming_model";
    cfg.tts05b.tokenizerPath = "tokenizer/qwen2.5-0.5b/tokenizer.json";
    cfg.tts05b.specialTokensPath = "tokenizer/qwen2.5-0.5b/special_tokens.json";
    cfg.tts05b.cfgScale = 1.5f;
    cfg.tts05b.inferenceSteps = 5;
    cfg.tts15b.enabled = false;
    cfg.asr.enabled = false;

    HttpServer server;
    bool initOk = server.init(cfg);
    check(initOk, "E2E HTTP+TTS: server init");
    if (!initOk) return;

    std::thread serverThread([&server]() {
        server.run();
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    httplib::Client cli("127.0.0.1", 19877);
    cli.set_connection_timeout(30);
    cli.set_read_timeout(60);

    // POST /v1/audio/speech -> 200 + WAV
    {
        json body;
        body["model"] = "tts-1";
        body["input"] = "Hello world.";
        body["voice"] = "alloy";
        body["response_format"] = "wav";
        auto res = cli.Post("/v1/audio/speech", body.dump(), "application/json");
        check(res != nullptr, "E2E HTTP+TTS: POST responds");
        if (res) {
            check(res->status == 200, "E2E HTTP+TTS: POST -> 200");
            check(res->get_header_value("Content-Type") == "audio/wav",
                  "E2E HTTP+TTS: Content-Type == audio/wav");
            // Check RIFF header
            check(res->body.size() > 44, "E2E HTTP+TTS: body > 44 bytes");
            if (res->body.size() > 4) {
                check(res->body.substr(0, 4) == "RIFF", "E2E HTTP+TTS: RIFF header");
            }
        }
    }

    server.stop();
    serverThread.join();
    fprintf(stderr, "  HTTP + TTS model test done\n");
}

// ── E2E Test Runner ──

int runE2ETests() {
    std::cout << "=== VibeVoice E2E Tests ===" << std::endl;

    testHttpEndpoints();
    testTTSE2E();
    testSTTE2E();
    testHttpWithModel();

    std::cout << "\n=== E2E Results ===" << std::endl;
    std::cout << "Passed: " << testsPassed << std::endl;
    std::cout << "Failed: " << testsFailed << std::endl;

    return testsFailed > 0 ? 1 : 0;
}
