#include <iostream>
#include <cstdlib>
#include <csignal>
#include <string>

#include <cuda_runtime.h>

#include "server/http_server.h"
#include "utils/logger.h"

// ── Test entry points (defined in tests/) ──

int runUnitTests();   // tests/tests_unit.cpp
int runE2ETests();    // tests/tests_e2e.cpp

// ── Graceful Shutdown ──

static HttpServer* g_server = nullptr;

static void signalHandler(int) {
    if (g_server) g_server->stop();
}

// ── Main ──

int main(int argc, char* argv[]) {
    setvbuf(stderr, NULL, _IONBF, 0);  // Make stderr unbuffered
    Logger::instance().setLogFile("C:\\Users\\onion\\Desktop\\Workspace\\VibeVoiceWindowsApiServer\\server_debug.log");
    // Parse CLI args
    if (argc >= 2 && std::string(argv[1]) == "--test") {
        return runUnitTests();
    }
    if (argc >= 2 && std::string(argv[1]) == "--test-e2e") {
        return runE2ETests();
    }

    // Server mode
    std::string configPath = "config.json";
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "--config") {
            configPath = argv[i + 1];
        }
    }

    std::cout << "=== VibeVoice Windows API Server === BUILD_V4" << std::endl;

    // CUDA info
    int cuda_runtime_ver = 0;
    cudaRuntimeGetVersion(&cuda_runtime_ver);
    std::cout << "CUDA Runtime: " << (cuda_runtime_ver / 1000) << "."
              << ((cuda_runtime_ver % 1000) / 10) << std::endl;

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "GPU [" << i << "]: " << prop.name
                  << " (" << (prop.totalGlobalMem / (1024 * 1024)) << " MB)" << std::endl;
    }

    // Load config
    ServerConfig cfg;
    if (!loadServerConfig(configPath, cfg)) {
        LOG_ERROR("MAIN", "Failed to load config from %s", configPath.c_str());
        LOG_ERROR("MAIN", "Usage: vibevoice_server [--config config.json] [--test] [--test-e2e]");
        return 1;
    }

    // Start server
    HttpServer server;
    if (!server.init(cfg)) {
        LOG_ERROR("MAIN", "Failed to initialize server");
        return 1;
    }

    // Register signal handler for graceful shutdown
    g_server = &server;
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    server.run();  // blocking

    g_server = nullptr;
    LOG_INFO("MAIN", "Server stopped");
    return 0;
}
