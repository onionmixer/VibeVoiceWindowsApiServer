#pragma once
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <string>

// Windows <windows.h> defines ERROR as a macro (value 0). Undefine it.
#ifdef ERROR
#undef ERROR
#endif

enum class LogLevel { DEBUG = 0, INFO = 1, WARN = 2, ERROR = 3 };

class Logger {
public:
    static Logger& instance();

    void setLevel(LogLevel level);
    void setLogFile(const std::string& path);
    void close();

    void log(LogLevel level, const char* module, const char* fmt, ...);

    // Thread-local request ID (set from HTTP handlers)
    static void setRequestId(const std::string& id);
    static void clearRequestId();
    static const std::string& requestId();

    // RAII performance timer
    struct Timer {
        const char* module;
        const char* label;
        std::chrono::steady_clock::time_point start;

        Timer(const char* mod, const char* lbl)
            : module(mod), label(lbl), start(std::chrono::steady_clock::now()) {}

        ~Timer() {
            double ms = elapsedMs();
            Logger::instance().log(LogLevel::INFO, module, "%s took %.1f ms", label, ms);
        }

        double elapsedMs() const {
            auto now = std::chrono::steady_clock::now();
            return std::chrono::duration<double, std::milli>(now - start).count();
        }
    };

private:
    Logger() = default;
    ~Logger();
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    std::mutex mutex_;
    LogLevel level_ = LogLevel::INFO;
    FILE* file_ = nullptr;

    static thread_local std::string requestId_;

    void writeEntry(LogLevel lv, const char* module, const char* msg);
    const char* levelStr(LogLevel lv) const;
    std::string formatTimestamp() const;
};

// Convenience macros
#define LOG_DEBUG(mod, ...) Logger::instance().log(LogLevel::DEBUG, mod, __VA_ARGS__)
#define LOG_INFO(mod, ...)  Logger::instance().log(LogLevel::INFO,  mod, __VA_ARGS__)
#define LOG_WARN(mod, ...)  Logger::instance().log(LogLevel::WARN,  mod, __VA_ARGS__)
#define LOG_ERROR(mod, ...) Logger::instance().log(LogLevel::ERROR, mod, __VA_ARGS__)
#define LOG_TIMER(mod, label) Logger::Timer _ltimer_##__COUNTER__(mod, label)
