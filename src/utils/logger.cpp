#include "utils/logger.h"
#include <ctime>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
// Windows <windows.h> defines ERROR as a macro; we already undef'd it in logger.h
// but re-undef in case it got re-defined by windows.h inclusion here.
#ifdef ERROR
#undef ERROR
#endif
#endif

// Thread-local request ID
thread_local std::string Logger::requestId_;

Logger& Logger::instance() {
    static Logger logger;
    return logger;
}

Logger::~Logger() {
    close();
}

void Logger::setLevel(LogLevel level) {
    std::lock_guard<std::mutex> lock(mutex_);
    level_ = level;
}

void Logger::setLogFile(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_) {
        fclose(file_);
        file_ = nullptr;
    }
    file_ = fopen(path.c_str(), "a");
}

void Logger::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_) {
        fclose(file_);
        file_ = nullptr;
    }
}

void Logger::setRequestId(const std::string& id) {
    requestId_ = id;
}

void Logger::clearRequestId() {
    requestId_.clear();
}

const std::string& Logger::requestId() {
    return requestId_;
}

void Logger::log(LogLevel level, const char* module, const char* fmt, ...) {
    if (level < level_) return;

    char buf[2048];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    writeEntry(level, module, buf);
}

void Logger::writeEntry(LogLevel lv, const char* module, const char* msg) {
    std::string ts = formatTimestamp();
    const char* lvStr = levelStr(lv);

    char line[2560];
    if (requestId_.empty()) {
        snprintf(line, sizeof(line), "[%s] [%-5s] [%s] %s\n", ts.c_str(), lvStr, module, msg);
    } else {
        snprintf(line, sizeof(line), "[%s] [%-5s] [%s] [%s] %s\n",
                 ts.c_str(), lvStr, module, requestId_.c_str(), msg);
    }

    std::lock_guard<std::mutex> lock(mutex_);
    fputs(line, stderr);
    if (file_) {
        fputs(line, file_);
        fflush(file_);
    }
}

const char* Logger::levelStr(LogLevel lv) const {
    switch (lv) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO:  return "INFO";
        case LogLevel::WARN:  return "WARN";
        case LogLevel::ERROR: return "ERROR";
    }
    return "???";
}

std::string Logger::formatTimestamp() const {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm = {};
#ifdef _WIN32
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif

    char buf[32];
    snprintf(buf, sizeof(buf), "%04d-%02d-%02d %02d:%02d:%02d.%03d",
             tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
             tm.tm_hour, tm.tm_min, tm.tm_sec, (int)ms.count());
    return buf;
}
