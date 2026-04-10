#pragma once

#include <sstream>
#include <string>

#ifdef CUPF_ENABLE_LOG
#if defined(__has_include)
#if __has_include(<spdlog/spdlog.h>)
#include <spdlog/spdlog.h>
#else
#error "CUPF_ENABLE_LOG requires <spdlog/spdlog.h>"
#endif
#else
#include <spdlog/spdlog.h>
#endif
#endif

namespace newton_solver::utils {

enum class LogLevel {
    Debug = 0,
    Info = 1,
    Warn = 2,
    Error = 3,
    Off = 4
};

#ifdef CUPF_ENABLE_LOG

struct LoggerState {
    LogLevel level = LogLevel::Info;
    bool enabled = true;
};

inline LoggerState& loggerState()
{
    static LoggerState state;
    return state;
}

inline spdlog::level::level_enum toSpdlogLevel(LogLevel level)
{
    switch (level) {
        case LogLevel::Debug:
            return spdlog::level::debug;
        case LogLevel::Info:
            return spdlog::level::info;
        case LogLevel::Warn:
            return spdlog::level::warn;
        case LogLevel::Error:
            return spdlog::level::err;
        case LogLevel::Off:
            return spdlog::level::off;
        default:
            return spdlog::level::info;
    }
}

inline void applyLoggerState()
{
    const auto& state = loggerState();
    spdlog::set_pattern("[%l] %v");
    spdlog::set_level(state.enabled ? toSpdlogLevel(state.level) : spdlog::level::off);
}

inline void ensureLoggerConfigured()
{
    static bool configured = false;
    if (!configured) {
        applyLoggerState();
        configured = true;
    }
}

inline void initLogger(LogLevel level = LogLevel::Info, bool enabled = true)
{
    auto& state = loggerState();
    state.level = level;
    state.enabled = enabled;
    applyLoggerState();
}

inline void setLogLevel(LogLevel level)
{
    loggerState().level = level;
    applyLoggerState();
}

inline LogLevel getLogLevel()
{
    return loggerState().level;
}

inline void setLogEnabled(bool enabled)
{
    loggerState().enabled = enabled;
    applyLoggerState();
}

inline bool isLogEnabled()
{
    return loggerState().enabled;
}

template <typename T>
inline void log(LogLevel level, const T& message)
{
    ensureLoggerConfigured();
    std::ostringstream oss;
    oss << message;
    spdlog::log(toSpdlogLevel(level), oss.str());
}

inline void log(LogLevel level, const char* message)
{
    ensureLoggerConfigured();
    spdlog::log(toSpdlogLevel(level), message != nullptr ? message : "<null>");
}

#else

struct LoggerState {
    LogLevel level = LogLevel::Off;
    bool enabled = false;
};

inline LoggerState& loggerState()
{
    static LoggerState state;
    return state;
}

inline void applyLoggerState() {}

inline void ensureLoggerConfigured() {}

inline void initLogger(LogLevel level = LogLevel::Info, bool enabled = false)
{
    (void)level;
    (void)enabled;
}

inline void setLogLevel(LogLevel level)
{
    (void)level;
}

inline LogLevel getLogLevel()
{
    return LogLevel::Off;
}

inline void setLogEnabled(bool enabled)
{
    (void)enabled;
}

inline bool isLogEnabled()
{
    return false;
}

template <typename T>
inline void log(LogLevel level, const T& message)
{
    (void)level;
    (void)message;
}

inline void log(LogLevel level, const char* message)
{
    (void)level;
    (void)message;
}

#endif

template <typename T>
inline void logDebug(const T& message)
{
    log(LogLevel::Debug, message);
}

template <typename T>
inline void logInfo(const T& message)
{
    log(LogLevel::Info, message);
}

template <typename T>
inline void logWarn(const T& message)
{
    log(LogLevel::Warn, message);
}

template <typename T>
inline void logError(const T& message)
{
    log(LogLevel::Error, message);
}

}  // namespace newton_solver::utils

#ifdef CUPF_ENABLE_LOG

#ifndef LOG_DEBUG
#define LOG_DEBUG(message) \
    do { \
        ::newton_solver::utils::logDebug((message)); \
    } while (0)
#endif

#ifndef LOG_INFO
#define LOG_INFO(message) \
    do { \
        ::newton_solver::utils::logInfo((message)); \
    } while (0)
#endif

#ifndef LOG_WARN
#define LOG_WARN(message) \
    do { \
        ::newton_solver::utils::logWarn((message)); \
    } while (0)
#endif

#ifndef LOG_ERROR
#define LOG_ERROR(message) \
    do { \
        ::newton_solver::utils::logError((message)); \
    } while (0)
#endif

#else

#ifndef LOG_DEBUG
#define LOG_DEBUG(message) \
    do { \
    } while (0)
#endif

#ifndef LOG_INFO
#define LOG_INFO(message) \
    do { \
    } while (0)
#endif

#ifndef LOG_WARN
#define LOG_WARN(message) \
    do { \
    } while (0)
#endif

#ifndef LOG_ERROR
#define LOG_ERROR(message) \
    do { \
    } while (0)
#endif

#endif
