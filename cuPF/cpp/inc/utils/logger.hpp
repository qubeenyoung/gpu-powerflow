#pragma once

// ---------------------------------------------------------------------------
// logger.hpp — cuPF 로깅 레이어
//
// spdlog를 감싸는 얇은 래퍼다. 빌드 설정에 따라 두 가지 모드로 동작한다.
//
// 빌드 설정:
//   CUPF_ENABLE_LOG 정의   → spdlog를 사용하는 실제 로거 활성화
//   미정의                 → 모든 로그 호출이 no-op으로 컴파일됨
//
// 핵심 인터페이스:
//   LogLevel                     : Debug / Info / Warn / Error / Off
//   initLogger(level, enabled)   : 로거 초기화 및 레벨 설정
//   setLogLevel(level)           : 런타임 레벨 변경
//   setLogEnabled(bool)          : 전체 로깅 on/off
//   log(LogLevel, message)       : 범용 로그 출력 (std::ostringstream 경유)
//   logDebug/Info/Warn/Error(m)  : 레벨별 편의 함수
//
// 매크로 (printf 스타일, CUPF_ENABLE_LOG 의존):
//   CUPF_LOG_TRACE / CUPF_LOG_DEBUG / CUPF_LOG_INFO
//   CUPF_LOG_WARN  / CUPF_LOG_ERROR
//     → spdlog 네이티브 포맷 문자열 지원, 비활성 시 완전 제거
//
// 매크로 (단일 인수, 항상 정의):
//   LOG_DEBUG / LOG_INFO / LOG_WARN / LOG_ERROR
//     → log(LogLevel, message) 를 호출하는 단순 래퍼
// ---------------------------------------------------------------------------

#include <sstream>
#include <string>

#ifdef CUPF_ENABLE_LOG
  #include <spdlog/spdlog.h>
#endif

namespace newton_solver::utils {

// 로그 출력 레벨. Off는 로깅을 완전히 비활성화한다.
enum class LogLevel {
    Debug = 0,
    Info = 1,
    Warn = 2,
    Error = 3,
    Off = 4,
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
    const LoggerState& state = loggerState();
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
    LoggerState& state = loggerState();
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

  #ifndef CUPF_LOG_TRACE
    #define CUPF_LOG_TRACE(...)                                              \
        do {                                                                 \
            ::newton_solver::utils::ensureLoggerConfigured();                \
            spdlog::trace(__VA_ARGS__);                                      \
        } while (0)
  #endif

  #ifndef CUPF_LOG_DEBUG
    #define CUPF_LOG_DEBUG(...)                                              \
        do {                                                                 \
            ::newton_solver::utils::ensureLoggerConfigured();                \
            spdlog::debug(__VA_ARGS__);                                      \
        } while (0)
  #endif

  #ifndef CUPF_LOG_INFO
    #define CUPF_LOG_INFO(...)                                               \
        do {                                                                 \
            ::newton_solver::utils::ensureLoggerConfigured();                \
            spdlog::info(__VA_ARGS__);                                       \
        } while (0)
  #endif

  #ifndef CUPF_LOG_WARN
    #define CUPF_LOG_WARN(...)                                               \
        do {                                                                 \
            ::newton_solver::utils::ensureLoggerConfigured();                \
            spdlog::warn(__VA_ARGS__);                                       \
        } while (0)
  #endif

  #ifndef CUPF_LOG_ERROR
    #define CUPF_LOG_ERROR(...)                                              \
        do {                                                                 \
            ::newton_solver::utils::ensureLoggerConfigured();                \
            spdlog::error(__VA_ARGS__);                                      \
        } while (0)
  #endif

#else

  #ifndef CUPF_LOG_TRACE
    #define CUPF_LOG_TRACE(...)                                              \
        do {                                                                 \
        } while (0)
  #endif

  #ifndef CUPF_LOG_DEBUG
    #define CUPF_LOG_DEBUG(...)                                              \
        do {                                                                 \
        } while (0)
  #endif

  #ifndef CUPF_LOG_INFO
    #define CUPF_LOG_INFO(...)                                               \
        do {                                                                 \
        } while (0)
  #endif

  #ifndef CUPF_LOG_WARN
    #define CUPF_LOG_WARN(...)                                               \
        do {                                                                 \
        } while (0)
  #endif

  #ifndef CUPF_LOG_ERROR
    #define CUPF_LOG_ERROR(...)                                              \
        do {                                                                 \
        } while (0)
  #endif

#endif

#ifndef LOG_DEBUG
  #define LOG_DEBUG(message)                                                 \
      do {                                                                   \
          ::newton_solver::utils::logDebug((message));                       \
      } while (0)
#endif

#ifndef LOG_INFO
  #define LOG_INFO(message)                                                  \
      do {                                                                   \
          ::newton_solver::utils::logInfo((message));                        \
      } while (0)
#endif

#ifndef LOG_WARN
  #define LOG_WARN(message)                                                  \
      do {                                                                   \
          ::newton_solver::utils::logWarn((message));                        \
      } while (0)
#endif

#ifndef LOG_ERROR
  #define LOG_ERROR(message)                                                 \
      do {                                                                   \
          ::newton_solver::utils::logError((message));                       \
      } while (0)
#endif
