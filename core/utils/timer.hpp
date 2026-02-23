#pragma once
#include <string>

#define USE_BLOCK_TIMER

#ifdef USE_BLOCK_TIMER

#include "spdlog/spdlog.h" // spdlog 헤더
#include <chrono>


inline std::shared_ptr<spdlog::logger>& timer_logger() {
    static std::shared_ptr<spdlog::logger> logger;
    return logger;
}

inline void init_timer_logger(std::shared_ptr<spdlog::logger> logger) {
    timer_logger() = std::move(logger);
    timer_logger()->set_level(spdlog::level::info);
    spdlog::info("[TIMER] Initialized!");
}

/**
 * @brief RAII 기반 타이머 클래스 
 * 객체 소멸 시 spdlog::debug 레벨로 시간 자동 출력
 */
class BlockTimer {
public:
    BlockTimer(const std::string& name)
        : m_Name(name),
          m_StartTime(std::chrono::high_resolution_clock::now()) {}

    ~BlockTimer() {
        auto endTime = std::chrono::high_resolution_clock::now();

        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(endTime - m_StartTime).count();
        double duration_sec = duration_us / 1000000.0;

        // 소영수정: 간단한 포맷으로 출력 (name duration_sec)
        auto logger = timer_logger();
        if (logger)
            logger->info("{} {:.6f}", m_Name, duration_sec);
        else
            spdlog::info("{} {:.6f}", m_Name, duration_sec);
    }

private:
    std::string m_Name;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_StartTime;
};

#else
class BlockTimer {
public:
    inline BlockTimer(const std::string&) {}
    inline ~BlockTimer() {}
};

#endif // USE_BLOCK_TIMER