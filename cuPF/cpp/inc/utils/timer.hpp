#pragma once

// ---------------------------------------------------------------------------
// timer.hpp — 계층형 wall-clock 타이밍 수집
//
// 구조 개요:
//
//   TimingBuffer (thread-local)
//     └─ TimingSample[]  : (name, elapsed_us) 원시 샘플 목록
//
//   TimingRegistry (전역 싱글톤)
//     └─ TimingBuffer*[] : 모든 스레드 버퍼의 포인터 레지스트리
//     └─ snapshot()      : 전체 버퍼를 순회해 label별 count/total_us 집계
//     └─ reset()         : 전체 버퍼 clear
//
//   ScopedTimer (RAII)
//     - 생성 시 steady_clock::now() 기록
//     - 소멸 또는 stop() 호출 시 경과 시간을 threadTimingBuffer()에 append
//     - CUPF_ENABLE_LOG 가 정의된 경우 로그도 출력
//
// 빌드 설정:
//   CUPF_ENABLE_TIMING  정의 → 실제 타이밍 수집 활성화
//   미정의              → ScopedTimer/timingSnapshot/resetTimingCollector 가
//                         모두 no-op으로 컴파일됨 (런타임 오버헤드 없음)
//
// 사용 예:
//   {
//       ScopedTimer t("jacobian_fill");
//       ... // 측정 대상 코드
//   }  // t 소멸 시 자동 기록
//
//   auto entries = timingSnapshot();  // 집계 결과 획득
//   resetTimingCollector();           // 다음 solve 전에 리셋
// ---------------------------------------------------------------------------

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "logger.hpp"

namespace newton_solver::utils {

// timingSnapshot()이 반환하는 label별 집계 결과.
// name은 ScopedTimer 생성자에 전달한 문자열 포인터 (정적 수명 가정).
struct TimingEntry {
    const char* name = nullptr;
    uint64_t count = 0;
    uint64_t total_us = 0;
};

#ifdef CUPF_ENABLE_TIMING

namespace detail {

// 타이밍 버퍼에 저장되는 원시 샘플 (label + 경과 시간).
struct TimingSample {
    const char* name = nullptr;
    uint64_t elapsed_us = 0;
};

inline const char* safeTimingName(const char* name)
{
    return name != nullptr ? name : "<null>";
}

class TimingBuffer;

// 전역 싱글톤: 모든 스레드의 TimingBuffer 포인터를 등록/해제하고
// snapshot() 호출 시 전체 샘플을 label별로 집계한다.
// TimingBuffer 생성/소멸이 registerBuffer/unregisterBuffer를 자동 호출하므로
// thread_local 변수를 통해 생명주기가 자동 관리된다.
class TimingRegistry {
public:
    static TimingRegistry& instance()
    {
        static TimingRegistry* registry = new TimingRegistry();
        return *registry;
    }

    void registerBuffer(TimingBuffer* buffer)
    {
        std::lock_guard<std::mutex> lock(mu_);
        buffers_.push_back(buffer);
    }

    void unregisterBuffer(TimingBuffer* buffer)
    {
        std::lock_guard<std::mutex> lock(mu_);
        for (auto it = buffers_.begin(); it != buffers_.end(); ++it) {
            if (*it == buffer) {
                buffers_.erase(it);
                break;
            }
        }
    }

    std::vector<TimingEntry> snapshot() const;
    void reset();

private:
    mutable std::mutex mu_;
    std::vector<TimingBuffer*> buffers_;
};

// 스레드 로컬 샘플 버퍼.
// thread_local threadTimingBuffer()를 통해 접근하며, 생성/소멸 시
// TimingRegistry에 자동 등록/해제된다.
class TimingBuffer {
public:
    TimingBuffer()
    {
        samples_.reserve(128);
        TimingRegistry::instance().registerBuffer(this);
    }

    ~TimingBuffer()
    {
        TimingRegistry::instance().unregisterBuffer(this);
    }

    void append(const char* name, uint64_t elapsed_us)
    {
        samples_.push_back(TimingSample{name, elapsed_us});
    }

    void clear()
    {
        samples_.clear();
    }

    const std::vector<TimingSample>& samples() const
    {
        return samples_;
    }

private:
    std::vector<TimingSample> samples_;
};

inline TimingBuffer& threadTimingBuffer()
{
    thread_local TimingBuffer buffer;
    return buffer;
}

inline void recordTimingSample(const char* name, uint64_t elapsed_us)
{
    threadTimingBuffer().append(safeTimingName(name), elapsed_us);
}

inline std::vector<TimingEntry> TimingRegistry::snapshot() const
{
    std::lock_guard<std::mutex> lock(mu_);

    std::vector<TimingEntry> summary;
    summary.reserve(buffers_.size());

    std::unordered_map<std::string_view, size_t> index_by_name;
    for (const TimingBuffer* buffer : buffers_) {
        for (const TimingSample& sample : buffer->samples()) {
            std::string_view key(safeTimingName(sample.name));
            auto [it, inserted] = index_by_name.emplace(key, summary.size());
            if (inserted) {
                summary.push_back(TimingEntry{safeTimingName(sample.name), 0, 0});
            }

            TimingEntry& entry = summary[it->second];
            entry.count += 1;
            entry.total_us += sample.elapsed_us;
        }
    }

    return summary;
}

inline void TimingRegistry::reset()
{
    std::lock_guard<std::mutex> lock(mu_);
    for (TimingBuffer* buffer : buffers_) {
        buffer->clear();
    }
}

inline std::string makeTimingMessage(const char* name, double elapsed_ms)
{
    std::ostringstream oss;
    oss << "[cuPF][timer] label=" << safeTimingName(name)
        << " elapsed_ms=" << std::fixed << std::setprecision(3) << elapsed_ms;
    return oss.str();
}

}  // namespace detail

// RAII 타이머.
// - 생성 시 steady_clock으로 시작 시각을 기록한다.
// - 소멸 시(또는 stop() 명시 호출 시) 경과 시간을 threadTimingBuffer()에 append한다.
// - CUPF_ENABLE_LOG 가 정의된 경우 로그 레벨 level로 경과 시간을 출력한다.
// - stop()은 한 번만 유효하다; 이후 중복 호출은 no-op.
class ScopedTimer {
public:
    explicit ScopedTimer(const char* name, LogLevel level = LogLevel::Info)
        : name_(name)
        , level_(level)
        , start_time_(Clock::now())
    {}

    ~ScopedTimer()
    {
        stop();
    }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
    ScopedTimer(ScopedTimer&&) = delete;
    ScopedTimer& operator=(ScopedTimer&&) = delete;

    void stop()
    {
        if (stopped_) {
            return;
        }

        end_time_ = Clock::now();
        stopped_ = true;

        const uint64_t elapsed_us = elapsedMicroseconds();
        detail::recordTimingSample(name_, elapsed_us);

#ifdef CUPF_ENABLE_LOG
        log(level_, detail::makeTimingMessage(name_, static_cast<double>(elapsed_us) / 1000.0));
#endif
    }

    void reset()
    {
        start_time_ = Clock::now();
        end_time_ = start_time_;
        stopped_ = false;
    }

    double elapsedSeconds() const
    {
        return static_cast<double>(elapsedMicroseconds()) / 1000000.0;
    }

    double elapsedMilliseconds() const
    {
        return static_cast<double>(elapsedMicroseconds()) / 1000.0;
    }

private:
    using Clock = std::chrono::steady_clock;

    uint64_t elapsedMicroseconds() const
    {
        const auto end_time = stopped_ ? end_time_ : Clock::now();
        const auto elapsed_us =
            std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time_).count();
        return static_cast<uint64_t>(elapsed_us);
    }

    const char* name_;
    LogLevel level_ = LogLevel::Info;
    Clock::time_point start_time_;
    Clock::time_point end_time_ = start_time_;
    bool stopped_ = false;
};

inline std::vector<TimingEntry> timingSnapshot()
{
    return detail::TimingRegistry::instance().snapshot();
}

inline void resetTimingCollector()
{
    detail::TimingRegistry::instance().reset();
}

#else

class ScopedTimer {
public:
    explicit ScopedTimer(const char* name, LogLevel level = LogLevel::Info)
        : name_(name)
        , level_(level)
    {}

    ~ScopedTimer() = default;

    void stop()
    {
        stopped_ = true;
    }

    void reset()
    {
        stopped_ = false;
    }

    double elapsedSeconds() const
    {
        return 0.0;
    }

    double elapsedMilliseconds() const
    {
        return 0.0;
    }

private:
    const char* name_;
    LogLevel level_ = LogLevel::Off;
    bool stopped_ = false;
};

inline std::vector<TimingEntry> timingSnapshot()
{
    return {};
}

inline void resetTimingCollector() {}

#endif

}  // namespace newton_solver::utils
