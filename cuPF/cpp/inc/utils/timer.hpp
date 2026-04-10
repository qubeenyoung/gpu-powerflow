#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "logger.hpp"

namespace newton_solver::utils {

struct TimingEntry {
    const char* name = nullptr;
    uint64_t    count = 0;
    uint64_t    total_us = 0;
};

#ifdef CUPF_ENABLE_TIMING

namespace detail {

struct TimingSample {
    const char* name = nullptr;
    uint64_t    elapsed_us = 0;
};

class TimingBuffer;

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
    mutable std::mutex       mu_;
    std::vector<TimingBuffer*> buffers_;
};

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
    threadTimingBuffer().append(name, elapsed_us);
}

inline std::vector<TimingEntry> TimingRegistry::snapshot() const
{
    std::lock_guard<std::mutex> lock(mu_);

    std::vector<TimingEntry> summary;
    summary.reserve(buffers_.size());

    std::unordered_map<std::string_view, size_t> index_by_name;
    for (const TimingBuffer* buffer : buffers_) {
        for (const TimingSample& sample : buffer->samples()) {
            std::string_view key(sample.name);
            auto [it, inserted] = index_by_name.emplace(key, summary.size());
            if (inserted) {
                summary.push_back(TimingEntry{sample.name, 0, 0});
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

}  // namespace detail

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

    void stop()
    {
        if (stopped_) {
            return;
        }

        end_time_ = Clock::now();
        stopped_ = true;

        detail::recordTimingSample(name_, elapsedMicroseconds());

#ifdef CUPF_ENABLE_LOG
        log(level_, std::string(name_) + " took " + std::to_string(elapsedSeconds()) + " sec");
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
