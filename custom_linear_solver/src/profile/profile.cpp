#include "profile/profile.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

namespace cls::profile {

namespace {
const char* dtype_name(DType dt) {
    switch (dt) {
        case DType::Float32: return "float32";
        case DType::Float64: return "float64";
        case DType::Int32:   return "int32";
    }
    return "unknown";
}
std::size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::Float32: return 4;
        case DType::Float64: return 8;
        case DType::Int32:   return 4;
    }
    return 0;
}
}  // namespace

namespace {

struct PendingGpu {
    std::string name;
    cudaStream_t stream;
    cudaEvent_t e_start;
    cudaEvent_t e_stop;
};

struct CpuRow {
    std::string name;
    double start_ms;
    double duration_ms;
};

struct State {
    std::atomic<bool> on{false};
    std::mutex mu;
    std::string dir;
    std::string dumps_dir;
    std::string csv_path;
    std::chrono::steady_clock::time_point epoch;
    std::vector<PendingGpu> gpu_pending;
    std::vector<CpuRow> cpu_rows;
    std::unordered_map<std::string, int> dump_seq;
    bool csv_header_written = false;
    bool atexit_registered = false;
};

State& state() {
    static State s;
    return s;
}

double now_ms_since(std::chrono::steady_clock::time_point tp) {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now() - tp).count();
}

long long now_ns() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

void resolve_gpu_locked(State& s) {
    if (s.gpu_pending.empty()) return;
    for (auto& p : s.gpu_pending) {
        cudaEventSynchronize(p.e_stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, p.e_start, p.e_stop);
        // start_ms is approximated as "now - duration" since cudaEvent gives no absolute
        // timestamp. For the typical use (relative ordering, duration histograms) this is
        // sufficient; combine with NVTX + nsys for absolute timeline.
        double start_ms = now_ms_since(s.epoch) - static_cast<double>(ms);
        s.cpu_rows.push_back(CpuRow{"gpu:" + p.name, start_ms, static_cast<double>(ms)});
        cudaEventDestroy(p.e_start);
        cudaEventDestroy(p.e_stop);
    }
    s.gpu_pending.clear();
}

void write_csv_locked(State& s) {
    if (s.cpu_rows.empty()) return;
    FILE* fp = std::fopen(s.csv_path.c_str(), "a");
    if (!fp) return;
    if (!s.csv_header_written) {
        std::fprintf(fp, "kind,name,start_ms,duration_ms\n");
        s.csv_header_written = true;
    }
    for (const auto& r : s.cpu_rows) {
        const char* colon = std::strchr(r.name.c_str(), ':');
        std::string kind, name;
        if (colon) {
            kind.assign(r.name.c_str(), colon);
            name.assign(colon + 1);
        } else {
            kind = "cpu";
            name = r.name;
        }
        std::fprintf(fp, "%s,%s,%.6f,%.6f\n", kind.c_str(), name.c_str(),
                     r.start_ms, r.duration_ms);
    }
    std::fclose(fp);
    s.cpu_rows.clear();
}

void atexit_flush() { flush(); }

}  // namespace

void init() {
    State& s = state();
    std::lock_guard<std::mutex> lk(s.mu);
    if (s.on.load(std::memory_order_relaxed)) return;
    const char* env = std::getenv("CLS_PROFILE_DIR");
    if (!env || !*env) return;
    std::error_code ec;
    std::filesystem::create_directories(env, ec);
    if (ec) {
        std::fprintf(stderr, "[cls::profile] cannot create %s: %s\n", env, ec.message().c_str());
        return;
    }
    s.dir = env;
    s.dumps_dir = s.dir + "/dumps";
    std::filesystem::create_directories(s.dumps_dir, ec);
    s.csv_path = s.dir + "/timers.csv";
    s.epoch = std::chrono::steady_clock::now();
    s.on.store(true, std::memory_order_release);
    if (!s.atexit_registered) {
        std::atexit(atexit_flush);
        s.atexit_registered = true;
    }
}

void flush() {
    State& s = state();
    std::lock_guard<std::mutex> lk(s.mu);
    if (!s.on.load(std::memory_order_acquire)) return;
    resolve_gpu_locked(s);
    write_csv_locked(s);
}

bool enabled() { return state().on.load(std::memory_order_acquire); }

NvtxScope::NvtxScope(const char* name) : active_(1) { nvtxRangePushA(name); }
NvtxScope::~NvtxScope() { if (active_) nvtxRangePop(); }

CpuScope::CpuScope(const char* name) : name_(name), start_ns_(0), active_(0) {
    if (!enabled()) return;
    start_ns_ = now_ns();
    active_ = 1;
}
CpuScope::~CpuScope() {
    if (!active_) return;
    const long long end_ns = now_ns();
    State& s = state();
    std::lock_guard<std::mutex> lk(s.mu);
    if (!s.on.load(std::memory_order_relaxed)) return;
    using namespace std::chrono;
    double start_ms = duration<double, std::milli>(
        steady_clock::time_point{nanoseconds(start_ns_)} - s.epoch).count();
    double duration_ms = static_cast<double>(end_ns - start_ns_) / 1.0e6;
    s.cpu_rows.push_back(CpuRow{"cpu:" + std::string(name_), start_ms, duration_ms});
}

GpuScope::GpuScope(const char* name, cudaStream_t stream)
    : name_(name), stream_(stream), e_start_(nullptr), e_stop_(nullptr), active_(0) {
    if (!enabled()) return;
    if (cudaEventCreate(&e_start_) != cudaSuccess) return;
    if (cudaEventCreate(&e_stop_) != cudaSuccess) {
        cudaEventDestroy(e_start_);
        e_start_ = nullptr;
        return;
    }
    cudaEventRecord(e_start_, stream);
    active_ = 1;
}
GpuScope::~GpuScope() {
    if (!active_) return;
    cudaEventRecord(e_stop_, stream_);
    State& s = state();
    std::lock_guard<std::mutex> lk(s.mu);
    if (!s.on.load(std::memory_order_relaxed)) {
        cudaEventDestroy(e_start_);
        cudaEventDestroy(e_stop_);
        return;
    }
    s.gpu_pending.push_back(PendingGpu{name_, stream_, e_start_, e_stop_});
}

void dump_device(const char* name, const void* device_ptr, std::size_t n_elem,
                 DType dtype, cudaStream_t stream)
{
    if (!enabled() || !name || !device_ptr || n_elem == 0) return;
    const std::size_t bytes = n_elem * dtype_size(dtype);
    if (bytes == 0) return;
    std::vector<unsigned char> host(bytes);
    cudaError_t err;
    if (stream) {
        err = cudaMemcpyAsync(host.data(), device_ptr, bytes, cudaMemcpyDeviceToHost, stream);
        if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
    } else {
        err = cudaMemcpy(host.data(), device_ptr, bytes, cudaMemcpyDeviceToHost);
    }
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[cls::profile] dump_device(%s) memcpy failed: %s\n",
                     name, cudaGetErrorString(err));
        return;
    }

    State& s = state();
    int seq;
    std::string base;
    {
        std::lock_guard<std::mutex> lk(s.mu);
        if (!s.on.load(std::memory_order_relaxed)) return;
        seq = s.dump_seq[name]++;
        char buf[32];
        std::snprintf(buf, sizeof(buf), "_%04d", seq);
        base = s.dumps_dir + "/" + name + buf;
    }
    const std::string bin_path = base + ".bin";
    const std::string json_path = base + ".json";
    if (FILE* fp = std::fopen(bin_path.c_str(), "wb")) {
        std::fwrite(host.data(), 1, bytes, fp);
        std::fclose(fp);
    } else {
        std::fprintf(stderr, "[cls::profile] dump_device(%s) cannot open %s\n",
                     name, bin_path.c_str());
        return;
    }
    if (FILE* fp = std::fopen(json_path.c_str(), "w")) {
        std::fprintf(fp,
                     "{\"name\":\"%s\",\"dtype\":\"%s\",\"n_elem\":%zu,\"seq\":%d}\n",
                     name, dtype_name(dtype), n_elem, seq);
        std::fclose(fp);
    }
}

}  // namespace cls::profile
