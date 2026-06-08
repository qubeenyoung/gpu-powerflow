#pragma once

// Public profiling toolkit for custom_linear_solver experiments.
//
// Purpose: shared instrumentation primitives (CPU/GPU timers, NVTX scopes, device dumps) so
// kernel experiments record measurements through a stable API instead of hand-patching code
// each iteration.
//
// Compile-time gate
// -----------------
// CMake sets `CLS_ENABLE_PROFILE` to ON by default for Debug builds, OFF for Release/
// RelWithDebInfo. When OFF, every macro and class in this header collapses to inline empty
// stubs (the compiler removes the calls entirely), profile.cpp is NOT compiled, and NVTX is
// NOT linked. Override with `-DCLS_ENABLE_PROFILE=ON` to instrument a non-Debug build.
//
// Runtime activation (only relevant when compiled in)
// ---------------------------------------------------
// `CLS_PROFILE_DIR=/some/path` env var enables CSV / dump output. Unset → no file IO, just
// NVTX push/pop (essentially free when no profiler is attached).
//
// Output (only when both compiled-in AND CLS_PROFILE_DIR set)
// -----------------------------------------------------------
//   ${CLS_PROFILE_DIR}/timers.csv     header: kind,name,start_ms,duration_ms
//                                     appended line per CpuScope / GpuScope.
//   ${CLS_PROFILE_DIR}/dumps/<name>_<seq>.bin   raw device bytes (host-side memcpy).
//   ${CLS_PROFILE_DIR}/dumps/<name>_<seq>.json  {"name":..,"dtype":..,"n_elem":..,"seq":..}
//
// Usage
// -----
//   #include "profile/profile.hpp"
//   cls::profile::init();        // safe to call repeatedly; called from Solver ctor
//   {
//     CLS_PROFILE_NVTX("factorize");
//     CLS_PROFILE_CPU("factorize_host");
//     CLS_PROFILE_GPU("factorize_gpu", stream);
//     CLS_PROFILE_DUMP("front_post_factor", d_front, n_elems,
//                      cls::profile::DType::Float32, stream);
//   }

#include <cstddef>
#include <cuda_runtime.h>

#ifndef CLS_ENABLE_PROFILE
#define CLS_ENABLE_PROFILE 0
#endif

namespace cls::profile {

enum class DType {
    Float32 = 0,
    Float64 = 1,
    Int32 = 2,
};

#if CLS_ENABLE_PROFILE

// Real implementations live in profile.cpp.
void init();
void flush();
bool enabled();
void dump_device(const char* name, const void* device_ptr, std::size_t n_elem,
                 DType dtype, cudaStream_t stream);

class NvtxScope {
 public:
    explicit NvtxScope(const char* name);
    ~NvtxScope();
    NvtxScope(const NvtxScope&) = delete;
    NvtxScope& operator=(const NvtxScope&) = delete;
 private:
    int active_;
};

class CpuScope {
 public:
    explicit CpuScope(const char* name);
    ~CpuScope();
    CpuScope(const CpuScope&) = delete;
    CpuScope& operator=(const CpuScope&) = delete;
 private:
    const char* name_;
    long long start_ns_;
    int active_;
};

class GpuScope {
 public:
    GpuScope(const char* name, cudaStream_t stream);
    ~GpuScope();
    GpuScope(const GpuScope&) = delete;
    GpuScope& operator=(const GpuScope&) = delete;
 private:
    const char* name_;
    cudaStream_t stream_;
    cudaEvent_t e_start_;
    cudaEvent_t e_stop_;
    int active_;
};

#else  // ---------- profile disabled: header-only inline stubs ----------

inline void init() {}
inline void flush() {}
inline bool enabled() { return false; }
inline void dump_device(const char*, const void*, std::size_t, DType, cudaStream_t) {}

struct NvtxScope { explicit NvtxScope(const char*) {} };
struct CpuScope { explicit CpuScope(const char*) {} };
struct GpuScope { GpuScope(const char*, cudaStream_t) {} };

#endif

}  // namespace cls::profile

#define CLS_PROFILE_CONCAT_INNER(a, b) a##b
#define CLS_PROFILE_CONCAT(a, b) CLS_PROFILE_CONCAT_INNER(a, b)

#if CLS_ENABLE_PROFILE
#define CLS_PROFILE_NVTX(name) \
    ::cls::profile::NvtxScope CLS_PROFILE_CONCAT(_cls_nvtx_, __LINE__)(name)
#define CLS_PROFILE_CPU(name) \
    ::cls::profile::CpuScope CLS_PROFILE_CONCAT(_cls_cpu_, __LINE__)(name)
#define CLS_PROFILE_GPU(name, stream) \
    ::cls::profile::GpuScope CLS_PROFILE_CONCAT(_cls_gpu_, __LINE__)(name, stream)
#define CLS_PROFILE_DUMP(name, ptr, n, dtype, stream) \
    ::cls::profile::dump_device((name), (ptr), (n), (dtype), (stream))
#else
#define CLS_PROFILE_NVTX(name) ((void)0)
#define CLS_PROFILE_CPU(name) ((void)0)
#define CLS_PROFILE_GPU(name, stream) ((void)0)
#define CLS_PROFILE_DUMP(name, ptr, n, dtype, stream) ((void)0)
#endif
