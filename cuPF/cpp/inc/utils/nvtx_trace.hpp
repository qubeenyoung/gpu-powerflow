#pragma once

// ---------------------------------------------------------------------------
// nvtx_trace.hpp — NVTX 프로파일링 레이어
//
// NVIDIA Nsight Systems에서 시각적으로 확인할 수 있는 NVTX 범위(range)와
// 이벤트(mark)를 RAII 방식으로 삽입하는 래퍼다.
//
// 빌드 설정:
//   CUPF_ENABLE_NVTX 정의  → nvToolsExt를 사용하는 실제 NVTX 호출 활성화
//   미정의                 → ScopedNvtxRange / markNvtxEvent / CUPF_NVTX_RANGE 가
//                            모두 no-op으로 컴파일됨
//
// 핵심 API:
//   ScopedNvtxRange(name, color, category)
//     - 생성 시 nvtxRangePushEx, 소멸 시 nvtxRangePop 호출 (RAII)
//     - setNvtxEnabled(false) 이면 push/pop 생략
//
//   markNvtxEvent(name, color, category)
//     - 단일 시점 이벤트를 타임라인에 표시 (nvtxMarkEx)
//
//   CUPF_NVTX_RANGE(name)   매크로
//     - 현재 스코프 전체를 NVTX 범위로 감싸는 단축 매크로
//     - 내부적으로 __LINE__ 기반 유니크 변수명을 생성
//
//   CUPF_NVTX_MARK(name)    매크로
//     - markNvtxEvent 단축 매크로
//
// NvtxColor:
//   Tableau 10 팔레트에서 가져온 ARGB 값 (Blue/Orange/Red/Teal/Green/Purple/Gray).
//   각 Op 종류에 고유 색상을 할당하면 Nsight Systems 타임라인이 읽기 쉬워진다.
// ---------------------------------------------------------------------------

#include <cstdint>

#ifdef CUPF_ENABLE_NVTX
  #include <nvtx3/nvToolsExt.h>
#endif

namespace newton_solver::utils {

// Tableau 10 팔레트 기반 ARGB 색상. Nsight Systems 타임라인에서 Op별 구분에 사용.
enum class NvtxColor : uint32_t {
    Blue = 0xFF4E79A7u,
    Orange = 0xFFF28E2Bu,
    Red = 0xFFE15759u,
    Teal = 0xFF76B7B2u,
    Green = 0xFF59A14Fu,
    Purple = 0xFFB07AA1u,
    Gray = 0xFF9C9C9Cu,
};

struct NvtxState {
    bool enabled = true;
};

inline NvtxState& nvtxState()
{
    static NvtxState state;
    return state;
}

inline void setNvtxEnabled(bool enabled)
{
    nvtxState().enabled = enabled;
}

inline bool isNvtxEnabled()
{
    return nvtxState().enabled;
}

#ifdef CUPF_ENABLE_NVTX

inline bool nvtxAvailable()
{
    return true;
}

namespace detail {

inline nvtxEventAttributes_t makeEventAttributes(const char* name, uint32_t argb, uint32_t category)
{
    nvtxEventAttributes_t attr{};
    attr.version = NVTX_VERSION;
    attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType = NVTX_COLOR_ARGB;
    attr.color = argb;
    attr.category = category;
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = name != nullptr ? name : "<null>";
    return attr;
}

}  // namespace detail

class ScopedNvtxRange {
public:
    explicit ScopedNvtxRange(const char* name,
                             NvtxColor color = NvtxColor::Blue,
                             uint32_t category = 0)
        : active_(isNvtxEnabled())
    {
        if (active_) {
            const auto attr = detail::makeEventAttributes(
                name, static_cast<uint32_t>(color), category);
            nvtxRangePushEx(&attr);
        }
    }

    ~ScopedNvtxRange()
    {
        if (active_) {
            nvtxRangePop();
        }
    }

    ScopedNvtxRange(const ScopedNvtxRange&) = delete;
    ScopedNvtxRange& operator=(const ScopedNvtxRange&) = delete;
    ScopedNvtxRange(ScopedNvtxRange&&) = delete;
    ScopedNvtxRange& operator=(ScopedNvtxRange&&) = delete;

private:
    bool active_ = false;
};

inline void markNvtxEvent(const char* name,
                          NvtxColor color = NvtxColor::Orange,
                          uint32_t category = 0)
{
    if (!isNvtxEnabled()) {
        return;
    }

    const auto attr = detail::makeEventAttributes(name, static_cast<uint32_t>(color), category);
    nvtxMarkEx(&attr);
}

#else

inline bool nvtxAvailable()
{
    return false;
}

class ScopedNvtxRange {
public:
    explicit ScopedNvtxRange(const char* name,
                             NvtxColor color = NvtxColor::Blue,
                             uint32_t category = 0)
    {
        (void)name;
        (void)color;
        (void)category;
    }
};

inline void markNvtxEvent(const char* name,
                          NvtxColor color = NvtxColor::Orange,
                          uint32_t category = 0)
{
    (void)name;
    (void)color;
    (void)category;
}

#endif

}  // namespace newton_solver::utils

#define CUPF_NVTX_DETAIL_CONCAT_INNER(x, y) x##y
#define CUPF_NVTX_DETAIL_CONCAT(x, y) CUPF_NVTX_DETAIL_CONCAT_INNER(x, y)

#ifndef CUPF_NVTX_RANGE
  #define CUPF_NVTX_RANGE(name)                                                   \
      ::newton_solver::utils::ScopedNvtxRange                                     \
          CUPF_NVTX_DETAIL_CONCAT(cupf_nvtx_scope_, __LINE__)((name))
#endif

#ifndef CUPF_NVTX_MARK
  #define CUPF_NVTX_MARK(name)                                                    \
      do {                                                                        \
          ::newton_solver::utils::markNvtxEvent((name));                          \
      } while (0)
#endif
