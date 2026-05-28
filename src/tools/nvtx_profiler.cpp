#include "tools/nvtx_profiler.hpp"

#if defined(PROFILE_ENABLE_NVTX)
#include <cstdint>
#include <string_view>

#include <nvtx3/nvToolsExt.h>
#endif

namespace sparse_direct::nvtx_profiler {
namespace {

#if defined(PROFILE_ENABLE_NVTX)
std::uint32_t color_from_name(std::string_view name)
{
    constexpr std::uint32_t colors[] = {
        0xff1f77b4,
        0xffff7f0e,
        0xff2ca02c,
        0xffd62728,
        0xff9467bd,
        0xff8c564b,
        0xffe377c2,
        0xff7f7f7f,
        0xffbcbd22,
        0xff17becf,
    };

    std::uint32_t hash = 2166136261u;
    for (char ch : name) {
        hash ^= static_cast<unsigned char>(ch);
        hash *= 16777619u;
    }

    return colors[hash % (sizeof(colors) / sizeof(colors[0]))];
}

nvtxEventAttributes_t event_attributes(const std::string& name, const std::string& category)
{
    nvtxEventAttributes_t event{};
    event.version = NVTX_VERSION;
    event.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    event.colorType = NVTX_COLOR_ARGB;
    event.color = color_from_name(category.empty() ? name : category);
    event.messageType = NVTX_MESSAGE_TYPE_ASCII;
    event.message.ascii = name.c_str();
    return event;
}
#endif

}  // namespace

Range::Range(const std::string& name, const std::string& category)
{
#if defined(PROFILE_ENABLE_NVTX)
    const nvtxEventAttributes_t event = event_attributes(name, category);
    nvtxRangePushEx(&event);
#else
    (void)name;
    (void)category;
#endif
}

Range::~Range() noexcept
{
#if defined(PROFILE_ENABLE_NVTX)
    nvtxRangePop();
#endif
}

void mark(const std::string& name)
{
#if defined(PROFILE_ENABLE_NVTX)
    const nvtxEventAttributes_t event = event_attributes(name, {});
    nvtxMarkEx(&event);
#else
    (void)name;
#endif
}

}  // namespace sparse_direct::nvtx_profiler
