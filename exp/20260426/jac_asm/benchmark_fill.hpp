#pragma once

#include "benchmark_support.hpp"
#include "common/jacobian_build.hpp"
#include "edge/edge_build.hpp"

namespace exp20260426::jac_asm_bench {

float runEdgeFill(const Options& options,
                  const CaseData& data,
                  const EdgeYbusMap& edge_map,
                  const JacobianPattern& pattern,
                  const JacobianMap& map);

float runEdgeFillNoAtomic(const Options& options,
                          const CaseData& data,
                          const EdgeYbusMap& edge_map,
                          const JacobianPattern& pattern,
                          const JacobianMap& map);

float runVertexFill(const Options& options,
                    const CaseData& data,
                    const BusIndexMap& index,
                    const JacobianPattern& pattern,
                    const JacobianMap& map);

}  // namespace exp20260426::jac_asm_bench
