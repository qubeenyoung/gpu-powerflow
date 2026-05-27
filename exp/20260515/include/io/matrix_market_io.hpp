#pragma once

#include "io/linear_system.hpp"

#include <filesystem>

namespace benchio {

/// Load a real/integer/pattern general Matrix Market coordinate file as fp32 CSR.
CsrMatrix<float, int> load_matrix_market_csr_fp32(const std::filesystem::path& path);

}  // namespace benchio
