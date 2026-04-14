#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace exp_20260413::iterative {

struct LinearSystemSnapshot {
    int32_t rows = 0;
    int32_t cols = 0;
    std::vector<int32_t> row_ptr;
    std::vector<int32_t> col_idx;
    std::vector<double> values;
    std::vector<double> rhs;
    std::vector<double> x_direct;
};

std::string iter_dir_name(int32_t iter);

void write_csr(const std::filesystem::path& path,
               int32_t rows,
               int32_t cols,
               const std::vector<int32_t>& row_ptr,
               const std::vector<int32_t>& col_idx,
               const std::vector<double>& values);

void write_vector(const std::filesystem::path& path,
                  const std::vector<double>& values);

void write_metadata(const std::filesystem::path& path,
                    const std::vector<std::pair<std::string, std::string>>& entries);

LinearSystemSnapshot read_snapshot(const std::filesystem::path& snapshot_dir);

std::vector<std::filesystem::path> find_snapshot_dirs(const std::filesystem::path& root);

double inf_norm(const std::vector<double>& values);

double residual_inf_norm(const LinearSystemSnapshot& snapshot,
                         const std::vector<double>& x);

double diff_inf_norm(const std::vector<double>& lhs,
                     const std::vector<double>& rhs);

}  // namespace exp_20260413::iterative
