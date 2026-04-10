#pragma once

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "logger.hpp"

#ifdef DUMP_DATA
#ifndef CUPF_ENABLE_DUMP
#define CUPF_ENABLE_DUMP
#endif
#endif

namespace newton_solver {

template <typename T, typename IndexType>
struct CSRMatrix;

template <typename T, typename IndexType>
struct CSCMatrix;

template <typename T, typename IndexType>
struct COOMatrix;

}  // namespace newton_solver

namespace newton_solver::utils {

struct DumpState {
    std::filesystem::path directory = "dump";
    bool enabled = false;
};

inline DumpState& dumpState()
{
    static DumpState state;
    return state;
}

#ifdef CUPF_ENABLE_DUMP

inline void setDumpDirectory(const std::string& directory)
{
    dumpState().directory = directory;
    std::filesystem::create_directories(dumpState().directory);
}

inline std::string getDumpDirectory()
{
    return dumpState().directory.string();
}

inline void setDumpEnabled(bool enabled)
{
    dumpState().enabled = enabled;
}

inline bool isDumpEnabled()
{
    return dumpState().enabled;
}

inline std::string makeDumpFilePath(const std::string& name,
                                    int iteration,
                                    const std::string& extension = ".txt")
{
    std::filesystem::create_directories(dumpState().directory);
    return (dumpState().directory / (name + "_iter" + std::to_string(iteration) + extension)).string();
}

inline std::ofstream openDumpFile(const std::string& path)
{
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open file: " + path);
    }
    return out;
}

inline void writeMatrixHeader(std::ofstream& out,
                              const std::string& type,
                              std::size_t rows,
                              std::size_t cols,
                              std::size_t nnz)
{
    out << "type " << type << '\n';
    out << "rows " << rows << '\n';
    out << "cols " << cols << '\n';
    out << "nnz " << nnz << '\n';
}

template <typename T>
inline void writeNamedLine(std::ofstream& out, const std::string& name, const std::vector<T>& values)
{
    out << name;
    for (const auto& value : values) {
        out << " " << value;
    }
    out << '\n';
}

template <typename T>
inline bool dumpVector(const std::string& name, int iteration, const std::vector<T>& values)
{
    if (!isDumpEnabled()) {
        return false;
    }

    try {
        const std::string path = makeDumpFilePath(name, iteration);
        std::ofstream out = openDumpFile(path);

        out << "type vector\n";
        out << "size " << values.size() << '\n';
        out << "values\n";
        for (std::size_t i = 0; i < values.size(); ++i) {
            out << i << " " << values[i] << '\n';
        }

        logInfo(std::string("Dumped vector: ") + path);
        return true;
    } catch (const std::exception& e) {
        logWarn(std::string("Failed to dump vector '") + name + "': " + e.what());
        return false;
    }
}

template <typename T, typename IndexType>
inline bool dumpCSRMatrix(const std::string& name,
                          int iteration,
                          const ::newton_solver::CSRMatrix<T, IndexType>& matrix)
{
    if (!isDumpEnabled()) {
        return false;
    }

    try {
        const std::string path = makeDumpFilePath(name, iteration);
        std::ofstream out = openDumpFile(path);

        writeMatrixHeader(out, "csr_matrix", matrix.rows, matrix.cols, matrix.values.size());
        writeNamedLine(out, "row_ptr", matrix.row_ptr);
        writeNamedLine(out, "col_idx", matrix.col_idx);
        writeNamedLine(out, "values", matrix.values);

        logInfo(std::string("Dumped CSR matrix: ") + path);
        return true;
    } catch (const std::exception& e) {
        logWarn(std::string("Failed to dump CSR matrix '") + name + "': " + e.what());
        return false;
    }
}

template <typename T, typename IndexType>
inline bool dumpCSCMatrix(const std::string& name,
                          int iteration,
                          const ::newton_solver::CSCMatrix<T, IndexType>& matrix)
{
    if (!isDumpEnabled()) {
        return false;
    }

    try {
        const std::string path = makeDumpFilePath(name, iteration);
        std::ofstream out = openDumpFile(path);

        writeMatrixHeader(out, "csc_matrix", matrix.rows, matrix.cols, matrix.values.size());
        writeNamedLine(out, "col_ptr", matrix.col_ptr);
        writeNamedLine(out, "row_idx", matrix.row_idx);
        writeNamedLine(out, "values", matrix.values);

        logInfo(std::string("Dumped CSC matrix: ") + path);
        return true;
    } catch (const std::exception& e) {
        logWarn(std::string("Failed to dump CSC matrix '") + name + "': " + e.what());
        return false;
    }
}

template <typename T, typename IndexType>
inline bool dumpCOOMatrix(const std::string& name,
                          int iteration,
                          const ::newton_solver::COOMatrix<T, IndexType>& matrix)
{
    if (!isDumpEnabled()) {
        return false;
    }

    try {
        const std::string path = makeDumpFilePath(name, iteration);
        std::ofstream out = openDumpFile(path);

        writeMatrixHeader(out, "coo_matrix", matrix.rows, matrix.cols, matrix.values.size());
        writeNamedLine(out, "row_idx", matrix.row_idx);
        writeNamedLine(out, "col_idx", matrix.col_idx);
        writeNamedLine(out, "values", matrix.values);

        logInfo(std::string("Dumped COO matrix: ") + path);
        return true;
    } catch (const std::exception& e) {
        logWarn(std::string("Failed to dump COO matrix '") + name + "': " + e.what());
        return false;
    }
}

#else

inline void setDumpDirectory(const std::string& directory)
{
    dumpState().directory = directory;
}

inline std::string getDumpDirectory()
{
    return dumpState().directory.string();
}

inline void setDumpEnabled(bool enabled)
{
    (void)enabled;
}

inline bool isDumpEnabled()
{
    return false;
}

inline std::string makeDumpFilePath(const std::string& name,
                                    int iteration,
                                    const std::string& extension = ".txt")
{
    return (dumpState().directory / (name + "_iter" + std::to_string(iteration) + extension)).string();
}

inline std::ofstream openDumpFile(const std::string& path)
{
    (void)path;
    return std::ofstream();
}

inline void writeMatrixHeader(std::ofstream& out,
                              const std::string& type,
                              std::size_t rows,
                              std::size_t cols,
                              std::size_t nnz)
{
    (void)out;
    (void)type;
    (void)rows;
    (void)cols;
    (void)nnz;
}

template <typename T>
inline void writeNamedLine(std::ofstream& out, const std::string& name, const std::vector<T>& values)
{
    (void)out;
    (void)name;
    (void)values;
}

template <typename T>
inline bool dumpVector(const std::string& name, int iteration, const std::vector<T>& values)
{
    (void)name;
    (void)iteration;
    (void)values;
    return false;
}

template <typename T, typename IndexType>
inline bool dumpCSRMatrix(const std::string& name,
                          int iteration,
                          const ::newton_solver::CSRMatrix<T, IndexType>& matrix)
{
    (void)name;
    (void)iteration;
    (void)matrix;
    return false;
}

template <typename T, typename IndexType>
inline bool dumpCSCMatrix(const std::string& name,
                          int iteration,
                          const ::newton_solver::CSCMatrix<T, IndexType>& matrix)
{
    (void)name;
    (void)iteration;
    (void)matrix;
    return false;
}

template <typename T, typename IndexType>
inline bool dumpCOOMatrix(const std::string& name,
                          int iteration,
                          const ::newton_solver::COOMatrix<T, IndexType>& matrix)
{
    (void)name;
    (void)iteration;
    (void)matrix;
    return false;
}

#endif

template <typename T, typename IndexType>
inline bool dumpMatrix(const std::string& name,
                       int iteration,
                       const ::newton_solver::CSRMatrix<T, IndexType>& matrix)
{
    return dumpCSRMatrix(name, iteration, matrix);
}

template <typename T, typename IndexType>
inline bool dumpMatrix(const std::string& name,
                       int iteration,
                       const ::newton_solver::CSCMatrix<T, IndexType>& matrix)
{
    return dumpCSCMatrix(name, iteration, matrix);
}

template <typename T, typename IndexType>
inline bool dumpMatrix(const std::string& name,
                       int iteration,
                       const ::newton_solver::COOMatrix<T, IndexType>& matrix)
{
    return dumpCOOMatrix(name, iteration, matrix);
}

}  // namespace newton_solver::utils
