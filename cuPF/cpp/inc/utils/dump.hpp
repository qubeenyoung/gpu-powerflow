#pragma once

// ---------------------------------------------------------------------------
// dump.hpp — NR 반복 중 행렬·벡터 덤프 유틸리티
//
// 디버깅 목적으로 각 반복(iteration)에서 벡터, 배열, CSR 행렬을 파일로
// 저장한다. 출력 형식은 사람이 읽을 수 있는 텍스트(키-값 한 줄씩).
//
// 파일 경로 규칙:
//   {directory}/{name}_iter{iteration}.txt
//
// 빌드 설정:
//   CUPF_ENABLE_DUMP 정의  → 실제 파일 I/O 활성화
//   미정의                 → 모든 dump 호출이 즉시 false를 반환하는 no-op
//                            (런타임 오버헤드 없음)
//
// 주요 함수:
//   setDumpDirectory(dir)           : 덤프 저장 디렉터리 설정 (기본값: "dump")
//   setDumpEnabled(bool)            : 덤프 on/off 전환
//   dumpVector(name, iter, values)  : std::vector<T> 저장
//   dumpArray(name, iter, ptr, n)   : raw 포인터 배열 저장
//   dumpCSR(name, iter, ...)        : CSR 희소 행렬(indptr/indices/values) 저장
//   dumpCSRView(name, iter, view)   : CSRView<T,I> 래퍼 오버로드
//
// 전역 편의 함수 (namespace 밖):
//   dump_vector(label, ptr, n)      : dumpArray(..., iter=0) 호출
//   dump_csr(label, ...)            : dumpCSR(..., iter=0) 호출
// ---------------------------------------------------------------------------

#include <algorithm>
#include <complex>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "logger.hpp"

template <typename T, typename IndexType>
struct CSRView;

namespace newton_solver::utils {

// 덤프 전역 상태 (디렉터리, 활성화 여부).
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
inline void writeNamedLine(std::ofstream& out, const std::string& name, const T* values, std::size_t count)
{
    out << name;
    for (std::size_t i = 0; i < count; ++i) {
        out << " " << values[i];
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

template <typename T>
inline bool dumpArray(const std::string& name, int iteration, const T* data, int32_t size)
{
    if (!isDumpEnabled()) {
        return false;
    }
    if (data == nullptr || size < 0) {
        logWarn(std::string("Failed to dump array '") + name + "': invalid pointer or size");
        return false;
    }

    std::vector<T> values(static_cast<std::size_t>(size));
    for (int32_t i = 0; i < size; ++i) {
        values[static_cast<std::size_t>(i)] = data[i];
    }
    return dumpVector(name, iteration, values);
}

template <typename ValueType, typename IndexType>
inline bool dumpCSR(const std::string& name,
                    int iteration,
                    const IndexType* indptr,
                    const IndexType* indices,
                    const ValueType* values,
                    IndexType n_rows,
                    IndexType n_cols = IndexType(-1))
{
    if (!isDumpEnabled()) {
        return false;
    }
    if (indptr == nullptr || indices == nullptr || values == nullptr || n_rows < 0) {
        logWarn(std::string("Failed to dump CSR matrix '") + name + "': invalid pointer or size");
        return false;
    }

    try {
        const std::size_t row_count = static_cast<std::size_t>(n_rows);
        const std::size_t nnz = static_cast<std::size_t>(indptr[n_rows]);
        IndexType inferred_cols = n_cols;
        if (inferred_cols < 0) {
            inferred_cols = 0;
            for (std::size_t i = 0; i < nnz; ++i) {
                inferred_cols = std::max(inferred_cols, static_cast<IndexType>(indices[i] + 1));
            }
        }

        const std::string path = makeDumpFilePath(name, iteration);
        std::ofstream out = openDumpFile(path);
        writeMatrixHeader(out, "csr_matrix", row_count, static_cast<std::size_t>(inferred_cols), nnz);
        writeNamedLine(out, "row_ptr", indptr, row_count + 1);
        writeNamedLine(out, "col_idx", indices, nnz);
        writeNamedLine(out, "values", values, nnz);

        logInfo(std::string("Dumped CSR matrix: ") + path);
        return true;
    } catch (const std::exception& e) {
        logWarn(std::string("Failed to dump CSR matrix '") + name + "': " + e.what());
        return false;
    }
}

template <typename ValueType, typename IndexType>
inline bool dumpCSRView(const std::string& name,
                        int iteration,
                        const ::CSRView<ValueType, IndexType>& matrix)
{
    return dumpCSR(name,
                   iteration,
                   matrix.indptr,
                   matrix.indices,
                   matrix.data,
                   matrix.rows,
                   matrix.cols);
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
inline void writeNamedLine(std::ofstream& out, const std::string& name, const T* values, std::size_t count)
{
    (void)out;
    (void)name;
    (void)values;
    (void)count;
}

template <typename T>
inline bool dumpVector(const std::string& name, int iteration, const std::vector<T>& values)
{
    (void)name;
    (void)iteration;
    (void)values;
    return false;
}

template <typename T>
inline bool dumpArray(const std::string& name, int iteration, const T* data, int32_t size)
{
    (void)name;
    (void)iteration;
    (void)data;
    (void)size;
    return false;
}

template <typename ValueType, typename IndexType>
inline bool dumpCSR(const std::string& name,
                    int iteration,
                    const IndexType* indptr,
                    const IndexType* indices,
                    const ValueType* values,
                    IndexType n_rows,
                    IndexType n_cols = IndexType(-1))
{
    (void)name;
    (void)iteration;
    (void)indptr;
    (void)indices;
    (void)values;
    (void)n_rows;
    (void)n_cols;
    return false;
}

template <typename ValueType, typename IndexType>
inline bool dumpCSRView(const std::string& name,
                        int iteration,
                        const ::CSRView<ValueType, IndexType>& matrix)
{
    (void)name;
    (void)iteration;
    (void)matrix;
    return false;
}

#endif

}  // namespace newton_solver::utils

template <typename T>
inline void dump_vector(const std::string& label, const T* data, int32_t n)
{
    (void)::newton_solver::utils::dumpArray(label, 0, data, n);
}

template <typename ValueType, typename IndexType = int32_t>
inline void dump_csr(const std::string& label,
                     const IndexType* indptr,
                     const IndexType* indices,
                     const ValueType* data,
                     IndexType n_rows)
{
    (void)::newton_solver::utils::dumpCSR(label, 0, indptr, indices, data, n_rows);
}
