#include "nr_data.hpp"
#include "io.hpp"
#include "spdlog/spdlog.h"
#include <nlohmann/json.hpp>
#include <fstream>

// 소영수정: CUDA 사용 시에만 CUDA 헤더 포함
#ifdef USE_CUDA
#include "cuda_runtime.h"
#include "cuda_utils.cuh"
#endif

using namespace pf_io;

void
nr_data::NRData::load_data(const std::string& case_name) 
{
    std::string data_path = std::string(DATASET_ROOT) + "/" + case_name;

    try {
        std::string ybus_file = data_path + "/Ybus.npz";
        Ybus = load_complex_csc<EigenSparseMatCSC>(ybus_file);

        std::string sbus_file = data_path + "/Sbus.npy";
        std::vector<std::complex<double>> sbus_vec = load_complex_vector(sbus_file);
        Sbus = Eigen::Map<VectorXcd>(sbus_vec.data(), sbus_vec.size());                 // std vector to eigen vector

        std::string v0_file = data_path + "/V0.npy";
        std::vector<std::complex<double>> v0_vec = load_complex_vector(v0_file);
        V0 = Eigen::Map<VectorXcd>(v0_vec.data(), v0_vec.size());

        std::string pv_file = data_path + "/pv.npy";
        std::vector<int32_t> pv_vec = load_int32_vector(pv_file);
        pv = Eigen::Map<VectorXi32>(pv_vec.data(), pv_vec.size());
        
        std::string pq_file = data_path + "/pq.npy";
        std::vector<int32_t> pq_vec = load_int32_vector(pq_file);
        pq = Eigen::Map<VectorXi32>(pq_vec.data(), pq_vec.size());
        
        spdlog::info("Successfully loaded all data for {}.", case_name);
        spdlog::debug("Ybus: {}x{}, Sbus: {}, V0: {}, pv: {}, pq: {}",
            Ybus.rows(), Ybus.cols(), Sbus.size(), V0.size(), pv.size(), pq.size());

        } catch (const std::exception& e) {
            spdlog::error("Failed to load data for {}: {}", case_name, e.what());
            throw; 
        }
}

// 소영수정: CUDA 사용 시에만 컴파일
#ifdef USE_CUDA
nr_data::NRDeviceData
nr_data::NRData::to_device(void)
{
    spdlog::info("Transferring data to device (GPU)...");

    NRDeviceData d_data; // 반환할 디바이스 구조체

    try {
        int32_t nbus = this->V0.size();
        int32_t npv = this->pv.size();
        int32_t npq = this->pq.size();

        size_t vector_bytes = nbus * sizeof(cuFloatComplex);
        
        CUDA_CHECK(cudaMalloc(&d_data.Sbus, vector_bytes));
        CUDA_CHECK(cudaMemcpy(d_data.Sbus, this->Sbus.data(), vector_bytes, cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaMalloc(&d_data.V0, vector_bytes));
        CUDA_CHECK(cudaMemcpy(d_data.V0, this->V0.data(), vector_bytes, cudaMemcpyHostToDevice));

        size_t pv_bytes = npv * sizeof(int32_t);
        if (pv_bytes > 0) { 
            CUDA_CHECK(cudaMalloc(&d_data.pv, pv_bytes));
            CUDA_CHECK(cudaMemcpy(d_data.pv, this->pv.data(), pv_bytes, cudaMemcpyHostToDevice));
        }

        size_t pq_bytes = npq * sizeof(int32_t);
        if (pq_bytes > 0) {
            CUDA_CHECK(cudaMalloc(&d_data.pq, pq_bytes));
            CUDA_CHECK(cudaMemcpy(d_data.pq, this->pq.data(), pq_bytes, cudaMemcpyHostToDevice));
        }
                
        this->Ybus.makeCompressed();         
        Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> Ycsr = this->Ybus;
        
        int32_t nnz = Ycsr.nonZeros();
        int32_t n_rows = Ycsr.rows();
        int32_t n_cols = Ycsr.cols();

        cudaMalloc(&d_data.Ybus_val, sizeof(cuDoubleComplex) * nnz);
        cudaMalloc(&d_data.Ybus_col_ind, sizeof(int32_t) * nnz);
        cudaMalloc(&d_data.Ybus_row_ptr, sizeof(int32_t) * (n_rows + 1));

        cudaMemcpy(d_data.Ybus_val, Ycsr.valuePtr(), sizeof(cuDoubleComplex) * nnz, cudaMemcpyHostToDevice);
        cudaMemcpy(d_data.Ybus_col_ind, Ycsr.innerIndexPtr(), sizeof(int32_t) * nnz, cudaMemcpyHostToDevice);
        cudaMemcpy(d_data.Ybus_row_ptr, Ycsr.outerIndexPtr(), sizeof(int32_t) * (n_rows + 1), cudaMemcpyHostToDevice);

        spdlog::info("Data transfer to GPU complete.");

    } catch (const std::exception& e) {
        spdlog::error("Failed during data transfer to device: {}", e.what());
        throw; // 예외를 다시 던져서 호출자가 알 수 있도록 함
    }
    
    return d_data;
}

// 생성자: 모든 포인터를 null로 초기화
nr_data::NRDeviceData::NRDeviceData() :
    Ybus_val(nullptr), Ybus_col_ind(nullptr), Ybus_row_ptr(nullptr),
    Sbus(nullptr), V0(nullptr), pv(nullptr), pq(nullptr) {}

nr_data::NRDeviceData::~NRDeviceData()
{
    cudaFree(Ybus_val);
    cudaFree(Ybus_col_ind);
    cudaFree(Ybus_row_ptr);
    cudaFree(Sbus);
    cudaFree(V0);
    cudaFree(pv);
    cudaFree(pq);

    spdlog::debug("NRData_Device deallocated");
}
#endif  // 소영수정: USE_CUDA

void
nr_data::NRResult::save_result(const std::string& case_name)
{
    std::string result_path = std::string(DATASET_ROOT) + "/" + case_name;

    try {
        // eigen vector to std vector
        std::vector<std::complex<double>> v_stdvec(V.data(), V.data() + V.size());

        std::string v_file = result_path + "/V.npy";
        save_vector(v_file, v_stdvec); 

        nlohmann::json summary;
        summary["converged"] = converged;
        summary["iter"] = iter;
        summary["normF"] = normF;

        // .json 파일로 저장 
        std::string summary_file = result_path + "/summary.json"; 
        std::ofstream outfile(summary_file);
        
        if (outfile.is_open()) {
            outfile << summary.dump(4); 
            outfile.close();
            spdlog::info("Saved summary to {}", summary_file);
        } else {
            spdlog::error("Failed to open file for writing: {}", summary_file);
        }

    } catch (const std::exception& e) {
        spdlog::error("Failed to save results for {}: {}", case_name, e.what());
    }
}

