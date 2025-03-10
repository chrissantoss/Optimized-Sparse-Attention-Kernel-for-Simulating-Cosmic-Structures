#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../cuda/sparse_attention.cuh"

namespace py = pybind11;

// Helper function to check CUDA errors
void check_cuda_error(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error in ") + file + " at line " + 
            std::to_string(line) + ": " + cudaGetErrorString(error));
    }
}

#define CHECK_CUDA(call) check_cuda_error(call, __FILE__, __LINE__)

// Convert dense mask to CSR format
py::tuple dense_mask_to_csr_py(py::array_t<bool> mask) {
    py::buffer_info mask_info = mask.request();
    
    if (mask_info.ndim != 2) {
        throw std::runtime_error("Mask must be a 2D array");
    }
    
    int N = mask_info.shape[0];
    if (N != mask_info.shape[1]) {
        throw std::runtime_error("Mask must be square");
    }
    
    // Create CSR matrix
    CSRMatrix csr_matrix;
    bool* mask_ptr = static_cast<bool*>(mask_info.ptr);
    
    CHECK_CUDA(dense_mask_to_csr(mask_ptr, N, &csr_matrix));
    
    // Copy CSR data to host for returning to Python
    int* h_row_ptrs = new int[N + 1];
    int* h_col_indices = new int[csr_matrix.nnz];
    float* h_values = new float[csr_matrix.nnz];
    
    CHECK_CUDA(cudaMemcpy(h_row_ptrs, csr_matrix.row_ptrs, (N + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_col_indices, csr_matrix.col_indices, csr_matrix.nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_values, csr_matrix.values, csr_matrix.nnz * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Create numpy arrays for returning to Python
    py::array_t<int> row_ptrs({N + 1});
    py::array_t<int> col_indices({csr_matrix.nnz});
    py::array_t<float> values({csr_matrix.nnz});
    
    std::memcpy(row_ptrs.mutable_data(), h_row_ptrs, (N + 1) * sizeof(int));
    std::memcpy(col_indices.mutable_data(), h_col_indices, csr_matrix.nnz * sizeof(int));
    std::memcpy(values.mutable_data(), h_values, csr_matrix.nnz * sizeof(float));
    
    // Free memory
    delete[] h_row_ptrs;
    delete[] h_col_indices;
    delete[] h_values;
    free_csr_matrix(&csr_matrix);
    
    return py::make_tuple(row_ptrs, col_indices, values, N, csr_matrix.nnz);
}

// Sparse attention forward pass
py::array_t<float> sparse_attention_forward_py(
    py::array_t<float> q,
    py::array_t<float> k,
    py::array_t<float> v,
    py::array_t<int> row_ptrs,
    py::array_t<int> col_indices,
    py::array_t<float> values,
    int N,
    int nnz
) {
    py::buffer_info q_info = q.request();
    py::buffer_info k_info = k.request();
    py::buffer_info v_info = v.request();
    py::buffer_info row_ptrs_info = row_ptrs.request();
    py::buffer_info col_indices_info = col_indices.request();
    py::buffer_info values_info = values.request();
    
    if (q_info.ndim != 2 || k_info.ndim != 2 || v_info.ndim != 2) {
        throw std::runtime_error("Q, K, V must be 2D arrays");
    }
    
    int N_q = q_info.shape[0];
    int D = q_info.shape[1];
    
    if (N_q != N || k_info.shape[0] != N || v_info.shape[0] != N ||
        k_info.shape[1] != D || v_info.shape[1] != D) {
        throw std::runtime_error("Q, K, V must have compatible shapes");
    }
    
    // Create CSR matrix
    CSRMatrix csr_matrix;
    csr_matrix.rows = N;
    csr_matrix.cols = N;
    csr_matrix.nnz = nnz;
    
    // Allocate device memory for CSR matrix
    CHECK_CUDA(cudaMalloc(&csr_matrix.row_ptrs, (N + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&csr_matrix.col_indices, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&csr_matrix.values, nnz * sizeof(float)));
    
    // Copy CSR data to device
    CHECK_CUDA(cudaMemcpy(csr_matrix.row_ptrs, row_ptrs_info.ptr, (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(csr_matrix.col_indices, col_indices_info.ptr, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(csr_matrix.values, values_info.ptr, nnz * sizeof(float), cudaMemcpyHostToDevice));
    
    // Allocate device memory for Q, K, V, O
    half *d_Q, *d_K, *d_V, *d_O;
    CHECK_CUDA(cudaMalloc(&d_Q, N * D * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_K, N * D * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_V, N * D * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_O, N * D * sizeof(half)));
    
    // Convert float to half and copy to device
    half* h_Q = new half[N * D];
    half* h_K = new half[N * D];
    half* h_V = new half[N * D];
    
    float* q_ptr = static_cast<float*>(q_info.ptr);
    float* k_ptr = static_cast<float*>(k_info.ptr);
    float* v_ptr = static_cast<float*>(v_info.ptr);
    
    for (int i = 0; i < N * D; i++) {
        h_Q[i] = __float2half(q_ptr[i]);
        h_K[i] = __float2half(k_ptr[i]);
        h_V[i] = __float2half(v_ptr[i]);
    }
    
    CHECK_CUDA(cudaMemcpy(d_Q, h_Q, N * D * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K, N * D * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V, N * D * sizeof(half), cudaMemcpyHostToDevice));
    
    // Allocate workspace
    size_t workspace_size = get_sparse_attention_workspace_size(N, D, &csr_matrix);
    float* d_workspace;
    CHECK_CUDA(cudaMalloc(&d_workspace, workspace_size));
    
    // Create CUDA stream
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    
    // Run sparse attention
    CHECK_CUDA(sparse_attention_forward(
        d_Q, d_K, d_V, &csr_matrix, d_O, d_workspace, workspace_size, stream));
    
    // Copy result back to host
    half* h_O = new half[N * D];
    CHECK_CUDA(cudaMemcpy(h_O, d_O, N * D * sizeof(half), cudaMemcpyDeviceToHost));
    
    // Convert half to float for Python
    py::array_t<float> output({N, D});
    float* output_ptr = static_cast<float*>(output.request().ptr);
    
    for (int i = 0; i < N * D; i++) {
        output_ptr[i] = __half2float(h_O[i]);
    }
    
    // Free memory
    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_O;
    
    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_O));
    CHECK_CUDA(cudaFree(d_workspace));
    
    free_csr_matrix(&csr_matrix);
    CHECK_CUDA(cudaStreamDestroy(stream));
    
    return output;
}

// Print CUDA device info
py::dict get_device_info() {
    int device;
    cudaDeviceProp props;
    
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&props, device));
    
    py::dict info;
    info["name"] = props.name;
    info["compute_capability"] = std::to_string(props.major) + "." + std::to_string(props.minor);
    info["total_memory_gb"] = props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0);
    info["multiprocessors"] = props.multiProcessorCount;
    info["max_threads_per_block"] = props.maxThreadsPerBlock;
    info["max_threads_per_mp"] = props.maxThreadsPerMultiProcessor;
    info["warp_size"] = props.warpSize;
    info["shared_memory_per_block_kb"] = props.sharedMemPerBlock / 1024;
    info["has_tensor_cores"] = props.major >= 7;
    
    return info;
}

PYBIND11_MODULE(sparse_attention_cuda, m) {
    m.doc() = "CUDA implementation of sparse attention for cosmic simulations";
    
    m.def("dense_mask_to_csr", &dense_mask_to_csr_py, 
          "Convert a dense binary mask to CSR format",
          py::arg("mask"));
    
    m.def("sparse_attention_forward", &sparse_attention_forward_py,
          "Sparse attention forward pass",
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("row_ptrs"), py::arg("col_indices"), py::arg("values"),
          py::arg("N"), py::arg("nnz"));
    
    m.def("get_device_info", &get_device_info,
          "Get CUDA device information");
} 