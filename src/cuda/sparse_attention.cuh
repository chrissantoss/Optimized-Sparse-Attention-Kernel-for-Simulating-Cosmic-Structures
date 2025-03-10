#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cutlass/gemm/device/gemm.h>

// Compressed Sparse Row (CSR) format for sparse matrices
struct CSRMatrix {
    int* row_ptrs;      // Row pointers (size: rows + 1)
    int* col_indices;   // Column indices (size: nnz)
    float* values;      // Values (size: nnz)
    int rows;           // Number of rows
    int cols;           // Number of columns
    int nnz;            // Number of non-zero elements
};

// Convert a dense binary mask to CSR format
cudaError_t dense_mask_to_csr(
    const bool* mask,           // Input dense mask (NxN)
    int N,                      // Dimension of the mask
    CSRMatrix* csr_matrix       // Output CSR matrix
);

// Free CSR matrix memory
void free_csr_matrix(CSRMatrix* csr_matrix);

// Sparse attention forward pass
// Computes: O = softmax(Q * K^T) * V, but only for non-zero entries in the mask
cudaError_t sparse_attention_forward(
    const half* Q,              // Query matrix (NxD)
    const half* K,              // Key matrix (NxD)
    const half* V,              // Value matrix (NxD)
    const CSRMatrix* mask,      // Sparsity mask in CSR format
    half* O,                    // Output matrix (NxD)
    float* workspace,           // Workspace memory
    size_t workspace_size,      // Workspace size in bytes
    cudaStream_t stream         // CUDA stream
);

// Sparse attention backward pass (if time allows)
cudaError_t sparse_attention_backward(
    const half* Q,              // Query matrix (NxD)
    const half* K,              // Key matrix (NxD)
    const half* V,              // Value matrix (NxD)
    const half* dO,             // Gradient of output (NxD)
    const CSRMatrix* mask,      // Sparsity mask in CSR format
    half* dQ,                   // Gradient of Q (NxD)
    half* dK,                   // Gradient of K (NxD)
    half* dV,                   // Gradient of V (NxD)
    float* workspace,           // Workspace memory
    size_t workspace_size,      // Workspace size in bytes
    cudaStream_t stream         // CUDA stream
);

// Calculate required workspace size for sparse attention
size_t get_sparse_attention_workspace_size(
    int N,                      // Number of particles
    int D,                      // Feature dimension
    const CSRMatrix* mask       // Sparsity mask in CSR format
);

// Debug utilities
void print_kernel_info();
void print_csr_stats(const CSRMatrix* csr_matrix); 