#include "sparse_attention.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/tensor_fill.h>

// Constants for kernel tuning
#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024

// Helper function to check CUDA errors
#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return err; \
    } \
} while(0)

// Convert a dense binary mask to CSR format
cudaError_t dense_mask_to_csr(const bool* mask, int N, CSRMatrix* csr_matrix) {
    // Count non-zero elements
    int nnz = 0;
    for (int i = 0; i < N * N; i++) {
        if (mask[i]) nnz++;
    }
    
    // Allocate memory for CSR format
    CHECK_CUDA(cudaMalloc(&csr_matrix->row_ptrs, (N + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&csr_matrix->col_indices, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&csr_matrix->values, nnz * sizeof(float)));
    
    // Fill row pointers and column indices on CPU, then copy to GPU
    int* h_row_ptrs = new int[N + 1];
    int* h_col_indices = new int[nnz];
    float* h_values = new float[nnz];
    
    h_row_ptrs[0] = 0;
    int idx = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (mask[i * N + j]) {
                h_col_indices[idx] = j;
                h_values[idx] = 1.0f;  // All non-zero values are 1
                idx++;
            }
        }
        h_row_ptrs[i + 1] = idx;
    }
    
    // Copy to GPU
    CHECK_CUDA(cudaMemcpy(csr_matrix->row_ptrs, h_row_ptrs, (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(csr_matrix->col_indices, h_col_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(csr_matrix->values, h_values, nnz * sizeof(float), cudaMemcpyHostToDevice));
    
    // Set matrix properties
    csr_matrix->rows = N;
    csr_matrix->cols = N;
    csr_matrix->nnz = nnz;
    
    // Free host memory
    delete[] h_row_ptrs;
    delete[] h_col_indices;
    delete[] h_values;
    
    return cudaSuccess;
}

// Free CSR matrix memory
void free_csr_matrix(CSRMatrix* csr_matrix) {
    if (csr_matrix->row_ptrs) cudaFree(csr_matrix->row_ptrs);
    if (csr_matrix->col_indices) cudaFree(csr_matrix->col_indices);
    if (csr_matrix->values) cudaFree(csr_matrix->values);
    csr_matrix->row_ptrs = nullptr;
    csr_matrix->col_indices = nullptr;
    csr_matrix->values = nullptr;
    csr_matrix->rows = 0;
    csr_matrix->cols = 0;
    csr_matrix->nnz = 0;
}

// CUDA kernel for computing QK^T for sparse attention
__global__ void sparse_qk_kernel(
    const half* Q,              // Query matrix (NxD)
    const half* K,              // Key matrix (NxD)
    const int* row_ptrs,        // CSR row pointers
    const int* col_indices,     // CSR column indices
    float* S,                   // Output scores (nnz)
    int N,                      // Number of particles
    int D                       // Feature dimension
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= row_ptrs[N]) return;  // nnz
    
    // Find row and column for this non-zero element
    int row = 0;
    while (row_ptrs[row + 1] <= idx) row++;
    int col = col_indices[idx - row_ptrs[row]];
    
    // Compute dot product between Q[row] and K[col]
    float sum = 0.0f;
    for (int d = 0; d < D; d++) {
        sum += __half2float(Q[row * D + d]) * __half2float(K[col * D + d]);
    }
    
    // Store result
    S[idx] = sum;
}

// CUDA kernel for applying softmax along rows in sparse format
__global__ void sparse_softmax_kernel(
    float* S,                   // Scores (nnz)
    const int* row_ptrs,        // CSR row pointers
    int N                       // Number of particles
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    
    int start = row_ptrs[row];
    int end = row_ptrs[row + 1];
    if (start == end) return;  // Empty row
    
    // Find max value for numerical stability
    float max_val = S[start];
    for (int i = start + 1; i < end; i++) {
        max_val = max(max_val, S[i]);
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = start; i < end; i++) {
        S[i] = expf(S[i] - max_val);
        sum += S[i];
    }
    
    // Normalize
    if (sum > 0.0f) {
        for (int i = start; i < end; i++) {
            S[i] /= sum;
        }
    }
}

// CUDA kernel for computing P*V for sparse attention
__global__ void sparse_pv_kernel(
    const float* P,             // Attention weights (nnz)
    const half* V,              // Value matrix (NxD)
    const int* row_ptrs,        // CSR row pointers
    const int* col_indices,     // CSR column indices
    half* O,                    // Output matrix (NxD)
    int N,                      // Number of particles
    int D                       // Feature dimension
) {
    int row = blockIdx.x;
    int d = threadIdx.x;
    
    if (row >= N || d >= D) return;
    
    int start = row_ptrs[row];
    int end = row_ptrs[row + 1];
    
    float sum = 0.0f;
    for (int i = start; i < end; i++) {
        int col = col_indices[i - start];
        sum += P[i] * __half2float(V[col * D + d]);
    }
    
    O[row * D + d] = __float2half(sum);
}

// Main sparse attention forward function
cudaError_t sparse_attention_forward(
    const half* Q,              // Query matrix (NxD)
    const half* K,              // Key matrix (NxD)
    const half* V,              // Value matrix (NxD)
    const CSRMatrix* mask,      // Sparsity mask in CSR format
    half* O,                    // Output matrix (NxD)
    float* workspace,           // Workspace memory
    size_t workspace_size,      // Workspace size in bytes
    cudaStream_t stream         // CUDA stream
) {
    int N = mask->rows;
    int D = 0;  // We need to infer D from the workspace size
    
    // Calculate D based on workspace size
    D = (workspace_size / sizeof(float) - mask->nnz) / N;
    
    // Check if workspace is large enough
    size_t required_size = mask->nnz * sizeof(float);
    if (workspace_size < required_size) {
        fprintf(stderr, "Workspace too small: %zu bytes required, %zu provided\n",
                required_size, workspace_size);
        return cudaErrorInvalidValue;
    }
    
    // Use workspace for attention scores
    float* S = workspace;
    
    // Step 1: Compute QK^T for non-zero elements in the mask
    int threads_per_block = min(BLOCK_SIZE, mask->nnz);
    int num_blocks = (mask->nnz + threads_per_block - 1) / threads_per_block;
    sparse_qk_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        Q, K, mask->row_ptrs, mask->col_indices, S, N, D);
    CHECK_CUDA(cudaGetLastError());
    
    // Step 2: Apply softmax along rows
    threads_per_block = min(BLOCK_SIZE, N);
    num_blocks = (N + threads_per_block - 1) / threads_per_block;
    sparse_softmax_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        S, mask->row_ptrs, N);
    CHECK_CUDA(cudaGetLastError());
    
    // Step 3: Compute P*V
    dim3 grid(N);
    dim3 block(min(D, MAX_THREADS_PER_BLOCK));
    sparse_pv_kernel<<<grid, block, 0, stream>>>(
        S, V, mask->row_ptrs, mask->col_indices, O, N, D);
    CHECK_CUDA(cudaGetLastError());
    
    return cudaSuccess;
}

// Calculate required workspace size for sparse attention
size_t get_sparse_attention_workspace_size(
    int N,                      // Number of particles
    int D,                      // Feature dimension
    const CSRMatrix* mask       // Sparsity mask in CSR format
) {
    // We need space for the attention scores (nnz floats)
    return mask->nnz * sizeof(float);
}

// Debug utilities
void print_kernel_info() {
    int device;
    cudaDeviceProp props;
    
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&props, device);
    
    printf("CUDA Device Info:\n");
    printf("  Device name: %s\n", props.name);
    printf("  Compute capability: %d.%d\n", props.major, props.minor);
    printf("  Total global memory: %.2f GB\n", props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Multiprocessors: %d\n", props.multiProcessorCount);
    printf("  Max threads per block: %d\n", props.maxThreadsPerBlock);
    printf("  Max threads per multiprocessor: %d\n", props.maxThreadsPerMultiProcessor);
    printf("  Warp size: %d\n", props.warpSize);
    printf("  Shared memory per block: %zu KB\n", props.sharedMemPerBlock / 1024);
    printf("  Tensor cores: %s\n", (props.major >= 7) ? "Yes" : "No");
}

void print_csr_stats(const CSRMatrix* csr_matrix) {
    printf("CSR Matrix Stats:\n");
    printf("  Rows: %d\n", csr_matrix->rows);
    printf("  Columns: %d\n", csr_matrix->cols);
    printf("  Non-zero elements: %d\n", csr_matrix->nnz);
    printf("  Sparsity: %.2f%%\n", 100.0f * (1.0f - (float)csr_matrix->nnz / (csr_matrix->rows * csr_matrix->cols)));
    
    // Copy row pointers to host to analyze row distribution
    int* h_row_ptrs = new int[csr_matrix->rows + 1];
    cudaMemcpy(h_row_ptrs, csr_matrix->row_ptrs, (csr_matrix->rows + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Calculate statistics
    int min_nnz_per_row = csr_matrix->nnz;
    int max_nnz_per_row = 0;
    float avg_nnz_per_row = 0.0f;
    
    for (int i = 0; i < csr_matrix->rows; i++) {
        int row_nnz = h_row_ptrs[i + 1] - h_row_ptrs[i];
        min_nnz_per_row = min(min_nnz_per_row, row_nnz);
        max_nnz_per_row = max(max_nnz_per_row, row_nnz);
        avg_nnz_per_row += row_nnz;
    }
    avg_nnz_per_row /= csr_matrix->rows;
    
    printf("  Min non-zeros per row: %d\n", min_nnz_per_row);
    printf("  Max non-zeros per row: %d\n", max_nnz_per_row);
    printf("  Avg non-zeros per row: %.2f\n", avg_nnz_per_row);
    
    delete[] h_row_ptrs;
} 