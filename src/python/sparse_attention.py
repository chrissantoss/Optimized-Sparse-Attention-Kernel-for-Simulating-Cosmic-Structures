import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, Optional, Dict, Any

try:
    from .sparse_attention_cuda import dense_mask_to_csr, sparse_attention_forward, get_device_info
    CUDA_AVAILABLE = True
except ImportError:
    print("Warning: CUDA extension not available. Using fallback implementation.")
    CUDA_AVAILABLE = False

def create_sparsity_mask_from_positions(
    positions: np.ndarray,
    threshold: float,
    periodic: bool = False,
    box_size: Optional[float] = None
) -> np.ndarray:
    """
    Create a sparsity mask based on particle positions.
    
    Args:
        positions: Particle positions, shape (N, 3)
        threshold: Distance threshold for interactions
        periodic: Whether to use periodic boundary conditions
        box_size: Size of the simulation box (required if periodic=True)
        
    Returns:
        Binary mask of shape (N, N)
    """
    N = positions.shape[0]
    mask = np.zeros((N, N), dtype=bool)
    
    if periodic and box_size is None:
        raise ValueError("box_size must be provided when periodic=True")
    
    for i in range(N):
        for j in range(N):
            if i == j:
                # Particles always interact with themselves
                mask[i, j] = True
                continue
            
            # Calculate distance
            dist = np.linalg.norm(positions[i] - positions[j])
            
            if periodic and box_size is not None:
                # Apply periodic boundary conditions
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dz = positions[i, 2] - positions[j, 2]
                
                if abs(dx) > box_size / 2:
                    dx = box_size - abs(dx)
                if abs(dy) > box_size / 2:
                    dy = box_size - abs(dy)
                if abs(dz) > box_size / 2:
                    dz = box_size - abs(dz)
                
                dist = np.sqrt(dx**2 + dy**2 + dz**2)
            
            # Set mask based on distance threshold
            mask[i, j] = dist < threshold
    
    return mask

def create_block_sparsity_mask(N: int, block_size: int) -> np.ndarray:
    """
    Create a block-sparse mask.
    
    Args:
        N: Number of particles
        block_size: Size of each block
        
    Returns:
        Binary mask of shape (N, N)
    """
    num_blocks = (N + block_size - 1) // block_size
    mask = np.zeros((N, N), dtype=bool)
    
    for i in range(num_blocks):
        for j in range(num_blocks):
            # Only include blocks on the diagonal and adjacent to it
            if abs(i - j) <= 1:
                i_start = i * block_size
                i_end = min((i + 1) * block_size, N)
                j_start = j * block_size
                j_end = min((j + 1) * block_size, N)
                
                mask[i_start:i_end, j_start:j_end] = True
    
    return mask

def sparse_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """
    Apply sparse attention using the CUDA kernel.
    
    Args:
        q: Query matrix, shape (N, D)
        k: Key matrix, shape (N, D)
        v: Value matrix, shape (N, D)
        mask: Binary mask, shape (N, N)
        
    Returns:
        Output matrix, shape (N, D)
    """
    if not CUDA_AVAILABLE:
        return _sparse_attention_fallback(q, k, v, mask)
    
    # Convert mask to CSR format
    row_ptrs, col_indices, values, N, nnz = dense_mask_to_csr(mask)
    
    # Run sparse attention
    output = sparse_attention_forward(q, k, v, row_ptrs, col_indices, values, N, nnz)
    
    return output

def _sparse_attention_fallback(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """
    Fallback implementation of sparse attention using NumPy.
    
    Args:
        q: Query matrix, shape (N, D)
        k: Key matrix, shape (N, D)
        v: Value matrix, shape (N, D)
        mask: Binary mask, shape (N, N)
        
    Returns:
        Output matrix, shape (N, D)
    """
    N, D = q.shape
    
    # Compute QK^T
    qk = np.matmul(q, k.T)
    
    # Apply mask
    qk_masked = qk * mask
    
    # Apply softmax along rows
    max_vals = np.max(qk_masked, axis=1, keepdims=True)
    exp_qk = np.exp(qk_masked - max_vals) * mask
    sum_exp = np.sum(exp_qk, axis=1, keepdims=True)
    attention = exp_qk / (sum_exp + 1e-10)
    
    # Compute output
    output = np.matmul(attention, v)
    
    return output

# JAX custom operation for sparse attention
@partial(jax.custom_jvp, nondiff_argnums=(3,))
def sparse_attention_jax(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    mask: np.ndarray
) -> jnp.ndarray:
    """
    JAX wrapper for sparse attention.
    
    Args:
        q: Query matrix, shape (N, D)
        k: Key matrix, shape (N, D)
        v: Value matrix, shape (N, D)
        mask: Binary mask, shape (N, N)
        
    Returns:
        Output matrix, shape (N, D)
    """
    # Convert JAX arrays to NumPy
    q_np = np.array(q)
    k_np = np.array(k)
    v_np = np.array(v)
    
    # Run sparse attention
    output_np = sparse_attention(q_np, k_np, v_np, mask)
    
    # Convert back to JAX
    return jnp.array(output_np)

@sparse_attention_jax.defjvp
def sparse_attention_jvp(mask, primals, tangents):
    """
    Custom JVP rule for sparse attention.
    
    This is a simplified implementation that doesn't compute the exact gradients.
    For a complete implementation, we would need to implement the backward pass in CUDA.
    """
    q, k, v = primals
    dq, dk, dv = tangents
    
    # Forward pass
    output = sparse_attention_jax(q, k, v, mask)
    
    # Approximate JVP (this is not exact)
    # For a proper implementation, we would need the backward pass in CUDA
    doutput = sparse_attention_jax(dq, k, v, mask) + \
              sparse_attention_jax(q, dk, v, mask) + \
              sparse_attention_jax(q, k, dv, mask)
    
    return output, doutput

def benchmark_sparse_vs_dense(
    N: int,
    D: int,
    sparsity: float,
    num_runs: int = 10
) -> Dict[str, Any]:
    """
    Benchmark sparse attention against dense attention.
    
    Args:
        N: Number of particles
        D: Feature dimension
        sparsity: Target sparsity (fraction of zeros in the mask)
        num_runs: Number of benchmark runs
        
    Returns:
        Dictionary with benchmark results
    """
    import time
    import torch
    
    # Create random data
    q = np.random.normal(size=(N, D)).astype(np.float32)
    k = np.random.normal(size=(N, D)).astype(np.float32)
    v = np.random.normal(size=(N, D)).astype(np.float32)
    
    # Create mask with target sparsity
    mask = np.random.random((N, N)) > sparsity
    mask = mask.astype(bool)
    
    # PyTorch dense attention for comparison
    q_torch = torch.tensor(q, dtype=torch.float32)
    k_torch = torch.tensor(k, dtype=torch.float32)
    v_torch = torch.tensor(v, dtype=torch.float32)
    mask_torch = torch.tensor(mask, dtype=torch.bool)
    
    def dense_attention_torch():
        qk = torch.matmul(q_torch, k_torch.t())
        qk_masked = qk.masked_fill(~mask_torch, -1e9)
        attention = torch.softmax(qk_masked, dim=1)
        return torch.matmul(attention, v_torch)
    
    # Benchmark sparse attention
    sparse_times = []
    for _ in range(num_runs):
        start = time.time()
        sparse_attention(q, k, v, mask)
        sparse_times.append(time.time() - start)
    
    # Benchmark dense attention
    dense_times = []
    for _ in range(num_runs):
        start = time.time()
        dense_attention_torch()
        dense_times.append(time.time() - start)
    
    # Calculate memory usage
    sparse_memory = q.nbytes + k.nbytes + v.nbytes + mask.sum() * (4 + 4 + 4)  # CSR format
    dense_memory = q.nbytes + k.nbytes + v.nbytes + N * N * 4  # Dense mask
    
    # Results
    results = {
        "N": N,
        "D": D,
        "sparsity": sparsity,
        "nnz": mask.sum(),
        "sparse_time_ms": np.mean(sparse_times) * 1000,
        "dense_time_ms": np.mean(dense_times) * 1000,
        "speedup": np.mean(dense_times) / np.mean(sparse_times),
        "sparse_memory_mb": sparse_memory / (1024 * 1024),
        "dense_memory_mb": dense_memory / (1024 * 1024),
        "memory_reduction": dense_memory / sparse_memory,
    }
    
    return results

def print_device_info():
    """Print CUDA device information."""
    if CUDA_AVAILABLE:
        info = get_device_info()
        print("CUDA Device Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    else:
        print("CUDA is not available.") 