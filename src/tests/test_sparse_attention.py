import numpy as np
import pytest

# Try to import torch, but make it optional
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Some tests will be skipped.")

from ..python.sparse_attention import (
    sparse_attention,
    create_sparsity_mask_from_positions,
    create_block_sparsity_mask,
    _sparse_attention_fallback
)

def test_create_sparsity_mask_from_positions():
    """Test creating a sparsity mask from particle positions."""
    # Create a simple test case with 4 particles
    positions = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 5, 0],
        [5, 5, 5]
    ])
    
    # Test with threshold = 2.0
    mask = create_sparsity_mask_from_positions(positions, threshold=2.0)
    
    # Expected mask:
    # - Particle 0 interacts with itself and particle 1
    # - Particle 1 interacts with itself and particle 0
    # - Particle 2 interacts with itself
    # - Particle 3 interacts with itself
    expected_mask = np.array([
        [True, True, False, False],
        [True, True, False, False],
        [False, False, True, False],
        [False, False, False, True]
    ])
    
    np.testing.assert_array_equal(mask, expected_mask)
    
    # Test with periodic boundary conditions
    # Not testing this in detail, just checking it runs
    mask_periodic = create_sparsity_mask_from_positions(
        positions, threshold=2.0, periodic=True, box_size=10.0
    )
    assert mask_periodic.shape == (4, 4)

def test_create_block_sparsity_mask():
    """Test creating a block-sparse mask."""
    # Test with N=6, block_size=2
    mask = create_block_sparsity_mask(N=6, block_size=2)
    
    # Expected mask: block-diagonal with adjacent blocks
    expected_mask = np.array([
        [True, True, True, True, False, False],
        [True, True, True, True, False, False],
        [True, True, True, True, True, True],
        [True, True, True, True, True, True],
        [False, False, True, True, True, True],
        [False, False, True, True, True, True]
    ])
    
    np.testing.assert_array_equal(mask, expected_mask)

def test_sparse_attention_fallback():
    """Test the fallback implementation of sparse attention."""
    # Skip if PyTorch is not available
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")
    
    # Create a simple test case
    N, D = 4, 3
    q = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1]
    ], dtype=np.float32)
    k = q.copy()
    v = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ], dtype=np.float32)
    
    # Full mask (dense attention)
    mask_full = np.ones((N, N), dtype=bool)
    
    # Compute with fallback implementation
    output_fallback = _sparse_attention_fallback(q, k, v, mask_full)
    
    # Compute with PyTorch for reference
    q_torch = torch.tensor(q, dtype=torch.float32)
    k_torch = torch.tensor(k, dtype=torch.float32)
    v_torch = torch.tensor(v, dtype=torch.float32)
    
    qk = torch.matmul(q_torch, k_torch.t())
    attention = torch.softmax(qk, dim=1)
    output_torch = torch.matmul(attention, v_torch).numpy()
    
    # Check results match
    np.testing.assert_allclose(output_fallback, output_torch, rtol=1e-5)
    
    # Test with sparse mask
    mask_sparse = np.array([
        [True, True, False, False],
        [True, True, False, False],
        [False, False, True, True],
        [False, False, True, True]
    ])
    
    # Compute with fallback implementation
    output_sparse = _sparse_attention_fallback(q, k, v, mask_sparse)
    
    # Compute with PyTorch for reference
    qk_masked = qk.clone()
    qk_masked[~torch.tensor(mask_sparse)] = -1e9
    attention_masked = torch.softmax(qk_masked, dim=1)
    output_torch_masked = torch.matmul(attention_masked, v_torch).numpy()
    
    # Check results match
    np.testing.assert_allclose(output_sparse, output_torch_masked, rtol=1e-5)

def test_sparse_attention_cuda():
    """Test the CUDA implementation of sparse attention."""
    try:
        from ..python.sparse_attention import CUDA_AVAILABLE
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA not available")
    except ImportError:
        pytest.skip("CUDA not available")
    
    # Create a simple test case
    N, D = 4, 3
    q = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1]
    ], dtype=np.float32)
    k = q.copy()
    v = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ], dtype=np.float32)
    
    # Full mask (dense attention)
    mask_full = np.ones((N, N), dtype=bool)
    
    # Compute with CUDA implementation
    output_cuda = sparse_attention(q, k, v, mask_full)
    
    # Compute with fallback implementation for reference
    output_fallback = _sparse_attention_fallback(q, k, v, mask_full)
    
    # Check results match
    np.testing.assert_allclose(output_cuda, output_fallback, rtol=1e-4)
    
    # Test with sparse mask
    mask_sparse = np.array([
        [True, True, False, False],
        [True, True, False, False],
        [False, False, True, True],
        [False, False, True, True]
    ])
    
    # Compute with CUDA implementation
    output_cuda_sparse = sparse_attention(q, k, v, mask_sparse)
    
    # Compute with fallback implementation for reference
    output_fallback_sparse = _sparse_attention_fallback(q, k, v, mask_sparse)
    
    # Check results match
    np.testing.assert_allclose(output_cuda_sparse, output_fallback_sparse, rtol=1e-4)

def test_large_sparse_attention():
    """Test sparse attention with larger matrices."""
    try:
        from ..python.sparse_attention import CUDA_AVAILABLE
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA not available")
    except ImportError:
        pytest.skip("CUDA not available")
    
    # Create a larger test case
    N, D = 128, 64
    q = np.random.normal(size=(N, D)).astype(np.float32)
    k = np.random.normal(size=(N, D)).astype(np.float32)
    v = np.random.normal(size=(N, D)).astype(np.float32)
    
    # Create a block-sparse mask
    mask = create_block_sparsity_mask(N, block_size=16)
    
    # Compute with CUDA implementation
    output_cuda = sparse_attention(q, k, v, mask)
    
    # Check output shape
    assert output_cuda.shape == (N, D)
    
    # Check output values are finite
    assert np.all(np.isfinite(output_cuda)) 