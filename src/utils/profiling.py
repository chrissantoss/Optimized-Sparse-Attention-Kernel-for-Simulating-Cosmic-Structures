import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
import subprocess
import os
import json

def time_function(func, *args, **kwargs):
    """
    Time a function call.
    
    Args:
        func: Function to time
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Tuple of (result, elapsed_time)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed_time = time.time() - start_time
    return result, elapsed_time

def benchmark_function(func, num_runs=10, *args, **kwargs):
    """
    Benchmark a function over multiple runs.
    
    Args:
        func: Function to benchmark
        num_runs: Number of runs
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Dictionary with benchmark results
    """
    times = []
    for _ in range(num_runs):
        _, elapsed_time = time_function(func, *args, **kwargs)
        times.append(elapsed_time)
    
    return {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
        "times": times
    }

def plot_benchmark_comparison(benchmarks, title="Benchmark Comparison"):
    """
    Plot a comparison of benchmark results.
    
    Args:
        benchmarks: Dictionary mapping names to benchmark results
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(benchmarks.keys())
    means = [benchmarks[name]["mean_time"] * 1000 for name in names]  # Convert to ms
    stds = [benchmarks[name]["std_time"] * 1000 for name in names]    # Convert to ms
    
    # Bar plot
    bars = ax.bar(names, means, yerr=stds, capsize=10, alpha=0.7)
    
    # Add values on top of bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{mean:.2f} ms', ha='center', va='bottom')
    
    ax.set_ylabel('Time (ms)')
    ax.set_title(title)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    return fig

def memory_usage(obj):
    """
    Estimate memory usage of a Python object.
    
    Args:
        obj: Python object
        
    Returns:
        Memory usage in bytes
    """
    import sys
    
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    elif isinstance(obj, (list, tuple)):
        return sum(memory_usage(x) for x in obj)
    elif isinstance(obj, dict):
        return sum(memory_usage(k) + memory_usage(v) for k, v in obj.items())
    else:
        return sys.getsizeof(obj)

def run_nsight_compute(cuda_program, args=None, output_file=None):
    """
    Run NVIDIA Nsight Compute profiler on a CUDA program.
    
    Args:
        cuda_program: Path to the CUDA program
        args: Arguments to pass to the program
        output_file: Output file for profiling results
        
    Returns:
        Profiling results as a string
    """
    if args is None:
        args = []
    
    if output_file is None:
        output_file = "nsight_compute_profile.ncu-rep"
    
    cmd = ["ncu", "--export", output_file, cuda_program] + args
    
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.decode()
    except subprocess.CalledProcessError as e:
        print(f"Error running Nsight Compute: {e}")
        print(f"stderr: {e.stderr.decode()}")
        return None

def run_nsight_systems(cuda_program, args=None, output_file=None):
    """
    Run NVIDIA Nsight Systems profiler on a CUDA program.
    
    Args:
        cuda_program: Path to the CUDA program
        args: Arguments to pass to the program
        output_file: Output file for profiling results
        
    Returns:
        Profiling results as a string
    """
    if args is None:
        args = []
    
    if output_file is None:
        output_file = "nsight_systems_profile.nsys-rep"
    
    cmd = ["nsys", "profile", "--stats=true", "-o", output_file, cuda_program] + args
    
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.decode()
    except subprocess.CalledProcessError as e:
        print(f"Error running Nsight Systems: {e}")
        print(f"stderr: {e.stderr.decode()}")
        return None

def parse_nsight_compute_metrics(ncu_output):
    """
    Parse metrics from Nsight Compute output.
    
    Args:
        ncu_output: Output from Nsight Compute
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Example parsing logic (adjust based on actual output format)
    lines = ncu_output.split('\n')
    for line in lines:
        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                try:
                    # Try to convert to number if possible
                    value = float(value)
                except ValueError:
                    pass
                metrics[key] = value
    
    return metrics

def visualize_kernel_performance(metrics, title="Kernel Performance"):
    """
    Visualize kernel performance metrics.
    
    Args:
        metrics: Dictionary of performance metrics
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Extract relevant metrics (adjust based on what's available)
    relevant_metrics = {
        "SM Occupancy": metrics.get("SM Occupancy (%)", 0),
        "Memory Throughput": metrics.get("Memory Throughput (%)", 0),
        "Compute Throughput": metrics.get("Compute Throughput (%)", 0),
        "Tensor Core Usage": metrics.get("Tensor Core Usage (%)", 0)
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(relevant_metrics.keys())
    values = list(relevant_metrics.values())
    
    # Bar plot
    bars = ax.bar(names, values, alpha=0.7)
    
    # Add values on top of bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}%', ha='center', va='bottom')
    
    ax.set_ylabel('Percentage (%)')
    ax.set_title(title)
    ax.set_ylim(0, 105)  # Leave room for text
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    return fig

def debug_csr_matrix(row_ptrs, col_indices, values, N):
    """
    Debug a CSR matrix.
    
    Args:
        row_ptrs: Row pointers
        col_indices: Column indices
        values: Values
        N: Matrix dimension
        
    Returns:
        Dictionary with debug information
    """
    # Convert to numpy arrays if not already
    row_ptrs = np.array(row_ptrs)
    col_indices = np.array(col_indices)
    values = np.array(values)
    
    # Basic checks
    assert len(row_ptrs) == N + 1, f"Row pointers length should be {N+1}, got {len(row_ptrs)}"
    assert row_ptrs[0] == 0, f"First row pointer should be 0, got {row_ptrs[0]}"
    assert row_ptrs[-1] == len(col_indices), f"Last row pointer should be {len(col_indices)}, got {row_ptrs[-1]}"
    assert len(col_indices) == len(values), f"Column indices and values should have same length"
    
    # Compute statistics
    nnz = len(col_indices)
    sparsity = 1.0 - (nnz / (N * N))
    
    row_nnz = np.diff(row_ptrs)
    min_nnz_per_row = np.min(row_nnz)
    max_nnz_per_row = np.max(row_nnz)
    avg_nnz_per_row = np.mean(row_nnz)
    
    # Check for invalid column indices
    invalid_cols = np.any((col_indices < 0) | (col_indices >= N))
    
    # Reconstruct dense matrix for visualization
    dense = np.zeros((N, N))
    for i in range(N):
        for j in range(row_ptrs[i], row_ptrs[i+1]):
            col = col_indices[j]
            if 0 <= col < N:
                dense[i, col] = values[j]
    
    return {
        "N": N,
        "nnz": nnz,
        "sparsity": sparsity,
        "min_nnz_per_row": min_nnz_per_row,
        "max_nnz_per_row": max_nnz_per_row,
        "avg_nnz_per_row": avg_nnz_per_row,
        "invalid_columns": invalid_cols,
        "dense_matrix": dense
    }

def visualize_sparsity_pattern(csr_info, max_size=1000):
    """
    Visualize the sparsity pattern of a matrix.
    
    Args:
        csr_info: CSR matrix debug information
        max_size: Maximum size to visualize
        
    Returns:
        Matplotlib figure
    """
    dense = csr_info["dense_matrix"]
    N = dense.shape[0]
    
    # If matrix is too large, sample it
    if N > max_size:
        step = N // max_size
        dense = dense[::step, ::step]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot sparsity pattern (non-zero elements)
    ax.spy(dense, markersize=0.5, aspect='equal')
    
    ax.set_title(f"Sparsity Pattern (N={N}, NNZ={csr_info['nnz']}, Sparsity={csr_info['sparsity']:.2%})")
    
    return fig 