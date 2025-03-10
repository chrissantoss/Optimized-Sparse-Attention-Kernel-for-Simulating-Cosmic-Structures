import numpy as np
import time
import argparse
import torch
import matplotlib.pyplot as plt
import os
import json
from typing import Dict, List, Tuple, Any
from pathlib import Path

from ..python.sparse_attention import (
    sparse_attention,
    create_sparsity_mask_from_positions,
    create_block_sparsity_mask,
    benchmark_sparse_vs_dense,
    print_device_info
)

# Add a custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def run_benchmark(
    particle_counts: List[int],
    feature_dim: int,
    sparsity: float,
    num_runs: int = 5,
    output_dir: str = "benchmark_results"
):
    """
    Run benchmarks for different numbers of particles.
    
    Args:
        particle_counts: List of particle counts to benchmark
        feature_dim: Feature dimension
        sparsity: Target sparsity (fraction of zeros in the mask)
        num_runs: Number of runs for each benchmark
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Print CUDA device info
    print("CUDA Device Information:")
    try:
        device_info = print_device_info()
        print()
    except:
        print("CUDA not available\n")
    
    results = {}
    
    for N in particle_counts:
        print(f"Benchmarking with {N} particles...")
        
        # Run benchmark
        result = benchmark_sparse_vs_dense(N, feature_dim, sparsity, num_runs)
        
        # Print results
        print(f"  Sparsity: {result['sparsity']:.2%}")
        print(f"  Non-zero elements: {result['nnz']}")
        print(f"  Sparse time: {result['sparse_time_ms']:.2f} ms")
        print(f"  Dense time: {result['dense_time_ms']:.2f} ms")
        print(f"  Speedup: {result['speedup']:.2f}x")
        print(f"  Sparse memory: {result['sparse_memory_mb']:.2f} MB")
        print(f"  Dense memory: {result['dense_memory_mb']:.2f} MB")
        print(f"  Memory reduction: {result['memory_reduction']:.2f}x")
        print()
        
        # Store results
        results[N] = result
    
    # Save results to JSON
    results_file = os.path.join(output_dir, f"benchmark_N{min(particle_counts)}-{max(particle_counts)}_D{feature_dim}_S{sparsity:.2f}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    # Plot results
    plot_benchmark_results(results, output_dir)
    
    return results

def plot_benchmark_results(results: Dict[int, Dict[str, Any]], output_dir: str):
    """
    Plot benchmark results.
    
    Args:
        results: Dictionary mapping particle counts to benchmark results
        output_dir: Directory to save plots
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    particle_counts = sorted(list(map(int, results.keys())))
    
    # Extract data
    sparse_times = [results[N]["sparse_time_ms"] for N in particle_counts]
    dense_times = [results[N]["dense_time_ms"] for N in particle_counts]
    speedups = [results[N]["speedup"] for N in particle_counts]
    sparse_memory = [results[N]["sparse_memory_mb"] for N in particle_counts]
    dense_memory = [results[N]["dense_memory_mb"] for N in particle_counts]
    memory_reductions = [results[N]["memory_reduction"] for N in particle_counts]
    
    # Plot runtime
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(particle_counts, sparse_times, 'o-', linewidth=2, label='Sparse Attention')
    ax.plot(particle_counts, dense_times, 'o-', linewidth=2, label='Dense Attention')
    ax.set_xlabel('Number of Particles')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Attention Time vs. Number of Particles')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "runtime.png"), dpi=300)
    plt.close()
    
    # Plot speedup
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(particle_counts, speedups, 'o-', linewidth=2)
    ax.set_xlabel('Number of Particles')
    ax.set_ylabel('Speedup (x)')
    ax.set_title('Speedup vs. Number of Particles')
    ax.set_xscale('log', base=2)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "speedup.png"), dpi=300)
    plt.close()
    
    # Plot memory usage
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(particle_counts, sparse_memory, 'o-', linewidth=2, label='Sparse Attention')
    ax.plot(particle_counts, dense_memory, 'o-', linewidth=2, label='Dense Attention')
    ax.set_xlabel('Number of Particles')
    ax.set_ylabel('Memory Usage (MB)')
    ax.set_title('Memory Usage vs. Number of Particles')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "memory.png"), dpi=300)
    plt.close()
    
    # Plot memory reduction
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(particle_counts, memory_reductions, 'o-', linewidth=2)
    ax.set_xlabel('Number of Particles')
    ax.set_ylabel('Memory Reduction (x)')
    ax.set_title('Memory Reduction vs. Number of Particles')
    ax.set_xscale('log', base=2)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "memory_reduction.png"), dpi=300)
    plt.close()

def benchmark_sparsity_levels(
    N: int,
    feature_dim: int,
    sparsity_levels: List[float],
    num_runs: int = 5,
    output_dir: str = "benchmark_results"
):
    """
    Run benchmarks for different sparsity levels.
    
    Args:
        N: Number of particles
        feature_dim: Feature dimension
        sparsity_levels: List of sparsity levels to benchmark
        num_runs: Number of runs for each benchmark
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    for sparsity in sparsity_levels:
        print(f"Benchmarking with sparsity {sparsity:.2%}...")
        
        # Run benchmark
        result = benchmark_sparse_vs_dense(N, feature_dim, sparsity, num_runs)
        
        # Print results
        print(f"  Non-zero elements: {result['nnz']}")
        print(f"  Sparse time: {result['sparse_time_ms']:.2f} ms")
        print(f"  Dense time: {result['dense_time_ms']:.2f} ms")
        print(f"  Speedup: {result['speedup']:.2f}x")
        print(f"  Memory reduction: {result['memory_reduction']:.2f}x")
        print()
        
        # Store results
        results[sparsity] = result
    
    # Save results to JSON
    results_file = os.path.join(output_dir, f"benchmark_sparsity_N{N}_D{feature_dim}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    # Plot results
    plot_sparsity_results(results, output_dir)
    
    return results

def plot_sparsity_results(results: Dict[float, Dict[str, Any]], output_dir: str):
    """
    Plot benchmark results for different sparsity levels.
    
    Args:
        results: Dictionary mapping sparsity levels to benchmark results
        output_dir: Directory to save plots
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    sparsity_levels = sorted(list(map(float, results.keys())))
    
    # Extract data
    sparse_times = [results[s]["sparse_time_ms"] for s in sparsity_levels]
    dense_times = [results[s]["dense_time_ms"] for s in sparsity_levels]
    speedups = [results[s]["speedup"] for s in sparsity_levels]
    memory_reductions = [results[s]["memory_reduction"] for s in sparsity_levels]
    
    # Plot runtime
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sparsity_levels, sparse_times, 'o-', linewidth=2, label='Sparse Attention')
    ax.plot(sparsity_levels, dense_times, 'o-', linewidth=2, label='Dense Attention')
    ax.set_xlabel('Sparsity')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Attention Time vs. Sparsity')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "runtime_vs_sparsity.png"), dpi=300)
    plt.close()
    
    # Plot speedup
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sparsity_levels, speedups, 'o-', linewidth=2)
    ax.set_xlabel('Sparsity')
    ax.set_ylabel('Speedup (x)')
    ax.set_title('Speedup vs. Sparsity')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "speedup_vs_sparsity.png"), dpi=300)
    plt.close()
    
    # Plot memory reduction
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sparsity_levels, memory_reductions, 'o-', linewidth=2)
    ax.set_xlabel('Sparsity')
    ax.set_ylabel('Memory Reduction (x)')
    ax.set_title('Memory Reduction vs. Sparsity')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "memory_reduction_vs_sparsity.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark sparse attention kernel")
    parser.add_argument("--mode", choices=["particles", "sparsity"], default="particles",
                        help="Benchmark mode: vary number of particles or sparsity levels")
    parser.add_argument("--particles", type=int, nargs="+", default=[512, 1024, 2048, 4096],
                        help="Number of particles to benchmark")
    parser.add_argument("--feature-dim", type=int, default=64,
                        help="Feature dimension")
    parser.add_argument("--sparsity", type=float, default=0.95,
                        help="Target sparsity (fraction of zeros in the mask)")
    parser.add_argument("--sparsity-levels", type=float, nargs="+", default=[0.5, 0.75, 0.9, 0.95, 0.99],
                        help="Sparsity levels to benchmark")
    parser.add_argument("--num-runs", type=int, default=5,
                        help="Number of runs for each benchmark")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    if args.mode == "particles":
        run_benchmark(
            args.particles,
            args.feature_dim,
            args.sparsity,
            args.num_runs,
            args.output_dir
        )
    else:
        benchmark_sparsity_levels(
            args.particles[0],
            args.feature_dim,
            args.sparsity_levels,
            args.num_runs,
            args.output_dir
        ) 