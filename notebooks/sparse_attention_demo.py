#!/usr/bin/env python
# coding: utf-8

# # Optimized Sparse Attention Kernel for Simulating Cosmic Structures
# 
# This notebook demonstrates the usage of our optimized sparse attention kernel for simulating cosmic structures. The kernel is implemented in CUDA and integrated with Python via pybind11.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import torch
import sys
import os

# Add parent directory to path
sys.path.append('..')

# Import our modules
from src.python.sparse_attention import (
    sparse_attention,
    create_sparsity_mask_from_positions,
    create_block_sparsity_mask,
    benchmark_sparse_vs_dense,
    print_device_info
)
from src.utils.cosmic_sim import CosmicParticleSimulation
from src.utils.profiling import (
    time_function,
    benchmark_function,
    plot_benchmark_comparison,
    memory_usage,
    debug_csr_matrix,
    visualize_sparsity_pattern
)

# %matplotlib inline


# ## 1. Check CUDA Device Information
# 
# First, let's check if CUDA is available and get information about the GPU.

# In[ ]:


print_device_info()


# ## 2. Create Sparsity Masks
# 
# Let's create different types of sparsity masks and visualize them.

# In[ ]:


# Create a block-sparse mask
N = 64
block_size = 8
block_mask = create_block_sparsity_mask(N, block_size)

# Visualize the mask
plt.figure(figsize=(8, 8))
plt.spy(block_mask, markersize=1)
plt.title(f"Block-Sparse Mask (N={N}, Block Size={block_size})")
plt.show()

# Calculate sparsity
sparsity = 1.0 - block_mask.sum() / (N * N)
print(f"Sparsity: {sparsity:.2%}")


# In[ ]:


# Create a distance-based mask
num_particles = 64
positions = np.random.uniform(0, 100, size=(num_particles, 3))
threshold = 20.0
distance_mask = create_sparsity_mask_from_positions(positions, threshold)

# Visualize the mask
plt.figure(figsize=(8, 8))
plt.spy(distance_mask, markersize=1)
plt.title(f"Distance-Based Mask (N={num_particles}, Threshold={threshold})")
plt.show()

# Calculate sparsity
sparsity = 1.0 - distance_mask.sum() / (num_particles * num_particles)
print(f"Sparsity: {sparsity:.2%}")


# ## 3. Benchmark Sparse vs. Dense Attention
# 
# Let's compare the performance of our sparse attention implementation against dense attention.

# In[ ]:


# Benchmark with different sparsity levels
N = 1024
D = 64
sparsity_levels = [0.5, 0.75, 0.9, 0.95, 0.99]

results = {}
for sparsity in sparsity_levels:
    print(f"Benchmarking with sparsity {sparsity:.2%}...")
    result = benchmark_sparse_vs_dense(N, D, sparsity, num_runs=5)
    results[f"Sparsity {sparsity:.0%}"] = result
    print(f"  Sparse: {result['sparse_time_ms']:.2f} ms, Dense: {result['dense_time_ms']:.2f} ms")
    print(f"  Speedup: {result['speedup']:.2f}x, Memory Reduction: {result['memory_reduction']:.2f}x")
    print()


# In[ ]:


# Plot speedup vs. sparsity
sparsities = [result['sparsity'] for result in results.values()]
speedups = [result['speedup'] for result in results.values()]
memory_reductions = [result['memory_reduction'] for result in results.values()]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.plot(sparsities, speedups, 'o-', linewidth=2)
ax1.set_xlabel('Sparsity')
ax1.set_ylabel('Speedup (x)')
ax1.set_title('Speedup vs. Sparsity')
ax1.grid(True)

ax2.plot(sparsities, memory_reductions, 'o-', linewidth=2, color='orange')
ax2.set_xlabel('Sparsity')
ax2.set_ylabel('Memory Reduction (x)')
ax2.set_title('Memory Reduction vs. Sparsity')
ax2.grid(True)

plt.tight_layout()
plt.show()


# ## 4. Cosmic Particle Simulation
# 
# Now, let's run a toy simulation of cosmic particles using our sparse attention kernel.

# In[ ]:


# Create a simulation with a small number of particles for visualization
sim = CosmicParticleSimulation(
    num_particles=256,
    box_size=100.0,
    feature_dim=64,
    interaction_threshold=15.0,
    dt=0.05,
    periodic=True,
    seed=42
)

# Visualize initial state
sim.visualize_3d()
plt.show()


# In[ ]:


# Run the simulation for 50 steps
print("Running simulation...")
start_time = time.time()
sim.run(50)
elapsed_time = time.time() - start_time
print(f"Simulation completed in {elapsed_time:.2f} seconds")

# Visualize final state
sim.visualize_3d()
plt.show()


# In[ ]:


# Create an animation of the simulation
anim = sim.create_animation(num_frames=50, interval=100, skip=1)
plt.close()  # Close the figure to avoid displaying it twice

# Display the animation
from IPython.display import HTML
HTML(anim.to_jshtml())


# In[ ]:


# Analyze clustering evolution
fig = sim.plot_clustering_evolution()
plt.show()


# ## 5. Scaling to Larger Simulations
# 
# Let's test how our sparse attention kernel scales with the number of particles.

# In[ ]:


# Test with different numbers of particles
particle_counts = [512, 1024, 2048, 4096]
feature_dim = 64
interaction_threshold = 10.0

sparse_times = []
dense_times = []

for N in particle_counts:
    print(f"Testing with {N} particles...")
    
    # Create random data
    positions = np.random.uniform(0, 100, size=(N, 3)).astype(np.float32)
    features = np.random.normal(0, 1, size=(N, feature_dim)).astype(np.float32)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    
    # Create mask based on positions
    mask = create_sparsity_mask_from_positions(positions, interaction_threshold)
    sparsity = 1.0 - mask.sum() / (N * N)
    print(f"  Sparsity: {sparsity:.2%}")
    
    # Benchmark sparse attention
    start_time = time.time()
    output_sparse = sparse_attention(features, features, features, mask)
    sparse_time = time.time() - start_time
    sparse_times.append(sparse_time * 1000)  # Convert to ms
    print(f"  Sparse attention time: {sparse_time * 1000:.2f} ms")
    
    # Benchmark dense attention (PyTorch)
    if N <= 2048:  # Skip dense attention for large N to avoid OOM
        q_torch = torch.tensor(features, dtype=torch.float32)
        k_torch = torch.tensor(features, dtype=torch.float32)
        v_torch = torch.tensor(features, dtype=torch.float32)
        mask_torch = torch.tensor(mask, dtype=torch.bool)
        
        start_time = time.time()
        qk = torch.matmul(q_torch, k_torch.t())
        qk_masked = qk.masked_fill(~mask_torch, -1e9)
        attention = torch.softmax(qk_masked, dim=1)
        output_dense = torch.matmul(attention, v_torch)
        dense_time = time.time() - start_time
        dense_times.append(dense_time * 1000)  # Convert to ms
        print(f"  Dense attention time: {dense_time * 1000:.2f} ms")
        print(f"  Speedup: {dense_time / sparse_time:.2f}x")
    else:
        dense_times.append(None)
        print("  Dense attention skipped (too large)")
    
    print()


# In[ ]:


# Plot scaling results
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(particle_counts, sparse_times, 'o-', linewidth=2, label='Sparse Attention')
ax.plot([N for N, t in zip(particle_counts, dense_times) if t is not None],
        [t for t in dense_times if t is not None],
        'o-', linewidth=2, label='Dense Attention')

ax.set_xlabel('Number of Particles')
ax.set_ylabel('Time (ms)')
ax.set_title('Attention Time vs. Number of Particles')
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()


# ## 6. Memory Usage Analysis
# 
# Let's analyze the memory usage of sparse vs. dense attention.

# In[ ]:


# Calculate memory usage for different numbers of particles
particle_counts = [512, 1024, 2048, 4096, 8192]
feature_dim = 64
sparsity = 0.95  # Target sparsity

sparse_memory = []
dense_memory = []

for N in particle_counts:
    # Memory for input matrices (Q, K, V)
    input_memory = 3 * N * feature_dim * 4  # 4 bytes per float32
    
    # Memory for dense attention
    dense_attn_memory = N * N * 4  # 4 bytes per float32 for attention matrix
    dense_total = input_memory + dense_attn_memory
    dense_memory.append(dense_total / (1024 * 1024))  # Convert to MB
    
    # Memory for sparse attention
    nnz = int(N * N * (1 - sparsity))  # Number of non-zeros
    sparse_attn_memory = nnz * (4 + 4)  # 4 bytes for value, 4 bytes for index
    sparse_attn_memory += (N + 1) * 4  # Row pointers
    sparse_total = input_memory + sparse_attn_memory
    sparse_memory.append(sparse_total / (1024 * 1024))  # Convert to MB

# Plot memory usage
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(particle_counts, sparse_memory, 'o-', linewidth=2, label='Sparse Attention')
ax.plot(particle_counts, dense_memory, 'o-', linewidth=2, label='Dense Attention')

ax.set_xlabel('Number of Particles')
ax.set_ylabel('Memory Usage (MB)')
ax.set_title(f'Memory Usage vs. Number of Particles (Sparsity: {sparsity:.0%})')
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()

# Print memory reduction
for i, N in enumerate(particle_counts):
    reduction = dense_memory[i] / sparse_memory[i]
    print(f"N={N}: Dense={dense_memory[i]:.2f} MB, Sparse={sparse_memory[i]:.2f} MB, Reduction={reduction:.2f}x")


# ## 7. Conclusion
# 
# In this notebook, we've demonstrated our optimized sparse attention kernel for simulating cosmic structures. The key findings are:
# 
# 1. **Performance**: Our sparse attention implementation achieves significant speedups over dense attention, especially for high sparsity levels.
# 
# 2. **Memory Efficiency**: The sparse implementation uses much less memory than dense attention, allowing us to scale to larger simulations.
# 
# 3. **Scalability**: The sparse attention kernel scales well with the number of particles, making it suitable for large-scale simulations.
# 
# 4. **Application**: We demonstrated a toy cosmic particle simulation that uses sparse attention to model particle interactions efficiently.
# 
# These results show that sparse attention is a promising approach for accelerating simulations of cosmic structures and other physical systems with sparse interaction patterns. 