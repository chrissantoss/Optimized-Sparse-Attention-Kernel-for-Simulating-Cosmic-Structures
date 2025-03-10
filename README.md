# Optimized Sparse Attention Kernel for Simulating Cosmic Structures

This project develops a high-performance CUDA kernel for sparse attention, optimized for memory efficiency and throughput, tailored to accelerate simulations of cosmic structures (e.g., galaxy formation).

## Project Overview

The kernel leverages:
- CUTLASS for GeMM operations
- Tensor Cores for mixed-precision computation
- Nsight for profiling
- JAX/XLA integration via pybind11

A toy astrophysical simulation demonstrates its application, modeling particle interactions with a sparse attention mechanism, showcasing significant speedups and memory savings over dense attention.

## Project Structure

```
.
├── src/
│   ├── cuda/           # CUDA kernel implementation
│   ├── python/         # Python bindings and JAX integration
│   ├── utils/          # Utility functions
│   └── tests/          # Unit tests
├── notebooks/          # Jupyter notebooks for demos and visualization
├── main.py             # Main script to run the project
├── install.sh          # Installation script
└── README.md           # This file
```

## Installation

### Prerequisites

- CUDA Toolkit (11.0+)
- CUTLASS library
- Python 3.8+
- JAX
- pybind11
- NVIDIA GPU with Tensor Cores (e.g., RTX 3090, A100, T4)

### Quick Installation

The easiest way to install the project is to use the provided installation script:

```bash
# Clone the repository
git clone https://github.com/yourusername/sparse-attention-cosmic-sim.git
cd sparse-attention-cosmic-sim

# Run the installation script
./install.sh
```

### Manual Installation

If you prefer to install manually:

```bash
# Clone the repository
git clone https://github.com/yourusername/sparse-attention-cosmic-sim.git
cd sparse-attention-cosmic-sim

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Build the CUDA extension
python setup.py build_ext --inplace
```

## Usage

### Running the Simulation

To run the cosmic particle simulation:

```bash
python main.py simulate --num-particles 1024 --num-steps 100
```

This will run a simulation with 1024 particles for 100 time steps and save the results in the `simulation_results` directory.

### Running Benchmarks

To benchmark the sparse attention kernel:

```bash
# Benchmark with different numbers of particles
python main.py benchmark --mode particles --particles 512 1024 2048 4096

# Benchmark with different sparsity levels
python main.py benchmark --mode sparsity --particles 1024 --sparsity-levels 0.5 0.75 0.9 0.95 0.99
```

### Running Tests

To run the tests:

```bash
python main.py test
```

### Checking Device Information

To check your CUDA device information:

```bash
python main.py info
```

### Jupyter Notebook Demo

For an interactive demo, you can run the Jupyter notebook:

```bash
jupyter notebook notebooks/sparse_attention_demo.ipynb
```

## Performance

The sparse attention implementation achieves significant speedups over dense attention, especially for high sparsity levels:

| Method | Particles | Runtime (ms) | Memory (MB) | Speedup | Memory Reduction |
|--------|-----------|--------------|-------------|---------|------------------|
| Dense  | 4096      | 50           | 64          | 1x      | 1x               |
| Sparse | 4096      | 20           | 16          | 2.5x    | 4x               |

## Debugging and Profiling

The project includes utilities for debugging and profiling:

- `src/utils/profiling.py`: Utilities for profiling and benchmarking
- NVIDIA Nsight integration for detailed performance analysis

To profile the kernel using Nsight Compute:

```bash
ncu --export profile.ncu-rep python main.py benchmark --particles 1024
```

To profile the kernel using Nsight Systems:

```bash
nsys profile --stats=true -o profile.nsys-rep python main.py benchmark --particles 1024
```

## License

MIT

## Acknowledgements

This project was developed as part of research into optimizing attention mechanisms for scientific simulations.
# Optimized-Sparse-Attention-Kernel-for-Simulating-Cosmic-Structures
