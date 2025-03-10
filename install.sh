#!/bin/bash
# Installation script for Optimized Sparse Attention Kernel

set -e  # Exit on error

# Print header
echo "====================================================="
echo "Installing Optimized Sparse Attention Kernel"
echo "====================================================="

# Check if CUDA is available
if command -v nvcc &> /dev/null; then
    echo "CUDA found: $(nvcc --version | head -n1)"
    CUDA_PATH=$(which nvcc | rev | cut -d'/' -f3- | rev)
    echo "CUDA_PATH: $CUDA_PATH"
else
    echo "Warning: CUDA not found. The sparse attention kernel will use the fallback implementation."
    echo "To use the CUDA implementation, please install CUDA Toolkit 11.0 or later."
fi

# Check if CUTLASS is available
if [ -d "$HOME/cutlass" ]; then
    echo "CUTLASS found at $HOME/cutlass"
    CUTLASS_PATH="$HOME/cutlass"
else
    echo "CUTLASS not found. Downloading CUTLASS..."
    git clone https://github.com/NVIDIA/cutlass.git "$HOME/cutlass"
    CUTLASS_PATH="$HOME/cutlass"
    echo "CUTLASS downloaded to $CUTLASS_PATH"
fi

# Export environment variables
export CUDA_PATH=$CUDA_PATH
export CUTLASS_PATH=$CUTLASS_PATH

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Build the CUDA extension
echo "Building CUDA extension..."
python setup.py build_ext --inplace

# Run tests
echo "Running tests..."
python -m pytest -xvs src/tests

echo "====================================================="
echo "Installation complete!"
echo "====================================================="
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the simulation, use:"
echo "  python main.py simulate"
echo ""
echo "To run benchmarks, use:"
echo "  python main.py benchmark"
echo ""
echo "For more information, run:"
echo "  python main.py --help"
echo "=====================================================" 