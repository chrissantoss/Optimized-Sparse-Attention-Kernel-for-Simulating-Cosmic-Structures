# Notebooks

This directory contains Jupyter notebooks for demonstrating and visualizing the sparse attention kernel.

## Converting Python Script to Jupyter Notebook

The `sparse_attention_demo.py` file is a Python script formatted to be easily converted to a Jupyter notebook. To convert it, you can use the `p2j` (py2jupyter) tool:

```bash
# Install p2j if you don't have it
pip install p2j

# Convert the Python script to a Jupyter notebook
p2j sparse_attention_demo.py
```

Alternatively, you can use Jupyter's built-in conversion tool:

```bash
# Install jupyter if you don't have it
pip install jupyter

# Convert the Python script to a Jupyter notebook
jupyter nbconvert --to notebook --execute sparse_attention_demo.py
```

## Running the Notebook

Once you have the notebook, you can run it using Jupyter:

```bash
jupyter notebook sparse_attention_demo.ipynb
```

## Notebook Contents

The `sparse_attention_demo` notebook demonstrates:

1. **CUDA Device Information**: Checking if CUDA is available and getting information about the GPU.
2. **Sparsity Masks**: Creating and visualizing different types of sparsity masks.
3. **Benchmarking**: Comparing the performance of sparse vs. dense attention.
4. **Cosmic Simulation**: Running a toy simulation of cosmic particles using sparse attention.
5. **Scaling Analysis**: Testing how the sparse attention kernel scales with the number of particles.
6. **Memory Usage**: Analyzing the memory usage of sparse vs. dense attention.

## Troubleshooting

If you encounter issues with the notebook:

1. Make sure you have all the required dependencies installed (see `requirements.txt` in the root directory).
2. Ensure that CUDA and CUTLASS are properly installed if you want to use the CUDA implementation.
3. If CUDA is not available, the code will fall back to a NumPy implementation, which will be slower but still functional. 