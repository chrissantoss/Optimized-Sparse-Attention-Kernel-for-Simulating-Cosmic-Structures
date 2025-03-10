import os
import sys
import platform
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Find CUDA and CUTLASS paths
cuda_path = os.environ.get('CUDA_PATH', '/usr/local/cuda')
cutlass_path = os.environ.get('CUTLASS_PATH', os.path.expanduser('~/cutlass'))

# Check if CUDA is available
if not os.path.exists(cuda_path):
    print(f"CUDA not found at {cuda_path}. Please set CUDA_PATH environment variable.")
    sys.exit(1)

# Check if CUTLASS is available
if not os.path.exists(cutlass_path):
    print(f"CUTLASS not found at {cutlass_path}. Please set CUTLASS_PATH environment variable.")
    sys.exit(1)

# Detect CUDA architecture
def get_cuda_arch():
    # Default to Volta (sm_70) if detection fails
    default_arch = "70"
    
    try:
        import subprocess
        result = subprocess.run([os.path.join(cuda_path, 'bin', 'nvcc'), '--version'], 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Try to detect GPU architecture using deviceQuery
        device_query_path = os.path.join(cuda_path, 'samples', 'bin', 'x86_64', 'linux', 'release', 'deviceQuery')
        if os.path.exists(device_query_path):
            result = subprocess.run([device_query_path], 
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            for line in result.stdout.split('\n'):
                if 'Compute Capability' in line:
                    parts = line.split(':')[1].strip().split('.')
                    return parts[0] + parts[1]
        
        return default_arch
    except Exception as e:
        print(f"Failed to detect CUDA architecture: {e}")
        return default_arch

cuda_arch = get_cuda_arch()

# Custom build extension for CUDA
class CUDAExtension(Extension):
    def __init__(self, name, sources, *args, **kwargs):
        super().__init__(name, sources, *args, **kwargs)

class BuildExt(build_ext):
    def build_extensions(self):
        # Customize compiler flags
        for ext in self.extensions:
            if isinstance(ext, CUDAExtension):
                self.build_cuda_extension(ext)
            else:
                super().build_extensions()
    
    def build_cuda_extension(self, ext):
        # Build CUDA extension using nvcc
        src_dir = os.path.dirname(self.get_ext_fullpath(ext.name))
        os.makedirs(src_dir, exist_ok=True)
        
        # Build command
        build_cmd = [
            'nvcc',
            '-shared',
            '-Xcompiler', '-fPIC',
            f'-arch=sm_{cuda_arch}',
            '-std=c++14',
            '-O3',
            '-I', os.path.join(cuda_path, 'include'),
            '-I', os.path.join(cutlass_path, 'include'),
            '-I', os.path.join(cutlass_path, 'tools', 'util', 'include'),
            '-L', os.path.join(cuda_path, 'lib64'),
            '-lcudart', '-lcublas',
        ]
        
        # Add pybind11 include path
        import pybind11
        build_cmd.extend(['-I', pybind11.get_include()])
        
        # Add source files
        build_cmd.extend(ext.sources)
        
        # Output file
        output_file = self.get_ext_fullpath(ext.name)
        build_cmd.extend(['-o', output_file])
        
        # Run build command
        self.announce(f'Building CUDA extension: {" ".join(build_cmd)}', level=3)
        import subprocess
        subprocess.check_call(build_cmd)

# Define extension
sparse_attention_ext = CUDAExtension(
    'src.python.sparse_attention_cuda',
    ['src/python/sparse_attention_bindings.cpp', 'src/cuda/sparse_attention.cu'],
)

# Setup
setup(
    name='sparse_attention',
    version='0.1.0',
    description='Optimized Sparse Attention Kernel for Simulating Cosmic Structures',
    author='Your Name',
    author_email='your.email@example.com',
    packages=['src', 'src.python', 'src.utils', 'src.tests'],
    ext_modules=[sparse_attention_ext],
    cmdclass={'build_ext': BuildExt},
    install_requires=[
        'numpy>=1.20.0',
        'jax>=0.3.4',
        'jaxlib>=0.3.4',
        'pybind11>=2.9.0',
        'matplotlib>=3.5.0',
        'pytest>=7.0.0',
    ],
    python_requires='>=3.8',
) 