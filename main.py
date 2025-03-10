#!/usr/bin/env python3
"""
Optimized Sparse Attention Kernel for Simulating Cosmic Structures

This script provides a command-line interface to run various components of the project.
"""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(
        description="Optimized Sparse Attention Kernel for Simulating Cosmic Structures",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Simulation command
    sim_parser = subparsers.add_parser("simulate", help="Run cosmic particle simulation")
    sim_parser.add_argument("--num-particles", type=int, default=1024,
                        help="Number of particles in the simulation")
    sim_parser.add_argument("--box-size", type=float, default=100.0,
                        help="Size of the simulation box")
    sim_parser.add_argument("--feature-dim", type=int, default=64,
                        help="Dimension of particle features")
    sim_parser.add_argument("--interaction-threshold", type=float, default=15.0,
                        help="Distance threshold for particle interactions")
    sim_parser.add_argument("--dt", type=float, default=0.05,
                        help="Time step")
    sim_parser.add_argument("--num-steps", type=int, default=100,
                        help="Number of time steps to simulate")
    sim_parser.add_argument("--periodic", action="store_true",
                        help="Use periodic boundary conditions")
    sim_parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    sim_parser.add_argument("--output-dir", type=str, default="simulation_results",
                        help="Directory to save results")
    sim_parser.add_argument("--no-visualize", action="store_true",
                        help="Disable visualization")
    sim_parser.add_argument("--no-animation", action="store_true",
                        help="Disable animation creation")
    
    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark sparse attention kernel")
    bench_parser.add_argument("--mode", choices=["particles", "sparsity"], default="particles",
                        help="Benchmark mode: vary number of particles or sparsity levels")
    bench_parser.add_argument("--particles", type=int, nargs="+", default=[512, 1024, 2048, 4096],
                        help="Number of particles to benchmark")
    bench_parser.add_argument("--feature-dim", type=int, default=64,
                        help="Feature dimension")
    bench_parser.add_argument("--sparsity", type=float, default=0.95,
                        help="Target sparsity (fraction of zeros in the mask)")
    bench_parser.add_argument("--sparsity-levels", type=float, nargs="+", default=[0.5, 0.75, 0.9, 0.95, 0.99],
                        help="Sparsity levels to benchmark")
    bench_parser.add_argument("--num-runs", type=int, default=5,
                        help="Number of runs for each benchmark")
    bench_parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Directory to save results")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("--verbose", "-v", action="count", default=0,
                        help="Increase verbosity")
    
    # Device info command
    info_parser = subparsers.add_parser("info", help="Print device information")
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Run the selected command
    if args.command == "simulate":
        from src.utils.run_simulation import run_simulation
        run_simulation(
            args.num_particles,
            args.box_size,
            args.feature_dim,
            args.interaction_threshold,
            args.dt,
            args.num_steps,
            args.periodic,
            args.seed,
            args.output_dir,
            not args.no_visualize,
            not args.no_animation
        )
    
    elif args.command == "benchmark":
        from src.utils.benchmark import run_benchmark, benchmark_sparsity_levels
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
    
    elif args.command == "test":
        import pytest
        sys.exit(pytest.main(["-v" * args.verbose, "src/tests"]))
    
    elif args.command == "info":
        try:
            from src.python.sparse_attention import print_device_info
            print_device_info()
        except ImportError:
            print("CUDA extension not available.")

if __name__ == "__main__":
    main() 