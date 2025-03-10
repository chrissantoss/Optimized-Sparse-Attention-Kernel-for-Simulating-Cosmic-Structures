import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import os
import time
from typing import Optional

from ..utils.cosmic_sim import CosmicParticleSimulation

def run_simulation(
    num_particles: int,
    box_size: float,
    feature_dim: int,
    interaction_threshold: float,
    dt: float,
    num_steps: int,
    periodic: bool,
    seed: Optional[int],
    output_dir: str,
    visualize: bool = True,
    create_animation: bool = True,
    animation_frames: int = 100,
    animation_interval: int = 50,
    animation_skip: int = 1
):
    """
    Run a cosmic particle simulation.
    
    Args:
        num_particles: Number of particles in the simulation
        box_size: Size of the simulation box
        feature_dim: Dimension of particle features
        interaction_threshold: Distance threshold for particle interactions
        dt: Time step
        num_steps: Number of time steps to simulate
        periodic: Whether to use periodic boundary conditions
        seed: Random seed
        output_dir: Directory to save results
        visualize: Whether to visualize the simulation
        create_animation: Whether to create an animation
        animation_frames: Number of frames in the animation
        animation_interval: Interval between frames (ms)
        animation_skip: Number of steps to skip between frames
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create simulation
    print(f"Creating simulation with {num_particles} particles...")
    sim = CosmicParticleSimulation(
        num_particles=num_particles,
        box_size=box_size,
        feature_dim=feature_dim,
        interaction_threshold=interaction_threshold,
        dt=dt,
        periodic=periodic,
        seed=seed
    )
    
    # Visualize initial state
    if visualize:
        print("Visualizing initial state...")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        sim.visualize_3d(ax=ax)
        plt.savefig(os.path.join(output_dir, "initial_state.png"), dpi=300)
        plt.close()
    
    # Run simulation
    print(f"Running simulation for {num_steps} steps...")
    start_time = time.time()
    sim.run(num_steps)
    elapsed_time = time.time() - start_time
    print(f"Simulation completed in {elapsed_time:.2f} seconds")
    print(f"Average time per step: {elapsed_time / num_steps * 1000:.2f} ms")
    
    # Visualize final state
    if visualize:
        print("Visualizing final state...")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        sim.visualize_3d(ax=ax)
        plt.savefig(os.path.join(output_dir, "final_state.png"), dpi=300)
        plt.close()
    
    # Create animation
    if create_animation:
        print("Creating animation...")
        anim = sim.create_animation(
            num_frames=animation_frames,
            interval=animation_interval,
            skip=animation_skip,
            filename=os.path.join(output_dir, "animation.gif")
        )
    
    # Analyze clustering
    print("Analyzing clustering...")
    fig = sim.plot_clustering_evolution()
    plt.savefig(os.path.join(output_dir, "clustering_evolution.png"), dpi=300)
    plt.close()
    
    # Save simulation parameters
    with open(os.path.join(output_dir, "simulation_params.txt"), 'w') as f:
        f.write(f"Number of particles: {num_particles}\n")
        f.write(f"Box size: {box_size}\n")
        f.write(f"Feature dimension: {feature_dim}\n")
        f.write(f"Interaction threshold: {interaction_threshold}\n")
        f.write(f"Time step: {dt}\n")
        f.write(f"Number of steps: {num_steps}\n")
        f.write(f"Periodic boundary conditions: {periodic}\n")
        f.write(f"Random seed: {seed}\n")
        f.write(f"Simulation time: {elapsed_time:.2f} seconds\n")
        f.write(f"Average time per step: {elapsed_time / num_steps * 1000:.2f} ms\n")
    
    return sim

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cosmic particle simulation")
    parser.add_argument("--num-particles", type=int, default=1024,
                        help="Number of particles in the simulation")
    parser.add_argument("--box-size", type=float, default=100.0,
                        help="Size of the simulation box")
    parser.add_argument("--feature-dim", type=int, default=64,
                        help="Dimension of particle features")
    parser.add_argument("--interaction-threshold", type=float, default=15.0,
                        help="Distance threshold for particle interactions")
    parser.add_argument("--dt", type=float, default=0.05,
                        help="Time step")
    parser.add_argument("--num-steps", type=int, default=100,
                        help="Number of time steps to simulate")
    parser.add_argument("--periodic", action="store_true",
                        help="Use periodic boundary conditions")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--output-dir", type=str, default="simulation_results",
                        help="Directory to save results")
    parser.add_argument("--no-visualize", action="store_true",
                        help="Disable visualization")
    parser.add_argument("--no-animation", action="store_true",
                        help="Disable animation creation")
    parser.add_argument("--animation-frames", type=int, default=100,
                        help="Number of frames in the animation")
    parser.add_argument("--animation-interval", type=int, default=50,
                        help="Interval between frames (ms)")
    parser.add_argument("--animation-skip", type=int, default=1,
                        help="Number of steps to skip between frames")
    
    args = parser.parse_args()
    
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
        not args.no_animation,
        args.animation_frames,
        args.animation_interval,
        args.animation_skip
    ) 