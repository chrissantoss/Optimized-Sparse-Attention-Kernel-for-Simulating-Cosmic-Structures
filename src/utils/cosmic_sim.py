import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Tuple, List, Optional, Callable
from ..python.sparse_attention import sparse_attention, create_sparsity_mask_from_positions

class CosmicParticleSimulation:
    """
    A toy simulation of cosmic particles using sparse attention.
    
    This simulation models a system of particles in 3D space, where each particle
    interacts with nearby particles through a sparse attention mechanism.
    """
    
    def __init__(
        self,
        num_particles: int = 4096,
        box_size: float = 100.0,
        feature_dim: int = 64,
        interaction_threshold: float = 10.0,
        dt: float = 0.01,
        periodic: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize the simulation.
        
        Args:
            num_particles: Number of particles in the simulation
            box_size: Size of the simulation box
            feature_dim: Dimension of particle features
            interaction_threshold: Distance threshold for particle interactions
            dt: Time step
            periodic: Whether to use periodic boundary conditions
            seed: Random seed
        """
        self.num_particles = num_particles
        self.box_size = box_size
        self.feature_dim = feature_dim
        self.interaction_threshold = interaction_threshold
        self.dt = dt
        self.periodic = periodic
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize particle positions and velocities
        self.positions = np.random.uniform(0, box_size, size=(num_particles, 3))
        self.velocities = np.random.normal(0, 1, size=(num_particles, 3))
        
        # Initialize particle features
        self.features = np.random.normal(0, 1, size=(num_particles, feature_dim))
        
        # Normalize features
        self.features = self.features / np.linalg.norm(self.features, axis=1, keepdims=True)
        
        # Initialize history for visualization
        self.position_history = [self.positions.copy()]
        self.time = 0.0
    
    def update_sparsity_mask(self) -> np.ndarray:
        """
        Update the sparsity mask based on current particle positions.
        
        Returns:
            Binary mask of shape (N, N)
        """
        return create_sparsity_mask_from_positions(
            self.positions,
            self.interaction_threshold,
            self.periodic,
            self.box_size
        )
    
    def compute_interaction(self) -> np.ndarray:
        """
        Compute particle interactions using sparse attention.
        
        Returns:
            Interaction forces, shape (N, 3)
        """
        # Create query, key, value matrices
        q = self.features.astype(np.float32)
        k = self.features.astype(np.float32)
        v = np.concatenate([self.positions, self.velocities], axis=1).astype(np.float32)
        
        # Update sparsity mask
        mask = self.update_sparsity_mask()
        
        # Apply sparse attention
        output = sparse_attention(q, k, v, mask)
        
        # Extract position and velocity updates
        pos_updates = output[:, :3]
        vel_updates = output[:, 3:6]
        
        # Compute forces (simplified model)
        forces = pos_updates - self.positions + 0.1 * vel_updates
        
        return forces
    
    def step(self) -> None:
        """
        Advance the simulation by one time step.
        """
        # Compute forces
        forces = self.compute_interaction()
        
        # Update velocities
        self.velocities += forces * self.dt
        
        # Update positions
        self.positions += self.velocities * self.dt
        
        # Apply periodic boundary conditions
        if self.periodic:
            self.positions = self.positions % self.box_size
        
        # Update time
        self.time += self.dt
        
        # Store position for history
        self.position_history.append(self.positions.copy())
    
    def run(self, num_steps: int) -> None:
        """
        Run the simulation for a specified number of steps.
        
        Args:
            num_steps: Number of time steps to simulate
        """
        for _ in range(num_steps):
            self.step()
    
    def visualize_3d(
        self,
        step: Optional[int] = None,
        ax: Optional[plt.Axes] = None,
        color_by: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        marker_size: float = 10.0
    ) -> plt.Axes:
        """
        Visualize the particle system in 3D.
        
        Args:
            step: Time step to visualize (None for latest)
            ax: Matplotlib axes to plot on (None to create new)
            color_by: Function to determine particle colors
            marker_size: Size of particle markers
            
        Returns:
            Matplotlib axes
        """
        if step is None:
            positions = self.positions
        else:
            positions = self.position_history[step]
        
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        
        # Determine colors
        if color_by is None:
            colors = np.linalg.norm(self.velocities, axis=1)
        else:
            colors = color_by(positions)
        
        # Plot particles
        scatter = ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            c=colors,
            cmap='viridis',
            s=marker_size,
            alpha=0.8
        )
        
        # Set axis limits
        ax.set_xlim(0, self.box_size)
        ax.set_ylim(0, self.box_size)
        ax.set_zlim(0, self.box_size)
        
        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Particle System at t={self.time:.2f}')
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Velocity Magnitude')
        
        return ax
    
    def create_animation(
        self,
        num_frames: int = 100,
        interval: int = 50,
        skip: int = 1,
        filename: Optional[str] = None
    ) -> FuncAnimation:
        """
        Create an animation of the particle system.
        
        Args:
            num_frames: Number of frames in the animation
            interval: Interval between frames (ms)
            skip: Number of steps to skip between frames
            filename: Filename to save the animation (None to not save)
            
        Returns:
            Matplotlib animation
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Run simulation if needed
        steps_needed = num_frames * skip
        current_steps = len(self.position_history) - 1
        if current_steps < steps_needed:
            self.run(steps_needed - current_steps)
        
        # Setup plot
        positions = self.position_history[0]
        scatter = ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            c=np.linalg.norm(self.velocities, axis=1),
            cmap='viridis',
            s=10,
            alpha=0.8
        )
        
        # Set axis limits
        ax.set_xlim(0, self.box_size)
        ax.set_ylim(0, self.box_size)
        ax.set_zlim(0, self.box_size)
        
        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        title = ax.set_title('Particle System at t=0.00')
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Velocity Magnitude')
        
        def update(frame):
            # Update positions
            positions = self.position_history[frame * skip]
            scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
            
            # Update colors
            colors = np.linalg.norm(self.velocities, axis=1)
            scatter.set_array(colors)
            
            # Update title
            title.set_text(f'Particle System at t={frame * skip * self.dt:.2f}')
            
            return scatter,
        
        # Create animation
        anim = FuncAnimation(
            fig, update, frames=num_frames, interval=interval, blit=True
        )
        
        # Save animation if filename is provided
        if filename is not None:
            anim.save(filename, writer='pillow', fps=30)
        
        return anim
    
    def analyze_clustering(self) -> Tuple[float, float]:
        """
        Analyze particle clustering in the simulation.
        
        Returns:
            Tuple of (clustering_coefficient, average_distance)
        """
        # Compute pairwise distances
        distances = np.zeros((self.num_particles, self.num_particles))
        for i in range(self.num_particles):
            for j in range(i + 1, self.num_particles):
                dist = np.linalg.norm(self.positions[i] - self.positions[j])
                
                if self.periodic:
                    # Apply periodic boundary conditions
                    dx = self.positions[i, 0] - self.positions[j, 0]
                    dy = self.positions[i, 1] - self.positions[j, 1]
                    dz = self.positions[i, 2] - self.positions[j, 2]
                    
                    if abs(dx) > self.box_size / 2:
                        dx = self.box_size - abs(dx)
                    if abs(dy) > self.box_size / 2:
                        dy = self.box_size - abs(dy)
                    if abs(dz) > self.box_size / 2:
                        dz = self.box_size - abs(dz)
                    
                    dist = np.sqrt(dx**2 + dy**2 + dz**2)
                
                distances[i, j] = distances[j, i] = dist
        
        # Compute clustering coefficient (simplified)
        mask = distances < self.interaction_threshold
        clustering_coefficient = mask.sum() / (self.num_particles * (self.num_particles - 1))
        
        # Compute average distance
        avg_distance = distances.sum() / (self.num_particles * (self.num_particles - 1))
        
        return clustering_coefficient, avg_distance
    
    def plot_clustering_evolution(self) -> plt.Figure:
        """
        Plot the evolution of clustering over time.
        
        Returns:
            Matplotlib figure
        """
        # Calculate clustering for each time step
        clustering_coeffs = []
        avg_distances = []
        
        for positions in self.position_history:
            # Store current positions
            current_positions = self.positions.copy()
            
            # Set positions to historical positions
            self.positions = positions
            
            # Calculate clustering
            cc, ad = self.analyze_clustering()
            clustering_coeffs.append(cc)
            avg_distances.append(ad)
            
            # Restore current positions
            self.positions = current_positions
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Time array
        times = np.arange(len(self.position_history)) * self.dt
        
        # Plot clustering coefficient
        ax1.plot(times, clustering_coeffs, 'b-', linewidth=2)
        ax1.set_ylabel('Clustering Coefficient')
        ax1.set_title('Evolution of Particle Clustering')
        ax1.grid(True)
        
        # Plot average distance
        ax2.plot(times, avg_distances, 'r-', linewidth=2)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Average Distance')
        ax2.grid(True)
        
        plt.tight_layout()
        
        return fig 