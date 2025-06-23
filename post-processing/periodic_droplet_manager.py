import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from typing import Union, List, Tuple, Dict, Any, Optional

class PeriodicDropletManager:
    """
    A class for managing droplet locations in a periodic cubic box and 
    performing efficient neighbor searches.
    """
    
    def __init__(self, droplet_locations: np.ndarray, box_size: np.ndarray):
        """
        Initialize the PeriodicDropletManager with droplet locations and box dimensions.
        
        Args:
            droplet_locations: Array of shape (n_droplets, 3) representing 3D coordinates
            box_size: Array of shape (3,) representing dimensions of the periodic box
        """
        self.droplet_locations = np.asarray(droplet_locations)
        self.box_size = np.asarray(box_size)
        
        # Validate inputs
        if self.droplet_locations.ndim != 2 or self.droplet_locations.shape[1] != 3:
            raise ValueError("Droplet locations must be a 2D array with shape (n_droplets, 3)")
        
        if self.box_size.shape != (3,):
            raise ValueError("Box size must be an array with shape (3,)")
        
        # Create spatial index (KD-Tree) with periodic boundaries
        self.tree = cKDTree(self.droplet_locations, boxsize=self.box_size)
    
    def find_neighbors(self, query_points: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors for each query point considering periodic boundaries.
        
        Args:
            query_points: Single point of shape (3,) or array of shape (n_queries, 3)
            k: Number of nearest neighbors to find
            
        Returns:
            Tuple containing:
                - distances: Array of shape (n_queries, k) with distances to neighbors
                - indices: Array of shape (n_queries, k) with indices of neighbors
        """
        # Handle single query point case
        query_points = np.asarray(query_points)
        if query_points.ndim == 1:
            query_points = query_points.reshape(1, -1)
            
        # Validate input
        if query_points.shape[1] != 3:
            raise ValueError("Query points must have 3 dimensions (x, y, z)")
        
        # Find k nearest neighbors
        distances, indices = self.tree.query(query_points, k=k)
        
        return distances, indices
    
    def compute_periodic_distance(self, point1: np.ndarray, point2: np.ndarray) -> np.ndarray:
        """
        Compute the distance between two points considering periodic boundaries.
        
        Args:
            point1: Array of shape (3,) representing first point
            point2: Array of shape (3,) representing second point
            
        Returns:
            Array of shape (3,) with the periodic displacement vector
        """
        # Calculate difference
        diff = point2 - point1
        
        # Apply periodic boundary conditions
        diff = np.remainder(diff + 0.5 * self.box_size, self.box_size) - 0.5 * self.box_size
        
        return diff
    
    def compute_periodic_centroid(self, indices: np.ndarray) -> np.ndarray:
        """
        Compute the centroid of a set of droplets while correctly handling
        periodic boundary conditions.
        
        Args:
            indices: Array of indices corresponding to droplets in the original array
            
        Returns:
            Array of shape (3,) representing the centroid coordinates
        """
        if len(indices) == 0:
            raise ValueError("Cannot compute centroid of empty set")
        
        # Get the droplet locations
        points = self.droplet_locations[indices]
        
        # Use the first point as a reference
        reference = points[0].copy()
        
        # Calculate periodic distances from reference
        rel_positions = np.zeros_like(points)
        for i, point in enumerate(points):
            rel_positions[i] = reference + self.compute_periodic_distance(reference, point)
        
        # Calculate centroid in the relative coordinate system
        centroid_rel = np.mean(rel_positions, axis=0)
        
        # Map the centroid back to the periodic box
        centroid = np.remainder(centroid_rel, self.box_size)
        
        return centroid
    
    def generate_superdroplet_info(self, query_points: np.ndarray, k: int) -> List[Dict[str, Any]]:
        """
        Generate comprehensive information about superdroplets and their associated
        actual droplets.
        
        Args:
            query_points: Single point of shape (3,) or array of shape (n_queries, 3)
            k: Number of nearest neighbors to find
            
        Returns:
            List of dictionaries where each dictionary contains:
                - 'query_point': Array of shape (3,) with query point coordinates
                - 'neighbor_indices': Array of shape (k,) with indices of neighbors
                - 'centroid': Array of shape (3,) with centroid coordinates
        """
        # Handle single query point case
        query_points = np.asarray(query_points)
        single_query = False
        if query_points.ndim == 1:
            query_points = query_points.reshape(1, -1)
            single_query = True
            
        # Find nearest neighbors
        distances, indices = self.find_neighbors(query_points, k)
        
        # Generate superdroplet info
        result = []
        for i in range(len(query_points)):
            neighbor_indices = indices[i]
            # Make sure neighbor_indices is flattened if it's a 2D array
            if neighbor_indices.ndim > 1:
                neighbor_indices = neighbor_indices.flatten()
                
            centroid = self.compute_periodic_centroid(neighbor_indices)
            
            info = {
                'query_point': query_points[i],
                'neighbor_indices': neighbor_indices,
                'centroid': centroid
            }
            result.append(info)
        
        # If input was a single point, return single result
        if single_query:
            return result[0]
        
        return result
    
    def visualize(self, query_points: np.ndarray, k: int, figsize: Tuple[int, int] = (12, 8), 
                 alpha_reference: float = 0.3, marker_size_query: int = 100, marker_size_neighbor: int = 50):
        """
        Visualize droplets, query points, neighbors and centroids in a 3D plot.
        
        Args:
            query_points: Points to query, shape (n_queries, 3)
            k: Number of nearest neighbors to find
            figsize: Figure size tuple
            alpha_reference: Alpha transparency for reference points
            marker_size_query: Size of query point markers
            marker_size_neighbor: Size of neighbor markers
        """
        # Get info about query points and their neighbors
        if isinstance(query_points, np.ndarray) and query_points.ndim == 1:
            superdroplet_info = [self.generate_superdroplet_info(query_points, k)]
        else:
            superdroplet_info = self.generate_superdroplet_info(query_points, k)
        
        # Create color map for visualization
        n_query_points = len(superdroplet_info)
        colors = list(mcolors.TABLEAU_COLORS.values())
        if n_query_points > len(colors):
            additional_colors = plt.cm.rainbow(np.linspace(0, 1, n_query_points))
            colors = additional_colors
        
        # Create 3D plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        ## Plot reference points in grey
        #ax.scatter(self.droplet_locations[:, 0], self.droplet_locations[:, 1], self.droplet_locations[:, 2], 
        #          c='lightgrey', alpha=alpha_reference, s=20, label='Reference Droplets')
        
        # Plot each query point and its neighbors with the same color
        for i, info in enumerate(superdroplet_info):
            query_point = info['query_point']
            neighbor_indices = info['neighbor_indices']
            centroid = info['centroid']
            
            # Get the neighbors
            neighbor_points = self.droplet_locations[neighbor_indices]
            
            # Plot the neighbors
            ax.scatter(neighbor_points[:, 0], neighbor_points[:, 1], neighbor_points[:, 2],
                      c=[colors[i]], s=marker_size_neighbor, alpha=0.6, label=f'Neighbors of Query {i+1}')
            
            # Plot the query point
            ax.scatter(query_point[0], query_point[1], query_point[2],
                      c=[colors[i]], s=marker_size_query, marker='*', label=f'Query Point {i+1}')
            
            # Plot the centroid
            ax.scatter(centroid[0], centroid[1], centroid[2],
                      c=[colors[i]], s=marker_size_query, marker='o', edgecolors='black', 
                      linewidth=2, label=f'Centroid {i+1}')
            
            # Draw lines from query point to neighbors
            for neighbor in neighbor_points:
                ax.plot([query_point[0], neighbor[0]],
                        [query_point[1], neighbor[1]],
                        [query_point[2], neighbor[2]],
                        c=colors[i], alpha=0.3)
            
            # Draw line from query point to centroid (thicker)
            ax.plot([query_point[0], centroid[0]],
                    [query_point[1], centroid[1]],
                    [query_point[2], centroid[2]],
                    c=colors[i], alpha=0.8, linewidth=2, linestyle='--')
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Droplets with Periodic Boundary Conditions')
        
        # Set axis limits
        ax.set_xlim([0, self.box_size[0]])
        ax.set_ylim([0, self.box_size[1]])
        ax.set_zlim([0, self.box_size[2]])
        
        # Add legend with unique entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Adjust layout
        plt.tight_layout()
        
        # Show plot
        plt.show()
