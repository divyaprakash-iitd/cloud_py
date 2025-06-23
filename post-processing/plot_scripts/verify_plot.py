import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

# Load the binary file
data = np.load('outdir_sk1_nm300_nx32/003000/superdroplet_neighbors.npy', allow_pickle=True)

# Randomly select 5 superdroplets
n_to_plot = 10
if len(data) < n_to_plot:
    n_to_plot = len(data)  # Adjust if fewer than 5 superdroplets exist
indices = np.random.choice(len(data), n_to_plot, replace=False)
selected_data = data[indices]

# Define colors for each superdroplet
colors = list(mcolors.TABLEAU_COLORS.values())[:n_to_plot]  # Use Tableau colors

# Create 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot each selected superdroplet
for i, (entry, color) in enumerate(zip(selected_data, colors)):
    icdfr = entry['icdfr']
    
    # Plot superdroplet
    ax.scatter(entry['super_x'], entry['super_y'], entry['super_z'], 
               c=color, marker='*', s=100, label=f'Superdroplet {i+1} (icdfr={icdfr})')
    
    # Plot centroid
    ax.scatter(entry['centroid_x'], entry['centroid_y'], entry['centroid_z'], 
               c=color, marker='o', s=100, edgecolors='black', label=f'Centroid {i+1}')
    
    # Plot neighbors
    ax.scatter(entry['neighbor_x'], entry['neighbor_y'], entry['neighbor_z'], 
               c=color, marker='.', s=100, alpha=0.6, label=f'Neighbors {i+1}')
    
    # Connect superdroplet to neighbors
    for nx, ny, nz in zip(entry['neighbor_x'], entry['neighbor_y'], entry['neighbor_z']):
        ax.plot([entry['super_x'], nx], [entry['super_y'], ny], [entry['super_z'], nz], 
                c=color, alpha=0.3)
    
    # Connect superdroplet to centroid
    ax.plot([entry['super_x'], entry['centroid_x']], 
            [entry['super_y'], entry['centroid_y']], 
            [entry['super_z'], entry['centroid_z']], 
            c=color, linestyle='--', alpha=0.8)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(f'5 Randomly Selected Superdroplets with Neighbors')

# Add legend with unique entries
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))  # Remove duplicates
#ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))

# Adjust layout and show plot
plt.tight_layout()
plot_filename = 'sdrop_clusters'
plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
plt.show()
