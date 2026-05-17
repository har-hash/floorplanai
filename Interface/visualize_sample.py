import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def plot_sample(sample_idx=0, data_path=r"H:\EDAI 2\Interface\static\Data\data_train_converted.pkl", output_path="sample_visualization.png"):
    # Load dataset
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    samples = data['data']
    
    if sample_idx >= len(samples):
        print("Sample index out of range!")
        return
        
    sample = samples[sample_idx]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.invert_yaxis() # Often needed for image coordinates
    
    # 1. Plot building boundary
    # boundary has shape (N, 4) presumably [x, y, ...]
    if hasattr(sample, 'boundary') and sample.boundary is not None and len(sample.boundary) > 0:
        bnd = sample.boundary[:, :2] # taking just x and y
        b_poly = Polygon(bnd, closed=True, fill=False, edgecolor='black', linewidth=3, zorder=1)
        ax.add_patch(b_poly)
        # also scatter the points to see the vertices
        ax.scatter(bnd[:, 0], bnd[:, 1], c='black', s=20)
    
    # 2. Plot rooms (Nodes)
    nodes = []
    # If rBoundary exists, we can draw the rooms as well
    if hasattr(sample, 'rBoundary'):
        for i, room_coords in enumerate(sample.rBoundary):
            r_poly = Polygon(room_coords, closed=True, fill=True, facecolor=f'C{i%10}', alpha=0.3, edgecolor='white', zorder=2)
            ax.add_patch(r_poly)
            # Center of the room based on bounding box
            cx = (sample.box[i][0] + sample.box[i][2]) / 2.0
            cy = (sample.box[i][1] + sample.box[i][3]) / 2.0
            nodes.append((cx, cy))
            
            ax.text(cx, cy, f"R{i}", fontsize=12, ha='center', va='center', fontweight='bold', color='black', zorder=4)
            # Optionally draw box type
            ax.text(cx, cy+10, f"type:{sample.box[i][4]}", fontsize=8, ha='center', va='center', color='darkblue', zorder=4)

    # 3. Plot room graph (Edges)
    if hasattr(sample, 'edge'):
        for edge in sample.edge:
            u, v, e_type = edge[0], edge[1], edge[2]
            if u < len(nodes) and v < len(nodes):
                xu, yu = nodes[u]
                xv, yv = nodes[v]
                ax.plot([xu, xv], [yu, yv], color='red', linestyle='--', linewidth=2, zorder=3)
                
                # Plot edge type
                mx, my = (xu + xv)/2, (yu + yv)/2
                ax.text(mx, my, str(e_type), fontsize=9, color='red', backgroundcolor='white')

    ax.autoscale()
    ax.set_aspect('equal', adjustable='box')
    plt.title(f"Building Boundary and Room Graph - Sample {sample_idx}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    plot_sample()
