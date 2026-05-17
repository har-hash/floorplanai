import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Import the previously built modules
from boundary_encoder import BoundaryContextEncoder
from constraint_mapper import UserConstraintMapper
from gat_module import RoomGraphNetwork
from generative_trainer import RoomGeneratorHead

class AIArchitect(nn.Module):
    """
    The complete model pipeline assembling all checkpoints.
    Input: Boundary Image, Room Types array, Adjacency Graph
    Output: Predicted bounding boxes for each room type
    """
    def __init__(self, num_room_types=20):
        super(AIArchitect, self).__init__()
        
        # 1. Spatial Boundary Encoder (Output: 128-dim)
        self.boundary_encoder = BoundaryContextEncoder(latent_dim=128, handle_orientations=True)
        
        # 2. User Constraint Mapper (Output: 64-dim)
        self.constraint_mapper = UserConstraintMapper(num_room_types=num_room_types, constraint_dim=64)
        
        # 3. Room Graph Message Passing Network (Output: 128-dim per node)
        self.room_gat = RoomGraphNetwork(num_room_types=num_room_types, embedding_dim=64, hidden_dim=128)
        
        # 4. Generative Decoder Head (Input: 128 + 192 = 320-dim, Output: 4-dim (x,y,w,h))
        self.generator = RoomGeneratorHead(node_embed_dim=128, global_cond_dim=192, hidden_dim=256)

    def forward(self, boundary_img, room_counts, room_types, adjacency):
        """
        Forward pass connecting all modules to generate the "Cloud of Rectangles"
        """
        # A. Get Global Context
        boundary_feature = self.boundary_encoder(boundary_img)      # (B, 128)
        constraint_feature = self.constraint_mapper(room_counts)    # (B, 64)
        global_cond = torch.cat([boundary_feature, constraint_feature], dim=1) # (B, 192)
        
        # B. Get Local Graph Context
        node_embeddings = self.room_gat(room_types, adjacency)      # (B, N, 128)
        
        # C. Predict Coordinates
        pred_boxes = self.generator(node_embeddings, global_cond)   # (B, N, 4)
        
        return pred_boxes

def generate_and_visualize(model):
    """
    Inference Pipeline Test:
    Input a simple boundary and room graph, and output the "Cloud of Rectangles"
    """
    model.eval()
    
    # --- 1. Define Inference Input ---
    B, N = 1, 6 # 1 Sample, 6 Rooms Requested
    num_room_types = 20
    
    # 1.1 Simple rectangular Boundary Mask (100x100 box inside the 256x256 image)
    boundary_img = torch.zeros(B, 1, 256, 256)
    boundary_img[:, :, 80:180, 80:180] = 1.0 # 1 inside, 0 outside
    
    # 1.2 User Constraints Example:
    # Let's say: 1 Living(0), 2 Bed(1), 1 Bath(2), 1 Kitchen(3), 1 Dining(4)
    room_counts = torch.zeros(B, num_room_types)
    room_counts[0, 0] = 1 # Living
    room_counts[0, 1] = 2 # Bed
    room_counts[0, 2] = 1 # Bath
    room_counts[0, 3] = 1 # Kitchen
    room_counts[0, 4] = 1 # Dining
    
    # 1.3 Ordered sequence of these rooms as individual nodes for the Graph
    room_types = torch.tensor([[0, 1, 1, 2, 3, 4]]) 
    
    # 1.4 Dummy Adjacency Graph connecting them (All connected to Living(0))
    # We also connect Kitchen(4) to Dining(5)
    adjacency = torch.zeros(B, N, N)
    edges = [(0,1), (0,2), (0,3), (0,4), (0,5), (4,5)]
    for u, v in edges:
        adjacency[0, u, v] = 1.0
        adjacency[0, v, u] = 1.0
    for i in range(N):
        adjacency[0, i, i] = 1.0 # Self loops
        
    print("Inputs prepared. Running Inference Model...")
    
    # --- 2. Run Inference ---
    with torch.no_grad():
        predicted_boxes = model(boundary_img, room_counts, room_types, adjacency)
    
    # --- 3. Visualize the "Cloud of Rectangles" ---
    boxes = predicted_boxes[0].numpy()  # Extract the single generated batch
    room_seq = room_types[0].numpy()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.invert_yaxis()
    ax.set_xlim(0, 256)
    ax.set_ylim(256, 0)
    
    # Draw true Boundary Mask Box Limit (80 to 180) for visual reference
    b_limit = Polygon([[80, 80], [180, 80], [180, 180], [80, 180]], 
                      closed=True, fill=False, edgecolor='black', linewidth=4, linestyle='--')
    ax.add_patch(b_limit)
    ax.text(80, 75, "Hard Boundary Limit", color='black', fontweight='bold')
    
    room_names = {0: "Living", 1: "Bed", 2: "Bath", 3: "Kitchen", 4: "Dining"}
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#D1A3FF']
    
    # Plot predicted Cloud of Rectangles
    for i, (x_c, y_c, w, h) in enumerate(boxes):
        r_type = room_seq[i]
        color = colors[r_type % len(colors)]
        
        # Convert absolute centroid (x,y) and size (w,h) to corner points
        x_min = x_c - (w / 2.0)
        y_min = y_c - (h / 2.0)
        
        # Draw the rectangle
        rect_poly = Polygon(
            [[x_min, y_min], [x_min+w, y_min], [x_min+w, y_min+h], [x_min, y_min+h]],
            closed=True, fill=True, facecolor=color, alpha=0.5, edgecolor='black', linewidth=2
        )
        ax.add_patch(rect_poly)
        
        # Plot centroid point
        ax.scatter(x_c, y_c, color='black', s=30)
        ax.text(x_c, y_c-10, room_names.get(r_type, f"R{r_type}"), 
                fontsize=10, ha='center', fontweight='bold', color='black',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
                
    # Draw valid graph edges predicted between the room centroids
    for u, v in edges:
        xu, yu = boxes[u][0], boxes[u][1]
        xv, yv = boxes[v][0], boxes[v][1]
        ax.plot([xu, xv], [yu, yv], color='red', linestyle=':', linewidth=2)

    ax.set_aspect('equal', adjustable='box')
    plt.title("Checkpoint 2.3: Initial Output (Cloud of Rectangles)")
    
    # Save image
    out_img = "cloud_of_rectangles_inference.png"
    plt.savefig(out_img, dpi=150)
    print(f"Inference complete! Output saved to: {out_img}")

if __name__ == "__main__":
    print("Initializing Unified AI Architect Model...")
    model = AIArchitect()
    generate_and_visualize(model)
