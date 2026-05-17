import torch
import torch.nn as nn

class UserConstraintMapper(nn.Module):
    """
    Maps structured user constraints (e.g., room counts) into a fixed-length 
    conditioning feature vector.
    """
    def __init__(self, num_room_types=20, constraint_dim=64):
        """
        Args:
            num_room_types (int): Number of separate room categories (e.g. bedroom, bathroom, etc.)
            constraint_dim (int): The dimensional size of the output constraint vector.
        """
        super(UserConstraintMapper, self).__init__()
        
        # A simple Multi-Layer Perceptron (MLP) to process the raw counts
        self.mlp = nn.Sequential(
            nn.Linear(num_room_types, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, constraint_dim),
            nn.ReLU(inplace=True) # Ensure non-negative features, or remove based on preference
        )
        
    def forward(self, constraint_counts):
        """
        Args:
            constraint_counts (torch.Tensor): A tensor of shape (Batch, num_room_types)
                containing the requested count for each room type.
        Returns:
            torch.Tensor: The embedded constraint vector of shape (Batch, constraint_dim)
        """
        # (Batch, constraint_dim)
        return self.mlp(constraint_counts)

if __name__ == "__main__":
    from boundary_encoder import BoundaryContextEncoder
    
    print("Initializing components...")
    # 1. Initialize Boundary Encoder
    boundary_encoder = BoundaryContextEncoder(latent_dim=128, handle_orientations=True)
    boundary_encoder.eval()
    
    # 2. Initialize Constraint Mapper
    constraint_mapper = UserConstraintMapper(num_room_types=20, constraint_dim=64)
    constraint_mapper.eval()
    
    # --- Simulate Inputs ---
    batch_size = 2

    # Dummy Raster Boundary (Batch, Channels, H, W)
    dummy_boundary_raster = torch.zeros(batch_size, 1, 256, 256)

    # Dummy User Constraints (Batch, num_room_types)
    # E.g., Room types index: 0=Living, 1=Bed, 2=Bath, etc.
    # Let's say sample 0 wants: 3 beds, 2 baths.
    dummy_constraints = torch.zeros(batch_size, 20)
    dummy_constraints[0, 1] = 3.0 # 3 Bedrooms
    dummy_constraints[0, 2] = 2.0 # 2 Bathrooms
    dummy_constraints[1, 1] = 1.0 # 1 Bedroom
    dummy_constraints[1, 2] = 1.0 # 1 Bathroom
    
    print("\n--- Forward Pass ---")
    with torch.no_grad():
        # Extracted boundary vector
        boundary_vector = boundary_encoder(dummy_boundary_raster)
        print(f"Boundary Vector Shape:   {boundary_vector.shape}") # Expected: (2, 128)
        
        # Extracted constraint vector
        constraint_vector = constraint_mapper(dummy_constraints)
        print(f"Constraint Vector Shape: {constraint_vector.shape}") # Expected: (2, 64)
        
        # === Checkpoint Goal: Concatenation ===
        # Concatenate along the feature dimension (dim=1)
        final_conditioning_vector = torch.cat([boundary_vector, constraint_vector], dim=1)
        
    print(f"\nFinal Concatenated Conditioning Vector Shape: {final_conditioning_vector.shape}")
    print("Checkpoint 1.3 Successful: Merged boundary and constraints into a single vector!")
