import torch
import torch.nn as nn
import torchvision.models as models

class BoundaryContextEncoder(nn.Module):
    """
    A CNN encoder based on ResNet-50 that takes a 256x256 boundary raster
    and extracts a fixed-length latent feature vector.
    
    To handle different building orientations, we implement Rotation Pooling:
    The module rotates the input image by 0, 90, 180, and 270 degrees, 
    passes all variants through the backbone, and max-pools their features.
    This creates a rotationally invariant latent representation.
    """
    def __init__(self, latent_dim=256, use_pretrained=True, handle_orientations=True):
        super(BoundaryContextEncoder, self).__init__()
        
        self.handle_orientations = handle_orientations
        
        # Initialize ResNet-50 backbone
        # We use pretrained weights to help with feature extraction
        weights = models.ResNet50_Weights.DEFAULT if use_pretrained else None
        resnet = models.resnet50(weights=weights)
        
        # Remove the final fully connected layer (classifier)
        # This leaves us with a backbone ending in an AdaptiveAvgPool2d (output size 1x1)
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1] 
        )
        
        resnet_out_dim = 2048 # ResNet-50 output feature dimension
        
        # Projection head to get to the desired fixed latent dimension
        self.fc = nn.Sequential(
            nn.Linear(resnet_out_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, latent_dim)
        )

    def forward_single(self, x):
        """ Forward pass for a single orientation. """
        # If input is a 1-channel grayscale binary raster, duplicate it to 3 channels for ResNet
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
            
        # Extract features
        features = self.backbone(x)
        features = torch.flatten(features, 1) # Shape: (B, 2048)
        
        # Project to latent space
        latent = self.fc(features) # Shape: (B, latent_dim)
        return latent

    def forward(self, x):
        """
        Input: x of shape (B, C, 256, 256)
        Output: latent vector of shape (B, latent_dim)
        """
        # Option 1: Direct pass without rotation handling
        if not self.handle_orientations:
            return self.forward_single(x)
        
        # Option 2: Rotation Pooling for Orientation Invariance
        latent_features = []
        
        # Rotate by 0, 90, 180, 270 degrees
        for k in range(4):
            # torch.rot90 rotates along H, W dimensions (dims 2 and 3) by k * 90 degrees
            x_rot = torch.rot90(x, k, [2, 3])
            
            latent = self.forward_single(x_rot)
            latent_features.append(latent.unsqueeze(1)) # Add a dimension for pooling (B, 1, latent_dim)
            
        # Stack along new dimension: (B, 4, latent_dim)
        latents_stacked = torch.cat(latent_features, dim=1)
        
        # Max pool over the 4 rotation orientations to enforce rotational invariance
        latent_inv, _ = torch.max(latents_stacked, dim=1) # (B, latent_dim)
        
        return latent_inv

if __name__ == "__main__":
    # Test the encoder
    print("Initializing BoundaryContextEncoder...")
    model = BoundaryContextEncoder(latent_dim=128, handle_orientations=True)
    model.eval()
    
    # Simulate a batch of 2 rasterized boundary images: (Batch, Channels, H, W)
    # Background=0, Boundary=1
    dummy_input = torch.zeros(2, 1, 256, 256)
    
    print("Testing forward pass with rotation pooling...")
    with torch.no_grad():
        output = model(dummy_input)
        
    print(f"Input raster shape:  {dummy_input.shape}")
    print(f"Output latent shape: {output.shape}")
