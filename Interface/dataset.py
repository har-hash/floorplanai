import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw

class EDAIDataset(Dataset):
    """
    Production-Level PyTorch Dataset for loading and processing the 
    EDAI 2 generative floorplan `.pkl` database.
    """
    def __init__(self, pkl_path, num_room_types=20, max_samples=None):
        self.pkl_path = pkl_path
        self.num_room_types = num_room_types
        
        print(f"Loading dataset from {pkl_path}... this may take a moment.")
        with open(self.pkl_path, 'rb') as f:
            raw_data = pickle.load(f)
            
        # The exact structure identified in Phase 1:
        all_samples = raw_data['data']
        
        # Filter out corrupted/incomplete samples missing required 'box' data
        total_before = len(all_samples)
        self.samples = [
            s for s in all_samples
            if hasattr(s, 'box') and s.box is not None and len(s.box) > 0
            and all(len(b) >= 5 for b in s.box)
        ]
        skipped = total_before - len(self.samples)
        if skipped > 0:
            print(f"  [Warning] Filtered out {skipped}/{total_before} samples with missing/invalid 'box' data.")
        
        # Optional truncater for rapid localized testing
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
            
        print(f"Successfully loaded {len(self.samples)} valid samples.")

    def __len__(self):
        return len(self.samples)

    def _rasterize_boundary(self, boundary_coords, image_size=256):
        """
        Takes N x 4 array (where first 2 columns are raw x, y integer coords),
        and dynamically rasterizes it into a discrete 256x256 binary mask tensor.
        1.0 inside the footprint, 0.0 outside.
        """
        # Create a blank black image
        img = Image.new('L', (image_size, image_size), 0)

        if boundary_coords is not None and len(boundary_coords) > 0:
            # Extract just x, y pairs
            xy_points = boundary_coords[:, :2].tolist()

            # Format requires a flat list or list of tuples: [(x,y), (x,y), ...]
            # Clamp coordinates to valid image bounds [0, image_size)
            xy_tuples = [
                (max(0, min(image_size - 1, int(pt[0]))),
                 max(0, min(image_size - 1, int(pt[1]))))
                for pt in xy_points
            ]

            # Only draw if we have valid points
            if xy_tuples:
                draw = ImageDraw.Draw(img)
                # Fill the exact spatial polygon with white (255)
                draw.polygon(xy_tuples, outline=255, fill=255)
            
        # Convert to numpy block, normalize to [0.0, 1.0], convert to FloatTensor
        mask = np.array(img, dtype=np.float32) / 255.0
        # Return shape: (1, 256, 256) matching PyTorch Image Ch conventions
        return torch.tensor(mask).unsqueeze(0)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 1. Rasterize the building boundary logic
        # Extract boundary (usually N x 4 from inspect output)
        if hasattr(sample, 'boundary') and sample.boundary is not None:
            boundary = self._rasterize_boundary(sample.boundary)
        else:
            boundary = torch.zeros(1, 256, 256)

        # 2. Extract Room Topologies (Nodes)
        # CRITICAL: Validate that sample.box exists and has data
        if not hasattr(sample, 'box') or sample.box is None or len(sample.box) == 0:
            raise ValueError(f"Sample {idx}: Missing or empty 'box' attribute. Cannot extract room topology.")

        # box shape: (N, 5) -> [xmin, ymin, xmax, ymax, room_type]
        num_rooms = len(sample.box)

        # Room types: extracted integer classification
        # CRITICAL: Validate box element structure before unpacking
        room_types_list = []
        for i, b in enumerate(sample.box):
            if len(b) < 5:
                raise ValueError(f"Sample {idx}, Box {i}: Expected 5 elements [xmin, ymin, xmax, ymax, room_type], got {len(b)}")
            room_types_list.append(int(b[4]))
        room_types = torch.tensor(room_types_list, dtype=torch.long)

        # Generate the constraints vector (e.g., [1 Living, 2 Bed, 1 Bath...])
        room_counts = torch.zeros(self.num_room_types)
        for rt in room_types:
            if rt < self.num_room_types:
                room_counts[rt] += 1.0

        # Target bounding boxes: Reformat pure coordinates into (x_centroid, y_centroid, w, h)
        target_boxes = torch.zeros(num_rooms, 4)
        for i, b in enumerate(sample.box):
            xmin, ymin, xmax, ymax, _ = b
            w = xmax - xmin
            h = ymax - ymin
            x_c = xmin + (w / 2.0)
            y_c = ymin + (h / 2.0)
            target_boxes[i] = torch.tensor([x_c, y_c, w, h], dtype=torch.float32)

        # 3. Extract Adjacency (Edges)
        # Undirected graph initialization
        adjacency = torch.zeros((num_rooms, num_rooms), dtype=torch.float32)
        
        # Add explicit cross-room constraints via the 'edge' array
        if hasattr(sample, 'edge'):
            for e in sample.edge:
                u, v = int(e[0]), int(e[1])
                # Safety check guaranteeing edges point to valid nodes
                if u < num_rooms and v < num_rooms:
                    adjacency[u, v] = 1.0
                    adjacency[v, u] = 1.0
        
        # Explicit Identity self-loops so GAT doesn't delete the node's original self-embedding
        adjacency.fill_diagonal_(1.0)
        
        return {
            'boundary': boundary,
            'room_counts': room_counts,
            'room_types': room_types,
            'adjacency': adjacency,
            'target_boxes': target_boxes,
            'num_rooms': num_rooms
        }

def edai_collate_fn(batch):
    """
    Since every floorplan has a completely different number of rooms (nodes),
    custom collation explicitly zero-pads smaller networks within a batch 
    so they stack seamlessly onto the GPU.
    """
    max_rooms = max([item['num_rooms'] for item in batch])
    batch_size = len(batch)

    # 1. Uniform components (Fixed Sizes)
    boundaries = torch.stack([item['boundary'] for item in batch])        # (B, 1, 256, 256)
    room_counts = torch.stack([item['room_counts'] for item in batch])    # (B, num_room_types)

    # 2. Dynamic components (Require 0-Padding)
    # Output structure initialization
    b_room_types = torch.zeros(batch_size, max_rooms, dtype=torch.long)
    b_adjacency = torch.zeros(batch_size, max_rooms, max_rooms, dtype=torch.float32)
    b_target_boxes = torch.zeros(batch_size, max_rooms, 4, dtype=torch.float32)

    # We also need a Node Validity mask (1 if real room, 0 if it's just padding)
    # so our Multi-Objective Loss logic from Checkpoint 2.2 knows to IGNORE fake zero-rooms
    b_node_masks = torch.zeros(batch_size, max_rooms, dtype=torch.float32)

    for i, item in enumerate(batch):
        r = item['num_rooms']
        b_room_types[i, :r] = item['room_types']
        b_adjacency[i, :r, :r] = item['adjacency']
        b_target_boxes[i, :r, :] = item['target_boxes']
        b_node_masks[i, :r] = 1.0
        
        # CRITICAL FIX: Padded mock-nodes previously had completely 0.0 adjacency rows.
        # Inside the GAT layer, this forced attention weights to [-inf, -inf...].
        # Softmax([-inf, -inf]) causes a Division by Zero -> NaN tensor corruption!
        # Giving padded nodes a fake self-loop safely routes the math to [1.0] attention.
        for pad_idx in range(r, max_rooms):
            b_adjacency[i, pad_idx, pad_idx] = 1.0

    return {
        'boundary_img': boundaries,
        'room_counts': room_counts,
        'room_types': b_room_types,
        'adjacency': b_adjacency,
        'target_boxes': b_target_boxes,
        'node_mask': b_node_masks
    }

if __name__ == "__main__":
    print("Testing Production DataLoader (Checkpoint 3.1)...")
    
    # Target our primary local pickle data source
    data_path = r"H:\EDAI 2\Interface\static\Data\data_train_converted.pkl"
    
    # Restrict load size strictly for instant unit testing right now
    dataset = EDAIDataset(data_path, max_samples=100)
    
    # Initialize the DataLoader wrapping the generic Dataset operations 
    # to test exact iteration properties spanning multiple batch matrices
    try:
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=edai_collate_fn)
        
        batch = next(iter(dataloader))
        print("\n--- Batch Assembled Successfully ---")
        print(f"Boundary Raster Stack: {batch['boundary_img'].shape}")
        print(f"Constraints Matrix:    {batch['room_counts'].shape}")
        print(f"Padded Room Types:     {batch['room_types'].shape}")
        print(f"Padded Adjacency:      {batch['adjacency'].shape}")
        print(f"Padded Target Boxes:   {batch['target_boxes'].shape}")
        print(f"Validity Masks:        {batch['node_mask'].shape}")
        
        # Verify valid boundaries max/min math rules
        print(f"\nExample Node Validity Mask (0 = Pad): {batch['node_mask'][0].tolist()}")
        print("\nCheckpoint 3.1 Functioning Native! Memory pads dynamically managed.")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nError: {e}")
