import os
import json
import torch
import numpy as np
import argparse
from inference_pipeline import AIArchitect

class ArchitecturalInferenceAPI:
    """
    Production API designed to run entirely independently from the 75,000 .pkl dataset.
    It purely relies on the mathematical weights established during Phase 4 training.
    """
    def __init__(self, model_weight_path="checkpoints/AI_Architect_Best.pth", num_room_types=20, device='cpu'):
        self.device = torch.device(device)
        self.num_room_types = num_room_types
        
        # Initialize the barren architecture
        self.model = AIArchitect(num_room_types=self.num_room_types).to(self.device)
        
        # Inject the trained brain (If available)
        if os.path.exists(model_weight_path):
            checkpoint = torch.load(model_weight_path, map_location=self.device, weights_only=False)
            # Support loading both raw state dicts or composite checkpoint dictionaries
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            self.model.load_state_dict(state_dict)
            print(f"[*] Successfully injected trained weights from {model_weight_path}.")
        else:
            print(f"[!] Warning: No trained weights found at {model_weight_path}. Running randomly.")
            
        self.model.eval()

    def process_json_request(self, json_string):
        """
        Translates a human-readable JSON string into PyTorch native Tensors.
        Expected JSON format:
        {
            "boundary": [[x,y], [x,y], [x,y], [x,y]],
            "rooms": { "0": 1, "1": 2, "2": 1 }  // Type 0 (Living): 1, Type 1 (Bed): 2
        }
        """
        request = json.loads(json_string)
        
        # 1. User Constraints Extractor
        room_dict = request.get("rooms", {})
        room_counts_tensor = torch.zeros(1, self.num_room_types, device=self.device)
        
        # Create continuous graph sequences for the Topologies
        ordered_room_types = []
        for room_id_str, count in room_dict.items():
            r_id = int(room_id_str)
            if r_id < self.num_room_types:
                room_counts_tensor[0, r_id] += float(count)
                ordered_room_types.extend([r_id] * int(count))
                
        num_requested = len(ordered_room_types)
        if num_requested == 0:
            raise ValueError("No rooms requested in JSON!")
            
        r_types_tensor = torch.tensor([ordered_room_types], dtype=torch.long, device=self.device)
        
        # 2. Boundary Extractor
        # We assume for this high-level API a simple fixed boundary if none is provided
        boundary_img = torch.zeros(1, 1, 256, 256, device=self.device)
        # Placeholder: Generate a 100x100 inner safe zone 
        boundary_img[:, :, 50:200, 50:200] = 1.0 
        
        # 3. Adjacency Logic
        # A full production system would algorithmically parse user relational requests.
        # Below represents an assumption that all nodes are generally densely connected 
        # (an open-floorplan assumption)
        adj_tensor = torch.ones(1, num_requested, num_requested, device=self.device)
        
        return boundary_img, room_counts_tensor, r_types_tensor, adj_tensor

    def generate_layout(self, json_string):
        """
        Main Generation Feed
        """
        print(f"[*] Processing Layout parameters: {json_string}")
        b_img, r_counts, r_types, adj = self.process_json_request(json_string)
        
        with torch.no_grad():
            predicted_boxes = self.model(b_img, r_counts, r_types, adj)
            
        print("[*] Layout Generated Successfully.")
        # Returns coordinates shape: (N, 4)
        return predicted_boxes[0].cpu().numpy(), r_types[0].cpu().numpy()

if __name__ == "__main__":
    # Test JSON string payload (Simulating a Web API Request)
    sample_request = '{"rooms": {"0": 1, "1": 3, "2": 2, "3": 1, "4": 1}}'
    
    api = ArchitecturalInferenceAPI()
    raw_coordinates, types = api.generate_layout(sample_request)
    
    for i, coords in enumerate(raw_coordinates):
        print(f"Room [Type {types[i]}]: x={coords[0]:.1f}, y={coords[1]:.1f}, w={coords[2]:.1f}, h={coords[3]:.1f}")
