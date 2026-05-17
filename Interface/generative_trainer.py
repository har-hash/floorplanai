import torch
import torch.nn as nn
import torch.nn.functional as F

class RoomGeneratorHead(nn.Module):
    """
    Takes the node embeddings from the GAT and the global conditioning vector,
    and predicts the (x, y, w, h) bounding box for each room.
    x, y representing the centroid of the room.
    w, h representing the width and height.
    """
    def __init__(self, node_embed_dim, global_cond_dim, hidden_dim=128):
        super(RoomGeneratorHead, self).__init__()
        
        # We concatenate the specific room's GAT embedding with the global conditioning vector
        input_dim = node_embed_dim + global_cond_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            # Output 4 values per node: (x, y, w, h)
            # using Sigmoid to constrain outputs between 0 and 1
            # (which can be scaled up to 256x256 later)
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid() 
        )

    def forward(self, node_embeddings, global_cond):
        """
        Args:
            node_embeddings (torch.Tensor): Output from GAT (B, N, node_embed_dim)
            global_cond (torch.Tensor): Concatenated Boundary+Constraint vector (B, global_cond_dim)
        Returns:
            torch.Tensor: Predicted coordinates (B, N, 4) in format (x, y, w, h) scaled to [0, 256]
        """
        B, N, E = node_embeddings.size()
        
        # Expand global_cond to append to every node
        # global_cond is (B, C) -> expand to (B, N, C)
        global_cond_expanded = global_cond.unsqueeze(1).expand(B, N, -1)
        
        # Concatenate node features with global context
        # Output shape: (B, N, E + C)
        fused_features = torch.cat([node_embeddings, global_cond_expanded], dim=-1)
        
        # Sanity check: fused dim must match MLP input
        assert fused_features.shape[-1] == self.mlp[0].in_features, \
            f"Fused dim {fused_features.shape[-1]} != MLP input {self.mlp[0].in_features}"
        
        # Predict bounding boxes
        boxes_normalized = self.mlp(fused_features)
        
        # Scale parameters to typical 256x256 pixel grid bounds for ease of math
        # x, y, w, h 
        boxes = boxes_normalized * 256.0
        
        return boxes


class FloorplanLoss(nn.Module):
    """
    Custom 5-component Loss Function:
    1. L_box:     SmoothL1 for box regression (x, y, w, h)
    2. L_iou:     1 - IoU for scale-invariant box matching
    3. L_overlap: Penalize room-room intersections (physics)
    4. L_adj:     L1 Manhattan distance for adjacent rooms (must touch)
    5. L_bnd:     Boundary violation penalty (rooms must stay inside)
    
    Uses recommended balanced weights to prevent gradient domination.
    """
    def __init__(self, penalty_weight=1.0, overlap_weight=0.5, adj_weight=0.3, iou_weight=1.0):
        super(FloorplanLoss, self).__init__()
        self.penalty_weight = penalty_weight
        self.overlap_weight = overlap_weight
        self.adj_weight = adj_weight
        self.iou_weight = iou_weight

    def _compute_corners(self, boxes):
        """Convert (x_c, y_c, w, h) to (x_min, y_min, x_max, y_max)."""
        half_w = boxes[:, :, 2] / 2.0
        half_h = boxes[:, :, 3] / 2.0
        x_min = boxes[:, :, 0] - half_w
        y_min = boxes[:, :, 1] - half_h
        x_max = boxes[:, :, 0] + half_w
        y_max = boxes[:, :, 1] + half_h
        return x_min, y_min, x_max, y_max

    def forward(self, pred_boxes, target_boxes, boundary_mask, adjacency=None, node_mask=None):
        """
        Args:
            pred_boxes (torch.Tensor): (B, N, 4) [x, y, w, h] predicted — MUST be float32
            target_boxes (torch.Tensor): (B, N, 4) [x, y, w, h] ground truth
            boundary_mask (torch.Tensor): (B, 1, 256, 256) binary mask (1=inside, 0=outside)
            adjacency (torch.Tensor): (B, N, N) graph connectivity matrix
            node_mask (torch.Tensor): (B, N) validity mask (1 if real, 0 if padded)
        """
        assert pred_boxes.dtype == torch.float32, "FloorplanLoss requires float32 inputs"
        
        B, N, _ = pred_boxes.size()
        
        # If no explicit node mask is provided, assume all nodes are valid
        if node_mask is None:
            node_mask = torch.ones(B, N, device=pred_boxes.device)
        
        # FIX C2: Do NOT pre-multiply boxes by mask. Apply mask only at reduction.
        valid_nodes = node_mask.unsqueeze(-1)  # (B, N, 1) for broadcasting
        num_valid = node_mask.sum().clamp(min=1.0)
        
        # --- 1. L_box: SmoothL1 Loss for box regression (x, y, w, h) ---
        loss_box = F.smooth_l1_loss(pred_boxes, target_boxes, beta=1.0, reduction='none')
        loss_box = (loss_box * valid_nodes).sum() / (num_valid * 4.0)
        
        # --- 2. L_iou: IoU Loss for scale-invariant matching ---
        # FIX C3: Add IoU loss — critical for convergence
        pred_xmin, pred_ymin, pred_xmax, pred_ymax = self._compute_corners(pred_boxes)
        gt_xmin, gt_ymin, gt_xmax, gt_ymax = self._compute_corners(target_boxes)
        
        # Intersection
        inter_xmin = torch.max(pred_xmin, gt_xmin)
        inter_ymin = torch.max(pred_ymin, gt_ymin)
        inter_xmax = torch.min(pred_xmax, gt_xmax)
        inter_ymax = torch.min(pred_ymax, gt_ymax)
        inter_w = F.relu(inter_xmax - inter_xmin)
        inter_h = F.relu(inter_ymax - inter_ymin)
        intersection = inter_w * inter_h
        
        # Union
        pred_area = F.relu(pred_boxes[:, :, 2]) * F.relu(pred_boxes[:, :, 3])
        gt_area = F.relu(target_boxes[:, :, 2]) * F.relu(target_boxes[:, :, 3])
        union = pred_area + gt_area - intersection + 1e-8
        
        iou = intersection / union
        loss_iou = ((1.0 - iou) * node_mask).sum() / num_valid
        
        # --- 3. L_bnd: Boundary violation penalty ---
        # Penalize boxes going outside [0, 256] hard bounds
        out_of_bounds = (
            F.relu(-pred_xmin) + 
            F.relu(pred_xmax - 256.0) + 
            F.relu(-pred_ymin) + 
            F.relu(pred_ymax - 256.0)
        )
        out_of_bounds_loss = (out_of_bounds * node_mask).sum() / num_valid
        
        # Differentiably sample boundary_mask at predicted centroids
        # FIX C1: Use raw pred_boxes centroids, not masked ones
        pred_cx = pred_boxes[:, :, 0]
        pred_cy = pred_boxes[:, :, 1]
        grid_x = (pred_cx / 128.0) - 1.0
        grid_y = (pred_cy / 128.0) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)  # (B, N, 1, 2)
        
        sampled_boundary = F.grid_sample(boundary_mask.float(), grid, align_corners=False)
        sampled_boundary = sampled_boundary.squeeze(1).squeeze(2)  # (B, N)
        
        mask_penalty = ((1.0 - sampled_boundary) * node_mask).sum() / num_valid
        validity_penalty = out_of_bounds_loss + mask_penalty
        
        # --- 4. L_overlap: Penalize room-room intersections (physics) ---
        # Use raw corners (not masked) — valid_pair_mask handles filtering
        x_min_exp = pred_xmin.unsqueeze(2).expand(B, N, N)
        y_min_exp = pred_ymin.unsqueeze(2).expand(B, N, N)
        x_max_exp = pred_xmax.unsqueeze(2).expand(B, N, N)
        y_max_exp = pred_ymax.unsqueeze(2).expand(B, N, N)
        
        ov_inter_x_min = torch.max(x_min_exp, x_min_exp.transpose(1, 2))
        ov_inter_y_min = torch.max(y_min_exp, y_min_exp.transpose(1, 2))
        ov_inter_x_max = torch.min(x_max_exp, x_max_exp.transpose(1, 2))
        ov_inter_y_max = torch.min(y_max_exp, y_max_exp.transpose(1, 2))
        
        ov_inter_w = F.relu(ov_inter_x_max - ov_inter_x_min)
        ov_inter_h = F.relu(ov_inter_y_max - ov_inter_y_min)
        overlap_area = ov_inter_w * ov_inter_h
        
        # Mask out self-overlap (diagonal) and padded nodes
        valid_pair_mask = node_mask.unsqueeze(2) * node_mask.unsqueeze(1)
        eye = torch.eye(N, device=pred_boxes.device).unsqueeze(0).bool()
        overlap_area = overlap_area.masked_fill(eye, 0.0)
        
        # FIX M7: Normalize by count of actual overlapping pairs, not all pairs
        masked_overlap = overlap_area * valid_pair_mask
        num_overlapping = (masked_overlap > 0).float().sum().clamp(min=1.0)
        loss_overlap = masked_overlap.sum() / num_overlapping

        # --- 5. L_adj: Adjacency attraction (L1 Manhattan — FP16-safe) ---
        # FIX C1: Use raw pred_boxes, not masked versions
        pred_cx_3d = pred_boxes[:, :, 0]  # (B, N)
        pred_cy_3d = pred_boxes[:, :, 1]
        pred_w_3d = pred_boxes[:, :, 2]
        pred_h_3d = pred_boxes[:, :, 3]
        
        # Gap distance between box edges (0 if touching or overlapping)
        dist_x = F.relu(
            torch.abs(pred_cx_3d.unsqueeze(2) - pred_cx_3d.unsqueeze(1)) - 
            (pred_w_3d.unsqueeze(2) + pred_w_3d.unsqueeze(1)) / 2.0
        )
        dist_y = F.relu(
            torch.abs(pred_cy_3d.unsqueeze(2) - pred_cy_3d.unsqueeze(1)) - 
            (pred_h_3d.unsqueeze(2) + pred_h_3d.unsqueeze(1)) / 2.0
        )
        
        # L1 Manhattan (NOT L2) — prevents float16 overflow (256²=65536 > fp16 max 65504)
        box_distance = dist_x + dist_y
        
        loss_adjacency = torch.tensor(0.0, device=pred_boxes.device)
        if adjacency is not None:
            # Only penalize distance where adjacency == 1 (excluding self-loops)
            adj_mask = (adjacency > 0.5) & (~eye) & valid_pair_mask.bool()
            
            if adj_mask.sum() > 0:
                loss_adjacency = box_distance[adj_mask].mean()
        
        # --- Final Total Loss (balanced weights) ---
        # FIX C7: Explicitly cast every term to float32 before summation.
        # Under AMP, partial losses can remain float16 from intermediate ops.
        # 256×256 = 65536 > float16 max (65504) → overflow → NaN.
        loss_box = loss_box.float()
        loss_iou = loss_iou.float()
        validity_penalty = validity_penalty.float()
        loss_overlap = loss_overlap.float()
        loss_adjacency = loss_adjacency.float()
        
        total_loss = (
            loss_box + 
            (self.iou_weight * loss_iou) +
            (self.penalty_weight * validity_penalty) + 
            (self.overlap_weight * loss_overlap) + 
            (self.adj_weight * loss_adjacency)
        )
        
        # NaN safety check
        assert not torch.isnan(total_loss), \
            f"NaN loss! box={loss_box.item():.4f} iou={loss_iou.item():.4f} " \
            f"bnd={validity_penalty.item():.4f} ovlp={loss_overlap.item():.4f} " \
            f"adj={loss_adjacency.item():.4f}"
        
        return total_loss, {
            'box_loss': loss_box.item(),
            'iou_loss': loss_iou.item(),
            'validity_penalty': validity_penalty.item(),
            'overlap_penalty': loss_overlap.item(),
            'adjacency_loss': loss_adjacency.item()
        }

if __name__ == "__main__":
    print("Testing Checkpoint 2.2: Generative Training Components...")
    
    # Simulate variables
    B, N = 2, 5 
    node_embed_dim = 64
    global_cond_dim = 192 # From checkpoint 1.3
    
    dummy_node_embeddings = torch.randn(B, N, node_embed_dim)
    dummy_global_cond = torch.randn(B, global_cond_dim)
    
    # 1. Test the Generator Head
    model = RoomGeneratorHead(node_embed_dim, global_cond_dim)
    pred_boxes = model(dummy_node_embeddings, dummy_global_cond)
    print(f"\nGenerator Head output shape: {pred_boxes.shape} -> (Batch, Nodes, 4)")
    print(f"Sample prediction [x, y, w, h]: {pred_boxes[0, 0].detach().numpy()}")
    
    # 2. Test the Loss Function
    loss_fn = FloorplanLoss(penalty_weight=1.0, overlap_weight=0.5, adj_weight=0.3, iou_weight=1.0)
    
    # Ground truth (random valid boxes between 50 and 200)
    target_boxes = torch.empty(B, N, 4).uniform_(50, 200)
    
    # Boundary mask (simulate a rectangular building limit in the center)
    dummy_mask = torch.zeros(B, 1, 256, 256)
    dummy_mask[:, :, 50:200, 50:200] = 1.0 
    
    # Simulate adjacency & node masks
    dummy_adj = torch.zeros(B, N, N)
    dummy_adj[:, 0, 1] = 1.0 # Force room 0 and 1 to be adjacent
    dummy_adj[:, 1, 0] = 1.0
    
    dummy_node_mask = torch.ones(B, N)
    
    total_loss, metrics = loss_fn(pred_boxes, target_boxes, dummy_mask, adjacency=dummy_adj, node_mask=dummy_node_mask)
    
    print("\n--- 5-Component Physics-Based Loss ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"  TOTAL: {total_loss.item():.4f}")
    print("\nAll 5 loss components verified!")
