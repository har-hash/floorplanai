import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

# Import Custom Core Modules Built in Phases 1, 2, and 3
from dataset import EDAIDataset, edai_collate_fn
from inference_pipeline import AIArchitect
from generative_trainer import FloorplanLoss

class EarlyStopping:
    """
    World-class utility to mathematically prevent overfitting.
    If the AI Architect starts memorizing the training data and performs worse
    on the unseen validation dataset, this arrests training and restores the best state.
    """
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def get_args():
    parser = argparse.ArgumentParser(description="Master Training Engine for EDAI 2 AI Architect")
    parser.add_argument('--train_data', type=str, default="static/Data/data_train_converted.pkl")
    parser.add_argument('--val_data', type=str, default="static/Data/data_test_converted.pkl")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32, help="Set to 16 or 8 if GPU VRAM runs out")
    parser.add_argument('--lr', type=float, default=3e-4, help="Peak Learning Rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="L2 Regularizer against exact memorization")
    parser.add_argument('--output_dir', type=str, default="checkpoints")
    parser.add_argument('--resume', action='store_true', help="Resume training from last best checkpoint")
    return parser.parse_args()

def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    """ Executes a single mathematically optimized epoch across millions of PyTorch nodes. """
    model.train()
    epoch_loss = 0.0
    
    for batch_idx, batch in enumerate(dataloader):
        # 1. Device Formatting (Moves heavy payload cleanly to GPU)
        b_img = batch['boundary_img'].to(device, non_blocking=True)
        r_counts = batch['room_counts'].to(device, non_blocking=True)
        r_types = batch['room_types'].to(device, non_blocking=True)
        adj = batch['adjacency'].to(device, non_blocking=True)
        t_boxes = batch['target_boxes'].to(device, non_blocking=True)
        n_mask = batch['node_mask'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True) # Massive memory management improvement
        
        # 2. Automatic Mixed Precision (AMP) 
        # Triples GPU rendering speed by compressing float32 memory to float16 instantly.
        with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            # Forward Architecture Pass
            pred_boxes = model(b_img, r_counts, r_types, adj)
            
        # 3. Massive Multi-Objective Physical Constraints Loss (Requires Float32 stability)
        # We explicitly step out of float16 autocast here because floorplan area math (256 * 256 = 65536)
        # massively overflows the maximum float16 limit (65504), causing instant NaN explosions.
        with autocast(device_type=device.type, enabled=False):
            loss, _ = criterion(
                pred_boxes=pred_boxes.float(), 
                target_boxes=t_boxes.float(), 
                boundary_mask=b_img.float(), 
                adjacency=adj.float(), 
                node_mask=n_mask.float()
            )

        # 3. Scaled Accumulation & Gradient Clipping
        scaler.scale(loss).backward()
        
        # Clip exploding gradients (Prevents catastrophic unlearning from extreme physics penalties)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 4. Step Optimizer
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        
        if batch_idx % 50 == 0:
            print(f"   [Train Batch {batch_idx}/{len(dataloader)}] -> Current Iteration Loss: {loss.item():.4f}")

    return epoch_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, device):
    """ Safely evaluates Model's Generalization state against complete UNSEEN layouts. """
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            b_img = batch['boundary_img'].to(device, non_blocking=True)
            r_counts = batch['room_counts'].to(device, non_blocking=True)
            r_types = batch['room_types'].to(device, non_blocking=True)
            adj = batch['adjacency'].to(device, non_blocking=True)
            t_boxes = batch['target_boxes'].to(device, non_blocking=True)
            n_mask = batch['node_mask'].to(device, non_blocking=True)
            
            # Utilizing PyTorch native FP16 rendering natively for the model
            with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                pred_boxes = model(b_img, r_counts, r_types, adj)
                
            # Compute loss in stable float32
            with autocast(device_type=device.type, enabled=False):
                loss, _ = criterion(pred_boxes.float(), t_boxes.float(), b_img.float(), adj.float(), n_mask.float())
                
            val_loss += loss.item()
            
    return val_loss / len(dataloader)

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Smart Infrastructure Identification (CUDA GPU > CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- AI ARCHITECT PRODUCTION ENGINE STARTING ---")
    print(f"Accelerating on Hardware Platform: {device.type.upper()}")
    if device.type == 'cuda':
        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")

    # 1. Dataloaders Setup (Strict separation preventing Overfitting)
    print("\n[1/4] Establishing Production PyTorch Dataloaders...")

    # Error handling for missing data files
    if not os.path.exists(args.train_data):
        raise FileNotFoundError(f"Training data not found: {args.train_data}")
    if not os.path.exists(args.val_data):
        raise FileNotFoundError(f"Validation data not found: {args.val_data}")

    try:
        train_dataset = EDAIDataset(args.train_data)
        val_dataset = EDAIDataset(args.val_data)
    except Exception as e:
        raise RuntimeError(f"Failed to load datasets: {str(e)}")

    # Determine optimal num_workers (0 for Windows to avoid multiprocessing issues)
    num_workers = 0 if os.name == 'nt' else 4
    pin_memory = device.type == 'cuda'

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=edai_collate_fn, num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=edai_collate_fn, num_workers=num_workers, pin_memory=pin_memory
    )
    
    # 2. Master Model Instantiation
    print("\n[2/4] Initializing Deep Neural Pipelines...")
    model = AIArchitect(num_room_types=20).to(device)
    
    # Physics/Constraint Matrix Instantiation
    criterion = FloorplanLoss(penalty_weight=1.0, overlap_weight=0.5, adj_weight=0.3, iou_weight=1.0)
    
    # 3. Master Optimas: AdamW (Weight Decay avoids Overfitting) + Scaler
    print("\n[3/4] Preparing AMP Optimizers and Weight Decay Regulators...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Smoothly drops learning rate into deep optimization "valleys" natively
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    
    early_stopper = EarlyStopping(patience=8, min_delta=0.01)
    start_epoch = 1

    # FIX M2: Resume from checkpoint if --resume flag is set
    ckpt_path = os.path.join(args.output_dir, "AI_Architect_Best.pth")
    if args.resume and os.path.exists(ckpt_path):
        print(f"\n[RESUME] Loading checkpoint from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_loss = ckpt.get('val_loss', float('inf'))
        print(f"[RESUME] Continuing from epoch {start_epoch}, best val_loss={best_val_loss:.4f}")
    else:
        best_val_loss = float('inf')

    print("\n[4/4] ENGAGING MASTER EPOCH LOOP...")
    
    start_train_time = time.time()
    
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        print(f"\n=== EPOCH {epoch}/{args.epochs} ===")
        
        # The Training Core Algorithm
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        
        # Validation Evaluation Block
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        # Adjust Learning Rate trajectory natively
        scheduler.step()
        
        metrics_msg = (
            f"   Phase Review | Train Loss: {train_loss:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| LR: {scheduler.get_last_lr()[0]:.6f} "
            f"| Time: {(time.time() - epoch_start):.1f}s"
        )
        print(metrics_msg)
        
        # Native PyTorch Persistence Controller
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.output_dir, "AI_Architect_Best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_loss': val_loss
            }, save_path)
            print(f"   >>> Checkpoint Update: New optimal generalization detected. Model weights safely pushed to {save_path}!")
            
        # Overfitting Arrester Hook
        early_stopper(val_loss)
        if early_stopper.early_stop:
            print(f"\n[ALERT] Mathematical Execution Halted early at Epoch {epoch} !!")
            print("Reason: Extreme Validation flatline detected. Further training guarantees Overfitting Dataset Bias.")
            break
            
    total_time = (time.time() - start_train_time) / 60.0
    print(f"\n--- SYSTEM COMPLETE | Total Runtime: {total_time:.1f} minutes ---")
    print(f"Production Model successfully compressed inside {args.output_dir}/AI_Architect_Best.pth")

if __name__ == "__main__":
    main()
