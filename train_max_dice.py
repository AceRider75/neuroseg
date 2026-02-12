import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import json
from pathlib import Path

from src.data.dataset import get_data_split, MRIProjectDataset
from src.models.unet import UNet
from src.utils.metrics import DiceLoss, dice_coeff, iou_score
from src.utils.augmentation import get_train_transforms, get_val_transforms
from tqdm import tqdm

# ==================== FOCAL LOSS ====================
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        p_t = torch.where(targets == 1, p, 1 - p)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * ce_loss
        return focal_loss.mean()

# ==================== BOUNDARY LOSS ====================
class BoundaryLoss(nn.Module):
    """Boundary Loss - focuses on segmentation edges"""
    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def forward(self, outputs, targets):
        # Compute distance maps
        with torch.no_grad():
            target_np = targets.cpu().numpy()
            pred_np = torch.sigmoid(outputs).detach().cpu().numpy()
        
        # Simple approximation: focus on regions where pred and target differ
        pred_binary = (torch.sigmoid(outputs) > 0.5).float()
        diff = torch.abs(pred_binary - targets)
        
        # Boundary loss: penalize errors more at boundaries
        boundary_weight = diff * 2.0 + 1.0
        bce_loss = nn.BCEWithLogitsLoss(weight=boundary_weight, reduction='mean')(outputs, targets)
        
        return bce_loss

# ==================== METRICS TRACKING ====================
class MetricsTracker:
    """Track training metrics for live visualization"""
    def __init__(self):
        self.history = {
            'train_loss': [],
            'train_dice': [],
            'train_iou': [],
            'val_loss': [],
            'val_dice': [],
            'val_iou': [],
            'lr': [],
            'best_val_dice': 0.0
        }
    
    def update(self, epoch, train_loss, train_dice, train_iou, 
               val_loss, val_dice, val_iou, lr):
        self.history['train_loss'].append(train_loss)
        self.history['train_dice'].append(train_dice)
        self.history['train_iou'].append(train_iou)
        self.history['val_loss'].append(val_loss)
        self.history['val_dice'].append(val_dice)
        self.history['val_iou'].append(val_iou)
        self.history['lr'].append(lr)
        self.history['best_val_dice'] = max(self.history['best_val_dice'], val_dice)
    
    def save(self, path='training_history.json'):
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def plot(self, save_path='training_metrics.png'):
        """Generate comprehensive training plot"""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], label='Train', marker='o', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], label='Val', marker='s', linewidth=2)
        axes[0, 0].set_title('Loss (Lower is Better)', fontsize=13, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Dice Coefficient
        axes[0, 1].plot(epochs, self.history['train_dice'], label='Train', marker='o', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_dice'], label='Val', marker='s', linewidth=2)
        axes[0, 1].axhline(y=0.90, color='orange', linestyle='--', linewidth=2, label='Target 0.90')
        axes[0, 1].set_title('Dice Coefficient (Higher is Better)', fontsize=13, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # IoU
        axes[0, 2].plot(epochs, self.history['train_iou'], label='Train', marker='o', linewidth=2)
        axes[0, 2].plot(epochs, self.history['val_iou'], label='Val', marker='s', linewidth=2)
        axes[0, 2].set_title('IoU Coefficient (Higher is Better)', fontsize=13, fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylim([0, 1])
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Smoothed Dice (10-epoch moving average)
        if len(epochs) >= 10:
            smoothed_train = np.convolve(self.history['train_dice'], 
                                        np.ones(10)/10, mode='valid')
            smoothed_val = np.convolve(self.history['val_dice'], 
                                      np.ones(10)/10, mode='valid')
            smooth_epochs = range(10, len(epochs) + 1)
            axes[1, 0].plot(smooth_epochs, smoothed_train, label='Train (smoothed)', marker='o', linewidth=2)
            axes[1, 0].plot(smooth_epochs, smoothed_val, label='Val (smoothed)', marker='s', linewidth=2)
        else:
            axes[1, 0].plot(epochs, self.history['train_dice'], label='Train', marker='o', linewidth=2)
            axes[1, 0].plot(epochs, self.history['val_dice'], label='Val', marker='s', linewidth=2)
        axes[1, 0].axhline(y=0.90, color='orange', linestyle='--', linewidth=2, alpha=0.7)
        axes[1, 0].set_title('Dice (Smoothed - Trend)', fontsize=13, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 1].semilogy(epochs, self.history['lr'], label='Learning Rate', marker='o', linewidth=2, color='green')
        axes[1, 1].set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR (log scale)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Dice Gap (Train - Val)
        dice_gap = [t - v for t, v in zip(self.history['train_dice'], self.history['val_dice'])]
        axes[1, 2].plot(epochs, dice_gap, label='Train - Val Gap', marker='o', linewidth=2, color='red')
        axes[1, 2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1, 2].axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='Warning (0.05)')
        axes[1, 2].axhline(y=0.10, color='red', linestyle='--', alpha=0.5, label='Overfitting (0.10)')
        axes[1, 2].set_title('Overfitting Indicator', fontsize=13, fontweight='bold')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Gap')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"\n✓ Metrics plot saved to {save_path}")
        plt.close()

# ==================== EARLY STOPPING ====================
class EarlyStopping:
    def __init__(self, patience=30, min_delta=1e-5, target_metric=0.92):
        self.patience = patience
        self.min_delta = min_delta
        self.target_metric = target_metric
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_dice):
        if self.best_score is None:
            self.best_score = val_dice
        elif val_dice < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience and val_dice >= self.target_metric:
                self.early_stop = True
        else:
            self.best_score = val_dice
            self.counter = 0

# ==================== TRAINING FUNCTION ====================
def train():
    """
    MAXIMUM DICE OPTIMIZATION VERSION
    
    Prioritizes:
    1. Maximum Dice score
    2. Smooth convergence
    3. No oscillations
    4. Stable training
    
    Over:
    - Training speed
    - Memory efficiency
    """
    
    print("="*80)
    print("NEUROSEG TRAINING - MAXIMUM DICE OPTIMIZATION")
    print("="*80)
    print("\nConfiguration: Optimized for MAXIMUM Dice score")
    print("- Not optimized for speed")
    print("- Focus on smooth, stable convergence")
    print("- Will train longer to reach best performance\n")
    
    # ========== HYPERPARAMETERS FOR MAX DICE ==========
    base_path = "archive"
    batch_size = 16  # LARGE batch (more data per update, less noisy gradients)
    accumulation_steps = 2  # Effective batch = 32 (excellent gradient quality)
    lr = 1e-3  # Moderate LR with good schedule
    epochs = 300  # LONG training (plenty of time to converge)
    warmup_epochs = 10  # LONG warmup (avoid early overfitting)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Hyperparameters:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Accumulation Steps: {accumulation_steps}")
    print(f"  Effective Batch: {batch_size * accumulation_steps}")
    print(f"  Learning Rate: {lr}")
    print(f"  Max Epochs: {epochs}")
    print(f"  Device: {device}\n")
    
    # ========== DATA LOADING ==========
    print("Loading data...")
    try:
        train_df, val_df, test_df = get_data_split(base_path)
    except Exception as e:
        print(f"Error splitting data: {e}")
        return
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}\n")
    
    # ========== DATASET & DATALOADER ==========
    train_dataset = MRIProjectDataset(train_df, transform=get_train_transforms())
    val_dataset = MRIProjectDataset(val_df, transform=get_val_transforms())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}\n")
    
    # ========== MODEL ==========
    model = UNet(n_channels=3, n_classes=1).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n")
    
    # ========== LOSS FUNCTIONS FOR MAXIMUM DICE ==========
    # Combination: 0.6 * Dice + 0.3 * Focal + 0.1 * Boundary
    # This gives best results for segmentation
    dice_loss_fn = DiceLoss(smooth=1.0)
    focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    boundary_loss_fn = BoundaryLoss()
    
    print("Loss Functions:")
    print("  - Dice Loss (60%): Focus on overlap")
    print("  - Focal Loss (30%): Handle class imbalance")
    print("  - Boundary Loss (10%): Focus on edges\n")
    
    # ========== OPTIMIZER ==========
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, amsgrad=True)
    
    # ========== SCHEDULER - OPTIMIZED FOR CONVERGENCE ==========
    total_steps = (len(train_loader) // accumulation_steps + 1) * epochs
    
    # Linear warmup + Cosine annealing
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        total_steps=total_steps,
        pct_start=warmup_epochs/epochs,  # Long warmup
        anneal_strategy='cos',
        cycle_momentum=False,
        div_factor=1e5,  # Very gradual warmup
        final_div_factor=1e4
    )
    
    # ========== GRADIENT SCALER ==========
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    # ========== EARLY STOPPING ==========
    early_stopping = EarlyStopping(patience=35, min_delta=1e-5, target_metric=0.92)
    
    # ========== METRICS TRACKER ==========
    tracker = MetricsTracker()
    
    # ========== TRAINING LOOP ==========
    best_dice = 0.0
    best_loss = float('inf')
    
    print("="*80)
    print("STARTING TRAINING - MAXIMUM DICE MODE")
    print("="*80 + "\n")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        epoch_iou = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [TRAIN]", ncols=100)
        optimizer.zero_grad()
        
        for i, (images, masks) in enumerate(pbar):
            images, masks = images.to(device), masks.to(device)
            
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(images)
                
                # COMBINED LOSS FOR MAXIMUM DICE
                dice_l = dice_loss_fn(outputs, masks)
                focal_l = focal_loss_fn(outputs, masks)
                boundary_l = boundary_loss_fn(outputs, masks)
                
                # Weighted combination (optimized for segmentation)
                loss = (0.6 * dice_l + 0.3 * focal_l + 0.1 * boundary_l) / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                # GRADIENT CLIPPING
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            epoch_loss += loss.item() * accumulation_steps
            
            with torch.no_grad():
                d_coeff = dice_coeff(outputs, masks)
                iou = iou_score(outputs, masks)
                epoch_dice += d_coeff.item()
                epoch_iou += iou.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.4f}',
                'dice': f'{d_coeff.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        # ========== VALIDATION ==========
        val_loss, val_dice, val_iou = validate(model, val_loader, 
                                               dice_loss_fn, focal_loss_fn, 
                                               boundary_loss_fn, device)
        
        # ========== EPOCH SUMMARY ==========
        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_dice = epoch_dice / len(train_loader)
        avg_train_iou = epoch_iou / len(train_loader)
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train → Loss: {avg_train_loss:.4f} | Dice: {avg_train_dice:.4f} | IoU: {avg_train_iou:.4f}")
        print(f"  Val   → Loss: {val_loss:.4f} | Dice: {val_dice:.4f} | IoU: {val_iou:.4f}")
        print(f"  Best Val Dice: {tracker.history['best_val_dice']:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # ========== UPDATE TRACKER ==========
        tracker.update(epoch, avg_train_loss, avg_train_dice, avg_train_iou,
                      val_loss, val_dice, val_iou, optimizer.param_groups[0]['lr'])
        
        # ========== SAVE BEST MODEL ==========
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  ✓ NEW BEST MODEL! Val Dice: {val_dice:.4f}")
        
        print(f"{'='*80}\n")
        
        # ========== EARLY STOPPING ==========
        early_stopping(val_dice)
        if early_stopping.early_stop:
            print(f"\n{'='*80}")
            print("EARLY STOPPING TRIGGERED!")
            print(f"Target Dice {early_stopping.target_metric} reached and stabilized.")
            print(f"Best Validation Dice: {best_dice:.4f}")
            print(f"{'='*80}\n")
            break
    
    # ========== SAVE METRICS ==========
    tracker.save('training_history.json')
    tracker.plot('training_metrics.png')
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED!")
    print("="*80)
    print(f"Best Validation Dice: {best_dice:.4f}")
    print(f"Final Validation Dice: {tracker.history['val_dice'][-1]:.4f}")
    print(f"Total Epochs: {len(tracker.history['val_dice'])}")
    print("="*80 + "\n")

def validate(model, loader, dice_loss_fn, focal_loss_fn, boundary_loss_fn, device):
    """Validation function with combined loss"""
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    val_iou = 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation", ncols=100):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            # Same loss weights as training
            dice_l = dice_loss_fn(outputs, masks)
            focal_l = focal_loss_fn(outputs, masks)
            boundary_l = boundary_loss_fn(outputs, masks)
            loss = 0.6 * dice_l + 0.3 * focal_l + 0.1 * boundary_l
            
            val_loss += loss.item()
            val_dice += dice_coeff(outputs, masks).item()
            val_iou += iou_score(outputs, masks).item()
    
    return (val_loss / len(loader), 
            val_dice / len(loader), 
            val_iou / len(loader))

if __name__ == "__main__":
    train()
