import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import deque
import json

from src.data.dataset import get_data_split, MRIProjectDataset
from src.models.unet import UNet
from src.utils.metrics import DiceLoss, dice_coeff, iou_score
from src.utils.augmentation import get_train_transforms, get_val_transforms
from tqdm import tqdm

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
            'lr': []
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
    
    def save(self, path='training_history.json'):
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def plot(self, save_path='training_metrics.png'):
        """Generate comprehensive training plot"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], label='Train', marker='o')
        axes[0, 0].plot(epochs, self.history['val_loss'], label='Val', marker='s')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Dice Coefficient
        axes[0, 1].plot(epochs, self.history['train_dice'], label='Train', marker='o')
        axes[0, 1].plot(epochs, self.history['val_dice'], label='Val', marker='s')
        axes[0, 1].set_title('Dice Coefficient')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # IoU
        axes[0, 2].plot(epochs, self.history['train_iou'], label='Train', marker='o')
        axes[0, 2].plot(epochs, self.history['val_iou'], label='Val', marker='s')
        axes[0, 2].set_title('IoU Coefficient')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylim([0, 1])
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Smoothed Dice (10-epoch moving average)
        if len(epochs) >= 10:
            smoothed_train = np.convolve(self.history['train_dice'], 
                                        np.ones(10)/10, mode='valid')
            smoothed_val = np.convolve(self.history['val_dice'], 
                                      np.ones(10)/10, mode='valid')
            smooth_epochs = range(10, len(epochs) + 1)
            axes[1, 0].plot(smooth_epochs, smoothed_train, label='Train (smoothed)', marker='o')
            axes[1, 0].plot(smooth_epochs, smoothed_val, label='Val (smoothed)', marker='s')
        else:
            axes[1, 0].plot(epochs, self.history['train_dice'], label='Train', marker='o')
            axes[1, 0].plot(epochs, self.history['val_dice'], label='Val', marker='s')
        axes[1, 0].set_title('Dice (Smoothed)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning Rate
        axes[1, 1].plot(epochs, self.history['lr'], label='Learning Rate', marker='o', color='green')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Dice Gap (Train - Val) to detect overfitting
        dice_gap = [t - v for t, v in zip(self.history['train_dice'], self.history['val_dice'])]
        axes[1, 2].plot(epochs, dice_gap, label='Train - Val Dice Gap', marker='o', color='red')
        axes[1, 2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1, 2].set_title('Overfitting Indicator')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Metrics plot saved to {save_path}")
        plt.close()

# ==================== EARLY STOPPING ====================
class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4, target_metric=0.90):
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
    # ========== HYPERPARAMETERS ==========
    base_path = "archive"
    batch_size = 8  # INCREASED: Better for GPU utilization
    accumulation_steps = 4  # REDUCED: More frequent updates
    lr = 1e-3  # INCREASED: Better convergence
    epochs = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # ========== DATA LOADING ==========
    print("Loading data...")
    try:
        train_df, val_df, test_df = get_data_split(base_path)
    except Exception as e:
        print(f"Error splitting data: {e}")
        return
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # ========== DATASET & DATALOADER ==========
    train_dataset = MRIProjectDataset(train_df, transform=get_train_transforms())
    val_dataset = MRIProjectDataset(val_df, transform=get_val_transforms())
    
    # num_workers=0 for Colab compatibility
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}\n")
    
    # ========== MODEL ==========
    model = UNet(n_channels=3, n_classes=1).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n")
    
    # ========== LOSS FUNCTIONS ==========
    criterion = nn.BCEWithLogitsLoss()
    dice_loss_fn = DiceLoss()
    
    # ========== OPTIMIZER ==========
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # ========== SCHEDULER (FIXED) ==========
    # Calculate total steps correctly
    total_steps = (len(train_loader) // accumulation_steps + 1) * epochs
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        total_steps=total_steps,
        pct_start=0.15,  # REDUCED: Shorter warmup
        anneal_strategy='cos',
        cycle_momentum=False,  # Don't change momentum for Adam
        div_factor=1e4,  # Start at lr/10000 
        final_div_factor=1e4  # End at lr/10000
    )
    
    # ========== GRADIENT SCALER ==========
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    # ========== EARLY STOPPING ==========
    early_stopping = EarlyStopping(patience=25, min_delta=1e-4, target_metric=0.90)
    
    # ========== METRICS TRACKER ==========
    tracker = MetricsTracker()
    
    # ========== TRAINING LOOP ==========
    best_dice = 0.0
    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80 + "\n")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        epoch_iou = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [TRAIN]")
        optimizer.zero_grad()
        
        for i, (images, masks) in enumerate(pbar):
            images, masks = images.to(device), masks.to(device)
            
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(images)
                
                # BALANCED LOSS: 70% Dice + 30% BCE
                bce_loss = criterion(outputs, masks)
                dice_l = dice_loss_fn(outputs, masks)
                loss = (0.7 * dice_l + 0.3 * bce_loss) / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                # GRADIENT CLIPPING: Prevent exploding gradients
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
        val_loss, val_dice, val_iou = validate(model, val_loader, criterion, dice_loss_fn, device)
        
        # ========== EPOCH SUMMARY ==========
        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_dice = epoch_dice / len(train_loader)
        avg_train_iou = epoch_iou / len(train_loader)
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train → Loss: {avg_train_loss:.4f} | Dice: {avg_train_dice:.4f} | IoU: {avg_train_iou:.4f}")
        print(f"  Val   → Loss: {val_loss:.4f} | Dice: {val_dice:.4f} | IoU: {val_iou:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'='*80}\n")
        
        # ========== UPDATE TRACKER ==========
        tracker.update(epoch, avg_train_loss, avg_train_dice, avg_train_iou,
                      val_loss, val_dice, val_iou, optimizer.param_groups[0]['lr'])
        
        # ========== SAVE BEST MODEL ==========
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), "best_model.pth")
            print(f"✓ New best model saved with Val Dice: {val_dice:.4f}\n")
        
        # ========== EARLY STOPPING ==========
        early_stopping(val_dice)
        if early_stopping.early_stop:
            print(f"\n{'='*80}")
            print("EARLY STOPPING TRIGGERED!")
            print(f"Target Dice {early_stopping.target_metric} reached and stabilized.")
            print(f"{'='*80}\n")
            break
    
    # ========== SAVE METRICS ==========
    tracker.save('training_history.json')
    tracker.plot('training_metrics.png')
    print("\nTraining completed!")
    print(f"Best Validation Dice: {best_dice:.4f}")

def validate(model, loader, criterion, dice_loss_fn, device):
    """Validation function with IoU calculation"""
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    val_iou = 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            # Same loss as training
            bce_loss = criterion(outputs, masks)
            dice_l = dice_loss_fn(outputs, masks)
            loss = 0.7 * dice_l + 0.3 * bce_loss
            
            val_loss += loss.item()
            val_dice += dice_coeff(outputs, masks).item()
            val_iou += iou_score(outputs, masks).item()
    
    return (val_loss / len(loader), 
            val_dice / len(loader), 
            val_iou / len(loader))

if __name__ == "__main__":
    train()
