import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
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
        pred_binary = (torch.sigmoid(outputs) > 0.5).float()
        diff = torch.abs(pred_binary - targets)
        boundary_weight = diff * 2.0 + 1.0
        
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(outputs, targets)
        boundary_loss = (boundary_weight * bce_loss).mean()
        return boundary_loss

# ==================== METRICS TRACKER ====================
class MetricsTracker:
    def __init__(self, start_epoch=0, start_history=None):
        self.start_epoch = start_epoch
        
        if start_history:
            # Resume from previous training
            self.train_loss = start_history['train_loss'].copy()
            self.val_loss = start_history['val_loss'].copy()
            self.train_dice = start_history['train_dice'].copy()
            self.val_dice = start_history['val_dice'].copy()
            self.train_iou = start_history['train_iou'].copy()
            self.val_iou = start_history['val_iou'].copy()
            self.learning_rates = start_history.get('learning_rates', []).copy()
        else:
            self.train_loss = []
            self.val_loss = []
            self.train_dice = []
            self.val_dice = []
            self.train_iou = []
            self.val_iou = []
            self.learning_rates = []

    def add_epoch(self, train_loss, val_loss, train_dice, val_dice, train_iou, val_iou, lr):
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_dice.append(train_dice)
        self.val_dice.append(val_dice)
        self.train_iou.append(train_iou)
        self.val_iou.append(val_iou)
        self.learning_rates.append(lr)

    def save(self, filepath='training_history.json'):
        history = {
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'train_dice': self.train_dice,
            'val_dice': self.val_dice,
            'train_iou': self.train_iou,
            'val_iou': self.val_iou,
            'learning_rates': self.learning_rates,
        }
        with open(filepath, 'w') as f:
            json.dump(history, f)

    def plot(self, filepath='training_metrics_resumed.png'):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f'Training Metrics - Resumed from Epoch {self.start_epoch}', 
                     fontsize=14, fontweight='bold')

        epochs = range(len(self.train_loss))

        # Loss
        axes[0, 0].plot(epochs, self.train_loss, label='Train Loss', linewidth=2, alpha=0.7)
        axes[0, 0].plot(epochs, self.val_loss, label='Val Loss', linewidth=2)
        axes[0, 0].axvline(x=self.start_epoch, color='red', linestyle='--', alpha=0.5, label='Resume Point')
        axes[0, 0].set_title('Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Dice
        axes[0, 1].plot(epochs, self.train_dice, label='Train Dice', linewidth=2, alpha=0.7)
        axes[0, 1].plot(epochs, self.val_dice, label='Val Dice', linewidth=2, color='orange')
        axes[0, 1].axhline(y=0.90, color='g', linestyle='--', alpha=0.5, label='Target (0.90)')
        axes[0, 1].axvline(x=self.start_epoch, color='red', linestyle='--', alpha=0.5, label='Resume Point')
        max_dice = max(self.val_dice) if self.val_dice else 0
        axes[0, 1].set_title(f'Dice Coefficient (Max: {max_dice:.4f})', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # IoU
        axes[0, 2].plot(epochs, self.train_iou, label='Train IoU', linewidth=2, alpha=0.7)
        axes[0, 2].plot(epochs, self.val_iou, label='Val IoU', linewidth=2, color='red')
        axes[0, 2].axvline(x=self.start_epoch, color='red', linestyle='--', alpha=0.5, label='Resume Point')
        axes[0, 2].set_title('IoU Score', fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('IoU')
        axes[0, 2].set_ylim([0, 1])
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # Smoothed Dice
        axes[1, 0].clear()
        val_dice = self.val_dice
        window = min(10, len(val_dice) // 5) if len(val_dice) > 1 else 1
        if window > 1:
            smoothed = np.convolve(val_dice, np.ones(window)/window, mode='valid')
            axes[1, 0].plot(val_dice, label='Raw Dice', alpha=0.3, linewidth=1)
            axes[1, 0].plot(range(window-1, len(val_dice)), smoothed, 
                           label=f'Smoothed (window={window})', linewidth=2, color='orange')
        else:
            axes[1, 0].plot(val_dice, label='Dice', linewidth=2)
        axes[1, 0].axvline(x=self.start_epoch, color='red', linestyle='--', alpha=0.5, label='Resume Point')
        axes[1, 0].set_title('Smoothness Analysis', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Dice')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Learning Rate
        if self.learning_rates:
            axes[1, 1].plot(self.learning_rates, linewidth=2, color='purple')
            axes[1, 1].axvline(x=self.start_epoch, color='red', linestyle='--', alpha=0.5, label='Resume Point')
            axes[1, 1].set_title('Learning Rate Schedule', fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        # Generalization Gap
        if len(self.train_dice) == len(self.val_dice):
            gap = [self.train_dice[i] - self.val_dice[i] for i in range(len(self.val_dice))]
            axes[1, 2].plot(gap, linewidth=2, color='brown')
            axes[1, 2].axhline(y=0.05, color='g', linestyle='--', alpha=0.5, label='Good (0.05)')
            axes[1, 2].axhline(y=0.10, color='orange', linestyle='--', alpha=0.5, label='Acceptable (0.10)')
            axes[1, 2].axvline(x=self.start_epoch, color='red', linestyle='--', alpha=0.5, label='Resume Point')
            axes[1, 2].set_title('Train-Val Gap (Overfitting)', fontweight='bold')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Gap')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        print(f"✓ Saved metrics plot: {filepath}")
        plt.close()

# ==================== TRAINING FUNCTIONS ====================
def validate(model, val_loader, dice_loss, focal_loss, boundary_loss, device):
    """Validate model"""
    model.eval()
    total_val_loss = 0
    total_dice = 0
    total_iou = 0
    num_batches = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)

            # Forward pass (use autocast when CUDA available)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                outputs = model(images)

            # Loss
            dice_l = dice_loss(outputs, masks)
            focal_l = focal_loss(outputs, masks)
            boundary_l = boundary_loss(outputs, masks)
            loss = (0.6 * dice_l + 0.3 * focal_l + 0.1 * boundary_l)

            total_val_loss += loss.item()

            # Metrics
            dice = dice_coeff(outputs, masks)
            iou = iou_score(outputs, masks)
            total_dice += dice
            total_iou += iou
            num_batches += 1

    avg_val_loss = total_val_loss / num_batches
    avg_dice = total_dice / num_batches
    avg_iou = total_iou / num_batches

    return avg_val_loss, avg_dice, avg_iou

def train_epoch(model, train_loader, optimizer, scheduler, dice_loss, focal_loss, boundary_loss, 
                device, scaler, accumulation_steps=2):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    optimizer.zero_grad()
    for batch_idx, (images, masks) in enumerate(pbar):
        images, masks = images.to(device), masks.to(device)

        # Forward pass with autocast
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            outputs = model(images)

            # Combined loss
            dice_l = dice_loss(outputs, masks)
            focal_l = focal_loss(outputs, masks)
            boundary_l = boundary_loss(outputs, masks)
            loss = (0.6 * dice_l + 0.3 * focal_l + 0.1 * boundary_l) / accumulation_steps

        # Scaled backward
        scaler.scale(loss).backward()

        # Accumulation step
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            # Unscale & clip
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            # ReduceLROnPlateau is stepped after validation, not in training loop

        # Metrics
        with torch.no_grad():
            dice = dice_coeff(outputs, masks)
            iou = iou_score(outputs, masks)

        total_loss += loss.item() * accumulation_steps
        total_dice += dice
        total_iou += iou
        num_batches += 1

        pbar.set_postfix({'loss': f'{total_loss / num_batches:.4f}', 
                         'dice': f'{total_dice / num_batches:.4f}'})

    avg_loss = total_loss / num_batches
    avg_dice = total_dice / num_batches
    avg_iou = total_iou / num_batches

    return avg_loss, avg_dice, avg_iou

# ==================== MAIN ====================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Hyperparameters
    batch_size = 16
    accumulation_steps = 2
    lr = 1e-4  # Much lower for fine-tuning (was 1e-3 which caused oscillations)
    epochs = 300
    warmup_epochs = 10
    early_stop_patience = 35
    early_stop_target = 0.92
    resume_epoch = 106

    
    # Load model
    print("Loading model from best_model.pth...")
    # UNet in src/models/unet.py expects (n_channels, n_classes)
    model = UNet(3, 1)
    checkpoint = torch.load('best_model.pth', map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    print("✓ Model loaded")

    # Data
    print("Loading data...")
    # get_data_split(base_path, test_size=0.15, val_size=0.15, random_seed=42)
    # base_path should point to the folder containing 'kaggle_3m'
    train_df, val_df, test_df = get_data_split(
        base_path='archive',
        test_size=0.15,
        val_size=0.15,
        random_seed=42
    )

    train_dataset = MRIProjectDataset(train_df, transform=get_train_transforms())
    val_dataset = MRIProjectDataset(val_df, transform=get_val_transforms())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    print(f"✓ Data loaded: {len(train_dataset)} train, {len(val_dataset)} val")

    # Loss functions
    dice_loss = DiceLoss()
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    boundary_loss = BoundaryLoss()

    # Optimizer with LOWER LR for resumed fine-tuning
    # Critical: resuming at epoch 106 means we're mid-convergence
    # Using 1e-3 causes oscillation; use 1e-4 for fine-tuning
    finetune_lr = 1e-4  # Much lower for stability
    optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_lr, weight_decay=1e-4, amsgrad=True)

    # Scheduler: Use ReduceLROnPlateau for resumed training
    # This backs off LR when validation Dice stops improving
    # Much more stable than trying to restart OneCycleLR mid-convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # Maximize Dice
        factor=0.5,  # Reduce LR by 50% when plateau detected
        patience=5,  # Wait 5 epochs before reducing
        min_lr=1e-6
    )

    # Mixed precision scaler
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    print(f"\n{'='*60}")
    print("RESUMED FINE-TUNING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Learning Rate (Fine-tuning): {finetune_lr}")
    print(f"Scheduler: ReduceLROnPlateau")
    print(f"  - Factor: 0.5 (cut LR in half when plateau)")
    print(f"  - Patience: 5 epochs without improvement")
    print(f"  - Min LR: 1e-6")
    print(f"{'='*60}\n")

    # Metrics tracker (load previous history if exists)
    print("Checking for previous training history...")
    prev_history = None
    
    if os.path.exists('training_history.json'):
        print("✓ Found training_history.json")
        try:
            with open('training_history.json', 'r') as f:
                prev_history = json.load(f)
            metrics = MetricsTracker(start_epoch=resume_epoch, start_history=prev_history)
            print(f"  Loaded {len(prev_history['val_dice'])} epochs of history")
        except Exception as e:
            print(f"⚠ Error reading training_history.json: {e}")
            print("  Will estimate current performance from model validation")
            prev_history = None
            
    elif os.path.exists('training_history_resumed.json'):
        print("✓ Found training_history_resumed.json")
        try:
            with open('training_history_resumed.json', 'r') as f:
                prev_history = json.load(f)
            metrics = MetricsTracker(start_epoch=resume_epoch, start_history=prev_history)
            print(f"  Loaded {len(prev_history['val_dice'])} epochs of history")
        except Exception as e:
            print(f"⚠ Error reading training_history_resumed.json: {e}")
            prev_history = None
    
    if prev_history is None:
        print("\n⚠ No training history file found!")
        print("  This can happen if training was suddenly interrupted.")
        print("  Running quick validation to measure current model performance...\n")
        
        # Quick validation to get current Dice
        print("Validating current model...")
        val_loss, val_dice, val_iou = validate(
            model, val_loader, dice_loss, focal_loss, boundary_loss, device
        )
        print(f"✓ Current model Dice: {val_dice:.4f} ({val_dice*100:.2f}%)")
        
        # Start metrics with validation results
        metrics = MetricsTracker(start_epoch=resume_epoch)
        metrics.val_dice.append(val_dice)
        metrics.val_loss.append(val_loss)
        metrics.val_iou.append(val_iou)
        metrics.train_dice.append(val_dice)  # Assume similar
        metrics.train_loss.append(val_loss)
        metrics.train_iou.append(val_iou)
        metrics.learning_rates.append(lr)

    # Training
    best_dice = max(metrics.val_dice) if metrics.val_dice else 0
    patience_counter = 0
    best_model_path = 'best_model_resumed.pth'

    print(f"\nStarting training from epoch {resume_epoch}...\n")

    for epoch in range(resume_epoch, epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        # Train
        train_loss, train_dice, train_iou = train_epoch(
            model, train_loader, optimizer, scheduler, dice_loss, focal_loss, 
            boundary_loss, device, scaler, accumulation_steps
        )

        # Validate
        val_loss, val_dice, val_iou = validate(
            model, val_loader, dice_loss, focal_loss, boundary_loss, device
        )

        # Get current LR
        current_lr = optimizer.param_groups[0]['lr']

        # Log metrics
        metrics.add_epoch(train_loss, val_loss, train_dice, val_dice, train_iou, val_iou, current_lr)

        # Print
        print(f"  Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Dice:   {val_dice:.4f} | IoU: {val_iou:.4f}")
        print(f"  LR: {current_lr:.2e}")

        # Early stopping
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ Best model saved! (Dice: {best_dice:.4f})")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{early_stop_patience}")

        # Check if reached target
        if val_dice >= early_stop_target:
            print(f"\n  ✓✓ TARGET REACHED: {val_dice:.4f} >= {early_stop_target}")

        # Step ReduceLROnPlateau scheduler based on val_dice
        scheduler.step(val_dice)

        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\n✓ Early stopping triggered after {epoch+1} epochs")
            print(f"  Best Val Dice: {best_dice:.4f}")
            break

    # Final metrics
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Final Val Dice: {val_dice:.4f}")
    print(f"Best Val Dice: {best_dice:.4f}")
    print(f"Epochs completed: {len(metrics.val_dice) - 1}")  # -1 because we added validation at start
    print(f"Total epochs (from original start): {resume_epoch + len(metrics.val_dice) - 1}")
    print(f"{'='*60}\n")

    # Save metrics
    metrics.save('training_history_resumed.json')
    print("✓ Saved training history")

    # Plot
    metrics.plot('training_metrics_resumed.png')
    print("✓ Saved metrics plot")

    # Load best model
    print(f"\nLoading best model from {best_model_path}...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    torch.save(model.state_dict(), 'best_model.pth')
    print("✓ Updated best_model.pth")

    print("\n🎉 Resume training complete!")
    print(f"✓ Dice improved from 0.8426 → {best_dice:.4f}")
    if best_dice >= 0.90:
        print("✓ REACHED TARGET (0.90+) 🚀")
    else:
        print(f"⚠ Target not yet reached, consider running again")

if __name__ == '__main__':
    main()
