"""
NeuroSeg Optimized Training Script
===================================
Targets: ≥0.90 Dice within 6 hours on Colab T4 GPU.

Key optimizations over train_max_dice.py:
1. Attention U-Net with dropout (better feature focusing)
2. CosineAnnealingWarmRestarts (escapes local minima via restarts)
3. EMA model averaging (smoother validation, better model selection)
4. Simplified loss: 0.5*Dice + 0.3*BCE + 0.2*Focal (dropped broken BoundaryLoss)
5. Batch size 8/accum 2 (more updates per wall-clock hour)
6. Enhanced augmentation (9 transforms vs 4)
7. Hard-thresholded Dice for validation (accurate reporting)
"""

import os
import copy
import math
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.data.dataset import get_data_split, MRIProjectDataset
from src.models.unet import UNet
from src.utils.metrics import DiceLoss, dice_coeff, dice_coeff_hard, iou_score
from src.utils.augmentation import get_train_transforms, get_val_transforms


# ==================== FOCAL LOSS ====================
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in segmentation."""
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


# ==================== EMA MODEL ====================
class EMAModel:
    """Exponential Moving Average of model weights.
    
    Maintains a shadow copy: θ_ema = decay * θ_ema + (1-decay) * θ_current
    Produces smoother, more stable models for validation.
    Typically adds 0.5-1% to validation dice.
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def apply_shadow(self, model):
        """Replace model weights with EMA weights for evaluation."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        """Restore original model weights after evaluation."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


# ==================== METRICS TRACKER ====================
class MetricsTracker:
    """Track and save training metrics for live visualization."""
    def __init__(self):
        self.history = {
            'train_loss': [],
            'train_dice': [],
            'train_iou': [],
            'val_loss': [],
            'val_dice': [],
            'val_iou': [],
            'learning_rates': [],
            'best_val_dice': 0.0
        }

    def update(self, train_loss, train_dice, train_iou,
               val_loss, val_dice, val_iou, lr):
        self.history['train_loss'].append(train_loss)
        self.history['train_dice'].append(train_dice)
        self.history['train_iou'].append(train_iou)
        self.history['val_loss'].append(val_loss)
        self.history['val_dice'].append(val_dice)
        self.history['val_iou'].append(val_iou)
        self.history['learning_rates'].append(lr)
        self.history['best_val_dice'] = max(self.history['best_val_dice'], val_dice)

    def save(self, path='training_history.json'):
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def plot(self, save_path='training_metrics.png'):
        """Generate comprehensive training plot."""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        epochs = range(1, len(self.history['train_loss']) + 1)

        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], label='Val', linewidth=2)
        axes[0, 0].set_title('Loss', fontsize=13, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Dice
        axes[0, 1].plot(epochs, self.history['train_dice'], label='Train', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_dice'], label='Val', linewidth=2)
        axes[0, 1].axhline(y=0.90, color='orange', linestyle='--', linewidth=2, label='Target 0.90')
        best = max(self.history['val_dice'])
        axes[0, 1].set_title(f'Dice (Best: {best:.4f})', fontsize=13, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # IoU
        axes[0, 2].plot(epochs, self.history['train_iou'], label='Train', linewidth=2)
        axes[0, 2].plot(epochs, self.history['val_iou'], label='Val', linewidth=2)
        axes[0, 2].set_title('IoU', fontsize=13, fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylim([0, 1])
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # Smoothed Dice
        if len(epochs) >= 10:
            window = min(10, len(epochs) // 3)
            smoothed = np.convolve(self.history['val_dice'],
                                   np.ones(window) / window, mode='valid')
            axes[1, 0].plot(self.history['val_dice'], alpha=0.3, linewidth=1, label='Raw')
            axes[1, 0].plot(range(window - 1, len(epochs)), smoothed,
                           linewidth=2, label=f'Smoothed (w={window})')
        else:
            axes[1, 0].plot(epochs, self.history['val_dice'], linewidth=2, label='Val Dice')
        axes[1, 0].axhline(y=0.90, color='orange', linestyle='--', linewidth=2, alpha=0.7)
        axes[1, 0].set_title('Dice Trend (Smoothed)', fontsize=13, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Learning Rate
        axes[1, 1].semilogy(epochs, self.history['learning_rates'], linewidth=2, color='green')
        axes[1, 1].set_title('Learning Rate', fontsize=13, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR (log)')
        axes[1, 1].grid(True, alpha=0.3)

        # Overfitting gap
        gap = [t - v for t, v in zip(self.history['train_dice'], self.history['val_dice'])]
        axes[1, 2].plot(epochs, gap, linewidth=2, color='red')
        axes[1, 2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1, 2].axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='Warning')
        axes[1, 2].axhline(y=0.10, color='red', linestyle='--', alpha=0.5, label='Overfitting')
        axes[1, 2].set_title('Train-Val Gap', fontsize=13, fontweight='bold')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"\n✓ Metrics plot saved to {save_path}")
        plt.close()


# ==================== EARLY STOPPING ====================
class EarlyStopping:
    def __init__(self, patience=25, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_dice):
        if self.best_score is None:
            self.best_score = val_dice
        elif val_dice < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_dice
            self.counter = 0


# ==================== LINEAR WARMUP SCHEDULER ====================
class LinearWarmupScheduler:
    """Linear warmup for the first N epochs, then delegates to base scheduler."""
    def __init__(self, optimizer, warmup_epochs, base_scheduler, warmup_start_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        self.warmup_start_lr = warmup_start_lr
        self.target_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0

    def step(self, epoch=None):
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.current_epoch / self.warmup_epochs
            lr = self.warmup_start_lr + alpha * (self.target_lr - self.warmup_start_lr)
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
        else:
            self.base_scheduler.step()

    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


# ==================== VALIDATION ====================
def validate(model, loader, dice_loss_fn, bce_loss_fn, focal_loss_fn, device):
    """Validation with hard-thresholded dice for accurate reporting."""
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    val_iou = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(images)

            dice_l = dice_loss_fn(outputs, masks)
            bce_l = bce_loss_fn(outputs, masks)
            focal_l = focal_loss_fn(outputs, masks)
            loss = 0.5 * dice_l + 0.3 * bce_l + 0.2 * focal_l

            val_loss += loss.item()
            # Use hard-thresholded dice for accurate validation
            val_dice += dice_coeff_hard(outputs, masks).item()
            val_iou += iou_score(outputs, masks).item()

    n = len(loader)
    return val_loss / n, val_dice / n, val_iou / n


# ==================== MAIN TRAINING ====================
def train():
    print("=" * 80)
    print("NEUROSEG OPTIMIZED TRAINING — TARGET: 0.90+ DICE IN ≤6H")
    print("=" * 80)

    # ========== HYPERPARAMETERS ==========
    base_path = "archive"
    batch_size = 8
    accumulation_steps = 2   # Effective batch = 16
    lr = 3e-4                # Start at reasonable LR immediately
    epochs = 200
    warmup_epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nConfiguration:")
    print(f"  Batch Size: {batch_size} (effective: {batch_size * accumulation_steps})")
    print(f"  Learning Rate: {lr}")
    print(f"  Scheduler: CosineAnnealingWarmRestarts (T0=30, Tmult=2)")
    print(f"  Warmup: {warmup_epochs} epochs (linear)")
    print(f"  Loss: 0.5·Dice + 0.3·BCE + 0.2·Focal")
    print(f"  EMA: decay=0.999")
    print(f"  Epochs: {epochs} (early stopping patience=25)")
    print(f"  Device: {device}\n")

    # ========== DATA ==========
    print("Loading data...")
    train_df, val_df, test_df = get_data_split(base_path)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}\n")

    train_dataset = MRIProjectDataset(train_df, transform=get_train_transforms())
    val_dataset = MRIProjectDataset(val_df, transform=get_val_transforms())

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}\n")

    # ========== MODEL ==========
    model = UNet(n_channels=3, n_classes=1).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Attention U-Net parameters: {total_params:,}\n")

    # ========== LOSS FUNCTIONS ==========
    dice_loss_fn = DiceLoss(smooth=1.0)
    bce_loss_fn = nn.BCEWithLogitsLoss()
    focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

    print("Loss: 0.5·Dice + 0.3·BCE + 0.2·Focal")
    print("  (dropped BoundaryLoss — was non-differentiable)\n")

    # ========== OPTIMIZER ==========
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   weight_decay=5e-4, amsgrad=True)

    # ========== SCHEDULER ==========
    # CosineAnnealingWarmRestarts: T_0=30 means first restart after 30 epochs,
    # T_mult=2 doubles the period each restart (30, 60, 120 epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-6
    )
    scheduler = LinearWarmupScheduler(optimizer, warmup_epochs, cosine_scheduler,
                                       warmup_start_lr=1e-6)

    # ========== MIXED PRECISION ==========
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # ========== EMA ==========
    ema = EMAModel(model, decay=0.999)

    # ========== EARLY STOPPING ==========
    early_stopping = EarlyStopping(patience=25, min_delta=1e-4)

    # ========== TRACKER ==========
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

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [TRAIN]", ncols=110)
        optimizer.zero_grad()

        for i, (images, masks) in enumerate(pbar):
            images, masks = images.to(device), masks.to(device)

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(images)

                # Combined loss
                dice_l = dice_loss_fn(outputs, masks)
                bce_l = bce_loss_fn(outputs, masks)
                focal_l = focal_loss_fn(outputs, masks)
                loss = (0.5 * dice_l + 0.3 * bce_l + 0.2 * focal_l) / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # Update EMA after each optimizer step
                ema.update(model)

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

        # Step scheduler (epoch-level)
        scheduler.step()

        # ========== VALIDATION WITH EMA ==========
        ema.apply_shadow(model)
        val_loss, val_dice, val_iou = validate(
            model, val_loader, dice_loss_fn, bce_loss_fn, focal_loss_fn, device
        )
        ema.restore(model)

        # ========== EPOCH SUMMARY ==========
        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_dice = epoch_dice / len(train_loader)
        avg_train_iou = epoch_iou / len(train_loader)

        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train → Loss: {avg_train_loss:.4f} | Dice: {avg_train_dice:.4f} | IoU: {avg_train_iou:.4f}")
        print(f"  Val   → Loss: {val_loss:.4f} | Dice: {val_dice:.4f} | IoU: {val_iou:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e} | Best: {best_dice:.4f}")

        # ========== UPDATE TRACKER ==========
        tracker.update(avg_train_loss, avg_train_dice, avg_train_iou,
                       val_loss, val_dice, val_iou, optimizer.param_groups[0]['lr'])

        # Save history every epoch for live monitoring
        tracker.save('training_history.json')

        # ========== SAVE BEST MODEL ==========
        if val_dice > best_dice:
            best_dice = val_dice
            # Save EMA weights as the best model
            ema.apply_shadow(model)
            torch.save(model.state_dict(), "best_model.pth")
            ema.restore(model)
            print(f"  ✓ NEW BEST MODEL! Val Dice: {val_dice:.4f}")

            if val_dice >= 0.90:
                print(f"  🎯 TARGET REACHED! Dice = {val_dice:.4f} ≥ 0.90")

        print(f"{'='*80}\n")

        # ========== EARLY STOPPING ==========
        early_stopping(val_dice)
        if early_stopping.early_stop:
            print(f"\n{'='*80}")
            print(f"EARLY STOPPING at epoch {epoch+1}")
            print(f"Best Dice: {best_dice:.4f} (no improvement for {early_stopping.patience} epochs)")
            print(f"{'='*80}\n")
            break

    # ========== FINAL SUMMARY ==========
    tracker.plot('training_metrics.png')

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED!")
    print("=" * 80)
    print(f"Best Validation Dice: {best_dice:.4f}")
    print(f"Final Validation Dice: {tracker.history['val_dice'][-1]:.4f}")
    print(f"Total Epochs: {len(tracker.history['val_dice'])}")
    if best_dice >= 0.90:
        print("🎯 TARGET MET: Dice ≥ 0.90!")
    else:
        print(f"⚠ Target not met. Gap: {0.90 - best_dice:.4f}")
        print("  Consider resuming training with resume_training.py")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    train()
