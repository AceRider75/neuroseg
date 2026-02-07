import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data.dataset import get_data_split, MRIProjectDataset
from src.models.unet import UNet
from src.utils.metrics import DiceLoss, dice_coeff
from src.utils.augmentation import get_train_transforms, get_val_transforms
from tqdm import tqdm

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0, target_metric=0.85):
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

def train():
    # Parameters
    base_path = "archive"
    batch_size = 4 
    accumulation_steps = 8 # Effective batch size = 32
    lr = 3e-4 
    epochs = 100 # Increased for better convergence
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data Split
    try:
        train_df, val_df, test_df = get_data_split(base_path)
    except Exception as e:
        print(f"Error splitting data: {e}")
        return
    
    # Datasets
    train_dataset = MRIProjectDataset(train_df, transform=get_train_transforms())
    val_dataset = MRIProjectDataset(val_df, transform=get_val_transforms())
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Model
    model = UNet(n_channels=3, n_classes=1).to(device)
    
    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    dice_loss_fn = DiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # OneCycleLR Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, 
        steps_per_epoch=len(train_loader) // accumulation_steps, 
        epochs=epochs
    )
    
    # Scaler for Mixed Precision
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    # Early Stopping
    early_stopping = EarlyStopping(patience=15, target_metric=0.85)
    
    # Training Loop
    best_dice = 0.0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        pbar = tqdm(train_loader, desc="Training")
        optimizer.zero_grad()
        
        for i, (images, masks) in enumerate(pbar):
            images, masks = images.to(device), masks.to(device)
            
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(images)
                # Combined Loss: 1.0 * Dice + 0.5 * BCE (Better balance for small regions)
                bce_loss = criterion(outputs, masks)
                dice_l = dice_loss_fn(outputs, masks)
                loss = (0.5 * bce_loss + dice_l) / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            epoch_loss += loss.item() * accumulation_steps
            with torch.no_grad():
                d_coeff = dice_coeff(outputs, masks)
                epoch_dice += d_coeff.item()
            
            pbar.set_postfix({
                'loss': loss.item() * accumulation_steps, 
                'dice': d_coeff.item(),
                'lr': optimizer.param_groups[0]['lr']
            })
            
        # Validation
        val_loss, val_dice = validate(model, val_loader, criterion, dice_loss_fn, device)
        
        print(f"Summary - Train Loss: {epoch_loss/len(train_loader):.4f}, Train Dice: {epoch_dice/len(train_loader):.4f}")
        print(f"Summary - Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), "best_model.pth")
            print(f"New best model saved with Val Dice: {val_dice:.4f}")

        # Check Early Stopping
        early_stopping(val_dice)
        if early_stopping.early_stop:
            print("Early stopping triggered! Target accuracy reached and stabilized.")
            break

def validate(model, loader, criterion, dice_loss_fn, device):
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            bce_loss = criterion(outputs, masks)
            dice_l = dice_loss_fn(outputs, masks)
            loss = bce_loss + dice_l
            
            val_loss += loss.item()
            val_dice += dice_coeff(outputs, masks).item()
            
    return val_loss / len(loader), val_dice / len(loader)

if __name__ == "__main__":
    train()
