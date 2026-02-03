import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data.dataset import get_data_split, MRIProjectDataset
from src.models.unet import UNet
from src.utils.metrics import DiceLoss, dice_coeff
from src.utils.augmentation import get_train_transforms, get_val_transforms
from tqdm import tqdm

def train():
    # Parameters
    base_path = "archive"
    batch_size = 4  # Reduced batch size for stability
    lr = 1e-4
    epochs = 50
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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    # Training Loop
    best_dice = 0.0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        pbar = tqdm(train_loader, desc="Training")
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Combined Loss: BCE + Dice
            bce_loss = criterion(outputs, masks)
            dice_l = dice_loss_fn(outputs, masks)
            loss = bce_loss + dice_l
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            with torch.no_grad():
                d_coeff = dice_coeff(outputs, masks)
                epoch_dice += d_coeff.item()
            
            pbar.set_postfix({'loss': loss.item(), 'dice': d_coeff.item()})
            
        # Validation
        val_loss, val_dice = validate(model, val_loader, criterion, dice_loss_fn, device)
        scheduler.step(val_loss)
        
        print(f"Summary - Train Loss: {epoch_loss/len(train_loader):.4f}, Train Dice: {epoch_dice/len(train_loader):.4f}")
        print(f"Summary - Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), "best_model.pth")
            print(f"New best model saved with Val Dice: {val_dice:.4f}")

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
