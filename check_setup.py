import torch
import torch.nn as nn
from src.data.dataset import get_data_split, MRIProjectDataset
from src.models.unet import UNet
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

def check():
    print("--- NeuroSeg Setup Verification ---")
    print("Checking torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA Device:", torch.cuda.get_device_name(0))
    
    # Check data split
    print("\nVerifying Data Split...")
    try:
        train_df, val_df, test_df = get_data_split("archive")
        print("Data split successful.")
        print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
    except Exception as e:
        print(f"Data split failed: {e}")
        return

    # Check model initialization
    print("\nVerifying Model Architecture...")
    try:
        model = UNet(n_channels=3, n_classes=1)
        dummy_input = torch.randn(1, 3, 256, 256)
        output = model(dummy_input)
        print("Model initialization and forward pass successful.")
        print("Output shape:", output.shape)
    except Exception as e:
        print(f"Model check failed: {e}")
        return

    # Check dataset
    print("\nVerifying Dataset Access...")
    try:
        dataset = MRIProjectDataset(train_df.head(10))
        img, mask = dataset[0]
        print("Dataset access successful.")
        print("Image shape:", img.shape, "Mask shape:", mask.shape)
        print("Image mean:", img.mean().item(), "std:", img.std().item())
    except Exception as e:
        print(f"Dataset check failed: {e}")
        return
    
    print("\n--- Setup Verified Successfully! ---")

if __name__ == "__main__":
    check()
