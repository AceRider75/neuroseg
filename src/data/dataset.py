import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from glob import glob

class MRIProjectDataset(Dataset):
    """
    Custom Dataset for MRI Brain Tumor Segmentation.
    """
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        mask_path = self.df.iloc[idx]['mask_path']

        # Read images
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Normalization and Tensor conversion
        # Image normalization: Z-score normalization as per requirements
        image = image.astype(np.float32)
        image = (image - np.mean(image)) / (np.std(image) + 1e-8)
        
        # Mask normalization: [0, 1]
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=0) # (1, H, W)
        
        # Transpose image to (C, H, W)
        if len(image.shape) == 3:
            image = np.transpose(image, (2, 0, 1))
        else:
            image = np.expand_dims(image, axis=0)

        return torch.from_numpy(image), torch.from_numpy(mask)

def get_data_split(base_path, test_size=0.15, val_size=0.15, random_seed=42):
    """
    Perform subject-level splitting to prevent data leakage.
    """
    all_images = glob(os.path.join(base_path, "kaggle_3m/*/*[!_mask].tif"))
    
    data = []
    for img_path in all_images:
        mask_path = img_path.replace(".tif", "_mask.tif")
        if not os.path.exists(mask_path):
            continue
            
        patient_id = os.path.basename(os.path.dirname(img_path))
        data.append({
            'patient_id': patient_id,
            'image_path': img_path,
            'mask_path': mask_path
        })
    
    df = pd.DataFrame(data)
    
    if df.empty:
        raise ValueError(f"No images found in {base_path}")
        
    # Subject-level split
    patients = df['patient_id'].unique()
    np.random.seed(random_seed)
    np.random.shuffle(patients)
    
    num_test = int(len(patients) * test_size)
    num_val = int(len(patients) * val_size)
    
    test_patients = patients[:num_test]
    val_patients = patients[num_test:num_test+num_val]
    train_patients = patients[num_test+num_val:]
    
    train_df = df[df['patient_id'].isin(train_patients)].reset_index(drop=True)
    val_df = df[df['patient_id'].isin(val_patients)].reset_index(drop=True)
    test_df = df[df['patient_id'].isin(test_patients)].reset_index(drop=True)
    
    print(f"Dataset Split Summary:")
    print(f"Total Patients: {len(patients)}")
    print(f"Train: {len(train_df)} slices ({len(train_patients)} patients)")
    print(f"Val:   {len(val_df)} slices ({len(val_patients)} patients)")
    print(f"Test:  {len(test_df)} slices ({len(test_patients)} patients)")
    
    return train_df, val_df, test_df
