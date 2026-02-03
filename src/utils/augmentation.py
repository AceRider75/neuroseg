import albumentations as A

def get_train_transforms():
    """
    Required Transforms: Elastic deformation, Rotation (±15◦ ), Horizontal Flip, Brightness Jitter.
    Augmentations must be deterministic and paired.
    """
    return A.Compose([
        A.Rotate(limit=15, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ElasticTransform(alpha=1, sigma=50, p=0.2),
    ])

def get_val_transforms():
    """
    No augmentations for validation/test, only resizing if needed.
    """
    return None
