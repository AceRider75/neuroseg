import albumentations as A

def get_train_transforms():
    """
    Enhanced augmentation pipeline for brain MRI segmentation.
    
    Required: Elastic deformation, Rotation (±15°), Horizontal Flip, Brightness Jitter.
    Added: VerticalFlip, GridDistortion, GaussNoise, CoarseDropout, ShiftScaleRotate.
    
    All augmentations are deterministic and paired (image + mask transform together).
    """
    return A.Compose([
        # Spatial transforms
        A.Rotate(limit=20, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=0, p=0.3),

        # Deformation transforms (crucial for medical imaging)
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),

        # Intensity transforms (image-only, masks unaffected)
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.GaussNoise(var_limit=(0.0, 0.01), p=0.2),

        # Occlusion robustness
        A.CoarseDropout(max_holes=4, max_height=20, max_width=20,
                        min_holes=1, min_height=8, min_width=8,
                        fill_value=0, p=0.2),
    ])

def get_val_transforms():
    """
    No augmentations for validation/test, only resizing if needed.
    """
    return None
