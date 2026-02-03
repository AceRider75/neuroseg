import cv2
import numpy as np
import matplotlib.pyplot as plt

def overlay_mask(image, mask, alpha=0.5, color=(255, 0, 0)):
    """
    Overlay mask onto MRI image.
    0 = Healthy, 1 = Tumor
    Red channel for tumor.
    """
    # Denormalize image if necessary (assuming [0, 1] or Z-score)
    # For visualization, we'll just scale to [0, 255]
    img_min, img_max = image.min(), image.max()
    image_display = ((image - img_min) / (img_max - img_min + 1e-8) * 255).astype(np.uint8)
    
    if len(image_display.shape) == 2:
        image_display = cv2.cvtColor(image_display, cv2.COLOR_GRAY2RGB)
    
    mask_display = (mask > 0.5).astype(np.uint8) * 255
    
    overlay = image_display.copy()
    colored_mask = np.zeros_like(image_display)
    colored_mask[mask > 0.5] = color
    
    cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0, overlay)
    
    return overlay

def plot_samples(images, masks, predictions=None, n=3):
    """
    Plot sample images, masks, and optionally predictions.
    """
    plt.figure(figsize=(15, 5 * (2 if predictions is None else 3)))
    
    for i in range(min(n, len(images))):
        # Original Image
        plt.subplot(n, 3 if predictions is not None else 2, i * (3 if predictions is not None else 2) + 1)
        plt.imshow(images[i].transpose(1, 2, 0))
        plt.title(f"Sample {i+1} - MRI")
        plt.axis('off')
        
        # Ground Truth Mask
        plt.subplot(n, 3 if predictions is not None else 2, i * (3 if predictions is not None else 2) + 2)
        plt.imshow(masks[i][0], cmap='gray')
        plt.title(f"Sample {i+1} - GT Mask")
        plt.axis('off')
        
        if predictions is not None:
            # Prediction
            plt.subplot(n, 3, i * 3 + 3)
            plt.imshow(predictions[i][0], cmap='gray')
            plt.title(f"Sample {i+1} - Prediction")
            plt.axis('off')
            
    plt.tight_layout()
    plt.show()
