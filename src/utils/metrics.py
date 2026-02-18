import torch

def dice_coeff(input, target, smooth=1.0):
    """
    Compute soft Dice Similarity Coefficient (for training loss).
    Uses sigmoid probabilities — suitable for differentiable loss.
    """
    input = torch.sigmoid(input)
    
    # Flatten tensors
    input = input.view(-1)
    target = target.view(-1)
    
    intersection = (input * target).sum()
    dice = (2. * intersection + smooth) / (input.sum() + target.sum() + smooth)
    
    return dice

def dice_coeff_hard(input, target, threshold=0.5, smooth=1.0):
    """
    Compute hard-thresholded Dice for accurate validation reporting.
    Uses binary predictions — gives the true segmentation Dice score.
    """
    input = (torch.sigmoid(input) > threshold).float()
    
    input = input.view(-1)
    target = target.view(-1)
    
    intersection = (input * target).sum()
    dice = (2. * intersection + smooth) / (input.sum() + target.sum() + smooth)
    
    return dice

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        return 1 - dice_coeff(input, target, self.smooth)

def iou_score(input, target, smooth=1.0):
    """
    Compute Intersection over Union.
    """
    input = torch.sigmoid(input)
    input = (input > 0.5).float()
    
    input = input.view(-1)
    target = target.view(-1)
    
    intersection = (input * target).sum()
    total = (input + target).sum()
    union = total - intersection
    
    return (intersection + smooth) / (union + smooth)
