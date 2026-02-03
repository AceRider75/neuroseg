import torch

def dice_coeff(input, target, smooth=1.0):
    """
    Compute Dice Similarity Coefficient.
    """
    input = torch.sigmoid(input)
    
    # Flatten tensors
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
