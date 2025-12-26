"""Loss functions for drug-drug interaction prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.
    
    Args:
        alpha: Weighting factor for rare class
        gamma: Focusing parameter
        reduction: Reduction method
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            inputs: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            
        Returns:
            Focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedCrossEntropyLoss(nn.Module):
    """Weighted Cross Entropy Loss for class imbalance.
    
    Args:
        class_weights: Weights for each class
    """
    
    def __init__(self, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.class_weights = class_weights
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted cross entropy loss.
        
        Args:
            inputs: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            
        Returns:
            Weighted cross entropy loss
        """
        if self.class_weights is not None:
            return F.cross_entropy(inputs, targets, weight=self.class_weights)
        else:
            return F.cross_entropy(inputs, targets)


class LabelSmoothingLoss(nn.Module):
    """Label Smoothing Cross Entropy Loss.
    
    Args:
        smoothing: Label smoothing factor
        num_classes: Number of classes
    """
    
    def __init__(self, smoothing: float = 0.1, num_classes: int = 2):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute label smoothing loss.
        
        Args:
            inputs: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            
        Returns:
            Label smoothing loss
        """
        log_preds = F.log_softmax(inputs, dim=1)
        true_dist = torch.zeros_like(log_preds)
        true_dist.fill_(self.smoothing / (self.num_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)
        
        return torch.mean(torch.sum(-true_dist * log_preds, dim=1))


def create_loss_function(
    loss_name: str,
    class_weights: Optional[torch.Tensor] = None,
    **kwargs
) -> nn.Module:
    """Create a loss function.
    
    Args:
        loss_name: Name of the loss function
        class_weights: Optional class weights
        **kwargs: Additional loss parameters
        
    Returns:
        Loss function instance
    """
    if loss_name == "cross_entropy":
        return WeightedCrossEntropyLoss(class_weights)
    elif loss_name == "focal":
        return FocalLoss(**kwargs)
    elif loss_name == "label_smoothing":
        return LabelSmoothingLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
