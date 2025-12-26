"""Utility functions for drug-drug interaction prediction."""

import random
import logging
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
from omegaconf import DictConfig


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: str = "auto") -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device: Device specification ("auto", "cuda", "mps", "cpu")
        
    Returns:
        PyTorch device object
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def setup_logging(log_level: str = "INFO", log_dir: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files (optional)
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger("ddi_prediction")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_dir specified)
    if log_dir:
        import os
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, "ddi_prediction.log"))
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: Dict[str, float],
    filepath: str,
    config: Optional[DictConfig] = None
) -> None:
    """Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        metrics: Evaluation metrics
        filepath: Path to save checkpoint
        config: Configuration object (optional)
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
        'config': config
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: PyTorch model (optional)
        optimizer: Optimizer (optional)
        device: Device to load checkpoint on
        
    Returns:
        Dictionary containing checkpoint data
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def early_stopping(
    current_loss: float,
    best_loss: float,
    patience: int,
    min_delta: float,
    patience_counter: int
) -> Tuple[bool, int]:
    """Check if early stopping criteria are met.
    
    Args:
        current_loss: Current validation loss
        best_loss: Best validation loss so far
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        patience_counter: Current patience counter
        
    Returns:
        Tuple of (should_stop, updated_patience_counter)
    """
    if current_loss < best_loss - min_delta:
        return False, 0
    else:
        patience_counter += 1
        return patience_counter >= patience, patience_counter


def format_time(seconds: float) -> str:
    """Format time in human-readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"
