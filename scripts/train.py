"""Training script for drug-drug interaction prediction."""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import wandb

from src.utils import (
    set_seed, get_device, setup_logging, save_checkpoint, 
    load_checkpoint, early_stopping, format_time, count_parameters
)
from src.data import generate_synthetic_ddi_data, create_data_loaders, get_class_weights
from src.models import create_model
from src.losses import create_loss_function
from src.metrics import MetricsCalculator


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    logger: logging.Logger
) -> Dict[str, float]:
    """Train model for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device to run on
        logger: Logger instance
        
    Returns:
        Dictionary of training metrics
    """
    model.train()
    metrics_calc = MetricsCalculator()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (features, labels) in enumerate(train_loader):
        features = features.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(features)
        loss = loss_fn(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            metrics_calc.update(predictions, labels, probabilities)
            total_loss += loss.item()
            num_batches += 1
    
    # Compute epoch metrics
    metrics = metrics_calc.compute_metrics()
    metrics['loss'] = total_loss / num_batches
    
    return metrics


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    logger: logging.Logger
) -> Dict[str, float]:
    """Validate model for one epoch.
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        loss_fn: Loss function
        device: Device to run on
        logger: Logger instance
        
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    metrics_calc = MetricsCalculator()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(features)
            loss = loss_fn(logits, labels)
            
            # Compute metrics
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            metrics_calc.update(predictions, labels, probabilities)
            total_loss += loss.item()
            num_batches += 1
    
    # Compute epoch metrics
    metrics = metrics_calc.compute_metrics()
    metrics['loss'] = total_loss / num_batches
    
    return metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train drug-drug interaction prediction model")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Setup
    set_seed(config.seed, config.deterministic)
    device = get_device(config.training.device)
    logger = setup_logging(config.logging.log_level, config.logging.log_dir)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting training with config: {args.config}")
    logger.info(f"Device: {device}")
    logger.info(f"Output directory: {output_dir}")
    
    # Initialize wandb if enabled
    if config.logging.use_wandb:
        wandb.init(
            project=config.logging.wandb_project,
            config=OmegaConf.to_container(config, resolve=True),
            name=f"ddi_prediction_{config.model.name}"
        )
    
    # Generate synthetic data
    logger.info("Generating synthetic drug-drug interaction data...")
    drug_pairs = generate_synthetic_ddi_data(
        num_samples=config.data.num_samples,
        interaction_ratio=0.3,
        seed=config.seed
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader, scaler = create_data_loaders(
        drug_pairs,
        batch_size=config.data.batch_size,
        train_split=config.data.train_split,
        val_split=config.data.val_split,
        test_split=config.data.test_split,
        fingerprint_type=config.data.fingerprint_type,
        fingerprint_bits=config.data.fingerprint_bits,
        seed=config.seed
    )
    
    # Create model
    model = create_model(
        model_name=config.model.name,
        input_dim=config.model.input_dim,
        hidden_dims=config.model.hidden_dims,
        dropout=config.model.dropout,
        activation=config.model.activation
    )
    model = model.to(device)
    
    logger.info(f"Model created with {count_parameters(model)} parameters")
    
    # Create loss function
    class_weights = get_class_weights(drug_pairs)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    
    loss_fn = create_loss_function(
        loss_name=config.loss.name,
        class_weights=class_weights,
        focal_alpha=config.loss.focal_alpha,
        focal_gamma=config.loss.focal_gamma
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Training loop
    start_time = time.time()
    best_val_loss = float('inf')
    patience_counter = 0
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['loss']
    
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, config.training.epochs):
        epoch_start_time = time.time()
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device, logger)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, loss_fn, device, logger)
        
        epoch_time = time.time() - epoch_start_time
        
        # Log metrics
        logger.info(
            f"Epoch {epoch+1}/{config.training.epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val AUC: {val_metrics.get('roc_auc', 0.0):.4f}, "
            f"Time: {format_time(epoch_time)}"
        )
        
        # Log to wandb
        if config.logging.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_metrics['loss'],
                "val_loss": val_metrics['loss'],
                "val_roc_auc": val_metrics.get('roc_auc', 0.0),
                "val_accuracy": val_metrics.get('accuracy', 0.0),
                "epoch_time": epoch_time
            })
        
        # Save checkpoint
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pth"
        save_checkpoint(
            model, optimizer, epoch, val_metrics['loss'], val_metrics,
            str(checkpoint_path), config
        )
        
        # Early stopping
        should_stop, patience_counter = early_stopping(
            val_metrics['loss'], best_val_loss, config.training.patience,
            config.training.min_delta, patience_counter
        )
        
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            # Save best model
            best_path = output_dir / "best_model.pth"
            save_checkpoint(
                model, optimizer, epoch, val_metrics['loss'], val_metrics,
                str(best_path), config
            )
            logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
        
        if should_stop:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    total_time = time.time() - start_time
    logger.info(f"Training completed in {format_time(total_time)}")
    
    # Final evaluation on test set
    logger.info("Evaluating on test set...")
    test_metrics = validate_epoch(model, test_loader, loss_fn, device, logger)
    
    logger.info("Test Results:")
    for metric, value in test_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save final results
    results_path = output_dir / "final_results.yaml"
    OmegaConf.save(test_metrics, results_path)
    
    if config.logging.use_wandb:
        wandb.log({"test_" + k: v for k, v in test_metrics.items()})
        wandb.finish()
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
