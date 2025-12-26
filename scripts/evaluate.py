"""Evaluation script for drug-drug interaction prediction."""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

from src.utils import get_device, setup_logging, load_checkpoint
from src.data import generate_synthetic_ddi_data, create_data_loaders
from src.models import create_model
from src.losses import create_loss_function
from src.metrics import MetricsCalculator, compute_uncertainty_metrics, create_leaderboard


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    logger: logging.Logger,
    output_dir: Path
) -> Dict[str, Any]:
    """Evaluate model on test set.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        loss_fn: Loss function
        device: Device to run on
        logger: Logger instance
        output_dir: Output directory for plots
        
    Returns:
        Dictionary of evaluation results
    """
    model.eval()
    metrics_calc = MetricsCalculator()
    total_loss = 0.0
    num_batches = 0
    
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
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
            
            # Store for detailed analysis
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    metrics = metrics_calc.compute_metrics()
    metrics['loss'] = total_loss / num_batches
    
    # Compute uncertainty metrics
    uncertainty_metrics = compute_uncertainty_metrics(
        torch.tensor(all_predictions),
        torch.tensor(all_probabilities),
        torch.tensor(all_labels)
    )
    metrics.update(uncertainty_metrics)
    
    # Generate plots
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Generating evaluation plots...")
    metrics_calc.plot_confusion_matrix(str(output_dir / "confusion_matrix.png"))
    metrics_calc.plot_roc_curve(str(output_dir / "roc_curve.png"))
    metrics_calc.plot_precision_recall_curve(str(output_dir / "pr_curve.png"))
    metrics_calc.plot_calibration_curve(str(output_dir / "calibration_curve.png"))
    
    return metrics


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate drug-drug interaction prediction model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of test samples")
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Setup
    device = get_device(config.training.device)
    logger = setup_logging(config.logging.log_level)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Evaluating model: {args.model_path}")
    logger.info(f"Device: {device}")
    logger.info(f"Output directory: {output_dir}")
    
    # Generate test data
    logger.info("Generating test data...")
    drug_pairs = generate_synthetic_ddi_data(
        num_samples=args.num_samples,
        interaction_ratio=0.3,
        seed=config.seed
    )
    
    # Create test data loader
    _, _, test_loader, scaler = create_data_loaders(
        drug_pairs,
        batch_size=config.data.batch_size,
        train_split=0.0,
        val_split=0.0,
        test_split=1.0,
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
    
    # Load checkpoint
    checkpoint = load_checkpoint(args.model_path, model, device=device)
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Create loss function
    from src.data import get_class_weights
    class_weights = get_class_weights(drug_pairs)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    
    loss_fn = create_loss_function(
        loss_name=config.loss.name,
        class_weights=class_weights,
        focal_alpha=config.loss.focal_alpha,
        focal_gamma=config.loss.focal_gamma
    )
    
    # Evaluate model
    logger.info("Evaluating model...")
    results = evaluate_model(model, test_loader, loss_fn, device, logger, output_dir)
    
    # Print results
    logger.info("Evaluation Results:")
    logger.info("=" * 50)
    for metric, value in results.items():
        logger.info(f"{metric:20s}: {value:.4f}")
    
    # Save results
    results_path = output_dir / "evaluation_results.yaml"
    OmegaConf.save(results, results_path)
    logger.info(f"Results saved to: {results_path}")
    
    # Create leaderboard
    leaderboard = create_leaderboard({config.model.name: results})
    logger.info("\nLeaderboard:")
    logger.info(leaderboard)
    
    # Save leaderboard
    leaderboard_path = output_dir / "leaderboard.txt"
    with open(leaderboard_path, 'w') as f:
        f.write(leaderboard)
    
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
