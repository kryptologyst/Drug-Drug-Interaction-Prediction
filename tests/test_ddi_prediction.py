"""Tests for drug-drug interaction prediction."""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.data import DDIDataset, generate_synthetic_ddi_data, create_data_loaders
from src.models import MLPModel, GNNModel, MPNNModel, create_model
from src.losses import FocalLoss, WeightedCrossEntropyLoss, create_loss_function
from src.metrics import MetricsCalculator
from src.utils import set_seed, get_device, count_parameters


class TestData:
    """Test data loading and preprocessing."""
    
    def test_ddi_dataset(self):
        """Test DDIDataset creation and functionality."""
        drug_pairs = [
            ("CCO", "CCN", 1),
            ("CCCC", "CCO", 0),
            ("CC(=O)O", "CN(C)C=O", 1)
        ]
        
        dataset = DDIDataset(drug_pairs, fingerprint_type="morgan", fingerprint_bits=512)
        
        assert len(dataset) == 3
        assert len(dataset.samples) == 3
        
        # Test sample access
        features, label = dataset[0]
        assert isinstance(features, torch.Tensor)
        assert features.shape == (1024,)  # 512 * 2 for concatenated fingerprints
        assert isinstance(label, int)
        assert label in [0, 1]
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        drug_pairs = generate_synthetic_ddi_data(num_samples=100, seed=42)
        
        assert len(drug_pairs) == 100
        assert all(len(pair) == 3 for pair in drug_pairs)
        assert all(isinstance(pair[2], int) for pair in drug_pairs)
        assert all(pair[2] in [0, 1] for pair in drug_pairs)
    
    def test_data_loaders(self):
        """Test data loader creation."""
        drug_pairs = generate_synthetic_ddi_data(num_samples=100, seed=42)
        
        train_loader, val_loader, test_loader, scaler = create_data_loaders(
            drug_pairs, batch_size=16, seed=42
        )
        
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None
        assert scaler is not None
        
        # Test batch iteration
        for batch_features, batch_labels in train_loader:
            assert batch_features.shape[0] <= 16
            assert batch_features.shape[1] == 1024
            assert batch_labels.shape[0] == batch_features.shape[0]
            break


class TestModels:
    """Test model implementations."""
    
    def test_mlp_model(self):
        """Test MLP model creation and forward pass."""
        model = MLPModel(input_dim=1024, hidden_dims=[256, 64], dropout=0.2)
        
        assert model is not None
        assert count_parameters(model) > 0
        
        # Test forward pass
        x = torch.randn(4, 1024)
        output = model(x)
        
        assert output.shape == (4, 2)
        assert not torch.isnan(output).any()
    
    def test_gnn_model(self):
        """Test GNN model creation."""
        model = GNNModel(input_dim=1024, hidden_dim=128, num_layers=3)
        
        assert model is not None
        assert count_parameters(model) > 0
        
        # Test forward pass
        x = torch.randn(8, 1024)
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 0, 5, 6, 7, 4]])
        output = model(x, edge_index)
        
        assert output.shape == (4, 2)  # Assuming 4 pairs
        assert not torch.isnan(output).any()
    
    def test_model_creation(self):
        """Test model creation function."""
        mlp_model = create_model("mlp", input_dim=1024)
        assert isinstance(mlp_model, MLPModel)
        
        gnn_model = create_model("gnn", input_dim=1024)
        assert isinstance(gnn_model, GNNModel)
        
        mpnn_model = create_model("mpnn", input_dim=1024)
        assert isinstance(mpnn_model, MPNNModel)


class TestLosses:
    """Test loss functions."""
    
    def test_focal_loss(self):
        """Test focal loss computation."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        
        logits = torch.randn(4, 2)
        labels = torch.randint(0, 2, (4,))
        
        loss = loss_fn(logits, labels)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert not torch.isnan(loss)
    
    def test_weighted_cross_entropy(self):
        """Test weighted cross entropy loss."""
        class_weights = torch.tensor([1.0, 2.0])
        loss_fn = WeightedCrossEntropyLoss(class_weights)
        
        logits = torch.randn(4, 2)
        labels = torch.randint(0, 2, (4,))
        
        loss = loss_fn(logits, labels)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert not torch.isnan(loss)
    
    def test_loss_creation(self):
        """Test loss function creation."""
        ce_loss = create_loss_function("cross_entropy")
        assert isinstance(ce_loss, WeightedCrossEntropyLoss)
        
        focal_loss = create_loss_function("focal", focal_alpha=0.25, focal_gamma=2.0)
        assert isinstance(focal_loss, FocalLoss)


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_metrics_calculator(self):
        """Test metrics calculator."""
        calc = MetricsCalculator()
        
        # Add some predictions
        predictions = torch.tensor([0, 1, 0, 1])
        labels = torch.tensor([0, 1, 1, 1])
        probabilities = torch.tensor([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.2, 0.8]])
        
        calc.update(predictions, labels, probabilities)
        
        metrics = calc.compute_metrics()
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'roc_auc' in metrics
        assert 'pr_auc' in metrics
        
        assert all(isinstance(v, float) for v in metrics.values())
        assert all(0 <= v <= 1 for v in metrics.values() if v != 'loss')


class TestUtils:
    """Test utility functions."""
    
    def test_seed_setting(self):
        """Test random seed setting."""
        set_seed(42, deterministic=True)
        
        # Test that seeds are set
        import random
        import numpy as np
        
        # These should be deterministic after setting seed
        random_val = random.random()
        np_val = np.random.random()
        
        assert isinstance(random_val, float)
        assert isinstance(np_val, float)
    
    def test_device_selection(self):
        """Test device selection."""
        device = get_device("auto")
        assert isinstance(device, torch.device)
        
        cpu_device = get_device("cpu")
        assert cpu_device.type == "cpu"


if __name__ == "__main__":
    pytest.main([__file__])
