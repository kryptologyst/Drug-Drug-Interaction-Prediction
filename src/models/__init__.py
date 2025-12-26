"""Model implementations for drug-drug interaction prediction."""

import logging
from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphSAGE, global_mean_pool
from torch_geometric.data import Data, Batch

logger = logging.getLogger(__name__)


class MLPModel(nn.Module):
    """Multi-layer perceptron for drug-drug interaction prediction.
    
    Args:
        input_dim: Input feature dimension
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout rate
        activation: Activation function name
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dims: List[int] = [256, 64],
        dropout: float = 0.2,
        activation: str = "relu"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 2))
        
        self.network = nn.Sequential(*layers)
        
        logger.info(f"Created MLP model with {self.count_parameters()} parameters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Logits [batch_size, 2]
        """
        return self.network(x)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GNNModel(nn.Module):
    """Graph Neural Network for drug-drug interaction prediction.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        num_layers: Number of GNN layers
        dropout: Dropout rate
        gnn_type: Type of GNN ("gcn", "gat", "sage")
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        gnn_type: str = "gcn"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            if gnn_type == "gcn":
                layer = GCNConv(hidden_dim, hidden_dim)
            elif gnn_type == "gat":
                layer = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
            elif gnn_type == "sage":
                layer = GraphSAGE(hidden_dim, hidden_dim, num_layers=1)
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")
            
            self.gnn_layers.append(layer)
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for concatenated features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
        
        logger.info(f"Created {gnn_type.upper()} model with {self.count_parameters()} parameters")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Logits [batch_size, 2]
        """
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        
        # GNN layers
        for layer in self.gnn_layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling and classification
        # For drug-drug interaction, we need to handle pairs
        # This is a simplified version - in practice, you'd need more sophisticated handling
        batch_size = x.size(0) // 2  # Assuming pairs of drugs
        x1 = x[:batch_size]  # First drug
        x2 = x[batch_size:batch_size*2]  # Second drug
        
        # Concatenate features
        combined = torch.cat([x1, x2], dim=1)
        
        return self.classifier(combined)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MPNNModel(nn.Module):
    """Message Passing Neural Network for drug-drug interaction prediction.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        num_layers: Number of MPNN layers
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Message passing layers
        self.message_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()
        
        for i in range(num_layers):
            self.message_layers.append(nn.Linear(hidden_dim * 2, hidden_dim))
            self.update_layers.append(nn.GRU(hidden_dim, hidden_dim))
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
        
        logger.info(f"Created MPNN model with {self.count_parameters()} parameters")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Logits [batch_size, 2]
        """
        # Input projection
        x = self.input_proj(x)
        
        # Message passing
        for i in range(self.num_layers):
            # Compute messages
            row, col = edge_index
            messages = self.message_layers[i](torch.cat([x[row], x[col]], dim=1))
            
            # Aggregate messages
            message_aggr = torch.zeros_like(x)
            message_aggr.scatter_add_(0, col.unsqueeze(-1).expand_as(messages), messages)
            
            # Update node features
            x, _ = self.update_layers[i](message_aggr.unsqueeze(0), x.unsqueeze(0))
            x = x.squeeze(0)
            
            # Apply dropout
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # For drug-drug interaction prediction
        batch_size = x.size(0) // 2
        x1 = x[:batch_size]
        x2 = x[batch_size:batch_size*2]
        
        # Concatenate features
        combined = torch.cat([x1, x2], dim=1)
        
        return self.classifier(combined)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(
    model_name: str,
    input_dim: int = 1024,
    **kwargs
) -> nn.Module:
    """Create a model instance.
    
    Args:
        model_name: Name of the model ("mlp", "gnn", "mpnn")
        input_dim: Input feature dimension
        **kwargs: Additional model parameters
        
    Returns:
        Model instance
    """
    if model_name == "mlp":
        return MLPModel(input_dim=input_dim, **kwargs)
    elif model_name == "gnn":
        return GNNModel(input_dim=input_dim, **kwargs)
    elif model_name == "mpnn":
        return MPNNModel(input_dim=input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")


class EnsembleModel(nn.Module):
    """Ensemble of multiple models for improved prediction.
    
    Args:
        models: List of models to ensemble
        weights: Optional weights for each model
    """
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super().__init__()
        
        self.models = nn.ModuleList(models)
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            self.weights = weights
        
        logger.info(f"Created ensemble with {len(models)} models")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with ensemble prediction.
        
        Args:
            x: Input features
            
        Returns:
            Weighted average of model predictions
        """
        predictions = []
        
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Weighted average
        ensemble_pred = torch.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            ensemble_pred += weight * pred
        
        return ensemble_pred
