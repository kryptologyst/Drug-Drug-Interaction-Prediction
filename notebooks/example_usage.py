"""Example notebook for drug-drug interaction prediction."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
from src.data import generate_synthetic_ddi_data, create_data_loaders, DDIDataset
from src.models import create_model
from src.losses import create_loss_function
from src.metrics import MetricsCalculator
from src.utils import set_seed, get_device, count_parameters

# Set random seed for reproducibility
set_seed(42, deterministic=True)

print("Drug-Drug Interaction Prediction Example")
print("=" * 50)

# 1. Generate synthetic data
print("\n1. Generating synthetic drug-drug interaction data...")
drug_pairs = generate_synthetic_ddi_data(
    num_samples=1000,
    interaction_ratio=0.3,
    seed=42
)

print(f"Generated {len(drug_pairs)} drug pairs")
print(f"Sample pairs: {drug_pairs[:3]}")

# 2. Create data loaders
print("\n2. Creating data loaders...")
train_loader, val_loader, test_loader, scaler = create_data_loaders(
    drug_pairs,
    batch_size=32,
    train_split=0.7,
    val_split=0.15,
    test_split=0.15,
    fingerprint_type="morgan",
    fingerprint_bits=512,
    seed=42
)

print(f"Train batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

# 3. Create model
print("\n3. Creating model...")
model = create_model(
    model_name="mlp",
    input_dim=1024,
    hidden_dims=[256, 64],
    dropout=0.2,
    activation="relu"
)

device = get_device("auto")
model = model.to(device)

print(f"Model created with {count_parameters(model)} parameters")
print(f"Using device: {device}")

# 4. Create loss function
print("\n4. Creating loss function...")
loss_fn = create_loss_function("cross_entropy")
print("Cross-entropy loss created")

# 5. Training setup
print("\n5. Setting up training...")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

print(f"Training for {num_epochs} epochs")

# 6. Training loop
print("\n6. Training model...")
train_losses = []
val_losses = []
val_aucs = []

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    train_batches = 0
    
    for batch_features, batch_labels in train_loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()
        logits = model(batch_features)
        loss = loss_fn(logits, batch_labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_batches += 1
    
    avg_train_loss = train_loss / train_batches
    train_losses.append(avg_train_loss)
    
    # Validation
    model.eval()
    val_metrics_calc = MetricsCalculator()
    val_loss = 0.0
    val_batches = 0
    
    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            logits = model(batch_features)
            loss = loss_fn(logits, batch_labels)
            
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            val_metrics_calc.update(predictions, batch_labels, probabilities)
            val_loss += loss.item()
            val_batches += 1
    
    avg_val_loss = val_loss / val_batches
    val_losses.append(avg_val_loss)
    
    val_metrics = val_metrics_calc.compute_metrics()
    val_auc = val_metrics.get('roc_auc', 0.0)
    val_aucs.append(val_auc)
    
    print(f"Epoch {epoch+1}/{num_epochs}: "
          f"Train Loss: {avg_train_loss:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, "
          f"Val AUC: {val_auc:.4f}")

# 7. Plot training curves
print("\n7. Plotting training curves...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Loss curves
ax1.plot(train_losses, label='Training Loss', color='blue')
ax1.plot(val_losses, label='Validation Loss', color='red')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True)

# AUC curve
ax2.plot(val_aucs, label='Validation AUC', color='green')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('AUC')
ax2.set_title('Validation AUC')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# 8. Final evaluation
print("\n8. Final evaluation on test set...")
model.eval()
test_metrics_calc = MetricsCalculator()
test_loss = 0.0
test_batches = 0

with torch.no_grad():
    for batch_features, batch_labels in test_loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        
        logits = model(batch_features)
        loss = loss_fn(logits, batch_labels)
        
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)
        
        test_metrics_calc.update(predictions, batch_labels, probabilities)
        test_loss += loss.item()
        test_batches += 1

test_metrics = test_metrics_calc.compute_metrics()
test_metrics['loss'] = test_loss / test_batches

print("Test Results:")
for metric, value in test_metrics.items():
    print(f"  {metric}: {value:.4f}")

# 9. Generate evaluation plots
print("\n9. Generating evaluation plots...")
test_metrics_calc.plot_confusion_matrix()
test_metrics_calc.plot_roc_curve()
test_metrics_calc.plot_precision_recall_curve()
test_metrics_calc.plot_calibration_curve()

# 10. Example prediction
print("\n10. Example prediction...")
example_drug1 = "CCO"  # Ethanol
example_drug2 = "CCN"  # Ethylamine

example_dataset = DDIDataset(
    [(example_drug1, example_drug2, 0)],
    fingerprint_type="morgan",
    fingerprint_bits=512
)

if len(example_dataset) > 0:
    features, _ = example_dataset[0]
    features = features.unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(features)
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(logits, dim=1).item()
    
    prob_no_interaction = probabilities[0][0].item()
    prob_interaction = probabilities[0][1].item()
    
    print(f"Drug 1: {example_drug1}")
    print(f"Drug 2: {example_drug2}")
    print(f"Prediction: {'Interaction' if prediction == 1 else 'No Interaction'}")
    print(f"Probability of no interaction: {prob_no_interaction:.3f}")
    print(f"Probability of interaction: {prob_interaction:.3f}")

print("\nExample completed successfully!")
print("\nTo run the interactive demo, use: streamlit run demo/app.py")
print("To train with different configurations, use: python scripts/train.py --config configs/gnn.yaml")
