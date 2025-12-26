# Drug-Drug Interaction Prediction

**RESEARCH DEMONSTRATION ONLY - NOT FOR CLINICAL USE**

A reproducible healthcare AI project for predicting drug-drug interactions using molecular fingerprints and graph neural networks.

## Overview

This project implements state-of-the-art methods for predicting drug-drug interactions (DDIs) using:
- Molecular fingerprints (Morgan, MACCS, RDKit)
- Graph neural networks (MPNN, GCN, GraphSAGE)
- Multi-task learning for interaction type prediction
- Comprehensive evaluation with calibration and uncertainty quantification

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run training
python scripts/train.py --config configs/baseline.yaml

# Launch interactive demo
streamlit run demo/app.py

# Run evaluation
python scripts/evaluate.py --model_path checkpoints/best_model.pth
```

## Project Structure

```
src/
├── models/          # Model implementations
├── data/           # Data loading and preprocessing
├── losses/         # Loss functions
├── metrics/        # Evaluation metrics
├── utils/          # Utility functions
├── train.py        # Training script
└── eval.py         # Evaluation script

configs/            # Configuration files
scripts/            # Training and evaluation scripts
demo/               # Interactive Streamlit demo
tests/              # Unit tests
assets/             # Generated plots and results
```

## Features

- **Multiple Model Architectures**: MLP, GNN, MPNN, GraphSAGE
- **Comprehensive Evaluation**: ROC-AUC, PR-AUC, calibration plots, uncertainty quantification
- **Explainability**: SHAP values, attention maps, molecular visualization
- **Interactive Demo**: Upload drug pairs, visualize predictions and explanations
- **Synthetic Data**: Generates realistic drug-drug interaction datasets for demonstration

## Safety and Compliance

- **Research Only**: Not intended for clinical use or diagnostic purposes
- **No PHI/PII**: All data is synthetic or properly anonymized
- **Uncertainty Reporting**: Models provide confidence intervals and uncertainty estimates
- **Bias Monitoring**: Fairness evaluation across different drug classes

## Limitations

- Models trained on synthetic data for demonstration purposes
- Performance metrics are for research evaluation only
- Not validated for clinical use
- Requires clinician supervision for any medical applications

## Citation

If you use this code in your research, please cite:

```bibtex
@software{drug_interaction_prediction,
  title={Drug-Drug Interaction Prediction: A Research Demonstration},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Drug-Drug-Interaction-Prediction}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
# Drug-Drug-Interaction-Prediction
