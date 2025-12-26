#!/usr/bin/env python3
"""Setup script for drug-drug interaction prediction project."""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up Drug-Drug Interaction Prediction Project")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create necessary directories
    directories = [
        "checkpoints",
        "logs", 
        "evaluation_results",
        "assets",
        "data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    
    # Install base requirements
    if not run_command("pip install -r requirements.txt", "Installing base requirements"):
        print("❌ Failed to install base requirements")
        sys.exit(1)
    
    # Install development dependencies
    if not run_command("pip install pytest black ruff mypy pre-commit", "Installing development dependencies"):
        print("⚠️  Failed to install development dependencies (optional)")
    
    # Setup pre-commit hooks
    if Path(".pre-commit-config.yaml").exists():
        if not run_command("pre-commit install", "Setting up pre-commit hooks"):
            print("⚠️  Failed to setup pre-commit hooks (optional)")
    
    # Run tests
    print("\n🧪 Running tests...")
    if not run_command("python -m pytest tests/ -v", "Running unit tests"):
        print("⚠️  Some tests failed (check test output above)")
    
    # Create initial model checkpoint (optional)
    print("\n🏗️  Creating initial model checkpoint...")
    if not run_command("python scripts/train.py --config configs/baseline.yaml --output_dir checkpoints", "Training initial model"):
        print("⚠️  Failed to create initial model (you can train manually later)")
    
    print("\n🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Run the interactive demo: streamlit run demo/app.py")
    print("2. Train with different models: python scripts/train.py --config configs/gnn.yaml")
    print("3. Evaluate models: python scripts/evaluate.py --model_path checkpoints/best_model.pth")
    print("4. Run the example notebook: python notebooks/example_usage.py")
    
    print("\n⚠️  IMPORTANT DISCLAIMER:")
    print("This is a research demonstration tool. It is NOT intended for clinical use.")
    print("Always consult qualified healthcare professionals for medical advice.")


if __name__ == "__main__":
    main()
