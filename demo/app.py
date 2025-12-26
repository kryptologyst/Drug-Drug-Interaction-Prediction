"""Interactive Streamlit demo for drug-drug interaction prediction."""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple

from src.utils import get_device, load_checkpoint
from src.data import DDIDataset
from src.models import create_model
from src.losses import create_loss_function
from src.metrics import MetricsCalculator
from rdkit import Chem
from rdkit.Chem import Draw
import io


# Page configuration
st.set_page_config(
    page_title="Drug-Drug Interaction Prediction",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for disclaimer
st.markdown("""
<style>
.disclaimer {
    background-color: #ffebee;
    border: 2px solid #f44336;
    border-radius: 5px;
    padding: 10px;
    margin: 10px 0;
    font-weight: bold;
    color: #d32f2f;
}
</style>
""", unsafe_allow_html=True)

# Disclaimer banner
st.markdown("""
<div class="disclaimer">
⚠️ RESEARCH DEMONSTRATION ONLY - NOT FOR CLINICAL USE ⚠️<br>
This tool is for educational and research purposes only. It does not provide medical advice, 
diagnosis, or treatment recommendations. Always consult qualified healthcare professionals 
for medical decisions.
</div>
""", unsafe_allow_html=True)

# Title and description
st.title("💊 Drug-Drug Interaction Prediction")
st.markdown("""
This interactive demo predicts potential drug-drug interactions using molecular fingerprints 
and machine learning models. Enter SMILES strings for two drugs to get predictions and explanations.
""")

# Sidebar for model selection
st.sidebar.header("Model Configuration")
model_path = st.sidebar.selectbox(
    "Select Model",
    ["checkpoints/best_model.pth", "checkpoints/checkpoint_epoch_50.pth"],
    help="Choose a trained model checkpoint"
)

# Check if model exists
if not Path(model_path).exists():
    st.error(f"Model file not found: {model_path}")
    st.info("Please train a model first using: `python scripts/train.py`")
    st.stop()

# Load model and configuration
@st.cache_resource
def load_model_and_config(model_path: str):
    """Load model and configuration."""
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        config = checkpoint.get('config')
        
        if config is None:
            st.error("No configuration found in checkpoint")
            return None, None
        
        # Create model
        model = create_model(
            model_name=config.model.name,
            input_dim=config.model.input_dim,
            hidden_dims=config.model.hidden_dims,
            dropout=config.model.dropout,
            activation=config.model.activation
        )
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, config
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, config = load_model_and_config(model_path)

if model is None or config is None:
    st.stop()

# Display model info
st.sidebar.markdown("### Model Information")
st.sidebar.write(f"**Model Type:** {config.model.name.upper()}")
st.sidebar.write(f"**Input Dimension:** {config.model.input_dim}")
st.sidebar.write(f"**Hidden Layers:** {config.model.hidden_dims}")
st.sidebar.write(f"**Dropout:** {config.model.dropout}")

# Main interface
col1, col2 = st.columns(2)

with col1:
    st.header("Drug Input")
    
    # Drug 1 input
    drug1_smiles = st.text_input(
        "Drug 1 SMILES",
        value="CCO",  # Ethanol
        help="Enter the SMILES string for the first drug"
    )
    
    # Drug 2 input
    drug2_smiles = st.text_input(
        "Drug 2 SMILES",
        value="CCN",  # Ethylamine
        help="Enter the SMILES string for the second drug"
    )
    
    # Validate SMILES
    def validate_smiles(smiles: str) -> bool:
        """Validate SMILES string."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    drug1_valid = validate_smiles(drug1_smiles)
    drug2_valid = validate_smiles(drug2_smiles)
    
    if not drug1_valid:
        st.error("Invalid SMILES for Drug 1")
    if not drug2_valid:
        st.error("Invalid SMILES for Drug 2")
    
    # Predict button
    predict_button = st.button("🔮 Predict Interaction", disabled=not (drug1_valid and drug2_valid))

with col2:
    st.header("Molecular Structures")
    
    if drug1_valid and drug2_valid:
        try:
            # Draw molecules
            mol1 = Chem.MolFromSmiles(drug1_smiles)
            mol2 = Chem.MolFromSmiles(drug2_smiles)
            
            # Create molecule images
            img1 = Draw.MolToImage(mol1, size=(300, 300))
            img2 = Draw.MolToImage(mol2, size=(300, 300))
            
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                st.image(img1, caption="Drug 1", use_column_width=True)
            with col2_2:
                st.image(img2, caption="Drug 2", use_column_width=True)
                
        except Exception as e:
            st.error(f"Error drawing molecules: {e}")

# Prediction results
if predict_button and drug1_valid and drug2_valid:
    st.header("Prediction Results")
    
    try:
        # Create dataset for prediction
        drug_pairs = [(drug1_smiles, drug2_smiles, 0)]  # Label doesn't matter for prediction
        
        dataset = DDIDataset(
            drug_pairs,
            fingerprint_type=config.data.fingerprint_type,
            fingerprint_bits=config.data.fingerprint_bits
        )
        
        if len(dataset) == 0:
            st.error("Failed to process drug pair")
            st.stop()
        
        # Get features
        features, _ = dataset[0]
        features = features.unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            logits = model(features)
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()
        
        # Display results
        prob_no_interaction = probabilities[0][0].item()
        prob_interaction = probabilities[0][1].item()
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Prediction")
            if prediction == 1:
                st.error("🚨 **INTERACTION DETECTED**")
                st.write(f"Probability: {prob_interaction:.3f}")
            else:
                st.success("✅ **NO INTERACTION**")
                st.write(f"Probability: {prob_no_interaction:.3f}")
        
        with col4:
            st.subheader("Confidence")
            confidence = max(prob_no_interaction, prob_interaction)
            st.metric("Confidence", f"{confidence:.3f}")
            
            # Confidence bar
            st.progress(confidence)
        
        # Detailed probabilities
        st.subheader("Detailed Probabilities")
        prob_df = pd.DataFrame({
            'Class': ['No Interaction', 'Interaction'],
            'Probability': [prob_no_interaction, prob_interaction]
        })
        
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(prob_df['Class'], prob_df['Probability'], 
                     color=['green', 'red'], alpha=0.7)
        ax.set_ylabel('Probability')
        ax.set_title('Prediction Probabilities')
        ax.set_ylim(0, 1)
        
        # Add probability values on bars
        for bar, prob in zip(bars, prob_df['Probability']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{prob:.3f}', ha='center', va='bottom')
        
        st.pyplot(fig)
        
        # Uncertainty analysis
        st.subheader("Uncertainty Analysis")
        entropy = -np.sum(probabilities.numpy() * np.log(probabilities.numpy() + 1e-8))
        max_prob = torch.max(probabilities).item()
        
        col5, col6 = st.columns(2)
        with col5:
            st.metric("Entropy", f"{entropy:.3f}")
        with col6:
            st.metric("Max Probability", f"{max_prob:.3f}")
        
        # Interpretation
        st.subheader("Interpretation")
        if confidence > 0.8:
            st.success("High confidence prediction")
        elif confidence > 0.6:
            st.warning("Moderate confidence prediction")
        else:
            st.error("Low confidence prediction - consider additional validation")
        
        if entropy > 0.7:
            st.info("High uncertainty - model is unsure about this prediction")
        elif entropy > 0.4:
            st.info("Moderate uncertainty")
        else:
            st.info("Low uncertainty - model is confident about this prediction")
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")

# Additional features
st.header("Additional Features")

# Batch prediction
st.subheader("Batch Prediction")
uploaded_file = st.file_uploader(
    "Upload CSV file with drug pairs",
    type=['csv'],
    help="CSV should have columns: drug1_smiles, drug2_smiles"
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        if 'drug1_smiles' in df.columns and 'drug2_smiles' in df.columns:
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            if st.button("🔮 Predict All Pairs"):
                predictions = []
                probabilities = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, row in df.iterrows():
                    status_text.text(f"Processing pair {i+1}/{len(df)}")
                    
                    try:
                        # Create dataset
                        drug_pairs = [(row['drug1_smiles'], row['drug2_smiles'], 0)]
                        dataset = DDIDataset(
                            drug_pairs,
                            fingerprint_type=config.data.fingerprint_type,
                            fingerprint_bits=config.data.fingerprint_bits
                        )
                        
                        if len(dataset) > 0:
                            features, _ = dataset[0]
                            features = features.unsqueeze(0)
                            
                            with torch.no_grad():
                                logits = model(features)
                                probs = torch.softmax(logits, dim=1)
                                pred = torch.argmax(logits, dim=1).item()
                            
                            predictions.append(pred)
                            probabilities.append(probs[0][1].item())  # Probability of interaction
                        else:
                            predictions.append(-1)  # Error
                            probabilities.append(0.0)
                    
                    except Exception as e:
                        st.warning(f"Error processing pair {i+1}: {e}")
                        predictions.append(-1)
                        probabilities.append(0.0)
                    
                    progress_bar.progress((i + 1) / len(df))
                
                # Add results to dataframe
                df['prediction'] = predictions
                df['interaction_probability'] = probabilities
                
                st.write("Results:")
                st.dataframe(df)
                
                # Download results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name="ddi_predictions.csv",
                    mime="text/csv"
                )
        
        else:
            st.error("CSV must contain 'drug1_smiles' and 'drug2_smiles' columns")
    
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")

# Model performance
st.subheader("Model Performance")
if st.button("📊 Show Model Performance"):
    st.info("Model performance metrics would be displayed here based on test set evaluation.")
    st.write("Run `python scripts/evaluate.py --model_path checkpoints/best_model.pth` to see detailed performance metrics.")

# Footer
st.markdown("---")
st.markdown("""
<div class="disclaimer">
⚠️ **IMPORTANT DISCLAIMER** ⚠️<br>
This is a research demonstration tool. Predictions are based on synthetic data and 
should not be used for clinical decision-making. Always consult qualified healthcare 
professionals for medical advice.
</div>
""", unsafe_allow_html=True)
