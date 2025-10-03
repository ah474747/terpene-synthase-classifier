#!/usr/bin/env python3
"""
Germacrene Synthase Binary Classifier
=====================================

This script implements a binary classifier for Germacrene synthase prediction using:
- Pre-trained protein language models (ESM-2, ProtT5)
- Stratified k-fold cross-validation
- Semi-supervised learning with pseudo-labeling
- XGBoost gradient boosting classifier

Author: AI Assistant
Date: 2024
"""

import os
import sys
import pickle
import joblib
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Machine learning imports
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# BioPython for FASTA parsing
from Bio import SeqIO
from Bio.Seq import Seq

# Deep learning imports for protein language models
import torch
from transformers import (
    EsmModel, EsmTokenizer,
    T5EncoderModel, T5Tokenizer
)

warnings.filterwarnings('ignore')

class TerpeneClassifier:
    """
    Main class for Germacrene synthase binary classification
    """
    
    def __init__(self, model_name: str = 'esm2_t33_650M_UR50D', 
                 device: str = 'auto', random_state: int = 42):
        """
        Initialize the classifier
        
        Args:
            model_name: Name of the protein language model to use
            device: Device to run the model on ('auto', 'cpu', 'cuda')
            random_state: Random seed for reproducibility
        """
        self.model_name = model_name
        self.random_state = random_state
        self.device = self._setup_device(device)
        self.plm_model = None
        self.plm_tokenizer = None
        self.xgb_model = None
        self.scaler = StandardScaler()
        self.embedding_dim = None
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup the device for model inference"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        return torch.device(device)
    
    def load_sequences(self, filepath: str) -> pd.DataFrame:
        """
        Load sequences from a FASTA file
        
        Args:
            filepath: Path to the FASTA file
            
        Returns:
            DataFrame with 'sequence' and 'id' columns
        """
        print(f"Loading sequences from {filepath}...")
        
        sequences = []
        ids = []
        
        try:
            for record in SeqIO.parse(filepath, "fasta"):
                sequences.append(str(record.seq))
                ids.append(record.id)
        except Exception as e:
            print(f"Error reading FASTA file: {e}")
            return pd.DataFrame()
        
        df = pd.DataFrame({
            'id': ids,
            'sequence': sequences
        })
        
        print(f"Loaded {len(df)} sequences")
        return df
    
    def parse_marts_db(self, filepath: str) -> pd.DataFrame:
        """
        Parse MARTS-DB file to create product and binary labels
        
        Args:
            filepath: Path to the MARTS-DB file
            
        Returns:
            DataFrame with 'sequence', 'id', 'product', and 'is_germacrene' columns
        """
        print(f"Parsing MARTS-DB file: {filepath}")
        
        # Load the sequences
        df = self.load_sequences(filepath)
        
        if df.empty:
            return df
        
        # Extract product information from sequence IDs
        # This is a simplified approach - in practice, you'd need the actual MARTS-DB format
        df['product'] = df['id'].apply(self._extract_product_from_id)
        
        # Create binary labels for Germacrene
        df['is_germacrene'] = df['product'].apply(
            lambda x: 1 if 'germacrene' in x.lower() else 0
        )
        
        print(f"Found {df['is_germacrene'].sum()} Germacrene synthases out of {len(df)} sequences")
        return df
    
    def _extract_product_from_id(self, seq_id: str) -> str:
        """
        Extract product name from sequence ID
        This is a simplified implementation - adjust based on your MARTS-DB format
        """
        # Common terpene synthase product patterns
        products = [
            'germacrene', 'limonene', 'pinene', 'myrcene', 'ocimene',
            'caryophyllene', 'humulene', 'farnesene', 'bisabolene',
            'selinene', 'eudesmol', 'cadinol', 'cedrol', 'santalene'
        ]
        
        seq_id_lower = seq_id.lower()
        for product in products:
            if product in seq_id_lower:
                return product
        
        return 'unknown'
    
    def generate_embeddings(self, sequences: List[str], model_name: str = None) -> pd.DataFrame:
        """
        Generate protein embeddings using a pre-trained protein language model
        
        Args:
            sequences: List of protein sequences
            model_name: Name of the PLM to use (overrides self.model_name)
            
        Returns:
            DataFrame with sequence IDs and embedding vectors
        """
        if model_name is None:
            model_name = self.model_name
        
        print(f"Generating embeddings using {model_name}...")
        print(f"Processing {len(sequences)} sequences on {self.device}")
        
        # Load the model and tokenizer
        self._load_plm_model(model_name)
        
        embeddings = []
        sequence_ids = []
        
        # Process sequences in batches to manage memory
        batch_size = 8 if self.device.type == 'cuda' else 4
        
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_sequences = sequences[i:i + batch_size]
            batch_ids = [f"seq_{i+j}" for j in range(len(batch_sequences))]
            
            batch_embeddings = self._process_batch(batch_sequences)
            
            embeddings.extend(batch_embeddings)
            sequence_ids.extend(batch_ids)
        
        # Create DataFrame
        embedding_df = pd.DataFrame({
            'id': sequence_ids,
            'embedding': embeddings
        })
        
        # Set embedding dimension
        self.embedding_dim = len(embeddings[0]) if embeddings else None
        
        print(f"Generated embeddings with dimension: {self.embedding_dim}")
        return embedding_df
    
    def _load_plm_model(self, model_name: str):
        """Load the protein language model and tokenizer"""
        try:
            if 'esm' in model_name.lower():
                self._load_esm_model(model_name)
            elif 'prot' in model_name.lower() or 't5' in model_name.lower():
                self._load_prott5_model(model_name)
            else:
                raise ValueError(f"Unsupported model: {model_name}")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise
    
    def _load_esm_model(self, model_name: str):
        """Load ESM model and tokenizer"""
        print(f"Loading ESM model: {model_name}")
        self.plm_tokenizer = EsmTokenizer.from_pretrained(f"facebook/{model_name}")
        self.plm_model = EsmModel.from_pretrained(f"facebook/{model_name}")
        self.plm_model.to(self.device)
        self.plm_model.eval()
    
    def _load_prott5_model(self, model_name: str):
        """Load ProtT5 model and tokenizer"""
        print(f"Loading ProtT5 model: {model_name}")
        self.plm_tokenizer = T5Tokenizer.from_pretrained(f"Rostlab/{model_name}")
        self.plm_model = T5EncoderModel.from_pretrained(f"Rostlab/{model_name}")
        self.plm_model.to(self.device)
        self.plm_model.eval()
    
    def _process_batch(self, sequences: List[str]) -> List[np.ndarray]:
        """Process a batch of sequences to generate embeddings"""
        # Tokenize sequences
        if 'esm' in self.model_name.lower():
            inputs = self.plm_tokenizer(
                sequences, 
                padding=True, 
                truncation=True, 
                max_length=1024,
                return_tensors="pt"
            )
        else:  # ProtT5
            inputs = self.plm_tokenizer(
                sequences, 
                padding=True, 
                truncation=True, 
                max_length=512,
                return_tensors="pt"
            )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.plm_model(**inputs)
            
            # Get the last hidden states
            last_hidden_states = outputs.last_hidden_state
            
            # Average pool over sequence length (excluding padding tokens)
            attention_mask = inputs['attention_mask']
            sequence_lengths = attention_mask.sum(dim=1, keepdim=True)
            
            # Mask out padding tokens and compute mean
            masked_embeddings = last_hidden_states * attention_mask.unsqueeze(-1)
            pooled_embeddings = masked_embeddings.sum(dim=1) / sequence_lengths
            
            # Convert to numpy and return as list
            return [emb.cpu().numpy() for emb in pooled_embeddings]
    
    def train_initial_model(self, X: np.ndarray, y: np.ndarray, 
                           n_splits: int = 5) -> Dict:
        """
        Train initial XGBoost classifier using stratified k-fold cross-validation
        
        Args:
            X: Feature matrix (embeddings)
            y: Target labels
            n_splits: Number of folds for cross-validation
            
        Returns:
            Dictionary containing training results and models
        """
        print(f"Training initial model with stratified {n_splits}-fold CV...")
        
        # Setup stratified k-fold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        # Store results
        results = {
            'f1_scores': [],
            'precision_scores': [],
            'recall_scores': [],
            'auc_pr_scores': [],
            'models': [],
            'predictions': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            print(f"\nFold {fold + 1}/{n_splits}")
            
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Calculate scale_pos_weight for class imbalance
            pos_count = np.sum(y_train)
            neg_count = len(y_train) - pos_count
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
            
            print(f"Training samples: {len(y_train)} (positive: {pos_count}, negative: {neg_count})")
            print(f"Test samples: {len(y_test)}")
            print(f"Scale pos weight: {scale_pos_weight:.2f}")
            
            # Initialize XGBoost classifier
            xgb_model = xgb.XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
                eval_metric='logloss',
                early_stopping_rounds=10,
                verbosity=0
            )
            
            # Train model
            xgb_model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = xgb_model.predict(X_test)
            y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            # Calculate AUC-PR
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
            auc_pr = auc(recall_curve, precision_curve)
            
            print(f"F1-Score: {f1:.3f}")
            print(f"Precision: {precision:.3f}")
            print(f"Recall: {recall:.3f}")
            print(f"AUC-PR: {auc_pr:.3f}")
            
            # Store results
            results['f1_scores'].append(f1)
            results['precision_scores'].append(precision)
            results['recall_scores'].append(recall)
            results['auc_pr_scores'].append(auc_pr)
            results['models'].append(xgb_model)
            results['predictions'].append({
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            })
        
        # Print summary statistics
        print(f"\n=== Cross-Validation Results ===")
        print(f"F1-Score: {np.mean(results['f1_scores']):.3f} ± {np.std(results['f1_scores']):.3f}")
        print(f"Precision: {np.mean(results['precision_scores']):.3f} ± {np.std(results['precision_scores']):.3f}")
        print(f"Recall: {np.mean(results['recall_scores']):.3f} ± {np.std(results['recall_scores']):.3f}")
        print(f"AUC-PR: {np.mean(results['auc_pr_scores']):.3f} ± {np.std(results['auc_pr_scores']):.3f}")
        
        return results
    
    def pseudo_label_data(self, unlabeled_embeddings: np.ndarray, 
                         confidence_threshold: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use the best model to pseudo-label unlabeled data
        
        Args:
            unlabeled_embeddings: Embeddings of unlabeled sequences
            confidence_threshold: Minimum confidence for pseudo-labeling
            
        Returns:
            Tuple of (pseudo_labels, confidence_scores)
        """
        print(f"Pseudo-labeling {len(unlabeled_embeddings)} sequences with threshold {confidence_threshold}")
        
        if not self.xgb_model:
            raise ValueError("No trained model available for pseudo-labeling")
        
        # Get predictions
        pseudo_proba = self.xgb_model.predict_proba(unlabeled_embeddings)[:, 1]
        
        # Create pseudo-labels based on confidence threshold
        pseudo_labels = np.zeros(len(unlabeled_embeddings))
        high_confidence_mask = pseudo_proba >= confidence_threshold
        pseudo_labels[high_confidence_mask] = 1
        
        print(f"Pseudo-labeled {np.sum(pseudo_labels)} sequences as Germacrene synthases")
        print(f"Confidence range: {pseudo_proba.min():.3f} - {pseudo_proba.max():.3f}")
        
        return pseudo_labels, pseudo_proba
    
    def train_final_model(self, X: np.ndarray, y: np.ndarray) -> xgb.XGBClassifier:
        """
        Train the final XGBoost model on the complete dataset
        
        Args:
            X: Complete feature matrix
            y: Complete target labels
            
        Returns:
            Trained XGBoost classifier
        """
        print("Training final model on complete dataset...")
        
        # Calculate scale_pos_weight
        pos_count = np.sum(y)
        neg_count = len(y) - pos_count
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        print(f"Total samples: {len(y)} (positive: {pos_count}, negative: {neg_count})")
        print(f"Scale pos weight: {scale_pos_weight:.2f}")
        
        # Initialize and train final model
        self.xgb_model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            eval_metric='logloss',
            verbosity=0
        )
        
        self.xgb_model.fit(X, y)
        
        print("Final model training completed!")
        return self.xgb_model
    
    def predict_germacrene(self, sequence: str) -> float:
        """
        Predict if a sequence is a Germacrene synthase
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            Confidence score (0-1) for Germacrene synthase prediction
        """
        if not self.xgb_model:
            raise ValueError("No trained model available. Train the model first.")
        
        # Generate embedding for the sequence
        embedding_df = self.generate_embeddings([sequence])
        embedding = embedding_df['embedding'].iloc[0]
        
        # Reshape for prediction
        embedding = embedding.reshape(1, -1)
        
        # Scale the embedding
        embedding_scaled = self.scaler.transform(embedding)
        
        # Make prediction
        confidence = self.xgb_model.predict_proba(embedding_scaled)[0, 1]
        
        return confidence
    
    def save_model(self, filepath: str):
        """Save the trained model and scaler"""
        if not self.xgb_model:
            raise ValueError("No trained model to save")
        
        model_data = {
            'xgb_model': self.xgb_model,
            'scaler': self.scaler,
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model and scaler"""
        model_data = joblib.load(filepath)
        
        self.xgb_model = model_data['xgb_model']
        self.scaler = model_data['scaler']
        self.model_name = model_data['model_name']
        self.embedding_dim = model_data['embedding_dim']
        self.random_state = model_data['random_state']
        
        print(f"Model loaded from {filepath}")
    
    def plot_results(self, results: Dict, save_path: str = None):
        """Plot training results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # F1-Score
        axes[0, 0].boxplot(results['f1_scores'])
        axes[0, 0].set_title('F1-Score Distribution')
        axes[0, 0].set_ylabel('F1-Score')
        
        # Precision
        axes[0, 1].boxplot(results['precision_scores'])
        axes[0, 1].set_title('Precision Distribution')
        axes[0, 1].set_ylabel('Precision')
        
        # Recall
        axes[1, 0].boxplot(results['recall_scores'])
        axes[1, 0].set_title('Recall Distribution')
        axes[1, 0].set_ylabel('Recall')
        
        # AUC-PR
        axes[1, 1].boxplot(results['auc_pr_scores'])
        axes[1, 1].set_title('AUC-PR Distribution')
        axes[1, 1].set_ylabel('AUC-PR')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Results plot saved to {save_path}")
        
        plt.show()


def main():
    """
    Main function to run the complete pipeline
    """
    print("=== Germacrene Synthase Binary Classifier ===")
    
    # Initialize classifier
    classifier = TerpeneClassifier(model_name='esm2_t33_650M_UR50D')
    
    # Step 1: Load and prepare data
    print("\n=== Step 1: Data Loading ===")
    
    # Load MARTS-DB data (you'll need to provide the actual file path)
    marts_file = "data/marts_db.fasta"  # Update this path
    if os.path.exists(marts_file):
        marts_df = classifier.parse_marts_db(marts_file)
    else:
        print(f"MARTS-DB file not found at {marts_file}. Please provide the correct path.")
        return
    
    # Load Uniprot and NCBI data
    uniprot_file = "data/uniprot_sequences.fasta"  # Update this path
    ncbi_file = "data/ncbi_sequences.fasta"  # Update this path
    
    uniprot_df = None
    ncbi_df = None
    
    if os.path.exists(uniprot_file):
        uniprot_df = classifier.load_sequences(uniprot_file)
    else:
        print(f"Uniprot file not found at {uniprot_file}")
    
    if os.path.exists(ncbi_file):
        ncbi_df = classifier.load_sequences(ncbi_file)
    else:
        print(f"NCBI file not found at {ncbi_file}")
    
    # Step 2: Generate embeddings
    print("\n=== Step 2: Feature Engineering ===")
    
    # Generate embeddings for labeled data
    labeled_embeddings = classifier.generate_embeddings(marts_df['sequence'].tolist())
    
    # Combine embeddings with labels
    labeled_data = pd.concat([
        marts_df[['id', 'is_germacrene']].reset_index(drop=True),
        labeled_embeddings['embedding'].apply(pd.Series)
    ], axis=1)
    
    # Prepare features and labels
    feature_columns = [col for col in labeled_data.columns if col not in ['id', 'is_germacrene']]
    X_labeled = labeled_data[feature_columns].values
    y_labeled = labeled_data['is_germacrene'].values
    
    # Step 3: Initial model training
    print("\n=== Step 3: Initial Model Training ===")
    
    # Scale features
    classifier.scaler.fit(X_labeled)
    X_labeled_scaled = classifier.scaler.transform(X_labeled)
    
    # Train initial model
    initial_results = classifier.train_initial_model(X_labeled_scaled, y_labeled)
    
    # Select best model (highest F1-score)
    best_fold = np.argmax(initial_results['f1_scores'])
    classifier.xgb_model = initial_results['models'][best_fold]
    
    print(f"Best model selected from fold {best_fold + 1} with F1-score: {initial_results['f1_scores'][best_fold]:.3f}")
    
    # Step 4: Semi-supervised learning
    print("\n=== Step 4: Semi-Supervised Learning ===")
    
    if uniprot_df is not None or ncbi_df is not None:
        # Combine unlabeled data
        unlabeled_sequences = []
        if uniprot_df is not None:
            unlabeled_sequences.extend(uniprot_df['sequence'].tolist())
        if ncbi_df is not None:
            unlabeled_sequences.extend(ncbi_df['sequence'].tolist())
        
        # Generate embeddings for unlabeled data
        unlabeled_embeddings = classifier.generate_embeddings(unlabeled_sequences)
        
        # Prepare unlabeled features
        unlabeled_features = unlabeled_embeddings['embedding'].apply(pd.Series)
        X_unlabeled = unlabeled_features.values
        X_unlabeled_scaled = classifier.scaler.transform(X_unlabeled)
        
        # Pseudo-label unlabeled data
        pseudo_labels, pseudo_confidence = classifier.pseudo_label_data(X_unlabeled_scaled)
        
        # Combine labeled and pseudo-labeled data
        X_combined = np.vstack([X_labeled_scaled, X_unlabeled_scaled])
        y_combined = np.hstack([y_labeled, pseudo_labels])
        
        # Retrain on combined dataset
        print("\nRetraining on combined dataset...")
        combined_results = classifier.train_initial_model(X_combined, y_combined)
        
        # Select best model from combined training
        best_combined_fold = np.argmax(combined_results['f1_scores'])
        classifier.xgb_model = combined_results['models'][best_combined_fold]
        
        print(f"Best combined model from fold {best_combined_fold + 1} with F1-score: {combined_results['f1_scores'][best_combined_fold]:.3f}")
    
    # Step 5: Final model training
    print("\n=== Step 5: Final Model Training ===")
    
    # Train final model on all available data
    if 'X_combined' in locals():
        final_model = classifier.train_final_model(X_combined, y_combined)
    else:
        final_model = classifier.train_final_model(X_labeled_scaled, y_labeled)
    
    # Save the model
    model_path = "models/germacrene_classifier.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    classifier.save_model(model_path)
    
    # Plot results
    results_path = "results/training_results.png"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    if 'combined_results' in locals():
        classifier.plot_results(combined_results, results_path)
    else:
        classifier.plot_results(initial_results, results_path)
    
    print("\n=== Training Complete ===")
    print(f"Final model saved to: {model_path}")
    print(f"Results plot saved to: {results_path}")
    
    # Example prediction
    print("\n=== Example Prediction ===")
    if len(marts_df) > 0:
        example_sequence = marts_df['sequence'].iloc[0]
        confidence = classifier.predict_germacrene(example_sequence)
        print(f"Example sequence prediction: {confidence:.3f}")
        print(f"Actual label: {marts_df['is_germacrene'].iloc[0]}")


if __name__ == "__main__":
    main()

