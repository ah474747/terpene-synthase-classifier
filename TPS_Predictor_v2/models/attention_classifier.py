"""
Attention-Based Multi-Class Classifier for Terpene Product Prediction

This module implements an attention-based neural network for predicting
terpene products from protein sequences using SaProt embeddings and
molecular fingerprints.

Based on research best practices for enzyme product prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import logging
from pathlib import Path
import pickle
from dataclasses import dataclass
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for the attention model"""
    protein_embedding_dim: int = 1280  # SaProt embedding dimension
    molecular_fingerprint_dim: int = 2223  # Combined fingerprint dimension
    hidden_dim: int = 512
    num_attention_heads: int = 8
    num_classes: int = 10
    dropout_rate: float = 0.3
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    early_stopping_patience: int = 10

class AttentionLayer(nn.Module):
    """Multi-head attention layer for protein embeddings"""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads"
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        
        # Linear transformations
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # Output projection
        output = self.out_linear(attended)
        output = self.dropout(output)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + x)
        
        return output, attention_weights

class TerpenePredictor(nn.Module):
    """Attention-based terpene product predictor"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Protein embedding processing
        self.protein_projection = nn.Linear(config.protein_embedding_dim, config.hidden_dim)
        self.protein_attention = AttentionLayer(config.hidden_dim, config.num_attention_heads)
        
        # Molecular fingerprint processing
        self.molecular_projection = nn.Linear(config.molecular_fingerprint_dim, config.hidden_dim)
        
        # Fusion layer
        self.fusion_layer = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 4, config.num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, protein_embeddings, molecular_fingerprints):
        """
        Forward pass
        
        Args:
            protein_embeddings: [batch_size, protein_embedding_dim]
            molecular_fingerprints: [batch_size, molecular_fingerprint_dim]
        
        Returns:
            logits: [batch_size, num_classes]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size = protein_embeddings.size(0)
        
        # Process protein embeddings
        protein_proj = self.protein_projection(protein_embeddings)
        protein_proj = protein_proj.unsqueeze(1)  # Add sequence dimension
        
        # Apply attention (treating single embedding as sequence of length 1)
        protein_attended, attention_weights = self.protein_attention(protein_proj)
        protein_features = protein_attended.squeeze(1)  # Remove sequence dimension
        
        # Process molecular fingerprints
        molecular_features = self.molecular_projection(molecular_fingerprints)
        
        # Fusion
        combined_features = torch.cat([protein_features, molecular_features], dim=1)
        fused_features = self.fusion_layer(combined_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits, attention_weights

class TerpenePredictorTrainer:
    """Trainer for the terpene predictor model"""
    
    def __init__(self, config: ModelConfig, device: str = "auto"):
        self.config = config
        self.device = self._setup_device(device)
        
        # Initialize model
        self.model = TerpenePredictor(config).to(self.device)
        
        # Initialize optimizer and loss
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-5
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Training history
        self.train_history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Label encoder
        self.label_encoder = LabelEncoder()
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        logger.info(f"Using device: {device}")
        return torch.device(device)
    
    def prepare_data(self, 
                    protein_embeddings: np.ndarray,
                    molecular_fingerprints: np.ndarray,
                    labels: List[str],
                    test_size: float = 0.2,
                    random_state: int = 42) -> Tuple[torch.Tensor, ...]:
        """Prepare data for training"""
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Split data
        X_protein_train, X_protein_test, X_mol_train, X_mol_test, y_train, y_test = train_test_split(
            protein_embeddings, molecular_fingerprints, encoded_labels,
            test_size=test_size, random_state=random_state, stratify=encoded_labels
        )
        
        # Convert to tensors
        X_protein_train = torch.FloatTensor(X_protein_train).to(self.device)
        X_protein_test = torch.FloatTensor(X_protein_test).to(self.device)
        X_mol_train = torch.FloatTensor(X_mol_train).to(self.device)
        X_mol_test = torch.FloatTensor(X_mol_test).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        y_test = torch.LongTensor(y_test).to(self.device)
        
        logger.info(f"Training set: {len(X_protein_train)} samples")
        logger.info(f"Test set: {len(X_protein_test)} samples")
        logger.info(f"Number of classes: {len(self.label_encoder.classes_)}")
        
        return X_protein_train, X_protein_test, X_mol_train, X_mol_test, y_train, y_test
    
    def train_epoch(self, X_protein, X_mol, y):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        # Create batches
        num_batches = len(X_protein) // self.config.batch_size
        if len(X_protein) % self.config.batch_size != 0:
            num_batches += 1
        
        for i in range(num_batches):
            start_idx = i * self.config.batch_size
            end_idx = min((i + 1) * self.config.batch_size, len(X_protein))
            
            batch_protein = X_protein[start_idx:end_idx]
            batch_mol = X_mol[start_idx:end_idx]
            batch_y = y[start_idx:end_idx]
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, _ = self.model(batch_protein, batch_mol)
            loss = self.criterion(logits, batch_y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        avg_loss = total_loss / num_batches
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, X_protein, X_mol, y):
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            num_batches = len(X_protein) // self.config.batch_size
            if len(X_protein) % self.config.batch_size != 0:
                num_batches += 1
            
            for i in range(num_batches):
                start_idx = i * self.config.batch_size
                end_idx = min((i + 1) * self.config.batch_size, len(X_protein))
                
                batch_protein = X_protein[start_idx:end_idx]
                batch_mol = X_mol[start_idx:end_idx]
                batch_y = y[start_idx:end_idx]
                
                logits, _ = self.model(batch_protein, batch_mol)
                loss = self.criterion(logits, batch_y)
                
                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        avg_loss = total_loss / num_batches
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, X_protein_train, X_mol_train, y_train, X_protein_val, X_mol_val, y_val):
        """Train the model"""
        logger.info("Starting training...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(X_protein_train, X_mol_train, y_train)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(X_protein_val, X_mol_val, y_val)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Store history
            self.train_history['loss'].append(train_loss)
            self.train_history['accuracy'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_accuracy'].append(val_acc)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.save_model("best_model.pth")
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
                logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        logger.info("Training completed!")
    
    def evaluate(self, X_protein_test, X_mol_test, y_test):
        """Evaluate the model"""
        self.model.eval()
        
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            num_batches = len(X_protein_test) // self.config.batch_size
            if len(X_protein_test) % self.config.batch_size != 0:
                num_batches += 1
            
            for i in range(num_batches):
                start_idx = i * self.config.batch_size
                end_idx = min((i + 1) * self.config.batch_size, len(X_protein_test))
                
                batch_protein = X_protein_test[start_idx:end_idx]
                batch_mol = X_mol_test[start_idx:end_idx]
                
                logits, _ = self.model(batch_protein, batch_mol)
                probabilities = F.softmax(logits, dim=1)
                _, predicted = torch.max(logits, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Convert predictions back to labels
        predicted_labels = self.label_encoder.inverse_transform(all_predictions)
        true_labels = self.label_encoder.inverse_transform(y_test.cpu().numpy())
        
        # Calculate metrics
        accuracy = (np.array(all_predictions) == y_test.cpu().numpy()).mean()
        
        # Classification report
        report = classification_report(true_labels, predicted_labels, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        
        return {
            'accuracy': accuracy,
            'predictions': predicted_labels,
            'probabilities': np.array(all_probabilities),
            'true_labels': true_labels,
            'classification_report': report,
            'confusion_matrix': cm
        }
    
    def save_model(self, filename: str):
        """Save model and related components"""
        save_path = Path("data/cache") / filename
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'label_encoder': self.label_encoder,
            'train_history': self.train_history
        }, save_path)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, filename: str):
        """Load model and related components"""
        load_path = Path("data/cache") / filename
        
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['config']
        self.label_encoder = checkpoint['label_encoder']
        self.train_history = checkpoint['train_history']
        
        logger.info(f"Model loaded from {load_path}")

def main():
    """Main function to demonstrate the attention model"""
    logger.info("Starting attention-based terpene predictor...")
    
    # Create sample data
    num_samples = 1000
    protein_embedding_dim = 1280
    molecular_fingerprint_dim = 2223
    
    # Generate random data for demonstration
    protein_embeddings = np.random.randn(num_samples, protein_embedding_dim)
    molecular_fingerprints = np.random.randn(num_samples, molecular_fingerprint_dim)
    
    # Create sample labels
    products = ['limonene', 'pinene', 'myrcene', 'linalool', 'germacrene_a', 
                'germacrene_d', 'caryophyllene', 'humulene', 'farnesene', 'bisabolene']
    labels = np.random.choice(products, num_samples)
    
    # Create model config
    config = ModelConfig(
        protein_embedding_dim=protein_embedding_dim,
        molecular_fingerprint_dim=molecular_fingerprint_dim,
        num_classes=len(products)
    )
    
    # Initialize trainer
    trainer = TerpenePredictorTrainer(config)
    
    # Prepare data
    X_protein_train, X_protein_test, X_mol_train, X_mol_test, y_train, y_test = trainer.prepare_data(
        protein_embeddings, molecular_fingerprints, labels
    )
    
    # Split training data for validation
    X_protein_train, X_protein_val, X_mol_train, X_mol_val, y_train, y_val = train_test_split(
        X_protein_train, X_mol_train, y_train, test_size=0.2, random_state=42
    )
    
    # Train model
    trainer.train(X_protein_train, X_mol_train, y_train, X_protein_val, X_mol_val, y_val)
    
    # Evaluate model
    results = trainer.evaluate(X_protein_test, X_mol_test, y_test)
    
    # Print results
    print(f"\nModel Evaluation Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Classification Report:")
    print(classification_report(results['true_labels'], results['predictions']))

if __name__ == "__main__":
    main()
