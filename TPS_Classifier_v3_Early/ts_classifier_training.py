#!/usr/bin/env python3
"""
Module 3: Streamlined PLM Fusion Model for Terpene Synthase Classification

This script implements the complete PyTorch multi-modal model architecture,
optimized training loop with Focal Loss, AMP, and gradient accumulation.

Architecture: ESM2 (E_PLM) + Engineered Features (E_Eng) → Fusion → Multi-Label Classification
Optimization: Focal Loss, Mixed Precision, Gradient Accumulation for Macro F1 Score
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import json
import os
from pathlib import Path
from typing import Tuple, Dict, Optional
import logging
from tqdm import tqdm
import warnings
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hyperparameters
LATENT_DIM = 256
FUSED_DIM = 512
N_CLASSES = 30
ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 50
PATIENCE = 10


class TSGSDDataset(Dataset):
    """
    Custom PyTorch Dataset for TS-GSD features
    """
    
    def __init__(self, features_path: str):
        """
        Initialize dataset from features file
        
        Args:
            features_path: Path to TS-GSD_final_features.pkl
        """
        logger.info(f"Loading dataset from {features_path}")
        
        with open(features_path, 'rb') as f:
            self.data = pickle.load(f)
        
        # Extract features and targets
        self.E_plm = torch.FloatTensor(self.data['E_plm'])
        self.E_eng = torch.FloatTensor(self.data['E_eng'])
        self.Y = torch.FloatTensor(self.data['Y'])
        
        logger.info(f"Dataset loaded:")
        logger.info(f"  - E_PLM shape: {self.E_plm.shape}")
        logger.info(f"  - E_Eng shape: {self.E_eng.shape}")
        logger.info(f"  - Y shape: {self.Y.shape}")
        logger.info(f"  - Samples: {len(self)}")
    
    def __len__(self):
        return len(self.E_plm)
    
    def __getitem__(self, idx):
        """
        Return E_PLM, E_Eng, and Y tensors for a given index
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (E_PLM, E_Eng, Y) tensors
        """
        return self.E_plm[idx], self.E_eng[idx], self.Y[idx]


class FocalLoss(nn.Module):
    """
    Focal Loss for Multi-Label Classification
    
    Addresses class imbalance by down-weighting easy examples
    and focusing on hard examples.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss
        
        Args:
            alpha: Weighting factor for rare class (default: 0.25)
            gamma: Focusing parameter (default: 2.0)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Loss
        
        Args:
            inputs: Predicted probabilities (N, C)
            targets: Binary targets (N, C)
            
        Returns:
            Focal loss value
        """
        # Compute binary cross-entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Compute p_t (probability of true class)
        p_t = torch.exp(-bce_loss)
        
        # Compute focal weight
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class PLMEncoder(nn.Module):
    """
    Protein Language Model Encoder for ESM2 embeddings
    
    Reduces 1280-D ESM2 embeddings to 256-D latent space
    """
    
    def __init__(self, input_dim: int = 1280, latent_dim: int = LATENT_DIM, dropout: float = 0.1):
        """
        Initialize PLM Encoder
        
        Args:
            input_dim: Input dimension (ESM2 embedding size)
            latent_dim: Output latent dimension
            dropout: Dropout rate
        """
        super(PLMEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: ESM2 embeddings (N, 1280)
            
        Returns:
            Latent representation (N, 256)
        """
        return self.encoder(x)


class FeatureEncoder(nn.Module):
    """
    Engineered Feature Encoder
    
    Projects 64-D engineered features to 256-D latent space
    """
    
    def __init__(self, input_dim: int = 64, latent_dim: int = LATENT_DIM, dropout: float = 0.1):
        """
        Initialize Feature Encoder
        
        Args:
            input_dim: Input dimension (engineered feature size)
            latent_dim: Output latent dimension
            dropout: Dropout rate
        """
        super(FeatureEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Engineered features (N, 64)
            
        Returns:
            Latent representation (N, 256)
        """
        return self.encoder(x)


class TPSClassifier(nn.Module):
    """
    Multi-Modal Terpene Synthase Classifier
    
    Fuses ESM2 embeddings and engineered features for multi-label classification
    """
    
    def __init__(self, 
                 plm_dim: int = 1280,
                 eng_dim: int = 64,
                 latent_dim: int = LATENT_DIM,
                 n_classes: int = N_CLASSES,
                 dropout: float = 0.1):
        """
        Initialize TPS Classifier
        
        Args:
            plm_dim: ESM2 embedding dimension
            eng_dim: Engineered feature dimension
            latent_dim: Latent space dimension
            n_classes: Number of output classes
            dropout: Dropout rate
        """
        super(TPSClassifier, self).__init__()
        
        # Encoders
        self.plm_encoder = PLMEncoder(plm_dim, latent_dim, dropout)
        self.feature_encoder = FeatureEncoder(eng_dim, latent_dim, dropout)
        
        # Fusion and prediction layers
        self.fusion_dim = latent_dim * 2  # Concatenated latent vectors
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize model weights using Xavier initialization
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, e_plm: torch.Tensor, e_eng: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            e_plm: ESM2 embeddings (N, 1280)
            e_eng: Engineered features (N, 64)
            
        Returns:
            Logits for each class (N, 30)
        """
        # Encode both modalities
        plm_latent = self.plm_encoder(e_plm)  # (N, 256)
        eng_latent = self.feature_encoder(e_eng)  # (N, 256)
        
        # Fuse latent representations
        fused = torch.cat([plm_latent, eng_latent], dim=1)  # (N, 512)
        
        # Predict classes
        logits = self.classifier(fused)  # (N, 30)
        
        return logits
    
    def predict_proba(self, e_plm: torch.Tensor, e_eng: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities
        
        Args:
            e_plm: ESM2 embeddings
            e_eng: Engineered features
            
        Returns:
            Class probabilities (N, 30)
        """
        with torch.no_grad():
            logits = self.forward(e_plm, e_eng)
            probabilities = torch.sigmoid(logits)
        return probabilities


class TPSModelTrainer:
    """
    Trainer class for TPS Classifier
    """
    
    def __init__(self, 
                 model: TPSClassifier,
                 device: torch.device,
                 learning_rate: float = LEARNING_RATE,
                 accumulation_steps: int = ACCUMULATION_STEPS):
        """
        Initialize trainer
        
        Args:
            model: TPS Classifier model
            device: PyTorch device
            learning_rate: Learning rate
            accumulation_steps: Gradient accumulation steps
        """
        self.model = model.to(device)
        self.device = device
        self.accumulation_steps = accumulation_steps
        
        # Optimizer and loss
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores = []
        self.best_f1 = 0.0
        self.patience_counter = 0
        
        logger.info(f"Trainer initialized on {device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def compute_metrics(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, float]:
        """
        Compute evaluation metrics for multi-label classification
        
        Args:
            y_pred: Predicted probabilities
            y_true: True labels
            
        Returns:
            Dictionary of metrics
        """
        # Convert to numpy
        y_pred_np = y_pred.cpu().numpy()
        y_true_np = y_true.cpu().numpy()
        
        # Apply threshold for binary predictions
        y_pred_binary = (y_pred_np > 0.5).astype(int)
        
        # For multi-label classification, compute metrics per label then average
        try:
            # Compute F1 scores for each label
            f1_scores = []
            precision_scores = []
            recall_scores = []
            
            for i in range(y_true_np.shape[1]):
                if y_true_np[:, i].sum() > 0:  # Only compute if label exists
                    f1 = f1_score(y_true_np[:, i], y_pred_binary[:, i], zero_division=0)
                    precision = precision_score(y_true_np[:, i], y_pred_binary[:, i], zero_division=0)
                    recall = recall_score(y_true_np[:, i], y_pred_binary[:, i], zero_division=0)
                    
                    f1_scores.append(f1)
                    precision_scores.append(precision)
                    recall_scores.append(recall)
            
            # Average over labels
            macro_f1 = np.mean(f1_scores) if f1_scores else 0.0
            macro_precision = np.mean(precision_scores) if precision_scores else 0.0
            macro_recall = np.mean(recall_scores) if recall_scores else 0.0
            
            # Micro F1 (overall)
            micro_f1 = f1_score(y_true_np.flatten(), y_pred_binary.flatten(), zero_division=0)
            
        except Exception as e:
            logger.warning(f"Error computing metrics: {e}")
            macro_f1 = 0.0
            micro_f1 = 0.0
            macro_precision = 0.0
            macro_recall = 0.0
        
        return {
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall
        }
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (e_plm, e_eng, y) in enumerate(tqdm(train_loader, desc="Training")):
            e_plm = e_plm.to(self.device)
            e_eng = e_eng.to(self.device)
            y = y.to(self.device)
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = self.model(e_plm, e_eng)
                    loss = self.criterion(logits, y) / self.accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                logits = self.model(e_plm, e_eng)
                loss = self.criterion(logits, y) / self.accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.accumulation_steps
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate for one epoch
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (validation loss, macro F1 score)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for e_plm, e_eng, y in tqdm(val_loader, desc="Validation"):
                e_plm = e_plm.to(self.device)
                e_eng = e_eng.to(self.device)
                y = y.to(self.device)
                
                # Forward pass
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        logits = self.model(e_plm, e_eng)
                        loss = self.criterion(logits, y)
                else:
                    logits = self.model(e_plm, e_eng)
                    loss = self.criterion(logits, y)
                
                total_loss += loss.item()
                
                # Collect predictions and targets
                probabilities = torch.sigmoid(logits)
                all_predictions.append(probabilities)
                all_targets.append(y)
        
        # Concatenate all predictions
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute metrics
        metrics = self.compute_metrics(all_predictions, all_targets)
        
        return total_loss / len(val_loader), metrics['macro_f1']
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader, 
              num_epochs: int = NUM_EPOCHS,
              patience: int = PATIENCE,
              save_dir: str = "models") -> Dict:
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            patience: Early stopping patience
            save_dir: Directory to save checkpoints
            
        Returns:
            Training history
        """
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        # Create save directory
        Path(save_dir).mkdir(exist_ok=True)
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_f1 = self.validate_epoch(val_loader)
            
            # Update history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_f1_scores.append(val_f1)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
            
            # Save best model
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.patience_counter = 0
                
                # Save checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_f1': self.best_f1,
                    'val_f1': val_f1,
                    'val_loss': val_loss
                }
                
                torch.save(checkpoint, f"{save_dir}/best_model.pth")
                logger.info(f"New best model saved with F1: {val_f1:.4f}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info(f"Training completed. Best F1: {self.best_f1:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_f1_scores': self.val_f1_scores,
            'best_f1': self.best_f1
        }
    
    def plot_training_history(self, save_path: str = "training_history.png"):
        """
        Plot training history
        
        Args:
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(self.train_losses, label='Train Loss')
        axes[0].plot(self.val_losses, label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # F1 score plot
        axes[1].plot(self.val_f1_scores, label='Val F1', color='green')
        axes[1].axhline(y=self.best_f1, color='red', linestyle='--', label=f'Best F1: {self.best_f1:.4f}')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Macro F1 Score')
        axes[1].set_title('Validation Macro F1 Score')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Training history saved to {save_path}")


def get_data_loaders(features_path: str, 
                    batch_size: int = BATCH_SIZE,
                    train_ratio: float = 0.8,
                    val_ratio: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing
    
    Args:
        features_path: Path to features file
        batch_size: Batch size
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load full dataset
    full_dataset = TSGSDDataset(features_path)
    
    # Split dataset
    n_samples = len(full_dataset)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    logger.info(f"Data loaders created:")
    logger.info(f"  - Train: {len(train_dataset)} samples")
    logger.info(f"  - Val: {len(val_dataset)} samples")
    logger.info(f"  - Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader


def main():
    """Main function to run the training pipeline"""
    
    print("🧬 TS Classifier Training Pipeline - Module 3")
    print("=" * 60)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load features
    features_path = "TS-GSD_final_features.pkl"
    if not Path(features_path).exists():
        print(f"❌ Features file not found: {features_path}")
        print("Please run Module 2 first to generate features")
        return
    
    # Create data loaders
    train_loader, val_loader, test_loader = get_data_loaders(features_path)
    
    # Initialize model
    model = TPSClassifier()
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Initialize trainer
    trainer = TPSModelTrainer(model, device)
    
    # Train model
    history = trainer.train(train_loader, val_loader)
    
    # Plot training history
    trainer.plot_training_history()
    
    print(f"\n🎉 Training completed!")
    print(f"📊 Best Macro F1 Score: {trainer.best_f1:.4f}")
    print(f"💾 Best model saved to: models/best_model.pth")
    print(f"🎯 Ready for evaluation and deployment!")


if __name__ == "__main__":
    main()
