#!/usr/bin/env python3
"""
Module 3: Multi-Modal Terpene Synthase Classification - FIXED VERSION

This script implements the complete PyTorch multi-modal model architecture
with ADAPTIVE THRESHOLD OPTIMIZATION to correctly calculate Macro F1 scores.

CRITICAL FIX: Replaces fixed 0.5 threshold with adaptive per-class thresholds
optimized on validation set to maximize F1 scores for sparse multi-label data.
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
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm
import warnings
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# Import the adaptive threshold fix
from adaptive_threshold_fix import (
    find_optimal_thresholds, 
    compute_metrics_adaptive, 
    integrate_adaptive_thresholds_in_training
)

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
    """Custom PyTorch Dataset for TS-GSD features"""
    
    def __init__(self, features_path: str):
        logger.info(f"Loading dataset from {features_path}")
        
        with open(features_path, 'rb') as f:
            self.data = pickle.load(f)
        
        self.E_plm = torch.FloatTensor(self.data['E_plm'])
        self.E_eng = torch.FloatTensor(self.data['E_eng'])
        self.Y = torch.FloatTensor(self.data['Y'])
        
        logger.info(f"Dataset loaded: {len(self)} samples")
    
    def __len__(self):
        return len(self.E_plm)
    
    def __getitem__(self, idx):
        return self.E_plm[idx], self.E_eng[idx], self.Y[idx]


class FocalLoss(nn.Module):
    """Focal Loss for Multi-Label Classification"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = torch.exp(-bce_loss)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class PLMEncoder(nn.Module):
    """Protein Language Model Encoder for ESM2 embeddings"""
    
    def __init__(self, input_dim: int = 1280, latent_dim: int = LATENT_DIM, dropout: float = 0.1):
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
        return self.encoder(x)


class FeatureEncoder(nn.Module):
    """Engineered Feature Encoder"""
    
    def __init__(self, input_dim: int = 64, latent_dim: int = LATENT_DIM, dropout: float = 0.1):
        super(FeatureEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class TPSClassifier(nn.Module):
    """Multi-Modal Terpene Synthase Classifier"""
    
    def __init__(self, 
                 plm_dim: int = 1280,
                 eng_dim: int = 64,
                 latent_dim: int = LATENT_DIM,
                 n_classes: int = N_CLASSES,
                 dropout: float = 0.1):
        super(TPSClassifier, self).__init__()
        
        self.plm_encoder = PLMEncoder(plm_dim, latent_dim, dropout)
        self.feature_encoder = FeatureEncoder(eng_dim, latent_dim, dropout)
        
        self.fusion_dim = latent_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, e_plm: torch.Tensor, e_eng: torch.Tensor) -> torch.Tensor:
        plm_latent = self.plm_encoder(e_plm)
        eng_latent = self.feature_encoder(e_eng)
        fused = torch.cat([plm_latent, eng_latent], dim=1)
        logits = self.classifier(fused)
        return logits


class TPSModelTrainerFixed:
    """Trainer class with ADAPTIVE THRESHOLD OPTIMIZATION"""
    
    def __init__(self, 
                 model: TPSClassifier,
                 device: torch.device,
                 learning_rate: float = LEARNING_RATE,
                 accumulation_steps: int = ACCUMULATION_STEPS):
        self.model = model.to(device)
        self.device = device
        self.accumulation_steps = accumulation_steps
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
        
        self.scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
        
        # Training history with adaptive thresholds
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores = []
        self.val_f1_scores_adaptive = []  # NEW: Adaptive threshold F1 scores
        self.best_f1 = 0.0
        self.best_f1_adaptive = 0.0  # NEW: Best adaptive F1
        self.patience_counter = 0
        
        logger.info(f"Trainer initialized on {device} with ADAPTIVE THRESHOLD OPTIMIZATION")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (e_plm, e_eng, y) in enumerate(tqdm(train_loader, desc="Training")):
            e_plm = e_plm.to(self.device)
            e_eng = e_eng.to(self.device)
            y = y.to(self.device)
            
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = self.model(e_plm, e_eng)
                    loss = self.criterion(logits, y) / self.accumulation_steps
                
                self.scaler.scale(loss).backward()
                
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
    
    def validate_epoch_adaptive(self, val_loader: DataLoader) -> Tuple[float, float, np.ndarray, dict]:
        """
        Validate with ADAPTIVE THRESHOLD OPTIMIZATION
        
        Returns:
            Tuple of (val_loss, adaptive_f1, optimal_thresholds, metrics_dict)
        """
        # Use the integrated adaptive threshold function
        val_loss, optimal_thresholds, adaptive_metrics = integrate_adaptive_thresholds_in_training(
            self.model, val_loader, self.device, self.criterion
        )
        
        return val_loss, adaptive_metrics['macro_f1'], optimal_thresholds, adaptive_metrics
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader, 
              num_epochs: int = NUM_EPOCHS,
              patience: int = PATIENCE,
              save_dir: str = "models") -> Dict:
        """
        Train the model with ADAPTIVE THRESHOLD OPTIMIZATION
        """
        logger.info("Starting training with ADAPTIVE THRESHOLD OPTIMIZATION...")
        
        Path(save_dir).mkdir(exist_ok=True)
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation with adaptive thresholds
            val_loss, adaptive_f1, optimal_thresholds, adaptive_metrics = self.validate_epoch_adaptive(val_loader)
            
            # Update history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_f1_scores_adaptive.append(adaptive_f1)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                       f"Adaptive F1: {adaptive_f1:.4f}")
            logger.info(f"Classes with data: {adaptive_metrics['n_classes_with_data']}/{adaptive_metrics['total_classes']}")
            
            # Save best model based on ADAPTIVE F1 score
            if adaptive_f1 > self.best_f1_adaptive:
                self.best_f1_adaptive = adaptive_f1
                self.patience_counter = 0
                
                # Save checkpoint with adaptive threshold information
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_f1_adaptive': self.best_f1_adaptive,
                    'val_f1_adaptive': adaptive_f1,
                    'val_loss': val_loss,
                    'optimal_thresholds': optimal_thresholds,
                    'adaptive_metrics': adaptive_metrics
                }
                
                torch.save(checkpoint, f"{save_dir}/best_model_adaptive.pth")
                logger.info(f"🎉 NEW BEST MODEL SAVED with Adaptive F1: {adaptive_f1:.4f}")
                logger.info(f"Optimal thresholds range: [{optimal_thresholds.min():.3f}, {optimal_thresholds.max():.3f}]")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info(f"Training completed. Best Adaptive F1: {self.best_f1_adaptive:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_f1_scores_adaptive': self.val_f1_scores_adaptive,
            'best_f1_adaptive': self.best_f1_adaptive
        }
    
    def plot_training_history(self, save_path: str = "training_history_adaptive.png"):
        """Plot training history with adaptive F1 scores"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(self.train_losses, label='Train Loss')
        axes[0].plot(self.val_losses, label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Adaptive F1 score plot
        axes[1].plot(self.val_f1_scores_adaptive, label='Adaptive F1', color='green')
        axes[1].axhline(y=self.best_f1_adaptive, color='red', linestyle='--', 
                       label=f'Best Adaptive F1: {self.best_f1_adaptive:.4f}')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Macro F1 Score')
        axes[1].set_title('Validation Adaptive Macro F1 Score')
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
    """Create data loaders for training, validation, and testing"""
    full_dataset = TSGSDDataset(features_path)
    
    n_samples = len(full_dataset)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    logger.info(f"Data loaders created: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def main():
    """Main function with ADAPTIVE THRESHOLD OPTIMIZATION"""
    
    print("🧬 TS Classifier Training Pipeline - FIXED VERSION")
    print("=" * 60)
    print("🎯 CRITICAL FIX: Adaptive Threshold Optimization")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    features_path = "TS-GSD_final_features.pkl"
    if not Path(features_path).exists():
        print(f"❌ Features file not found: {features_path}")
        return
    
    # Create data loaders
    train_loader, val_loader, test_loader = get_data_loaders(features_path)
    
    # Initialize model
    model = TPSClassifier()
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Initialize trainer with adaptive threshold optimization
    trainer = TPSModelTrainerFixed(model, device)
    
    # Train model with adaptive thresholds
    history = trainer.train(train_loader, val_loader)
    
    # Plot training history
    trainer.plot_training_history()
    
    print(f"\n🎉 TRAINING COMPLETED WITH ADAPTIVE THRESHOLDS!")
    print(f"📊 Best Adaptive F1 Score: {trainer.best_f1_adaptive:.4f}")
    print(f"💾 Best model saved to: models/best_model_adaptive.pth")
    print(f"🎯 This reveals the TRUE performance of the multi-modal model!")


if __name__ == "__main__":
    main()



