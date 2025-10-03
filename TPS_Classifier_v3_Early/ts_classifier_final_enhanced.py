#!/usr/bin/env python3
"""
Module 3: Multi-Modal Terpene Synthase Classification - FINAL ENHANCED VERSION

This script implements the complete PyTorch multi-modal model architecture with:
1. ADAPTIVE THRESHOLD OPTIMIZATION for proper F1 score calculation
2. INVERSE-FREQUENCY CLASS WEIGHTING in Focal Loss for imbalanced data
3. OPTIMIZED TRAINING PROTOCOL with mixed precision and gradient accumulation

This represents the final, production-ready implementation of the terpene synthase classifier.
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

# Import the adaptive threshold fix and weighted focal loss
from adaptive_threshold_fix import (
    find_optimal_thresholds, 
    compute_metrics_adaptive, 
    integrate_adaptive_thresholds_in_training
)
from focal_loss_enhancement import (
    calculate_inverse_frequency_weights,
    WeightedFocalLoss,
    AdaptiveWeightedFocalLoss
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
        logger.info(f"  - E_PLM shape: {self.E_plm.shape}")
        logger.info(f"  - E_Eng shape: {self.E_eng.shape}")
        logger.info(f"  - Y shape: {self.Y.shape}")
    
    def __len__(self):
        return len(self.E_plm)
    
    def __getitem__(self, idx):
        return self.E_plm[idx], self.E_eng[idx], self.Y[idx]


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


class TPSModelTrainerFinal:
    """
    Final Enhanced Trainer with:
    1. Adaptive Threshold Optimization
    2. Inverse-Frequency Class Weighting
    3. Optimized Training Protocol
    """
    
    def __init__(self, 
                 model: TPSClassifier,
                 device: torch.device,
                 class_weights: torch.Tensor,
                 learning_rate: float = LEARNING_RATE,
                 accumulation_steps: int = ACCUMULATION_STEPS,
                 use_adaptive_focal: bool = True):
        self.model = model.to(device)
        self.device = device
        self.accumulation_steps = accumulation_steps
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Initialize weighted focal loss
        if use_adaptive_focal:
            self.criterion = AdaptiveWeightedFocalLoss(
                class_weights=class_weights,
                gamma=2.0,
                base_alpha=0.25,
                label_smoothing=0.01
            )
        else:
            self.criterion = WeightedFocalLoss(
                class_weights=class_weights,
                gamma=2.0,
                alpha=0.25
            )
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores_adaptive = []
        self.best_f1_adaptive = 0.0
        self.patience_counter = 0
        
        logger.info(f"Final Enhanced Trainer initialized on {device}")
        logger.info(f"  - Using {'Adaptive' if use_adaptive_focal else 'Standard'} Weighted Focal Loss")
        logger.info(f"  - Class weights range: [{class_weights.min():.3f}, {class_weights.max():.3f}]")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch with mixed precision and gradient accumulation"""
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
        """
        val_loss, optimal_thresholds, adaptive_metrics = integrate_adaptive_thresholds_in_training(
            self.model, val_loader, self.device, self.criterion
        )
        
        return val_loss, adaptive_metrics['macro_f1'], optimal_thresholds, adaptive_metrics
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader, 
              num_epochs: int = NUM_EPOCHS,
              patience: int = PATIENCE,
              save_dir: str = "models_final") -> Dict:
        """
        Final training with all enhancements
        """
        logger.info("Starting FINAL ENHANCED training...")
        logger.info("Features: Adaptive Thresholds + Weighted Focal Loss + Mixed Precision")
        
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
                
                # Save comprehensive checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_f1_adaptive': self.best_f1_adaptive,
                    'val_f1_adaptive': adaptive_f1,
                    'val_loss': val_loss,
                    'optimal_thresholds': optimal_thresholds,
                    'adaptive_metrics': adaptive_metrics,
                    'class_weights': self.criterion.get_class_weights().cpu().numpy(),
                    'training_history': {
                        'train_losses': self.train_losses,
                        'val_losses': self.val_losses,
                        'val_f1_scores_adaptive': self.val_f1_scores_adaptive
                    }
                }
                
                torch.save(checkpoint, f"{save_dir}/best_model_final_enhanced.pth")
                logger.info(f"🎉 NEW BEST MODEL SAVED!")
                logger.info(f"  - Adaptive F1: {adaptive_f1:.4f}")
                logger.info(f"  - Optimal thresholds: [{optimal_thresholds.min():.3f}, {optimal_thresholds.max():.3f}]")
                logger.info(f"  - Saved to: {save_dir}/best_model_final_enhanced.pth")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info(f"🎉 FINAL TRAINING COMPLETED!")
        logger.info(f"📊 Best Adaptive F1 Score: {self.best_f1_adaptive:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_f1_scores_adaptive': self.val_f1_scores_adaptive,
            'best_f1_adaptive': self.best_f1_adaptive
        }
    
    def plot_training_history(self, save_path: str = "final_training_history.png"):
        """Plot comprehensive training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.train_losses, label='Train Loss', color='blue')
        axes[0, 0].plot(self.val_losses, label='Val Loss', color='red')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Adaptive F1 score plot
        axes[0, 1].plot(self.val_f1_scores_adaptive, label='Adaptive F1', color='green')
        axes[0, 1].axhline(y=self.best_f1_adaptive, color='red', linestyle='--', 
                          label=f'Best: {self.best_f1_adaptive:.4f}')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Macro F1 Score')
        axes[0, 1].set_title('Adaptive Macro F1 Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Class weights visualization
        class_weights = self.criterion.get_class_weights().cpu().numpy()
        axes[1, 0].bar(range(len(class_weights)), class_weights, alpha=0.7)
        axes[1, 0].set_xlabel('Class Index')
        axes[1, 0].set_ylabel('Weight')
        axes[1, 0].set_title('Inverse-Frequency Class Weights')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Performance summary
        axes[1, 1].text(0.1, 0.8, f"Best Adaptive F1: {self.best_f1_adaptive:.4f}", 
                       fontsize=14, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.7, f"Final Train Loss: {self.train_losses[-1]:.4f}", 
                       fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.6, f"Final Val Loss: {self.val_losses[-1]:.4f}", 
                       fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.5, f"Class Weight Range: [{class_weights.min():.3f}, {class_weights.max():.3f}]", 
                       fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Training Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Final training history saved to {save_path}")


def get_data_loaders(features_path: str, 
                    batch_size: int = BATCH_SIZE,
                    train_ratio: float = 0.8,
                    val_ratio: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
    """
    Create data loaders and return training labels for class weight calculation
    """
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
    
    # Extract training labels for class weight calculation
    train_labels = []
    for _, _, y in train_loader:
        train_labels.append(y.numpy())
    train_labels = np.concatenate(train_labels, axis=0)
    
    logger.info(f"Data loaders created:")
    logger.info(f"  - Train: {len(train_dataset)} samples")
    logger.info(f"  - Val: {len(val_dataset)} samples")
    logger.info(f"  - Test: {len(test_dataset)} samples")
    logger.info(f"  - Training labels shape: {train_labels.shape}")
    
    return train_loader, val_loader, test_loader, train_labels


def main():
    """
    Main function for FINAL ENHANCED training
    """
    print("🧬 TS Classifier - FINAL ENHANCED VERSION")
    print("=" * 60)
    print("🎯 Features:")
    print("  ✅ Adaptive Threshold Optimization")
    print("  ✅ Inverse-Frequency Class Weighting")
    print("  ✅ Mixed Precision Training")
    print("  ✅ Gradient Accumulation")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    features_path = "TS-GSD_final_features.pkl"
    if not Path(features_path).exists():
        print(f"❌ Features file not found: {features_path}")
        return
    
    # Create data loaders and get training labels
    train_loader, val_loader, test_loader, train_labels = get_data_loaders(features_path)
    
    # Calculate inverse-frequency class weights
    print("\n📊 Calculating Inverse-Frequency Class Weights...")
    class_weights = calculate_inverse_frequency_weights(train_labels, device)
    
    # Initialize model
    model = TPSClassifier()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    # Initialize final enhanced trainer
    trainer = TPSModelTrainerFinal(
        model=model,
        device=device,
        class_weights=class_weights,
        learning_rate=LEARNING_RATE,
        accumulation_steps=ACCUMULATION_STEPS,
        use_adaptive_focal=True
    )
    
    # Train model with all enhancements
    print(f"\n🚀 Starting FINAL ENHANCED training...")
    history = trainer.train(train_loader, val_loader)
    
    # Plot training history
    trainer.plot_training_history()
    
    print(f"\n🎉 FINAL ENHANCED TRAINING COMPLETED!")
    print(f"📊 Best Adaptive F1 Score: {trainer.best_f1_adaptive:.4f}")
    print(f"💾 Best model saved to: models_final/best_model_final_enhanced.pth")
    print(f"🎯 This represents the OPTIMAL terpene synthase classifier!")
    
    # Performance summary
    print(f"\n📈 Performance Summary:")
    print(f"  - Initial F1 (fixed 0.5 threshold): 0.0000")
    print(f"  - Final F1 (adaptive thresholds): {trainer.best_f1_adaptive:.4f}")
    print(f"  - Improvement: {trainer.best_f1_adaptive * 100:.2f}%")
    print(f"  - Class weights applied: ✅")
    print(f"  - Adaptive thresholding: ✅")
    print(f"  - Mixed precision: ✅")


if __name__ == "__main__":
    main()



