#!/usr/bin/env python3
"""
Complete Multi-Modal Terpene Synthase Classifier

This script implements the final, complete multi-modal classifier that integrates:
1. ESM2 protein language model features (1280D)
2. Engineered biochemical features (64D)  
3. Structural graph features from AlphaFold structures (256D)

This represents the complete implementation of the multi-modal architecture
with all three modalities successfully integrated.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging
from tqdm import tqdm

# Import our custom components
from structural_graph_pipeline import ProteinGraph, GCNEncoder, StructuralGraphProcessor
from ts_classifier_final_enhanced import PLMEncoder, FeatureEncoder
from focal_loss_enhancement import AdaptiveWeightedFocalLoss
from adaptive_threshold_fix import find_optimal_thresholds, compute_metrics_adaptive

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hyperparameters
LATENT_DIM = 256
FUSED_DIM = 768  # 256 (PLM) + 256 (Structural) + 256 (Engineered)
N_CLASSES = 30
BATCH_SIZE = 8  # Smaller batch size for graph data


class CompleteMultiModalClassifier(nn.Module):
    """
    Complete multi-modal terpene synthase classifier
    
    Integrates all three modalities:
    1. ESM2 protein language model features
    2. Engineered biochemical features
    3. Structural graph features from AlphaFold
    """
    
    def __init__(self, 
                 plm_dim: int = 1280,
                 eng_dim: int = 64,
                 latent_dim: int = LATENT_DIM,
                 n_classes: int = N_CLASSES,
                 dropout: float = 0.1):
        """
        Initialize complete multi-modal classifier
        
        Args:
            plm_dim: ESM2 feature dimension
            eng_dim: Engineered feature dimension
            latent_dim: Latent dimension for each modality
            n_classes: Number of functional ensemble classes
            dropout: Dropout rate
        """
        super(CompleteMultiModalClassifier, self).__init__()
        
        # Individual encoders
        self.plm_encoder = PLMEncoder(plm_dim, latent_dim, dropout)
        self.eng_encoder = FeatureEncoder(eng_dim, latent_dim, dropout)
        self.structural_encoder = GCNEncoder(
            input_dim=20,  # Amino acid features
            hidden_dim=128,
            output_dim=latent_dim,
            num_layers=3,
            dropout=dropout
        )
        
        # Fusion layer
        self.fusion_dim = latent_dim * 3  # PLM + Structural + Engineered
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )
        
        self._initialize_weights()
        
        logger.info(f"Complete Multi-Modal Classifier initialized:")
        logger.info(f"  - PLM Encoder: {plm_dim} -> {latent_dim}")
        logger.info(f"  - Structural Encoder: Graph -> {latent_dim}")
        logger.info(f"  - Engineered Encoder: {eng_dim} -> {latent_dim}")
        logger.info(f"  - Fusion: {self.fusion_dim} -> 256 -> {n_classes}")
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, graph_data, e_plm: torch.Tensor, e_eng: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through complete multi-modal architecture
        
        Args:
            graph_data: ProteinGraph object or batch of graphs
            e_plm: ESM2 features (batch_size, 1280)
            e_eng: Engineered features (batch_size, 64)
            
        Returns:
            Logits for functional ensemble classification (batch_size, 30)
        """
        # Encode each modality
        plm_features = self.plm_encoder(e_plm)  # (batch_size, 256)
        eng_features = self.eng_encoder(e_eng)  # (batch_size, 256)
        
        # Handle structural encoding
        if isinstance(graph_data, list):
            # Batch of graphs
            structural_features = []
            for graph in graph_data:
                struct_feat = self.structural_encoder(graph)
                structural_features.append(struct_feat)
            structural_features = torch.cat(structural_features, dim=0)
        else:
            # Single graph
            structural_features = self.structural_encoder(graph_data)
        
        # Ensure all features have the same batch size
        if structural_features.shape[0] != plm_features.shape[0]:
            # Handle batch size mismatch (some graphs might be missing)
            min_batch_size = min(plm_features.shape[0], structural_features.shape[0])
            plm_features = plm_features[:min_batch_size]
            eng_features = eng_features[:min_batch_size]
            structural_features = structural_features[:min_batch_size]
        
        # Fuse all modalities
        fused_features = torch.cat([plm_features, structural_features, eng_features], dim=1)
        
        # Apply fusion layer
        fused_features = self.fusion_layer(fused_features)
        
        # Final classification
        logits = self.classifier(fused_features)
        
        return logits


class CompleteMultiModalDataset(Dataset):
    """
    Dataset for complete multi-modal features
    """
    
    def __init__(self, 
                 features_path: str,
                 graph_data_path: str,
                 manifest_path: str):
        """
        Initialize complete multi-modal dataset
        
        Args:
            features_path: Path to ESM2 + Engineered features
            graph_data_path: Path to protein graph data
            manifest_path: Path to structural manifest
        """
        logger.info("Loading complete multi-modal dataset...")
        
        # Load sequence and engineered features
        with open(features_path, 'rb') as f:
            self.features_data = pickle.load(f)
        
        self.E_plm = torch.FloatTensor(self.features_data['E_plm'])
        self.E_eng = torch.FloatTensor(self.features_data['E_eng'])
        self.Y = torch.FloatTensor(self.features_data['Y'])
        
        # Load graph data
        with open(graph_data_path, 'rb') as f:
            self.graph_data = pickle.load(f)
        
        # Load manifest and filter to high-confidence structures
        self.manifest_df = pd.read_csv(manifest_path)
        self.manifest_df = self.manifest_df[self.manifest_df['confidence_level'] == 'high']
        
        # Create mapping from features to graphs
        self.valid_indices = []
        self.uniprot_ids = self.features_data.get('uniprot_ids', [])
        
        for idx, uniprot_id in enumerate(self.uniprot_ids):
            if uniprot_id in self.graph_data:
                self.valid_indices.append(idx)
        
        logger.info(f"Complete multi-modal dataset loaded:")
        logger.info(f"  - Total samples: {len(self.E_plm)}")
        logger.info(f"  - High-confidence structures: {len(self.manifest_df)}")
        logger.info(f"  - Multi-modal samples: {len(self.valid_indices)}")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        uniprot_id = self.uniprot_ids[actual_idx]
        
        # Get sequence and engineered features
        e_plm = self.E_plm[actual_idx]
        e_eng = self.E_eng[actual_idx]
        y = self.Y[actual_idx]
        
        # Get graph data
        graph = self.graph_data[uniprot_id]
        
        return graph, e_plm, e_eng, y


class CompleteMultiModalTrainer:
    """
    Trainer for complete multi-modal classifier
    """
    
    def __init__(self, 
                 model: CompleteMultiModalClassifier,
                 device: torch.device,
                 class_weights: torch.Tensor,
                 learning_rate: float = 1e-4,
                 accumulation_steps: int = 2):  # Smaller accumulation for graph data
        """
        Initialize complete multi-modal trainer
        """
        self.model = model.to(device)
        self.device = device
        self.accumulation_steps = accumulation_steps
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Loss function
        self.criterion = AdaptiveWeightedFocalLoss(
            class_weights=class_weights,
            gamma=2.0,
            base_alpha=0.25,
            label_smoothing=0.01
        )
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores = []
        self.best_f1 = 0.0
        self.patience_counter = 0
        
        logger.info(f"Complete Multi-Modal Trainer initialized on {device}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (graphs, e_plm, e_eng, y) in enumerate(tqdm(train_loader, desc="Training")):
            e_plm = e_plm.to(self.device)
            e_eng = e_eng.to(self.device)
            y = y.to(self.device)
            
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = self.model(graphs, e_plm, e_eng)
                    loss = self.criterion(logits, y) / self.accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                logits = self.model(graphs, e_plm, e_eng)
                loss = self.criterion(logits, y) / self.accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.accumulation_steps
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate epoch with adaptive thresholds"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for graphs, e_plm, e_eng, y in val_loader:
                e_plm = e_plm.to(self.device)
                e_eng = e_eng.to(self.device)
                y = y.to(self.device)
                
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        logits = self.model(graphs, e_plm, e_eng)
                        loss = self.criterion(logits, y)
                else:
                    logits = self.model(graphs, e_plm, e_eng)
                    loss = self.criterion(logits, y)
                
                total_loss += loss.item()
                
                # Collect predictions
                probabilities = torch.sigmoid(logits)
                all_predictions.append(probabilities.cpu().numpy())
                all_targets.append(y.cpu().numpy())
        
        # Calculate adaptive F1 score
        y_pred_proba = np.concatenate(all_predictions, axis=0)
        y_true = np.concatenate(all_targets, axis=0)
        
        optimal_thresholds = find_optimal_thresholds(y_true, y_pred_proba)
        adaptive_metrics = compute_metrics_adaptive(y_true, y_pred_proba, optimal_thresholds)
        
        return total_loss / len(val_loader), adaptive_metrics['macro_f1']
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader, 
              num_epochs: int = 30,
              patience: int = 10,
              save_dir: str = "models_complete") -> Dict:
        """
        Train the complete multi-modal model
        """
        logger.info("Starting complete multi-modal training...")
        
        Path(save_dir).mkdir(exist_ok=True)
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss, adaptive_f1 = self.validate_epoch(val_loader)
            
            # Update history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_f1_scores.append(adaptive_f1)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                       f"Adaptive F1: {adaptive_f1:.4f}")
            
            # Save best model
            if adaptive_f1 > self.best_f1:
                self.best_f1 = adaptive_f1
                self.patience_counter = 0
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_f1': self.best_f1,
                    'val_f1': adaptive_f1,
                    'val_loss': val_loss,
                    'training_history': {
                        'train_losses': self.train_losses,
                        'val_losses': self.val_losses,
                        'val_f1_scores': self.val_f1_scores
                    }
                }
                
                torch.save(checkpoint, f"{save_dir}/complete_multimodal_best.pth")
                logger.info(f"🎉 NEW BEST MODEL SAVED! F1: {adaptive_f1:.4f}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info(f"Complete multi-modal training completed. Best F1: {self.best_f1:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_f1_scores': self.val_f1_scores,
            'best_f1': self.best_f1
        }


def custom_collate_fn(batch):
    """
    Custom collate function for multi-modal data with protein graphs
    """
    graphs, e_plm_batch, e_eng_batch, y_batch = zip(*batch)
    
    # Stack tensors
    e_plm = torch.stack(e_plm_batch)
    e_eng = torch.stack(e_eng_batch)
    y = torch.stack(y_batch)
    
    # Keep graphs as list
    return list(graphs), e_plm, e_eng, y


def create_complete_multimodal_dataset(features_path: str,
                                     manifest_path: str,
                                     structures_dir: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create complete multi-modal dataset with all three modalities
    """
    logger.info("Creating complete multi-modal dataset...")
    
    # Load manifest
    manifest_df = pd.read_csv(manifest_path)
    
    # Create protein graphs for all high-confidence structures
    logger.info("Creating protein graphs for all structures...")
    graphs = {}
    
    processor = StructuralGraphProcessor()
    high_conf_df = manifest_df[manifest_df['confidence_level'] == 'high']
    
    for idx, row in tqdm(high_conf_df.iterrows(), total=len(high_conf_df), desc="Processing structures"):
        uniprot_id = row['uniprot_id']
        pdb_path = row['file_path']
        
        if Path(pdb_path).exists():
            graph = processor.create_protein_graph(uniprot_id, pdb_path)
            if graph is not None:
                graphs[uniprot_id] = graph
    
    logger.info(f"Created {len(graphs)} protein graphs")
    
    # Save graph data
    graph_data_path = "complete_protein_graphs.pkl"
    with open(graph_data_path, 'wb') as f:
        pickle.dump(graphs, f)
    
    # Create dataset
    dataset = CompleteMultiModalDataset(features_path, graph_data_path, manifest_path)
    
    # Create data loaders
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    
    logger.info(f"Complete multi-modal dataloaders created:")
    logger.info(f"  - Train: {len(train_dataset)} samples")
    logger.info(f"  - Val: {len(val_dataset)} samples")
    logger.info(f"  - Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader, graphs


def demonstrate_complete_multimodal():
    """
    Demonstrate the complete multi-modal classifier
    """
    print("🧬 Complete Multi-Modal Terpene Synthase Classifier")
    print("="*70)
    
    # Configuration
    features_path = "TS-GSD_final_features.pkl"
    manifest_path = "alphafold_structural_manifest.csv"
    structures_dir = "alphafold_structures/pdb"
    
    # Check if files exist
    if not Path(features_path).exists():
        print(f"❌ Features file not found: {features_path}")
        return
    
    if not Path(manifest_path).exists():
        print(f"❌ Manifest file not found: {manifest_path}")
        return
    
    if not Path(structures_dir).exists():
        print(f"❌ Structures directory not found: {structures_dir}")
        return
    
    try:
        # Create complete multi-modal dataset (sample for demo)
        print("\n🔍 Creating complete multi-modal dataset (sample)...")
        
        # Load a small sample for demonstration
        manifest_df = pd.read_csv(manifest_path)
        sample_manifest = manifest_df[manifest_df['confidence_level'] == 'high'].head(3)
        
        # Create sample graphs
        processor = StructuralGraphProcessor()
        sample_graphs = {}
        
        for idx, row in sample_manifest.iterrows():
            uniprot_id = row['uniprot_id']
            pdb_path = row['file_path']
            
            if Path(pdb_path).exists():
                graph = processor.create_protein_graph(uniprot_id, pdb_path)
                if graph is not None:
                    sample_graphs[uniprot_id] = graph
        
        if sample_graphs:
            print(f"✅ Created {len(sample_graphs)} sample graphs")
            
            # Test complete multi-modal model
            print(f"\n🧠 Testing Complete Multi-Modal Model...")
            
            # Initialize model
            model = CompleteMultiModalClassifier()
            
            # Create sample data
            sample_uniprot = list(sample_graphs.keys())[0]
            sample_graph = sample_graphs[sample_uniprot]
            sample_e_plm = torch.randn(1, 1280)  # Sample ESM2 features
            sample_e_eng = torch.randn(1, 64)    # Sample engineered features
            
            # Test forward pass
            with torch.no_grad():
                logits = model(sample_graph, sample_e_plm, sample_e_eng)
                probabilities = torch.sigmoid(logits)
                
                print(f"  - Input graph: {sample_graph.node_features.shape[0]} nodes, {sample_graph.edge_index.shape[1]} edges")
                print(f"  - ESM2 features: {sample_e_plm.shape}")
                print(f"  - Engineered features: {sample_e_eng.shape}")
                print(f"  - Output logits: {logits.shape}")
                print(f"  - Output probabilities: {probabilities.shape}")
                print(f"  - Probability range: [{probabilities.min():.3f}, {probabilities.max():.3f}]")
            
            print(f"\n✅ Complete multi-modal classifier demonstration successful!")
            print(f"🎯 Ready for full-scale multi-modal training!")
            
            # Show model architecture
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\n📊 Model Architecture:")
            print(f"  - Total parameters: {total_params:,}")
            print(f"  - Trainable parameters: {trainable_params:,}")
            print(f"  - Modalities integrated: 3 (ESM2 + Structural + Engineered)")
            
        else:
            print(f"❌ No graphs created successfully")
    
    except Exception as e:
        logger.error(f"Complete multi-modal demonstration failed: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    demonstrate_complete_multimodal()
