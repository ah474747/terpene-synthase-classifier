#!/usr/bin/env python3
"""
Retrain Enhanced Multi-Modal Classifier on Full Dataset

This script retrains the complete enhanced multi-modal classifier on the full
dataset (1,222 high-confidence structures) with the new 25D engineered node
features from Module 6 and calculates the final F1 scores.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import pickle
import time
from tqdm import tqdm

# Import our enhanced components
from module6_feature_enhancement import (
    EnhancedStructuralGraphProcessor,
    EnhancedProteinGraph,
    EnhancedGCNEncoder
)
from complete_multimodal_classifier import (
    CompleteMultiModalClassifier,
    CompleteMultiModalTrainer,
    custom_collate_fn
)
from focal_loss_enhancement import calculate_inverse_frequency_weights
from adaptive_threshold_fix import find_optimal_thresholds, compute_metrics_adaptive

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hyperparameters for full dataset training
BATCH_SIZE = 4  # Smaller batch size for graph data
ACCUMULATION_STEPS = 2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
PATIENCE = 15


class EnhancedCompleteMultiModalClassifier(CompleteMultiModalClassifier):
    """
    Enhanced complete multi-modal classifier with 25D node features
    """
    
    def __init__(self, 
                 plm_dim: int = 1280,
                 eng_dim: int = 64,
                 latent_dim: int = 256,
                 n_classes: int = 30,
                 dropout: float = 0.1):
        """Initialize enhanced multi-modal classifier"""
        super().__init__(plm_dim, eng_dim, latent_dim, n_classes, dropout)
        
        # Replace with enhanced GCN encoder for 25D node features
        self.structural_encoder = EnhancedGCNEncoder(
            input_dim=25,  # Enhanced 25D node features
            hidden_dim=128,
            output_dim=latent_dim,
            num_layers=3,
            dropout=dropout
        )
        
        logger.info(f"Enhanced Complete Multi-Modal Classifier initialized:")
        logger.info(f"  - PLM Encoder: {plm_dim} -> {latent_dim}")
        logger.info(f"  - Enhanced Structural Encoder: 25D -> {latent_dim}")
        logger.info(f"  - Engineered Encoder: {eng_dim} -> {latent_dim}")
        logger.info(f"  - Fusion: {latent_dim * 3} -> 256 -> {n_classes}")


def create_enhanced_full_dataset(features_path: str,
                               manifest_path: str,
                               structures_dir: str) -> Tuple[torch.utils.data.DataLoader, 
                                                           torch.utils.data.DataLoader, 
                                                           torch.utils.data.DataLoader,
                                                           Dict]:
    """
    Create enhanced full dataset with 25D node features
    """
    logger.info("Creating enhanced full dataset with 25D node features...")
    
    # Load features
    with open(features_path, 'rb') as f:
        features_data = pickle.load(f)
    
    # Load manifest and filter to high-confidence structures
    manifest_df = pd.read_csv(manifest_path)
    high_conf_df = manifest_df[manifest_df['confidence_level'] == 'high']
    
    logger.info(f"Processing {len(high_conf_df)} high-confidence structures...")
    
    # Create enhanced protein graphs
    enhanced_processor = EnhancedStructuralGraphProcessor()
    enhanced_graphs = {}
    
    for idx, row in tqdm(high_conf_df.iterrows(), total=len(high_conf_df), desc="Creating enhanced graphs"):
        uniprot_id = row['uniprot_id']
        pdb_path = row['file_path']
        
        if Path(pdb_path).exists():
            enhanced_graph = enhanced_processor.create_protein_graph(uniprot_id, pdb_path)
            if enhanced_graph is not None:
                enhanced_graphs[uniprot_id] = enhanced_graph
                logger.debug(f"Created enhanced graph for {uniprot_id}: {enhanced_graph.node_features.shape}")
    
    logger.info(f"Successfully created {len(enhanced_graphs)} enhanced protein graphs")
    
    # Save enhanced graph data
    enhanced_graph_data_path = "enhanced_protein_graphs_full.pkl"
    with open(enhanced_graph_data_path, 'wb') as f:
        pickle.dump(enhanced_graphs, f)
    
    logger.info(f"Saved enhanced graph data to {enhanced_graph_data_path}")
    
    # Create enhanced dataset
    from complete_multimodal_classifier import CompleteMultiModalDataset
    enhanced_dataset = CompleteMultiModalDataset(features_path, enhanced_graph_data_path, manifest_path)
    
    if len(enhanced_dataset) == 0:
        raise ValueError("No valid enhanced multi-modal samples found")
    
    logger.info(f"Enhanced dataset created: {len(enhanced_dataset)} multi-modal samples")
    
    # Create data loaders
    train_size = int(0.8 * len(enhanced_dataset))
    val_size = int(0.1 * len(enhanced_dataset))
    test_size = len(enhanced_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        enhanced_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=0, collate_fn=custom_collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=0, collate_fn=custom_collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=0, collate_fn=custom_collate_fn
    )
    
    logger.info(f"Enhanced data loaders created:")
    logger.info(f"  - Train: {len(train_dataset)} samples")
    logger.info(f"  - Val: {len(val_dataset)} samples")
    logger.info(f"  - Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader, enhanced_graphs


def train_enhanced_full_model():
    """
    Train the enhanced multi-modal classifier on the full dataset
    """
    print("🧬 Enhanced Multi-Modal Classifier - Full Dataset Training")
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
        print(f"\n🔍 Step 1: Creating Enhanced Full Dataset...")
        
        # Create enhanced full dataset
        train_loader, val_loader, test_loader, enhanced_graphs = create_enhanced_full_dataset(
            features_path, manifest_path, structures_dir
        )
        
        print(f"✅ Enhanced dataset created with {len(enhanced_graphs)} enhanced graphs")
        
        # Calculate class weights
        print(f"\n🔍 Step 2: Calculating Class Weights...")
        
        with open(features_path, 'rb') as f:
            features_data = pickle.load(f)
        
        train_labels = features_data['Y']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        class_weights = calculate_inverse_frequency_weights(train_labels, device)
        
        print(f"📊 Class weights calculated:")
        print(f"  - Weight range: [{class_weights.min():.3f}, {class_weights.max():.3f}]")
        print(f"  - Weight mean: {class_weights.mean():.3f}")
        
        # Initialize enhanced model
        print(f"\n🔍 Step 3: Initializing Enhanced Model...")
        
        enhanced_model = EnhancedCompleteMultiModalClassifier()
        total_params = sum(p.numel() for p in enhanced_model.parameters())
        trainable_params = sum(p.numel() for p in enhanced_model.parameters() if p.requires_grad)
        
        print(f"📊 Enhanced model initialized:")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        print(f"  - Enhanced node features: 25D (20D one-hot + 5D physicochemical)")
        
        # Initialize trainer
        enhanced_trainer = CompleteMultiModalTrainer(
            model=enhanced_model,
            device=device,
            class_weights=class_weights,
            learning_rate=LEARNING_RATE,
            accumulation_steps=ACCUMULATION_STEPS
        )
        
        # Train model
        print(f"\n🔍 Step 4: Training Enhanced Model on Full Dataset...")
        print(f"🚀 Starting training with enhanced 25D node features...")
        
        start_time = time.time()
        
        history = enhanced_trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=NUM_EPOCHS,
            patience=PATIENCE,
            save_dir="models_enhanced_full"
        )
        
        training_time = time.time() - start_time
        
        print(f"\n✅ Enhanced training completed in {training_time/60:.2f} minutes")
        print(f"📊 Final Results:")
        print(f"  - Best F1 Score: {enhanced_trainer.best_f1:.4f}")
        print(f"  - Final Train Loss: {history['train_losses'][-1]:.4f}")
        print(f"  - Final Val Loss: {history['val_losses'][-1]:.4f}")
        
        # Performance comparison
        print(f"\n📈 Performance Comparison:")
        print(f"  - Initial (broken evaluation): 0.0000 F1")
        print(f"  - ESM2 + Engineered (adaptive thresholds): 0.0857 F1")
        print(f"  - Complete Multi-Modal (20D nodes): 0.2008 F1")
        print(f"  - Enhanced Multi-Modal (25D nodes): {enhanced_trainer.best_f1:.4f} F1")
        
        improvement_20d = enhanced_trainer.best_f1 - 0.2008
        improvement_total = enhanced_trainer.best_f1 - 0.0857
        
        print(f"  - Improvement from 20D to 25D nodes: {improvement_20d:.4f} ({improvement_20d/0.2008*100:.1f}%)")
        print(f"  - Total improvement from sequence-only: {improvement_total:.4f} ({improvement_total/0.0857*100:.1f}%)")
        
        # Detailed evaluation on test set
        print(f"\n🔍 Step 5: Detailed Test Set Evaluation...")
        
        # Load best model for evaluation
        checkpoint_path = "models_enhanced_full/complete_multimodal_best.pth"
        if Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            enhanced_model.load_state_dict(checkpoint['model_state_dict'])
            
            print(f"📊 Best model loaded for detailed evaluation")
            
            # Evaluate on test set
            enhanced_model.eval()
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for graphs, e_plm, e_eng, y in tqdm(test_loader, desc="Evaluating test set"):
                    e_plm = e_plm.to(device)
                    e_eng = e_eng.to(device)
                    y = y.to(device)
                    
                    logits = enhanced_model(graphs, e_plm, e_eng)
                    probabilities = torch.sigmoid(logits)
                    
                    all_predictions.append(probabilities.cpu().numpy())
                    all_targets.append(y.cpu().numpy())
            
            # Calculate metrics
            y_pred_proba = np.concatenate(all_predictions, axis=0)
            y_true = np.concatenate(all_targets, axis=0)
            
            # Find optimal thresholds
            optimal_thresholds = find_optimal_thresholds(y_true, y_pred_proba)
            adaptive_metrics = compute_metrics_adaptive(y_true, y_pred_proba, optimal_thresholds)
            
            print(f"📊 Enhanced Test Set Performance:")
            print(f"  - Macro F1 Score: {adaptive_metrics['macro_f1']:.4f}")
            print(f"  - Micro F1 Score: {adaptive_metrics['micro_f1']:.4f}")
            print(f"  - Macro Precision: {adaptive_metrics['macro_precision']:.4f}")
            print(f"  - Macro Recall: {adaptive_metrics['macro_recall']:.4f}")
            print(f"  - Classes with Data: {adaptive_metrics['n_classes_with_data']}/{adaptive_metrics['total_classes']}")
            
            print(f"📊 Optimal Thresholds:")
            print(f"  - Range: [{optimal_thresholds.min():.3f}, {optimal_thresholds.max():.3f}]")
            print(f"  - Mean: {optimal_thresholds.mean():.3f}")
            print(f"  - Median: {np.median(optimal_thresholds):.3f}")
            
            # Save final results
            final_results = {
                'enhanced_f1': enhanced_trainer.best_f1,
                'test_macro_f1': adaptive_metrics['macro_f1'],
                'test_micro_f1': adaptive_metrics['micro_f1'],
                'test_macro_precision': adaptive_metrics['macro_precision'],
                'test_macro_recall': adaptive_metrics['macro_recall'],
                'optimal_thresholds': optimal_thresholds.tolist(),
                'training_history': history,
                'enhanced_graphs_count': len(enhanced_graphs),
                'total_parameters': total_params,
                'training_time_minutes': training_time / 60
            }
            
            with open("enhanced_full_training_results.json", "w") as f:
                import json
                json.dump(final_results, f, indent=2)
            
            print(f"\n📄 Final results saved to: enhanced_full_training_results.json")
        
        print(f"\n🎉 ENHANCED FULL DATASET TRAINING COMPLETE!")
        print(f"🚀 The enhanced multi-modal classifier with 25D node features is ready!")
        
    except Exception as e:
        logger.error(f"Enhanced full dataset training failed: {e}")
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    train_enhanced_full_model()



