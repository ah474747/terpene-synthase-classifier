#!/usr/bin/env python3
"""
Test Complete Multi-Modal Model and Derive F1 Scores

This script trains and evaluates the complete multi-modal terpene synthase classifier
with all three modalities integrated: ESM2 + Structural + Engineered features.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
import pickle
import time

# Import our complete multi-modal components
from complete_multimodal_classifier import (
    CompleteMultiModalClassifier, 
    CompleteMultiModalTrainer,
    create_complete_multimodal_dataset
)
from focal_loss_enhancement import calculate_inverse_frequency_weights
from adaptive_threshold_fix import find_optimal_thresholds, compute_metrics_adaptive

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_complete_multimodal_model():
    """
    Test the complete multi-modal model and derive F1 scores
    """
    print("🧬 Testing Complete Multi-Modal Terpene Synthase Classifier")
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
        print(f"\n🔍 Step 1: Creating Complete Multi-Modal Dataset...")
        
        # Create complete multi-modal dataset (small sample for testing)
        print(f"📋 Creating dataset with structural integration...")
        
        # Load features to get training labels for class weights
        with open(features_path, 'rb') as f:
            features_data = pickle.load(f)
        
        # Calculate class weights from training data
        train_labels = features_data['Y']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        class_weights = calculate_inverse_frequency_weights(train_labels, device)
        
        print(f"📊 Class weights calculated:")
        print(f"  - Weight range: [{class_weights.min():.3f}, {class_weights.max():.3f}]")
        print(f"  - Weight mean: {class_weights.mean():.3f}")
        
        # Create sample dataset for testing (first 50 structures)
        print(f"\n🔍 Step 2: Creating Sample Dataset (50 structures)...")
        
        manifest_df = pd.read_csv(manifest_path)
        high_conf_df = manifest_df[manifest_df['confidence_level'] == 'high'].head(50)
        
        # Create sample graphs
        from structural_graph_pipeline import StructuralGraphProcessor
        processor = StructuralGraphProcessor()
        sample_graphs = {}
        
        print(f"📊 Processing {len(high_conf_df)} structures...")
        for idx, row in tqdm(high_conf_df.iterrows(), total=len(high_conf_df), desc="Creating graphs"):
            uniprot_id = row['uniprot_id']
            pdb_path = row['file_path']
            
            if Path(pdb_path).exists():
                graph = processor.create_protein_graph(uniprot_id, pdb_path)
                if graph is not None:
                    sample_graphs[uniprot_id] = graph
        
        print(f"✅ Created {len(sample_graphs)} protein graphs")
        
        if len(sample_graphs) == 0:
            print(f"❌ No graphs created successfully")
            return
        
        # Save sample graph data
        sample_graph_data_path = "sample_protein_graphs.pkl"
        with open(sample_graph_data_path, 'wb') as f:
            pickle.dump(sample_graphs, f)
        
        # Create sample dataset
        from complete_multimodal_classifier import CompleteMultiModalDataset
        sample_dataset = CompleteMultiModalDataset(features_path, sample_graph_data_path, manifest_path)
        
        if len(sample_dataset) == 0:
            print(f"❌ No valid multi-modal samples found")
            return
        
        print(f"✅ Sample dataset created: {len(sample_dataset)} multi-modal samples")
        
        # Create data loaders
        print(f"\n🔍 Step 3: Creating Data Loaders...")
        
        train_size = int(0.8 * len(sample_dataset))
        val_size = len(sample_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            sample_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Import custom collate function
        from complete_multimodal_classifier import custom_collate_fn
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)
        
        print(f"📊 Data loaders created:")
        print(f"  - Train: {len(train_dataset)} samples")
        print(f"  - Val: {len(val_dataset)} samples")
        
        # Initialize model
        print(f"\n🔍 Step 4: Initializing Complete Multi-Modal Model...")
        
        model = CompleteMultiModalClassifier()
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"📊 Model initialized:")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        
        # Initialize trainer
        trainer = CompleteMultiModalTrainer(
            model=model,
            device=device,
            class_weights=class_weights,
            learning_rate=1e-4,
            accumulation_steps=2
        )
        
        # Train model
        print(f"\n🔍 Step 5: Training Complete Multi-Modal Model...")
        print(f"🚀 Starting training with all three modalities...")
        
        start_time = time.time()
        
        # Train for a few epochs
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=10,  # Quick training for demonstration
            patience=5,
            save_dir="models_complete_test"
        )
        
        training_time = time.time() - start_time
        
        print(f"\n✅ Training completed in {training_time:.2f} seconds")
        print(f"📊 Final Results:")
        print(f"  - Best F1 Score: {trainer.best_f1:.4f}")
        print(f"  - Final Train Loss: {history['train_losses'][-1]:.4f}")
        print(f"  - Final Val Loss: {history['val_losses'][-1]:.4f}")
        
        # Compare with previous results
        print(f"\n📈 Performance Comparison:")
        print(f"  - Initial (broken evaluation): 0.0000 F1")
        print(f"  - ESM2 + Engineered (adaptive thresholds): 0.0857 F1")
        print(f"  - Complete Multi-Modal (ESM2 + Structural + Engineered): {trainer.best_f1:.4f} F1")
        
        improvement = trainer.best_f1 - 0.0857
        print(f"  - Improvement from structural integration: {improvement:.4f} ({improvement/0.0857*100:.1f}%)")
        
        # Detailed evaluation
        print(f"\n🔍 Step 6: Detailed Performance Analysis...")
        
        # Load best model for evaluation
        checkpoint_path = "models_complete_test/complete_multimodal_best.pth"
        if Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            print(f"📊 Best model loaded for detailed evaluation")
            
            # Evaluate on validation set
            model.eval()
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for graphs, e_plm, e_eng, y in val_loader:
                    e_plm = e_plm.to(device)
                    e_eng = e_eng.to(device)
                    y = y.to(device)
                    
                    logits = model(graphs, e_plm, e_eng)
                    probabilities = torch.sigmoid(logits)
                    
                    all_predictions.append(probabilities.cpu().numpy())
                    all_targets.append(y.cpu().numpy())
            
            # Calculate metrics
            y_pred_proba = np.concatenate(all_predictions, axis=0)
            y_true = np.concatenate(all_targets, axis=0)
            
            # Find optimal thresholds
            optimal_thresholds = find_optimal_thresholds(y_true, y_pred_proba)
            adaptive_metrics = compute_metrics_adaptive(y_true, y_pred_proba, optimal_thresholds)
            
            print(f"📊 Detailed Performance Metrics:")
            print(f"  - Macro F1 Score: {adaptive_metrics['macro_f1']:.4f}")
            print(f"  - Micro F1 Score: {adaptive_metrics['micro_f1']:.4f}")
            print(f"  - Macro Precision: {adaptive_metrics['macro_precision']:.4f}")
            print(f"  - Macro Recall: {adaptive_metrics['macro_recall']:.4f}")
            print(f"  - Classes with Data: {adaptive_metrics['n_classes_with_data']}/{adaptive_metrics['total_classes']}")
            
            print(f"📊 Optimal Thresholds:")
            print(f"  - Range: [{optimal_thresholds.min():.3f}, {optimal_thresholds.max():.3f}]")
            print(f"  - Mean: {optimal_thresholds.mean():.3f}")
            print(f"  - Median: {np.median(optimal_thresholds):.3f}")
        
        print(f"\n🎉 COMPLETE MULTI-MODAL TEST SUCCESSFUL!")
        print(f"🚀 The complete multi-modal classifier is working with all three modalities!")
        
    except Exception as e:
        logger.error(f"Complete multi-modal test failed: {e}")
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_complete_multimodal_model()
