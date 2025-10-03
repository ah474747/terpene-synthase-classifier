#!/usr/bin/env python3
"""
Batch inference with stabilized TPS classifier
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'terpene_classifier_v3'))

import json
import numpy as np
import torch
from tps.models.multimodal import FinalMultiModalClassifier
from tps.features.engineered import generate_engineered_features
from tps.features.structure import generate_gcn_features
from tps.retrieval.knn_head import KNNBlender
from tps.hierarchy.head import HierarchyHead
from tps.eval.calibration import CalibratedPredictor
from tps.utils import set_seed


def load_model(model_path: str):
    """Load the trained model"""
    model = FinalMultiModalClassifier()
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def load_knn_index(index_path: str):
    """Load kNN index"""
    import pickle
    with open(index_path, 'rb') as f:
        data = pickle.load(f)
    return data['blender']


def load_calibration(calibration_dir: str):
    """Load calibration data"""
    calibrator = CalibratedPredictor()
    
    # Load thresholds
    with open(os.path.join(calibration_dir, 'thresholds.json'), 'r') as f:
        thresholds_data = json.load(f)
    calibrator.set_thresholds(np.array(thresholds_data['thresholds']))
    
    return calibrator


def predict_sequence(model, sequence: str, uniprot_id: str, knn_blender=None, 
                    hierarchy_head=None, calibrator=None, use_knn=True, use_hierarchy=True):
    """Predict terpene class for a single sequence"""
    
    # Generate features
    eng_features = generate_engineered_features(sequence)
    graph, has_structure = generate_gcn_features(sequence, uniprot_id)
    
    # Mock ESM features (in real implementation, would use ESM2)
    esm_features = np.random.randn(1280).astype(np.float32)  # Placeholder
    
    # Convert to tensors
    esm_tensor = torch.tensor(esm_features).unsqueeze(0)
    eng_tensor = torch.tensor(eng_features).unsqueeze(0)
    struct_tensor = torch.tensor(graph.node_features).unsqueeze(0) if has_structure else None
    has_struct_tensor = torch.tensor([has_structure])
    
    # Model prediction
    with torch.no_grad():
        logits = model(esm_tensor, eng_tensor, struct_tensor, has_struct_tensor)
        raw_probs = torch.softmax(logits, dim=-1).numpy()[0]
    
    # kNN blending
    if use_knn and knn_blender is not None:
        # Mock embedding (in real implementation, would use ESM2)
        embedding = esm_features.reshape(1, -1)
        blended_probs = knn_blender.blend(raw_probs.reshape(1, -1), embedding)[0]
    else:
        blended_probs = raw_probs
    
    # Hierarchy masking
    if use_hierarchy and hierarchy_head is not None:
        # Mock latent features
        latent_features = torch.randn(1, 512)
        type_logits = hierarchy_head.type_head(latent_features)
        type_probs = torch.softmax(type_logits, dim=-1)
        
        fine_logits = torch.tensor(blended_probs).unsqueeze(0)
        masked_logits = hierarchy_head.apply_type_mask(fine_logits, type_probs)
        masked_probs = torch.softmax(masked_logits, dim=-1).numpy()[0]
    else:
        masked_probs = blended_probs
    
    # Calibration
    if calibrator is not None:
        calibrated_probs = calibrator.predict_proba(masked_probs.reshape(1, -1))[0]
    else:
        calibrated_probs = masked_probs
    
    # Apply thresholds and get predictions
    if calibrator is not None and calibrator.thresholds is not None:
        binary_preds = (calibrated_probs > calibrator.thresholds).astype(int)
        top_k_indices = np.argsort(calibrated_probs)[::-1][:3]
    else:
        binary_preds = (calibrated_probs > 0.5).astype(int)
        top_k_indices = np.argsort(calibrated_probs)[::-1][:3]
    
    return {
        'raw_probabilities': raw_probs.tolist(),
        'blended_probabilities': blended_probs.tolist(),
        'calibrated_probabilities': calibrated_probs.tolist(),
        'binary_predictions': binary_preds.tolist(),
        'top_k_indices': top_k_indices.tolist(),
        'top_k_probabilities': calibrated_probs[top_k_indices].tolist(),
        'has_structure': has_structure
    }


def main():
    parser = argparse.ArgumentParser(description='Batch inference with stabilized TPS classifier')
    parser.add_argument('--in', required=True, help='Input FASTA file')
    parser.add_argument('--out', required=True, help='Output JSONL file')
    parser.add_argument('--model', default='models_final_functional/complete_multimodal_best.pth', help='Model checkpoint path')
    parser.add_argument('--knn_index', help='kNN index path')
    parser.add_argument('--calibration', help='Calibration directory')
    parser.add_argument('--use-knn', action='store_true', help='Use kNN blending')
    parser.add_argument('--use-hierarchy', action='store_true', help='Use hierarchy masking')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = load_model(args.model)
    
    # Load kNN index if provided
    knn_blender = None
    if args.use_knn and args.knn_index:
        print(f"Loading kNN index from {args.knn_index}...")
        knn_blender = load_knn_index(args.knn_index)
    
    # Load calibration if provided
    calibrator = None
    if args.calibration:
        print(f"Loading calibration from {args.calibration}...")
        calibrator = load_calibration(args.calibration)
    
    # Initialize hierarchy head
    hierarchy_head = HierarchyHead() if args.use_hierarchy else None
    
    # Load sequences
    sequences = {}
    with open(getattr(args, 'in'), 'r') as f:
        current_id = None
        current_seq = ""
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    sequences[current_id] = current_seq
                current_id = line[1:].split()[0]
                current_seq = ""
            else:
                current_seq += line
        if current_id:
            sequences[current_id] = current_seq
    
    print(f"Loaded {len(sequences)} sequences")
    
    # Run predictions
    results = []
    for i, (seq_id, sequence) in enumerate(sequences.items()):
        if i % 10 == 0:
            print(f"Processing sequence {i+1}/{len(sequences)}: {seq_id}")
        
        prediction = predict_sequence(
            model, sequence, seq_id, knn_blender, 
            hierarchy_head, calibrator, args.use_knn, args.use_hierarchy
        )
        
        results.append({
            'sequence_id': seq_id,
            'sequence': sequence,
            'prediction': prediction
        })
    
    # Save results
    with open(args.out, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"✅ Predictions saved to {args.out}")
    print(f"   - {len(results)} sequences processed")
    print(f"   - kNN blending: {'ON' if args.use_knn and knn_blender else 'OFF'}")
    print(f"   - Hierarchy masking: {'ON' if args.use_hierarchy else 'OFF'}")
    print(f"   - Calibration: {'ON' if calibrator else 'OFF'}")


if __name__ == "__main__":
    main()