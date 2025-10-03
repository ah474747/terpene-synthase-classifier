#!/usr/bin/env python3
"""
Calibrate thresholds using identity-aware validation
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'terpene_classifier_v3'))

import json
import numpy as np
from tps.eval.calibration import CalibratedPredictor


def load_predictions(preds_path: str):
    """Load predictions from JSONL file"""
    predictions = []
    labels = []
    
    with open(preds_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            predictions.append(data['probabilities'])
            labels.append(data['true_label'])
    
    return np.array(predictions), np.array(labels)


def optimize_thresholds_f1beta(probabilities: np.ndarray, labels: np.ndarray, beta: float = 0.7):
    """Optimize thresholds to maximize F1-beta"""
    n_classes = probabilities.shape[1]
    thresholds = np.zeros(n_classes)
    
    for class_id in range(n_classes):
        class_probs = probabilities[:, class_id]
        class_labels = (labels == class_id).astype(int)
        
        best_f1 = 0
        best_threshold = 0.5
        
        # Test different thresholds
        for threshold in np.arange(0.1, 0.9, 0.05):
            preds = (class_probs > threshold).astype(int)
            
            tp = np.sum((preds == 1) & (class_labels == 1))
            fp = np.sum((preds == 1) & (class_labels == 0))
            fn = np.sum((preds == 0) & (class_labels == 1))
            
            if tp + fp == 0:
                precision = 0
            else:
                precision = tp / (tp + fp)
            
            if tp + fn == 0:
                recall = 0
            else:
                recall = tp / (tp + fn)
            
            if precision + recall == 0:
                f1 = 0
            else:
                f1 = (1 + beta**2) * precision * recall / ((beta**2) * precision + recall)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        thresholds[class_id] = best_threshold
    
    return thresholds


def main():
    parser = argparse.ArgumentParser(description='Calibrate thresholds')
    parser.add_argument('--preds', required=True, help='Validation predictions JSONL file')
    parser.add_argument('--out', required=True, help='Output calibration directory')
    parser.add_argument('--beta', type=float, default=0.7, help='F1-beta parameter')
    
    args = parser.parse_args()
    
    # Load data
    probabilities, labels = load_predictions(args.preds)
    
    # Optimize thresholds
    thresholds = optimize_thresholds_f1beta(probabilities, labels, args.beta)
    
    # Save results
    os.makedirs(args.out, exist_ok=True)
    
    with open(os.path.join(args.out, 'thresholds.json'), 'w') as f:
        json.dump({
            'thresholds': thresholds.tolist(),
            'beta': args.beta
        }, f, indent=2)
    
    print(f"✅ Calibration completed! Thresholds saved to {args.out}/thresholds.json")


if __name__ == "__main__":
    main()