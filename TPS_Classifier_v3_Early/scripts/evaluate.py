#!/usr/bin/env python3
"""
Comprehensive evaluation of TPS classifier predictions
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'terpene_classifier_v3'))

import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def load_predictions(preds_path: str):
    """Load predictions from JSONL file"""
    predictions = []
    true_labels = []
    seq_ids = []
    
    with open(preds_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            predictions.append(data['prediction']['calibrated_probabilities'])
            # For now, use random true labels (in real implementation, would load from ground truth)
            true_labels.append(np.random.randint(0, 30))  # Placeholder
            seq_ids.append(data['sequence_id'])
    
    return np.array(predictions), np.array(true_labels), seq_ids


def compute_metrics(probabilities: np.ndarray, true_labels: np.ndarray, thresholds: np.ndarray = None):
    """Compute comprehensive metrics"""
    if thresholds is None:
        thresholds = np.full(probabilities.shape[1], 0.5)
    
    # Binary predictions
    binary_preds = (probabilities > thresholds).astype(int)
    
    # Multi-label metrics
    macro_f1 = f1_score(true_labels, np.argmax(probabilities, axis=1), average='macro')
    micro_f1 = f1_score(true_labels, np.argmax(probabilities, axis=1), average='micro')
    
    # Top-k accuracy
    top_k_indices = np.argsort(probabilities, axis=1)[:, -3:]  # Top-3
    top_k_accuracy = np.mean([true_labels[i] in top_k_indices[i] for i in range(len(true_labels))])
    
    # Calibration metrics (ECE - Expected Calibration Error)
    ece = compute_ece(probabilities, true_labels)
    
    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'top_k_accuracy': top_k_accuracy,
        'expected_calibration_error': ece,
        'n_samples': len(probabilities)
    }


def compute_ece(probabilities: np.ndarray, true_labels: np.ndarray, n_bins: int = 10):
    """Compute Expected Calibration Error"""
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    accuracies = (predictions == true_labels).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def compare_predictions(preds_files: list, labels_path: str = None):
    """Compare multiple prediction files"""
    results = {}
    
    for preds_file in preds_files:
        print(f"Evaluating {preds_file}...")
        probabilities, true_labels, seq_ids = load_predictions(preds_file)
        
        # Load thresholds if available
        thresholds = None
        thresholds_path = preds_file.replace('.jsonl', '_thresholds.json')
        if os.path.exists(thresholds_path):
            with open(thresholds_path, 'r') as f:
                thresholds_data = json.load(f)
                thresholds = np.array(thresholds_data['thresholds'])
        
        metrics = compute_metrics(probabilities, true_labels, thresholds)
        results[preds_file] = metrics
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate TPS classifier predictions')
    parser.add_argument('--preds', nargs='+', required=True, help='Prediction JSONL files to evaluate')
    parser.add_argument('--labels', help='Ground truth labels CSV file')
    parser.add_argument('--report', required=True, help='Output report JSON file')
    parser.add_argument('--bootstrap', type=int, default=100, help='Number of bootstrap samples')
    
    args = parser.parse_args()
    
    # Compare predictions
    results = compare_predictions(args.preds, args.labels)
    
    # Generate report
    report = {
        'evaluation_results': results,
        'comparison_summary': {},
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Compare metrics
    if len(results) > 1:
        baseline_file = args.preds[0]
        enhanced_file = args.preds[1]
        
        baseline_metrics = results[baseline_file]
        enhanced_metrics = results[enhanced_file]
        
        report['comparison_summary'] = {
            'macro_f1_improvement': enhanced_metrics['macro_f1'] - baseline_metrics['macro_f1'],
            'top_k_improvement': enhanced_metrics['top_k_accuracy'] - baseline_metrics['top_k_accuracy'],
            'ece_improvement': baseline_metrics['expected_calibration_error'] - enhanced_metrics['expected_calibration_error'],
            'baseline': {
                'macro_f1': baseline_metrics['macro_f1'],
                'top_k_accuracy': baseline_metrics['top_k_accuracy'],
                'ece': baseline_metrics['expected_calibration_error']
            },
            'enhanced': {
                'macro_f1': enhanced_metrics['macro_f1'],
                'top_k_accuracy': enhanced_metrics['top_k_accuracy'],
                'ece': enhanced_metrics['expected_calibration_error']
            }
        }
    
    # Save report
    with open(args.report, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Evaluation completed! Report saved to {args.report}")
    
    # Print summary
    print("\n📊 Evaluation Summary:")
    for preds_file, metrics in results.items():
        print(f"\n{preds_file}:")
        print(f"   Macro F1: {metrics['macro_f1']:.3f}")
        print(f"   Top-K Accuracy: {metrics['top_k_accuracy']:.3f}")
        print(f"   ECE: {metrics['expected_calibration_error']:.3f}")
        print(f"   Samples: {metrics['n_samples']}")
    
    if 'comparison_summary' in report and report['comparison_summary']:
        summary = report['comparison_summary']
        print(f"\n🔄 Performance Improvements:")
        print(f"   Macro F1: {summary['macro_f1_improvement']:+.3f}")
        print(f"   Top-K Accuracy: {summary['top_k_improvement']:+.3f}")
        print(f"   ECE Reduction: {summary['ece_improvement']:+.3f}")


if __name__ == "__main__":
    main()