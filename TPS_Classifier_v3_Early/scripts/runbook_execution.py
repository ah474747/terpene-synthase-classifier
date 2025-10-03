#!/usr/bin/env python3
"""
Complete Performance Validation Runbook Execution
Measures F1 improvements on novel sequences using the stabilized TPS classifier
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'terpene_classifier_v3'))

import json
import numpy as np
import subprocess
import tempfile
from datetime import datetime


def create_mock_training_data(output_dir: str):
    """Create mock training data for demonstration"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create mock FASTA file
    fasta_path = os.path.join(output_dir, 'train.fasta')
    with open(fasta_path, 'w') as f:
        for i in range(100):
            f.write(f">TRAIN_{i:03d}\n")
            f.write("MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQTLSEQVQEELLSSQVTQELRALMDETAQN\n")
    
    # Create mock labels CSV
    labels_path = os.path.join(output_dir, 'train_labels.csv')
    with open(labels_path, 'w') as f:
        f.write("uniprot_id,ensemble_id\n")
        for i in range(100):
            f.write(f"TRAIN_{i:03d},{i % 30}\n")
    
    # Create mock validation FASTA
    val_fasta_path = os.path.join(output_dir, 'val.fasta')
    with open(val_fasta_path, 'w') as f:
        for i in range(50):
            f.write(f">VAL_{i:03d}\n")
            f.write("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n")
    
    # Create mock validation labels
    val_labels_path = os.path.join(output_dir, 'val_labels.csv')
    with open(val_labels_path, 'w') as f:
        f.write("uniprot_id,ensemble_id\n")
        for i in range(50):
            f.write(f"VAL_{i:03d},{i % 30}\n")
    
    # Create mock identity clusters
    clusters_path = os.path.join(output_dir, 'val_clusters.json')
    clusters = {}
    for i in range(50):
        clusters[f"VAL_{i:03d}"] = 0.3 + (i % 10) * 0.1  # Mock identity scores
    with open(clusters_path, 'w') as f:
        json.dump(clusters, f, indent=2)
    
    return fasta_path, labels_path, val_fasta_path, val_labels_path, clusters_path


def run_command(cmd: list, description: str):
    """Run a command and handle errors"""
    print(f"\n🔧 {description}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            if result.stdout:
                print("Output:", result.stdout[:500])  # First 500 chars
        else:
            print(f"❌ {description} failed")
            print("Error:", result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} timed out")
        return False
    except Exception as e:
        print(f"💥 {description} error: {e}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Execute complete performance validation runbook')
    parser.add_argument('--data_dir', default='runbook_data', help='Directory for runbook data')
    parser.add_argument('--results_dir', default='runbook_results', help='Directory for results')
    parser.add_argument('--alpha', type=float, default=0.7, help='kNN blending alpha')
    parser.add_argument('--beta', type=float, default=0.7, help='F1-beta parameter')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print("🚀 TPS Classifier Performance Validation Runbook")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Results directory: {args.results_dir}")
    print(f"kNN alpha: {args.alpha}")
    print(f"F1-beta: {args.beta}")
    print(f"Seed: {args.seed}")
    print("=" * 60)
    
    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, 'preds'), exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, 'reports'), exist_ok=True)
    
    # Step 1: Create mock training data
    print("\n📊 Step 1: Creating mock training data...")
    train_fasta, train_labels, val_fasta, val_labels, val_clusters = create_mock_training_data(args.data_dir)
    print(f"✅ Created training data: {train_fasta}, {train_labels}")
    print(f"✅ Created validation data: {val_fasta}, {val_labels}")
    
    # Step 2: Build kNN index
    print("\n🔍 Step 2: Building kNN index...")
    knn_index_path = os.path.join(args.results_dir, 'models', 'knn_index.pkl')
    cmd = [
        'python3', 'scripts/build_index.py',
        '--train_fasta', train_fasta,
        '--labels', train_labels,
        '--out', knn_index_path,
        '--alpha', str(args.alpha),
        '--seed', str(args.seed)
    ]
    if not run_command(cmd, "Building kNN index"):
        print("❌ Runbook failed at kNN index building")
        return 1
    
    # Step 3: Baseline predictions (no enhancements)
    print("\n📈 Step 3: Baseline predictions...")
    baseline_preds = os.path.join(args.results_dir, 'preds', 'val_base.jsonl')
    cmd = [
        'python3', 'scripts/predict.py',
        '--in', val_fasta,
        '--out', baseline_preds,
        '--seed', str(args.seed)
    ]
    if not run_command(cmd, "Baseline predictions"):
        print("❌ Runbook failed at baseline predictions")
        return 1
    
    # Step 4: Calibrate thresholds
    print("\n🎯 Step 4: Calibrating thresholds...")
    calibration_dir = os.path.join(args.results_dir, 'models', 'calibration')
    cmd = [
        'python3', 'scripts/calibrate_thresholds.py',
        '--preds', baseline_preds,
        '--out', calibration_dir,
        '--beta', str(args.beta)
    ]
    if not run_command(cmd, "Calibrating thresholds"):
        print("❌ Runbook failed at threshold calibration")
        return 1
    
    # Step 5: Enhanced predictions (with all improvements)
    print("\n🚀 Step 5: Enhanced predictions...")
    enhanced_preds = os.path.join(args.results_dir, 'preds', 'val_enhanced.jsonl')
    cmd = [
        'python3', 'scripts/predict.py',
        '--in', val_fasta,
        '--out', enhanced_preds,
        '--knn_index', knn_index_path,
        '--calibration', calibration_dir,
        '--use-knn',
        '--use-hierarchy',
        '--seed', str(args.seed)
    ]
    if not run_command(cmd, "Enhanced predictions"):
        print("❌ Runbook failed at enhanced predictions")
        return 1
    
    # Step 6: Evaluate and compare
    print("\n📊 Step 6: Evaluating performance...")
    evaluation_report = os.path.join(args.results_dir, 'reports', 'performance_comparison.json')
    cmd = [
        'python3', 'scripts/evaluate.py',
        '--preds', baseline_preds, enhanced_preds,
        '--report', evaluation_report,
        '--bootstrap', '1000'
    ]
    if not run_command(cmd, "Performance evaluation"):
        print("❌ Runbook failed at performance evaluation")
        return 1
    
    # Step 7: Generate summary report
    print("\n📋 Step 7: Generating summary report...")
    
    # Load evaluation results
    with open(evaluation_report, 'r') as f:
        eval_results = json.load(f)
    
    # Create summary
    summary = {
        'runbook_timestamp': datetime.now().isoformat(),
        'parameters': {
            'alpha': args.alpha,
            'beta': args.beta,
            'seed': args.seed
        },
        'data_summary': {
            'training_sequences': 100,
            'validation_sequences': 50,
            'n_classes': 30
        },
        'performance_comparison': eval_results.get('comparison_summary', {}),
        'files_generated': {
            'knn_index': knn_index_path,
            'baseline_predictions': baseline_preds,
            'enhanced_predictions': enhanced_preds,
            'calibration': calibration_dir,
            'evaluation_report': evaluation_report
        }
    }
    
    summary_path = os.path.join(args.results_dir, 'runbook_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Summary report saved to {summary_path}")
    
    # Final results
    print("\n" + "=" * 60)
    print("🎉 RUNBOOK EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    if 'performance_comparison' in summary and summary['performance_comparison']:
        comp = summary['performance_comparison']
        print(f"\n📈 PERFORMANCE IMPROVEMENTS:")
        print(f"   Macro F1: {comp.get('macro_f1_improvement', 0):+.3f}")
        print(f"   Top-K Accuracy: {comp.get('top_k_improvement', 0):+.3f}")
        print(f"   ECE Reduction: {comp.get('ece_improvement', 0):+.3f}")
    
    print(f"\n📁 Results saved in: {args.results_dir}")
    print(f"📊 Evaluation report: {evaluation_report}")
    print(f"📋 Summary: {summary_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())