#!/usr/bin/env python3
"""
Monitor K-Means Enhanced Training Progress
"""

import os
import time
import json
from pathlib import Path


def monitor_kmeans_training():
    """Monitor the k-means enhanced training progress"""
    
    print("=" * 70)
    print("K-MEANS ENHANCED TRAINING MONITOR")
    print("=" * 70)
    print(f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check for k-means model files
    kmeans_model_file = "models/germacrene_classifier_kmeans.pkl"
    kmeans_results_file = "results/kmeans_training_results.json"
    
    print(f"\n=== K-Means Enhanced Model Status ===")
    
    if os.path.exists(kmeans_model_file):
        file_size = os.path.getsize(kmeans_model_file)
        mod_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(kmeans_model_file)))
        print(f"✓ K-means model exists: {kmeans_model_file}")
        print(f"  - Size: {file_size:,} bytes")
        print(f"  - Last modified: {mod_time}")
        
        # Check if it's the full dataset model
        try:
            import joblib
            model_data = joblib.load(kmeans_model_file)
            training_sequences = model_data.get('training_sequences', 'Unknown')
            n_clusters = model_data.get('n_clusters', 'Unknown')
            print(f"  - Training sequences: {training_sequences}")
            print(f"  - Number of clusters: {n_clusters}")
            
            if training_sequences >= 1350:
                print(f"  - ✓ FULL DATASET MODEL!")
            else:
                print(f"  - ⚠ Limited dataset model ({training_sequences} sequences)")
                
        except Exception as e:
            print(f"  - Error reading model info: {e}")
    else:
        print(f"✗ K-means model not found: {kmeans_model_file}")
    
    # Check results
    if os.path.exists(kmeans_results_file):
        mod_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(kmeans_results_file)))
        print(f"✓ K-means results exist: {kmeans_results_file}")
        print(f"  - Last modified: {mod_time}")
        
        try:
            with open(kmeans_results_file, 'r') as f:
                results = json.load(f)
            
            print(f"  - Training sequences: {results.get('training_sequences', 'N/A')}")
            print(f"  - Number of clusters: {results.get('n_clusters', 'N/A')}")
            print(f"  - Enhanced features: {results.get('enhanced_feature_dimension', 'N/A')}")
            print(f"  - F1-Score: {results.get('cv_f1_mean', 'N/A'):.3f} ± {results.get('cv_f1_std', 'N/A'):.3f}")
            print(f"  - Best F1-Score: {results.get('best_f1_score', 'N/A'):.3f}")
            print(f"  - AUC-PR: {results.get('cv_auc_pr_mean', 'N/A'):.3f}")
            
        except Exception as e:
            print(f"  - Error reading results: {e}")
    else:
        print(f"✗ K-means results not found: {kmeans_results_file}")
    
    # Check for cluster visualization
    cluster_viz_file = "results/kmeans_cluster_analysis.png"
    if os.path.exists(cluster_viz_file):
        file_size = os.path.getsize(cluster_viz_file)
        mod_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(cluster_viz_file)))
        print(f"✓ Cluster visualization: {cluster_viz_file}")
        print(f"  - Size: {file_size:,} bytes")
        print(f"  - Last modified: {mod_time}")
    
    # Check for running processes
    print(f"\n=== Process Status ===")
    try:
        import subprocess
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        kmeans_processes = [line for line in result.stdout.split('\n') if 'train_with_kmeans.py' in line]
        
        if kmeans_processes:
            print("✓ K-means training process is running:")
            for process in kmeans_processes:
                print(f"  {process}")
        else:
            print("✗ No k-means training process found")
            
    except Exception as e:
        print(f"Could not check process status: {e}")
    
    # Show comparison with other models
    print(f"\n=== Model Comparison ===")
    
    # Standard XGBoost results
    robust_results_file = "results/robust_training_results.json"
    if os.path.exists(robust_results_file):
        try:
            with open(robust_results_file, 'r') as f:
                robust_results = json.load(f)
            
            print(f"Standard XGBoost (200 sequences):")
            print(f"  - F1-Score: {robust_results.get('cv_f1_mean', 'N/A'):.3f} ± {robust_results.get('cv_f1_std', 'N/A'):.3f}")
            print(f"  - Best F1-Score: {robust_results.get('best_f1_score', 'N/A'):.3f}")
            
        except Exception as e:
            print(f"Could not read robust results: {e}")
    
    # Full dataset results
    full_results_file = "results/full_training_results.json"
    if os.path.exists(full_results_file):
        try:
            with open(full_results_file, 'r') as f:
                full_results = json.load(f)
            
            print(f"Full Dataset XGBoost (500 sequences):")
            print(f"  - F1-Score: {full_results.get('cv_f1_mean', 'N/A'):.3f} ± {full_results.get('cv_f1_std', 'N/A'):.3f}")
            print(f"  - Best F1-Score: {full_results.get('best_f1_score', 'N/A'):.3f}")
            
        except Exception as e:
            print(f"Could not read full results: {e}")
    
    # Data summary
    print(f"\n=== Dataset Summary ===")
    training_file = "data/germacrene_training_data.csv"
    if os.path.exists(training_file):
        import pandas as pd
        df = pd.read_csv(training_file)
        print(f"✓ Complete dataset: {len(df)} sequences")
        print(f"  - Germacrene synthases: {df['target'].sum()}")
        print(f"  - Other synthases: {len(df) - df['target'].sum()}")
        print(f"  - Class balance: {df['target'].mean():.2%}")
    
    print(f"\n=== Next Steps ===")
    print("1. Wait for k-means enhanced training to complete")
    print("2. Compare performance with standard XGBoost")
    print("3. Analyze cluster patterns and visualizations")
    print("4. Test the enhanced model on new sequences")
    print("5. Deploy the best performing model")


def main():
    monitor_kmeans_training()


if __name__ == "__main__":
    main()

