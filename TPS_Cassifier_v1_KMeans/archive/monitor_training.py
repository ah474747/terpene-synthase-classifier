#!/usr/bin/env python3
"""
Monitor training progress for the complete dataset
"""

import os
import time
import json
from pathlib import Path


def check_training_progress():
    """Check the current training progress"""
    
    # Check if model files exist
    model_files = [
        "models/germacrene_classifier_full.pkl",
        "models/germacrene_classifier_robust.pkl"
    ]
    
    results_files = [
        "results/full_training_results.json",
        "results/robust_training_results.json"
    ]
    
    print("=== Training Progress Monitor ===")
    print(f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check for completed training
    for model_file in model_files:
        if os.path.exists(model_file):
            file_size = os.path.getsize(model_file)
            mod_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(model_file)))
            print(f"✓ Model file exists: {model_file}")
            print(f"  - Size: {file_size:,} bytes")
            print(f"  - Last modified: {mod_time}")
    
    # Check for results
    for results_file in results_files:
        if os.path.exists(results_file):
            mod_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(results_file)))
            print(f"✓ Results file exists: {results_file}")
            print(f"  - Last modified: {mod_time}")
            
            # Show results summary
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                print(f"  - Training sequences: {results.get('training_sequences', 'N/A')}")
                print(f"  - F1-Score: {results.get('cv_f1_mean', 'N/A'):.3f} ± {results.get('cv_f1_std', 'N/A'):.3f}")
                print(f"  - Best F1-Score: {results.get('best_f1_score', 'N/A'):.3f}")
                print(f"  - AUC-PR: {results.get('cv_auc_pr_mean', 'N/A'):.3f} ± {results.get('cv_auc_pr_std', 'N/A'):.3f}")
                
            except Exception as e:
                print(f"  - Error reading results: {e}")
    
    # Check for running processes
    print("\n=== Process Status ===")
    try:
        import subprocess
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        python_processes = [line for line in result.stdout.split('\n') if 'python3 train_full_dataset.py' in line]
        
        if python_processes:
            print("✓ Full training process is running:")
            for process in python_processes:
                print(f"  {process}")
        else:
            print("✗ No full training process found")
            
    except Exception as e:
        print(f"Could not check process status: {e}")


def show_data_summary():
    """Show summary of available data"""
    print("\n=== Data Summary ===")
    
    # Check training data
    training_file = "data/germacrene_training_data.csv"
    if os.path.exists(training_file):
        import pandas as pd
        df = pd.read_csv(training_file)
        print(f"✓ Training data: {len(df)} sequences")
        print(f"  - Germacrene synthases: {df['target'].sum()}")
        print(f"  - Other synthases: {len(df) - df['target'].sum()}")
        print(f"  - Class balance: {df['target'].mean():.2%}")
    
    # Check enhanced data
    enhanced_file = "data/marts_db_enhanced.csv"
    if os.path.exists(enhanced_file):
        df = pd.read_csv(enhanced_file)
        print(f"✓ Enhanced data: {len(df)} sequences")
        print(f"  - Product categories: {df['product_category'].nunique()}")
        print(f"  - Germacrene types: {df[df['is_germacrene_family']]['germacrene_type'].nunique()}")


def main():
    """Main monitoring function"""
    check_training_progress()
    show_data_summary()
    
    print(f"\n=== Next Steps ===")
    print("1. Wait for training to complete (2-3 hours total)")
    print("2. Check results in results/full_training_results.json")
    print("3. Test the trained model with predict_with_trained_model.py")
    print("4. Use the model for predictions on new sequences")


if __name__ == "__main__":
    main()
