#!/usr/bin/env python3
"""
Dataset Validation Script for TS-GSD

This script validates the quality and integrity of the generated TS-GSD dataset.
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path

def validate_ts_gsd_dataset(dataset_path: str = "data/TS-GSD.csv"):
    """
    Validate the TS-GSD dataset for completeness and quality
    
    Args:
        dataset_path: Path to the TS-GSD dataset
    """
    print("🔍 Validating TS-GSD Dataset...")
    
    # Load dataset
    df = pd.read_csv(dataset_path)
    
    print(f"📊 Dataset Overview:")
    print(f"  - Total entries: {len(df)}")
    print(f"  - Columns: {len(df.columns)}")
    
    # Check required columns
    required_columns = [
        'uniprot_accession_id',
        'aa_sequence', 
        'product_smiles_list',
        'substrate_type',
        'rxn_class_i_ii_hybrid',
        'pfam_domain_ids',
        'taxonomy_phylum',
        'target_vector'
    ]
    
    print(f"\n✅ Required Columns Check:")
    missing_columns = []
    for col in required_columns:
        if col in df.columns:
            print(f"  ✓ {col}")
        else:
            print(f"  ❌ {col} - MISSING")
            missing_columns.append(col)
    
    if missing_columns:
        print(f"\n❌ Validation failed: Missing columns: {missing_columns}")
        return False
    
    # Validate data types and content
    print(f"\n🔍 Data Quality Checks:")
    
    # Check sequences
    sequence_lengths = df['aa_sequence'].str.len()
    print(f"  - Sequence lengths: {sequence_lengths.min()} - {sequence_lengths.max()} (avg: {sequence_lengths.mean():.1f})")
    
    # Check target vectors
    target_vectors = df['target_vector'].apply(json.loads)
    vector_lengths = target_vectors.apply(len)
    print(f"  - Target vector lengths: {vector_lengths.unique()}")
    
    # Check multi-label distribution
    num_products = target_vectors.apply(sum)
    print(f"  - Products per enzyme: {num_products.min()} - {num_products.max()} (avg: {num_products.mean():.2f})")
    
    # Check substrate types
    substrate_counts = df['substrate_type'].value_counts()
    print(f"  - Substrate types: {dict(substrate_counts)}")
    
    # Check reaction classes
    rxn_counts = df['rxn_class_i_ii_hybrid'].value_counts()
    print(f"  - Reaction classes: {dict(rxn_counts)}")
    
    # Check for missing values
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        print(f"\n⚠️  Missing values found:")
        for col, count in missing_data.items():
            if count > 0:
                print(f"    - {col}: {count} missing")
    else:
        print(f"\n✅ No missing values found")
    
    # Validate JSON fields
    print(f"\n🔍 JSON Field Validation:")
    
    try:
        # Test product_smiles_list
        smiles_parsed = df['product_smiles_list'].apply(json.loads)
        print(f"  ✓ product_smiles_list: Valid JSON")
    except Exception as e:
        print(f"  ❌ product_smiles_list: Invalid JSON - {e}")
        return False
    
    try:
        # Test target_vector
        targets_parsed = df['target_vector'].apply(json.loads)
        print(f"  ✓ target_vector: Valid JSON")
    except Exception as e:
        print(f"  ❌ target_vector: Invalid JSON - {e}")
        return False
    
    try:
        # Test pfam_domain_ids
        pfam_parsed = df['pfam_domain_ids'].apply(json.loads)
        print(f"  ✓ pfam_domain_ids: Valid JSON")
    except Exception as e:
        print(f"  ❌ pfam_domain_ids: Invalid JSON - {e}")
        return False
    
    # Functional ensemble distribution
    print(f"\n📊 Functional Ensemble Analysis:")
    all_products = []
    for products in target_vectors:
        all_products.extend([i for i, val in enumerate(products) if val == 1])
    
    ensemble_counts = pd.Series(all_products).value_counts().sort_index()
    print(f"  - Active ensembles: {len(ensemble_counts)}")
    print(f"  - Ensemble distribution: {dict(ensemble_counts.head(10))}")
    
    print(f"\n✅ Dataset validation completed successfully!")
    print(f"🎯 Dataset is ready for Module 2: Feature Extraction Pipeline")
    
    return True

def main():
    """Main validation function"""
    dataset_path = "data/TS-GSD.csv"
    
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found at: {dataset_path}")
        print("Please run the TS-GSD pipeline first: python ts_gsd_pipeline.py")
        return
    
    validate_ts_gsd_dataset(dataset_path)

if __name__ == "__main__":
    main()



