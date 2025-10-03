#!/usr/bin/env python3
"""
Generate Engineered Features for TS-GSD Dataset

Extracts biochemical/categorical features from the TS-GSD consolidated dataset
following the V3 model's feature engineering approach.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# Configuration
DATA_FILE = '../TPS_Classifier_v3_Early/TS-GSD_consolidated.csv'
OUTPUT_FILE = 'data/engineered_features.npy'
ENG_DIM = 64  # Total engineered feature dimension (matching V3)

def generate_engineered_features():
    """Generate engineered features from TS-GSD data"""
    
    print("📂 Loading TS-GSD consolidated dataset...")
    df = pd.read_csv(DATA_FILE)
    print(f"✅ Loaded {len(df)} samples")
    
    # 1. Terpene Type (categorical -> one-hot)
    terpene_types = ['mono', 'sesq', 'di', 'tri', 'pt', 'other']
    df['terpene_type'] = df['terpene_type'].fillna('other')
    
    # Create one-hot encoding for terpene type
    type_features = np.zeros((len(df), len(terpene_types)), dtype=np.float32)
    for i, ttype in enumerate(df['terpene_type']):
        if ttype in terpene_types:
            type_features[i, terpene_types.index(ttype)] = 1.0
        else:
            type_features[i, terpene_types.index('other')] = 1.0
    
    print(f"✅ Terpene type features: {type_features.shape[1]} dimensions")
    
    # 2. Enzyme Class (categorical -> one-hot)
    # Class 1 or Class 2
    df['enzyme_class'] = df['enzyme_class'].fillna(0)
    class_features = np.zeros((len(df), 2), dtype=np.float32)
    for i, eclass in enumerate(df['enzyme_class']):
        if eclass == 1:
            class_features[i, 0] = 1.0
        elif eclass == 2:
            class_features[i, 1] = 1.0
    
    print(f"✅ Enzyme class features: {class_features.shape[1]} dimensions")
    
    # 3. Kingdom (categorical -> one-hot)
    kingdoms = df['kingdom'].unique()
    kingdom_features = np.zeros((len(df), len(kingdoms)), dtype=np.float32)
    kingdom_to_idx = {k: i for i, k in enumerate(kingdoms)}
    for i, kingdom in enumerate(df['kingdom']):
        if pd.notna(kingdom) and kingdom in kingdom_to_idx:
            kingdom_features[i, kingdom_to_idx[kingdom]] = 1.0
    
    print(f"✅ Kingdom features: {kingdom_features.shape[1]} dimensions")
    
    # 4. Number of products (normalized)
    num_products = df['num_products'].fillna(1).values.astype(np.float32)
    max_products = num_products.max()
    normalized_products = (num_products / max_products if max_products > 0 else num_products).reshape(-1, 1)
    
    print(f"✅ Product count feature: 1 dimension (max={int(max_products)})")
    
    # 5. Combine real features
    real_features = np.concatenate([
        type_features,           # ~6 dimensions (terpene types)
        class_features,          # 2 dimensions (Class 1, Class 2)
        kingdom_features,        # ~3 dimensions (Plantae, Bacteria, Fungi, etc.)
        normalized_products      # 1 dimension (normalized product count)
    ], axis=1)
    
    real_feature_dim = real_features.shape[1]
    print(f"\n📊 Real categorical features: {real_feature_dim} dimensions")
    
    # 6. Add placeholder features to reach ENG_DIM (64)
    # These would be replaced with structural/mechanistic features in full model
    placeholder_dim = ENG_DIM - real_feature_dim
    
    if placeholder_dim > 0:
        # Use fixed seed for reproducibility
        np.random.seed(42)
        placeholder_features = np.random.uniform(
            0, 1, 
            size=(len(df), placeholder_dim)
        ).astype(np.float32)
        
        engineered_features = np.concatenate([real_features, placeholder_features], axis=1)
        print(f"📊 Placeholder features: {placeholder_dim} dimensions (for future structural data)")
    else:
        engineered_features = real_features[:, :ENG_DIM]
    
    print(f"\n✅ Total engineered features: {engineered_features.shape}")
    print(f"   - Real features: {real_feature_dim} dimensions")
    print(f"   - Placeholder features: {max(0, placeholder_dim)} dimensions")
    
    # Save features
    Path('data').mkdir(exist_ok=True)
    np.save(OUTPUT_FILE, engineered_features)
    print(f"\n💾 Saved engineered features to {OUTPUT_FILE}")
    
    # Save feature info
    feature_info = {
        'total_dim': int(engineered_features.shape[1]),
        'real_dim': int(real_feature_dim),
        'placeholder_dim': int(max(0, placeholder_dim)),
        'terpene_types': terpene_types,
        'n_kingdoms': len(kingdoms),
        'kingdoms': list(kingdoms)
    }
    
    import json
    with open('data/engineered_feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    print(f"💾 Saved feature info to data/engineered_feature_info.json")
    
    return engineered_features

if __name__ == '__main__':
    print("="*60)
    print("🔧 Generating Engineered Features for TS-GSD Dataset")
    print("="*60 + "\n")
    
    features = generate_engineered_features()
    
    print("\n" + "="*60)
    print("✅ Engineered features generation complete!")
    print("="*60)

