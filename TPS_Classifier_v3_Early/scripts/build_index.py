#!/usr/bin/env python3
"""
Build kNN index from training data
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'terpene_classifier_v3'))

import numpy as np
import pandas as pd
from tps.retrieval.knn_head import KNNBlender
from tps.features.engineered import generate_engineered_features
from tps.utils import set_seed


def load_training_data(fasta_path: str, labels_path: str):
    """Load training sequences and labels"""
    sequences = {}
    labels = {}
    
    # Load FASTA sequences
    with open(fasta_path, 'r') as f:
        current_id = None
        current_seq = ""
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    sequences[current_id] = current_seq
                current_id = line[1:].split()[0]  # Get first part of header
                current_seq = ""
            else:
                current_seq += line
        if current_id:
            sequences[current_id] = current_seq
    
    # Load labels
    df = pd.read_csv(labels_path)
    for _, row in df.iterrows():
        labels[row['uniprot_id']] = row['ensemble_id']
    
    return sequences, labels


def generate_embeddings(sequences: dict):
    """Generate ESM embeddings (placeholder - using engineered features for now)"""
    embeddings = {}
    
    for seq_id, sequence in sequences.items():
        # For now, use engineered features as embedding proxy
        # In real implementation, this would use ESM2 model
        features = generate_engineered_features(sequence)
        
        # Pad to 1280D to simulate ESM2
        embedding = np.zeros(1280, dtype=np.float32)
        embedding[:len(features)] = features
        embeddings[seq_id] = embedding
    
    return embeddings


def build_knn_index(sequences: dict, labels: dict, output_path: str, k: int = 5, alpha: float = 0.7, seed: int = 42):
    """Build and save kNN index"""
    set_seed(seed)
    
    print(f"Building kNN index for {len(sequences)} training sequences...")
    
    # Generate embeddings
    embeddings = generate_embeddings(sequences)
    
    # Prepare data arrays
    embedding_array = []
    label_array = []
    seq_ids = []
    
    for seq_id in sequences.keys():
        if seq_id in embeddings and seq_id in labels:
            embedding_array.append(embeddings[seq_id])
            label_array.append(labels[seq_id])
            seq_ids.append(seq_id)
    
    embedding_array = np.array(embedding_array)
    label_array = np.array(label_array)
    
    print(f"Generated embeddings for {len(embedding_array)} sequences")
    
    # Build kNN blender
    blender = KNNBlender(k=k, alpha=alpha)
    blender.fit(embedding_array, label_array)
    
    # Save index
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # For now, save as pickle (in real implementation, would use FAISS)
    import pickle
    with open(output_path, 'wb') as f:
        pickle.dump({
            'blender': blender,
            'embeddings': embedding_array,
            'labels': label_array,
            'seq_ids': seq_ids
        }, f)
    
    print(f"✅ kNN index saved to {output_path}")
    print(f"   - {len(embedding_array)} training sequences")
    print(f"   - k={k}, alpha={alpha}")
    
    return blender


def main():
    parser = argparse.ArgumentParser(description='Build kNN index from training data')
    parser.add_argument('--train_fasta', required=True, help='Training sequences FASTA file')
    parser.add_argument('--labels', required=True, help='Training labels CSV file')
    parser.add_argument('--out', required=True, help='Output index path')
    parser.add_argument('--k', type=int, default=5, help='Number of neighbors')
    parser.add_argument('--alpha', type=float, default=0.7, help='Blending weight')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Load training data
    sequences, labels = load_training_data(args.train_fasta, args.labels)
    
    # Build index
    blender = build_knn_index(sequences, labels, args.out, args.k, args.alpha, args.seed)
    
    print("🎉 kNN index building completed successfully!")


if __name__ == "__main__":
    main()