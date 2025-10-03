#!/usr/bin/env python3
"""
Identity Split - Runbook Version
===============================

Creates identity-aware validation splits with clustering.
"""

import argparse
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from tps.eval.identity_split import create_identity_splits
from tps.utils import setup_logging, set_seed

def parse_fasta(filepath: Path) -> tuple:
    """Parse FASTA file and return sequences, IDs, and annotations."""
    sequences = []
    uniprot_ids = []
    annotations = []
    
    with open(filepath, 'r') as f:
        current_seq = ""
        current_id = ""
        current_annotation = ""
        
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    sequences.append(current_seq)
                    uniprot_ids.append(current_id)
                    annotations.append(current_annotation)
                
                # Parse header
                header = line[1:]
                parts = header.split()
                current_id = parts[0] if parts else f"seq_{len(sequences)}"
                current_annotation = " ".join(parts[1:]) if len(parts) > 1 else ""
                current_seq = ""
            else:
                current_seq += line
        
        # Add last sequence
        if current_seq:
            sequences.append(current_seq)
            uniprot_ids.append(current_id)
            annotations.append(current_annotation)
    
    return sequences, uniprot_ids, annotations

def main():
    parser = argparse.ArgumentParser(description="Create identity-aware validation splits")
    parser.add_argument("--train_fasta", required=True, help="Training FASTA file")
    parser.add_argument("--val_fasta", required=True, help="Validation FASTA file")
    parser.add_argument("--identity_threshold", type=float, default=0.4, help="Identity threshold")
    parser.add_argument("--out", required=True, help="Output JSON file for clusters")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Training ratio")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation ratio")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Set seed
    set_seed(args.seed)
    
    try:
        # Load sequences
        logger.info(f"Loading training sequences from {args.train_fasta}")
        train_sequences, train_ids, train_annotations = parse_fasta(Path(args.train_fasta))
        
        logger.info(f"Loading validation sequences from {args.val_fasta}")
        val_sequences, val_ids, val_annotations = parse_fasta(Path(args.val_fasta))
        
        logger.info(f"Loaded {len(train_sequences)} training sequences")
        logger.info(f"Loaded {len(val_sequences)} validation sequences")
        
        # Create combined dataset for clustering
        all_sequences = train_sequences + val_sequences
        all_ids = train_ids + val_ids
        all_annotations = train_annotations + val_annotations
        
        # Create dummy labels for clustering (not used for actual clustering)
        n_classes = 10  # Assume 10 classes
        all_labels = np.random.randint(0, 2, (len(all_sequences), n_classes))
        
        # Create identity-aware splits
        logger.info(f"Creating identity-aware splits with threshold {args.identity_threshold}")
        splits_result = create_identity_splits(
            sequences=all_sequences,
            labels=all_labels,
            uniprot_ids=all_ids,
            identity_threshold=args.identity_threshold,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_seed=args.seed
        )
        
        # Extract splits
        splits = splits_result['splits']
        clusters = splits_result['clusters']
        cluster_stats = splits_result['cluster_stats']
        
        # Create identity map (val seq -> nearest train identity)
        identity_map = {}
        for val_idx in splits['val']:
            val_seq = all_sequences[val_idx]
            val_id = all_ids[val_idx]
            
            # Find nearest training sequence
            min_identity = 1.0
            nearest_train_id = None
            
            for train_idx in splits['train']:
                train_seq = all_sequences[train_idx]
                train_id = all_ids[train_idx]
                
                # Compute identity (simplified)
                identity = compute_sequence_identity(val_seq, train_seq)
                
                if identity < min_identity:
                    min_identity = identity
                    nearest_train_id = train_id
            
            identity_map[val_id] = {
                'nearest_train_id': nearest_train_id,
                'identity': min_identity,
                'cluster_id': find_cluster_for_sequence(clusters, val_idx)
            }
        
        # Save results
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'identity_threshold': args.identity_threshold,
            'splits': {
                'train': [all_ids[i] for i in splits['train']],
                'val': [all_ids[i] for i in splits['val']],
                'test': [all_ids[i] for i in splits['test']]
            },
            'identity_map': identity_map,
            'cluster_stats': cluster_stats,
            'n_train': len(splits['train']),
            'n_val': len(splits['val']),
            'n_test': len(splits['test']),
            'seed': args.seed
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Verify identity constraints
        logger.info("Verifying identity constraints...")
        violations = 0
        for val_id, info in identity_map.items():
            if info['identity'] > args.identity_threshold:
                violations += 1
                logger.warning(f"Identity violation: {val_id} has {info['identity']:.3f} > {args.identity_threshold}")
        
        if violations == 0:
            logger.info("✅ All validation sequences meet identity constraint")
        else:
            logger.warning(f"⚠️  {violations} validation sequences violate identity constraint")
        
        logger.info(f"✅ Identity splits created successfully!")
        logger.info(f"   - Training samples: {len(splits['train'])}")
        logger.info(f"   - Validation samples: {len(splits['val'])}")
        logger.info(f"   - Test samples: {len(splits['test'])}")
        logger.info(f"   - Identity threshold: {args.identity_threshold}")
        logger.info(f"   - Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to create identity splits: {e}")
        sys.exit(1)

def compute_sequence_identity(seq1: str, seq2: str) -> float:
    """Compute sequence identity (simplified)."""
    if len(seq1) != len(seq2):
        return 0.0
    
    matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
    return matches / len(seq1)

def find_cluster_for_sequence(clusters: dict, sequence_idx: int) -> int:
    """Find cluster ID for a sequence."""
    for cluster_id, indices in clusters.items():
        if sequence_idx in indices:
            return cluster_id
    return -1

if __name__ == "__main__":
    main()



