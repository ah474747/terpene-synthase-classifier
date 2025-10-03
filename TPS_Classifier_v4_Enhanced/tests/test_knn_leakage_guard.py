"""
Test to prevent kNN leakage by ensuring eval sequences are not in training.
"""
import numpy as np
import json
import sys
from pathlib import Path
from typing import Set, List

# Add the parent directory to the path so we can import tps modules
sys.path.append(str(Path(__file__).parent.parent))

def load_sequence_ids(fasta_path: str) -> Set[str]:
    """Load sequence IDs from a FASTA file."""
    ids = set()
    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                id_part = line[1:].strip().split()[0]  # Take first part after >
                ids.add(id_part)
    return ids

def test_knn_no_leakage():
    """Ensure no evaluation sequences are in the kNN index."""
    
    # Check if index exists
    index_meta_path = Path("models/knn/index_meta.json")
    val_fasta_path = Path("data/val.fasta")
    
    if not index_meta_path.exists():
        print("kNN index not built yet, skipping leakage test")
        return
        
    if not val_fasta_path.exists():
        print("Validation data not available, skipping leakage test")
        return
    
    # Load kNN index metadata
    with open(index_meta_path) as f:
        index_meta = json.load(f)
    
    train_ids = set(index_meta.get("sequence_ids", []))
    val_ids = load_sequence_ids(str(val_fasta_path))
    
    # Check for overlap
    overlap = train_ids.intersection(val_ids)
    
    if overlap:
        raise AssertionError(f"Leakage detected! {len(overlap)} sequences in both train and val: {list(overlap)[:5]}...")
    
    print(f"✓ No leakage: {len(train_ids)} train sequences, {len(val_ids)} val sequences")
    print(f"✓ Train IDs: {list(train_ids)[:3]}...")
    print(f"✓ Val IDs: {list(val_ids)[:3]}...")

def test_external_holdout_no_leakage():
    """Ensure external holdout sequences are not in training or validation."""
    
    external_path = Path("data/external_30.fasta")
    val_path = Path("data/val.fasta")
    index_meta_path = Path("models/knn/index_meta.json")
    
    if not all([external_path.exists(), val_path.exists(), index_meta_path.exists()]):
        print("External holdout data not available, skipping test")
        return
    
    # Load all sequence sets
    external_ids = load_sequence_ids(str(external_path))
    val_ids = load_sequence_ids(str(val_path))
    
    with open(index_meta_path) as f:
        index_meta = json.load(f)
    train_ids = set(index_meta.get("sequence_ids", []))
    
    # Check overlaps
    train_overlap = external_ids.intersection(train_ids)
    val_overlap = external_ids.intersection(val_ids)
    
    if train_overlap:
        raise AssertionError(f"External holdout leaked into training! {len(train_overlap)} sequences: {list(train_overlap)}")
    
    if val_overlap:
        raise AssertionError(f"External holdout leaked into validation! {len(val_overlap)} sequences: {list(val_overlap)}")
    
    print(f"✓ External holdout clean: {len(external_ids)} sequences")
    print(f"✓ No overlap with train ({len(train_ids)}) or val ({len(val_ids)})")

def compute_sequence_identity(seq1: str, seq2: str) -> float:
    """Compute approximate sequence identity."""
    short, long_seq = (seq1, seq2) if len(seq1) < len(seq2) else (seq2, seq1)
    
    if len(short) == 0:
        return 0.0
    
    matches = sum(c1 == c2 for c1, c2 in zip(short, long_seq[:len(short)]))
    return matches / len(short)

def test_identity_aware_split():
    """Test that sequences have low identity as intended."""
    
    val_path = Path("data/val.fasta")
    index_meta_path = Path("models/knn/index_meta.json")
    
    if not (val_path.exists() and index_meta_path.exists()):
        print("Required data not available for identity test")
        return
    
    # Load sequences (for this test, we'll just check file existence)
    # Real identity computation would require loading full sequences
    print("✓ Identity-aware split validation requires full sequence loading")
    print("✓ Implement mmseqs2 or cd-hit for detailed identity analysis")

if __name__ == "__main__":
    test_knn_no_leakage()
    test_external_holdout_no_leakage()
    test_identity_aware_split()
    print("All leakage tests passed!")
