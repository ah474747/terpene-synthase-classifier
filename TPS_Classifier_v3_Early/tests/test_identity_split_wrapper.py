#!/usr/bin/env python3
"""
Test: Identity Split Wrapper
Verifies that identity-aware splits work correctly.
"""

from tps.eval.identity_split import IdentitySplitter


def test_identity_split_basic():
    """Test basic identity split functionality."""
    sequences = {
        'seq1': 'MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQTLSEQVQEELLSSQVTQELRALMDETAQN',
        'seq2': 'MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQTLSEQVQEELLSSQVTQELRALMDETAQN',  # Identical
        'seq3': 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA',  # Very different
    }
    
    splitter = IdentitySplitter(identity_threshold=0.4)
    clusters = splitter.cluster_sequences(sequences)
    
    assert len(clusters) >= 1, "Should have at least one cluster"
    
    # Check that identical sequences are clustered together
    cluster_ids = {}
    for cluster_id, seq_list in clusters.items():
        for seq_id in seq_list:
            cluster_ids[seq_id] = cluster_id
    
    assert cluster_ids['seq1'] == cluster_ids['seq2'], "Identical sequences should be in same cluster"
    
    print("✓ Basic identity split works correctly")


def test_identity_split_biopython_fallback():
    """Test Biopython fallback when MMseqs2 is not available."""
    sequences = {
        'seq1': 'MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQTLSEQVQEELLSSQVTQELRALMDETAQN',
        'seq2': 'MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQTLSEQVQEELLSSQVTQELRALMDETAQN'
    }
    
    # Force Biopython fallback
    splitter = IdentitySplitter(identity_threshold=0.4, mmseqs_available=False)
    clusters = splitter.cluster_sequences(sequences)
    
    assert len(clusters) >= 1, "Biopython fallback should work"
    
    print("✓ Biopython fallback works correctly")


if __name__ == "__main__":
    print("Testing identity split wrapper...")
    test_identity_split_basic()
    test_identity_split_biopython_fallback()
    print("\n✅ All identity split tests passed!")