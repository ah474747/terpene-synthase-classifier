#!/usr/bin/env python3
"""
Filter UniProt FASTA file to remove sequences that were already in MARTS-DB training data.
This ensures we only predict on truly novel sequences for proper generalization testing.
"""

import pickle
import pandas as pd
from pathlib import Path
from Bio import SeqIO
import hashlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_sequence_hash(sequence):
    """Get a hash of the sequence for comparison"""
    return hashlib.md5(sequence.encode()).hexdigest()

def load_training_sequences():
    """Load sequences from the training data"""
    logger.info("Loading training sequences from MARTS-DB...")
    
    with open('data/cache/terpene_predictor_pipeline.pkl', 'rb') as f:
        pipeline_data = pickle.load(f)
    
    processed_data = pipeline_data['processed_data']
    
    # Get unique sequences and their hashes
    training_sequences = set()
    training_hashes = set()
    
    for _, row in processed_data.iterrows():
        seq = row['sequence'].upper().strip()
        training_sequences.add(seq)
        training_hashes.add(get_sequence_hash(seq))
    
    logger.info(f"Loaded {len(training_sequences)} unique training sequences")
    return training_sequences, training_hashes

def filter_fasta_file(input_fasta, output_fasta, training_sequences, training_hashes):
    """Filter FASTA file to remove sequences already in training data"""
    
    logger.info(f"Filtering {input_fasta}...")
    
    filtered_sequences = []
    total_sequences = 0
    duplicate_sequences = 0
    
    # Parse FASTA file
    for record in SeqIO.parse(input_fasta, "fasta"):
        total_sequences += 1
        
        # Get sequence and normalize
        sequence = str(record.seq).upper().strip()
        seq_hash = get_sequence_hash(sequence)
        
        # Check if sequence is in training data
        if sequence in training_sequences or seq_hash in training_hashes:
            duplicate_sequences += 1
            logger.debug(f"Removing duplicate sequence: {record.id}")
        else:
            filtered_sequences.append(record)
    
    # Write filtered sequences
    SeqIO.write(filtered_sequences, output_fasta, "fasta")
    
    logger.info(f"Filtering complete!")
    logger.info(f"Total sequences in UniProt file: {total_sequences}")
    logger.info(f"Sequences already in training data: {duplicate_sequences}")
    logger.info(f"Novel sequences for prediction: {len(filtered_sequences)}")
    logger.info(f"Filtered FASTA saved to: {output_fasta}")
    
    return len(filtered_sequences), duplicate_sequences

def main():
    """Main function"""
    
    input_fasta = "uniprotkb_terpene_synthase_AND_reviewed_2025_09_24.fasta"
    output_fasta = "uniprot_novel_sequences.fasta"
    
    # Check if input file exists
    if not Path(input_fasta).exists():
        logger.error(f"Input FASTA file not found: {input_fasta}")
        return
    
    # Load training sequences
    training_sequences, training_hashes = load_training_sequences()
    
    # Filter FASTA file
    novel_count, duplicate_count = filter_fasta_file(
        input_fasta, output_fasta, training_sequences, training_hashes
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"FILTERING SUMMARY")
    print(f"{'='*60}")
    print(f"Original UniProt sequences: {novel_count + duplicate_count}")
    print(f"Sequences in training data: {duplicate_count}")
    print(f"Novel sequences for testing: {novel_count}")
    print(f"Novelty rate: {novel_count/(novel_count + duplicate_count)*100:.1f}%")
    print(f"Filtered file: {output_fasta}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
