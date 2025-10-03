#!/usr/bin/env python3
"""
CORRECTED: Filter UniProt FASTA file to remove sequences that were already in MARTS-DB training data.
This version correctly compares UniProt IDs instead of amino acid sequences.
"""

import pandas as pd
from pathlib import Path
from Bio import SeqIO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_training_uniprot_ids():
    """Load UniProt IDs from the MARTS-DB training data"""
    logger.info("Loading training UniProt IDs from MARTS-DB...")
    
    # Load MARTS-DB data directly
    marts_data = pd.read_csv('reactions.csv')
    
    # Get unique UniProt IDs from MARTS-DB
    training_uniprot_ids = set(marts_data['Uniprot_ID'].dropna())
    
    logger.info(f"Loaded {len(training_uniprot_ids)} unique UniProt IDs from MARTS-DB")
    return training_uniprot_ids

def filter_fasta_file(input_fasta, output_fasta, training_uniprot_ids):
    """Filter FASTA file to remove sequences already in training data by UniProt ID"""
    
    logger.info(f"Filtering {input_fasta}...")
    
    filtered_sequences = []
    total_sequences = 0
    duplicate_sequences = 0
    
    # Parse FASTA file
    for record in SeqIO.parse(input_fasta, "fasta"):
        total_sequences += 1
        
        # Extract UniProt ID from sequence ID
        seq_id = record.id
        if '|' in seq_id:
            uniprot_id = seq_id.split('|')[1]
        else:
            uniprot_id = seq_id
        
        # Check if UniProt ID is in training data
        if uniprot_id in training_uniprot_ids:
            duplicate_sequences += 1
            logger.debug(f"Removing duplicate sequence: {seq_id} (UniProt: {uniprot_id})")
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
    output_fasta = "uniprot_novel_sequences_CORRECTED.fasta"
    
    # Check if input file exists
    if not Path(input_fasta).exists():
        logger.error(f"Input FASTA file not found: {input_fasta}")
        return
    
    # Load training UniProt IDs
    training_uniprot_ids = load_training_uniprot_ids()
    
    # Filter FASTA file
    novel_count, duplicate_count = filter_fasta_file(
        input_fasta, output_fasta, training_uniprot_ids
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"CORRECTED FILTERING SUMMARY")
    print(f"{'='*60}")
    print(f"Original UniProt sequences: {novel_count + duplicate_count}")
    print(f"Sequences in MARTS-DB training data: {duplicate_count}")
    print(f"Truly novel sequences for testing: {novel_count}")
    print(f"Novelty rate: {novel_count/(novel_count + duplicate_count)*100:.1f}%")
    print(f"Corrected filtered file: {output_fasta}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
