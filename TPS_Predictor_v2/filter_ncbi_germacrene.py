#!/usr/bin/env python3
"""
Filter NCBI Germacrene FASTA file to remove sequences already in MARTS-DB training data.
This creates a truly novel dataset for testing the Germacrene classifier.
"""

import pandas as pd
from pathlib import Path
from Bio import SeqIO
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_uniprot_id(header: str) -> str:
    """Extracts UniProt ID from a FASTA header."""
    # Try UniProt format first
    match = re.search(r'\|([A-Z0-9]+)\|', header)
    if match:
        return match.group(1)
    
    # Try NCBI format patterns
    match = re.search(r'([A-Z0-9]{6,10})', header)
    if match:
        return match.group(1)
    
    # Fallback to first part
    return header.split(' ')[0]

def load_training_uniprot_ids(marts_db_path: str = 'reactions_expanded.csv') -> set:
    """Load UniProt IDs from the MARTS-DB training data."""
    logger.info("Loading training UniProt IDs from MARTS-DB...")
    marts_df = pd.read_csv(marts_db_path)
    
    # Get unique UniProt IDs from MARTS-DB
    training_uniprot_ids = set(marts_df['Uniprot_ID'].dropna().astype(str).str.strip().unique())
    
    logger.info(f"Loaded {len(training_uniprot_ids)} unique UniProt IDs from MARTS-DB")
    return training_uniprot_ids

def filter_ncbi_fasta(input_fasta: str, output_fasta: str, training_uniprot_ids: set):
    """Filter NCBI FASTA file to remove sequences already in training data."""
    
    logger.info(f"Filtering {input_fasta}...")
    
    filtered_sequences = []
    total_sequences = 0
    duplicate_sequences = 0
    
    # Parse FASTA file
    for record in SeqIO.parse(input_fasta, "fasta"):
        total_sequences += 1
        
        # Extract UniProt ID from the record ID
        uniprot_id = extract_uniprot_id(record.id)
        
        # Check if UniProt ID is in training data
        if uniprot_id in training_uniprot_ids:
            duplicate_sequences += 1
            logger.debug(f"Removing duplicate sequence: {uniprot_id}")
        else:
            filtered_sequences.append(record)
    
    # Write filtered sequences
    SeqIO.write(filtered_sequences, output_fasta, "fasta")
    
    logger.info(f"Filtering complete!")
    logger.info(f"Total sequences in NCBI file: {total_sequences}")
    logger.info(f"Sequences already in training data: {duplicate_sequences}")
    logger.info(f"Novel sequences for prediction: {len(filtered_sequences)}")
    logger.info(f"Filtered FASTA saved to: {output_fasta}")
    
    return len(filtered_sequences), duplicate_sequences

def main():
    """Main function"""
    
    input_fasta = "NCBI_germacrene.fasta"
    output_fasta = "NCBI_novel_germacrene_sequences.fasta"
    marts_db_path = "reactions_expanded.csv"
    
    # Check if input files exist
    if not Path(input_fasta).exists():
        logger.error(f"Input FASTA file not found: {input_fasta}")
        return
    if not Path(marts_db_path).exists():
        logger.error(f"MARTS-DB file not found: {marts_db_path}")
        return
    
    # Load training UniProt IDs
    training_uniprot_ids = load_training_uniprot_ids(marts_db_path)
    
    # Filter FASTA file
    novel_count, duplicate_count = filter_ncbi_fasta(
        input_fasta, output_fasta, training_uniprot_ids
    )
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"NCBI GERMACRENE FILTERING SUMMARY")
    print(f"{'='*70}")
    print(f"Original NCBI sequences: {novel_count + duplicate_count}")
    print(f"Sequences in MARTS-DB training data: {duplicate_count}")
    print(f"Truly novel sequences for testing: {novel_count}")
    print(f"Novelty rate: {novel_count/(novel_count + duplicate_count)*100:.1f}%")
    print(f"Filtered file: {output_fasta}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
