#!/usr/bin/env python3
"""
Filter NCBI Germacrene FASTA file to separate high-confidence and putative sequences.
This ensures we only use high-quality annotations for training.
"""

import pandas as pd
from pathlib import Path
from Bio import SeqIO
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def filter_ncbi_by_confidence(input_fasta: str, high_confidence_output: str, putative_output: str):
    """Filter NCBI FASTA file by confidence level based on annotations."""
    
    logger.info(f"Filtering {input_fasta} by confidence level...")
    
    high_confidence_sequences = []
    putative_sequences = []
    
    # Parse FASTA file
    for record in SeqIO.parse(input_fasta, "fasta"):
        description = record.description.lower()
        
        # Check for putative annotations
        if 'putative' in description:
            putative_sequences.append(record)
        else:
            high_confidence_sequences.append(record)
    
    # Write filtered sequences
    SeqIO.write(high_confidence_sequences, high_confidence_output, "fasta")
    SeqIO.write(putative_sequences, putative_output, "fasta")
    
    logger.info(f"Filtering complete!")
    logger.info(f"High confidence sequences: {len(high_confidence_sequences)}")
    logger.info(f"Putative sequences: {len(putative_sequences)}")
    logger.info(f"High confidence FASTA saved to: {high_confidence_output}")
    logger.info(f"Putative FASTA saved to: {putative_output}")
    
    return len(high_confidence_sequences), len(putative_sequences)

def main():
    """Main function"""
    
    input_fasta = "NCBI_germacrene.fasta"
    high_confidence_output = "NCBI_high_confidence_germacrene.fasta"
    putative_output = "NCBI_putative_germacrene.fasta"
    
    # Check if input file exists
    if not Path(input_fasta).exists():
        logger.error(f"Input FASTA file not found: {input_fasta}")
        return
    
    # Filter by confidence level
    high_conf_count, putative_count = filter_ncbi_by_confidence(
        input_fasta, high_confidence_output, putative_output
    )
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"NCBI GERMACRENE CONFIDENCE FILTERING SUMMARY")
    print(f"{'='*70}")
    print(f"Original NCBI sequences: {high_conf_count + putative_count}")
    print(f"High confidence sequences: {high_conf_count} ({high_conf_count/(high_conf_count + putative_count)*100:.1f}%)")
    print(f"Putative sequences (excluded): {putative_count} ({putative_count/(high_conf_count + putative_count)*100:.1f}%)")
    print(f"High confidence file: {high_confidence_output}")
    print(f"Putative file: {putative_output}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
