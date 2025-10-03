#!/usr/bin/env python3
"""
Add high-confidence NCBI Germacrene sequences to MARTS-DB training data.
This expands the training dataset with diverse, high-quality Germacrene sequences.
"""

import pandas as pd
from pathlib import Path
from Bio import SeqIO
import logging
import re
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

def load_marts_db(marts_db_path: str = 'reactions_expanded.csv') -> pd.DataFrame:
    """Loads the existing MARTS-DB data."""
    return pd.read_csv(marts_db_path)

def get_training_uniprot_ids(marts_df: pd.DataFrame) -> set:
    """Extracts unique UniProt IDs from the MARTS-DB DataFrame."""
    return set(marts_df['Uniprot_ID'].dropna().astype(str).str.strip().unique())

def parse_ncbi_germacrene_fasta(fasta_path: str, existing_uniprot_ids: set) -> List[Dict[str, Any]]:
    """
    Parses the high-confidence NCBI Germacrene FASTA file and identifies novel sequences.
    Returns a list of dictionaries for novel sequences.
    """
    novel_sequences_data = []
    total_ncbi_fasta = 0
    already_in_marts_db = 0

    logger.info(f"Loading high-confidence NCBI Germacrene sequences from: {fasta_path}")
    for record in SeqIO.parse(fasta_path, "fasta"):
        total_ncbi_fasta += 1
        uniprot_id = extract_uniprot_id(record.id)
        
        if uniprot_id in existing_uniprot_ids:
            already_in_marts_db += 1
        else:
            # This is a novel high-confidence Germacrene synthase
            # Extract organism from description
            organism = "unknown"
            if 'OS=' in record.description:
                organism = record.description.split('OS=')[-1].split('OX=')[0].strip()
            elif '[' in record.description and ']' in record.description:
                organism = record.description.split('[')[-1].split(']')[0].strip()
            
            novel_sequences_data.append({
                'Uniprot_ID': uniprot_id,
                'Enzyme_marts_ID': f"ncbi_{uniprot_id}",  # Create NCBI-specific ID
                'Enzyme_name': record.description,  # Use description as enzyme name
                'Species': organism,
                'Sequence': str(record.seq),
                'Product_name': 'Germacrene',  # Explicitly Germacrene
                'Product_SMILES': 'C=C1CCC(C(C)=C)C(C)C1',  # Generic Germacrene SMILES
                'EC_number': 'unknown',  # Can be researched if needed
                'Confidence': 1.0,  # High confidence as they are curated NCBI sequences
                'Source': 'NCBI_High_Confidence_Germacrene',
                'Reference': 'NCBI_Curated_Database'
            })
    
    logger.info(f"Total NCBI Germacrene sequences: {total_ncbi_fasta}")
    logger.info(f"Sequences already in MARTS-DB: {already_in_marts_db}")
    logger.info(f"Novel sequences (not in MARTS-DB): {len(novel_sequences_data)}")
    
    return novel_sequences_data

def main():
    marts_db_path = 'reactions_expanded.csv'
    ncbi_fasta_path = 'NCBI_high_confidence_germacrene.fasta'
    output_marts_db_path = 'reactions_with_ncbi_germacrene.csv'
    backup_marts_db_path = 'reactions_expanded_backup.csv'

    logger.info("Starting MARTS-DB expansion with high-confidence NCBI Germacrene sequences...")

    # 1. Load existing MARTS-DB data
    marts_df = load_marts_db(marts_db_path)
    original_marts_size = len(marts_df)
    logger.info(f"Original MARTS-DB size: {original_marts_size} sequences")

    # 2. Get existing UniProt IDs from MARTS-DB
    existing_uniprot_ids = get_training_uniprot_ids(marts_df)

    # 3. Parse the NCBI Germacrene FASTA and get novel sequences
    novel_ncbi_data = parse_ncbi_germacrene_fasta(ncbi_fasta_path, existing_uniprot_ids)
    
    if not novel_ncbi_data:
        logger.info("No novel NCBI Germacrene sequences found to add. Exiting.")
        return

    logger.info(f"Found {len(novel_ncbi_data)} novel NCBI Germacrene sequences to add")

    # 4. Create a DataFrame for novel sequences
    novel_df = pd.DataFrame(novel_ncbi_data)

    # 5. Backup original MARTS-DB
    marts_df.to_csv(backup_marts_db_path, index=False)
    logger.info(f"Creating backup: {backup_marts_db_path}")

    # 6. Concatenate original and novel data
    expanded_marts_df = pd.concat([marts_df, novel_df], ignore_index=True)
    
    # Ensure no duplicate UniProt_IDs if any slipped through
    expanded_marts_df.drop_duplicates(subset=['Uniprot_ID'], inplace=True)

    # 7. Save the expanded dataset
    expanded_marts_df.to_csv(output_marts_db_path, index=False)
    logger.info(f"Saving expanded dataset: {output_marts_db_path}")
    logger.info(f"Expanded MARTS-DB size: {len(expanded_marts_df)} sequences")
    logger.info(f"Added {len(novel_ncbi_data)} novel NCBI Germacrene sequences")

    # 8. Analyze the expanded dataset
    germacrene_count = sum(1 for _, row in expanded_marts_df.iterrows() 
                          if 'germacrene' in str(row['Product_name']).lower())
    total_germacrene_percentage = germacrene_count / len(expanded_marts_df) * 100

    print(f"\n{'='*70}")
    print(f"MARTS-DB EXPANSION WITH NCBI GERMACRENE COMPLETE")
    print(f"{'='*70}")
    print(f"Original sequences: {original_marts_size}")
    print(f"Novel NCBI Germacrene sequences added: {len(novel_ncbi_data)}")
    print(f"New total sequences: {len(expanded_marts_df)}")
    print(f"Total Germacrene sequences: {germacrene_count}")
    print(f"Germacrene percentage: {total_germacrene_percentage:.1f}%")
    print(f"Expansion: {((len(expanded_marts_df) - original_marts_size) / original_marts_size) * 100:.1f}% increase")
    print(f"Backup saved to: {backup_marts_db_path}")
    print(f"Expanded data saved to: {output_marts_db_path}")
    print(f"{'='*70}")
    logger.info("✅ MARTS-DB expansion with NCBI Germacrene sequences completed successfully!")
    logger.info(f"Ready to retrain model with expanded dataset: {output_marts_db_path}")

if __name__ == "__main__":
    main()
