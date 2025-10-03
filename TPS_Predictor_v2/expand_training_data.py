#!/usr/bin/env python3
"""
Expand MARTS-DB training data with novel Germacrene synthase sequences from UniProt.
This script adds 13 experimentally validated Germacrene synthases to improve model training.
"""

import pandas as pd
from Bio import SeqIO
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_novel_germacrene_sequences():
    """Load the 13 novel Germacrene synthase sequences from UniProt FASTA"""
    
    germacrene_fasta = '../terpene_synthase_predictor/uniprotkb_germacrene_synthase_AND_revie_2025_09_24.fasta'
    
    # Load MARTS-DB to check which sequences are already present
    marts_data = pd.read_csv('reactions.csv')
    marts_uniprot_ids = set(marts_data['Uniprot_ID'].dropna())
    
    novel_sequences = []
    
    logger.info(f"Loading Germacrene sequences from: {germacrene_fasta}")
    
    for record in SeqIO.parse(germacrene_fasta, 'fasta'):
        # Extract UniProt ID from sequence ID
        seq_id = record.id
        if '|' in seq_id:
            uniprot_id = seq_id.split('|')[1]
            enzyme_name = seq_id.split('|')[2] if len(seq_id.split('|')) > 2 else 'Unknown'
        else:
            uniprot_id = seq_id
            enzyme_name = 'Unknown'
        
        # Only include sequences NOT already in MARTS-DB
        if uniprot_id not in marts_uniprot_ids:
            # Extract organism from enzyme name (format: ENZYME_ORGANISM)
            organism = enzyme_name.split('_')[-1] if '_' in enzyme_name else 'Unknown'
            
            novel_sequences.append({
                'Uniprot_ID': uniprot_id,
                'Enzyme_marts_ID': f'germacrene_{uniprot_id}',  # Create unique ID
                'Enzyme_name': enzyme_name,
                'Species': organism,
                'Product_name': 'Germacrene',  # Generic Germacrene product
                'Product_smiles': 'C/C1=C/C[C@H](C(C)(C)O)CC/C(C)=C/CC1',  # Generic Germacrene SMILES
                'Product_chebi': 'CHEBI:15393',  # Germacrene CHEBI ID
                'Sequence': str(record.seq),
                'Source': 'UniProt_Germacrene_FASTA',
                'Reference': 'UniProt_curated'
            })
    
    logger.info(f"Found {len(novel_sequences)} novel Germacrene sequences to add")
    return novel_sequences

def expand_marts_db():
    """Expand MARTS-DB with novel Germacrene sequences"""
    
    logger.info("Loading existing MARTS-DB data...")
    marts_data = pd.read_csv('reactions.csv')
    
    logger.info(f"Original MARTS-DB size: {len(marts_data)} sequences")
    
    # Load novel sequences
    novel_sequences = load_novel_germacrene_sequences()
    
    if not novel_sequences:
        logger.warning("No novel sequences found to add")
        return
    
    # Create DataFrame from novel sequences
    novel_df = pd.DataFrame(novel_sequences)
    
    # Combine with existing data
    expanded_data = pd.concat([marts_data, novel_df], ignore_index=True)
    
    # Save expanded dataset
    backup_file = 'reactions_backup.csv'
    expanded_file = 'reactions_expanded.csv'
    
    logger.info(f"Creating backup: {backup_file}")
    marts_data.to_csv(backup_file, index=False)
    
    logger.info(f"Saving expanded dataset: {expanded_file}")
    expanded_data.to_csv(expanded_file, index=False)
    
    logger.info(f"Expanded MARTS-DB size: {len(expanded_data)} sequences")
    logger.info(f"Added {len(novel_sequences)} novel Germacrene sequences")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TRAINING DATA EXPANSION COMPLETE")
    print(f"{'='*60}")
    print(f"Original sequences: {len(marts_data)}")
    print(f"Novel Germacrene sequences added: {len(novel_sequences)}")
    print(f"New total sequences: {len(expanded_data)}")
    print(f"Expansion: {len(novel_sequences)/len(marts_data)*100:.1f}% increase")
    print(f"Backup saved to: {backup_file}")
    print(f"Expanded data saved to: {expanded_file}")
    print(f"{'='*60}")
    
    return expanded_file

def main():
    """Main function"""
    
    logger.info("Starting MARTS-DB expansion with novel Germacrene sequences...")
    
    # Check if Germacrene FASTA file exists
    germacrene_fasta = Path('../terpene_synthase_predictor/uniprotkb_germacrene_synthase_AND_revie_2025_09_24.fasta')
    if not germacrene_fasta.exists():
        logger.error(f"Germacrene FASTA file not found: {germacrene_fasta}")
        return
    
    # Expand the dataset
    expanded_file = expand_marts_db()
    
    if expanded_file:
        logger.info("✅ Training data expansion completed successfully!")
        logger.info(f"Ready to retrain model with expanded dataset: {expanded_file}")
    else:
        logger.error("❌ Training data expansion failed")

if __name__ == "__main__":
    main()
