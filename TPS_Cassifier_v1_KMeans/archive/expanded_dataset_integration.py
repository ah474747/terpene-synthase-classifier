#!/usr/bin/env python3
"""
Expanded Dataset Integration for Germacrene Synthase Classification
Combines MARTS-DB and NCBI datasets with proper deduplication
"""

import os
import pandas as pd
import numpy as np
from Bio import SeqIO
import hashlib
from pathlib import Path
import logging
from typing import List, Dict, Tuple
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExpandedDatasetIntegrator:
    """Integrate MARTS-DB and NCBI datasets for germacrene synthase classification"""
    
    def __init__(self, output_dir: str = "data/expanded_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'marts_germacrene': 0,
            'ncbi_germacrene': 0,
            'overlap_sequences': 0,
            'unique_germacrene': 0,
            'non_germacrene': 0,
            'total_sequences': 0
        }
    
    def load_marts_db_data(self) -> List[Dict]:
        """Load germacrene sequences from MARTS-DB"""
        logger.info("Loading MARTS-DB germacrene sequences...")
        
        # Load MARTS-DB data
        df = pd.read_csv('data/marts_db_enhanced.csv')
        germacrene_df = df[df['is_germacrene'] == 1]
        
        marts_sequences = []
        for _, row in germacrene_df.iterrows():
            sequence = row.get('Aminoacid_sequence', '')
            if pd.isna(sequence) or sequence == '':
                continue
                
            marts_sequences.append({
                'id': row.get('Enzyme_marts_ID', 'unknown'),
                'description': f"{row.get('Enzyme_name', '')} [{row.get('Species', '')}]",
                'sequence': sequence,
                'length': len(sequence),
                'source': 'marts_db',
                'hash': hashlib.md5(sequence.encode()).hexdigest(),
                'is_germacrene': 1
            })
        
        self.stats['marts_germacrene'] = len(marts_sequences)
        logger.info(f"Loaded {len(marts_sequences)} germacrene sequences from MARTS-DB")
        return marts_sequences
    
    def load_ncbi_data(self) -> List[Dict]:
        """Load germacrene sequences from NCBI"""
        logger.info("Loading NCBI germacrene sequences...")
        
        ncbi_sequences = []
        for record in SeqIO.parse('data/phase2/processed/germacrene_sequences.fasta', 'fasta'):
            ncbi_sequences.append({
                'id': record.id,
                'description': record.description,
                'sequence': str(record.seq),
                'length': len(record.seq),
                'source': 'ncbi',
                'hash': hashlib.md5(str(record.seq).encode()).hexdigest(),
                'is_germacrene': 1
            })
        
        self.stats['ncbi_germacrene'] = len(ncbi_sequences)
        logger.info(f"Loaded {len(ncbi_sequences)} germacrene sequences from NCBI")
        return ncbi_sequences
    
    def load_non_germacrene_data(self) -> List[Dict]:
        """Load non-germacrene sequences from MARTS-DB"""
        logger.info("Loading non-germacrene sequences from MARTS-DB...")
        
        # Load MARTS-DB data
        df = pd.read_csv('data/marts_db_enhanced.csv')
        non_germacrene_df = df[df['is_germacrene'] == 0]
        
        non_germacrene_sequences = []
        for _, row in non_germacrene_df.iterrows():
            sequence = row.get('Aminoacid_sequence', '')
            if pd.isna(sequence) or sequence == '':
                continue
                
            non_germacrene_sequences.append({
                'id': row.get('Enzyme_marts_ID', 'unknown'),
                'description': f"{row.get('Enzyme_name', '')} [{row.get('Species', '')}]",
                'sequence': sequence,
                'length': len(sequence),
                'source': 'marts_db',
                'hash': hashlib.md5(sequence.encode()).hexdigest(),
                'is_germacrene': 0
            })
        
        self.stats['non_germacrene'] = len(non_germacrene_sequences)
        logger.info(f"Loaded {len(non_germacrene_sequences)} non-germacrene sequences from MARTS-DB")
        return non_germacrene_sequences
    
    def deduplicate_germacrene_sequences(self, marts_sequences: List[Dict], 
                                       ncbi_sequences: List[Dict]) -> Tuple[List[Dict], int]:
        """Remove duplicate germacrene sequences and track overlap"""
        logger.info("Deduplicating germacrene sequences...")
        
        # Create hash sets for overlap detection
        marts_hashes = {seq['hash'] for seq in marts_sequences}
        ncbi_hashes = {seq['hash'] for seq in ncbi_sequences}
        
        # Find overlapping sequences
        overlap_hashes = marts_hashes.intersection(ncbi_hashes)
        self.stats['overlap_sequences'] = len(overlap_hashes)
        
        # Combine sequences, prioritizing MARTS-DB for duplicates
        unique_sequences = []
        seen_hashes = set()
        
        # Add MARTS-DB sequences first
        for seq in marts_sequences:
            if seq['hash'] not in seen_hashes:
                unique_sequences.append(seq)
                seen_hashes.add(seq['hash'])
        
        # Add NCBI sequences that don't overlap
        for seq in ncbi_sequences:
            if seq['hash'] not in seen_hashes:
                unique_sequences.append(seq)
                seen_hashes.add(seq['hash'])
        
        self.stats['unique_germacrene'] = len(unique_sequences)
        logger.info(f"Found {len(overlap_hashes)} overlapping sequences")
        logger.info(f"Total unique germacrene sequences: {len(unique_sequences)}")
        
        return unique_sequences, len(overlap_hashes)
    
    def create_expanded_dataset(self) -> pd.DataFrame:
        """Create the expanded dataset combining all sequences"""
        logger.info("Creating expanded dataset...")
        
        # Load all data
        marts_germacrene = self.load_marts_db_data()
        ncbi_germacrene = self.load_ncbi_data()
        non_germacrene = self.load_non_germacrene_data()
        
        # Deduplicate germacrene sequences
        unique_germacrene, overlap_count = self.deduplicate_germacrene_sequences(
            marts_germacrene, ncbi_germacrene
        )
        
        # Combine all sequences
        all_sequences = unique_germacrene + non_germacrene
        self.stats['total_sequences'] = len(all_sequences)
        
        # Create DataFrame
        df = pd.DataFrame(all_sequences)
        
        # Add additional features
        df['sequence_length'] = df['sequence'].str.len()
        df['gc_content'] = df['sequence'].apply(self._calculate_gc_content)
        df['molecular_weight'] = df['sequence'].apply(self._calculate_molecular_weight)
        
        logger.info(f"Created expanded dataset with {len(df)} sequences")
        logger.info(f"Germacrene sequences: {len(unique_germacrene)}")
        logger.info(f"Non-germacrene sequences: {len(non_germacrene)}")
        logger.info(f"Class balance: {len(unique_germacrene)/len(df)*100:.1f}% positive")
        
        return df
    
    def _calculate_gc_content(self, sequence: str) -> float:
        """Calculate GC content for protein sequence (not applicable, but included for completeness)"""
        if not sequence:
            return 0.0
        # For proteins, this is not meaningful, but we'll calculate it anyway
        gc_count = sequence.count('G') + sequence.count('C')
        return gc_count / len(sequence) if len(sequence) > 0 else 0.0
    
    def _calculate_molecular_weight(self, sequence: str) -> float:
        """Calculate approximate molecular weight of protein sequence"""
        if not sequence:
            return 0.0
        
        # Amino acid molecular weights (Da)
        aa_weights = {
            'A': 89.09, 'R': 174.20, 'N': 132.12, 'D': 133.10, 'C': 121.16,
            'Q': 146.15, 'E': 147.13, 'G': 75.07, 'H': 155.16, 'I': 131.17,
            'L': 131.17, 'K': 146.19, 'M': 149.21, 'F': 165.19, 'P': 115.13,
            'S': 105.09, 'T': 119.12, 'W': 204.23, 'Y': 181.19, 'V': 117.15
        }
        
        total_weight = sum(aa_weights.get(aa, 0) for aa in sequence.upper())
        # Subtract weight of water molecules (18.015 Da per peptide bond)
        peptide_bonds = len(sequence) - 1
        return total_weight - (peptide_bonds * 18.015)
    
    def save_dataset(self, df: pd.DataFrame) -> None:
        """Save the expanded dataset in multiple formats"""
        logger.info("Saving expanded dataset...")
        
        # Save as CSV
        csv_path = self.output_dir / "expanded_germacrene_dataset.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved dataset to {csv_path}")
        
        # Save as FASTA
        fasta_path = self.output_dir / "expanded_germacrene_dataset.fasta"
        with open(fasta_path, 'w') as f:
            for _, row in df.iterrows():
                f.write(f">{row['id']} {row['description']}\n")
                f.write(f"{row['sequence']}\n")
        logger.info(f"Saved FASTA to {fasta_path}")
        
        # Save statistics
        stats_path = self.output_dir / "dataset_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        logger.info(f"Saved statistics to {stats_path}")
        
        # Save germacrene sequences separately
        germacrene_df = df[df['is_germacrene'] == 1]
        germacrene_path = self.output_dir / "germacrene_sequences.fasta"
        with open(germacrene_path, 'w') as f:
            for _, row in germacrene_df.iterrows():
                f.write(f">{row['id']} {row['description']}\n")
                f.write(f"{row['sequence']}\n")
        logger.info(f"Saved {len(germacrene_df)} germacrene sequences to {germacrene_path}")
    
    def print_summary(self) -> None:
        """Print dataset summary"""
        print("\n" + "="*60)
        print("EXPANDED DATASET SUMMARY")
        print("="*60)
        print(f"MARTS-DB germacrene sequences: {self.stats['marts_germacrene']}")
        print(f"NCBI germacrene sequences: {self.stats['ncbi_germacrene']}")
        print(f"Overlapping sequences: {self.stats['overlap_sequences']}")
        print(f"Unique germacrene sequences: {self.stats['unique_germacrene']}")
        print(f"Non-germacrene sequences: {self.stats['non_germacrene']}")
        print(f"Total sequences: {self.stats['total_sequences']}")
        print(f"Class balance: {self.stats['unique_germacrene']/self.stats['total_sequences']*100:.1f}% positive")
        print("="*60)

def main():
    """Main function to create expanded dataset"""
    integrator = ExpandedDatasetIntegrator()
    
    # Create expanded dataset
    df = integrator.create_expanded_dataset()
    
    # Save dataset
    integrator.save_dataset(df)
    
    # Print summary
    integrator.print_summary()
    
    print("\n✓ Expanded dataset creation completed successfully!")
    print(f"✓ Dataset saved to: {integrator.output_dir}")

if __name__ == "__main__":
    main()
