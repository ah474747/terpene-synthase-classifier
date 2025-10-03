#!/usr/bin/env python3
"""
Process downloaded terpene synthase sequences from NCBI and UniProt
"""

import os
import re
import hashlib
import pandas as pd
import numpy as np
from Bio import SeqIO
from typing import List, Dict, Set, Tuple
import json
from pathlib import Path
import argparse
from collections import defaultdict, Counter
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TerpeneSequenceProcessor:
    """Process and clean downloaded terpene synthase sequences"""
    
    def __init__(self, input_dir: str = "data/phase2/downloaded", output_dir: str = "data/phase2/processed"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Known terpene products for parsing
        self.terpene_products = {
            # Monoterpenes
            'limonene', 'pinene', 'myrcene', 'ocimene', 'linalool', 'geraniol', 'menthol',
            'camphor', 'borneol', 'cineole', 'terpinene', 'phellandrene', 'sabinene',
            'camphene', 'carene', 'terpinolene', 'thujene', 'fenchone', 'thujone',
            
            # Sesquiterpenes
            'germacrene', 'caryophyllene', 'humulene', 'farnesene', 'bisabolene', 'selinene',
            'elemene', 'gurjunene', 'patchoulene', 'valencene', 'nootkatone', 'santalene',
            'cedrene', 'vetivene', 'guaiene', 'bulnesene', 'spathulenol', 'cubebene',
            'copaene', 'muurolene', 'amorphene', 'germacrene-d', 'germacrene-a', 'germacrene-b',
            'germacrene-c', 'germacrene-e', 'germacrene-f', 'germacrene-g', 'germacrene-h',
            
            # Diterpenes
            'taxadiene', 'kaurene', 'abietadiene', 'pimaradiene', 'labdadiene', 'copalyl',
            'ent-kaurene', 'ent-copalyl', 'ent-pimaradiene', 'ent-abietadiene',
            
            # Triterpenes
            'squalene', 'lupeol', 'beta-amyrin', 'alpha-amyrin', 'taraxerol', 'friedelin',
            'lupane', 'ursane', 'oleanane', 'taraxerane', 'friedelane', 'hopane',
            
            # Other terpenoids
            'carotenoid', 'sterol', 'phytol', 'retinol', 'dolichol', 'ubiquinone'
        }
        
        # Quality filters
        self.min_length = 200
        self.max_length = 1000
        # Note: GC content filter removed - not applicable to protein sequences
        
        # Statistics
        self.stats = {
            'total_sequences': 0,
            'after_length_filter': 0,
            'after_gc_filter': 0,
            'after_duplicate_removal': 0,
            'after_quality_filter': 0,
            'germacrene_sequences': 0,
            'other_terpene_sequences': 0,
            'unknown_sequences': 0
        }
    
    def load_fasta_files(self) -> List[Dict]:
        """Load all FASTA files from input directory"""
        sequences = []
        
        if not self.input_dir.exists():
            logger.error(f"Input directory {self.input_dir} does not exist")
            return sequences
        
        fasta_files = list(self.input_dir.glob("*.fasta")) + list(self.input_dir.glob("*.fa"))
        
        if not fasta_files:
            logger.error(f"No FASTA files found in {self.input_dir}")
            return sequences
        
        logger.info(f"Found {len(fasta_files)} FASTA files")
        
        for fasta_file in fasta_files:
            logger.info(f"Processing {fasta_file.name}")
            try:
                for record in SeqIO.parse(fasta_file, "fasta"):
                    seq_info = {
                        'id': record.id,
                        'description': record.description,
                        'sequence': str(record.seq),
                        'length': len(record.seq),
                        'source_file': fasta_file.name,
                        'source': 'ncbi' if 'ncbi' in fasta_file.name.lower() else 'uniprot'
                    }
                    sequences.append(seq_info)
            except Exception as e:
                logger.error(f"Error processing {fasta_file}: {e}")
                continue
        
        self.stats['total_sequences'] = len(sequences)
        logger.info(f"Loaded {len(sequences)} sequences")
        return sequences
    
    def apply_quality_filters(self, sequences: List[Dict]) -> List[Dict]:
        """Apply quality filters to protein sequences"""
        filtered_sequences = []
        
        for seq in sequences:
            # Length filter
            if not (self.min_length <= seq['length'] <= self.max_length):
                continue
            
            # Check for valid amino acids
            valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
            if not all(aa in valid_aa for aa in seq['sequence']):
                continue
            
            # Check for stop codons (X) - these are incomplete sequences
            if 'X' in seq['sequence']:
                continue
            
            filtered_sequences.append(seq)
        
        self.stats['after_length_filter'] = len([s for s in sequences if self.min_length <= s['length'] <= self.max_length])
        self.stats['after_quality_filter'] = len(filtered_sequences)
        logger.info(f"After quality filters: {len(filtered_sequences)} sequences")
        return filtered_sequences
    
    def remove_duplicates(self, sequences: List[Dict]) -> List[Dict]:
        """Remove duplicate sequences using MD5 hashing"""
        seen_hashes = set()
        unique_sequences = []
        
        for seq in sequences:
            # Create MD5 hash of sequence
            seq_hash = hashlib.md5(seq['sequence'].encode()).hexdigest()
            
            if seq_hash not in seen_hashes:
                seen_hashes.add(seq_hash)
                seq['sequence_hash'] = seq_hash
                unique_sequences.append(seq)
        
        self.stats['after_duplicate_removal'] = len(unique_sequences)
        logger.info(f"After duplicate removal: {len(unique_sequences)} sequences")
        return unique_sequences
    
    def parse_product_information(self, sequences: List[Dict]) -> List[Dict]:
        """Parse terpene product information from descriptions"""
        for seq in sequences:
            description = seq['description'].lower()
            
            # Check for "putative" - exclude these
            if 'putative' in description:
                seq['is_putative'] = True
                seq['parsed_products'] = []
                continue
            
            seq['is_putative'] = False
            
            # Find terpene products in description
            found_products = []
            for product in self.terpene_products:
                if product in description:
                    found_products.append(product)
            
            seq['parsed_products'] = found_products
            
            # Classify sequence type
            if 'germacrene' in description or 'germacrene' in found_products:
                seq['sequence_type'] = 'germacrene'
                self.stats['germacrene_sequences'] += 1
            elif found_products:
                seq['sequence_type'] = 'other_terpene'
                self.stats['other_terpene_sequences'] += 1
            else:
                seq['sequence_type'] = 'unknown'
                self.stats['unknown_sequences'] += 1
        
        return sequences
    
    def filter_putative_sequences(self, sequences: List[Dict]) -> List[Dict]:
        """Remove sequences marked as putative"""
        non_putative = [seq for seq in sequences if not seq.get('is_putative', False)]
        
        self.stats['after_quality_filter'] = len(non_putative)
        logger.info(f"After removing putative sequences: {len(non_putative)} sequences")
        return non_putative
    
    def create_summary_statistics(self, sequences: List[Dict]) -> Dict:
        """Create summary statistics"""
        if not sequences:
            return {}
        
        # Basic statistics
        lengths = [seq['length'] for seq in sequences]
        gc_contents = []
        for seq in sequences:
            gc_content = (seq['sequence'].count('G') + seq['sequence'].count('C')) / len(seq['sequence'])
            gc_contents.append(gc_content)
        
        # Product statistics
        product_counts = Counter()
        for seq in sequences:
            for product in seq.get('parsed_products', []):
                product_counts[product] += 1
        
        # Source statistics
        source_counts = Counter(seq['source'] for seq in sequences)
        
        # Sequence type statistics
        type_counts = Counter(seq['sequence_type'] for seq in sequences)
        
        summary = {
            'total_sequences': len(sequences),
            'length_stats': {
                'mean': np.mean(lengths),
                'std': np.std(lengths),
                'min': np.min(lengths),
                'max': np.max(lengths),
                'median': np.median(lengths)
            },
            'gc_content_stats': {
                'mean': np.mean(gc_contents),
                'std': np.std(gc_contents),
                'min': np.min(gc_contents),
                'max': np.max(gc_contents),
                'median': np.median(gc_contents)
            },
            'product_counts': dict(product_counts.most_common(20)),
            'source_counts': dict(source_counts),
            'sequence_type_counts': dict(type_counts),
            'processing_stats': self.stats
        }
        
        return summary
    
    def save_processed_data(self, sequences: List[Dict], summary: Dict):
        """Save processed sequences and summary"""
        # Save sequences as CSV
        df = pd.DataFrame(sequences)
        csv_path = self.output_dir / "processed_sequences.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {len(sequences)} sequences to {csv_path}")
        
        # Save summary statistics
        summary_path = self.output_dir / "processing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Saved summary to {summary_path}")
        
        # Save sequences as FASTA
        fasta_path = self.output_dir / "processed_sequences.fasta"
        with open(fasta_path, 'w') as f:
            for seq in sequences:
                f.write(f">{seq['id']} {seq['description']}\n")
                f.write(f"{seq['sequence']}\n")
        logger.info(f"Saved FASTA to {fasta_path}")
        
        # Save germacrene sequences separately
        germacrene_seqs = [seq for seq in sequences if seq['sequence_type'] == 'germacrene']
        if germacrene_seqs:
            germacrene_path = self.output_dir / "germacrene_sequences.fasta"
            with open(germacrene_path, 'w') as f:
                for seq in germacrene_seqs:
                    f.write(f">{seq['id']} {seq['description']}\n")
                    f.write(f"{seq['sequence']}\n")
            logger.info(f"Saved {len(germacrene_seqs)} germacrene sequences to {germacrene_path}")
    
    def process_all(self) -> Tuple[List[Dict], Dict]:
        """Process all sequences through the pipeline"""
        logger.info("Starting sequence processing pipeline")
        
        # Load sequences
        sequences = self.load_fasta_files()
        if not sequences:
            return [], {}
        
        # Apply filters
        sequences = self.apply_quality_filters(sequences)
        sequences = self.remove_duplicates(sequences)
        sequences = self.parse_product_information(sequences)
        sequences = self.filter_putative_sequences(sequences)
        
        # Create summary
        summary = self.create_summary_statistics(sequences)
        
        # Save results
        self.save_processed_data(sequences, summary)
        
        logger.info("Processing pipeline completed")
        return sequences, summary

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Process downloaded terpene synthase sequences")
    parser.add_argument("--input-dir", default="data/phase2/downloaded", 
                       help="Directory containing downloaded FASTA files")
    parser.add_argument("--output-dir", default="data/phase2/processed",
                       help="Directory to save processed data")
    
    args = parser.parse_args()
    
    # Create processor
    processor = TerpeneSequenceProcessor(args.input_dir, args.output_dir)
    
    # Process sequences
    sequences, summary = processor.process_all()
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total sequences processed: {summary.get('total_sequences', 0)}")
    print(f"Final sequences after filtering: {len(sequences)}")
    print(f"Germacrene sequences: {summary.get('sequence_type_counts', {}).get('germacrene', 0)}")
    print(f"Other terpene sequences: {summary.get('sequence_type_counts', {}).get('other_terpene', 0)}")
    print(f"Unknown sequences: {summary.get('sequence_type_counts', {}).get('unknown', 0)}")
    
    if summary.get('product_counts'):
        print("\nTop 10 terpene products found:")
        for product, count in list(summary['product_counts'].items())[:10]:
            print(f"  {product}: {count}")
    
    print(f"\nProcessed data saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
