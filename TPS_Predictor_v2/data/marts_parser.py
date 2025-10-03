"""
MARTS-DB Data Parser

This module handles parsing and integration of MARTS-DB reaction data
into the terpene synthase prediction pipeline.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from pathlib import Path
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MARTSRecord:
    """Structured record for MARTS-DB data"""
    enzyme_id: str
    uniprot_id: str
    enzyme_name: str
    sequence: str
    organism: str
    kingdom: str
    enzyme_type: str
    enzyme_class: str
    substrate_name: str
    substrate_smiles: str
    substrate_chebi_id: str
    product_name: str
    product_smiles: str
    product_chebi_id: str
    confidence: float
    source: str
    reference: str

class MARTSDBParser:
    """Parser for MARTS-DB reaction data"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Terpene product mappings for standardization
        self.terpene_product_mappings = {
            # Monoterpenes
            'limonene': ['limonene', '(4s)-limonene', '(4r)-limonene'],
            'pinene': ['α-pinene', 'alpha-pinene', '(-)-α-pinene', '(+)-α-pinene', 
                      'β-pinene', 'beta-pinene', '(-)-β-pinene', '(+)-β-pinene'],
            'myrcene': ['β-myrcene', 'beta-myrcene', 'myrcene'],
            'linalool': ['linalool', '(s)-linalool', '(r)-linalool'],
            'terpinene': ['α-terpinene', 'β-terpinene', 'γ-terpinene', 'terpinolene'],
            'phellandrene': ['α-phellandrene', 'β-phellandrene'],
            'thujene': ['α-thujene', 'β-thujene'],
            'sabinene': ['sabinene'],
            
            # Sesquiterpenes
            'caryophyllene': ['(−)-β-caryophyllene', 'β-caryophyllene', 'caryophyllene'],
            'humulene': ['(1e,4e,8e)-α-humulene', 'α-humulene', 'humulene'],
            'germacrene_a': ['(−)-germacrene a', 'germacrene a', 'germacrene-a'],
            'germacrene_d': ['(-)-germacrene d', 'germacrene d', 'germacrene-d'],
            'farnesene': ['trans-β-farnesene', 'β-farnesene', 'farnesene'],
            'bisabolene': ['β-bisabolene', '(s)-β-bisabolene', 'bisabolene'],
            
            # Diterpenes
            'copalyl_diphosphate': ['ent-copalyl diphosphate', 'copalyl diphosphate', 
                                   '(+)-copalyl diphosphate'],
            'miltiradiene': ['miltiradiene'],
            'manoyl_oxide': ['(13r)-manoyl oxide', 'manoyl oxide'],
        }
        
        # Reverse mapping for lookup
        self.product_to_standard = {}
        for standard_name, variants in self.terpene_product_mappings.items():
            for variant in variants:
                self.product_to_standard[variant.lower()] = standard_name
    
    def load_marts_data(self, csv_file: str) -> pd.DataFrame:
        """Load MARTS-DB CSV data"""
        logger.info(f"Loading MARTS-DB data from {csv_file}")
        
        df = pd.read_csv(csv_file)
        logger.info(f"Loaded {len(df)} records from MARTS-DB")
        
        return df
    
    def filter_terpene_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter for terpene synthase data"""
        logger.info("Filtering for terpene synthase data...")
        
        # Filter for terpene-related products
        terpene_keywords = [
            'pinene', 'limonene', 'myrcene', 'linalool', 'germacrene', 
            'caryophyllene', 'humulene', 'farnesene', 'bisabolene',
            'terpinene', 'phellandrene', 'thujene', 'sabinene',
            'copalyl', 'miltiradiene', 'manoyl'
        ]
        
        # Create boolean mask for terpene products
        terpene_mask = df['Product_name'].str.contains(
            '|'.join(terpene_keywords), 
            case=False, 
            na=False
        )
        
        terpene_df = df[terpene_mask].copy()
        logger.info(f"Found {len(terpene_df)} terpene synthase records")
        
        return terpene_df
    
    def standardize_product_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize product names for consistent classification"""
        logger.info("Standardizing product names...")
        
        def map_to_standard(product_name):
            """Map product name to standard name"""
            if pd.isna(product_name):
                return 'unknown'
            
            product_lower = product_name.lower()
            
            # Direct mapping
            if product_lower in self.product_to_standard:
                return self.product_to_standard[product_lower]
            
            # Fuzzy matching for partial matches
            for variant, standard in self.product_to_standard.items():
                if variant in product_lower or product_lower in variant:
                    return standard
            
            return 'other'
        
        df['standard_product'] = df['Product_name'].apply(map_to_standard)
        
        # Log product distribution
        product_counts = df['standard_product'].value_counts()
        logger.info("Standardized product distribution:")
        for product, count in product_counts.head(10).items():
            logger.info(f"  {product}: {count}")
        
        return df
    
    def validate_sequences(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate amino acid sequences"""
        logger.info("Validating amino acid sequences...")
        
        # Valid amino acids
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        
        def is_valid_sequence(seq):
            if pd.isna(seq) or not isinstance(seq, str):
                return False
            
            # Check length
            if len(seq) < 50:
                return False
            
            # Check for valid amino acids
            seq_upper = seq.upper()
            if not all(aa in valid_aa for aa in seq_upper):
                return False
            
            return True
        
        df['valid_sequence'] = df['Aminoacid_sequence'].apply(is_valid_sequence)
        
        valid_df = df[df['valid_sequence']].copy()
        invalid_count = len(df) - len(valid_df)
        
        logger.info(f"Valid sequences: {len(valid_df)}")
        logger.info(f"Invalid sequences: {invalid_count}")
        
        return valid_df
    
    def create_balanced_dataset(self, df: pd.DataFrame, min_samples_per_class: int = 10) -> pd.DataFrame:
        """Create balanced dataset for training"""
        logger.info(f"Creating balanced dataset (min {min_samples_per_class} samples per class)...")
        
        # Filter classes with sufficient samples
        class_counts = df['standard_product'].value_counts()
        valid_classes = class_counts[class_counts >= min_samples_per_class].index
        
        logger.info(f"Valid classes (≥{min_samples_per_class} samples): {len(valid_classes)}")
        for class_name in valid_classes:
            logger.info(f"  {class_name}: {class_counts[class_name]} samples")
        
        # Filter for valid classes
        balanced_df = df[df['standard_product'].isin(valid_classes)].copy()
        
        # Balance classes by sampling
        balanced_samples = []
        for class_name in valid_classes:
            class_data = balanced_df[balanced_df['standard_product'] == class_name]
            
            # Sample up to max_samples_per_class or use all available
            max_samples = min(len(class_data), 100)  # Cap at 100 samples per class
            sampled_data = class_data.sample(n=max_samples, random_state=42)
            balanced_samples.append(sampled_data)
        
        final_df = pd.concat(balanced_samples, ignore_index=True)
        
        logger.info(f"Final balanced dataset: {len(final_df)} samples")
        logger.info("Final class distribution:")
        final_counts = final_df['standard_product'].value_counts()
        for product, count in final_counts.items():
            logger.info(f"  {product}: {count}")
        
        return final_df
    
    def convert_to_records(self, df: pd.DataFrame) -> List[MARTSRecord]:
        """Convert DataFrame to MARTSRecord objects"""
        logger.info("Converting to MARTSRecord objects...")
        
        records = []
        for _, row in df.iterrows():
            record = MARTSRecord(
                enzyme_id=row['Enzyme_marts_ID'],
                uniprot_id=row['Uniprot_ID'] if pd.notna(row['Uniprot_ID']) else 'unknown',
                enzyme_name=row['Enzyme_name'],
                sequence=row['Aminoacid_sequence'],
                organism=row['Species'],
                kingdom=row['Kingdom'],
                enzyme_type=row['Type'],
                enzyme_class=str(row['Class']),
                substrate_name=row['Substrate_name'],
                substrate_smiles=row['Substrate_smiles'],
                substrate_chebi_id=row['Substrate_chebi_ID'],
                product_name=row['Product_name'],
                product_smiles=row['Product_smiles'],
                product_chebi_id=row['Product_chebi_ID'],
                confidence=0.9,  # High confidence for MARTS-DB data
                source='MARTS-DB',
                reference=row['Publication']
            )
            records.append(record)
        
        logger.info(f"Created {len(records)} MARTSRecord objects")
        return records
    
    def parse_marts_data(self, csv_file: str) -> List[MARTSRecord]:
        """Main parsing pipeline"""
        logger.info("Starting MARTS-DB data parsing pipeline...")
        
        # Load data
        df = self.load_marts_data(csv_file)
        
        # Filter for terpene data
        terpene_df = self.filter_terpene_data(df)
        
        # Standardize product names
        terpene_df = self.standardize_product_names(terpene_df)
        
        # Validate sequences
        valid_df = self.validate_sequences(terpene_df)
        
        # Create balanced dataset
        balanced_df = self.create_balanced_dataset(valid_df)
        
        # Convert to records
        records = self.convert_to_records(balanced_df)
        
        logger.info("MARTS-DB parsing pipeline completed!")
        return records

def main():
    """Test the MARTS-DB parser"""
    parser = MARTSDBParser()
    records = parser.parse_marts_data('reactions.csv')
    
    print(f"\nParsed {len(records)} terpene synthase records")
    print("\nSample records:")
    for i, record in enumerate(records[:3]):
        print(f"\nRecord {i+1}:")
        print(f"  Enzyme: {record.enzyme_name}")
        print(f"  Organism: {record.organism}")
        print(f"  Product: {record.product_name}")
        print(f"  Sequence length: {len(record.sequence)}")

if __name__ == "__main__":
    main()
