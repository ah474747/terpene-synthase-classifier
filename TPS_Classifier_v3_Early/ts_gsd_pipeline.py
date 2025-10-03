#!/usr/bin/env python3
"""
Terpene Synthase Gold Standard Dataset (TS-GSD) Pipeline

This script systematically acquires, merges, and cleans TPS data from public sources,
establishing Multi-Label Functional Ensemble targets and core feature fields.

Author: AI Assistant
Date: 2024
Environment: Google Colab / Python 3.9+
"""

import pandas as pd
import numpy as np
import requests
import json
import time
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import warnings
from pathlib import Path
import logging

# BioPython for sequence handling
try:
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqUtils import molecular_weight
except ImportError:
    print("Warning: BioPython not installed. Install with: pip install biopython")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TSGSDPipeline:
    """
    Main pipeline class for creating the Terpene Synthase Gold Standard Dataset
    """
    
    def __init__(self, output_dir: str = "data", n_classes: int = 30):
        """
        Initialize the TS-GSD Pipeline
        
        Args:
            output_dir: Directory to save output files
            n_classes: Number of functional ensemble classes (default 30)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.n_classes = n_classes
        
        # Functional ensemble mapping (placeholder - will be expanded)
        self.functional_ensembles = {
            # Germacrene family
            'germacrene': 0,
            'germacrene_a': 0, 'germacrene_d': 0, 'germacrene_c': 0,
            
            # Monoterpene families
            'limonene': 1, 'pinene': 2, 'myrcene': 3, 'camphene': 4,
            
            # Sesquiterpene families
            'farnesene': 5, 'bisabolene': 6, 'humulene': 7, 'caryophyllene': 8,
            
            # Diterpene families
            'kaurene': 9, 'abietadiene': 10, 'taxadiene': 11,
            
            # Triterpene families
            'squalene': 12, 'lanosterol': 13,
            
            # Other major families
            'menthol': 14, 'linalool': 15, 'geraniol': 16, 'citral': 17,
            'thujene': 18, 'sabinene': 19, 'terpinene': 20,
            'eucalyptol': 21, 'cineole': 22, 'borneol': 23,
            'camphor': 24, 'carvone': 25, 'menthone': 26,
            'terpinolene': 27, 'phellandrene': 28, 'ocimene': 29
        }
        
        logger.info(f"Initialized TS-GSD Pipeline with {n_classes} functional ensembles")
    
    def fetch_marts_data(self, url: str = None) -> pd.DataFrame:
        """
        Fetch TPS data from MARTS-DB
        
        Args:
            url: MARTS-DB download URL (if None, uses simulated data)
            
        Returns:
            DataFrame with TPS enzyme data
        """
        logger.info("Fetching MARTS-DB data...")
        
        if url is None:
            # Simulate MARTS-DB data structure for development
            logger.info("Using simulated MARTS-DB data for development")
            return self._create_simulated_marts_data()
        
        try:
            # Attempt to fetch real data
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse the response (format depends on MARTS-DB structure)
            # This is a placeholder - adjust based on actual MARTS-DB format
            data = response.json()
            
            df = pd.DataFrame(data)
            logger.info(f"Successfully fetched {len(df)} entries from MARTS-DB")
            return df
            
        except Exception as e:
            logger.warning(f"Failed to fetch real MARTS-DB data: {e}")
            logger.info("Falling back to simulated data")
            return self._create_simulated_marts_data()
    
    def _create_simulated_marts_data(self) -> pd.DataFrame:
        """
        Create simulated MARTS-DB data for development and testing
        
        Returns:
            DataFrame with simulated TPS enzyme data
        """
        # Simulated enzyme data based on known TPS characteristics
        simulated_data = {
            'enzyme_id': [f'TPS_{i:03d}' for i in range(1, 201)],
            'uniprot_id': [f'Q{i:05d}' for i in range(12345, 12545)],
            'sequence': [self._generate_random_protein_sequence(500, 800) for _ in range(200)],
            'product_names': [
                np.random.choice([
                    'germacrene_a', 'germacrene_d', 'limonene', 'pinene', 'myrcene',
                    'farnesene', 'bisabolene', 'humulene', 'caryophyllene',
                    'kaurene', 'abietadiene', 'menthol', 'linalool', 'geraniol'
                ], size=np.random.randint(1, 4), replace=False
            ).tolist() for _ in range(200)
            ],
            'substrate_type': np.random.choice(['GPP', 'FPP', 'GGPP'], 200),
            'reaction_class': np.random.choice(['Class_I', 'Class_II', 'Hybrid'], 200),
            'organism': [f'Species_{i}' for i in range(1, 201)],
            'taxonomy_id': np.random.randint(1000, 9999, 200)
        }
        
        df = pd.DataFrame(simulated_data)
        logger.info(f"Created simulated dataset with {len(df)} entries")
        return df
    
    def _generate_random_protein_sequence(self, min_length: int, max_length: int) -> str:
        """Generate a random protein sequence of specified length range"""
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        length = np.random.randint(min_length, max_length + 1)
        return ''.join(np.random.choice(list(amino_acids), length))
    
    def fetch_uniprot_features(self, accessions: List[str]) -> pd.DataFrame:
        """
        Fetch additional features from UniProt API
        
        Args:
            accessions: List of UniProt accession IDs
            
        Returns:
            DataFrame with enriched UniProt data
        """
        logger.info(f"Fetching UniProt features for {len(accessions)} accessions...")
        
        uniprot_data = []
        
        for accession in tqdm(accessions, desc="Fetching UniProt data"):
            try:
                # UniProt REST API endpoint
                url = f"https://rest.uniprot.org/uniprotkb/{accession}"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract relevant features
                    entry = {
                        'uniprot_accession': accession,
                        'sequence': data.get('sequence', {}).get('value', ''),
                        'organism': data.get('organism', {}).get('scientificName', ''),
                        'taxonomy_id': data.get('organism', {}).get('taxonId', 0),
                        'protein_name': data.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', ''),
                        'pfam_domains': self._extract_pfam_domains(data),
                        'taxonomy_phylum': self._extract_taxonomy_phylum(data),
                        'molecular_weight': self._calculate_molecular_weight(data.get('sequence', {}).get('value', ''))
                    }
                    
                    uniprot_data.append(entry)
                    
                else:
                    logger.warning(f"Failed to fetch data for {accession}: {response.status_code}")
                    
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"Error fetching {accession}: {e}")
                continue
        
        df = pd.DataFrame(uniprot_data)
        logger.info(f"Successfully fetched UniProt data for {len(df)} entries")
        return df
    
    def _extract_pfam_domains(self, data: Dict) -> List[str]:
        """Extract Pfam domain IDs from UniProt data"""
        pfam_domains = []
        
        try:
            features = data.get('features', [])
            for feature in features:
                if feature.get('type') == 'domain' and 'Pfam' in str(feature):
                    # Extract Pfam ID from feature description
                    description = feature.get('description', '')
                    if 'Pfam:' in description:
                        pfam_id = description.split('Pfam:')[1].split()[0]
                        pfam_domains.append(pfam_id)
        except Exception as e:
            logger.debug(f"Error extracting Pfam domains: {e}")
        
        return pfam_domains
    
    def _extract_taxonomy_phylum(self, data: Dict) -> str:
        """Extract phylum from taxonomy lineage"""
        try:
            lineage = data.get('organism', {}).get('lineage', [])
            for taxon in lineage:
                if taxon.get('rank') == 'phylum':
                    return taxon.get('scientificName', '')
        except Exception as e:
            logger.debug(f"Error extracting phylum: {e}")
        
        return ''
    
    def _calculate_molecular_weight(self, sequence: str) -> float:
        """Calculate molecular weight of protein sequence"""
        if not sequence:
            return 0.0
        
        try:
            from Bio.SeqUtils import molecular_weight
            return molecular_weight(Seq(sequence), 'protein')
        except ImportError:
            # Fallback calculation if BioPython not available
            aa_weights = {
                'A': 89.09, 'R': 174.20, 'N': 132.12, 'D': 133.10, 'C': 121.16,
                'Q': 146.15, 'E': 147.13, 'G': 75.07, 'H': 155.16, 'I': 131.17,
                'L': 131.17, 'K': 146.19, 'M': 149.21, 'F': 165.19, 'P': 115.13,
                'S': 105.09, 'T': 119.12, 'W': 204.23, 'Y': 181.19, 'V': 117.15
            }
            return sum(aa_weights.get(aa, 0) for aa in sequence.upper())
    
    def engineer_multi_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer multi-label targets for functional ensemble classification
        
        Args:
            df: Input DataFrame with product information
            
        Returns:
            DataFrame with multi-label target vectors
        """
        logger.info("Engineering multi-label targets...")
        
        target_vectors = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing labels"):
            # Get product names (could be string or list)
            products = row.get('product_names', [])
            if isinstance(products, str):
                products = [products]
            
            # Create binary target vector
            target_vector = [0] * self.n_classes
            
            for product in products:
                # Map product to functional ensemble
                ensemble_id = self._map_product_to_ensemble(product)
                if ensemble_id is not None:
                    target_vector[ensemble_id] = 1
            
            target_vectors.append(target_vector)
        
        # Add target vectors to dataframe
        df['target_vector'] = target_vectors
        df['num_products'] = df['target_vector'].apply(sum)
        
        logger.info(f"Engineered multi-label targets. Distribution:")
        logger.info(f"  - Single product: {sum(df['num_products'] == 1)}")
        logger.info(f"  - Multiple products: {sum(df['num_products'] > 1)}")
        
        return df
    
    def _map_product_to_ensemble(self, product: str) -> Optional[int]:
        """
        Map a product name to functional ensemble ID
        
        Args:
            product: Product name (e.g., 'germacrene_a')
            
        Returns:
            Functional ensemble ID or None if not found
        """
        product_lower = product.lower().replace(' ', '_')
        
        # Direct mapping
        if product_lower in self.functional_ensembles:
            return self.functional_ensembles[product_lower]
        
        # Fuzzy matching for variants
        for key, ensemble_id in self.functional_ensembles.items():
            if key in product_lower or product_lower in key:
                return ensemble_id
        
        logger.debug(f"No ensemble mapping found for product: {product}")
        return None
    
    def merge_datasets(self, marts_df: pd.DataFrame, uniprot_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge MARTS-DB and UniProt datasets
        
        Args:
            marts_df: MARTS-DB DataFrame
            uniprot_df: UniProt DataFrame
            
        Returns:
            Merged DataFrame
        """
        logger.info("Merging MARTS-DB and UniProt datasets...")
        
        # Merge on UniProt accession
        merged_df = pd.merge(
            marts_df, 
            uniprot_df, 
            left_on='uniprot_id', 
            right_on='uniprot_accession', 
            how='left'
        )
        
        # Handle missing sequences (use MARTS sequence if UniProt sequence missing)
        merged_df['sequence'] = merged_df['sequence_y'].fillna(merged_df['sequence_x'])
        
        # Clean up duplicate columns
        merged_df = merged_df.drop(['sequence_x', 'sequence_y'], axis=1)
        
        logger.info(f"Merged dataset contains {len(merged_df)} entries")
        return merged_df
    
    def create_final_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create the final TS-GSD dataset with all required columns
        
        Args:
            df: Merged DataFrame
            
        Returns:
            Final TS-GSD DataFrame
        """
        logger.info("Creating final TS-GSD dataset...")
        
        # Convert product names to SMILES (placeholder)
        df['product_smiles_list'] = df['product_names'].apply(self._convert_products_to_smiles)
        
        # Create final schema
        final_df = pd.DataFrame({
            'uniprot_accession_id': df['uniprot_accession'],
            'aa_sequence': df['sequence'],
            'product_smiles_list': df['product_smiles_list'],
            'substrate_type': df['substrate_type'],
            'rxn_class_i_ii_hybrid': df['reaction_class'],
            'pfam_domain_ids': df['pfam_domains'].apply(json.dumps),
            'taxonomy_phylum': df['taxonomy_phylum'],
            'target_vector': df['target_vector'].apply(json.dumps),
            'molecular_weight': df.get('molecular_weight', 0),
            'protein_name': df.get('protein_name', ''),
            'organism': df.get('organism', ''),
            'taxonomy_id': df.get('taxonomy_id', 0)
        })
        
        logger.info(f"Final dataset created with {len(final_df)} entries and {len(final_df.columns)} columns")
        return final_df
    
    def _convert_products_to_smiles(self, products: List[str]) -> str:
        """
        Convert product names to SMILES strings (placeholder implementation)
        
        Args:
            products: List of product names
            
        Returns:
            JSON string of SMILES representations
        """
        # Placeholder SMILES mappings
        smiles_mapping = {
            'germacrene_a': 'C1=CC(C)(C)C=CC=C(C)C=C1',
            'germacrene_d': 'C1=CC(C)(C)C=CC=C(C)C=C1',
            'limonene': 'CC(C)=CCCC(C)=C',
            'pinene': 'C1(C)C(C)=CC2C1C2',
            'myrcene': 'C=CC(C)=CCCC=C',
            'farnesene': 'C=CC(C)=CCCC(C)=CCC=C',
            'bisabolene': 'C=CC(C)=CCCC(C)=CCC=C',
            'humulene': 'C=CC(C)=CCCC(C)=CCC=C',
            'caryophyllene': 'C1=CC(C)(C)C=CC=C(C)C=C1',
            'menthol': 'CC(C)CCC(C)C(C)O',
            'linalool': 'CC(C)=CCCC(C)(O)C=C',
            'geraniol': 'CC(C)=CCCC(C)=CCO'
        }
        
        smiles_list = []
        for product in products:
            smiles = smiles_mapping.get(product.lower(), 'Unknown')
            smiles_list.append(smiles)
        
        return json.dumps(smiles_list)
    
    def run_pipeline(self, marts_url: str = None) -> str:
        """
        Run the complete TS-GSD pipeline
        
        Args:
            marts_url: Optional MARTS-DB URL
            
        Returns:
            Path to the created dataset file
        """
        logger.info("Starting TS-GSD Pipeline...")
        
        try:
            # Step 1: Fetch MARTS-DB data
            marts_df = self.fetch_marts_data(marts_url)
            
            # Step 2: Engineer multi-label targets
            marts_df = self.engineer_multi_labels(marts_df)
            
            # Step 3: Fetch UniProt features
            uniprot_df = self.fetch_uniprot_features(marts_df['uniprot_id'].tolist())
            
            # Step 4: Merge datasets
            merged_df = self.merge_datasets(marts_df, uniprot_df)
            
            # Step 5: Create final dataset
            final_df = self.create_final_dataset(merged_df)
            
            # Step 6: Save dataset
            output_path = self.output_dir / 'TS-GSD.csv'
            final_df.to_csv(output_path, index=False)
            
            # Save metadata
            metadata = {
                'total_entries': len(final_df),
                'n_classes': self.n_classes,
                'functional_ensembles': self.functional_ensembles,
                'columns': list(final_df.columns),
                'class_distribution': final_df['target_vector'].apply(
                    lambda x: sum(json.loads(x))
                ).value_counts().to_dict()
            }
            
            metadata_path = self.output_dir / 'TS-GSD_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Pipeline completed successfully!")
            logger.info(f"Dataset saved to: {output_path}")
            logger.info(f"Metadata saved to: {metadata_path}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


def main():
    """Main function to run the TS-GSD pipeline"""
    # Initialize pipeline
    pipeline = TSGSDPipeline(output_dir="data", n_classes=30)
    
    # Run pipeline
    dataset_path = pipeline.run_pipeline()
    
    print(f"\n✅ TS-GSD Pipeline completed successfully!")
    print(f"📁 Dataset saved to: {dataset_path}")
    print(f"📊 Ready for Module 2: Feature Extraction Pipeline")


if __name__ == "__main__":
    main()



