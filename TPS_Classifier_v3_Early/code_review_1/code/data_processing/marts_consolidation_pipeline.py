#!/usr/bin/env python3
"""
MARTS-DB Consolidation Pipeline for TS-GSD

This script transforms the raw MARTS-DB data into a consolidated, multi-label,
feature-ready format for the Terpene Synthase Gold Standard Dataset (TS-GSD).

Based on the real MARTS-DB data structure with 1,273 unique enzymes.
"""

import pandas as pd
import numpy as np
import json
import requests
import time
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MARTSDBConsolidator:
    """
    Consolidates MARTS-DB data into enzyme-centric format with multi-label targets
    """
    
    def __init__(self, marts_file: str = "marts_db.csv", n_classes: int = 30):
        """
        Initialize the MARTS-DB consolidator
        
        Args:
            marts_file: Path to MARTS-DB CSV file
            n_classes: Number of functional ensemble classes
        """
        self.marts_file = marts_file
        self.n_classes = n_classes
        
        # Define functional ensemble mapping based on terpene type and product families
        self.functional_ensembles = {
            # Monoterpenes (0-9)
            'mono_pinene': 0,           # α-pinene, β-pinene, (-)-α-pinene
            'mono_limonene': 1,         # limonene, (-)-limonene
            'mono_myrcene': 2,          # myrcene, β-myrcene
            'mono_terpinene': 3,        # α-terpinene, β-terpinene, γ-terpinene
            'mono_phellandrene': 4,     # α-phellandrene, β-phellandrene
            'mono_sabinene': 5,         # sabinene, trans-sabinene
            'mono_thujene': 6,          # α-thujene, β-thujene
            'mono_terpinolene': 7,      # terpinolene
            'mono_camphene': 8,         # camphene
            'mono_bornane': 9,          # borneol, camphor, bornyl acetate
            
            # Sesquiterpenes (10-19)
            'sesq_germacrane': 10,      # germacrene A, D, etc.
            'sesq_caryophyllane': 11,   # caryophyllene, β-caryophyllene
            'sesq_humulane': 12,        # humulene, α-humulene
            'sesq_bisabolane': 13,      # bisabolene, α-bisabolene, β-bisabolene
            'sesq_farnesane': 14,       # farnesene, α-farnesene, β-farnesene
            'sesq_selinane': 15,        # selinene, α-selinene, β-selinene
            'sesq_elemane': 16,         # elemene, α-elemene, β-elemene
            'sesq_guaiacane': 17,       # guaiol, guaiene
            'sesq_cadinane': 18,        # cadinene, α-cadinene, γ-cadinene
            'sesq_eudesmane': 19,       # eudesmol, α-eudesmol, β-eudesmol
            
            # Diterpenes (20-24)
            'di_kaurane': 20,           # kaurene, ent-kaurene
            'di_abietane': 21,          # abietadiene, abietic acid
            'di_taxane': 22,            # taxadiene, taxol precursors
            'di_copalane': 23,          # copalyl diphosphate derivatives
            'di_phytane': 24,           # phytol derivatives
            
            # Triterpenes (25-27)
            'tri_squalene': 25,         # squalene
            'tri_lanostane': 26,        # lanosterol, cycloartenol
            'tri_oleanane': 27,         # oleanolic acid, β-amyrin
            
            # Specialized (28-29)
            'specialized_oxygenated': 28,  # oxygenated derivatives
            'specialized_cyclic': 29       # complex cyclic structures
        }
        
        # Product name to ensemble mapping (expanded for real data)
        self.product_to_ensemble = self._build_product_mapping()
        
        logger.info(f"Initialized MARTS-DB Consolidator with {n_classes} functional ensembles")
        logger.info(f"Loaded {len(self.functional_ensembles)} ensemble mappings")
    
    def _build_product_mapping(self) -> Dict[str, int]:
        """
        Build comprehensive mapping from product names to functional ensembles
        
        Returns:
            Dictionary mapping product names to ensemble IDs
        """
        mapping = {}
        
        # Monoterpene mappings
        mapping.update({
            # Pinene family
            'α-pinene': 0, 'β-pinene': 0, '(-)-α-pinene': 0, '(+)-α-pinene': 0,
            
            # Limonene family
            'limonene': 1, '(-)-limonene': 1, '(+)-limonene': 1, 'd-limonene': 1,
            
            # Myrcene family
            'myrcene': 2, 'β-myrcene': 2,
            
            # Terpinene family
            'α-terpinene': 3, 'β-terpinene': 3, 'γ-terpinene': 3,
            
            # Phellandrene family
            'α-phellandrene': 4, 'β-phellandrene': 4,
            
            # Sabinene family
            'sabinene': 5, 'trans-sabinene': 5,
            
            # Thujene family
            'α-thujene': 6, 'β-thujene': 6,
            
            # Others
            'terpinolene': 7, 'camphene': 8,
            'borneol': 9, 'camphor': 9, 'bornyl acetate': 9,
        })
        
        # Sesquiterpene mappings
        mapping.update({
            # Germacrane family
            'germacrene a': 10, 'germacrene d': 10, 'germacrene c': 10,
            'α-germacrene': 10, 'β-germacrene': 10,
            
            # Caryophyllane family
            'caryophyllene': 11, 'β-caryophyllene': 11, 'α-caryophyllene': 11,
            
            # Humulane family
            'humulene': 12, 'α-humulene': 12, 'β-humulene': 12,
            
            # Bisabolane family
            'bisabolene': 13, 'α-bisabolene': 13, 'β-bisabolene': 13,
            
            # Farnesane family
            'farnesene': 14, 'α-farnesene': 14, 'β-farnesene': 14,
            
            # Selinane family
            'selinene': 15, 'α-selinene': 15, 'β-selinene': 15,
            
            # Elemane family
            'elemene': 16, 'α-elemene': 16, 'β-elemene': 16,
            
            # Guaiacane family
            'guaiol': 17, 'guaiene': 17,
            
            # Cadinane family
            'cadinene': 18, 'α-cadinene': 18, 'γ-cadinene': 18,
            
            # Eudesmane family
            'eudesmol': 19, 'α-eudesmol': 19, 'β-eudesmol': 19,
        })
        
        # Diterpene mappings
        mapping.update({
            # Kaurane family
            'kaurene': 20, 'ent-kaurene': 20,
            
            # Abietane family
            'abietadiene': 21, 'abietic acid': 21,
            
            # Taxane family
            'taxadiene': 22,
            
            # Copalane family
            'copalyl diphosphate': 23,
        })
        
        # Triterpene mappings
        mapping.update({
            'squalene': 25, 'lanosterol': 26, 'cycloartenol': 26,
            'oleanolic acid': 27, 'β-amyrin': 27,
        })
        
        return mapping
    
    def consolidate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Consolidate MARTS-DB data to one row per unique enzyme
        
        Args:
            df: Raw MARTS-DB DataFrame
            
        Returns:
            Consolidated DataFrame with one row per UniProt ID
        """
        logger.info("Consolidating MARTS-DB data to enzyme-centric format...")
        
        # Group by UniProt_ID
        grouped = df.groupby('Uniprot_ID').agg({
            # Keep first occurrence of these fields
            'Enzyme_marts_ID': 'first',
            'Enzyme_name': 'first',
            'Aminoacid_sequence': 'first',
            'Species': 'first',
            'Kingdom': 'first',
            'Type': 'first',
            'Class': 'first',
            'Substrate_name': 'first',
            'Substrate_smiles': 'first',
            'Genbank_ID': 'first',
            
            # Aggregate these into lists
            'Product_name': list,
            'Product_smiles': list,
            'Product_chebi_ID': list,
            'Product_marts_ID': list,
            'Mechanism_marts_ID': list,
            'Publication': list
        }).reset_index()
        
        # Clean up the aggregated data
        consolidated_df = pd.DataFrame({
            'uniprot_accession_id': grouped['Uniprot_ID'],
            'enzyme_marts_id': grouped['Enzyme_marts_ID'],
            'enzyme_name': grouped['Enzyme_name'],
            'aa_sequence': grouped['Aminoacid_sequence'],
            'species': grouped['Species'],
            'kingdom': grouped['Kingdom'],
            'terpene_type': grouped['Type'],  # mono, sesq, di, tri, etc.
            'enzyme_class': grouped['Class'],  # 1, 2
            'substrate_name': grouped['Substrate_name'],
            'substrate_smiles': grouped['Substrate_smiles'],
            'genbank_id': grouped['Genbank_ID'],
            'product_names': grouped['Product_name'],
            'product_smiles': grouped['Product_smiles'],
            'product_chebi_ids': grouped['Product_chebi_ID'],
            'product_marts_ids': grouped['Product_marts_ID'],
            'mechanism_ids': grouped['Mechanism_marts_ID'],
            'publications': grouped['Publication']
        })
        
        # Add product count
        consolidated_df['num_products'] = consolidated_df['product_names'].apply(len)
        
        logger.info(f"Consolidated {len(df)} rows into {len(consolidated_df)} unique enzymes")
        logger.info(f"Product distribution: {consolidated_df['num_products'].value_counts().to_dict()}")
        
        return consolidated_df
    
    def generate_target_vectors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate multi-label target vectors based on product clustering
        
        Args:
            df: Consolidated DataFrame
            
        Returns:
            DataFrame with target vectors added
        """
        logger.info("Generating multi-label target vectors...")
        
        target_vectors = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing target vectors"):
            products = row['product_names']
            
            # Create binary target vector
            target_vector = [0] * self.n_classes
            
            # Map each product to functional ensemble
            for product in products:
                ensemble_id = self._map_product_to_ensemble(product)
                if ensemble_id is not None:
                    target_vector[ensemble_id] = 1
            
            target_vectors.append(target_vector)
        
        # Add target vectors to dataframe
        df['target_vector'] = target_vectors
        df['active_ensembles'] = df['target_vector'].apply(sum)
        
        logger.info(f"Generated target vectors. Active ensemble distribution:")
        logger.info(f"  - Single ensemble: {sum(df['active_ensembles'] == 1)}")
        logger.info(f"  - Multiple ensembles: {sum(df['active_ensembles'] > 1)}")
        
        return df
    
    def _map_product_to_ensemble(self, product: str) -> Optional[int]:
        """
        Map a product name to functional ensemble ID
        
        Args:
            product: Product name
            
        Returns:
            Functional ensemble ID or None if not found
        """
        product_lower = product.lower().strip()
        
        # Direct mapping
        if product_lower in self.product_to_ensemble:
            return self.product_to_ensemble[product_lower]
        
        # Fuzzy matching for variants
        for key, ensemble_id in self.product_to_ensemble.items():
            if key in product_lower or product_lower in key:
                return ensemble_id
        
        # Log unmapped products for analysis
        logger.debug(f"Unmapped product: {product}")
        return None
    
    def fetch_domain_ids(self, uniprot_ids: List[str]) -> pd.DataFrame:
        """
        Fetch Pfam/InterPro domain IDs from UniProt API
        
        Args:
            uniprot_ids: List of UniProt accession IDs
            
        Returns:
            DataFrame with domain information
        """
        logger.info(f"Fetching domain information for {len(uniprot_ids)} enzymes...")
        
        domain_data = []
        
        for uniprot_id in tqdm(uniprot_ids, desc="Fetching domains"):
            try:
                # UniProt REST API endpoint
                url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract domain information
                    pfam_domains = []
                    interpro_domains = []
                    
                    features = data.get('features', [])
                    for feature in features:
                        if feature.get('type') == 'domain':
                            # Extract Pfam domains
                            if 'Pfam' in str(feature):
                                description = feature.get('description', '')
                                if 'Pfam:' in description:
                                    pfam_id = description.split('Pfam:')[1].split()[0]
                                    pfam_domains.append(pfam_id)
                            
                            # Extract InterPro domains
                            if 'InterPro:' in str(feature):
                                description = feature.get('description', '')
                                if 'InterPro:' in description:
                                    interpro_id = description.split('InterPro:')[1].split()[0]
                                    interpro_domains.append(interpro_id)
                    
                    domain_data.append({
                        'uniprot_accession_id': uniprot_id,
                        'pfam_domains': pfam_domains,
                        'interpro_domains': interpro_domains,
                        'total_domains': len(pfam_domains) + len(interpro_domains)
                    })
                    
                else:
                    logger.warning(f"Failed to fetch {uniprot_id}: {response.status_code}")
                    domain_data.append({
                        'uniprot_accession_id': uniprot_id,
                        'pfam_domains': [],
                        'interpro_domains': [],
                        'total_domains': 0
                    })
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"Error fetching {uniprot_id}: {e}")
                domain_data.append({
                    'uniprot_accession_id': uniprot_id,
                    'pfam_domains': [],
                    'interpro_domains': [],
                    'total_domains': 0
                })
        
        df = pd.DataFrame(domain_data)
        logger.info(f"Successfully fetched domain data for {len(df)} enzymes")
        return df
    
    def create_final_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create the final TS-GSD consolidated dataset
        
        Args:
            df: DataFrame with all features
            
        Returns:
            Final TS-GSD DataFrame
        """
        logger.info("Creating final TS-GSD consolidated dataset...")
        
        # Create final schema
        final_df = pd.DataFrame({
            'uniprot_accession_id': df['uniprot_accession_id'],
            'enzyme_marts_id': df['enzyme_marts_id'],
            'enzyme_name': df['enzyme_name'],
            'aa_sequence': df['aa_sequence'],
            'species': df['species'],
            'kingdom': df['kingdom'],
            'terpene_type': df['terpene_type'],
            'enzyme_class': df['enzyme_class'],
            'substrate_name': df['substrate_name'],
            'substrate_smiles': df['substrate_smiles'],
            'product_names': df['product_names'].apply(json.dumps),
            'product_smiles': df['product_smiles'].apply(json.dumps),
            'num_products': df['num_products'],
            'target_vector': df['target_vector'].apply(json.dumps),
            'active_ensembles': df['active_ensembles'],
            'pfam_domains': df.get('pfam_domains', []).apply(json.dumps),
            'interpro_domains': df.get('interpro_domains', []).apply(json.dumps),
            'total_domains': df.get('total_domains', 0)
        })
        
        logger.info(f"Final dataset created with {len(final_df)} entries and {len(final_df.columns)} columns")
        return final_df
    
    def run_consolidation(self, fetch_domains: bool = True) -> str:
        """
        Run the complete consolidation pipeline
        
        Args:
            fetch_domains: Whether to fetch domain information from UniProt
            
        Returns:
            Path to the created consolidated dataset
        """
        logger.info("Starting MARTS-DB consolidation pipeline...")
        
        try:
            # Step 1: Load raw MARTS-DB data
            logger.info(f"Loading MARTS-DB data from {self.marts_file}...")
            raw_df = pd.read_csv(self.marts_file)
            logger.info(f"Loaded {len(raw_df)} rows from MARTS-DB")
            
            # Step 2: Consolidate to enzyme-centric format
            consolidated_df = self.consolidate_data(raw_df)
            
            # Step 3: Generate multi-label target vectors
            consolidated_df = self.generate_target_vectors(consolidated_df)
            
            # Step 4: Fetch domain information (optional)
            if fetch_domains:
                uniprot_ids = consolidated_df['uniprot_accession_id'].tolist()
                domain_df = self.fetch_domain_ids(uniprot_ids)
                
                # Merge domain data
                consolidated_df = pd.merge(
                    consolidated_df, 
                    domain_df, 
                    on='uniprot_accession_id', 
                    how='left'
                )
            else:
                # Add placeholder domain columns
                consolidated_df['pfam_domains'] = [[]] * len(consolidated_df)
                consolidated_df['interpro_domains'] = [[]] * len(consolidated_df)
                consolidated_df['total_domains'] = 0
            
            # Step 5: Create final dataset
            final_df = self.create_final_dataset(consolidated_df)
            
            # Step 6: Save consolidated dataset
            output_path = "TS-GSD_consolidated.csv"
            final_df.to_csv(output_path, index=False)
            
            # Save metadata
            metadata = {
                'total_enzymes': len(final_df),
                'n_classes': self.n_classes,
                'functional_ensembles': self.functional_ensembles,
                'product_to_ensemble_mapping': self.product_to_ensemble,
                'columns': list(final_df.columns),
                'terpene_type_distribution': consolidated_df['terpene_type'].value_counts().to_dict(),
                'enzyme_class_distribution': consolidated_df['enzyme_class'].value_counts().to_dict(),
                'ensemble_distribution': consolidated_df['active_ensembles'].value_counts().to_dict()
            }
            
            metadata_path = "TS-GSD_consolidated_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Consolidation completed successfully!")
            logger.info(f"Dataset saved to: {output_path}")
            logger.info(f"Metadata saved to: {metadata_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Consolidation failed: {e}")
            raise


def main():
    """Main function to run the MARTS-DB consolidation pipeline"""
    # Initialize consolidator
    consolidator = MARTSDBConsolidator(n_classes=30)
    
    # Run consolidation (set fetch_domains=False for faster testing)
    dataset_path = consolidator.run_consolidation(fetch_domains=False)
    
    print(f"\n✅ MARTS-DB Consolidation completed successfully!")
    print(f"📁 Consolidated dataset saved to: {dataset_path}")
    print(f"🎯 Ready for Module 2: ESM2 Feature Extraction")


if __name__ == "__main__":
    main()
