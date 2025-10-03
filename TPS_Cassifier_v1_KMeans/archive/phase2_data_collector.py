#!/usr/bin/env python3
"""
Phase 2: Large-Scale Terpene Synthase Data Collection
===================================================

This script implements a comprehensive data collection and processing pipeline for:
1. Downloading terpene synthase sequences from NCBI and UniProt
2. Deduplication and overlap removal
3. Product information parsing
4. Quality filtering (removing putative sequences)
5. Preparation for semi-supervised learning

Author: AI Assistant
Date: 2024
"""

import os
import sys
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import re
import json
from datetime import datetime
import warnings
from Bio import SeqIO, Entrez
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import hashlib

warnings.filterwarnings('ignore')


class TerpeneSynthaseDataCollector:
    """
    Comprehensive data collector for terpene synthase sequences
    """
    
    def __init__(self, email: str = "your_email@example.com", api_key: str = None, tool_name: str = "terpene_classifier"):
        """
        Initialize the data collector
        
        Args:
            email: Email for NCBI API (required for Entrez)
            api_key: NCBI API key for higher rate limits (optional)
            tool_name: Tool name for NCBI API identification
        """
        self.email = email
        self.api_key = api_key
        self.tool_name = tool_name
        
        # Rate limiting: 3 requests/sec without API key, 10 with API key
        self.requests_per_second = 10 if api_key else 3
        self.request_delay = 1.0 / self.requests_per_second
        # Set up Entrez with proper parameters
        Entrez.email = email
        Entrez.tool = tool_name
        if api_key:
            Entrez.api_key = api_key
        
        # Search terms for different databases
        self.search_terms = {
            'ncbi': [
                'terpene synthase',
                'terpene synthetase', 
                'terpenoid synthase',
                'terpenoid synthetase',
                'isoprenoid synthase',
                'monoterpene synthase',
                'sesquiterpene synthase',
                'diterpene synthase',
                'triterpene synthase'
            ],
            'uniprot': [
                'terpene synthase',
                'terpene synthetase',
                'terpenoid synthase', 
                'terpenoid synthetase',
                'isoprenoid synthase'
            ]
        }
        
        # Terpene product keywords for parsing
        self.terpene_products = {
            'germacrene': ['germacrene', 'germacrenol', 'germacradienol'],
            'limonene': ['limonene', 'limonol'],
            'pinene': ['pinene', 'pinenol'],
            'myrcene': ['myrcene', 'myrcenol'],
            'linalool': ['linalool', 'linalyl'],
            'menthol': ['menthol', 'menthone', 'menthene'],
            'camphor': ['camphor', 'campholenol'],
            'sabinene': ['sabinene', 'sabinol'],
            'carene': ['carene', 'carenol'],
            'terpinene': ['terpinene', 'terpineol'],
            'phellandrene': ['phellandrene', 'phellandrol'],
            'ocimene': ['ocimene', 'ocimenol'],
            'farnesene': ['farnesene', 'farnesol'],
            'humulene': ['humulene', 'humulol'],
            'caryophyllene': ['caryophyllene', 'caryophyllol'],
            'bisabolene': ['bisabolene', 'bisabolol'],
            'selinene': ['selinene', 'selinol'],
            'eudesmol': ['eudesmol', 'eudesmene'],
            'cadinene': ['cadinene', 'cadinol'],
            'muurolene': ['muurolene', 'muurolol'],
            'amorphene': ['amorphene', 'amorphenol'],
            'valencene': ['valencene', 'valencenol'],
            'nootkatone': ['nootkatone', 'nootkatol'],
            'santalene': ['santalene', 'santalol'],
            'bergamotene': ['bergamotene', 'bergamotol'],
            'elemene': ['elemene', 'elemol'],
            'guaiene': ['guaiene', 'guaienol'],
            'patchoulene': ['patchoulene', 'patchoulol'],
            'vetispiradiene': ['vetispiradiene', 'vetispiradienol'],
            'prezizaene': ['prezizaene', 'prezizaenol'],
            'zingiberene': ['zingiberene', 'zingiberenol'],
            'curcumene': ['curcumene', 'curcumenol'],
            'bisabolol': ['bisabolol', 'bisabolene'],
            'nerolidol': ['nerolidol', 'nerolidene'],
            'farnesol': ['farnesol', 'farnesene'],
            'geraniol': ['geraniol', 'geranial'],
            'citronellol': ['citronellol', 'citronellal'],
            'linalool': ['linalool', 'linalyl'],
            'menthol': ['menthol', 'menthone'],
            'thujone': ['thujone', 'thujol'],
            'carvone': ['carvone', 'carveol'],
            'pulegone': ['pulegone', 'pulegol'],
            'menthone': ['menthone', 'menthol'],
            'isomenthone': ['isomenthone', 'isomenthol'],
            'piperitone': ['piperitone', 'piperitol'],
            'thymol': ['thymol', 'thymene'],
            'carvacrol': ['carvacrol', 'carvacrene'],
            'eugenol': ['eugenol', 'eugenyl'],
            'safrole': ['safrole', 'safrenol'],
            'myristicin': ['myristicin', 'myristicene'],
            'elemicin': ['elemicin', 'elemicene'],
            'asarone': ['asarone', 'asarol'],
            'apiole': ['apiole', 'apiolene'],
            'dillapiole': ['dillapiole', 'dillapiolene'],
            'estragole': ['estragole', 'estragolol'],
            'anethole': ['anethole', 'anetholol'],
            'isoanethole': ['isoanethole', 'isoanetholol'],
            'propenyl': ['propenyl', 'propenol'],
            'allyl': ['allyl', 'allylol'],
            'benzyl': ['benzyl', 'benzylol'],
            'phenethyl': ['phenethyl', 'phenethylol'],
            'cinnamyl': ['cinnamyl', 'cinnamylol'],
            'vanillyl': ['vanillyl', 'vanillylol'],
            'guaiacyl': ['guaiacyl', 'guaiacylol'],
            'syringyl': ['syringyl', 'syringylol'],
            'coniferyl': ['coniferyl', 'coniferyol'],
            'sinapyl': ['sinapyl', 'sinapylol'],
            'coumaryl': ['coumaryl', 'coumarylol']
        }
        
        # Quality filters
        self.quality_filters = {
            'min_length': 50,  # Minimum sequence length
            'max_length': 2000,  # Maximum sequence length
            'exclude_putative': True,  # Exclude sequences with "putative" in description
            'exclude_fragment': True,  # Exclude sequences marked as fragments
            'exclude_partial': True,  # Exclude sequences marked as partial
        }
        
        # Output directories
        self.output_dir = Path("data/phase2")
        self.output_dir.mkdir(exist_ok=True)
        
        # Cache for deduplication
        self.sequence_cache = {}
        self.description_cache = {}
        
    def search_ncbi_sequences(self, search_term: str, max_results: int = 10000) -> List[Dict]:
        """
        Search NCBI for terpene synthase sequences with proper rate limiting
        
        Args:
            search_term: Search term for NCBI
            max_results: Maximum number of results to retrieve
            
        Returns:
            List of sequence information dictionaries
        """
        print(f"Searching NCBI for: '{search_term}'")
        
        try:
            # First, get the total count
            count_handle = Entrez.esearch(
                db="protein",
                term=search_term,
                retmax=0,  # Just get count
                retmode="xml"
            )
            count_results = Entrez.read(count_handle)
            count_handle.close()
            
            total_count = int(count_results['Count'])
            print(f"  Total results available: {total_count}")
            
            if total_count == 0:
                print(f"  No results found for '{search_term}'")
                return []
            
            # Respect rate limiting
            time.sleep(self.request_delay)
            
            # Search for protein sequences with proper query syntax
            search_handle = Entrez.esearch(
                db="protein",
                term=f'"{search_term}"[Title] OR "{search_term}"[Abstract]',
                retmax=min(max_results, total_count),
                retmode="xml"
            )
            search_results = Entrez.read(search_handle)
            search_handle.close()
            
            if not search_results['IdList']:
                print(f"  No results found for '{search_term}'")
                return []
            
            print(f"  Found {len(search_results['IdList'])} results")
            
            # Fetch sequence details in batches
            sequences = []
            batch_size = 100  # Larger batch size for efficiency
            
            for i in range(0, len(search_results['IdList']), batch_size):
                batch_ids = search_results['IdList'][i:i+batch_size]
                
                try:
                    # Respect rate limiting
                    time.sleep(self.request_delay)
                    
                    fetch_handle = Entrez.efetch(
                        db="protein",
                        id=batch_ids,
                        rettype="fasta",
                        retmode="text"
                    )
                    
                    fasta_data = fetch_handle.read()
                    fetch_handle.close()
                    
                    # Parse FASTA data
                    for record in SeqIO.parse(fasta_data, "fasta"):
                        seq_info = {
                            'id': record.id,
                            'description': record.description,
                            'sequence': str(record.seq),
                            'length': len(record.seq),
                            'source': 'ncbi',
                            'search_term': search_term
                        }
                        sequences.append(seq_info)
                    
                    print(f"    Processed batch {i//batch_size + 1}/{(len(search_results['IdList']) + batch_size - 1)//batch_size}")
                    
                except Exception as e:
                    print(f"    Error processing batch {i//batch_size + 1}: {e}")
                    continue
            
            print(f"  Successfully retrieved {len(sequences)} sequences")
            return sequences
            
        except Exception as e:
            print(f"  Error searching NCBI: {e}")
            return []
    
    def search_uniprot_sequences(self, search_term: str, max_results: int = 10000) -> List[Dict]:
        """
        Search UniProt for terpene synthase sequences
        
        Args:
            search_term: Search term for UniProt
            max_results: Maximum number of results to retrieve
            
        Returns:
            List of sequence information dictionaries
        """
        print(f"Searching UniProt for: '{search_term}'")
        
        try:
            # UniProt REST API endpoint
            base_url = "https://rest.uniprot.org/uniprotkb/search"
            
            # Search parameters - simplified query
            params = {
                'query': search_term,
                'format': 'fasta',
                'size': min(max_results, 10000),  # UniProt limit
                'compressed': 'false'
            }
            
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse FASTA data
            sequences = []
            fasta_data = response.text
            
            for record in SeqIO.parse(fasta_data, "fasta"):
                seq_info = {
                    'id': record.id,
                    'description': record.description,
                    'sequence': str(record.seq),
                    'length': len(record.seq),
                    'source': 'uniprot',
                    'search_term': search_term
                }
                sequences.append(seq_info)
            
            print(f"  Successfully retrieved {len(sequences)} sequences")
            return sequences
            
        except Exception as e:
            print(f"  Error searching UniProt: {e}")
            return []
    
    def collect_all_sequences(self, max_results_per_term: int = 5000) -> List[Dict]:
        """
        Collect sequences from all sources and search terms
        
        Args:
            max_results_per_term: Maximum results per search term
            
        Returns:
            List of all collected sequences
        """
        print("=" * 60)
        print("PHASE 2: LARGE-SCALE SEQUENCE COLLECTION")
        print("=" * 60)
        
        all_sequences = []
        
        # Collect from NCBI
        print(f"\n=== Collecting from NCBI ===")
        for term in self.search_terms['ncbi']:
            sequences = self.search_ncbi_sequences(term, max_results_per_term)
            all_sequences.extend(sequences)
            time.sleep(1)  # Be nice to NCBI
        
        # Collect from UniProt
        print(f"\n=== Collecting from UniProt ===")
        for term in self.search_terms['uniprot']:
            sequences = self.search_uniprot_sequences(term, max_results_per_term)
            all_sequences.extend(sequences)
            time.sleep(1)  # Be nice to UniProt
        
        print(f"\n=== Collection Summary ===")
        print(f"Total sequences collected: {len(all_sequences)}")
        
        # Save raw data
        raw_data_path = self.output_dir / "raw_sequences.json"
        with open(raw_data_path, 'w') as f:
            json.dump(all_sequences, f, indent=2)
        
        print(f"Raw data saved to: {raw_data_path}")
        
        return all_sequences
    
    def parse_terpene_products(self, description: str) -> Dict[str, List[str]]:
        """
        Parse terpene products from sequence description
        
        Args:
            description: Sequence description text
            
        Returns:
            Dictionary of detected terpene products
        """
        description_lower = description.lower()
        detected_products = {}
        
        for product_category, keywords in self.terpene_products.items():
            found_keywords = []
            for keyword in keywords:
                if keyword.lower() in description_lower:
                    found_keywords.append(keyword)
            
            if found_keywords:
                detected_products[product_category] = found_keywords
        
        return detected_products
    
    def is_quality_sequence(self, seq_info: Dict) -> Tuple[bool, str]:
        """
        Check if sequence meets quality criteria
        
        Args:
            seq_info: Sequence information dictionary
            
        Returns:
            Tuple of (is_quality, reason)
        """
        sequence = seq_info['sequence']
        description = seq_info['description'].lower()
        
        # Check length
        if len(sequence) < self.quality_filters['min_length']:
            return False, f"Too short ({len(sequence)} < {self.quality_filters['min_length']})"
        
        if len(sequence) > self.quality_filters['max_length']:
            return False, f"Too long ({len(sequence)} > {self.quality_filters['max_length']})"
        
        # Check for invalid characters
        valid_chars = set('ACDEFGHIKLMNPQRSTVWY')
        if not all(c in valid_chars for c in sequence):
            return False, "Contains invalid amino acid characters"
        
        # Check for putative sequences
        if self.quality_filters['exclude_putative'] and 'putative' in description:
            return False, "Contains 'putative' in description"
        
        # Check for fragments
        if self.quality_filters['exclude_fragment'] and 'fragment' in description:
            return False, "Contains 'fragment' in description"
        
        # Check for partial sequences
        if self.quality_filters['exclude_partial'] and 'partial' in description:
            return False, "Contains 'partial' in description"
        
        return True, "Passed all quality checks"
    
    def deduplicate_sequences(self, sequences: List[Dict]) -> List[Dict]:
        """
        Remove duplicate sequences and merge information
        
        Args:
            sequences: List of sequence information dictionaries
            
        Returns:
            Deduplicated list of sequences
        """
        print(f"\n=== Deduplication ===")
        print(f"Input sequences: {len(sequences)}")
        
        # Group by sequence hash
        sequence_groups = defaultdict(list)
        
        for seq_info in sequences:
            # Create hash of sequence
            seq_hash = hashlib.md5(seq_info['sequence'].encode()).hexdigest()
            sequence_groups[seq_hash].append(seq_info)
        
        # Merge groups
        deduplicated = []
        for seq_hash, group in sequence_groups.items():
            if len(group) == 1:
                # Single sequence, no merging needed
                deduplicated.append(group[0])
            else:
                # Multiple sequences with same sequence, merge information
                merged = self._merge_sequence_group(group)
                deduplicated.append(merged)
        
        print(f"After deduplication: {len(deduplicated)} sequences")
        print(f"Removed {len(sequences) - len(deduplicated)} duplicates")
        
        return deduplicated
    
    def _merge_sequence_group(self, group: List[Dict]) -> Dict:
        """
        Merge information from sequences with identical sequences
        
        Args:
            group: List of sequences with identical sequences
            
        Returns:
            Merged sequence information
        """
        # Use the first sequence as base
        merged = group[0].copy()
        
        # Merge descriptions
        descriptions = [seq['description'] for seq in group]
        merged['description'] = ' | '.join(descriptions)
        
        # Merge sources
        sources = list(set([seq['source'] for seq in group]))
        merged['source'] = '|'.join(sources)
        
        # Merge search terms
        search_terms = list(set([seq['search_term'] for seq in group]))
        merged['search_term'] = '|'.join(search_terms)
        
        # Add merge information
        merged['merge_count'] = len(group)
        merged['original_ids'] = [seq['id'] for seq in group]
        
        return merged
    
    def process_sequences(self, sequences: List[Dict]) -> pd.DataFrame:
        """
        Process and clean collected sequences
        
        Args:
            sequences: List of raw sequence information
            
        Returns:
            Processed DataFrame
        """
        print(f"\n=== Processing Sequences ===")
        print(f"Input sequences: {len(sequences)}")
        
        processed_data = []
        quality_stats = {
            'total': len(sequences),
            'passed': 0,
            'failed': 0,
            'reasons': defaultdict(int)
        }
        
        for i, seq_info in enumerate(sequences):
            if i % 1000 == 0:
                print(f"  Processing {i}/{len(sequences)}...")
            
            # Check quality
            is_quality, reason = self.is_quality_sequence(seq_info)
            
            if not is_quality:
                quality_stats['failed'] += 1
                quality_stats['reasons'][reason] += 1
                continue
            
            quality_stats['passed'] += 1
            
            # Parse terpene products
            products = self.parse_terpene_products(seq_info['description'])
            
            # Create processed record
            processed_record = {
                'id': seq_info['id'],
                'description': seq_info['description'],
                'sequence': seq_info['sequence'],
                'length': seq_info['length'],
                'source': seq_info['source'],
                'search_term': seq_info['search_term'],
                'terpene_products': json.dumps(products),
                'product_count': len(products),
                'is_germacrene': 'germacrene' in products,
                'merge_count': seq_info.get('merge_count', 1),
                'original_ids': json.dumps(seq_info.get('original_ids', [seq_info['id']]))
            }
            
            processed_data.append(processed_record)
        
        # Create DataFrame
        df = pd.DataFrame(processed_data)
        
        # Print quality statistics
        print(f"\n=== Quality Statistics ===")
        print(f"Total sequences: {quality_stats['total']}")
        print(f"Passed quality checks: {quality_stats['passed']}")
        print(f"Failed quality checks: {quality_stats['failed']}")
        print(f"Success rate: {quality_stats['passed']/quality_stats['total']*100:.1f}%")
        
        print(f"\nFailure reasons:")
        for reason, count in quality_stats['reasons'].items():
            print(f"  {reason}: {count}")
        
        return df
    
    def save_processed_data(self, df: pd.DataFrame) -> None:
        """
        Save processed data to files
        
        Args:
            df: Processed DataFrame
        """
        print(f"\n=== Saving Processed Data ===")
        
        # Save as CSV
        csv_path = self.output_dir / "processed_terpene_sequences.csv"
        df.to_csv(csv_path, index=False)
        print(f"CSV saved to: {csv_path}")
        
        # Save as FASTA
        fasta_path = self.output_dir / "processed_terpene_sequences.fasta"
        with open(fasta_path, 'w') as f:
            for _, row in df.iterrows():
                f.write(f">{row['id']} {row['description']}\n")
                f.write(f"{row['sequence']}\n")
        print(f"FASTA saved to: {fasta_path}")
        
        # Save summary statistics
        summary_path = self.output_dir / "processing_summary.json"
        summary = {
            'total_sequences': len(df),
            'sources': df['source'].value_counts().to_dict(),
            'germacrene_sequences': int(df['is_germacrene'].sum()),
            'germacrene_percentage': float(df['is_germacrene'].mean() * 100),
            'length_stats': {
                'mean': float(df['length'].mean()),
                'std': float(df['length'].std()),
                'min': int(df['length'].min()),
                'max': int(df['length'].max())
            },
            'product_distribution': df['product_count'].value_counts().to_dict(),
            'processing_timestamp': datetime.now().isoformat()
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_path}")
        
        # Print final statistics
        print(f"\n=== Final Statistics ===")
        print(f"Total processed sequences: {len(df)}")
        print(f"Germacrene sequences: {summary['germacrene_sequences']} ({summary['germacrene_percentage']:.1f}%)")
        print(f"Average sequence length: {summary['length_stats']['mean']:.1f} ± {summary['length_stats']['std']:.1f}")
        print(f"Length range: {summary['length_stats']['min']} - {summary['length_stats']['max']}")
        
        print(f"\nSource distribution:")
        for source, count in summary['sources'].items():
            print(f"  {source}: {count}")
        
        print(f"\nProduct distribution:")
        for count, freq in summary['product_distribution'].items():
            print(f"  {count} products: {freq} sequences")


def main():
    """Main execution function"""
    print("Phase 2: Large-Scale Terpene Synthase Data Collection")
    print("=" * 60)
    
    # Initialize collector
    # Note: Get an API key from https://www.ncbi.nlm.nih.gov/account/settings/ for higher rate limits
    collector = TerpeneSynthaseDataCollector(
        email="andrew.horwitz@gmail.com",  # Your email
        api_key="5c12c88e90ef2a5f15fdec7688d3ef30ad09",  # Your API key for 10 req/sec
        tool_name="terpene_classifier_phase2"
    )
    
    # Collect sequences
    print("Starting sequence collection...")
    raw_sequences = collector.collect_all_sequences(max_results_per_term=10000)  # Higher limit with API key
    
    if not raw_sequences:
        print("No sequences collected. Exiting.")
        return
    
    # Deduplicate
    deduplicated = collector.deduplicate_sequences(raw_sequences)
    
    # Process and clean
    processed_df = collector.process_sequences(deduplicated)
    
    # Save results
    collector.save_processed_data(processed_df)
    
    print(f"\n✓ Phase 2 data collection completed successfully!")
    print(f"✓ Ready for semi-supervised learning with {len(processed_df)} sequences")


if __name__ == "__main__":
    main()
