"""
UniProt Data Curator

This module handles curation of terpene synthase sequences from UniProt
with manual annotation and quality control.

Implements research best practices for protein sequence curation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import re
import logging
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
import requests
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UniProtRecord:
    """Structured record for UniProt terpene synthase data"""
    uniprot_id: str
    organism: str
    sequence: str
    product: str
    product_smiles: Optional[str]
    ec_number: str
    function_description: str
    confidence: float
    source: str
    reference: str

class UniProtCurator:
    """Curates terpene synthase data from UniProt"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # UniProt API base URL
        self.uniprot_base = "https://www.uniprot.org/uniprot"
        
        # Terpene synthase keywords for searching
        self.terpene_keywords = [
            "terpene synthase",
            "terpenoid synthase", 
            "monoterpene synthase",
            "sesquiterpene synthase",
            "diterpene synthase",
            "limonene synthase",
            "pinene synthase",
            "myrcene synthase",
            "linalool synthase",
            "germacrene synthase",
            "caryophyllene synthase",
            "humulene synthase",
            "farnesene synthase",
            "bisabolene synthase"
        ]
        
        # Product name mappings
        self.product_mappings = {
            "limonene": ["limonene", "d-limonene", "l-limonene"],
            "pinene": ["pinene", "α-pinene", "β-pinene", "alpha-pinene", "beta-pinene"],
            "myrcene": ["myrcene", "β-myrcene", "beta-myrcene"],
            "linalool": ["linalool", "l-linalool", "d-linalool"],
            "germacrene_a": ["germacrene a", "germacrene-a", "germacrene A"],
            "germacrene_d": ["germacrene d", "germacrene-d", "germacrene D"],
            "caryophyllene": ["caryophyllene", "β-caryophyllene", "beta-caryophyllene"],
            "humulene": ["humulene", "α-humulene", "alpha-humulene"],
            "farnesene": ["farnesene", "α-farnesene", "β-farnesene"],
            "bisabolene": ["bisabolene", "α-bisabolene", "β-bisabolene"],
        }

    def search_terpene_synthases(self, organism: Optional[str] = None) -> List[str]:
        """Search for terpene synthase entries in UniProt"""
        logger.info("Searching UniProt for terpene synthases...")
        
        uniprot_ids = []
        
        for keyword in self.terpene_keywords:
            try:
                ids = self._search_uniprot(keyword, organism)
                uniprot_ids.extend(ids)
                logger.info(f"Found {len(ids)} entries for '{keyword}'")
            except Exception as e:
                logger.error(f"Error searching for '{keyword}': {e}")
                continue
        
        # Remove duplicates
        uniprot_ids = list(set(uniprot_ids))
        logger.info(f"Total unique UniProt IDs found: {len(uniprot_ids)}")
        
        return uniprot_ids

    def _search_uniprot(self, keyword: str, organism: Optional[str] = None) -> List[str]:
        """Search UniProt for specific keyword"""
        query = f"keyword:{keyword}"
        if organism:
            query += f" AND organism:{organism}"
        
        params = {
            'query': query,
            'format': 'list',
            'limit': 1000
        }
        
        try:
            response = requests.get(self.uniprot_base, params=params)
            response.raise_for_status()
            
            # Parse response
            ids = [line.strip() for line in response.text.split('\n') if line.strip()]
            return ids
            
        except Exception as e:
            logger.error(f"Error in UniProt search: {e}")
            return []

    def fetch_uniprot_records(self, uniprot_ids: List[str]) -> List[UniProtRecord]:
        """Fetch detailed records for UniProt IDs"""
        logger.info(f"Fetching {len(uniprot_ids)} UniProt records...")
        
        records = []
        batch_size = 100
        
        for i in tqdm(range(0, len(uniprot_ids), batch_size), desc="Fetching records"):
            batch_ids = uniprot_ids[i:i+batch_size]
            batch_records = self._fetch_batch(batch_ids)
            records.extend(batch_records)
        
        logger.info(f"Fetched {len(records)} records")
        return records

    def _fetch_batch(self, uniprot_ids: List[str]) -> List[UniProtRecord]:
        """Fetch a batch of UniProt records"""
        records = []
        
        for uniprot_id in uniprot_ids:
            try:
                record = self._fetch_single_record(uniprot_id)
                if record:
                    records.append(record)
            except Exception as e:
                logger.error(f"Error fetching {uniprot_id}: {e}")
                continue
        
        return records

    def _fetch_single_record(self, uniprot_id: str) -> Optional[UniProtRecord]:
        """Fetch single UniProt record"""
        try:
            # Fetch FASTA
            fasta_url = f"{self.uniprot_base}/{uniprot_id}.fasta"
            fasta_response = requests.get(fasta_url)
            fasta_response.raise_for_status()
            
            # Parse FASTA
            fasta_record = SeqIO.read(fasta_response.text.split('\n'), "fasta")
            
            # Fetch detailed XML
            xml_url = f"{self.uniprot_base}/{uniprot_id}.xml"
            xml_response = requests.get(xml_url)
            xml_response.raise_for_status()
            
            # Parse XML for metadata
            metadata = self._parse_uniprot_xml(xml_response.text)
            
            # Create record
            record = UniProtRecord(
                uniprot_id=uniprot_id,
                organism=metadata.get('organism', 'Unknown'),
                sequence=str(fasta_record.seq),
                product=metadata.get('product', 'Unknown'),
                product_smiles=metadata.get('product_smiles'),
                ec_number=metadata.get('ec_number', ''),
                function_description=metadata.get('function', ''),
                confidence=metadata.get('confidence', 0.8),
                source='UniProt',
                reference=metadata.get('reference', '')
            )
            
            return record
            
        except Exception as e:
            logger.error(f"Error processing {uniprot_id}: {e}")
            return None

    def _parse_uniprot_xml(self, xml_content: str) -> Dict[str, str]:
        """Parse UniProt XML for metadata"""
        metadata = {}
        
        try:
            # Simple XML parsing (in production, use proper XML parser)
            # Extract organism
            org_match = re.search(r'<organism[^>]*>.*?<name[^>]*>([^<]+)</name>', xml_content, re.DOTALL)
            if org_match:
                metadata['organism'] = org_match.group(1)
            
            # Extract EC number
            ec_match = re.search(r'<dbReference type="EC" id="([^"]+)"', xml_content)
            if ec_match:
                metadata['ec_number'] = ec_match.group(1)
            
            # Extract function
            func_match = re.search(r'<comment type="function">.*?<text[^>]*>([^<]+)</text>', xml_content, re.DOTALL)
            if func_match:
                metadata['function'] = func_match.group(1)
            
            # Extract product information from function
            function_text = metadata.get('function', '').lower()
            for product, variants in self.product_mappings.items():
                if any(variant in function_text for variant in variants):
                    metadata['product'] = product
                    break
            
        except Exception as e:
            logger.error(f"Error parsing XML: {e}")
        
        return metadata

    def curate_records(self, records: List[UniProtRecord]) -> List[UniProtRecord]:
        """Curate and validate UniProt records"""
        logger.info("Starting UniProt record curation...")
        
        curated_records = []
        
        for record in tqdm(records, desc="Curating records"):
            # Validate sequence
            if not self._validate_sequence(record.sequence):
                logger.warning(f"Invalid sequence for {record.uniprot_id}")
                continue
            
            # Validate product
            if record.product == 'Unknown':
                logger.warning(f"Unknown product for {record.uniprot_id}")
                continue
            
            # Additional quality checks
            if len(record.sequence) < 200:
                logger.warning(f"Short sequence for {record.uniprot_id}")
                continue
            
            curated_records.append(record)
        
        logger.info(f"Curated {len(curated_records)} records from {len(records)} original records")
        return curated_records

    def _validate_sequence(self, sequence: str) -> bool:
        """Validate protein sequence"""
        if not sequence or len(sequence) < 50:
            return False
        
        valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
        if not all(aa in valid_amino_acids for aa in sequence.upper()):
            return False
        
        return True

    def save_records(self, records: List[UniProtRecord], filename: str = "uniprot_terpene_synthases.csv"):
        """Save records to CSV file"""
        data = []
        for record in records:
            data.append({
                'uniprot_id': record.uniprot_id,
                'organism': record.organism,
                'sequence': record.sequence,
                'product': record.product,
                'product_smiles': record.product_smiles,
                'ec_number': record.ec_number,
                'function_description': record.function_description,
                'confidence': record.confidence,
                'source': record.source,
                'reference': record.reference
            })
        
        df = pd.DataFrame(data)
        output_path = self.cache_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(records)} records to {output_path}")
        return output_path

def main():
    """Main function to curate UniProt data"""
    logger.info("Starting UniProt data curation...")
    
    # Initialize curator
    curator = UniProtCurator()
    
    # Search for terpene synthases
    uniprot_ids = curator.search_terpene_synthases()
    
    # Fetch records
    records = curator.fetch_uniprot_records(uniprot_ids)
    
    # Curate records
    curated_records = curator.curate_records(records)
    
    # Save results
    curator.save_records(curated_records)
    
    # Print summary
    print(f"\nUniProt Curation Summary:")
    print(f"Total IDs found: {len(uniprot_ids)}")
    print(f"Records fetched: {len(records)}")
    print(f"Curated records: {len(curated_records)}")
    print(f"Success rate: {len(curated_records)/len(records)*100:.1f}%")
    
    # Product distribution
    products = [r.product for r in curated_records]
    product_counts = pd.Series(products).value_counts()
    print(f"\nProduct distribution:")
    print(product_counts)

if __name__ == "__main__":
    main()
