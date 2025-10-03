#!/usr/bin/env python3
"""
Large-scale NCBI sequence downloader using EPost + EFetch approach
Designed for downloading ~60,000 terpene synthase sequences efficiently
"""

import os
import time
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NCBILargeScaleDownloader:
    """Download large datasets from NCBI using EPost + EFetch approach"""
    
    def __init__(self, email: str, api_key: str, tool_name: str = "terpene_classifier"):
        """
        Initialize the downloader
        
        Args:
            email: Email for NCBI API (required)
            api_key: NCBI API key for higher rate limits
            tool_name: Tool name for NCBI API identification
        """
        self.email = email
        self.api_key = api_key
        self.tool_name = tool_name
        
        # Rate limiting: 3 requests/sec without API key, 10 with API key
        self.requests_per_second = 10 if api_key else 3
        self.delay_between_requests = 1.0 / self.requests_per_second
        
        # Base URLs for E-utilities
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        
        logger.info(f"Initialized NCBI downloader with {self.requests_per_second} req/sec limit")
    
    def search_and_get_ids(self, search_term: str, max_results: int = 100000) -> List[str]:
        """
        Search NCBI and get protein IDs using ESearch
        
        Args:
            search_term: Search term for NCBI
            max_results: Maximum number of results to retrieve
            
        Returns:
            List of protein IDs
        """
        logger.info(f"Searching NCBI for: '{search_term}' (max {max_results} results)")
        
        # ESearch parameters
        params = {
            'db': 'protein',
            'term': search_term,
            'retmax': min(max_results, 100000),  # NCBI limit
            'retmode': 'xml',
            'email': self.email,
            'tool': self.tool_name
        }
        
        if self.api_key:
            params['api_key'] = self.api_key
        
        try:
            # Make ESearch request
            response = requests.get(f"{self.base_url}/esearch.fcgi", params=params)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.text)
            
            # Extract IDs
            id_elements = root.findall('.//Id')
            ids = [id_elem.text for id_elem in id_elements]
            
            total_found = int(root.find('.//Count').text)
            logger.info(f"Found {total_found} total results, retrieved {len(ids)} IDs")
            
            return ids
            
        except Exception as e:
            logger.error(f"Error searching NCBI: {e}")
            return []
    
    def epost_upload_ids(self, ids: List[str]) -> Optional[Tuple[str, str]]:
        """
        Upload protein IDs to NCBI History Server using EPost
        
        Args:
            ids: List of protein IDs to upload
            
        Returns:
            Tuple of (QueryKey, WebEnv) if successful, None otherwise
        """
        logger.info(f"Uploading {len(ids)} IDs to NCBI History Server")
        
        # EPost parameters
        params = {
            'db': 'protein',
            'email': self.email,
            'tool': self.tool_name
        }
        
        if self.api_key:
            params['api_key'] = self.api_key
        
        # Prepare ID list (comma-separated)
        id_string = ','.join(ids)
        
        try:
            # Make EPost request
            response = requests.post(
                f"{self.base_url}/epost.fcgi",
                params=params,
                data={'id': id_string}
            )
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.text)
            
            # Extract QueryKey and WebEnv
            query_key = root.find('.//QueryKey')
            web_env = root.find('.//WebEnv')
            
            if query_key is not None and web_env is not None:
                logger.info(f"Successfully uploaded IDs. QueryKey: {query_key.text}, WebEnv: {web_env.text[:20]}...")
                return query_key.text, web_env.text
            else:
                logger.error("Failed to get QueryKey and WebEnv from EPost response")
                return None
                
        except Exception as e:
            logger.error(f"Error uploading IDs to History Server: {e}")
            return None
    
    def efetch_batch_download(self, query_key: str, web_env: str, 
                            total_records: int, batch_size: int = 5000,
                            output_file: str = "ncbi_sequences.fasta") -> bool:
        """
        Download sequences in batches using EFetch
        
        Args:
            query_key: QueryKey from EPost
            web_env: WebEnv from EPost
            total_records: Total number of records to download
            batch_size: Number of records per batch
            output_file: Output FASTA file path
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Starting batch download of {total_records} sequences in batches of {batch_size}")
        
        # EFetch parameters
        params = {
            'db': 'protein',
            'query_key': query_key,
            'WebEnv': web_env,
            'rettype': 'fasta',
            'retmode': 'text',
            'email': self.email,
            'tool': self.tool_name
        }
        
        if self.api_key:
            params['api_key'] = self.api_key
        
        try:
            with open(output_file, 'w') as out_handle:
                for start_index in range(0, total_records, batch_size):
                    current_batch_size = min(batch_size, total_records - start_index)
                    
                    # Add batch parameters
                    batch_params = params.copy()
                    batch_params['retstart'] = start_index
                    batch_params['retmax'] = current_batch_size
                    
                    logger.info(f"Downloading batch {start_index//batch_size + 1}: "
                              f"records {start_index + 1} to {start_index + current_batch_size}")
                    
                    try:
                        # Make EFetch request
                        response = requests.get(f"{self.base_url}/efetch.fcgi", params=batch_params)
                        response.raise_for_status()
                        
                        # Write FASTA data to file
                        out_handle.write(response.text)
                        out_handle.flush()  # Ensure data is written
                        
                        logger.info(f"✓ Downloaded {current_batch_size} sequences")
                        
                        # Rate limiting delay
                        time.sleep(self.delay_between_requests)
                        
                    except Exception as e:
                        logger.error(f"Error downloading batch {start_index//batch_size + 1}: {e}")
                        time.sleep(10)  # Wait longer on error
                        continue
            
            logger.info(f"Download complete! Sequences saved to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error during batch download: {e}")
            return False
    
    def download_terpene_synthase_sequences(self, output_dir: str = "data/phase2/downloaded") -> bool:
        """
        Download terpene synthase sequences using the complete EPost + EFetch pipeline
        
        Args:
            output_dir: Directory to save downloaded sequences
            
        Returns:
            True if successful, False otherwise
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Search terms for terpene synthases
        search_terms = [
            '"terpene synthase"[Title] OR "terpene synthase"[Abstract]',
            '"terpenoid synthase"[Title] OR "terpenoid synthase"[Abstract]',
            '"isoprenoid synthase"[Title] OR "isoprenoid synthase"[Abstract]',
            '"sesquiterpene synthase"[Title] OR "sesquiterpene synthase"[Abstract]',
            '"monoterpene synthase"[Title] OR "monoterpene synthase"[Abstract]',
            '"diterpene synthase"[Title] OR "diterpene synthase"[Abstract]',
            '"triterpene synthase"[Title] OR "triterpene synthase"[Abstract]'
        ]
        
        all_ids = set()  # Use set to avoid duplicates
        
        # Step 1: Search and collect all IDs
        logger.info("=== Step 1: Searching and collecting protein IDs ===")
        for term in search_terms:
            ids = self.search_and_get_ids(term, max_results=20000)
            all_ids.update(ids)
            logger.info(f"Total unique IDs collected so far: {len(all_ids)}")
            time.sleep(self.delay_between_requests)
        
        if not all_ids:
            logger.error("No protein IDs found!")
            return False
        
        logger.info(f"=== Step 2: Downloading {len(all_ids)} sequences ===")
        
        # Convert set back to list
        id_list = list(all_ids)
        
        # Step 2: Upload IDs to History Server
        epost_result = self.epost_upload_ids(id_list)
        if not epost_result:
            logger.error("Failed to upload IDs to History Server")
            return False
        
        query_key, web_env = epost_result
        time.sleep(self.delay_between_requests)
        
        # Step 3: Batch download sequences
        output_file = output_path / "ncbi_terpene_synthase_sequences.fasta"
        success = self.efetch_batch_download(
            query_key, web_env, 
            total_records=len(id_list),
            batch_size=5000,
            output_file=str(output_file)
        )
        
        if success:
            logger.info(f"✓ Successfully downloaded {len(id_list)} terpene synthase sequences")
            logger.info(f"✓ Sequences saved to: {output_file}")
            return True
        else:
            logger.error("Failed to download sequences")
            return False

def main():
    """Main function to run the large-scale downloader"""
    # Use the credentials from the previous script
    email = "andrew.horwitz@gmail.com"
    api_key = "5c12c88e90ef2a5f15fdec7688d3ef30ad09"
    tool_name = "terpene_classifier_phase2"
    output_dir = "data/phase2/downloaded"
    
    print(f"Starting large-scale NCBI download with:")
    print(f"  Email: {email}")
    print(f"  API Key: {api_key[:20]}...")
    print(f"  Tool: {tool_name}")
    print(f"  Output: {output_dir}")
    print()
    
    # Initialize downloader
    downloader = NCBILargeScaleDownloader(
        email=email,
        api_key=api_key,
        tool_name=tool_name
    )
    
    # Download sequences
    success = downloader.download_terpene_synthase_sequences(output_dir)
    
    if success:
        print("✓ Download completed successfully!")
    else:
        print("✗ Download failed!")
        exit(1)

if __name__ == "__main__":
    main()
