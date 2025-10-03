#!/usr/bin/env python3
"""
Module 4.5: Bulk Structural Data Acquisition

This script automatically downloads AlphaFold predicted protein structures for all
1,273 unique UniProt IDs in the TS-GSD and performs quality control filtering
based on pLDDT confidence scores.

Features:
1. Bulk UniProt ID retrieval from TS-GSD
2. High-throughput AlphaFold structure download
3. Structural quality control with pLDDT filtering
4. Comprehensive manifest generation and reporting
"""

import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import shutil
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import json
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
ALPHAFOLD_BASE_URL = "https://alphafold.ebi.ac.uk/files/"
DOWNLOAD_TIMEOUT = 30
MAX_RETRIES = 3
BATCH_SIZE = 100
HIGH_CONFIDENCE_THRESHOLD = 70.0
MAX_WORKERS = 10  # For concurrent downloads


class AlphaFoldBulkDownloader:
    """
    High-throughput AlphaFold structure downloader with quality control
    """
    
    def __init__(self, output_dir: str = "alphafold_structures"):
        """
        Initialize the bulk downloader
        
        Args:
            output_dir: Directory to store downloaded structures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.mmcif_dir = self.output_dir / "mmcif"
        self.pdb_dir = self.output_dir / "pdb"
        self.mmcif_dir.mkdir(exist_ok=True)
        self.pdb_dir.mkdir(exist_ok=True)
        
        # Statistics tracking
        self.download_stats = {
            'total_requested': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'high_confidence_structures': 0,
            'low_confidence_structures': 0,
            'missing_structures': 0
        }
        
        logger.info(f"AlphaFold Bulk Downloader initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"High confidence threshold: {HIGH_CONFIDENCE_THRESHOLD}")
    
    def load_uniprot_ids(self, csv_path: str) -> List[str]:
        """
        Load unique UniProt IDs from TS-GSD consolidated file
        
        Args:
            csv_path: Path to TS-GSD_consolidated.csv
            
        Returns:
            List of unique UniProt IDs
        """
        logger.info(f"Loading UniProt IDs from {csv_path}")
        
        if not Path(csv_path).exists():
            logger.error(f"TS-GSD file not found: {csv_path}")
            raise FileNotFoundError(f"TS-GSD file not found: {csv_path}")
        
        # Load the consolidated dataset
        df = pd.read_csv(csv_path)
        
        # Extract unique UniProt IDs
        if 'uniprot_accession_id' in df.columns:
            uniprot_ids = df['uniprot_accession_id'].unique().tolist()
        elif 'Uniprot_ID' in df.columns:
            uniprot_ids = df['Uniprot_ID'].unique().tolist()
        else:
            logger.error("No UniProt ID column found in the dataset")
            raise ValueError("No UniProt ID column found")
        
        # Filter out any NaN values
        uniprot_ids = [uid for uid in uniprot_ids if pd.notna(uid) and uid.strip()]
        
        logger.info(f"Loaded {len(uniprot_ids)} unique UniProt IDs")
        logger.info(f"Sample IDs: {uniprot_ids[:5]}")
        
        self.download_stats['total_requested'] = len(uniprot_ids)
        
        return uniprot_ids
    
    def generate_alphafold_urls(self, uniprot_ids: List[str]) -> List[Dict[str, str]]:
        """
        Generate AlphaFold download URLs for UniProt IDs
        
        Args:
            uniprot_ids: List of UniProt accession IDs
            
        Returns:
            List of dictionaries with URL and metadata
        """
        logger.info("Generating AlphaFold download URLs...")
        
        urls = []
        for uniprot_id in uniprot_ids:
            # AlphaFold URL format: AF-{uniprot_id}-F1-model_v4.cif.gz
            mmcif_url = f"{ALPHAFOLD_BASE_URL}AF-{uniprot_id}-F1-model_v4.cif.gz"
            pdb_url = f"{ALPHAFOLD_BASE_URL}AF-{uniprot_id}-F1-model_v4.pdb"
            
            urls.append({
                'uniprot_id': uniprot_id,
                'mmcif_url': mmcif_url,
                'pdb_url': pdb_url,
                'mmcif_path': self.mmcif_dir / f"AF-{uniprot_id}-F1-model_v4.cif.gz",
                'pdb_path': self.pdb_dir / f"AF-{uniprot_id}-F1-model_v4.pdb"
            })
        
        logger.info(f"Generated {len(urls)} download URLs")
        return urls
    
    def download_single_structure(self, url_info: Dict[str, str]) -> Dict[str, any]:
        """
        Download a single AlphaFold structure
        
        Args:
            url_info: Dictionary containing URLs and paths
            
        Returns:
            Download result dictionary
        """
        uniprot_id = url_info['uniprot_id']
        result = {
            'uniprot_id': uniprot_id,
            'success': False,
            'format': None,
            'file_path': None,
            'error': None
        }
        
        # Try mmCIF format first (preferred)
        try:
            response = requests.get(url_info['mmcif_url'], timeout=DOWNLOAD_TIMEOUT)
            if response.status_code == 200:
                # Save compressed file
                with open(url_info['mmcif_path'], 'wb') as f:
                    f.write(response.content)
                
                # Extract the gzipped file
                extracted_path = url_info['mmcif_path'].with_suffix('')  # Remove .gz
                with gzip.open(url_info['mmcif_path'], 'rb') as f_in:
                    with open(extracted_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                # Remove compressed file
                url_info['mmcif_path'].unlink()
                
                result.update({
                    'success': True,
                    'format': 'mmcif',
                    'file_path': str(extracted_path)
                })
                
                logger.debug(f"Downloaded mmCIF: {uniprot_id}")
                return result
                
        except Exception as e:
            logger.debug(f"mmCIF download failed for {uniprot_id}: {e}")
        
        # Try PDB format as fallback
        try:
            response = requests.get(url_info['pdb_url'], timeout=DOWNLOAD_TIMEOUT)
            if response.status_code == 200:
                with open(url_info['pdb_path'], 'wb') as f:
                    f.write(response.content)
                
                result.update({
                    'success': True,
                    'format': 'pdb',
                    'file_path': str(url_info['pdb_path'])
                })
                
                logger.debug(f"Downloaded PDB: {uniprot_id}")
                return result
                
        except Exception as e:
            logger.debug(f"PDB download failed for {uniprot_id}: {e}")
        
        # Both formats failed
        result['error'] = f"Both mmCIF and PDB downloads failed"
        return result
    
    def bulk_download_structures(self, url_list: List[Dict[str, str]]) -> List[Dict[str, any]]:
        """
        Download structures in parallel with error handling
        
        Args:
            url_list: List of URL dictionaries
            
        Returns:
            List of download results
        """
        logger.info(f"Starting bulk download of {len(url_list)} structures...")
        logger.info(f"Using {MAX_WORKERS} concurrent workers")
        
        results = []
        completed = 0
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all download tasks
            future_to_url = {
                executor.submit(self.download_single_structure, url_info): url_info 
                for url_info in url_list
            }
            
            # Process completed downloads
            for future in as_completed(future_to_url):
                url_info = future_to_url[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['success']:
                        self.download_stats['successful_downloads'] += 1
                        logger.debug(f"✅ {result['uniprot_id']} ({result['format']})")
                    else:
                        self.download_stats['failed_downloads'] += 1
                        logger.debug(f"❌ {result['uniprot_id']}: {result['error']}")
                    
                except Exception as e:
                    self.download_stats['failed_downloads'] += 1
                    logger.error(f"Download error for {url_info['uniprot_id']}: {e}")
                    results.append({
                        'uniprot_id': url_info['uniprot_id'],
                        'success': False,
                        'error': str(e)
                    })
                
                completed += 1
                if completed % 50 == 0:
                    logger.info(f"Progress: {completed}/{len(url_list)} ({completed/len(url_list)*100:.1f}%)")
        
        logger.info(f"Bulk download completed: {self.download_stats['successful_downloads']} successful, {self.download_stats['failed_downloads']} failed")
        
        return results
    
    def extract_plddt_scores(self, structure_file: str) -> Optional[float]:
        """
        Extract average pLDDT score from structure file
        
        Args:
            structure_file: Path to structure file (mmCIF or PDB)
            
        Returns:
            Average pLDDT score or None if extraction fails
        """
        try:
            file_path = Path(structure_file)
            
            if file_path.suffix == '.cif':
                # Parse mmCIF file
                mmcif_dict = MMCIF2Dict(structure_file)
                
                # Extract pLDDT scores
                if '_atom_site.B_iso_or_equiv' in mmcif_dict:
                    b_factors = mmcif_dict['_atom_site.B_iso_or_equiv']
                    # Convert to float and filter out non-numeric values
                    plddt_scores = []
                    for b_factor in b_factors:
                        try:
                            score = float(b_factor)
                            if 0 <= score <= 100:  # pLDDT range
                                plddt_scores.append(score)
                        except (ValueError, TypeError):
                            continue
                    
                    if plddt_scores:
                        return np.mean(plddt_scores)
                
            elif file_path.suffix == '.pdb':
                # Parse PDB file
                parser = PDBParser(QUIET=True)
                structure = parser.get_structure('protein', structure_file)
                
                plddt_scores = []
                for model in structure:
                    for chain in model:
                        for residue in chain:
                            for atom in residue:
                                # B-factor in PDB corresponds to pLDDT in AlphaFold
                                b_factor = atom.get_bfactor()
                                if 0 <= b_factor <= 100:
                                    plddt_scores.append(b_factor)
                
                if plddt_scores:
                    return np.mean(plddt_scores)
            
        except Exception as e:
            logger.debug(f"Error extracting pLDDT from {structure_file}: {e}")
        
        return None
    
    def check_structural_quality(self, download_results: List[Dict[str, any]]) -> pd.DataFrame:
        """
        Perform quality control on downloaded structures
        
        Args:
            download_results: Results from bulk download
            
        Returns:
            DataFrame with quality control results
        """
        logger.info("Performing structural quality control...")
        
        quality_results = []
        
        for result in download_results:
            uniprot_id = result['uniprot_id']
            quality_info = {
                'uniprot_id': uniprot_id,
                'download_success': result['success'],
                'file_format': result.get('format', None),
                'file_path': result.get('file_path', None),
                'avg_plddt': None,
                'confidence_level': None,
                'quality_status': 'failed'
            }
            
            if result['success'] and result.get('file_path'):
                # Extract pLDDT scores
                avg_plddt = self.extract_plddt_scores(result['file_path'])
                
                if avg_plddt is not None:
                    quality_info.update({
                        'avg_plddt': avg_plddt,
                        'confidence_level': 'high' if avg_plddt >= HIGH_CONFIDENCE_THRESHOLD else 'low',
                        'quality_status': 'high_confidence' if avg_plddt >= HIGH_CONFIDENCE_THRESHOLD else 'low_confidence'
                    })
                    
                    if avg_plddt >= HIGH_CONFIDENCE_THRESHOLD:
                        self.download_stats['high_confidence_structures'] += 1
                    else:
                        self.download_stats['low_confidence_structures'] += 1
                else:
                    quality_info['quality_status'] = 'no_plddt_data'
                    logger.warning(f"No pLDDT data found for {uniprot_id}")
            else:
                self.download_stats['missing_structures'] += 1
                quality_info['quality_status'] = 'download_failed'
            
            quality_results.append(quality_info)
        
        # Create DataFrame
        quality_df = pd.DataFrame(quality_results)
        
        logger.info(f"Quality control completed:")
        logger.info(f"  - High confidence structures: {self.download_stats['high_confidence_structures']}")
        logger.info(f"  - Low confidence structures: {self.download_stats['low_confidence_structures']}")
        logger.info(f"  - Missing structures: {self.download_stats['missing_structures']}")
        
        return quality_df
    
    def generate_structural_manifest(self, quality_df: pd.DataFrame, 
                                   output_path: str = "alphafold_structural_manifest.csv") -> str:
        """
        Generate final structural manifest
        
        Args:
            quality_df: Quality control results
            output_path: Output file path
            
        Returns:
            Path to generated manifest
        """
        logger.info(f"Generating structural manifest: {output_path}")
        
        # Save detailed manifest
        quality_df.to_csv(output_path, index=False)
        
        # Generate summary report
        summary_report = {
            'timestamp': datetime.now().isoformat(),
            'download_statistics': self.download_stats,
            'quality_summary': {
                'total_structures': len(quality_df),
                'high_confidence': len(quality_df[quality_df['quality_status'] == 'high_confidence']),
                'low_confidence': len(quality_df[quality_df['quality_status'] == 'low_confidence']),
                'failed_downloads': len(quality_df[quality_df['quality_status'] == 'download_failed']),
                'success_rate': (self.download_stats['successful_downloads'] / self.download_stats['total_requested'] * 100) if self.download_stats['total_requested'] > 0 else 0,
                'high_confidence_rate': (self.download_stats['high_confidence_structures'] / self.download_stats['total_requested'] * 100) if self.download_stats['total_requested'] > 0 else 0
            },
            'plddt_statistics': {
                'mean_plddt': quality_df['avg_plddt'].mean() if quality_df['avg_plddt'].notna().any() else None,
                'median_plddt': quality_df['avg_plddt'].median() if quality_df['avg_plddt'].notna().any() else None,
                'min_plddt': quality_df['avg_plddt'].min() if quality_df['avg_plddt'].notna().any() else None,
                'max_plddt': quality_df['avg_plddt'].max() if quality_df['avg_plddt'].notna().any() else None
            }
        }
        
        # Save summary report
        summary_path = output_path.replace('.csv', '_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        logger.info(f"Structural manifest saved to: {output_path}")
        logger.info(f"Summary report saved to: {summary_path}")
        
        return output_path
    
    def print_download_summary(self):
        """Print comprehensive download summary"""
        print("\n" + "="*80)
        print("🧬 ALPHAFOLD BULK DOWNLOAD SUMMARY")
        print("="*80)
        
        print(f"\n📊 DOWNLOAD STATISTICS:")
        print(f"  📋 Total Requested: {self.download_stats['total_requested']}")
        print(f"  ✅ Successful Downloads: {self.download_stats['successful_downloads']}")
        print(f"  ❌ Failed Downloads: {self.download_stats['failed_downloads']}")
        success_rate = (self.download_stats['successful_downloads']/self.download_stats['total_requested']*100) if self.download_stats['total_requested'] > 0 else 0
        print(f"  📈 Success Rate: {success_rate:.1f}%")
        
        print(f"\n🎯 QUALITY CONTROL RESULTS:")
        print(f"  🟢 High Confidence (pLDDT ≥ {HIGH_CONFIDENCE_THRESHOLD}): {self.download_stats['high_confidence_structures']}")
        print(f"  🟡 Low Confidence (pLDDT < {HIGH_CONFIDENCE_THRESHOLD}): {self.download_stats['low_confidence_structures']}")
        print(f"  🔴 Missing Structures: {self.download_stats['missing_structures']}")
        high_conf_rate = (self.download_stats['high_confidence_structures']/self.download_stats['total_requested']*100) if self.download_stats['total_requested'] > 0 else 0
        print(f"  📊 High Confidence Rate: {high_conf_rate:.1f}%")
        
        print(f"\n📁 OUTPUT LOCATIONS:")
        print(f"  📂 Structures Directory: {self.output_dir}")
        print(f"  📄 mmCIF Files: {self.mmcif_dir}")
        print(f"  📄 PDB Files: {self.pdb_dir}")
        
        print(f"\n🚀 READY FOR PHASE 2:")
        print(f"  ✅ High-confidence structures ready for GCN pipeline")
        print(f"  ✅ Structural features can be extracted")
        print(f"  ✅ Multi-modal architecture can be completed")
        
        print("\n" + "="*80)


def main():
    """
    Main function for bulk AlphaFold structure download
    """
    print("🧬 AlphaFold Bulk Structure Downloader - Module 4.5")
    print("="*70)
    
    # Configuration
    ts_gsd_path = "TS-GSD_consolidated.csv"
    output_dir = "alphafold_structures"
    manifest_path = "alphafold_structural_manifest.csv"
    
    # Check if TS-GSD file exists
    if not Path(ts_gsd_path).exists():
        print(f"❌ TS-GSD file not found: {ts_gsd_path}")
        print("Please ensure TS-GSD_consolidated.csv exists in the current directory")
        return
    
    try:
        # Initialize downloader
        downloader = AlphaFoldBulkDownloader(output_dir)
        
        # Step 1: Load UniProt IDs
        print("\n🔍 Step 1: Loading UniProt IDs from TS-GSD...")
        uniprot_ids = downloader.load_uniprot_ids(ts_gsd_path)
        
        # Step 2: Generate download URLs
        print("\n🔗 Step 2: Generating AlphaFold download URLs...")
        url_list = downloader.generate_alphafold_urls(uniprot_ids)
        
        # Step 3: Bulk download structures
        print("\n⬇️  Step 3: Downloading AlphaFold structures...")
        download_results = downloader.bulk_download_structures(url_list)
        
        # Step 4: Quality control
        print("\n🔍 Step 4: Performing structural quality control...")
        quality_df = downloader.check_structural_quality(download_results)
        
        # Step 5: Generate manifest
        print("\n📄 Step 5: Generating structural manifest...")
        manifest_path = downloader.generate_structural_manifest(quality_df, manifest_path)
        
        # Print summary
        downloader.print_download_summary()
        
        print(f"\n🎉 BULK DOWNLOAD COMPLETE!")
        print(f"📄 Structural manifest: {manifest_path}")
        print(f"📂 Structures directory: {output_dir}")
        print(f"🚀 Ready for Phase 2 GCN integration!")
        
    except Exception as e:
        logger.error(f"Bulk download failed: {e}")
        print(f"\n❌ Download failed: {e}")


if __name__ == "__main__":
    main()
