#!/usr/bin/env python3
"""
Test script for AlphaFold bulk downloader - demonstrates functionality on small subset
"""

import pandas as pd
from pathlib import Path
from alphafold_bulk_downloader import AlphaFoldBulkDownloader

def test_small_subset():
    """Test the downloader on a small subset of UniProt IDs"""
    
    print("🧪 Testing AlphaFold Bulk Downloader on Small Subset")
    print("="*60)
    
    # Create a small test dataset
    test_uniprot_ids = [
        "P0C2A9",  # Example UniProt ID 1
        "Q9FJZ4",  # Example UniProt ID 2
        "P0C2B0",  # Example UniProt ID 3
        "Q9FJZ5",  # Example UniProt ID 4
        "P0C2B1"   # Example UniProt ID 5
    ]
    
    print(f"📋 Test UniProt IDs: {test_uniprot_ids}")
    
    # Initialize downloader with test directory
    downloader = AlphaFoldBulkDownloader("test_alphafold_structures")
    
    # Generate URLs
    print("\n🔗 Generating download URLs...")
    url_list = downloader.generate_alphafold_urls(test_uniprot_ids)
    print(f"Generated {len(url_list)} URLs")
    
    # Show URL format
    if url_list:
        sample_url = url_list[0]
        print(f"\n📄 Sample URL format:")
        print(f"  UniProt ID: {sample_url['uniprot_id']}")
        print(f"  mmCIF URL: {sample_url['mmcif_url']}")
        print(f"  PDB URL: {sample_url['pdb_url']}")
    
    # Test download (just first 2 to be quick)
    print(f"\n⬇️  Testing download (first 2 structures)...")
    test_results = downloader.bulk_download_structures(url_list[:2])
    
    # Show results
    print(f"\n📊 Download Results:")
    for result in test_results:
        status = "✅" if result['success'] else "❌"
        print(f"  {status} {result['uniprot_id']}: {result.get('format', 'failed')}")
        if result['success']:
            print(f"    File: {result['file_path']}")
    
    # Test quality control
    print(f"\n🔍 Testing quality control...")
    quality_df = downloader.check_structural_quality(test_results)
    
    print(f"\n📈 Quality Control Results:")
    print(quality_df[['uniprot_id', 'download_success', 'avg_plddt', 'confidence_level']].to_string(index=False))
    
    # Generate test manifest
    print(f"\n📄 Generating test manifest...")
    manifest_path = downloader.generate_structural_manifest(quality_df, "test_structural_manifest.csv")
    
    # Print summary
    downloader.print_download_summary()
    
    print(f"\n✅ Test completed successfully!")
    print(f"📁 Test files saved to: test_alphafold_structures/")
    print(f"📄 Test manifest: test_structural_manifest.csv")

if __name__ == "__main__":
    test_small_subset()



