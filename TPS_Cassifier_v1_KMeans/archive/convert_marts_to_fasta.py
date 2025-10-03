#!/usr/bin/env python3
"""
Convert MARTS-DB CSV to FASTA format for the classifier
"""

import pandas as pd
import os


def convert_marts_to_fasta(csv_file, fasta_file):
    """
    Convert MARTS-DB CSV to FASTA format
    
    Args:
        csv_file: Path to MARTS-DB CSV file
        fasta_file: Path to output FASTA file
    """
    print(f"Converting {csv_file} to {fasta_file}...")
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    print(f"Loaded {len(df)} records from CSV")
    print(f"Columns: {list(df.columns)}")
    
    # Check if we have the required columns
    required_cols = ['Enzyme_marts_ID', 'Aminoacid_sequence', 'Product_name']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        return False
    
    # Remove duplicates and empty sequences
    df = df.dropna(subset=['Aminoacid_sequence'])
    df = df[df['Aminoacid_sequence'].str.strip() != '']
    df = df.drop_duplicates(subset=['Enzyme_marts_ID', 'Aminoacid_sequence'])
    
    print(f"After filtering: {len(df)} unique sequences")
    
    # Write FASTA file
    with open(fasta_file, 'w') as f:
        for idx, row in df.iterrows():
            # Create sequence ID with product information
            seq_id = f"{row['Enzyme_marts_ID']}_{row['Product_name'].replace(' ', '_').replace('α', 'alpha').replace('β', 'beta')}"
            
            # Write FASTA entry
            f.write(f">{seq_id}\n")
            f.write(f"{row['Aminoacid_sequence']}\n")
    
    print(f"✓ Created FASTA file with {len(df)} sequences")
    return True


def main():
    """Main function"""
    csv_file = "data/marts_db.csv"
    fasta_file = "data/marts_db.fasta"
    
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found at {csv_file}")
        return
    
    success = convert_marts_to_fasta(csv_file, fasta_file)
    
    if success:
        print(f"\n✓ Conversion completed successfully!")
        print(f"FASTA file saved to: {fasta_file}")
        
        # Show some statistics
        df = pd.read_csv(csv_file)
        unique_products = df['Product_name'].nunique()
        germacrene_count = df[df['Product_name'].str.contains('germacrene', case=False, na=False)].shape[0]
        
        print(f"\nDataset Statistics:")
        print(f"- Total sequences: {len(df)}")
        print(f"- Unique products: {unique_products}")
        print(f"- Germacrene sequences: {germacrene_count}")
        print(f"- Other sequences: {len(df) - germacrene_count}")
    else:
        print("✗ Conversion failed!")


if __name__ == "__main__":
    main()

