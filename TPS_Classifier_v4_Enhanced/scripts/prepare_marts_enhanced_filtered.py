#!/usr/bin/env python3
import csv, argparse
from pathlib import Path

def is_valid_protein_sequence(seq):
    """Check if sequence contains only valid protein characters"""
    valid_chars = set('ACDEFGHIKLMNPQRSTVWY')
    invalid_chars = set(seq.upper()) - valid_chars
    return len(invalid_chars) == 0, invalid_chars

def main(args):
    sequences = []
    labels_used = []
    
    with open(args.input, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            seq_id = row['Enzyme_marts_ID'] or f"marts_seq_{i}"
            sequence = row['Aminoacid_sequence'].upper()
            
            # Filter out sequences with invalid characters
            is_valid, invalid_chars = is_valid_protein_sequence(sequence)
            if not is_valid:
                print(f"Skipping {seq_id}: contains invalid chars: {invalid_chars}")
                continue
            
            if args.label_column == 'product_class' and row.get('product_class'):
                label = row['product_class']
            elif args.label_column == 'enzyme_type' and row.get('enzyme_type'):
                label = row['enzyme_type']
            elif args.label_column == 'enzyme_class' and row.get('enzyme_class'):
                label = row['enzyme_class']
            else:
                label = row.get('Type', 'unknown')
            
            label = label.lower().strip()
            
            if sequence and len(sequence) > 50:
                sequences.append({
                    'id': seq_id,
                    'sequence': sequence,
                    'label': label
                })
                if label not in labels_used:
                    labels_used.append(label)
    
    # Write filtered training data CSV
    with open(args.output, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'sequence', 'label'])
        writer.writeheader()
        for seq in sequences:
            writer.writerow(seq)
    
    print(f"Prepared {len(sequences)} valid sequences")
    print(f"Unique labels ({args.label_column}): {len(labels_used)}")
    print(f"Labels: {sorted(labels_used)}")
    print(f"Written to: {args.output}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/marts_db_enhanced.csv")
    p.add_argument("--output", default="data/marts_enhanced_filtered.csv")
    p.add_argument("--label_column", choices=['product_class', 'enzyme_type', 'enzyme_class'], default='product_class')
    main(p.parse_args())

