"""
Create a large test dataset to demonstrate performance with 1000+ sequences
"""

import random
import os

def create_large_test_dataset(filename="large_test.fasta", num_sequences=1000):
    """Create a large FASTA file for testing performance."""
    
    # Base sequences for different types
    base_sequences = {
        'germacrene_a': "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELGERMACA",
        'germacrene_d': "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELGERMACD", 
        'limonene': "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELDDAAD",
        'pinene': "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELNSE",
        'myrcene': "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELRRAAW",
        'linalool': "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELGXGXXG",
        'geraniol': "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELHXXXH",
        'caryophyllene': "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELCARYOPH"
    }
    
    # Clean up sequences (remove invalid characters)
    for key in base_sequences:
        seq = base_sequences[key]
        # Replace X with A, numbers with A
        seq = seq.replace('X', 'A').replace('2', 'A').replace('3', 'A').replace('8', 'A')
        base_sequences[key] = seq
    
    print(f"Creating {num_sequences} sequences in {filename}...")
    
    with open(filename, 'w') as f:
        for i in range(num_sequences):
            # Randomly select sequence type
            seq_type = random.choice(list(base_sequences.keys()))
            base_seq = base_sequences[seq_type]
            
            # Add some random mutations to create diversity
            mutated_seq = list(base_seq)
            for j in range(len(mutated_seq)):
                if random.random() < 0.05:  # 5% mutation rate
                    mutated_seq[j] = random.choice('ACDEFGHIKLMNPQRSTVWY')
            
            mutated_seq = ''.join(mutated_seq)
            
            # Write to file
            f.write(f">{seq_type}_synthase_{i+1}\n")
            f.write(f"{mutated_seq}\n")
    
    print(f"✅ Created {filename} with {num_sequences} sequences")
    return filename

if __name__ == "__main__":
    create_large_test_dataset("large_test.fasta", 1000)
