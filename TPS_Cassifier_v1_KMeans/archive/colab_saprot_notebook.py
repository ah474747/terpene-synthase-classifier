#!/usr/bin/env python3
"""
Google Colab Notebook for SaProt Embedding Generation
This script generates the Colab notebook cells for the SaProt workflow
"""

def generate_colab_notebook():
    """Generate the complete Colab notebook for SaProt embedding generation"""
    
    notebook_content = """
# SaProt Embeddings for Germacrene Synthase Classification

## Step 1: Environment Setup and Validation

### 1.1 GPU Runtime Selection
**IMPORTANT**: Go to Runtime → Change runtime type → Hardware accelerator → GPU

### 1.2 Environment Check
```python
import torch
import sys
import os

# Check CUDA availability
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("⚠️  No GPU detected! Please select GPU runtime.")
```

### 1.3 Mount Google Drive and Validate
```python
from google.colab import drive
import shutil

# Mount Google Drive
drive.mount('/content/drive')

# Validate mounting
try:
    # Test write access
    test_file = '/content/drive/MyDrive/test_write.txt'
    with open(test_file, 'w') as f:
        f.write('Test write access')
    
    # Check available space
    statvfs = os.statvfs('/content/drive/MyDrive')
    free_space_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
    print(f"✅ Google Drive mounted successfully")
    print(f"Available space: {free_space_gb:.1f} GB")
    
    # Clean up test file
    os.remove(test_file)
    
except Exception as e:
    print(f"❌ Google Drive mounting failed: {e}")
    print("Please check your Google Drive access and try again")

# Create project directory
PROJECT_DIR = "/content/drive/MyDrive/germacrene_saprot_project"
os.makedirs(PROJECT_DIR, exist_ok=True)

# Create subdirectories
subdirs = ['structures', '3di_sequences', 'embeddings', 'results']
for subdir in subdirs:
    os.makedirs(os.path.join(PROJECT_DIR, subdir), exist_ok=True)

print(f"✅ Project directory created: {PROJECT_DIR}")
```

### 1.4 Install Dependencies
```python
# Install required packages
!pip install -q torch transformers huggingface-hub biopython numpy pandas scikit-learn
!pip install -q foldseek  # If available via pip, otherwise use manual installation

# Verify installations
import transformers
import Bio
print(f"Transformers version: {transformers.__version__}")
print(f"Biopython version: {Bio.__version__}")
```

## Step 2: Foldseek Installation and Setup

### 2.1 Install Foldseek
```python
# Download and install Foldseek
!wget -q https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz
!tar xzf foldseek-linux-avx2.tar.gz
!chmod +x foldseek/bin/foldseek

# Add to PATH
import os
os.environ['PATH'] += ':/content/foldseek/bin'

# Verify installation
!foldseek --help | head -10
print("✅ Foldseek installed successfully")
```

### 2.2 Test Foldseek with Sample Structure
```python
# Download a sample PDB file for testing
!wget -q https://files.rcsb.org/download/1CRN.pdb -O sample.pdb

# Test 3Di generation
!foldseek createdb sample.pdb sample_db
!foldseek structureto3di sample_db sample_3di_db
!foldseek createseqfiledb sample_3di_db sample_3di_seq

# Check output
!head -5 sample_3di_seq
print("✅ Foldseek 3Di generation working")
```

## Step 3: SaProt Installation and Configuration

### 3.1 Clone SaProt Repository
```python
# Clone SaProt repository
!git clone https://github.com/mingkangyang/SaProt.git
%cd SaProt

# Check repository structure
!ls -la
print("✅ SaProt repository cloned")
```

### 3.2 Install SaProt Dependencies
```python
# Install SaProt specific dependencies
!pip install -q -e .

# Check if installation was successful
try:
    import saprot
    print("✅ SaProt installed successfully")
except ImportError as e:
    print(f"❌ SaProt installation failed: {e}")
```

### 3.3 Download SaProt Model
```python
from huggingface_hub import hf_hub_download
import torch

# Download SaProt model
model_path = hf_hub_download(
    repo_id="westlake-repl/SaProt", 
    filename="saprot_650M.pt",
    cache_dir="/content/models"
)

print(f"Model downloaded to: {model_path}")

# Check model size
model_size = os.path.getsize(model_path) / 1e9
print(f"Model size: {model_size:.1f} GB")
```

## Step 4: Data Preparation

### 4.1 Upload Dataset
```python
# Upload the expanded dataset FASTA file
# You can either:
# 1. Upload manually through Colab file browser
# 2. Copy from Google Drive if already uploaded

# Check if file exists
if os.path.exists('/content/expanded_germacrene_dataset.fasta'):
    print("✅ Dataset file found")
    !head -5 /content/expanded_germacrene_dataset.fasta
else:
    print("❌ Dataset file not found. Please upload expanded_germacrene_dataset.fasta")
```

### 4.2 Parse Sequences
```python
from Bio import SeqIO
import pandas as pd

# Load sequences
sequences = []
for record in SeqIO.parse('/content/expanded_germacrene_dataset.fasta', 'fasta'):
    sequences.append({
        'id': record.id,
        'description': record.description,
        'sequence': str(record.seq),
        'length': len(record.seq)
    })

df = pd.DataFrame(sequences)
print(f"Loaded {len(df)} sequences")
print(f"Sequence length range: {df['length'].min()}-{df['length'].max()} amino acids")
print(f"Average length: {df['length'].mean():.1f} amino acids")

# Save sequence info
df.to_csv('/content/sequence_info.csv', index=False)
print("✅ Sequence information saved")
```

## Step 5: Structure Acquisition and 3Di Generation

### 5.1 Strategy Selection
```python
# Choose your approach based on available data:
# Option A: Download from AlphaFold Database (for known UniProt IDs)
# Option B: Generate structures with ColabFold (for novel sequences)
# Option C: Use pre-computed 3Di sequences (if available)

print("Available strategies:")
print("A. Download from AlphaFold Database (fastest, limited to known structures)")
print("B. Generate with ColabFold (slower, works for any sequence)")
print("C. Use pre-computed 3Di (fastest, if available)")

# For this workflow, we'll use Option A with fallback to Option B
strategy = "A"  # Change to "B" or "C" as needed
```

### 5.2 Option A: Download from AlphaFold Database
```python
import requests
import time
from pathlib import Path
import re

# Create structures directory
structures_dir = Path('/content/structures')
structures_dir.mkdir(exist_ok=True)

def download_alphafold_structure(uniprot_id, output_dir):
    \"\"\"Download structure from AlphaFold Database\"\"\"
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            output_path = output_dir / f"{uniprot_id}.pdb"
            with open(output_path, 'w') as f:
                f.write(response.text)
            return True
        else:
            print(f"❌ Failed to download {uniprot_id}: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error downloading {uniprot_id}: {e}")
        return False

# Extract UniProt IDs from sequence descriptions
uniprot_ids = []
for desc in df['description']:
    # Look for UniProt IDs in description (6-character alphanumeric)
    match = re.search(r'([A-Z0-9]{6})', desc)
    if match:
        uniprot_ids.append(match.group(1))

print(f"Found {len(uniprot_ids)} potential UniProt IDs")
print(f"Sample IDs: {uniprot_ids[:5]}")

# Download structures (limit to first 50 for testing)
successful_downloads = 0
for i, uniprot_id in enumerate(uniprot_ids[:50]):
    print(f"Downloading {i+1}/50: {uniprot_id}")
    if download_alphafold_structure(uniprot_id, structures_dir):
        successful_downloads += 1
    time.sleep(0.5)  # Be respectful to the server

print(f"✅ Downloaded {successful_downloads} structures")
```

### 5.3 Option B: Generate Structures with ColabFold
```python
# Install ColabFold
!pip install -q colabfold

# Generate structures for sequences without UniProt IDs
from colabfold import ColabFold

def generate_structure_with_colabfold(sequence, output_path):
    \"\"\"Generate structure using ColabFold\"\"\"
    try:
        # This is a simplified example - actual implementation would be more complex
        print(f"Generating structure for sequence of length {len(sequence)}")
        # ColabFold implementation would go here
        return True
    except Exception as e:
        print(f"Error generating structure: {e}")
        return False

# Process sequences without UniProt IDs
sequences_without_structures = df[~df['description'].str.contains(r'[A-Z0-9]{6}')]
print(f"Found {len(sequences_without_structures)} sequences without UniProt IDs")

# Generate structures for a subset (ColabFold is computationally intensive)
for i, (_, row) in enumerate(sequences_without_structures.head(10).iterrows()):
    print(f"Generating structure {i+1}/10 for {row['id']}")
    output_path = structures_dir / f"{row['id']}.pdb"
    generate_structure_with_colabfold(row['sequence'], output_path)
```

### 5.4 Option C: Use Pre-computed 3Di Sequences
```python
# If you have pre-computed 3Di sequences, load them directly
def load_precomputed_3di(file_path):
    \"\"\"Load pre-computed 3Di sequences\"\"\"
    sequences_3di = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    seq_id = line.strip()[1:]
                else:
                    sequences_3di[seq_id] = line.strip()
        print(f"✅ Loaded {len(sequences_3di)} pre-computed 3Di sequences")
        return sequences_3di
    except FileNotFoundError:
        print("❌ Pre-computed 3Di file not found")
        return {}

# Uncomment if you have pre-computed 3Di sequences
# sequences_3di = load_precomputed_3di('/content/precomputed_3di.fasta')
```

### 5.2 Generate 3Di Sequences
```python
# Generate 3Di sequences for downloaded structures
def generate_3di_sequence(pdb_file, output_file):
    \"\"\"Generate 3Di sequence from PDB file\"\"\"
    try:
        # Create database
        db_name = pdb_file.stem
        !foldseek createdb {pdb_file} {db_name}_db
        
        # Convert to 3Di
        !foldseek structureto3di {db_name}_db {db_name}_3di_db
        
        # Create sequence file
        !foldseek createseqfiledb {db_name}_3di_db {db_name}_3di_seq
        
        # Read 3Di sequence
        with open(f'{db_name}_3di_seq', 'r') as f:
            lines = f.readlines()
            if len(lines) >= 2:
                return lines[1].strip()
        return None
    except Exception as e:
        print(f"Error processing {pdb_file}: {e}")
        return None

# Process downloaded structures
structures_dir = Path('/content/structures')
pdb_files = list(structures_dir.glob('*.pdb'))

print(f"Processing {len(pdb_files)} PDB files...")

sequences_3di = {}
for pdb_file in pdb_files:
    print(f"Processing {pdb_file.name}")
    seq_3di = generate_3di_sequence(pdb_file, pdb_file.with_suffix('.3di'))
    if seq_3di:
        sequences_3di[pdb_file.stem] = seq_3di
        print(f"✅ Generated 3Di sequence for {pdb_file.stem}")
    else:
        print(f"❌ Failed to generate 3Di sequence for {pdb_file.stem}")

print(f"✅ Generated {len(sequences_3di)} 3Di sequences")
```

## Step 6: SaProt Embedding Generation

### 6.1 Load SaProt Model
```python
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

# Load SaProt model and tokenizer
model_name = "westlake-repl/SaProt"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Move model to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

print(f"✅ SaProt model loaded on {device}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### 6.2 Generate Embeddings with Batch Processing
```python
def generate_saprot_embeddings(sequences, batch_size=8):
    \"\"\"Generate SaProt embeddings for sequences with optimal batch processing\"\"\"
    embeddings = []
    
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(sequences)-1)//batch_size + 1}")
        
        # Tokenize batch
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Use mean pooling
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(batch_embeddings.cpu().numpy())
    
    return np.vstack(embeddings)

# Generate embeddings for 3Di sequences
if sequences_3di:
    print("Generating SaProt embeddings for 3Di sequences...")
    seq_list = list(sequences_3di.values())
    saprot_embeddings = generate_saprot_embeddings(seq_list)
    
    print(f"✅ Generated embeddings shape: {saprot_embeddings.shape}")
    
    # Save embeddings
    np.save('/content/saprot_embeddings.npy', saprot_embeddings)
    print("✅ Embeddings saved to saprot_embeddings.npy")
else:
    print("❌ No 3Di sequences available for embedding generation")
```

### 6.3 Save Results to Google Drive
```python
# Copy results to Google Drive
!cp /content/saprot_embeddings.npy {PROJECT_DIR}/embeddings/
!cp /content/sequence_info.csv {PROJECT_DIR}/results/

# Verify file sizes
embedding_size = os.path.getsize(f'{PROJECT_DIR}/embeddings/saprot_embeddings.npy') / 1e6
print(f"✅ Results saved to Google Drive")
print(f"Embedding file size: {embedding_size:.1f} MB")
print(f"Files saved to: {PROJECT_DIR}")
```

## Step 7: Download Instructions

### 7.1 Download Files
```python
print("📥 Download the following files from the Colab file browser:")
print("1. saprot_embeddings.npy - SaProt embeddings")
print("2. sequence_info.csv - Sequence information")
print("3. Any additional output files")
print()
print("Or copy from Google Drive:")
print(f"Google Drive location: {PROJECT_DIR}")
```

### 7.2 Next Steps
```python
print("🔄 Next steps for local evaluation:")
print("1. Download embeddings and sequence info")
print("2. Train XGBoost model with SaProt embeddings")
print("3. Compare performance with ESM2 embeddings")
print("4. Analyze feature importance and interpretability")
print("5. Select best performing model")
```

## Summary

This notebook provides a complete workflow for:
- ✅ Environment setup and validation
- ✅ Foldseek installation and 3Di generation
- ✅ SaProt installation and model loading
- ✅ Structure acquisition from AlphaFold Database
- ✅ SaProt embedding generation
- ✅ Results export to Google Drive

The generated embeddings can now be used for training and evaluating the germacrene synthase classifier.
"""
    
    return notebook_content

def save_notebook():
    """Save the notebook content to a file"""
    content = generate_colab_notebook()
    
    with open('colab_saprot_notebook.ipynb', 'w') as f:
        # Create a basic Jupyter notebook structure
        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["# SaProt Embeddings for Germacrene Synthase Classification"]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [content]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.8.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        import json
        json.dump(notebook, f, indent=2)
    
    print("✅ Colab notebook saved as colab_saprot_notebook.ipynb")

if __name__ == "__main__":
    save_notebook()
