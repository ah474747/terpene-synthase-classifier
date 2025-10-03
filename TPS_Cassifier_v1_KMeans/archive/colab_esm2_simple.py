#!/usr/bin/env python3
"""
Simple Google Colab Notebook for ESM2 Embedding Generation
No structure prediction - just reliable sequence embeddings
"""

def generate_esm2_notebook():
    """Generate a simple, working ESM2 embedding notebook"""
    
    notebook_cells = []
    
    # Title
    notebook_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# ESM2 Embeddings for Germacrene Synthase Classification\n",
            "\n",
            "## Simple, Fast, and Reliable Approach\n",
            "\n",
            "This notebook generates ESM2 sequence embeddings without structure prediction.\n",
            "\n",
            "**Benefits:**\n",
            "- ✅ No complex dependencies\n",
            "- ✅ Fast processing (~10-15 min for 100 sequences)\n",
            "- ✅ Proven effective for protein classification\n",
            "- ✅ Works reliably in Colab\n",
            "\n",
            "**CRITICAL**: Select GPU runtime → Runtime → Change runtime type → GPU"
        ]
    })
    
    # Cell 1: GPU Check
    notebook_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 1: GPU Check\n",
            "import torch\n",
            "import sys\n",
            "import os\n",
            "\n",
            "print(f\"Python version: {sys.version}\")\n",
            "print(f\"PyTorch version: {torch.__version__}\")\n",
            "print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
            "\n",
            "if torch.cuda.is_available():\n",
            "    print(f\"CUDA device: {torch.cuda.get_device_name(0)}\")\n",
            "    print(f\"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\")\n",
            "else:\n",
            "    print(\"⚠️  No GPU! Select GPU runtime: Runtime → Change runtime type → GPU\")"
        ]
    })
    
    # Cell 2: Google Drive
    notebook_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 2: Google Drive Setup\n",
            "from google.colab import drive\n",
            "\n",
            "drive.mount('/content/drive')\n",
            "\n",
            "# Test write access\n",
            "try:\n",
            "    test_file = '/content/drive/MyDrive/test_write.txt'\n",
            "    with open(test_file, 'w') as f:\n",
            "        f.write('Test')\n",
            "    os.remove(test_file)\n",
            "    print(\"✅ Google Drive mounted successfully\")\n",
            "except Exception as e:\n",
            "    print(f\"❌ Drive mount failed: {e}\")"
        ]
    })
    
    # Cell 3: Install ESM
    notebook_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 3: Install Dependencies\n",
            "!pip install -q fair-esm biopython numpy pandas tqdm\n",
            "\n",
            "import esm\n",
            "from Bio import SeqIO\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "from tqdm import tqdm\n",
            "\n",
            "print(\"✅ All dependencies installed\")"
        ]
    })
    
    # Cell 4: Load ESM2 Model
    notebook_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 4: Load ESM2 Model\n",
            "print(\"Loading ESM2 model (this may take a minute)...\")\n",
            "\n",
            "# Load ESM-2 model (650M parameters)\n",
            "model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()\n",
            "batch_converter = alphabet.get_batch_converter()\n",
            "model.eval()\n",
            "\n",
            "# Move to GPU\n",
            "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
            "model = model.to(device)\n",
            "\n",
            "print(f\"✅ ESM2 model loaded on {device}\")\n",
            "print(f\"Model parameters: {sum(p.numel() for p in model.parameters()):,}\")"
        ]
    })
    
    # Cell 5: Load Sequences
    notebook_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 5: Load Sequences\n",
            "# Configuration\n",
            "INPUT_FASTA_PATH = 'input_sequences.fasta'\n",
            "MAX_SEQUENCES = 100  # Adjust as needed (100 recommended)\n",
            "\n",
            "# Load sequences\n",
            "sequences = []\n",
            "for record in SeqIO.parse(INPUT_FASTA_PATH, 'fasta'):\n",
            "    sequences.append({\n",
            "        'id': record.id,\n",
            "        'description': record.description,\n",
            "        'sequence': str(record.seq),\n",
            "        'length': len(record.seq)\n",
            "    })\n",
            "\n",
            "df = pd.DataFrame(sequences)\n",
            "print(f\"Loaded {len(df)} sequences\")\n",
            "\n",
            "# Limit to MAX_SEQUENCES\n",
            "df = df.head(MAX_SEQUENCES)\n",
            "print(f\"Processing {len(df)} sequences\")\n",
            "print(f\"Length range: {df['length'].min()}-{df['length'].max()} aa\")\n",
            "print(f\"Average length: {df['length'].mean():.1f} aa\")"
        ]
    })
    
    # Cell 6: Generate Embeddings
    notebook_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 6: Generate ESM2 Embeddings\n",
            "import time\n",
            "\n",
            "def generate_esm2_embeddings(sequences_list, batch_size=8):\n",
            "    \"\"\"Generate ESM2 embeddings for sequences\"\"\"\n",
            "    all_embeddings = []\n",
            "    \n",
            "    # Clear GPU memory\n",
            "    torch.cuda.empty_cache()\n",
            "    \n",
            "    # Process in batches\n",
            "    for i in tqdm(range(0, len(sequences_list), batch_size)):\n",
            "        batch = sequences_list[i:i+batch_size]\n",
            "        \n",
            "        # Prepare batch data\n",
            "        batch_labels, batch_strs, batch_tokens = batch_converter(batch)\n",
            "        batch_tokens = batch_tokens.to(device)\n",
            "        \n",
            "        # Generate embeddings\n",
            "        with torch.no_grad():\n",
            "            results = model(batch_tokens, repr_layers=[33], return_contacts=False)\n",
            "        \n",
            "        # Extract embeddings (mean pooling over sequence length)\n",
            "        embeddings = results['representations'][33]\n",
            "        \n",
            "        # Mean pool over sequence length (excluding special tokens)\n",
            "        for j, (label, seq) in enumerate(batch):\n",
            "            seq_len = len(seq)\n",
            "            # Average over sequence (skip BOS/EOS tokens)\n",
            "            embedding = embeddings[j, 1:seq_len+1].mean(0)\n",
            "            all_embeddings.append(embedding.cpu().numpy())\n",
            "    \n",
            "    return np.vstack(all_embeddings)\n",
            "\n",
            "# Prepare sequences for ESM2\n",
            "sequences_for_esm = [(row['id'], row['sequence']) for _, row in df.iterrows()]\n",
            "\n",
            "print(f\"\\n🔄 Generating ESM2 embeddings for {len(sequences_for_esm)} sequences...\")\n",
            "print(f\"Batch size: 8\")\n",
            "print(f\"Estimated time: ~{len(sequences_for_esm) * 0.1:.1f} minutes\\n\")\n",
            "\n",
            "start_time = time.time()\n",
            "\n",
            "# Generate embeddings\n",
            "esm2_embeddings = generate_esm2_embeddings(sequences_for_esm, batch_size=8)\n",
            "\n",
            "end_time = time.time()\n",
            "total_time = end_time - start_time\n",
            "\n",
            "# Print results\n",
            "print(f\"\\n{'='*60}\")\n",
            "print(f\"✅ Generated embeddings shape: {esm2_embeddings.shape}\")\n",
            "print(f\"   ({esm2_embeddings.shape[0]} sequences × {esm2_embeddings.shape[1]} dimensions)\")\n",
            "print(f\"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)\")\n",
            "print(f\"📊 Average time per sequence: {total_time/len(sequences_for_esm):.2f} seconds\")\n",
            "print(f\"🧠 Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB\")\n",
            "print(f\"{'='*60}\")"
        ]
    })
    
    # Cell 7: Save Results
    notebook_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 7: Save to Google Drive\n",
            "# Create output directory\n",
            "output_dir = '/content/drive/MyDrive/ESM2_Embeddings_Results'\n",
            "os.makedirs(output_dir, exist_ok=True)\n",
            "\n",
            "# Save embeddings\n",
            "embedding_file = f\"{output_dir}/esm2_embeddings.npy\"\n",
            "np.save(embedding_file, esm2_embeddings)\n",
            "\n",
            "# Save sequence information\n",
            "sequence_info_file = f\"{output_dir}/sequence_info.csv\"\n",
            "df.to_csv(sequence_info_file, index=False)\n",
            "\n",
            "# Save metadata\n",
            "metadata = {\n",
            "    'num_sequences': len(df),\n",
            "    'embedding_dim': esm2_embeddings.shape[1],\n",
            "    'model': 'ESM2-650M',\n",
            "    'total_time_seconds': total_time,\n",
            "    'avg_time_per_sequence': total_time/len(df)\n",
            "}\n",
            "\n",
            "import json\n",
            "with open(f\"{output_dir}/metadata.json\", 'w') as f:\n",
            "    json.dump(metadata, f, indent=2)\n",
            "\n",
            "# Verify file sizes\n",
            "embedding_size = os.path.getsize(embedding_file) / 1e6\n",
            "\n",
            "print(f\"\\n✅ Results saved to Google Drive\")\n",
            "print(f\"📁 Location: {output_dir}\")\n",
            "print(f\"\\nFiles created:\")\n",
            "print(f\"  • esm2_embeddings.npy ({embedding_size:.1f} MB)\")\n",
            "print(f\"  • sequence_info.csv\")\n",
            "print(f\"  • metadata.json\")"
        ]
    })
    
    # Cell 8: Download Instructions
    notebook_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 📥 Download Instructions\n",
            "\n",
            "### Files Generated\n",
            "1. **esm2_embeddings.npy** - ESM2 embeddings (1280 dimensions)\n",
            "2. **sequence_info.csv** - Sequence metadata\n",
            "3. **metadata.json** - Processing information\n",
            "\n",
            "### How to Download\n",
            "\n",
            "#### Option 1: Google Drive\n",
            "1. Go to [Google Drive](https://drive.google.com)\n",
            "2. Navigate to `ESM2_Embeddings_Results` folder\n",
            "3. Download the files\n",
            "\n",
            "#### Option 2: Colab File Browser\n",
            "1. Click the folder icon in the left sidebar\n",
            "2. Navigate to `/content/drive/MyDrive/ESM2_Embeddings_Results/`\n",
            "3. Right-click files and select \"Download\"\n",
            "\n",
            "### Next Steps\n",
            "1. Download `esm2_embeddings.npy` and `sequence_info.csv`\n",
            "2. Train XGBoost classifier with these embeddings\n",
            "3. Compare performance with other models\n",
            "4. Analyze feature importance\n",
            "\n",
            "### Using the Embeddings\n",
            "```python\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "\n",
            "# Load embeddings\n",
            "embeddings = np.load('esm2_embeddings.npy')\n",
            "sequences = pd.read_csv('sequence_info.csv')\n",
            "\n",
            "print(f\"Embeddings shape: {embeddings.shape}\")\n",
            "print(f\"Sequences: {len(sequences)}\")\n",
            "```"
        ]
    })
    
    # Create notebook structure
    notebook = {
        "cells": notebook_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            },
            "accelerator": "GPU"
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    return notebook

def save_esm2_notebook():
    """Save the ESM2 notebook to file"""
    import json
    
    notebook = generate_esm2_notebook()
    
    with open('colab_esm2_simple.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print("✅ ESM2 Colab notebook saved as colab_esm2_simple.ipynb")

if __name__ == "__main__":
    save_esm2_notebook()
