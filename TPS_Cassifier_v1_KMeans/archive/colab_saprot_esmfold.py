#!/usr/bin/env python3
"""
Google Colab Notebook for SaProt Embedding Generation with ESMFold
This implementation uses ESMFold for structure prediction
"""

def generate_esmfold_notebook():
    """Generate the complete Colab notebook with ESMFold implementation"""
    
    notebook_cells = []
    
    # Title
    notebook_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# SaProt Embeddings for Germacrene Synthase Classification\n",
            "\n",
            "## Implementation with ESMFold for Structure Prediction\n",
            "\n",
            "This notebook generates SaProt embeddings using ESMFold for structure prediction."
        ]
    })
    
    # Phase 1: Environment Setup
    notebook_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🧪 Phase 1: Environment Setup and Validation\n",
            "\n",
            "### Important: Select GPU Runtime\n",
            "**CRITICAL**: Go to Runtime → Change runtime type → Hardware accelerator → GPU"
        ]
    })
    
    # 1.1 GPU Check
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
            "# Check CUDA availability\n",
            "print(f\"Python version: {sys.version}\")\n",
            "print(f\"PyTorch version: {torch.__version__}\")\n",
            "print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
            "\n",
            "if torch.cuda.is_available():\n",
            "    print(f\"CUDA device: {torch.cuda.get_device_name(0)}\")\n",
            "    print(f\"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\")\n",
            "    print(f\"CUDA version: {torch.version.cuda}\")\n",
            "else:\n",
            "    print(\"⚠️  No GPU detected! Please select GPU runtime.\")\n",
            "    print(\"Go to Runtime → Change runtime type → Hardware accelerator → GPU\")"
        ]
    })
    
    # 1.2 Google Drive Setup
    notebook_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 2: Google Drive Setup\n",
            "from google.colab import drive\n",
            "import shutil\n",
            "\n",
            "# Mount Google Drive\n",
            "drive.mount('/content/drive')\n",
            "\n",
            "# Validate mounting with write/read test\n",
            "try:\n",
            "    # Test write access\n",
            "    test_file = '/content/drive/MyDrive/test_write.txt'\n",
            "    with open(test_file, 'w') as f:\n",
            "        f.write('Test write access')\n",
            "    \n",
            "    # Test read access\n",
            "    with open(test_file, 'r') as f:\n",
            "        content = f.read()\n",
            "        if content == 'Test write access':\n",
            "            print(\"✅ Google Drive mounted successfully\")\n",
            "        else:\n",
            "            print(\"❌ Google Drive read/write test failed\")\n",
            "    \n",
            "    # Clean up test file\n",
            "    os.remove(test_file)\n",
            "    \n",
            "except Exception as e:\n",
            "    print(f\"❌ Google Drive mounting failed: {e}\")\n",
            "    print(\"Please check your Google Drive access and try again\")"
        ]
    })
    
    # 1.3 Space Monitoring
    notebook_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 3: Space Monitoring\n",
            "# Check available space\n",
            "MIN_GB_REQUIRED = 50\n",
            "\n",
            "try:\n",
            "    statvfs = os.statvfs('/content/drive/MyDrive')\n",
            "    free_space_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)\n",
            "    print(f\"Available space: {free_space_gb:.1f} GB\")\n",
            "    \n",
            "    if free_space_gb < MIN_GB_REQUIRED:\n",
            "        print(f\"⚠️  WARNING: Insufficient space! Required: {MIN_GB_REQUIRED} GB, Available: {free_space_gb:.1f} GB\")\n",
            "        print(\"Please free up space in your Google Drive before proceeding\")\n",
            "    else:\n",
            "        print(f\"✅ Sufficient space available ({free_space_gb:.1f} GB >= {MIN_GB_REQUIRED} GB)\")\n",
            "        \n",
            "except Exception as e:\n",
            "    print(f\"❌ Error checking space: {e}\")"
        ]
    })
    
    # 1.4 Dependency Installation
    notebook_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 4: Dependency Installation\n",
            "# Install required packages\n",
            "!pip install -q torch transformers huggingface-hub biopython numpy pandas scikit-learn requests tqdm\n",
            "!pip install -q fair-esm  # ESMFold implementation\n",
            "\n",
            "# Verify installations\n",
            "import transformers\n",
            "import Bio\n",
            "print(f\"Transformers version: {transformers.__version__}\")\n",
            "print(f\"Biopython version: {Bio.__version__}\")\n",
            "print(\"✅ Dependencies installed successfully\")"
        ]
    })
    
    # Phase 2: Software Installation
    notebook_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🛠️ Phase 2: Software and Model Acquisition"
        ]
    })
    
    # 2.1 Foldseek Installation
    notebook_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 5: Foldseek Installation\n",
            "# Download and install Foldseek\n",
            "!wget -q https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz\n",
            "!tar xzf foldseek-linux-avx2.tar.gz\n",
            "!chmod +x foldseek/bin/foldseek\n",
            "\n",
            "# Add to PATH\n",
            "os.environ['PATH'] += ':/content/foldseek/bin'\n",
            "\n",
            "# Validate installation\n",
            "!foldseek version\n",
            "print(\"✅ Foldseek installed successfully\")"
        ]
    })
    
    # 2.2 ESMFold Model Download
    notebook_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 6: ESMFold Model Download\n",
            "import esm\n",
            "\n",
            "# Load ESMFold model\n",
            "print(\"Downloading ESMFold model (this may take a few minutes)...\")\n",
            "esmfold_model = esm.pretrained.esmfold_v1()\n",
            "esmfold_model = esmfold_model.eval()\n",
            "device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')\n",
            "esmfold_model = esmfold_model.to(device)\n",
            "\n",
            "print(f\"✅ ESMFold model loaded on {device}\")"
        ]
    })
    
    # 2.3 SaProt Model Download
    notebook_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 7: SaProt Model Download\n",
            "from huggingface_hub import hf_hub_download\n",
            "\n",
            "# Download SaProt model\n",
            "print(\"Downloading SaProt model...\")\n",
            "model_path = hf_hub_download(\n",
            "    repo_id=\"westlake-repl/SaProt\",\n",
            "    filename=\"SaProt_650M_AF2.pt\",\n",
            "    cache_dir=\"/content/models\"\n",
            ")\n",
            "\n",
            "print(f\"Model downloaded to: {model_path}\")\n",
            "\n",
            "# Check model size\n",
            "model_size = os.path.getsize(model_path) / 1e9\n",
            "print(f\"Model size: {model_size:.1f} GB\")\n",
            "print(\"✅ SaProt model downloaded successfully\")"
        ]
    })
    
    # Phase 3: Data Processing
    notebook_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🧬 Phase 3: Data Processing and Embedding Generation\n",
            "\n",
            "### User Configuration\n",
            "Upload your `input_sequences.fasta` file before running the next cell."
        ]
    })
    
    # 3.1 Load Sequences
    notebook_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 8: Load Sequences\n",
            "from Bio import SeqIO\n",
            "import pandas as pd\n",
            "from pathlib import Path\n",
            "\n",
            "# Configuration\n",
            "INPUT_FASTA_PATH = 'input_sequences.fasta'\n",
            "MAX_SEQUENCES = 100  # Process first 100 sequences (adjust as needed)\n",
            "\n",
            "# Create structures directory\n",
            "structures_dir = Path('/content/structures')\n",
            "structures_dir.mkdir(exist_ok=True)\n",
            "\n",
            "# Load sequences from FASTA\n",
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
            "print(f\"Loaded {len(df)} sequences from {INPUT_FASTA_PATH}\")\n",
            "print(f\"Processing first {min(MAX_SEQUENCES, len(df))} sequences\")\n",
            "\n",
            "# Limit to MAX_SEQUENCES for manageable processing time\n",
            "df = df.head(MAX_SEQUENCES)\n",
            "print(f\"Sequence length range: {df['length'].min()}-{df['length'].max()} amino acids\")"
        ]
    })
    
    # 3.2 Generate Structures with ESMFold
    notebook_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 9: Generate Structures with ESMFold\n",
            "import time\n",
            "from tqdm import tqdm\n",
            "\n",
            "def predict_structure_esmfold(sequence, seq_id, output_dir):\n",
            "    \"\"\"Predict structure using ESMFold\"\"\"\n",
            "    try:\n",
            "        with torch.no_grad():\n",
            "            output = esmfold_model.infer_pdb(sequence)\n",
            "        \n",
            "        # Save PDB file\n",
            "        output_path = output_dir / f\"{seq_id}.pdb\"\n",
            "        with open(output_path, 'w') as f:\n",
            "            f.write(output)\n",
            "        \n",
            "        return True\n",
            "    except Exception as e:\n",
            "        print(f\"Error predicting structure for {seq_id}: {e}\")\n",
            "        return False\n",
            "\n",
            "# Generate structures\n",
            "print(\"🔄 Generating structures with ESMFold...\")\n",
            "print(\"⚠️  This will take approximately 1-2 minutes per sequence\")\n",
            "\n",
            "successful_predictions = 0\n",
            "start_time = time.time()\n",
            "\n",
            "for idx, row in tqdm(df.iterrows(), total=len(df)):\n",
            "    if predict_structure_esmfold(row['sequence'], row['id'], structures_dir):\n",
            "        successful_predictions += 1\n",
            "\n",
            "end_time = time.time()\n",
            "total_time = end_time - start_time\n",
            "\n",
            "print(f\"\\n✅ Generated {successful_predictions}/{len(df)} structures\")\n",
            "print(f\"⏱️  Total time: {total_time/60:.1f} minutes\")\n",
            "print(f\"📊 Average time per structure: {total_time/len(df):.1f} seconds\")"
        ]
    })
    
    # 3.3 Generate 3Di Sequences
    notebook_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 10: Generate 3Di Sequences\n",
            "import subprocess\n",
            "\n",
            "def generate_3di_sequence(pdb_file):\n",
            "    \"\"\"Generate 3Di sequence from PDB file using Foldseek\"\"\"\n",
            "    try:\n",
            "        db_name = pdb_file.stem\n",
            "        \n",
            "        # Create database\n",
            "        subprocess.run(['foldseek', 'createdb', str(pdb_file), f'{db_name}_db'], \n",
            "                      capture_output=True, check=True)\n",
            "        \n",
            "        # Convert to 3Di\n",
            "        subprocess.run(['foldseek', 'structureto3di', f'{db_name}_db', f'{db_name}_3di_db'], \n",
            "                      capture_output=True, check=True)\n",
            "        \n",
            "        # Create sequence file\n",
            "        subprocess.run(['foldseek', 'createseqfiledb', f'{db_name}_3di_db', f'{db_name}_3di_seq'], \n",
            "                      capture_output=True, check=True)\n",
            "        \n",
            "        # Read 3Di sequence\n",
            "        with open(f'{db_name}_3di_seq', 'r') as f:\n",
            "            lines = f.readlines()\n",
            "            if len(lines) >= 2:\n",
            "                return lines[1].strip()\n",
            "        return None\n",
            "    except Exception as e:\n",
            "        print(f\"Error processing {pdb_file}: {e}\")\n",
            "        return None\n",
            "\n",
            "# Process structures\n",
            "pdb_files = list(structures_dir.glob('*.pdb'))\n",
            "print(f\"Processing {len(pdb_files)} PDB files...\")\n",
            "\n",
            "sequences_3di = {}\n",
            "for pdb_file in tqdm(pdb_files):\n",
            "    seq_3di = generate_3di_sequence(pdb_file)\n",
            "    if seq_3di:\n",
            "        sequences_3di[pdb_file.stem] = seq_3di\n",
            "\n",
            "print(f\"\\n✅ Generated {len(sequences_3di)} 3Di sequences\")\n",
            "\n",
            "# Save 3Di sequences to FASTA\n",
            "with open('generated_3di_sequences.fasta', 'w') as f:\n",
            "    for seq_id, seq_3di in sequences_3di.items():\n",
            "        f.write(f\">{seq_id}\\n{seq_3di}\\n\")\n",
            "\n",
            "print(\"✅ 3Di sequences saved to generated_3di_sequences.fasta\")"
        ]
    })
    
    # 3.4 Load SaProt Model
    notebook_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 11: Load SaProt Model\n",
            "from transformers import AutoTokenizer, AutoModel\n",
            "import numpy as np\n",
            "\n",
            "# Load SaProt model and tokenizer\n",
            "print(\"Loading SaProt model...\")\n",
            "model_name = \"westlake-repl/SaProt\"\n",
            "tokenizer = AutoTokenizer.from_pretrained(model_name)\n",
            "saprot_model = AutoModel.from_pretrained(model_name)\n",
            "\n",
            "# Move model to GPU\n",
            "device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')\n",
            "saprot_model = saprot_model.to(device)\n",
            "saprot_model.eval()\n",
            "\n",
            "print(f\"✅ SaProt model loaded on {device}\")\n",
            "print(f\"Model parameters: {sum(p.numel() for p in saprot_model.parameters()):,}\")"
        ]
    })
    
    # 3.5 Generate SaProt Embeddings
    notebook_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 12: Generate SaProt Embeddings\n",
            "# Configuration\n",
            "BATCH_SIZE = 32  # Adjusted for better memory management\n",
            "\n",
            "def generate_saprot_embeddings(sequences, batch_size=BATCH_SIZE):\n",
            "    \"\"\"Generate SaProt embeddings for sequences with optimal batch processing\"\"\"\n",
            "    embeddings = []\n",
            "    \n",
            "    # Clear GPU memory\n",
            "    torch.cuda.empty_cache()\n",
            "    \n",
            "    for i in range(0, len(sequences), batch_size):\n",
            "        batch = sequences[i:i+batch_size]\n",
            "        print(f\"Processing batch {i//batch_size + 1}/{(len(sequences)-1)//batch_size + 1}\")\n",
            "        \n",
            "        # Tokenize batch\n",
            "        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)\n",
            "        inputs = {k: v.to(device) for k, v in inputs.items()}\n",
            "        \n",
            "        # Generate embeddings\n",
            "        with torch.no_grad():\n",
            "            outputs = saprot_model(**inputs)\n",
            "            # Use mean pooling\n",
            "            batch_embeddings = outputs.last_hidden_state.mean(dim=1)\n",
            "            embeddings.append(batch_embeddings.cpu().numpy())\n",
            "    \n",
            "    return np.vstack(embeddings)\n",
            "\n",
            "# Prepare 3Di sequences for embedding\n",
            "print(f\"Loaded {len(sequences_3di)} 3Di sequences\")\n",
            "\n",
            "# Generate embeddings\n",
            "if sequences_3di:\n",
            "    print(\"🔄 Generating SaProt embeddings...\")\n",
            "    \n",
            "    # Measure inference time\n",
            "    start_time = time.time()\n",
            "    \n",
            "    seq_list = list(sequences_3di.values())\n",
            "    saprot_embeddings = generate_saprot_embeddings(seq_list)\n",
            "    \n",
            "    end_time = time.time()\n",
            "    inference_time = end_time - start_time\n",
            "    \n",
            "    # Measure GPU memory\n",
            "    peak_memory = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0\n",
            "    \n",
            "    print(f\"\\n✅ Generated embeddings shape: {saprot_embeddings.shape}\")\n",
            "    print(f\"⏱️  Total inference time: {inference_time:.2f} seconds\")\n",
            "    print(f\"🧠 Peak GPU memory allocated: {peak_memory:.2f} GB\")\n",
            "    print(f\"📊 Average time per sequence: {inference_time/len(seq_list):.3f} seconds\")\n",
            "    \n",
            "    # Save embeddings\n",
            "    np.save('/content/saprot_embeddings.npy', saprot_embeddings)\n",
            "    print(\"✅ Embeddings saved to saprot_embeddings.npy\")\n",
            "else:\n",
            "    print(\"❌ No 3Di sequences available for embedding generation\")"
        ]
    })
    
    # Phase 4: Results Export
    notebook_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 💾 Phase 4: Results Export"
        ]
    })
    
    # 4.1 Save to Google Drive
    notebook_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 13: Save to Google Drive\n",
            "# Create output directory\n",
            "output_dir = '/content/drive/MyDrive/SaProt_Results'\n",
            "os.makedirs(output_dir, exist_ok=True)\n",
            "\n",
            "# Save embeddings\n",
            "embedding_file = f\"{output_dir}/saprot_embeddings.npy\"\n",
            "np.save(embedding_file, saprot_embeddings)\n",
            "\n",
            "# Save sequence information\n",
            "sequence_info_file = f\"{output_dir}/sequence_info.csv\"\n",
            "df.to_csv(sequence_info_file, index=False)\n",
            "\n",
            "# Save 3Di sequences\n",
            "if sequences_3di:\n",
            "    with open(f\"{output_dir}/generated_3di_sequences.fasta\", 'w') as f:\n",
            "        for seq_id, seq_3di in sequences_3di.items():\n",
            "            f.write(f\">{seq_id}\\n{seq_3di}\\n\")\n",
            "\n",
            "# Verify file sizes\n",
            "embedding_size = os.path.getsize(embedding_file) / 1e6\n",
            "print(f\"✅ Results saved to Google Drive\")\n",
            "print(f\"Embedding file size: {embedding_size:.1f} MB\")\n",
            "print(f\"Files saved to: {output_dir}\")"
        ]
    })
    
    # 4.2 Download Instructions
    notebook_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 📥 Download Instructions\n",
            "\n",
            "### Files Generated\n",
            "The following files have been saved to your Google Drive:\n",
            "\n",
            "1. **saprot_embeddings.npy** - SaProt embeddings (main output)\n",
            "2. **sequence_info.csv** - Sequence information and metadata\n",
            "3. **generated_3di_sequences.fasta** - 3Di sequences used for embedding generation\n",
            "\n",
            "### How to Download\n",
            "\n",
            "#### Option 1: Google Drive Web Interface\n",
            "1. Go to [Google Drive](https://drive.google.com)\n",
            "2. Navigate to the `SaProt_Results` folder\n",
            "3. Right-click on each file and select \"Download\"\n",
            "\n",
            "#### Option 2: Colab File Browser\n",
            "1. In the Colab interface, click on the folder icon in the left sidebar\n",
            "2. Navigate to `/content/drive/MyDrive/SaProt_Results/`\n",
            "3. Right-click on each file and select \"Download\"\n",
            "\n",
            "### Next Steps for Local Evaluation\n",
            "1. Download the embedding file (`saprot_embeddings.npy`)\n",
            "2. Download the sequence information (`sequence_info.csv`)\n",
            "3. Use these files to train your XGBoost model locally\n",
            "4. Compare performance with ESM2 embeddings\n",
            "5. Analyze feature importance and interpretability\n",
            "\n",
            "### File Locations\n",
            "- **Google Drive**: `/content/drive/MyDrive/SaProt_Results/`\n",
            "- **Local Colab**: `/content/saprot_embeddings.npy`"
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
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            },
            "accelerator": "GPU"
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    return notebook

def save_esmfold_notebook():
    """Save the ESMFold notebook to file"""
    import json
    
    notebook = generate_esmfold_notebook()
    
    with open('colab_saprot_esmfold.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print("✅ ESMFold Colab notebook saved as colab_saprot_esmfold.ipynb")

if __name__ == "__main__":
    save_esmfold_notebook()
