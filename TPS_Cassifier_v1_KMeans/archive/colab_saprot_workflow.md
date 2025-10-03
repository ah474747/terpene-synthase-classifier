# SaProt Embeddings Workflow in Google Colab

## Cursor AI Prompt: Protein Embeddings Workflow in Colab

### Persona and Goal
You are an expert bioinformatician and Python developer. Your task is to generate SaProt structural embeddings for a list of protein sequences provided in a local input_sequences.fasta file. The entire workflow must be executed within a Google Colab environment utilizing its free GPU runtime for accelerated performance.

### Step 1: Colab Environment Setup and Validation
Create a new Google Colab notebook. The very first executable cell should contain the following instructions and validation checks.

**Select GPU Runtime**: Explicitly document the requirement to select a GPU runtime (e.g., T4 GPU) from the Notebook Settings.

**Environment Check**: Write a Python code snippet that checks and prints the available CUDA/GPU details to confirm the GPU is active and correctly configured.

**Dependency Installation**: Generate a cell using `!pip install` to install all necessary Python libraries for SaProt and general data handling (e.g., torch, transformers, huggingface-hub, and any specific dependencies noted in the SaProt GitHub repository's requirements.txt). Use the `-q` flag for a quiet installation.

### Step 2: Install and Configure SaProt Software
In the subsequent cells, generate the code to download and set up the SaProt repository and its structural dependencies.

**Clone Repository**: Use `!git clone` to clone the SaProt GitHub repository into the Colab environment.

**Navigate and Setup**: Generate the necessary commands to navigate into the cloned directory and run any required setup/installation shell scripts or configuration (e.g., `!bash environment.sh` if provided, or similar setup).

**Model Download**: Include a command/script to download the required SaProt model checkpoint (e.g., SaProt_650M_AF2) from Hugging Face or another specified source into a designated local directory, ensuring the model weights are present for the next step.

### Step 3: Data Upload and Embedding Generation
Assume the user has uploaded the required input_sequences.fasta file to the Colab session's root directory.

**Embedding Script**: Write a final Python script that:

1. Imports the necessary SaProt components (model and alphabet loader).
2. Loads the pre-trained SaProt model checkpoint onto the available GPU device.
3. Loads the sequences from the input_sequences.fasta file.
4. Generates the structural embeddings for all sequences.
5. Saves the resulting embeddings (e.g., as a compressed NumPy array, .npy, or a pickled object) to a file named saprot_embeddings.npy in the Colab environment.

**Data Download Instruction**: Provide a final markdown or print statement that instructs the user to download the saprot_embeddings.npy file from the Colab file browser, as this is the final output.

## Enhanced Workflow for Germacrene Synthase Classification

### Additional Considerations for Our Specific Use Case

#### Step 0: Pre-Colab Preparation (Local MacBook Pro)
1. **Dataset Preparation**: Ensure the expanded_germacrene_dataset.fasta file is ready
2. **Google Drive Setup**: Create a dedicated folder for the project
3. **Structure Planning**: Plan the workflow to minimize data transfer

#### Step 1: Enhanced Colab Environment Setup
```python
# GPU Runtime Selection
# Go to Runtime → Change runtime type → Hardware accelerator → GPU

# Environment Validation
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "No GPU")

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

#### Step 2: Foldseek Installation and 3Di Generation
```python
# Install Foldseek
!wget https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz
!tar xvzf foldseek-linux-avx2.tar.gz
!chmod +x foldseek/bin/foldseek

# Add to PATH
import os
os.environ['PATH'] += ':/content/foldseek/bin'
```

#### Step 3: SaProt Installation and Configuration
```python
# Install SaProt dependencies
!pip install -q torch transformers huggingface-hub biopython

# Clone SaProt repository
!git clone https://github.com/mingkangyang/SaProt.git
%cd SaProt

# Download model
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(repo_id="westlake-repl/SaProt", filename="saprot_650M.pt")
```

#### Step 4: Structure Acquisition and 3Di Generation
```python
# Download structures from AlphaFold Database
# Process PDB files to generate 3Di sequences
# Save 3Di sequences for SaProt input
```

#### Step 5: SaProt Embedding Generation
```python
# Load SaProt model on GPU
# Process 3Di sequences in batches
# Generate structure-aware embeddings
# Save embeddings to Google Drive
```

#### Step 6: Local Evaluation (MacBook Pro)
```python
# Download embeddings from Google Drive
# Train XGBoost models with SaProt embeddings
# Compare with ESM2 embeddings
# Evaluate performance metrics
```

## Implementation Order

### Phase 1: Environment Setup and Validation
1. **Colab Setup**: Create notebook, select GPU runtime
2. **Environment Check**: Validate CUDA/GPU availability
3. **Dependency Installation**: Install required packages
4. **Google Drive Mount**: Set up data access

### Phase 2: Software Installation
1. **Foldseek Installation**: Install structural analysis tool
2. **SaProt Installation**: Clone repository and download model
3. **Validation**: Test installations with sample data

### Phase 3: Data Processing
1. **Structure Acquisition**: Download from AlphaFold Database
2. **3Di Generation**: Use Foldseek to create structural tokens
3. **Embedding Generation**: Use SaProt to create embeddings
4. **Data Export**: Save results to Google Drive

### Phase 4: Local Evaluation
1. **Data Download**: Retrieve embeddings from Google Drive
2. **Model Training**: Train XGBoost with SaProt embeddings
3. **Comparative Analysis**: Compare with ESM2 performance
4. **Results Analysis**: Evaluate improvements and interpretability

## Key Benefits of This Approach

### Computational Efficiency
- **GPU Acceleration**: 10-100x speedup for SaProt inference
- **Cloud Resources**: Offload intensive computations
- **Local Evaluation**: Keep model training on local machine

### Resource Management
- **Minimal Local Impact**: Reduce MacBook Pro resource usage
- **Scalable Processing**: Handle large datasets efficiently
- **Cost Effective**: Utilize free Colab GPU resources

### Workflow Optimization
- **Sequential Processing**: Logical order of operations
- **Validation Steps**: Ensure each step works before proceeding
- **Error Handling**: Robust pipeline with fallback options

## Expected Timeline

### Week 1: Environment Setup
- Colab notebook creation and configuration
- Software installation and validation
- Initial testing with sample data

### Week 2: Data Processing
- Structure acquisition from AlphaFold Database
- 3Di sequence generation with Foldseek
- SaProt embedding generation

### Week 3: Model Training and Evaluation
- Download embeddings to local machine
- Train XGBoost models with SaProt embeddings
- Comparative analysis with ESM2

### Week 4: Results Analysis and Optimization
- Performance evaluation and statistical testing
- Feature importance analysis
- Final model selection and deployment

## Success Metrics

### Technical Metrics
- **Embedding Generation**: Successful processing of all 1,373 sequences
- **Model Performance**: F1-score > 0.70 (current: 0.59)
- **Computational Efficiency**: Significant speedup over local processing

### Comparative Metrics
- **SaProt vs ESM2**: Statistical significance testing
- **Feature Importance**: Interpretability analysis
- **Generalization**: Cross-validation performance

## Risk Mitigation

### Technical Risks
- **GPU Availability**: Free Colab GPU limits
- **Model Size**: Memory constraints for large models
- **Data Transfer**: Large file upload/download times

### Mitigation Strategies
- **Batch Processing**: Process sequences in manageable batches
- **Alternative Resources**: Consider Colab Pro for better GPUs
- **Local Fallback**: ESM2 as backup option

## Next Steps

1. **Create Colab Notebook**: Set up the initial environment
2. **Install Dependencies**: Configure all required software
3. **Validate Environment**: Test with sample data
4. **Begin Data Processing**: Start with structure acquisition
5. **Generate Embeddings**: Use SaProt for structure-aware features
6. **Evaluate Performance**: Compare with existing ESM2 model

This comprehensive workflow ensures efficient, scalable, and robust implementation of SaProt embeddings for germacrene synthase classification.
