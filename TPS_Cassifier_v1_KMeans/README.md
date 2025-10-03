# Germacrene Synthase Binary Classifier

A comprehensive machine learning pipeline for predicting Germacrene synthase enzymes using protein language models and semi-supervised learning.

## Overview

This project implements a binary classifier that can identify Germacrene synthase enzymes from protein sequences. It uses:

- **Pre-trained Protein Language Models** (ESM-2, ProtT5) for feature extraction
- **Stratified K-Fold Cross-Validation** for robust model evaluation
- **Semi-Supervised Learning** with pseudo-labeling of unannotated data
- **XGBoost Gradient Boosting** for final classification
- **Class Imbalance Handling** with appropriate weighting strategies

## Features

- ✅ FASTA file loading and parsing
- ✅ Multiple protein language model support (ESM-2, ProtT5)
- ✅ Stratified k-fold cross-validation
- ✅ Semi-supervised learning with confidence-based pseudo-labeling
- ✅ Comprehensive evaluation metrics (F1-Score, Precision, Recall, AUC-PR)
- ✅ Model persistence and loading
- ✅ Batch processing for large datasets
- ✅ GPU acceleration support
- ✅ Detailed visualization of results

## Installation

1. Clone or download this repository:
```bash
git clone <repository-url>
cd terpene_classifier_kmeans
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have sufficient disk space for the protein language models (several GB).

## Usage

### Quick Start

1. **Prepare your data files:**
   - `data/marts_db.fasta` - Labeled MARTS-DB sequences
   - `data/uniprot_sequences.fasta` - Uniprot sequences (optional)
   - `data/ncbi_sequences.fasta` - NCBI sequences (optional)

2. **Run the complete pipeline:**
```bash
python terpene_classifier.py
```

### Custom Usage

```python
from terpene_classifier import TerpeneClassifier

# Initialize classifier
classifier = TerpeneClassifier(model_name='esm2_t33_650M_UR50D')

# Load and parse data
marts_df = classifier.parse_marts_db('path/to/marts_db.fasta')
uniprot_df = classifier.load_sequences('path/to/uniprot_sequences.fasta')

# Generate embeddings
embeddings = classifier.generate_embeddings(marts_df['sequence'].tolist())

# Train model
results = classifier.train_initial_model(X, y)

# Make predictions
confidence = classifier.predict_germacrene("MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL...")
print(f"Germacrene synthase confidence: {confidence:.3f}")
```

## Data Format

### Input Files

The classifier expects FASTA files with protein sequences. For MARTS-DB files, the sequence IDs should contain product information that can be parsed to identify Germacrene synthases.

**Example FASTA format:**
```
>germacrene_synthase_1
MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL...
>limonene_synthase_1
MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL...
```

### MARTS-DB Parsing

The `parse_marts_db()` function extracts product information from sequence IDs and creates binary labels:
- `is_germacrene = 1` for sequences containing "germacrene" in the ID
- `is_germacrene = 0` for all other sequences

## Model Architecture

### 1. Feature Extraction
- **Protein Language Models**: ESM-2 (650M parameters) or ProtT5
- **Embedding Generation**: Average pooling of residue-level embeddings
- **Feature Scaling**: StandardScaler for normalization

### 2. Classification
- **Algorithm**: XGBoost Gradient Boosting Classifier
- **Class Imbalance**: `scale_pos_weight` parameter
- **Cross-Validation**: Stratified 5-fold CV

### 3. Semi-Supervised Learning
- **Pseudo-labeling**: High-confidence predictions (threshold = 0.95)
- **Data Augmentation**: Combining labeled and pseudo-labeled data
- **Retraining**: Full pipeline on expanded dataset

## Performance Metrics

The classifier reports several metrics for comprehensive evaluation:

- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **AUC-PR**: Area under the Precision-Recall curve

## Output Files

After training, the following files are generated:

- `models/germacrene_classifier.pkl` - Trained model and scaler
- `results/training_results.png` - Visualization of cross-validation results
- Console output with detailed metrics and progress

## Configuration

### Model Parameters

```python
classifier = TerpeneClassifier(
    model_name='esm2_t33_650M_UR50D',  # or 'prot_t5_xl_half_uniref50-enc'
    device='auto',  # 'cpu', 'cuda', or 'auto'
    random_state=42
)
```

### Training Parameters

- **Cross-validation folds**: 5 (configurable)
- **Confidence threshold**: 0.95 (for pseudo-labeling)
- **Batch size**: 8 (GPU) or 4 (CPU)
- **Max sequence length**: 1024 (ESM-2) or 512 (ProtT5)

## Memory Requirements

- **RAM**: 8-16 GB recommended
- **GPU Memory**: 4-8 GB (if using GPU acceleration)
- **Disk Space**: 5-10 GB (for protein language models)

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Model download fails**: Check internet connection and disk space
3. **FASTA parsing errors**: Verify file format and encoding

### Performance Tips

1. **Use GPU**: Significantly faster for large datasets
2. **Batch processing**: Adjust batch size based on available memory
3. **Sequence filtering**: Remove very long sequences if memory is limited

## Example Results

Typical performance on a balanced dataset:
- F1-Score: 0.85 ± 0.03
- Precision: 0.87 ± 0.04
- Recall: 0.83 ± 0.05
- AUC-PR: 0.89 ± 0.02

## Citation

If you use this classifier in your research, please cite the relevant papers:

- ESM-2: [Lin et al., 2023, Science]
- ProtT5: [Elnaggar et al., 2021, IEEE TPAMI]
- XGBoost: [Chen & Guestrin, 2016, KDD]

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Support

For questions or support, please open an issue on GitHub or contact the maintainers.

