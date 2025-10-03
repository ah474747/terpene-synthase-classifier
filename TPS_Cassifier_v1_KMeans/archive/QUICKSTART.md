# Quick Start Guide

## 1. Installation

```bash
# Option 1: Use the installation script
./install.sh

# Option 2: Manual installation
pip install -r requirements.txt
python3 setup.py
```

## 2. Prepare Your Data

Place your FASTA files in the `data/` directory:
- `data/marts_db.fasta` - Labeled MARTS-DB sequences (required)
- `data/uniprot_sequences.fasta` - Uniprot sequences (optional)
- `data/ncbi_sequences.fasta` - NCBI sequences (optional)

## 3. Train the Model

```bash
python3 terpene_classifier.py
```

This will:
- Load and parse your data
- Generate protein embeddings
- Train the classifier with cross-validation
- Apply semi-supervised learning
- Save the trained model

## 4. Make Predictions

```bash
# Predict from FASTA file
python3 predict.py data/sample_sequences.fasta

# Predict single sequence
python3 predict.py -s "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL"

# Interactive mode
python3 predict.py --interactive
```

## 5. Test Everything Works

```bash
python3 test_classifier.py
```

## Expected Output

After training, you should see:
- Cross-validation results with F1-scores, precision, recall, and AUC-PR
- A trained model saved to `models/germacrene_classifier.pkl`
- Results visualization in `results/training_results.png`

## Troubleshooting

- **Memory issues**: Reduce batch size in the code or use CPU instead of GPU
- **Model download fails**: Check internet connection and disk space
- **FASTA parsing errors**: Verify file format and sequence IDs contain product information

## Need Help?

Check the full README.md for detailed documentation and examples.

