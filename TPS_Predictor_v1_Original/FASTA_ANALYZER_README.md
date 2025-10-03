# FASTA Sequence Analyzer for Germacrene Synthase Prediction

This tool analyzes FASTA files containing protein sequences and predicts whether each sequence is a Germacrene A synthase, Germacrene D synthase, or other type of synthase.

## Features

- **Three-class prediction**: Germacrene A, Germacrene D, or Other
- **High accuracy**: 89% F1-score on test data
- **Confidence scores**: Each prediction includes confidence level
- **CSV output**: Results saved in easy-to-use CSV format
- **Batch processing**: Analyze multiple sequences at once

## Usage

### Basic Usage
```bash
python3 fasta_analyzer.py your_sequences.fasta
```

### Specify Output File
```bash
python3 fasta_analyzer.py your_sequences.fasta -o results.csv
```

### Use Custom Model
```bash
python3 fasta_analyzer.py your_sequences.fasta -m custom_model.joblib
```

## Input Requirements

- **FASTA format**: Standard FASTA file with protein sequences
- **Valid amino acids**: Only standard 20 amino acids (ACDEFGHIKLMNPQRSTVWY)
- **Minimum length**: Sequences must be at least 50 amino acids long
- **No gaps**: Sequences should not contain gaps or special characters

## Output Files

The tool generates two CSV files:

### 1. Basic CSV (`filename_germacrene_predictions.csv`)
Contains the three columns you requested:
- `fasta_name`: The sequence identifier from FASTA header
- `sequence`: The protein sequence
- `prediction`: Model prediction (Germacrene A Synthase, Germacrene D Synthase, or Other Synthase)

### 2. Detailed CSV (`filename_germacrene_predictions_detailed.csv`)
Contains additional information:
- All basic columns plus:
- `confidence`: Prediction confidence (0-1)
- `prob_germacrene_a`: Probability of being Germacrene A synthase
- `prob_germacrene_d`: Probability of being Germacrene D synthase
- `prob_other`: Probability of being other synthase
- `sequence_length`: Length of the sequence

## Example

### Input FASTA file (`sequences.fasta`):
```
>synthase_1
MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELGERMACA
>synthase_2
MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELGERMACD
>synthase_3
MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELDDXXD
```

### Output CSV (`sequences_germacrene_predictions.csv`):
```csv
fasta_name,sequence,prediction
synthase_1,MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELGERMACA,Germacrene A Synthase
synthase_2,MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELGERMACD,Germacrene D Synthase
synthase_3,MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELDDXXD,Other Synthase
```

## Model Performance

- **Accuracy**: 89.0%
- **F1-score**: 89.1%
- **Precision**: 89.8%
- **Recall**: 89.0%

## Requirements

- Python 3.7+
- Required packages: pandas, numpy, scikit-learn, torch, transformers, biopython
- Trained model file: `germacrene_predictor.joblib`

## Troubleshooting

### Model Not Found
If you get "Model file not found", run:
```bash
python3 germacrene_predictor.py
```
This will train and save the model.

### Invalid Amino Acids
Sequences with non-standard amino acids (X, B, Z, etc.) will be skipped. Use only standard 20 amino acids.

### Short Sequences
Sequences shorter than 50 amino acids will be skipped as they may not be reliable for prediction.

## Notes

- The model uses ESM2 protein language model for sequence embeddings
- Predictions are based on protein sequence similarity and learned patterns
- High confidence predictions (>90%) are generally more reliable
- The model was trained on synthetic data and may need validation on real sequences
