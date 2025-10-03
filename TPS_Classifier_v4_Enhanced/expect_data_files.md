# Expected Data Files for Complete TPS Pipeline

Please provide the following files to complete the evaluation system:

## Required Files

### 1. Trained Model Weights
- **File**: `models/checkpoints/complete_multimodal_best.pth`
- **Description**: Your trained multimodal classifier weights
- **Format**: PyTorch state_dict (`.pth`)

### 2. Training Data
- **File**: `data/train.fasta` 
- **Format**: Standard FASTA format with sequence IDs
- **Example**:
```
>ID1
MKVFLILLFSLAASGLAEYAVLQVEQKLQGQSEKLLQHLENKTKTLQNQELQSRLDQLLDTAKK...
>ID2
MKTFLILLFSLAASGLAEYAVLQVEQKLQGQSEKLLQHLENKTKTLQNQELQSRLDQLLDTAKK...
```

- **File**: `data/train_labels.csv`
- **Format**: CSV with ID, class pairs
- **Example**:
```csv
ID1,Germacrene_A
ID2,Linalool
ID3,Pinene
```

### 3. Validation Data
- **File**: `data/val.fasta`
- **Format**: Standard FASTA format (same structure as train.fasta)

- **File**: `data/val_labels_binary.csv` 
- **Format**: CSV with binary labels matching the label order
- **Columns**: ID,Germacrene_A,Germacrene_C,Germacrene_D,Linalool,Limonene,Myrcene,Pinene,Sabinene,Terpinolene,Beta_caryophyllene,Alpha_humulene,Beta_farnesene,Sesquiterpene_general,Monoterpene_general,Terpene_synthase_general
- **Example**:
```csv
VAL001,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0
VAL002,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0
```

## Optional Files (Highly Recommended)

### External 30-Sequence Holdout
- **File**: `data/external_30.fasta`
- **Format**: Standard FASTA format (30 sequences for final validation)

- **File**: `data/external_30_labels_binary.csv`
- **Format**: Binary labels (same format as validation labels)

## Instructions

1. Place the model weights file: `models/checkpoints/complete_multimodal_best.pth`
2. Place training data files: `data/train.fasta` and `data/train_labels.csv`  
3. Place validation data files: `data/val.fasta` and `data/val_labels_binary.csv`
4. (Optional) Place external holdout files: `data/external_30.fasta` and `data/external_30_labels_binary.csv`
5. Run: `./create_complete_bundle.sh`

This will create a complete bundle with all the training data and model weights ready for end-to-end evaluation.


