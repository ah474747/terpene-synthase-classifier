# Terpene Synthase Product Predictor v2

A state-of-the-art machine learning system for predicting terpene products from synthase protein sequences using protein language models and molecular fingerprints.

## 🎯 Objective

Build a robust model that predicts specific terpene products (limonene, pinene, myrcene, etc.) synthesized by terpene synthases using:
- **SaProt** protein language model for sequence embeddings
- **Molecular fingerprints** for terpene product representation
- **Attention mechanisms** for interpretable predictions
- **Multi-class classification** for diverse terpene products

## 🏗️ Architecture

### Data Sources
- **BRENDA Database**: Curated enzyme data with verified products
- **UniProt**: Manual curation pipeline for sequence annotation
- **Literature Mining**: Extract verified products from research papers

### Model Components
- **SaProt Encoder**: Domain-specific protein embeddings
- **Molecular Encoder**: SMILES → RDKit fingerprints
- **Attention Classifier**: Multi-class prediction with interpretability
- **Ensemble Model**: Combine multiple approaches for robustness

### Key Features
- **Balanced Training**: Handle class imbalance properly
- **Cross-Validation**: Robust evaluation on real biological data
- **Biological Validation**: Test on unseen organisms
- **Uncertainty Quantification**: Confidence scores for predictions

## 📁 Project Structure

```
terpene_predictor_v2/
├── data/                    # Data collection and curation
├── models/                  # Model architectures
├── training/               # Training pipelines
├── evaluation/             # Evaluation and validation
├── utils/                  # Utility functions
├── config/                 # Configuration files
└── requirements.txt        # Dependencies
```

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Collect Data**:
   ```bash
   python data/brenda_collector.py
   python data/uniprot_curator.py
   ```

3. **Train Model**:
   ```bash
   python training/train_saprot.py
   ```

4. **Evaluate**:
   ```bash
   python evaluation/biological_validator.py
   ```

## 🔬 Research Foundation

This project implements state-of-the-art methods validated by recent research:
- Protein language models for enzyme-substrate prediction
- Molecular fingerprint integration for chemical structure
- Attention mechanisms for biological interpretability
- Multi-class classification for diverse enzyme products

## 📊 Expected Performance

- **Accuracy**: >85% on held-out test sets
- **Generalization**: Robust across different organisms
- **Interpretability**: Attention maps showing important sequence regions
- **Confidence**: Uncertainty quantification for predictions

## 🤝 Contributing

This is a research-grade implementation following best practices in:
- Data curation and quality control
- Model architecture and training
- Biological validation and evaluation
- Reproducible research standards
