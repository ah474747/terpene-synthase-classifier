# Terpene Synthase Product Predictor

A machine learning pipeline for predicting terpene synthase enzyme products from protein sequence data. This project collects protein sequences from UniProt and NCBI databases, extracts various features, and trains multiple machine learning models to predict the products of terpene synthase enzymes.

## Features

- **Data Collection**: Automated collection of terpene synthase protein sequences from UniProt and NCBI
- **Feature Extraction**: Comprehensive feature extraction including:
  - Amino acid composition
  - Physicochemical properties
  - K-mer frequencies
  - Motif patterns
  - Secondary structure predictions
- **Multiple ML Models**: Support for Random Forest, XGBoost, LightGBM, SVM, Logistic Regression, and Neural Networks
- **Hyperparameter Tuning**: Automated hyperparameter optimization
- **Model Evaluation**: Comprehensive evaluation with visualizations
- **Product Prediction**: Predict products for new protein sequences

## Installation

1. Clone or download this repository
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Basic Usage

Run the complete pipeline with default settings:

```bash
python main.py --email your_email@example.com
```

### 2. Custom Configuration

```bash
# Collect more data
python main.py --email your_email@example.com --uniprot-limit 1000 --ncbi-limit 1000

# Skip data collection and only train models
python main.py --skip-data --email your_email@example.com

# Evaluate model performance
python main.py --email your_email@example.com --evaluate
```

### 3. Individual Components

#### Data Collection
```python
from data_collector import TerpeneSynthaseDataCollector

collector = TerpeneSynthaseDataCollector(email="your_email@example.com")
proteins = collector.search_uniprot_terpene_synthases(limit=500)
annotated_proteins = collector.extract_product_annotations(proteins)
collector.save_data(annotated_proteins)
```

#### Feature Extraction
```python
from feature_extractor import ProteinFeatureExtractor

extractor = ProteinFeatureExtractor()
sequences = ["MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL"]
features_df = extractor.extract_all_features(sequences)
```

#### Model Training
```python
from model_trainer import TerpeneSynthasePredictor

trainer = TerpeneSynthasePredictor()
results = trainer.train_models(X, y)
```

## Project Structure

```
terpene_synthase_predictor/
├── main.py                 # Main pipeline script
├── data_collector.py       # Data collection from databases
├── feature_extractor.py    # Protein sequence feature extraction
├── model_trainer.py        # Machine learning model training
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── data/                  # Data directory (created automatically)
│   ├── terpene_synthase_data.json
│   ├── protein_features.csv
│   └── product_annotations.csv
└── models/                # Trained models (created automatically)
    └── *.pkl
```

## Data Sources

- **UniProt**: Comprehensive protein sequence database with functional annotations
- **NCB**: National Center for Biotechnology Information protein database

## Feature Types

### 1. Basic Sequence Features
- Molecular weight
- Isoelectric point
- Instability index
- Aromaticity
- Grand average of hydropathy (GRAVY)

### 2. Amino Acid Composition
- Percentage of each amino acid in the sequence
- 20 features (one for each standard amino acid)

### 3. Physicochemical Properties
- Average hydrophobicity, charge, size, and polarity
- Standard deviation and range of properties

### 4. K-mer Features
- Frequency of k-mers (subsequences of length k)
- K-mer diversity and entropy
- Supports k=2, 3, 4

### 5. Motif Features
- Presence of known terpene synthase motifs:
  - DDXXD (metal binding)
  - NSE (metal binding)
  - RRX8W (substrate binding)
  - GXGXXG (ATP binding)
  - HXXXH (metal binding)

### 6. Secondary Structure Features
- Helix and sheet propensity scores
- Based on amino acid propensities

## Machine Learning Models

The pipeline supports multiple machine learning algorithms:

1. **Random Forest**: Ensemble method with good interpretability
2. **XGBoost**: Gradient boosting with high performance
3. **LightGBM**: Fast gradient boosting framework
4. **SVM**: Support Vector Machine with RBF kernel
5. **Logistic Regression**: Linear model with regularization
6. **Neural Network**: Multi-layer perceptron

## Model Evaluation

The pipeline provides comprehensive model evaluation:

- **Accuracy**: Overall prediction accuracy
- **Cross-validation**: 5-fold cross-validation scores
- **Feature Importance**: Most important features for tree-based models
- **Confusion Matrix**: Detailed prediction breakdown
- **Classification Report**: Precision, recall, and F1-scores

## Hyperparameter Tuning

Automated hyperparameter optimization using GridSearchCV:

- **Random Forest**: n_estimators, max_depth, min_samples_split, min_samples_leaf
- **XGBoost**: n_estimators, max_depth, learning_rate, subsample
- **LightGBM**: n_estimators, max_depth, learning_rate, num_leaves

## Usage Examples

### Predict Products for New Sequences

```python
from main import TerpeneSynthasePipeline

# Initialize pipeline
pipeline = TerpeneSynthasePipeline(email="your_email@example.com")

# New sequences to predict
new_sequences = [
    "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL",
    "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL"
]

# Make predictions
predictions = pipeline.predict_products(new_sequences, model_name='Random Forest')

for pred in predictions:
    print(f"Sequence: {pred['sequence_id']}")
    print(f"Predicted Product: {pred['predicted_product']}")
    print(f"Confidence: {pred['confidence']:.4f}")
    print()
```

### Evaluate Model Performance

```python
# Evaluate specific model
pipeline.evaluate_model_performance('Random Forest')

# Get feature importance
importance_df = pipeline.model_trainer.get_feature_importance('Random Forest', top_n=20)
print(importance_df)
```

## Configuration

### Environment Variables

- Set your email address for NCBI API access
- Configure database limits based on your needs

### Customization

- Modify feature extraction parameters in `feature_extractor.py`
- Add new machine learning models in `model_trainer.py`
- Extend data collection sources in `data_collector.py`

## Troubleshooting

### Common Issues

1. **NCBI API Rate Limiting**: The pipeline includes rate limiting, but you may need to reduce batch sizes for large datasets
2. **Memory Issues**: For large datasets, consider processing in batches
3. **Missing Dependencies**: Ensure all requirements are installed

### Error Handling

The pipeline includes comprehensive error handling:
- Graceful handling of API failures
- Validation of input data
- Fallback options for missing features

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Citation

If you use this tool in your research, please cite:

```
Terpene Synthase Product Predictor
A machine learning pipeline for predicting terpene synthase enzyme products
https://github.com/yourusername/terpene_synthase_predictor
```

## Future Enhancements

- [ ] Support for multiple product predictions per enzyme
- [ ] Integration with additional databases
- [ ] Deep learning models (CNN, LSTM, Transformer)
- [ ] Web interface for easy usage
- [ ] API endpoint for programmatic access
- [ ] Support for protein structure-based features
- [ ] Integration with molecular docking results
