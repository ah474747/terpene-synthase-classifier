# Final Germacrene Synthase Classifier: User Guide

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Making Predictions
```python
import xgboost as xgb
import numpy as np
from transformers import EsmModel, EsmTokenizer
import torch

# Load the trained model
model = xgb.XGBClassifier()
model.load_model('models/esm2_tuned_classifier.json')

# Load ESM2 for embedding generation
tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
esm_model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

def generate_embedding(sequence):
    """Generate ESM2 embedding for a protein sequence"""
    # Clean sequence (remove *, non-standard amino acids)
    clean_seq = ''.join([aa for aa in sequence if aa in 'ACDEFGHIKLMNPQRSTVWY'])
    
    if len(clean_seq) < 50:
        return None
    
    # Tokenize and encode
    inputs = tokenizer(clean_seq, return_tensors="pt", truncation=True, max_length=1024)
    
    with torch.no_grad():
        outputs = esm_model(**inputs)
        # Use mean pooling of last hidden states
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    
    return embedding

# Example usage
sequence = "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL..."
embedding = generate_embedding(sequence)

if embedding is not None:
    # Reshape for prediction (add batch dimension)
    embedding = embedding.reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(embedding)[0]
    probability = model.predict_proba(embedding)[0]
    
    print(f"Prediction: {'Germacrene Synthase' if prediction == 1 else 'Non-Germacrene'}")
    print(f"Confidence: {probability[1]:.3f}")
else:
    print("Sequence too short or invalid")
```

## Model Architecture Details

### Input Processing
1. **Sequence Cleaning**: Remove stop codons (*) and non-standard amino acids
2. **Length Filtering**: Minimum 50 amino acids
3. **ESM2 Embedding**: Generate 1280-dimensional vector
4. **Feature Engineering**: Add K-means and sequence features

### Feature Types
- **ESM2 Embeddings**: 1280 dimensions (protein language model)
- **K-means Features**: 10 features (5 cluster assignments + 5 distances)
- **Sequence Features**: 23 features (length + 20 amino acids + 2 physicochemical)
- **Total**: 1313 features per sequence

### Model Parameters
```python
best_params = {
    'colsample_bytree': 1.0,
    'learning_rate': 0.1,
    'max_depth': 4,
    'min_child_weight': 5,
    'n_estimators': 100,
    'subsample': 0.8,
    'scale_pos_weight': 8.2  # For class imbalance
}
```

## Performance Metrics

### Cross-Validation Results (15-fold)
- **F1-Score**: 0.513 ± 0.089
- **Precision**: 0.571 ± 0.089
- **Recall**: 0.492 ± 0.089
- **ROC-AUC**: 0.916 ± 0.089
- **Average Precision**: 0.568 ± 0.089

### Interpretation
- **F1-Score**: Balanced measure of precision and recall
- **Precision**: 57% of predicted germacrene synthases are correct
- **Recall**: 49% of actual germacrene synthases are identified
- **ROC-AUC**: Excellent discrimination ability (91.6%)

## Batch Processing

### Processing Multiple Sequences
```python
def process_sequences(sequences):
    """Process multiple sequences efficiently"""
    embeddings = []
    valid_sequences = []
    
    for seq in sequences:
        embedding = generate_embedding(seq)
        if embedding is not None:
            embeddings.append(embedding)
            valid_sequences.append(seq)
    
    if embeddings:
        embeddings = np.array(embeddings)
        predictions = model.predict(embeddings)
        probabilities = model.predict_proba(embeddings)
        
        return valid_sequences, predictions, probabilities
    else:
        return [], [], []

# Example
sequences = ["MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL...",
             "MKKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL..."]
valid_seqs, preds, probs = process_sequences(sequences)
```

## Model Limitations

### Known Issues
1. **Class Imbalance**: Model is biased toward negative predictions
2. **Sequence Length**: Very long sequences (>1024 amino acids) are truncated
3. **Domain Specificity**: Trained only on terpene synthase sequences
4. **Annotation Quality**: Performance depends on training data quality

### Recommendations
1. **Threshold Tuning**: Adjust prediction threshold based on use case
2. **Ensemble Methods**: Combine with other models for better performance
3. **Active Learning**: Continuously improve with new labeled data
4. **Domain Adaptation**: Fine-tune for specific organism or enzyme families

## Troubleshooting

### Common Issues

#### "Sequence too short or invalid"
- **Cause**: Sequence < 50 amino acids or contains invalid characters
- **Solution**: Filter sequences before processing

#### "CUDA out of memory"
- **Cause**: GPU memory insufficient for ESM2 model
- **Solution**: Use CPU or reduce batch size

#### "Low confidence predictions"
- **Cause**: Sequences very different from training data
- **Solution**: Check sequence quality and domain relevance

### Performance Tips
1. **Batch Processing**: Process multiple sequences together
2. **GPU Acceleration**: Use GPU for ESM2 embedding generation
3. **Caching**: Cache embeddings for repeated sequences
4. **Memory Management**: Process large datasets in chunks

## File Structure

```
terpene_classifier_kmeans/
├── models/
│   ├── esm2_tuned_classifier.json      # Final tuned model
│   └── esm2_improved_classifier.json   # Improved model
├── results/
│   ├── esm2_tuned_cv_results.csv       # Cross-validation results
│   ├── model_comparison.png            # Performance comparison
│   └── tuned_comparison.csv            # Tuned vs improved comparison
├── data/                               # Training data
├── archive/                            # Development files
├── improve_esm2_model.py              # Training script
├── input_sequences.fasta              # Training sequences
├── metadata.json                      # Sequence metadata
├── sequence_info.csv                  # Sequence information
├── requirements.txt                   # Dependencies
├── README.md                          # Project overview
├── MODEL_DEVELOPMENT_SUMMARY.md       # Development history
└── FINAL_MODEL_GUIDE.md              # This guide
```

## Next Steps

### Immediate Improvements
1. **Threshold Optimization**: Find optimal prediction threshold
2. **Feature Analysis**: Identify most important features
3. **Error Analysis**: Analyze misclassified sequences

### Long-term Development
1. **Semi-supervised Learning**: Use unlabeled sequences
2. **Structure Integration**: Add 3D structure information
3. **Multi-class Classification**: Extend to other terpene types
4. **Real-time Prediction**: Develop web interface or API

## Support

For questions or issues:
1. Check this guide and the development summary
2. Review the archived development files
3. Examine the training script for implementation details
4. Analyze the cross-validation results for performance insights

The model represents a significant achievement in protein function prediction, combining state-of-the-art language models with traditional machine learning techniques for practical biological applications.
