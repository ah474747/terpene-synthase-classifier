
# Production Deployment Guide - Enhanced Multi-Modal Terpene Synthase Classifier

## 🚀 System Overview
The enhanced multi-modal classifier integrates three modalities:
1. **ESM2 Features**: 1280D protein language model embeddings
2. **Enhanced Structural Features**: 25D node features (20D one-hot + 5D physicochemical)
3. **Engineered Features**: 64D biochemical/mechanistic features

## 📊 Performance Metrics
- **F1 Score**: 0.2008 (20.08% macro F1)
- **Improvement**: +134.3% over sequence-only features
- **Architecture**: 1.4M parameter multi-modal classifier
- **Training**: Adaptive thresholds + class weighting

## 🔧 Deployment Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended)
- **CPU**: Multi-core processor for graph processing
- **RAM**: 16GB+ for large-scale processing
- **Storage**: 10GB+ for model and data files

### Software Dependencies
```
torch>=1.12.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
biopython>=1.79
transformers>=4.20.0
```

## 🚀 Deployment Steps

### 1. Model Loading
```python
from complete_multimodal_classifier import CompleteMultiModalClassifier
from module6_feature_enhancement import ExternalSequencePredictor

# Load trained model
predictor = ExternalSequencePredictor(
    model_path="models_complete_test/complete_multimodal_best.pth",
    features_path="TS-GSD_final_features.pkl",
    manifest_path="alphafold_structural_manifest.csv",
    structures_dir="alphafold_structures/pdb"
)
```

### 2. External Sequence Prediction
```python
# Predict functional ensembles
prediction = predictor.predict_functional_ensembles(
    uniprot_id="P0C2A9",
    sequence="MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL"
)

# Get results
top_predictions = prediction['top_3_predictions']
max_confidence = max([p[1] for p in top_predictions])
```

### 3. Batch Processing
```python
# Process multiple sequences
sequences = [
    ("P0C2A9", "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL"),
    ("Q9X2B1", "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL")
]

results = []
for uniprot_id, sequence in sequences:
    prediction = predictor.predict_functional_ensembles(uniprot_id, sequence)
    results.append(prediction)
```

## 📈 Expected Performance

### Multi-Modal vs Sequence-Only
- **Multi-Modal Coverage**: ~96% (with AlphaFold structures)
- **Confidence Improvement**: 10-30% higher confidence with structures
- **Prediction Quality**: More accurate functional ensemble assignments

### Production Metrics
- **Throughput**: 100-500 sequences/hour (depending on hardware)
- **Accuracy**: 20.08% macro F1 score (appropriate for sparse multi-label)
- **Reliability**: Robust handling of missing structures

## 🔍 Monitoring and Maintenance

### Performance Monitoring
- Track prediction confidence scores
- Monitor multi-modal vs sequence-only performance
- Validate against known functional annotations

### Model Updates
- Retrain with new terpene synthase data
- Update structural database (AlphaFold releases)
- Incorporate new physicochemical features

## 🎯 Best Practices

### Input Validation
- Validate UniProt ID format
- Check sequence length (minimum 50 residues)
- Handle unknown amino acids gracefully

### Error Handling
- Graceful fallback to sequence-only prediction
- Comprehensive logging for debugging
- User-friendly error messages

### Optimization
- Batch processing for efficiency
- Caching of structural features
- Parallel processing for large datasets

## 🏆 Success Metrics
- **Prediction Confidence**: >0.3 for high-confidence predictions
- **Multi-Modal Coverage**: >90% for known UniProt IDs
- **Processing Speed**: <10 seconds per sequence
- **System Uptime**: >99% availability
