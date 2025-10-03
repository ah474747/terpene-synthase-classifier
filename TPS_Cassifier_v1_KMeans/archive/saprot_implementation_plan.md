# SaProt Implementation Plan for Germacrene Synthase Classification

## Overview
This document outlines the implementation plan for integrating SaProt (Structure-aware Protein) embeddings into our germacrene synthase binary classifier, comparing performance against ESM2 embeddings.

## Technical Requirements

### 1. SaProt Model Requirements
- **Model**: SaProt 650M parameter model (recommended)
- **Dependencies**: 
  - Python 3.8+
  - PyTorch
  - Hugging Face Transformers
  - SaProt-specific libraries
- **Hardware**: 16GB RAM minimum (MacBook Pro compatible)
- **Processing**: CPU-based inference (slow but feasible)

### 2. Structural Data Requirements
- **Input**: Protein structures in PDB/mmCIF format
- **Source**: AlphaFold Database (preferred) or structure prediction
- **Tool**: Foldseek for 3Di sequence generation
- **Output**: Structure-aware (SA) tokens for SaProt

### 3. Foldseek Requirements
- **Purpose**: Convert 3D protein structures to structural alphabet (3Di) sequences
- **Input**: PDB/mmCIF files
- **Output**: 3Di sequences (structural tokens)
- **Installation**: Available for macOS
- **Computational**: Low to moderate (CPU-based)

## Implementation Strategy

### Phase 1: Data Preparation
1. **Combine Datasets**
   - MARTS-DB: 80 germacrene sequences
   - NCBI: 17 unique germacrene sequences
   - Total: 97 germacrene + 1,276 non-germacrene = 1,373 sequences

2. **Structure Acquisition**
   - Download from AlphaFold Database (preferred)
   - Use AlphaFold2/ESMFold for missing structures
   - Validate structure quality and completeness

### Phase 2: Structural Token Generation
1. **Install Foldseek**
   - Download and install Foldseek for macOS
   - Configure for 3Di sequence generation
   - Test with sample structures

2. **Generate 3Di Sequences**
   - Process all 1,373 protein structures
   - Convert to structural alphabet format
   - Validate 3Di sequence quality

### Phase 3: SaProt Embedding Generation
1. **Model Setup**
   - Load SaProt 650M model
   - Configure tokenizer for SA tokens
   - Set up batch processing

2. **Embedding Extraction**
   - Process sequences with SA tokens
   - Generate 1280-dimensional embeddings
   - Save embeddings for model training

### Phase 4: Comparative Evaluation
1. **Model Training**
   - Train XGBoost with SaProt embeddings
   - Use identical hyperparameters as ESM2 model
   - Apply class weights (scale_pos_weight ≈ 14)

2. **Performance Comparison**
   - F1-Score, Precision, Recall
   - AUC-ROC, AUC-PR
   - Statistical significance testing
   - Feature importance analysis

## Expected Challenges

### 1. Computational Resources
- **Structure Prediction**: Time-consuming for 1,373 sequences
- **SaProt Inference**: Hours of processing on CPU
- **Memory Usage**: 16GB RAM may be limiting

### 2. Data Availability
- **Missing Structures**: Some sequences may not have predicted structures
- **Quality Issues**: Predicted structures may be incomplete
- **Format Compatibility**: PDB/mmCIF format requirements

### 3. Technical Complexity
- **SA Token Generation**: Requires Foldseek expertise
- **Model Integration**: SaProt-specific implementation
- **Pipeline Coordination**: Multiple tools and formats

## Mitigation Strategies

### 1. Computational Efficiency
- **Batch Processing**: Process sequences in manageable batches
- **Parallel Processing**: Use multiple CPU cores where possible
- **Cloud Resources**: Consider cloud computing for intensive steps

### 2. Data Quality
- **AlphaFold Database**: Prioritize pre-predicted structures
- **Quality Filtering**: Remove low-confidence predictions
- **Fallback Options**: Use ESM2 for missing structures

### 3. Technical Support
- **Documentation**: Maintain detailed implementation notes
- **Testing**: Validate each step with sample data
- **Backup Plans**: ESM2 as fallback option

## Success Metrics

### 1. Performance Targets
- **F1-Score**: > 0.70 (current: 0.59)
- **Precision**: > 0.75 (current: 0.69)
- **Recall**: > 0.65 (current: 0.54)
- **AUC-PR**: > 0.80 (current: 0.72)

### 2. Comparative Analysis
- **SaProt vs ESM2**: Statistical significance testing
- **Feature Importance**: Interpretability analysis
- **Computational Cost**: Time and resource comparison

## Timeline

### Week 1: Setup and Data Preparation
- Install Foldseek and SaProt dependencies
- Download structures from AlphaFold Database
- Prepare combined dataset

### Week 2: Structural Token Generation
- Generate 3Di sequences with Foldseek
- Validate structural token quality
- Prepare SA token inputs

### Week 3: SaProt Embedding Generation
- Generate SaProt embeddings
- Validate embedding quality
- Prepare for model training

### Week 4: Model Training and Evaluation
- Train XGBoost with SaProt embeddings
- Compare performance with ESM2
- Analyze results and interpretability

## Next Steps
1. Begin with data integration and structure acquisition
2. Install and test Foldseek with sample structures
3. Set up SaProt model and tokenizer
4. Implement embedding generation pipeline
5. Train and evaluate comparative models
