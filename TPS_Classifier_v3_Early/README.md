# 🧬 Terpene Synthase Gold Standard Dataset (TS-GSD) Pipeline

A comprehensive, multi-modal deep learning system for predicting Terpene Synthase (TPS) product functional ensembles, optimized for Macro F1 score on limited and imbalanced datasets.

## 🎯 Project Overview

This project implements a three-module approach to build a high-performance TPS classifier:

1. **Module 1: TS-GSD Data Pipeline** ✅ (Complete)
2. **Module 2: Feature Extraction Pipeline** (Future)
3. **Module 3: Multi-Modal Deep Learning** (Future)

## 📋 Module 1: TS-GSD Data Acquisition and Curation Pipeline

### 🎯 Objectives

- Systematically acquire TPS data from MARTS-DB and UniProt
- Merge and clean datasets using UniProt accession IDs
- Engineer multi-label functional ensemble targets (30 classes)
- Output clean, machine-readable dataset ready for feature extraction

### 🏗️ Architecture

```
MARTS-DB Data → Multi-Label Engineering → UniProt Enrichment → Final TS-GSD
     ↓                    ↓                        ↓              ↓
Enzyme IDs,        Functional Ensemble      Sequence, Pfam,   Clean Dataset
Products,          Mapping (30 classes)     Taxonomy, etc.    (CSV/JSON)
Substrates
```

### 📊 Dataset Schema

| Column Name | Data Type | Purpose |
|-------------|-----------|---------|
| `uniprot_accession_id` | String | Primary Key |
| `aa_sequence` | String | Full protein sequence |
| `product_smiles_list` | JSON | Product SMILES representations |
| `substrate_type` | Categorical | GPP, FPP, GGPP, etc. |
| `rxn_class_i_ii_hybrid` | Categorical | Class 1, Class 2, Hybrid |
| `pfam_domain_ids` | JSON | Predicted conserved domains |
| `taxonomy_phylum` | String | Organism phylum |
| `target_vector` | JSON | Binary multi-label vector (30 classes) |

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

1. **Upload the notebook** to Google Colab
2. **Run Cell 1** to install dependencies
3. **Run Cells 2-9** sequentially to execute the pipeline
4. **Download** the generated `TS-GSD_demo.csv` file

### Option 2: Local Installation

```bash
# Clone or download the project
cd terpene_classifier_v3

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python ts_gsd_pipeline.py
```

### Option 3: Colab Setup Script

```bash
# Run the automated setup script
python setup_colab.py
```

## 📁 Project Structure

```
terpene_classifier_v3/
├── ts_gsd_pipeline.py          # Main pipeline implementation
├── setup_colab.py              # Colab environment setup
├── requirements.txt             # Python dependencies
├── TS-GSD_Pipeline_Demo.ipynb  # Interactive Colab notebook
├── README.md                   # This file
├── data/                       # Generated datasets
│   ├── TS-GSD.csv             # Final dataset
│   └── TS-GSD_metadata.json   # Dataset metadata
└── models/                     # Future: Model artifacts
```

## 🧬 Functional Ensemble Mapping

The system maps specific terpene products to 30 functional ensembles:

### Major Families:
- **Germacrene Family** (Ensemble 0): Germacrene A, D, C, etc.
- **Monoterpenes** (Ensembles 1-4): Limonene, Pinene, Myrcene, Camphene
- **Sesquiterpenes** (Ensembles 5-8): Farnesene, Bisabolene, Humulene, Caryophyllene
- **Diterpenes** (Ensembles 9-11): Kaurene, Abietadiene, Taxadiene
- **Triterpenes** (Ensembles 12-13): Squalene, Lanosterol
- **Oxygenated Terpenes** (Ensembles 14-29): Menthol, Linalool, Geraniol, etc.

## 🔧 Key Features

### Multi-Label Classification
- **Binary target vectors** for each enzyme
- **Multiple products per enzyme** support
- **Imbalanced dataset handling**

### Data Integration
- **MARTS-DB integration** for enzyme-reaction data
- **UniProt API** for sequence and taxonomic enrichment
- **Automatic fallback** to simulated data for development

### Robust Pipeline
- **Error handling** and logging
- **Rate limiting** for API calls
- **Progress tracking** with tqdm
- **Comprehensive validation**

## 📊 Example Usage

```python
from ts_gsd_pipeline import TSGSDPipeline

# Initialize pipeline
pipeline = TSGSDPipeline(output_dir="data", n_classes=30)

# Run complete pipeline
dataset_path = pipeline.run_pipeline()

print(f"Dataset created at: {dataset_path}")
```

## 🔬 Data Sources

### MARTS-DB
- **Primary source** for TPS enzyme data
- **Enzyme-reaction relationships**
- **Product mechanisms and substrates**

### UniProt
- **Sequence validation** and enrichment
- **Pfam domain predictions**
- **Taxonomic lineage information**
- **Protein annotations**

## 📈 Performance Considerations

### Multi-Label Optimization
- **Macro F1 score** as primary metric
- **Focal Loss** ready for imbalanced data
- **30 functional ensembles** to manage sparsity

### Scalability
- **Batch processing** for large datasets
- **Memory-efficient** data structures
- **GPU-ready** for future modules

## 🔮 Future Modules

### Module 2: Feature Extraction Pipeline
- **ESM2 embeddings** (650M parameters)
- **Structural features** from AlphaFold
- **Domain architecture** encoding

### Module 3: Multi-Modal Deep Learning
- **Focal Loss** implementation
- **Attention mechanisms**
- **Ensemble methods**

## 🐛 Troubleshooting

### Common Issues

1. **BioPython Import Error**
   ```bash
   pip install biopython
   ```

2. **UniProt API Rate Limiting**
   - The pipeline includes automatic rate limiting
   - Use smaller sample sizes for testing

3. **Memory Issues**
   - Reduce batch size in `fetch_uniprot_features()`
   - Process data in chunks

### Development Mode
The pipeline automatically falls back to simulated data if real data sources are unavailable, making it perfect for development and testing.

## 📚 Dependencies

### Core Requirements
- `pandas>=1.5.0` - Data manipulation
- `numpy>=1.21.0` - Numerical computing
- `requests>=2.28.0` - API calls
- `biopython>=1.79` - Bioinformatics
- `tqdm>=4.64.0` - Progress bars

### Future Modules
- `torch>=1.12.0` - Deep learning
- `transformers>=4.21.0` - Language models
- `scikit-learn>=1.1.0` - Machine learning

## 🤝 Contributing

This is a modular system designed for iterative development. Each module can be developed and tested independently while maintaining compatibility with the overall architecture.

## 📄 License

This project is designed for research and educational purposes. Please ensure compliance with data source licenses (UniProt, MARTS-DB) when using real data.

---

**Status**: Module 1 Complete ✅ | Module 2 In Development | Module 3 Planned

**Next Steps**: Implement ESM2 feature extraction pipeline for Module 2



