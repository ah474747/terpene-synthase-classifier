# Google Colab Training Instructions

## Quick Start

1. **Open Google Colab**: Go to [colab.research.google.com](https://colab.research.google.com)

2. **Upload the notebook**: Upload `terpene_prediction_colab.ipynb` from this directory

3. **Upload training data**: Upload the compressed file `colab_training_data.tar.gz` (362KB)

4. **Extract data**: In Colab, run:
   ```python
   !tar -xzf colab_training_data.tar.gz
   ```

5. **Run the notebook**: Execute cells 1-13 in sequence

## What You'll Get

- **Trained model**: `best_model.pth` (~15-30MB)
- **Training results**: `training_results.json` with performance metrics
- **Performance metrics**: Macro F1, precision, recall per class
- **Evaluation plots**: Training curves and confusion matrices

## Dataset Summary

- **1,326 protein sequences** from enhanced MARTS database
- **6 product classes**: diterpenes, germacrenes, monoterpenes, other, sesquiterpenes, triterpenes
- **Train/Val split**: 1,064 training + 262 validation
- **ESM2-650M embeddings**: 1,280-dimensional protein representations
- **Multimodal approach**: ESM embeddings + engineered features + structural features

## Expected Training Time

- **Google Colab GPU**: ~45-90 minutes
- **Google Colab CPU**: ~3-6 hours
- **Memory requirement**: 8-16 GB RAM (easily handled by Colab)

## Key Features

- **Numbered cells** for easy sequential execution
- **Memory efficient**: Batch processing for ESM embeddings
- **GPU acceleration**: Automatic CUDA detection
- **Model checkpointing**: Saves best performing model
- **Comprehensive evaluation**: Multiple metrics and visualizations

## Troubleshooting

- **Out of memory**: Reduce batch size in Cell 10
- **Missing dependencies**: Run Cell 1 with `%%capture` removed to see errors
- **Upload issues**: Ensure `data/` folder is created in `/content/` 

## After Training

Download the `best_model.pth` and `training_results.json` files to your local machine. You can then integrate them into your local TPS system by copying the checkpoint to `models/checkpoints/` and updating `EPS_TOPIC_KEY` accordingly.
