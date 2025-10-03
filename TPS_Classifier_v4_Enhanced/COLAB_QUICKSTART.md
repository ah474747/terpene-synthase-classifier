# Google Colab Quick Start Guide

## 🚀 Ultra-Fast Setup (3 steps!)

### Step 1: Upload Files
1. Go to **Files** tab in Colab (left sidebar)
2. Upload **both** files:
   - `terpene_colab.ipynb` (this notebook)
   - `colab_training_data.tar.gz` (training data)

### Step 2: Run Notebook  
1. Open `terpene_colab.ipynb`
2. Click **Runtime** → **Change runtime type** → **GPU**
3. Run all cells sequentially (Shift+Enter through each cell)

### Step 3: Download Results
The notebook will automatically download:
- `best_model.pth` - Trained terpene classifier
- `results.json` - Evaluation metrics and predictions

## ⏱️ Timeline
- **Setup**: 2 minutes
- **Training**: 45-90 minutes on GPU
- **Total**: < 1.5 hours

## 📊 Expected Results
- **Performance**: >70% macro F1
- **Classes**: dialterpenes, germacrenes, monoterpenes, other, sesquiterpenes, triterpenes
- **Data**: 1,064 training + 262 validation sequences

## 🔧 If Issues
- **OOM errors**: Reduce batch size in cell 8 (`batch_size=8`)
- **Slow training**: Ensure GPU is selected (Runtime → Change runtime type)
- **Missing files**: Verify both `.ipynb` and `.tar.gz` uploaded

## 📋 What Each Cell Does
1. **Setup** - Install dependencies
2. **Upload** - Extract training data
3. **ESM** - Protein embedding class
4. **Model** - Neural network architecture
5. **Dataset** - Data loading class
6. **Training** - Create datasets & embeddings
7. **Train Setup** - Initialize model & optimizers
8. **Training Loop** - 15 epochs with F1 tracking
9. **Evaluate** - Classification report & metrics
10. **Download** - Save results to disk

**Ready to train! 🧬🌿**
