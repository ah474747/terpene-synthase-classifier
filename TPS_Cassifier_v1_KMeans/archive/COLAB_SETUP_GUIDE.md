# Google Colab Setup Guide for SaProt Embeddings

## 🚀 Quick Start

### Step 1: Open Google Colab
1. Go to [Google Colab](https://colab.research.google.com/)
2. Sign in with your Google account
3. Click "New notebook" or "File" → "New notebook"

### Step 2: Upload the Notebook
1. Download the `colab_saprot_final.ipynb` file from your local machine
2. In Colab, go to "File" → "Upload notebook"
3. Select the `colab_saprot_final.ipynb` file
4. Click "Upload"

### Step 3: Select GPU Runtime
1. Go to "Runtime" → "Change runtime type"
2. Set "Hardware accelerator" to "GPU"
3. Choose "T4 GPU" (free) or "V100/A100" (Colab Pro)
4. Click "Save"

### Step 4: Upload Your Dataset
1. In the Colab interface, click the folder icon in the left sidebar
2. Click the upload button (📁 with up arrow)
3. Upload your `expanded_germacrene_dataset.fasta` file
4. Rename it to `input_sequences.fasta` if needed

## 📋 Prerequisites Checklist

Before running the notebook, ensure you have:

- [ ] Google account with Colab access
- [ ] `expanded_germacrene_dataset.fasta` file ready
- [ ] Sufficient Google Drive space (50+ GB recommended)
- [ ] Stable internet connection

## 🔧 Configuration

### User Configuration Cell
In the notebook, you'll need to modify these variables:

```python
# User Configuration
INPUT_FASTA_PATH = 'input_sequences.fasta'  # Path to your uploaded FASTA file
ACQUISITION_STRATEGY = 'AFDB'  # Choose: 'AFDB', 'COLABFOLD', or '3DI_PRECOMPUTED'
```

### Strategy Options
- **'AFDB'**: Download structures from AlphaFold Database (fastest, limited to known structures)
- **'COLABFOLD'**: Generate structures with ColabFold (slower, works for any sequence)
- **'3DI_PRECOMPUTED'**: Use pre-computed 3Di sequences (fastest, if available)

## 📊 Expected Timeline

### Phase 1: Environment Setup (5-10 minutes)
- GPU runtime selection and validation
- Google Drive mounting and space check
- Dependency installation

### Phase 2: Software Installation (10-15 minutes)
- Foldseek installation and validation
- SaProt repository setup
- Model download (650M parameters)

### Phase 3: Data Processing (30-60 minutes)
- Structure acquisition (depends on strategy)
- 3Di sequence generation
- SaProt embedding generation

### Phase 4: Results Export (5 minutes)
- Save to Google Drive
- Download instructions

## ⚠️ Important Notes

### GPU Availability
- Free Colab GPUs have time limits (typically 12 hours)
- Colab Pro offers longer sessions and better GPUs
- Save your work frequently to Google Drive

### Memory Management
- The 650M parameter SaProt model requires ~1.3 GB VRAM
- Batch processing is optimized for T4 GPU (16 GB VRAM)
- Monitor GPU memory usage during embedding generation

### Data Transfer
- Large embedding files will be saved to Google Drive
- Download times depend on your internet connection
- Consider using Google Drive desktop sync for large files

## 🐛 Troubleshooting

### Common Issues

#### 1. GPU Not Detected
```
⚠️ No GPU detected! Please select GPU runtime.
```
**Solution**: Go to Runtime → Change runtime type → GPU → Save

#### 2. Google Drive Mount Failed
```
❌ Google Drive mounting failed
```
**Solution**: 
- Check your Google account permissions
- Try remounting: `drive.mount('/content/drive', force_remount=True)`

#### 3. Insufficient Space
```
⚠️ WARNING: Insufficient space! Required: 50 GB, Available: 30.0 GB
```
**Solution**: Free up space in your Google Drive

#### 4. Foldseek Installation Failed
```
❌ Foldseek installation failed
```
**Solution**: The notebook will automatically retry with manual installation

#### 5. Model Download Failed
```
❌ Model download failed
```
**Solution**: Check your internet connection and try again

### Getting Help
- Check the Colab documentation: [colab.research.google.com](https://colab.research.google.com/)
- Google Colab FAQ: [research.google.com/colaboratory/faq.html](https://research.google.com/colaboratory/faq.html)
- SaProt GitHub: [github.com/mingkangyang/SaProt](https://github.com/mingkangyang/SaProt)

## 📁 Output Files

After successful completion, you'll have:

1. **`saprot_embeddings.npy`** - SaProt embeddings (main output)
2. **`sequence_info.csv`** - Sequence information and metadata
3. **`generated_3di_sequences.fasta`** - 3Di sequences used for embedding generation

## 🔄 Next Steps

1. **Download Results**: Use the provided download instructions
2. **Local Evaluation**: Train XGBoost models with SaProt embeddings
3. **Comparative Analysis**: Compare with ESM2 embeddings
4. **Model Selection**: Choose the best performing model

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all prerequisites are met
3. Ensure stable internet connection
4. Try restarting the Colab runtime

---

**Ready to start?** Open the `colab_saprot_final.ipynb` notebook in Google Colab and follow the steps above!
