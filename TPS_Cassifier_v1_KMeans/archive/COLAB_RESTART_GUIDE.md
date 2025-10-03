# How to Delete Colab Notebook and Start Over

## Option 1: Delete Current Notebook (Recommended)

### Step 1: Close the current notebook
1. In Google Colab, click "File" → "Close and halt"
2. This closes the notebook and stops the runtime

### Step 2: Delete from Google Drive (optional)
1. Go to [Google Drive](https://drive.google.com)
2. Navigate to "Colab Notebooks" folder (usually in "My Drive")
3. Find the old notebook file
4. Right-click and select "Remove" (moves to trash)

### Step 3: Upload the new notebook
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click "File" → "Upload notebook"
3. Select the new `colab_saprot_esmfold.ipynb` file
4. Click "Upload"

## Option 2: Restart Runtime Only

### If you want to keep the notebook but start fresh:
1. In Google Colab, click "Runtime" → "Factory reset runtime"
2. Confirm the action
3. This clears all variables and installed packages
4. Start running cells from the beginning

## Option 3: Create New Notebook from Scratch

### Step 1: Create new notebook
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click "File" → "New notebook"

### Step 2: Upload new notebook
1. Click "File" → "Upload notebook"
2. Select `colab_saprot_esmfold.ipynb`
3. Click "Upload"

## After Uploading New Notebook

### Step 1: Select GPU runtime
1. Go to "Runtime" → "Change runtime type"
2. Set "Hardware accelerator" to "GPU"
3. Click "Save"

### Step 2: Upload your dataset
1. Click the folder icon in the left sidebar
2. Click the upload button (📁 with up arrow)
3. Upload your `input_sequences.fasta` file

### Step 3: Run cells sequentially
1. Start with the first code cell (GPU Check)
2. Run cells one by one using the play button (▶️)
3. Wait for each cell to finish before proceeding

## Key Differences in New Notebook

### ESMFold Implementation
- Uses ESMFold instead of ColabFold
- Generates structures for any sequence
- No dependency on AlphaFold Database
- Fully implemented (no placeholder code)

### Processing Limits
- Default: processes first 100 sequences
- Adjustable: change `MAX_SEQUENCES` in cell 3.1
- Recommended: start with 10-20 sequences for testing

### Expected Timeline
- Structure generation: 1-2 minutes per sequence
- 100 sequences: approximately 2-3 hours
- 10 sequences: approximately 20-30 minutes

### Batch Processing
- Reduced batch size (32 instead of 128)
- Better memory management
- More stable for T4 GPU

## Verification Steps

### After upload, verify:
1. ✅ Notebook title shows "SaProt Embeddings for Germacrene Synthase Classification"
2. ✅ Cells are in correct order (see cell headers)
3. ✅ GPU runtime is selected
4. ✅ `input_sequences.fasta` is uploaded

## Tips

### For faster testing:
1. Edit cell 3.1 to set `MAX_SEQUENCES = 10`
2. Run the entire workflow on 10 sequences first
3. Once confirmed working, increase to 100 or more

### For monitoring progress:
1. Watch the output of each cell
2. Check GPU memory usage
3. Note the time per sequence
4. Adjust `MAX_SEQUENCES` based on time constraints

### For troubleshooting:
1. Read error messages carefully
2. Check GPU runtime is selected
3. Verify file uploads
4. Restart runtime if needed

## Quick Start Checklist

- [ ] Delete or close old notebook
- [ ] Upload new `colab_saprot_esmfold.ipynb`
- [ ] Select GPU runtime
- [ ] Upload `input_sequences.fasta`
- [ ] Set `MAX_SEQUENCES` (default: 100, testing: 10)
- [ ] Run cells sequentially
- [ ] Monitor progress and errors
- [ ] Download results from Google Drive

---

**Ready to start?** Follow Option 1 above to delete the current notebook and upload the new one.
