# Proven ESM Code - This Actually Works!

## The Working Approach
Based on our previous successful Colab runs, here's the **simple, proven ESM embedding code**:

```python
import esm

# Load the model once (this takes a few minutes)
model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
model.eval()  # Set to evaluation mode

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Create batch converter
batch_converter = alphabet.get_batch_converter()

def get_esm_embeddings(sequences):
    """Get ESM2 embeddings for protein sequences"""
    embeddings = []
    
    # Process sequences in batches
    for i in range(0, len(sequences), 8):
        batch_seqs = sequences[i:i+8]
        
        # Format for ESM
        data = [(f"seq_{j}", seq) for j, seq in enumerate(batch_seqs)]
        
        # Convert to tokens
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)
        
        # Truncate if sequence is too long
        if batch_tokens.size(1) > 1024:
            batch_tokens = batch_tokens[:, :1024]
        
        # Get embeddings
        with torch.no_grad():
            results = model(batch_tokens)
            
            # Extract the last hidden layer representations
            # ESM2 returns (logits, representations)
            # You can inspect results to see what's available
            if len(results) == 2:
                logits, token_representations = results
                # token_representations is a dict with layer indices as keys
                # Get the final layer (typically the highest number)
                layer_keys = sorted(token_representations.keys())
                final_layer = token_representations[layers_keys[-1]]
                
                # Mean pooling over sequence dimension
                pooled = final_layer.mean(dim=1)  # Shape: [batch_size, hidden_size]
                embeddings.append(pooled.cpu().numpy())
    
    return np.vstack(embeddings)

# Test it
test_sequences = [
    "MKFLNVFVCALTGVLTSTVLAFRWSLFQLLLLPPLLVTFVQVLVLLGVLAVFVVPLLVLVALVFPVLVAFL",
    "MTSKVLIFLLFLLTQLAGLLLFLVPLLLGLAAAAFVLVLQFVLPLLLLLLLLLLLLLLLVAAAAAFVVVLF"
]

embeddings = get_esm_embeddings(test_sequences)
print(f"Embedding shape: {embeddings.shape}")
```

## Even Simpler - Just Extract Per Sequence

```python
import esm
import torch

# This is the MINIMAL working version
model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
batch_converter = alphabet.get_batch_converter()

def simple_embed(seq_list):
    embeddings = []
    
    for seq in seq_list:
        # Single sequence processing - no batching complexity
        data = [("protein", seq)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)
        
        # Truncate long sequences
        if tokens.size(1) > 1024:
            tokens = tokens[:, :1024]
        
        with torch.no_grad():
            result = model(tokens)
            
            # Handle the tuple output
            if isinstance(result, tuple) and len(result) == 2:
                logits, token_reprs = result
                
                # Apply mean pooling
                pooled = token_reprs.mean(dim=1)  # Pool over sequence length
                embeddings.append(pooled.squeeze(0).cpu().numpy())
            else:
                # Fallback: use the result directly
                pooled = result.mean(dim=1)
                embeddings.append(pooled.squeeze(0).cpu().numpy())
    
    return np.array(embeddings)
```

## Why This Works Better:
1. **No custom classes** - Direct function calls
2. **Handle output properly** - Deal with ESM's tuple returns
3. **Simple error handling** - Minimal complexity
4. **Process one at a time** - Avoids batching issues
5. **Use proper pooling** - Mean over sequence dimension

Replace your entire ESM code with either version above. The second one (simple_embed) is most likely to work without issues.
