# ESM Layer Fix for Google Colab

## The Error
The ESM model is throwing a `KeyError: 32` because it can't find layer 32 in the representations.

## Quick Fix
Replace the problematic ESM embedding code with this corrected version:

```python
import esm

class ESMEmbedder:
    def __init__(self, model_id="esm2_t33_650M_UR50D"):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.alphabet = None
        
    def embed_mean(self, seqs):
        if self.model is None:
            print(f"Loading {self.model_id}...")
            self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(self.model_id)
            self.model = self.model.to(self.device)
            self.batch_converter = self.alphabet.get_batch_converter()
        
        embeddings = []
        for i in range(0, len(seqs), 8):
            batch = [(f"seq_{j}", seq) for j, seq in enumerate(seqs[i:i+8])]
            _, _, tokens = self.batch_converter(batch)
            tokens = tokens.to(self.device)
            
            with torch.no_grad():
                if tokens.size(1) > 1024:
                    tokens = tokens[:, :1024]
                results = self.model(tokens, return_pairs=False)
                
                # FIXED: Use logits instead of representations
                logits = results["logits"]  # [batch_size, seq_length, vocab_size]
                
                # Get the ESM pooler output (mean pooling)
                # ESM pooler performs mean pooling over sequence length
                pooled = logits.mean(dim=1)  # [batch_size, vocab_size]
                
                # Extract the first vocab_size-5 dimensions (remove special tokens)
                # This gives us the protein embedding
                protein_embedding = pooled[:, :-5]  # [batch_size, vocab_size-5]
                
                embeddings.append(protein_embedding.cpu().numpy())
        
        return np.vstack(embeddings)
```

## Alternative Fix (More Robust)
If the above doesn't work, use this approach:

```python
import esm

class ESMEmbedder:
    def __init__(self, model_id="esm2_t33_650M_UR50D"):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.alphabet = None
        
    def embed_mean(self, seqs):
        if self.model is None:
            print(f"Loading {self.model_id}...")
            self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(self.model_id)
            self.model = self.model.to(self.device)
            self.batch_converter = self.alphabet.get_batch_converter()
        
        embeddings = []
        for i in range(0, len(seqs), 8):
            batch = [(f"seq_{j}", seq) for j, seq in enumerate(seqs[i:i+8])]
            _, _, tokens = self.batch_converter(batch)
            tokens = tokens.to(self.device)
            
            with torch.no_grad():
                if tokens.size(1) > 1024:
                    tokens = tokens[:, :1024]
                
                # Use the last hidden states directly
                _, token_representations = self.model(tokens, return_pairs=False)
                
                # Apply mean pooling over sequence length
                # Exclude special tokens (CLS and PAD tokens at beginning/end)
                seq_lengths = (tokens != self.alphabet.padding_idx).sum(dim=1)
                pooled = token_representations.sum(dim=1) / seq_lengths.unsqueeze(1)
                
                embeddings.append(pooled.cpu().numpy())
        
        return np.vstack(embeddings)
```

## Steps to Fix:
1. **Copy one of these fixed ESMEmbedder classes**
2. **Re-run the ESM cell** with the new code  
3. **Re-run the Training cell** to create datasets again
4. **Continue with the rest** of the notebook

## Why This Happened:
ESM2 models have different layer structures than expected. The representation keys don't always match the layer numbers, so we need to use the logits or last hidden states instead.

The fixed version will work reliably across different ESM model versions.
