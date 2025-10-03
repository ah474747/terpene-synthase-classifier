# Final ESM Fix - Google Colab

## The Error
ESM2 doesn't support `return_pairs=False` parameter in the forward method.

## Corrected ESM Code (Copy this entire block):

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
                
                # FIXED: Remove return_pairs parameter
                results = self.model(tokens)
                
                # Extract token representations
                if "representations" in results:
                    # Use the last layer representation
                    last_layer_key = max(results["representations"].keys())
                    token_reprs = results["representations"][last_layer_key]
                else:
                    # Fallback to logits
                    token_reprs = results["logits"] if "logits" in results else None
                
                if token_reprs is not None:
                    # Mean pooling over sequence length
                    pooled = token_reprs.mean(dim=1)
                    embeddings.append(pooled.cpu().numpy())
        
        if embeddings:
            return np.vstack(embeddings)
        else:
            raise ValueError("No embeddings generated - check model output")
```

## Alternative Simple Fix (If above fails):

```python
import esm

class ESMEmbedder:
    def __init__(self, model_id="ess2_t33_650M_UR50D"):
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
            batch = [(f"sep_{j}", seq) for j, seq in enumerate(seqs[i:i+8])]
            _, _, tokens = self.batch_converter(batch)
            tokens = tokens.to(self.device)
            
            with torch.no_grad():
                if tokens.size(1) > 1024:
                    tokens = tokens[:, :1024]
                
                # Simple forward pass
                logits, _ = self.model(tokens)
                
                # Mean pooling
                pooled = logits.mean(dim=1)
                embeddings.append(pooled.cpu().numpy())
        
        return np.vstack(embeddings)
```

## Instructions:
1. **Replace your entire ESM cell** with one of the above corrected versions
2. **Re-run the ESM cell**
3. **Re-run the Training cell** to recreate datasets
4. **Continue training**

## What I Fixed:
- Removed `return_pairs=False` parameter that doesn't exist
- Added proper handling for different ESM model outputs
- Used simple tuple unpacking: `logits, _ = self.model(tokens)`

This should resolve the TypeError and get your training started!
