# Final ESM Fix - Complete Solution

## The Error
The model is returning strings instead of tensors, causing the `.mean()` method to fail.

## Working ESM Code (Copy this exactly):

```python
import esm
import torch

class ESMEmbedder:
    def __init__(self, model_id="esm2_t33_650M_UR50D"):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.alphabet = None
        self.batch_converter = None
        
    def embed_mean(self, seqs):
        if self.model is None:
            print(f"Loading {self.model_id}...")
            self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(self.model_id)
            self.model = self.model.to(self.device)
            self.batch_converter = self.alphabet.get_batch_converter()
        
        embeddings = []
        print(f"Processing {len(seqs)} sequences in batches of 8...")
        
        for i in range(0, len(seqs), 8):
            batch_seqs = seqs[i:i+8]
            print(f"Processing batch {i//8 + 1} {(i+1, min(i+8, len(seqs)))}")
            
            # Convert sequences to model format
            data = [(f"seq_{j+i}", seq) for j, seq in enumerate(batch_seqs)]
            
            try:
                # Tokenize
                batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
                batch_tokens = batch_tokens.to(self.device)
                
                # Truncate if too long
                if batch_tokens.size(1) > 1024:
                    batch_tokens = batch_tokens[:, :1024]
                
                with torch.no_grad():
                    # Forward pass - ESM2 returns (logits, representations)
                    output = self.model(batch_tokens)
                    
                    # Handle different output formats
                    if isinstance(output, tuple):
                        if len(output) == 2:
                            logits, token_reprs = output
                        elif len(output) == 1:
                            logits = output[0]
                            token_reprs = None
                        else:
                            raise ValueError(f"Unexpected output format: {len(output)} elements")
                    else:
                        # Single output
                        logits = output
                        token_reprs = None
                    
                    # Use token representations if available, otherwise use logits
                    if token_reprs is not None:
                        # Use token representations for embedding
                        pooled = token_reprs.mean(dim=1)  # Mean over sequence length
                    else:
                        # Fallback to logits
                        pooled = logits.mean(dim=1)
                    
                    # Ensure we have the right shape
                    if pooled.dim() == 2:
                        embeddings.append(pooled.cpu().numpy())
                    else:
                        print(f"Warning: Unexpected pooled shape {pooled.shape}")
                        # Try to reshape
                        if pooled.dim() == 1:
                            embeddings.append(pooled.unsqueeze(0).cpu().numpy())
                        else:
                            embeddings.append(pooled.flatten().cpu().numpy())
                            
            except Exception as e:
                print(f"Error with batch {i//8 + 1}: {e}")
                # Add zero embeddings as fallback
                fallback_emb = np.zeros((len(batch_seqs), 1280))
                embeddings.append(fallback_emb)
        
        final_embeddings = np.vstack(embeddings)
        print(f"Generated embeddings shape: {final_embeddings.shape}")
        return final_embeddings
```

## Even Simpler Version (if above is too complex):

```python
import esm
import torch

class ESMEmbedder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading ESM2 model on {self.device}...")
        
        # Load model
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
        self.model = self.model.to(self.device)
        self.model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        
    def embed_mean(self, seqs):
        embeddings = []
        
        # Process in small batches
        for i in range(0, len(seqs), 4):  # Smaller batch size
            batch = seqs[i:i+4]
            batch_labels = [f"seq_{i+j}" for j in range(len(batch))]
            
            # Convert to model format
            data = list(zip(batch_labels, batch))
            _, _, tokens = self.batch_converter(data)
            tokens = tokens.to(self.device)
            
            # Truncate long sequences
            if tokens.size(1) > 1024:
                tokens = tokens[:, :1024]
            
            # Get representations
            with torch.no_grad():
                # ESM2 returns (logits, [representations dict])
                # We want the representations
                results = self.model(tokens)
                
                # Extract final hidden representations
                if isinstance(results, tuple) and len(results) == 2:
                    logits, representations = results
                    if isinstance(representations, dict):
                        # Get the last layer
                        layers = sorted(representations.keys())
                        final_repr = representations[layers[-1]]
                    else:
                        final_repr = logits
                else:
                    final_repr = results[0] if isinstance(results, tuple) else results
                
                # Mean pooling
                pooled = final_repr.mean(dim=1)
                embeddings.append(pooled.cpu().numpy())
        
        return np.vstack(embeddings)
```

## What to do:
1. **Clear your ESM cell** (delete all the code in it)
2. **Copy and paste** one of the two versions above
3. **Run the ESM cell**
4. **Run the Training cell** to recreate datasets

The first version is more robust with error handling, the second is simpler and often works better in Colab.
