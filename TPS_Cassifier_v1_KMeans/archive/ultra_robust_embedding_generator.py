#!/usr/bin/env python3
"""
Ultra-robust embedding generator with maximum memory management
"""

import os
import gc
import torch
import numpy as np
import pandas as pd
from typing import List, Optional
from transformers import EsmModel, EsmTokenizer
import warnings
import time

warnings.filterwarnings('ignore')


class UltraRobustEmbeddingGenerator:
    """
    Ultra-robust embedding generator with maximum memory management
    """
    
    def __init__(self, model_name: str = 'esm2_t33_650M_UR50D', max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cpu')
        
        # Ultra-conservative memory settings
        self.batch_size = 1  # Single sequence at a time
        self.chunk_size = 10  # Process only 10 sequences per chunk
        self.max_sequences = None  # No limit for full dataset training
        
        # Memory cleanup frequency
        self.cleanup_frequency = 5  # Clean up every 5 sequences
        
    def _load_model_safely(self):
        """Load the ESM model with maximum error handling"""
        if self.model is not None:
            return True
            
        try:
            print(f"Loading ESM model: {self.model_name} (this may take a few minutes...)")
            
            # Clear any existing models
            if hasattr(self, 'model') and self.model is not None:
                del self.model
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
            
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Load tokenizer first
            self.tokenizer = EsmTokenizer.from_pretrained(f"facebook/{self.model_name}")
            
            # Load model with ultra-conservative settings
            self.model = EsmModel.from_pretrained(
                f"facebook/{self.model_name}",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                device_map=None,
                trust_remote_code=True
            )
            
            self.model.eval()
            print("✓ ESM model loaded successfully")
            return True
            
        except Exception as e:
            print(f"✗ Failed to load ESM model: {e}")
            return False
    
    def _cleanup_memory(self):
        """Aggressive memory cleanup"""
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(0.1)  # Brief pause to let system catch up
        except Exception as e:
            print(f"Warning: Memory cleanup failed: {e}")
    
    def _generate_sequence_embedding(self, sequence: str) -> np.ndarray:
        """Generate embedding for a single sequence with maximum robustness"""
        try:
            # Truncate sequence if too long
            if len(sequence) > self.max_length:
                sequence = sequence[:self.max_length]
            
            # Tokenize
            inputs = self.tokenizer(sequence, return_tensors="pt", padding=True, truncation=True)
            
            # Move to CPU device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling of last hidden states
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
            
            # Convert to numpy and cleanup
            embedding = embeddings.cpu().numpy()
            del inputs, outputs, embeddings
            
            return embedding
            
        except Exception as e:
            print(f"Warning: Failed to generate embedding for sequence (length: {len(sequence)}): {e}")
            # Return zero embedding as fallback
            return np.zeros(1280)  # ESM-2 embedding dimension
    
    def generate_embeddings(self, sequences: List[str]) -> pd.DataFrame:
        """
        Generate embeddings with ultra-conservative memory management
        """
        print(f"Generating embeddings for {len(sequences)} sequences...")
        
        # Limit sequences if specified
        if self.max_sequences is not None and len(sequences) > self.max_sequences:
            print(f"Limiting to first {self.max_sequences} sequences")
            sequences = sequences[:self.max_sequences]
        
        # Load model
        if not self._load_model_safely():
            raise RuntimeError("Could not load model")
        
        all_embeddings = []
        
        # Process in very small chunks
        total_chunks = (len(sequences) + self.chunk_size - 1) // self.chunk_size
        print(f"Processing {len(sequences)} sequences in {total_chunks} chunks of {self.chunk_size}")
        
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, len(sequences))
            chunk_sequences = sequences[start_idx:end_idx]
            
            print(f"Processing chunk {chunk_idx + 1}/{total_chunks} ({len(chunk_sequences)} sequences)")
            
            chunk_embeddings = []
            for seq_idx, sequence in enumerate(chunk_sequences):
                print(f"  Processing sequence {seq_idx + 1}/{len(chunk_sequences)}")
                
                try:
                    embedding = self._generate_sequence_embedding(sequence)
                    chunk_embeddings.append(embedding)
                    
                    # Cleanup every few sequences
                    if (seq_idx + 1) % self.cleanup_frequency == 0:
                        self._cleanup_memory()
                        
                except Exception as e:
                    print(f"  Warning: Failed sequence {seq_idx + 1}: {e}")
                    # Add zero embedding as fallback
                    chunk_embeddings.append(np.zeros(1280))
            
            # Convert chunk to numpy array
            chunk_array = np.array(chunk_embeddings)
            all_embeddings.append(chunk_array)
            
            # Cleanup after each chunk
            del chunk_embeddings, chunk_array
            self._cleanup_memory()
            
            print(f"  ✓ Completed chunk {chunk_idx + 1}/{total_chunks}")
        
        # Combine all embeddings
        print("Combining all embeddings...")
        final_embeddings = np.vstack(all_embeddings)
        
        # Create DataFrame
        embedding_df = pd.DataFrame(final_embeddings)
        embedding_df['sequence'] = sequences[:len(final_embeddings)]
        
        # Final cleanup
        del all_embeddings, final_embeddings
        self._cleanup_memory()
        
        print(f"✓ Generated {len(embedding_df)} embeddings successfully")
        print(f"✓ Embedding dimension: {embedding_df.shape[1] - 1}")
        
        return embedding_df
    
    def cleanup(self):
        """Clean up model and memory"""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            self._cleanup_memory()
            print("✓ Memory cleanup completed")
            
        except Exception as e:
            print(f"Warning: Cleanup failed: {e}")

