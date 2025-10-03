#!/usr/bin/env python3
"""
Robust embedding generator that handles memory issues and model loading problems
"""

import os
import gc
import torch
import numpy as np
import pandas as pd
from typing import List, Optional
from transformers import EsmModel, EsmTokenizer
import warnings

warnings.filterwarnings('ignore')


class RobustEmbeddingGenerator:
    """
    Robust embedding generator that handles memory issues and model loading
    """
    
    def __init__(self, model_name: str = 'esm2_t33_650M_UR50D', max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cpu')  # Force CPU to avoid CUDA issues
        
        # Memory management settings
        self.batch_size = 2  # Very small batch size
        self.max_sequences = None  # No limit for full dataset training
        
    def _load_model_safely(self):
        """Load the ESM model with error handling"""
        if self.model is not None:
            return True
            
        try:
            print(f"Loading ESM model: {self.model_name} (this may take a few minutes...)")
            
            # Load tokenizer first
            self.tokenizer = EsmTokenizer.from_pretrained(f"facebook/{self.model_name}")
            
            # Load model with specific settings for stability
            self.model = EsmModel.from_pretrained(
                f"facebook/{self.model_name}",
                torch_dtype=torch.float32,  # Use float32 for stability
                low_cpu_mem_usage=True,
                device_map=None  # Don't use device_map
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            print("✓ Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"✗ Model loading failed: {e}")
            return False
    
    def _process_single_sequence(self, sequence: str) -> Optional[np.ndarray]:
        """Process a single sequence safely"""
        try:
            # Truncate sequence if too long
            if len(sequence) > self.max_length:
                sequence = sequence[:self.max_length]
            
            # Tokenize
            inputs = self.tokenizer(
                sequence,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                last_hidden_states = outputs.last_hidden_state
                
                # Average pool over sequence length
                attention_mask = inputs['attention_mask']
                sequence_lengths = attention_mask.sum(dim=1, keepdim=True)
                
                masked_embeddings = last_hidden_states * attention_mask.unsqueeze(-1)
                pooled_embeddings = masked_embeddings.sum(dim=1) / sequence_lengths
                
                return pooled_embeddings.cpu().numpy()[0]
                
        except Exception as e:
            print(f"Error processing sequence: {e}")
            return None
    
    def generate_embeddings(self, sequences: List[str]) -> pd.DataFrame:
        """
        Generate embeddings for a list of sequences
        """
        print(f"Generating embeddings for {len(sequences)} sequences...")
        
        # Limit sequences if specified
        if self.max_sequences is not None and len(sequences) > self.max_sequences:
            print(f"Limiting to first {self.max_sequences} sequences")
            sequences = sequences[:self.max_sequences]
        
        # Load model
        if not self._load_model_safely():
            raise RuntimeError("Could not load model")
        
        embeddings = []
        valid_indices = []
        
        # Process sequences one by one to avoid memory issues
        for i, sequence in enumerate(sequences):
            if i % 10 == 0:
                print(f"Processing sequence {i+1}/{len(sequences)}")
            
            embedding = self._process_single_sequence(sequence)
            
            if embedding is not None:
                embeddings.append(embedding)
                valid_indices.append(i)
            else:
                print(f"Skipping sequence {i+1} due to processing error")
            
            # Clear cache periodically
            if i % 20 == 0:
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        if not embeddings:
            raise RuntimeError("No embeddings generated")
        
        # Create DataFrame
        embedding_df = pd.DataFrame({
            'id': [f"seq_{i}" for i in valid_indices],
            'embedding': embeddings
        })
        
        print(f"✓ Generated {len(embeddings)} embeddings successfully")
        print(f"✓ Embedding dimension: {len(embeddings[0])}")
        
        return embedding_df


def test_embedding_generation():
    """Test the embedding generation with a small sample"""
    print("Testing embedding generation...")
    
    # Create test sequences
    test_sequences = [
        "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL",
        "MSVSLSFAASATFGFRGGLGGFSRPAAAIKQWRCLPRIQCHSAEQSQSPLRRSGNYQPSIWTHDRIQSLTLSHTADEDDHGERIKLLKCQTNKLMEEKKGEVGEQLQLIDHLQQLGVAYHFKDEIKDTLRGFYASFEDISLQFKDNLHASALLFRLLRENGFSVSEDIFKKFKDDQKGQFEDRLQSQAEGLLSLYEASYLEKDGEELLHEAREFTTKHLKNLLEEEGSLKPGLIREQVAYALELPLNRRFQRLHTKWFIGAWQRDPTMDPALLLLAKLDFNALQNMYKRELNEVSRWWTDLGLPQKLPFFRDRLTENYLWAVVFAFEPDSWAFREMDTKTNCFITMIDDVYDVYGTLDELELFTDIMERWDVNAIDKLPEYMKICFLAVFNTVNDAGYEVMRDKGVNIIPYLKRAWAELCKMYMREARWYHTGYTPTLDEYLDGAWISISGALILSTAYCMGKDLTKEDLDKFSTYPSIVQPSCMLLRLHDDFGTSTEELARGDVQKAVQCCMHERKVPEAVAREHIKQVMEAKWRVLNGNRVAASSFEEYFQNVAINLPRAAQFFYGKGDGYANADGETQKQVMSLLIEPVQ"
    ]
    
    try:
        generator = RobustEmbeddingGenerator()
        embeddings_df = generator.generate_embeddings(test_sequences)
        
        print("✓ Embedding generation test successful!")
        print(f"Generated embeddings shape: {embeddings_df.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Embedding generation test failed: {e}")
        return False


if __name__ == "__main__":
    test_embedding_generation()
