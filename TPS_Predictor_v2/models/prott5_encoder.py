#!/usr/bin/env python3
"""
ProtT5 encoder implementation as a drop-in replacement for SaProtEncoder.
"""

import torch
import numpy as np
import logging
from typing import Optional, List, Tuple
from transformers import T5Tokenizer, T5EncoderModel
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ProteinEmbedding:
    """Protein embedding result"""
    sequence: str
    embedding: np.ndarray
    attention_weights: Optional[np.ndarray] = None
    sequence_length: int = 0
    model_name: str = ""

class ProtT5Encoder:
    """ProtT5 encoder for protein sequences"""
    
    def __init__(self, model_name: str = "Rostlab/prot_t5_xl_half_uniref50-enc", 
                 max_length: int = 1024, batch_size: int = 8, device=None):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device if device is not None else self._get_device()
        
        # Initialize model and tokenizer
        self._load_model()
        
    def _get_device(self):
        """Get the best available device"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        
        logger.info(f"Using device: {device}")
        return device
    
    def _load_model(self):
        """Load ProtT5 model and tokenizer"""
        try:
            logger.info(f"Loading ProtT5 model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(
                self.model_name,
                do_lower_case=False
            )
            
            # Load model
            self.model = T5EncoderModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
            )
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("ProtT5 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading ProtT5 model: {e}")
            raise RuntimeError(f"Failed to load ProtT5 model: {e}")
    
    def _validate_sequence(self, sequence: str) -> bool:
        """Validate protein sequence"""
        if not sequence or len(sequence) == 0:
            return False
        
        # Check for valid amino acids
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        sequence_chars = set(sequence.upper())
        
        if not sequence_chars.issubset(valid_aa):
            invalid_chars = sequence_chars - valid_aa
            logger.warning(f"Invalid amino acids found: {invalid_chars}")
            return False
        
        return True
    
    def encode_sequence(self, sequence: str, return_attention: bool = False) -> Optional[ProteinEmbedding]:
        """Encode a single protein sequence"""
        try:
            # Validate sequence
            if not self._validate_sequence(sequence):
                logger.error("Invalid protein sequence")
                return None
            
            # Truncate if too long
            if len(sequence) > self.max_length:
                sequence = sequence[:self.max_length]
                logger.warning(f"Sequence truncated to {self.max_length} amino acids")
            
            # Format sequence for ProtT5 (add spaces between amino acids)
            formatted_sequence = ' '.join(sequence)
            
            # Tokenize sequence
            inputs = self.tokenizer(
                formatted_sequence,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True, output_attentions=return_attention)
                
                # Extract embeddings (last hidden state)
                embeddings = outputs.last_hidden_state
                
                # Mean pooling over sequence length (excluding padding)
                attention_mask = inputs['attention_mask']
                masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
                pooled_embedding = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                pooled_embedding = pooled_embedding.squeeze(0)
                
                # Convert to numpy
                embedding_array = pooled_embedding.cpu().numpy()
                
                # Extract attention weights if requested
                attention_weights = None
                if return_attention and hasattr(outputs, 'attentions') and outputs.attentions is not None:
                    # ProtT5 returns attention weights from all layers
                    attention_weights = outputs.attentions[-1].cpu().numpy()  # Last layer
                    logger.info(f"Extracted attention weights shape: {attention_weights.shape}")
                
                return ProteinEmbedding(
                    sequence=sequence,
                    embedding=embedding_array,
                    attention_weights=attention_weights,
                    sequence_length=len(sequence),
                    model_name=self.model_name
                )
                
        except Exception as e:
            logger.error(f"Error encoding sequence: {e}")
            return None
    
    def encode_sequences(self, sequences: List[str], return_attention: bool = False) -> List[ProteinEmbedding]:
        """Encode multiple protein sequences"""
        logger.info(f"Encoding {len(sequences)} protein sequences...")
        
        embeddings = []
        
        # Process in batches
        for i in range(0, len(sequences), self.batch_size):
            batch_sequences = sequences[i:i+self.batch_size]
            
            batch_embeddings = self._encode_batch(batch_sequences, return_attention)
            embeddings.extend(batch_embeddings)
        
        logger.info(f"Successfully encoded {len(embeddings)} sequences")
        return embeddings
    
    def _encode_batch(self, sequences: List[str], return_attention: bool = False) -> List[ProteinEmbedding]:
        """Encode a batch of sequences"""
        embeddings = []
        
        for sequence in sequences:
            embedding = self.encode_sequence(sequence, return_attention)
            if embedding is not None:
                embeddings.append(embedding)
            else:
                logger.warning(f"Failed to encode sequence: {sequence[:50]}...")
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings"""
        return self.model.config.d_model
    
    def create_embedding_matrix(self, embeddings: List[ProteinEmbedding]) -> Tuple[np.ndarray, List[str]]:
        """Create embedding matrix for machine learning"""
        if not embeddings:
            return np.array([]), []
        
        embedding_vectors = [emb.embedding for emb in embeddings]
        sequences = [emb.sequence for emb in embeddings]
        
        embedding_matrix = np.array(embedding_vectors)
        
        logger.info(f"Created embedding matrix: {embedding_matrix.shape}")
        return embedding_matrix, sequences
    
    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            'model_name': self.model_name,
            'model_type': 'ProtT5',
            'embedding_dimension': self.get_embedding_dimension(),
            'max_length': self.max_length,
            'device': str(self.device)
        }
