#!/usr/bin/env python3
"""
Hybrid ensemble encoder combining ProtT5 and TerpeneMiner (ESM-1v) embeddings.
This combines the realistic attention patterns of ProtT5 with the diversity of ESM-1v.
"""

import torch
import numpy as np
import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass

# Import our individual encoders
from models.prott5_encoder import ProtT5Encoder, ProteinEmbedding as ProtT5Embedding
from models.terpeneminer_encoder import TerpeneMinerEncoder, ProteinEmbedding as TerpeneMinerEmbedding

logger = logging.getLogger(__name__)

@dataclass
class HybridEmbedding:
    """Hybrid protein embedding result combining ProtT5 and TerpeneMiner"""
    sequence: str
    prott5_embedding: np.ndarray
    terpeneminer_embedding: np.ndarray
    combined_embedding: np.ndarray
    prott5_attention_weights: Optional[np.ndarray] = None
    terpeneminer_attention_weights: Optional[np.ndarray] = None
    sequence_length: int = 0
    model_name: str = "Hybrid-ProtT5-TerpeneMiner"

class HybridEnsembleEncoder:
    """Hybrid ensemble encoder combining ProtT5 and TerpeneMiner"""
    
    def __init__(self, 
                 prott5_model_name: str = "Rostlab/prot_t5_xl_half_uniref50-enc",
                 terpeneminer_model_name: str = "facebook/esm1v_t33_650M_UR90S_1",
                 combination_method: str = "concatenate",  # "concatenate", "weighted_average", "learned"
                 max_length: int = 1024,
                 batch_size: int = 8,
                 device=None):
        
        self.combination_method = combination_method
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device if device is not None else self._get_device()
        
        # Initialize individual encoders
        logger.info("Initializing hybrid ensemble encoder...")
        self.prott5_encoder = ProtT5Encoder(
            model_name=prott5_model_name,
            max_length=max_length,
            batch_size=batch_size,
            device=self.device
        )
        
        self.terpeneminer_encoder = TerpeneMinerEncoder(
            model_name=terpeneminer_model_name,
            max_length=max_length,
            batch_size=batch_size,
            device=self.device
        )
        
        logger.info("Hybrid ensemble encoder initialized successfully")
    
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
    
    def _combine_embeddings(self, prott5_emb: np.ndarray, terpeneminer_emb: np.ndarray) -> np.ndarray:
        """Combine ProtT5 and TerpeneMiner embeddings"""
        if self.combination_method == "concatenate":
            # Simple concatenation
            combined = np.concatenate([prott5_emb, terpeneminer_emb], axis=0)
            
        elif self.combination_method == "weighted_average":
            # Weighted average (ProtT5 gets more weight due to better attention patterns)
            prott5_weight = 0.7
            terpeneminer_weight = 0.3
            
            # Normalize embeddings first
            prott5_norm = prott5_emb / np.linalg.norm(prott5_emb)
            terpeneminer_norm = terpeneminer_emb / np.linalg.norm(terpeneminer_emb)
            
            # Weighted average
            combined = prott5_weight * prott5_norm + terpeneminer_weight * terpeneminer_norm
            
        elif self.combination_method == "learned":
            # This would require training a combination layer
            # For now, use concatenation
            combined = np.concatenate([prott5_emb, terpeneminer_emb], axis=0)
            
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")
        
        return combined
    
    def encode_sequence(self, sequence: str, return_attention: bool = False) -> Optional[HybridEmbedding]:
        """Encode a single protein sequence using both encoders"""
        try:
            # Validate sequence
            if not self._validate_sequence(sequence):
                logger.error("Invalid protein sequence")
                return None
            
            logger.info(f"Encoding sequence with hybrid ensemble (length: {len(sequence)})")
            
            # Encode with ProtT5
            prott5_result = self.prott5_encoder.encode_sequence(sequence, return_attention=return_attention)
            if prott5_result is None:
                logger.error("Failed to encode sequence with ProtT5")
                return None
            
            # Encode with TerpeneMiner
            terpeneminer_result = self.terpeneminer_encoder.encode_sequence(sequence, return_attention=return_attention)
            if terpeneminer_result is None:
                logger.error("Failed to encode sequence with TerpeneMiner")
                return None
            
            # Combine embeddings
            combined_embedding = self._combine_embeddings(
                prott5_result.embedding, 
                terpeneminer_result.embedding
            )
            
            logger.info(f"Successfully encoded sequence with hybrid ensemble")
            logger.info(f"  ProtT5 embedding shape: {prott5_result.embedding.shape}")
            logger.info(f"  TerpeneMiner embedding shape: {terpeneminer_result.embedding.shape}")
            logger.info(f"  Combined embedding shape: {combined_embedding.shape}")
            
            return HybridEmbedding(
                sequence=sequence,
                prott5_embedding=prott5_result.embedding,
                terpeneminer_embedding=terpeneminer_result.embedding,
                combined_embedding=combined_embedding,
                prott5_attention_weights=prott5_result.attention_weights if return_attention else None,
                terpeneminer_attention_weights=terpeneminer_result.attention_weights if return_attention else None,
                sequence_length=len(sequence),
                model_name="Hybrid-ProtT5-TerpeneMiner"
            )
            
        except Exception as e:
            logger.error(f"Error encoding sequence with hybrid ensemble: {e}")
            return None
    
    def encode_sequences(self, sequences: List[str], return_attention: bool = False) -> List[HybridEmbedding]:
        """Encode multiple protein sequences using both encoders"""
        logger.info(f"Encoding {len(sequences)} protein sequences with hybrid ensemble...")
        
        embeddings = []
        
        # Process in batches
        for i in range(0, len(sequences), self.batch_size):
            batch_sequences = sequences[i:i+self.batch_size]
            
            batch_embeddings = self._encode_batch(batch_sequences, return_attention)
            embeddings.extend(batch_embeddings)
        
        logger.info(f"Successfully encoded {len(embeddings)} sequences with hybrid ensemble")
        return embeddings
    
    def _encode_batch(self, sequences: List[str], return_attention: bool = False) -> List[HybridEmbedding]:
        """Encode a batch of sequences"""
        embeddings = []
        
        for sequence in sequences:
            embedding = self.encode_sequence(sequence, return_attention)
            if embedding is not None:
                embeddings.append(embedding)
            else:
                logger.warning(f"Failed to encode sequence: {sequence[:50]}...")
        
        return embeddings
    
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
    
    def create_embedding_matrix(self, embeddings: List[HybridEmbedding]) -> Tuple[np.ndarray, List[str]]:
        """Create embedding matrix for machine learning using combined embeddings"""
        if not embeddings:
            return np.array([]), []
        
        embedding_vectors = [emb.combined_embedding for emb in embeddings]
        sequences = [emb.sequence for emb in embeddings]
        
        embedding_matrix = np.array(embedding_vectors)
        
        logger.info(f"Created hybrid embedding matrix: {embedding_matrix.shape}")
        return embedding_matrix, sequences
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the combined embeddings"""
        if self.combination_method == "concatenate":
            return self.prott5_encoder.get_embedding_dimension() + self.terpeneminer_encoder.get_embedding_dimension()
        elif self.combination_method == "weighted_average":
            # Both embeddings should have the same dimension for weighted average
            prott5_dim = self.prott5_encoder.get_embedding_dimension()
            terpeneminer_dim = self.terpeneminer_encoder.get_embedding_dimension()
            if prott5_dim != terpeneminer_dim:
                logger.warning(f"Different embedding dimensions: ProtT5={prott5_dim}, TerpeneMiner={terpeneminer_dim}")
            return max(prott5_dim, terpeneminer_dim)
        else:
            return self.prott5_encoder.get_embedding_dimension() + self.terpeneminer_encoder.get_embedding_dimension()
    
    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            'model_name': 'Hybrid-ProtT5-TerpeneMiner',
            'model_type': 'Hybrid Ensemble',
            'combination_method': self.combination_method,
            'prott5_info': self.prott5_encoder.get_model_info(),
            'terpeneminer_info': self.terpeneminer_encoder.get_model_info(),
            'combined_embedding_dimension': self.get_embedding_dimension(),
            'max_length': self.max_length,
            'device': str(self.device)
        }
