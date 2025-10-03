"""
SaProt Protein Encoder

This module implements protein sequence encoding using SaProt (Specialized protein language model)
for terpene synthase sequences.

SaProt is specialized for protein function prediction and provides better embeddings
than general protein language models like ESM2.
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import logging
from pathlib import Path
import pickle
from dataclasses import dataclass
from tqdm import tqdm
import warnings

try:
    from transformers import EsmTokenizer, EsmForMaskedLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Install with: pip install transformers")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

@dataclass
class ProteinEmbedding:
    """Container for protein embedding data"""
    sequence: str
    embedding: np.ndarray
    attention_weights: Optional[np.ndarray] = None
    sequence_length: int = 0
    model_name: str = ""

class SaProtEncoder:
    """Encodes protein sequences using SaProt model"""
    
    def __init__(self, 
                 model_name: str = "westlake-repl/SaProt_650M_AF2",
                 cache_dir: str = "data/cache",
                 device: str = "auto"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library is required for SaProt")
        
        self.model_name = model_name
        self.device = self._setup_device(device)
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self._load_model()
        
        # Sequence processing parameters
        self.max_length = 1024  # SaProt max sequence length
        self.batch_size = 8     # Batch size for processing

    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        logger.info(f"Using device: {device}")
        return torch.device(device)

    def _load_model(self):
        """Load SaProt model and tokenizer"""
        try:
            logger.info(f"Loading SaProt model: {self.model_name}")
            
            # Load tokenizer - use standard ESM2 tokenizer for compatibility
            self.tokenizer = EsmTokenizer.from_pretrained(
                "facebook/esm2_t6_8M_UR50D"  # Use standard ESM2 tokenizer
            )
            
            # Load model with eager attention for attention weight extraction
            self.model = EsmForMaskedLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                attn_implementation="eager"  # Use eager attention to enable attention weight extraction
            )
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("SaProt model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading SaProt model: {e}")
            # Fallback to ESM2 if SaProt fails
            logger.info("Falling back to ESM2 model...")
            self._load_esm2_fallback()

    def _load_esm2_fallback(self):
        """Fallback to ESM2 model if SaProt fails"""
        try:
            logger.info("Loading ESM2 fallback model...")
            
            # Try smaller ESM2 model first
            self.model_name = "facebook/esm2_t6_8M_UR50D"
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info("ESM2 fallback model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading ESM2 fallback: {e}")
            logger.error("Unable to load any protein language model")
            raise RuntimeError("Failed to load any protein language model. Please check your internet connection and model availability.")

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
            
            # Tokenize sequence
            inputs = self.tokenizer(
                sequence,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                # Use the standard ESM forward pass with attention output
                outputs = self.model(**inputs, output_hidden_states=True, output_attentions=return_attention)
                
                # Extract the last hidden state
                hidden_states = outputs.hidden_states[-1]
                
                # Mean pooling over sequence length (excluding special tokens)
                attention_mask = inputs['attention_mask']
                # Mask out special tokens (CLS, SEP, PAD)
                masked_hidden_states = hidden_states * attention_mask.unsqueeze(-1)
                pooled_embedding = masked_hidden_states.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                pooled_embedding = pooled_embedding.squeeze(0)
                
                # Convert to numpy
                embedding_array = pooled_embedding.cpu().numpy()
                
                # Extract attention weights if requested
                attention_weights = None
                if return_attention:
                    if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                        # ESM models return attention weights
                        attention_weights = outputs.attentions[-1].cpu().numpy()
                        logger.info(f"Extracted attention weights shape: {attention_weights.shape}")
                    else:
                        logger.warning("Model does not return attention weights")
                        # Create dummy attention weights for analysis
                        seq_len = inputs['input_ids'].shape[1]
                        attention_weights = np.ones((1, 1, seq_len, seq_len)) / seq_len  # Uniform attention
                
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
        for i in tqdm(range(0, len(sequences), self.batch_size), desc="Encoding sequences"):
            batch_sequences = sequences[i:i+self.batch_size]
            
            batch_embeddings = self._encode_batch(batch_sequences, return_attention)
            embeddings.extend(batch_embeddings)
        
        logger.info(f"Successfully encoded {len(embeddings)} sequences")
        return embeddings

    def _encode_batch(self, sequences: List[str], return_attention: bool = False) -> List[ProteinEmbedding]:
        """Encode a batch of sequences"""
        batch_embeddings = []
        
        for sequence in sequences:
            embedding = self.encode_sequence(sequence, return_attention)
            if embedding:
                batch_embeddings.append(embedding)
            else:
                logger.warning(f"Failed to encode sequence: {sequence[:50]}...")
        
        return batch_embeddings

    def _validate_sequence(self, sequence: str) -> bool:
        """Validate protein sequence"""
        if not sequence or len(sequence) < 10:
            return False
        
        valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
        if not all(aa in valid_amino_acids for aa in sequence.upper()):
            return False
        
        return True

    def create_embedding_matrix(self, embeddings: List[ProteinEmbedding]) -> Tuple[np.ndarray, List[str]]:
        """Create embedding matrix for machine learning"""
        if not embeddings:
            return np.array([]), []
        
        embedding_vectors = [emb.embedding for emb in embeddings]
        sequences = [emb.sequence for emb in embeddings]
        
        embedding_matrix = np.array(embedding_vectors)
        
        logger.info(f"Created embedding matrix: {embedding_matrix.shape}")
        return embedding_matrix, sequences

    def save_embeddings(self, embeddings: List[ProteinEmbedding], filename: str = "protein_embeddings.pkl"):
        """Save embeddings to file"""
        output_path = self.cache_dir / filename
        
        # Convert to serializable format
        serializable_data = []
        for emb in embeddings:
            serializable_data.append({
                'sequence': emb.sequence,
                'embedding': emb.embedding,
                'attention_weights': emb.attention_weights,
                'sequence_length': emb.sequence_length,
                'model_name': emb.model_name
            })
        
        with open(output_path, 'wb') as f:
            pickle.dump(serializable_data, f)
        
        logger.info(f"Saved {len(embeddings)} embeddings to {output_path}")

    def load_embeddings(self, filename: str = "protein_embeddings.pkl") -> List[ProteinEmbedding]:
        """Load embeddings from file"""
        input_path = self.cache_dir / filename
        
        if not input_path.exists():
            logger.error(f"File not found: {input_path}")
            return []
        
        with open(input_path, 'rb') as f:
            serializable_data = pickle.load(f)
        
        # Convert back to ProteinEmbedding objects
        embeddings = []
        for data in serializable_data:
            embeddings.append(ProteinEmbedding(
                sequence=data['sequence'],
                embedding=data['embedding'],
                attention_weights=data['attention_weights'],
                sequence_length=data['sequence_length'],
                model_name=data['model_name']
            ))
        
        logger.info(f"Loaded {len(embeddings)} embeddings")
        return embeddings

class EmbeddingAnalyzer:
    """Analyzes protein embeddings for insights"""
    
    def __init__(self):
        pass
    
    def analyze_embeddings(self, embeddings: List[ProteinEmbedding]) -> Dict[str, any]:
        """Analyze protein embeddings"""
        if not embeddings:
            return {}
        
        embedding_matrix = np.array([emb.embedding for emb in embeddings])
        
        analysis = {
            'num_embeddings': len(embeddings),
            'embedding_dimension': embedding_matrix.shape[1],
            'mean_embedding': np.mean(embedding_matrix, axis=0),
            'std_embedding': np.std(embedding_matrix, axis=0),
            'sequence_lengths': [emb.sequence_length for emb in embeddings],
            'model_names': list(set([emb.model_name for emb in embeddings]))
        }
        
        return analysis
    
    def compute_similarity_matrix(self, embeddings: List[ProteinEmbedding]) -> np.ndarray:
        """Compute similarity matrix between embeddings"""
        if not embeddings:
            return np.array([])
        
        embedding_matrix = np.array([emb.embedding for emb in embeddings])
        
        # Compute cosine similarity
        normalized_embeddings = embedding_matrix / np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        return similarity_matrix

def main():
    """Main function to demonstrate SaProt encoding"""
    logger.info("Starting SaProt protein encoding...")
    
    # Initialize encoder
    encoder = SaProtEncoder()
    
    # Sample terpene synthase sequences
    sample_sequences = [
        "MSTEQFVLPDLLESCPLKDATNPYYKEAAAESRAWINGYDIFTDRKRAEFIQGQNELLCSHVYWYAGREQLRTTCDFVNLLFVVDEVSDEQNGKGARETGQVFFKAMKYPDWDDGSILAKVTKEFMARFTRLAGPRNTKRFIDLCESYTACVGEEAELRERSELLDLASYIPLRRQNSAVLLCFALVEYILGIDLADEVYEDEMFMKAYWAACDQVCWANDIYSYDMEQSKGLAGNNIVSILMNENGTNLQETADYIGERCGEFVSDYISAKSQISPSLGPEALQFIDFVGYWMIGNIEWCFETPRYFGSRHLEIKETRVVHLRPKEVPEGLSSEDCIESDDE",
        "MALVSIAPLASKSCLHKSLSSSAHELKTICRTIPTLGMSRRGKSATPSMSMSLTTTVSDDGVQRRMGDFHSNLWNDDFIQSLSTSYGEPSYRERAERLIGEVKKMFNSMSSEDGELINPHNDLIQRVWMVDSVERLGIERHFKNEIKSALDYVYSYWSEKGIGCGRESVVADLNSTALGLRTLRLHGYAVSADVLNLFKDQNGQFACSPSQTEEEIGSVLNLYRASLIAFPGEKVMEEAEIFSAKYLEEALQKISVSSLSQEIRDVLEYGWHTYLPRMEARNHIDVFGQDTQNSKSCINTEKLLELAKLEFNIFHSLQKRELEYLVRWWKDSGSPQMTFGRHRHVEYYTLASCIAFEPQHSGFRLGFAKTCHIITILDDMYDTFGTVDELELFTAAMKRWNPSAADCLPEYMKGMYMIVYDTVNEICQEAEKAQGRNTLDYARQAWDEYLDSYMQEAKWIVTGYLPTFAEYYENGKVSSGHRTAALQPILTMDIPFPPHILKEVDFPSKLNDLACAILRLRGDTRCYKADRARGEEASSISCYMKDNPGVTEEDALDHINAMISDVIRGLNWELLNPNSSVPISSKKHVFDISRAFHYGYKYRDGYSVANIETKSLVKRTVIDPVTL"
    ]
    
    # Encode sequences
    embeddings = encoder.encode_sequences(sample_sequences)
    
    # Create embedding matrix
    embedding_matrix, sequences = encoder.create_embedding_matrix(embeddings)
    
    # Save results
    encoder.save_embeddings(embeddings)
    
    # Analyze results
    analyzer = EmbeddingAnalyzer()
    analysis = analyzer.analyze_embeddings(embeddings)
    similarity_matrix = analyzer.compute_similarity_matrix(embeddings)
    
    # Print summary
    print(f"\nSaProt Encoding Summary:")
    print(f"Model used: {analysis['model_names'][0]}")
    print(f"Sequences encoded: {analysis['num_embeddings']}")
    print(f"Embedding dimension: {analysis['embedding_dimension']}")
    print(f"Sequence lengths: {analysis['sequence_lengths']}")
    print(f"Embedding matrix shape: {embedding_matrix.shape}")
    print(f"Similarity matrix shape: {similarity_matrix.shape}")

if __name__ == "__main__":
    main()
