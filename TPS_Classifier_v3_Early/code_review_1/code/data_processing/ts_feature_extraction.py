#!/usr/bin/env python3
"""
Module 2: Multi-Modal Feature Extraction Pipeline for TS-GSD

This script processes the consolidated TS-GSD dataset to generate:
1. ESM2 embeddings (E_PLM) - High-dimensional protein language model features
2. Engineered features (E_Eng) - Categorical + structural placeholders
3. Final PyTorch-ready feature set for Module 3 training

Input: TS-GSD_consolidated.csv (1,273 unique enzymes)
Output: TS-GSD_final_features.pkl (PyTorch-ready features)
"""

import pandas as pd
import numpy as np
import torch
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm
import warnings

# ML and preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler

# Transformers for ESM2
try:
    from transformers import EsmModel, EsmTokenizer
    print("✅ Transformers library imported successfully")
except ImportError:
    print("❌ Transformers not installed. Install with: pip install transformers")
    raise

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class TSFeatureExtractor:
    """
    Multi-modal feature extraction for Terpene Synthase classification
    """
    
    def __init__(self, 
                 model_name: str = "facebook/esm2_t33_650M_UR50D",
                 device: str = None,
                 batch_size: int = 8,
                 eng_feature_dim: int = 64):
        """
        Initialize the TS Feature Extractor
        
        Args:
            model_name: ESM2 model name
            device: PyTorch device (auto-detect if None)
            batch_size: Batch size for ESM2 processing
            eng_feature_dim: Dimension of engineered feature vector
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.eng_feature_dim = eng_feature_dim
        
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initialized TS Feature Extractor")
        logger.info(f"Device: {self.device}")
        logger.info(f"ESM2 Model: {model_name}")
        logger.info(f"Batch Size: {batch_size}")
        logger.info(f"Engineered Feature Dim: {eng_feature_dim}")
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        
    def load_esm2_model(self):
        """
        Load ESM2 model and tokenizer
        """
        logger.info("Loading ESM2 model and tokenizer...")
        
        try:
            # Load tokenizer
            self.tokenizer = EsmTokenizer.from_pretrained(self.model_name)
            
            # Load model
            self.model = EsmModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            logger.info(f"✅ ESM2 model loaded successfully on {self.device}")
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load ESM2 model: {e}")
            raise
    
    def extract_esm2_embeddings(self, sequences: List[str]) -> np.ndarray:
        """
        Extract ESM2 embeddings for protein sequences
        
        Args:
            sequences: List of protein sequences
            
        Returns:
            NumPy array of shape (N, 1280) containing ESM2 embeddings
        """
        if self.model is None or self.tokenizer is None:
            self.load_esm2_model()
        
        logger.info(f"Extracting ESM2 embeddings for {len(sequences)} sequences...")
        
        embeddings = []
        
        # Process sequences in batches
        for i in tqdm(range(0, len(sequences), self.batch_size), desc="ESM2 Processing"):
            batch_sequences = sequences[i:i + self.batch_size]
            
            try:
                # Tokenize sequences
                batch_encoded = self.tokenizer(
                    batch_sequences,
                    padding=True,
                    truncation=True,
                    max_length=1024,  # ESM2 max length
                    return_tensors='pt'
                )
                
                # Move to device
                batch_encoded = {k: v.to(self.device) for k, v in batch_encoded.items()}
                
                # Extract embeddings
                with torch.no_grad():
                    outputs = self.model(**batch_encoded)
                    
                    # Use mean pooling over sequence length (excluding padding)
                    # Shape: (batch_size, seq_len, hidden_size)
                    attention_mask = batch_encoded['attention_mask']
                    token_embeddings = outputs.last_hidden_state
                    
                    # Mask out padding tokens
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    masked_embeddings = token_embeddings * input_mask_expanded
                    
                    # Sum over sequence length and divide by actual length
                    summed = torch.sum(masked_embeddings, 1)
                    summed_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    mean_pooled = summed / summed_mask
                    
                    # Move to CPU and convert to numpy
                    batch_embeddings = mean_pooled.cpu().numpy()
                    embeddings.extend(batch_embeddings)
                    
            except Exception as e:
                logger.warning(f"Error processing batch {i//self.batch_size}: {e}")
                # Add zero embeddings for failed sequences
                batch_size_actual = len(batch_sequences)
                zero_embeddings = np.zeros((batch_size_actual, 1280))
                embeddings.extend(zero_embeddings)
        
        embeddings_array = np.array(embeddings)
        logger.info(f"✅ ESM2 embeddings extracted: {embeddings_array.shape}")
        
        return embeddings_array
    
    def simulate_engineered_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate engineered features (E_Eng) with real categorical data + placeholders
        
        Args:
            df: Consolidated TS-GSD DataFrame
            
        Returns:
            NumPy array of shape (N, eng_feature_dim)
        """
        logger.info(f"Generating engineered features (dim={self.eng_feature_dim})...")
        
        n_samples = len(df)
        features = []
        
        # 1. Real categorical features from MARTS-DB
        
        # Terpene type encoding (one-hot)
        terpene_types = df['terpene_type'].values
        unique_types = np.unique(terpene_types)
        
        # Create one-hot encoding for terpene types
        type_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        type_features = type_encoder.fit_transform(terpene_types.reshape(-1, 1))
        
        # Enzyme class encoding (one-hot)
        enzyme_classes = df['enzyme_class'].astype(str).values
        class_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        class_features = class_encoder.fit_transform(enzyme_classes.reshape(-1, 1))
        
        # Species kingdom encoding (one-hot)
        kingdoms = df['kingdom'].values
        kingdom_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        kingdom_features = kingdom_encoder.fit_transform(kingdoms.reshape(-1, 1))
        
        # Number of products (normalized)
        num_products = df['num_products'].values
        max_products = np.max(num_products)
        normalized_products = num_products / max_products if max_products > 0 else num_products
        
        # Combine real features
        real_features = np.concatenate([
            type_features,           # ~11 dimensions (unique terpene types)
            class_features,          # 2 dimensions (Class 1, Class 2)
            kingdom_features,        # ~3 dimensions (Plantae, etc.)
            normalized_products.reshape(-1, 1)  # 1 dimension
        ], axis=1)
        
        real_feature_dim = real_features.shape[1]
        logger.info(f"Real categorical features: {real_feature_dim} dimensions")
        
        # 2. Placeholder features for structural/mechanistic data
        
        placeholder_dim = self.eng_feature_dim - real_feature_dim
        
        if placeholder_dim > 0:
            # Generate placeholder features (representing future structural data)
            np.random.seed(42)  # For reproducibility
            
            placeholder_features = np.random.uniform(
                0, 1, 
                size=(n_samples, placeholder_dim)
            )
            
            # Combine real and placeholder features
            engineered_features = np.concatenate([real_features, placeholder_features], axis=1)
        else:
            engineered_features = real_features
            # Truncate if we have too many real features
            engineered_features = engineered_features[:, :self.eng_feature_dim]
        
        logger.info(f"✅ Engineered features generated: {engineered_features.shape}")
        logger.info(f"  - Real features: {real_feature_dim} dimensions")
        logger.info(f"  - Placeholder features: {max(0, placeholder_dim)} dimensions")
        
        return engineered_features
    
    def extract_target_vectors(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract target vectors from the consolidated dataset
        
        Args:
            df: Consolidated TS-GSD DataFrame
            
        Returns:
            NumPy array of shape (N, 30) containing binary target vectors
        """
        logger.info("Extracting target vectors...")
        
        target_vectors = []
        
        for _, row in df.iterrows():
            target_str = row['target_vector']
            
            if isinstance(target_str, str):
                # Parse JSON string
                target_vector = json.loads(target_str)
            else:
                # Already a list
                target_vector = target_str
            
            target_vectors.append(target_vector)
        
        target_array = np.array(target_vectors)
        logger.info(f"✅ Target vectors extracted: {target_array.shape}")
        
        return target_array
    
    def create_final_dataset(self, 
                           df: pd.DataFrame,
                           E_plm: np.ndarray,
                           E_eng: np.ndarray,
                           Y: np.ndarray) -> Dict:
        """
        Create the final PyTorch-ready dataset
        
        Args:
            df: Original DataFrame
            E_plm: ESM2 embeddings
            E_eng: Engineered features
            Y: Target vectors
            
        Returns:
            Dictionary containing all features and metadata
        """
        logger.info("Creating final PyTorch-ready dataset...")
        
        # Create final dataset dictionary
        final_dataset = {
            # Core features
            'E_plm': E_plm.astype(np.float32),      # ESM2 embeddings
            'E_eng': E_eng.astype(np.float32),      # Engineered features
            'Y': Y.astype(np.int64),                # Target vectors
            
            # Metadata
            'uniprot_ids': df['uniprot_accession_id'].values,
            'enzyme_names': df['enzyme_name'].values,
            'species': df['species'].values,
            'terpene_types': df['terpene_type'].values,
            'enzyme_classes': df['enzyme_class'].values,
            'num_products': df['num_products'].values,
            
            # Feature dimensions
            'plm_dim': E_plm.shape[1],
            'eng_dim': E_eng.shape[1],
            'n_classes': Y.shape[1],
            'n_samples': len(df),
            
            # Model configuration
            'model_name': self.model_name,
            'device': str(self.device),
            'batch_size': self.batch_size
        }
        
        logger.info("✅ Final dataset created with features:")
        logger.info(f"  - E_PLM: {final_dataset['E_plm'].shape}")
        logger.info(f"  - E_Eng: {final_dataset['E_eng'].shape}")
        logger.info(f"  - Y: {final_dataset['Y'].shape}")
        
        return final_dataset
    
    def save_final_features(self, dataset: Dict, output_path: str = "TS-GSD_final_features.pkl"):
        """
        Save the final features to disk
        
        Args:
            dataset: Final dataset dictionary
            output_path: Output file path
        """
        logger.info(f"Saving final features to {output_path}...")
        
        with open(output_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        logger.info(f"✅ Final features saved to {output_path}")
        
        # Save metadata separately for easy inspection
        metadata_path = output_path.replace('.pkl', '_metadata.json')
        metadata = {
            'n_samples': dataset['n_samples'],
            'plm_dim': dataset['plm_dim'],
            'eng_dim': dataset['eng_dim'],
            'n_classes': dataset['n_classes'],
            'model_name': dataset['model_name'],
            'device': dataset['device'],
            'batch_size': dataset['batch_size'],
            'feature_shapes': {
                'E_plm': list(dataset['E_plm'].shape),
                'E_eng': list(dataset['E_eng'].shape),
                'Y': list(dataset['Y'].shape)
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Metadata saved to {metadata_path}")
    
    def run_feature_extraction(self, 
                              input_path: str = "TS-GSD_consolidated.csv",
                              output_path: str = "TS-GSD_final_features.pkl",
                              skip_esm2: bool = False) -> str:
        """
        Run the complete feature extraction pipeline
        
        Args:
            input_path: Path to consolidated TS-GSD CSV
            output_path: Path for output pickle file
            skip_esm2: Skip ESM2 extraction (for testing)
            
        Returns:
            Path to the created feature file
        """
        logger.info("Starting TS Feature Extraction Pipeline...")
        
        try:
            # Step 1: Load consolidated dataset
            logger.info(f"Loading consolidated dataset from {input_path}...")
            df = pd.read_csv(input_path)
            logger.info(f"Loaded {len(df)} enzymes from consolidated dataset")
            
            # Step 2: Extract target vectors
            Y = self.extract_target_vectors(df)
            
            # Step 3: Generate engineered features
            E_eng = self.simulate_engineered_features(df)
            
            # Step 4: Extract ESM2 embeddings (if not skipped)
            if skip_esm2:
                logger.info("Skipping ESM2 extraction (testing mode)")
                # Create dummy ESM2 embeddings for testing
                E_plm = np.random.randn(len(df), 1280).astype(np.float32)
            else:
                sequences = df['aa_sequence'].tolist()
                E_plm = self.extract_esm2_embeddings(sequences)
            
            # Step 5: Create final dataset
            final_dataset = self.create_final_dataset(df, E_plm, E_eng, Y)
            
            # Step 6: Save final features
            self.save_final_features(final_dataset, output_path)
            
            logger.info("✅ Feature extraction pipeline completed successfully!")
            
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Feature extraction failed: {e}")
            raise


def main():
    """Main function to run the feature extraction pipeline"""
    
    # Initialize feature extractor
    extractor = TSFeatureExtractor(
        model_name="facebook/esm2_t33_650M_UR50D",
        device=None,  # Auto-detect
        batch_size=8,
        eng_feature_dim=64
    )
    
    print("🧬 TS Feature Extraction Pipeline - Module 2")
    print("=" * 60)
    
    # Check if GPU is available
    if torch.cuda.is_available():
        print(f"🚀 GPU available: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  GPU not available - using CPU (will be slower)")
    
    print()
    
    # Run feature extraction
    try:
        output_path = extractor.run_feature_extraction(
            input_path="TS-GSD_consolidated.csv",
            output_path="TS-GSD_final_features.pkl",
            skip_esm2=False  # Set to True for testing without ESM2
        )
        
        print(f"\n🎉 Module 2 Complete!")
        print(f"📁 Final features saved to: {output_path}")
        print(f"🎯 Ready for Module 3: Multi-Modal Deep Learning Training")
        
    except Exception as e:
        print(f"\n❌ Feature extraction failed: {e}")
        print("💡 Try setting skip_esm2=True for testing without GPU")


if __name__ == "__main__":
    main()
