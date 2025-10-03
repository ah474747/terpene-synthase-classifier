"""
Feature extraction module for terpene synthase protein sequences.
This module converts protein sequences into numerical features for machine learning.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils import molecular_weight
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import re
from collections import Counter
import joblib
import os


class ProteinFeatureExtractor:
    """
    Extracts various features from protein sequences for machine learning.
    """
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.aa_properties = self._load_amino_acid_properties()
        self.scaler = StandardScaler()
        self.tfidf_vectorizer = None
        
    def _load_amino_acid_properties(self) -> Dict[str, Dict[str, float]]:
        """Load amino acid physicochemical properties."""
        return {
            'A': {'hydrophobicity': 1.8, 'charge': 0, 'size': 89, 'polarity': 0},
            'R': {'hydrophobicity': -4.5, 'charge': 1, 'size': 174, 'polarity': 1},
            'N': {'hydrophobicity': -3.5, 'charge': 0, 'size': 132, 'polarity': 1},
            'D': {'hydrophobicity': -3.5, 'charge': -1, 'size': 133, 'polarity': 1},
            'C': {'hydrophobicity': 2.5, 'charge': 0, 'size': 121, 'polarity': 0},
            'Q': {'hydrophobicity': -3.5, 'charge': 0, 'size': 146, 'polarity': 1},
            'E': {'hydrophobicity': -3.5, 'charge': -1, 'size': 147, 'polarity': 1},
            'G': {'hydrophobicity': -0.4, 'charge': 0, 'size': 75, 'polarity': 0},
            'H': {'hydrophobicity': -3.2, 'charge': 0.5, 'size': 155, 'polarity': 1},
            'I': {'hydrophobicity': 4.5, 'charge': 0, 'size': 131, 'polarity': 0},
            'L': {'hydrophobicity': 3.8, 'charge': 0, 'size': 131, 'polarity': 0},
            'K': {'hydrophobicity': -3.9, 'charge': 1, 'size': 146, 'polarity': 1},
            'M': {'hydrophobicity': 1.9, 'charge': 0, 'size': 149, 'polarity': 0},
            'F': {'hydrophobicity': 2.8, 'charge': 0, 'size': 165, 'polarity': 0},
            'P': {'hydrophobicity': -1.6, 'charge': 0, 'size': 115, 'polarity': 0},
            'S': {'hydrophobicity': -0.8, 'charge': 0, 'size': 105, 'polarity': 1},
            'T': {'hydrophobicity': -0.7, 'charge': 0, 'size': 119, 'polarity': 1},
            'W': {'hydrophobicity': -0.9, 'charge': 0, 'size': 204, 'polarity': 0},
            'Y': {'hydrophobicity': -1.3, 'charge': 0, 'size': 181, 'polarity': 1},
            'V': {'hydrophobicity': 4.2, 'charge': 0, 'size': 117, 'polarity': 0}
        }
    
    def extract_all_features(self, sequences: List[str], sequence_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Extract all features from protein sequences.
        
        Args:
            sequences: List of protein sequences
            sequence_ids: Optional list of sequence identifiers
            
        Returns:
            DataFrame with extracted features
        """
        print(f"Extracting features from {len(sequences)} sequences...")
        
        features_list = []
        
        for i, sequence in enumerate(sequences):
            if sequence_ids:
                seq_id = sequence_ids[i]
            else:
                seq_id = f"seq_{i}"
            
            # Extract different types of features
            features = {
                'sequence_id': seq_id,
                'sequence_length': len(sequence)
            }
            
            # Basic sequence features
            features.update(self._extract_basic_features(sequence))
            
            # Amino acid composition
            features.update(self._extract_aa_composition(sequence))
            
            # Physicochemical properties
            features.update(self._extract_physicochemical_features(sequence))
            
            # K-mer features
            features.update(self._extract_kmer_features(sequence))
            
            # Motif features
            features.update(self._extract_motif_features(sequence))
            
            # Secondary structure prediction features
            features.update(self._extract_secondary_structure_features(sequence))
            
            features_list.append(features)
        
        # Convert to DataFrame
        df = pd.DataFrame(features_list)
        
        # Handle missing values
        df = df.fillna(0)
        
        print(f"Extracted {len(df.columns)-1} features per sequence")
        return df
    
    def _extract_basic_features(self, sequence: str) -> Dict[str, float]:
        """Extract basic sequence features."""
        if not sequence:
            return {}
        
        try:
            analysis = ProteinAnalysis(sequence)
            
            return {
                'molecular_weight': analysis.molecular_weight(),
                'isoelectric_point': analysis.isoelectric_point(),
                'instability_index': analysis.instability_index(),
                'aromaticity': analysis.aromaticity(),
                'gravy': analysis.gravy()  # Grand average of hydropathy
            }
        except Exception as e:
            print(f"Error in basic feature extraction: {e}")
            return {
                'molecular_weight': 0,
                'isoelectric_point': 0,
                'instability_index': 0,
                'aromaticity': 0,
                'gravy': 0
            }
    
    def _extract_aa_composition(self, sequence: str) -> Dict[str, float]:
        """Extract amino acid composition features."""
        if not sequence:
            return {}
        
        # Count amino acids
        aa_counts = Counter(sequence.upper())
        total_length = len(sequence)
        
        # Calculate composition percentages
        composition = {}
        for aa in 'ACDEFGHIKLMNPQRSTVWY':
            count = aa_counts.get(aa, 0)
            composition[f'aa_comp_{aa}'] = count / total_length if total_length > 0 else 0
        
        return composition
    
    def _extract_physicochemical_features(self, sequence: str) -> Dict[str, float]:
        """Extract physicochemical property features."""
        if not sequence:
            return {}
        
        features = {}
        total_length = len(sequence)
        
        if total_length == 0:
            return features
        
        # Calculate average properties
        total_hydrophobicity = 0
        total_charge = 0
        total_size = 0
        total_polarity = 0
        
        for aa in sequence.upper():
            if aa in self.aa_properties:
                props = self.aa_properties[aa]
                total_hydrophobicity += props['hydrophobicity']
                total_charge += props['charge']
                total_size += props['size']
                total_polarity += props['polarity']
        
        features['avg_hydrophobicity'] = total_hydrophobicity / total_length
        features['avg_charge'] = total_charge / total_length
        features['avg_size'] = total_size / total_length
        features['avg_polarity'] = total_polarity / total_length
        
        # Calculate property distributions
        hydrophobicity_values = [self.aa_properties[aa]['hydrophobicity'] for aa in sequence.upper() if aa in self.aa_properties]
        if hydrophobicity_values:
            features['hydrophobicity_std'] = np.std(hydrophobicity_values)
            features['hydrophobicity_range'] = max(hydrophobicity_values) - min(hydrophobicity_values)
        
        return features
    
    def _extract_kmer_features(self, sequence: str, k_values: List[int] = [2, 3, 4]) -> Dict[str, float]:
        """Extract k-mer frequency features."""
        if not sequence:
            return {}
        
        features = {}
        sequence_upper = sequence.upper()
        
        for k in k_values:
            # Generate k-mers
            kmers = [sequence_upper[i:i+k] for i in range(len(sequence_upper) - k + 1)]
            kmer_counts = Counter(kmers)
            total_kmers = len(kmers)
            
            # Calculate k-mer frequencies
            for kmer, count in kmer_counts.items():
                if total_kmers > 0:
                    features[f'kmer_{k}_{kmer}'] = count / total_kmers
            
            # Add k-mer diversity metrics
            features[f'kmer_{k}_diversity'] = len(kmer_counts) / total_kmers if total_kmers > 0 else 0
            features[f'kmer_{k}_entropy'] = self._calculate_entropy(kmer_counts.values())
        
        return features
    
    def _calculate_entropy(self, counts: List[int]) -> float:
        """Calculate Shannon entropy of k-mer counts."""
        if not counts:
            return 0
        
        total = sum(counts)
        if total == 0:
            return 0
        
        entropy = 0
        for count in counts:
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _extract_motif_features(self, sequence: str) -> Dict[str, float]:
        """Extract motif-based features."""
        if not sequence:
            return {}
        
        features = {}
        sequence_upper = sequence.upper()
        
        # Common terpene synthase motifs
        motifs = {
            'DDXXD': r'D[^P][^P]D[^P]D',  # Metal binding motif
            'NSE': r'N[^P][^P]E',         # Metal binding motif
            'RRX8W': r'R[^P][^P][^P][^P][^P][^P][^P][^P]W',  # Substrate binding
            'GXGXXG': r'G[^P]G[^P][^P]G',  # ATP binding
            'HXXXH': r'H[^P][^P][^P]H',    # Metal binding
        }
        
        for motif_name, pattern in motifs.items():
            matches = len(re.findall(pattern, sequence_upper))
            features[f'motif_{motif_name}'] = matches
        
        # Calculate motif density
        total_motifs = sum(features.values())
        features['motif_density'] = total_motifs / len(sequence) if len(sequence) > 0 else 0
        
        return features
    
    def _extract_secondary_structure_features(self, sequence: str) -> Dict[str, float]:
        """Extract secondary structure prediction features."""
        if not sequence:
            return {}
        
        features = {}
        
        # Simple secondary structure prediction based on amino acid propensities
        helix_propensity = {'A': 1.42, 'E': 1.51, 'L': 1.21, 'M': 1.45, 'Q': 1.11, 'K': 1.16, 'R': 0.98}
        sheet_propensity = {'V': 1.70, 'I': 1.60, 'Y': 1.47, 'F': 1.38, 'W': 1.37, 'L': 1.30, 'T': 1.19}
        
        helix_score = sum(helix_propensity.get(aa, 0) for aa in sequence.upper())
        sheet_score = sum(sheet_propensity.get(aa, 0) for aa in sequence.upper())
        
        features['helix_propensity'] = helix_score / len(sequence) if len(sequence) > 0 else 0
        features['sheet_propensity'] = sheet_score / len(sequence) if len(sequence) > 0 else 0
        
        return features
    
    def extract_sequence_embeddings(self, sequences: List[str], method: str = "tfidf") -> np.ndarray:
        """
        Extract sequence embeddings using various methods.
        
        Args:
            sequences: List of protein sequences
            method: Embedding method ('tfidf', 'kmer', 'onehot')
            
        Returns:
            Array of sequence embeddings
        """
        if method == "tfidf":
            return self._extract_tfidf_embeddings(sequences)
        elif method == "kmer":
            return self._extract_kmer_embeddings(sequences)
        elif method == "onehot":
            return self._extract_onehot_embeddings(sequences)
        else:
            raise ValueError(f"Unknown embedding method: {method}")
    
    def _extract_tfidf_embeddings(self, sequences: List[str]) -> np.ndarray:
        """Extract TF-IDF embeddings from sequences."""
        # Convert sequences to k-mer strings
        kmer_sequences = []
        for seq in sequences:
            kmers = [seq[i:i+3] for i in range(len(seq) - 2)]
            kmer_sequences.append(' '.join(kmers))
        
        # Fit TF-IDF vectorizer
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 1))
            embeddings = self.tfidf_vectorizer.fit_transform(kmer_sequences)
        else:
            embeddings = self.tfidf_vectorizer.transform(kmer_sequences)
        
        return embeddings.toarray()
    
    def _extract_kmer_embeddings(self, sequences: List[str], k: int = 3) -> np.ndarray:
        """Extract k-mer frequency embeddings."""
        # Get all possible k-mers
        all_kmers = set()
        for seq in sequences:
            for i in range(len(seq) - k + 1):
                all_kmers.add(seq[i:i+k])
        
        # Create k-mer to index mapping
        kmer_to_idx = {kmer: i for i, kmer in enumerate(sorted(all_kmers))}
        
        # Create embeddings
        embeddings = np.zeros((len(sequences), len(kmer_to_idx)))
        for i, seq in enumerate(sequences):
            for j in range(len(seq) - k + 1):
                kmer = seq[j:j+k]
                if kmer in kmer_to_idx:
                    embeddings[i, kmer_to_idx[kmer]] += 1
        
        # Normalize by sequence length
        for i in range(len(sequences)):
            if len(sequences[i]) > 0:
                embeddings[i] /= len(sequences[i])
        
        return embeddings
    
    def _extract_onehot_embeddings(self, sequences: List[str]) -> np.ndarray:
        """Extract one-hot encoded embeddings."""
        # Pad sequences to same length
        max_length = max(len(seq) for seq in sequences) if sequences else 0
        
        # Create one-hot encoding
        embeddings = np.zeros((len(sequences), max_length, 20))  # 20 amino acids
        
        aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
        
        for i, seq in enumerate(sequences):
            for j, aa in enumerate(seq):
                if aa in aa_to_idx and j < max_length:
                    embeddings[i, j, aa_to_idx[aa]] = 1
        
        # Flatten to 2D
        embeddings = embeddings.reshape(len(sequences), -1)
        
        return embeddings
    
    def save_features(self, features_df: pd.DataFrame, filename: str = "protein_features.csv"):
        """Save extracted features to CSV file."""
        os.makedirs("data", exist_ok=True)
        filepath = os.path.join("data", filename)
        features_df.to_csv(filepath, index=False)
        print(f"Features saved to {filepath}")
    
    def load_features(self, filename: str = "protein_features.csv") -> pd.DataFrame:
        """Load features from CSV file."""
        filepath = os.path.join("data", filename)
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        else:
            print(f"File {filepath} not found")
            return pd.DataFrame()


def main():
    """Main function to demonstrate feature extraction."""
    # Sample sequences for testing
    sample_sequences = [
        "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL",
        "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL",
        "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL"
    ]
    
    # Initialize feature extractor
    extractor = ProteinFeatureExtractor()
    
    # Extract features
    features_df = extractor.extract_all_features(sample_sequences)
    
    print("Extracted features:")
    print(features_df.head())
    print(f"\nFeature matrix shape: {features_df.shape}")
    
    # Extract embeddings
    embeddings = extractor.extract_sequence_embeddings(sample_sequences, method="tfidf")
    print(f"Embedding matrix shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
