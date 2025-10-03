"""
Configuration file for Terpene Synthase Product Predictor v2

This file contains all configuration parameters for the predictor.
"""

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TerpenePredictorConfig:
    """Unified configuration for the entire system"""
    
    # Data parameters
    min_sequence_length: int = 50
    max_sequence_length: int = 2000
    min_confidence: float = 0.7
    min_samples_per_class: int = 10
    max_samples_per_class: int = 1000
    
    # Model parameters
    saprot_model_name: str = "nferruz/SaProt_650M"
    protein_embedding_dim: int = 1280
    molecular_fingerprint_dim: int = 2223
    hidden_dim: int = 512
    num_attention_heads: int = 8
    dropout_rate: float = 0.3
    
    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    early_stopping_patience: int = 10
    weight_decay: float = 1e-5
    
    # Validation parameters
    cv_folds: int = 5
    test_size: float = 0.2
    val_size: float = 0.2
    holdout_fraction: float = 0.2
    
    # System parameters
    device: str = "auto"
    cache_dir: str = "data/cache"
    random_state: int = 42
    log_level: str = "INFO"
    
    # Data sources
    use_marts_db: bool = True
    
    # Uncertainty quantification
    uncertainty_threshold: float = 0.8
    num_monte_carlo_samples: int = 100
    
    # Performance thresholds
    min_accuracy_threshold: float = 0.7
    min_f1_threshold: float = 0.6
    
    # Output settings
    save_models: bool = True
    save_results: bool = True
    plot_results: bool = True

# Default configuration
DEFAULT_CONFIG = TerpenePredictorConfig()

# Terpene product definitions
TERPENE_PRODUCTS = [
    "limonene",
    "pinene", 
    "myrcene",
    "linalool",
    "germacrene_a",
    "germacrene_d",
    "caryophyllene",
    "humulene",
    "farnesene",
    "bisabolene"
]

# Terpene product SMILES database
TERPENE_SMILES = {
    "limonene": "CC1=CCC(CC1)C(=C)C",
    "pinene": "CC1=CCC2CC1C2(C)C",
    "myrcene": "CC(=CCCC(=C)C)C",
    "linalool": "CC(C)=CCCC(C)(C=C)O",
    "germacrene_a": "CC1=CCCC(=C)C2CC1C2(C)C",
    "germacrene_d": "CC1=CCCC(=C)C2CC1C2(C)C",
    "caryophyllene": "CC1=CCCC(=C)C2CC1C2(C)C",
    "humulene": "CC1=CCCC(=C)C2CC1C2(C)C",
    "farnesene": "CC(=CCCC(=CCCC(=C)C)C)C",
    "bisabolene": "CC1=CCCC(=C)C2CC1C2(C)C",
}

# Known terpene synthase EC numbers
TERPENE_EC_NUMBERS = [
    "4.2.3.27",  # Limonene synthase
    "4.2.3.20",  # Pinene synthase
    "4.2.3.15",  # Myrcene synthase
    "4.2.3.14",  # Linalool synthase
    "4.2.3.70",  # Germacrene A synthase
    "4.2.3.75",  # Germacrene D synthase
    "4.2.3.97",  # Caryophyllene synthase
    "4.2.3.46",  # Humulene synthase
    "4.2.3.26",  # Farnesene synthase
    "4.2.3.47",  # Bisabolene synthase
]

def validate_config(config: TerpenePredictorConfig) -> bool:
    """Validate configuration parameters"""
    
    # Check numeric ranges
    if config.min_sequence_length < 10:
        raise ValueError("min_sequence_length must be >= 10")
    
    if config.max_sequence_length < config.min_sequence_length:
        raise ValueError("max_sequence_length must be >= min_sequence_length")
    
    if not 0 < config.min_confidence <= 1:
        raise ValueError("min_confidence must be between 0 and 1")
    
    if config.min_samples_per_class < 1:
        raise ValueError("min_samples_per_class must be >= 1")
    
    if config.max_samples_per_class < config.min_samples_per_class:
        raise ValueError("max_samples_per_class must be >= min_samples_per_class")
    
    # Check model parameters
    if config.protein_embedding_dim <= 0:
        raise ValueError("protein_embedding_dim must be > 0")
    
    if config.molecular_fingerprint_dim <= 0:
        raise ValueError("molecular_fingerprint_dim must be > 0")
    
    if config.hidden_dim <= 0:
        raise ValueError("hidden_dim must be > 0")
    
    if config.num_attention_heads <= 0:
        raise ValueError("num_attention_heads must be > 0")
    
    if config.hidden_dim % config.num_attention_heads != 0:
        raise ValueError("hidden_dim must be divisible by num_attention_heads")
    
    # Check training parameters
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    
    if config.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    
    if config.num_epochs <= 0:
        raise ValueError("num_epochs must be > 0")
    
    # Check validation parameters
    if not 0 < config.test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    
    if not 0 < config.val_size < 1:
        raise ValueError("val_size must be between 0 and 1")
    
    if config.test_size + config.val_size >= 1:
        raise ValueError("test_size + val_size must be < 1")
    
    if not 0 < config.holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1")
    
    return True

# Terpene product SMILES
TERPENE_SMILES = {
    "limonene": "CC1=CCC(CC1)C(=C)C",
    "pinene": "CC1=CCC2CC1C2(C)C",
    "myrcene": "CC(=CCCC(=C)C)C",
    "linalool": "CC(C)=CCCC(C)(C=C)O",
    "germacrene_a": "CC1=CCCC(=C)C2CC1C2(C)C",
    "germacrene_d": "CC1=CCCC(=C)C2CC1C2(C)C",
    "caryophyllene": "CC1=CCCC(=C)C2CC1C2(C)C",
    "humulene": "CC1=CCCC(=C)C2CC1C2(C)C",
    "farnesene": "CC(=CCCC(=CCCC(=C)C)C)C",
    "bisabolene": "CC1=CCCC(=C)C2CC1C2(C)C"
}

# EC numbers for terpene synthases
TERPENE_EC_NUMBERS = [
    "4.2.3.27",  # Limonene synthase
    "4.2.3.20",  # Pinene synthase
    "4.2.3.15",  # Myrcene synthase
    "4.2.3.14",  # Linalool synthase
    "4.2.3.70",  # Germacrene A synthase
    "4.2.3.75",  # Germacrene D synthase
    "4.2.3.97",  # Caryophyllene synthase
    "4.2.3.46",  # Humulene synthase
    "4.2.3.26",  # Farnesene synthase
    "4.2.3.47",  # Bisabolene synthase
]
