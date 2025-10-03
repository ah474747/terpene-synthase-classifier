"""
Configuration file for Germacrene Synthase Classifier
===================================================

This file contains all configurable parameters for the classifier.
Modify these values to customize the behavior of the classifier.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Data file paths
MARTS_DB_FILE = DATA_DIR / "marts_db.fasta"
UNIPROT_FILE = DATA_DIR / "uniprot_sequences.fasta"
NCBI_FILE = DATA_DIR / "ncbi_sequences.fasta"

# Model configuration
class ModelConfig:
    """Configuration for protein language models"""
    
    # Available models
    AVAILABLE_MODELS = {
        'esm2_t33_650M_UR50D': {
            'model_name': 'facebook/esm2_t33_650M_UR50D',
            'max_length': 1024,
            'embedding_dim': 1280
        },
        'esm2_t36_3B_UR50D': {
            'model_name': 'facebook/esm2_t36_3B_UR50D', 
            'max_length': 1024,
            'embedding_dim': 2560
        },
        'prot_t5_xl_half_uniref50-enc': {
            'model_name': 'Rostlab/prot_t5_xl_half_uniref50-enc',
            'max_length': 512,
            'embedding_dim': 1024
        }
    }
    
    # Default model
    DEFAULT_MODEL = 'esm2_t33_650M_UR50D'
    
    # Batch sizes
    GPU_BATCH_SIZE = 8
    CPU_BATCH_SIZE = 4


# Training configuration
class TrainingConfig:
    """Configuration for model training"""
    
    # Cross-validation
    N_FOLDS = 5
    
    # XGBoost parameters
    XGBOOST_PARAMS = {
        'random_state': 42,
        'eval_metric': 'logloss',
        'early_stopping_rounds': 10,
        'verbosity': 0,
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
    
    # Semi-supervised learning
    CONFIDENCE_THRESHOLD = 0.95
    MIN_SEQUENCE_LENGTH = 50
    MAX_SEQUENCE_LENGTH = 2000


# Evaluation configuration
class EvaluationConfig:
    """Configuration for model evaluation"""
    
    # Metrics to calculate
    METRICS = ['f1_score', 'precision_score', 'recall_score', 'auc_pr']
    
    # Plotting
    FIGURE_SIZE = (12, 10)
    DPI = 300
    
    # Confidence thresholds for prediction interpretation
    HIGH_CONFIDENCE_THRESHOLD = 0.8
    MODERATE_CONFIDENCE_THRESHOLD = 0.6
    LOW_CONFIDENCE_THRESHOLD = 0.4


# Data processing configuration
class DataConfig:
    """Configuration for data processing"""
    
    # Sequence validation
    VALID_AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')
    
    # FASTA parsing
    MAX_SEQUENCES_PER_FILE = 10000  # Limit for memory management
    
    # Product name mapping for MARTS-DB
    GERMACRENE_KEYWORDS = [
        'germacrene', 'germacrene-a', 'germacrene-b', 'germacrene-c', 'germacrene-d',
        'germacrene-e', 'germacrene-f', 'germacrene-g', 'germacrene-h'
    ]
    
    # Other terpene products (for reference)
    OTHER_TERPENE_KEYWORDS = [
        'limonene', 'pinene', 'myrcene', 'ocimene', 'caryophyllene', 'humulene',
        'farnesene', 'bisabolene', 'selinene', 'eudesmol', 'cadinol', 'cedrol',
        'santalene', 'nerolidol', 'linalool', 'terpineol'
    ]


# System configuration
class SystemConfig:
    """System and hardware configuration"""
    
    # Device settings
    DEVICE = 'auto'  # 'auto', 'cpu', 'cuda'
    
    # Memory management
    MAX_MEMORY_USAGE = 0.8  # Fraction of available memory to use
    
    # Parallel processing
    N_JOBS = -1  # Number of parallel jobs (-1 for all cores)
    
    # Random seed
    RANDOM_STATE = 42


# Logging configuration
class LoggingConfig:
    """Configuration for logging"""
    
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = RESULTS_DIR / 'training.log'


# Model persistence
class ModelPersistence:
    """Configuration for model saving/loading"""
    
    MODEL_FILE = MODELS_DIR / 'germacrene_classifier.pkl'
    EMBEDDINGS_FILE = MODELS_DIR / 'embeddings_cache.pkl'
    RESULTS_FILE = RESULTS_DIR / 'training_results.json'
    
    # Cache settings
    CACHE_EMBEDDINGS = True
    EMBEDDING_CACHE_SIZE = 1000


# Validation configuration
class ValidationConfig:
    """Configuration for model validation"""
    
    # Hold-out test set
    TEST_SIZE = 0.2
    
    # Stratification
    STRATIFY = True
    
    # Performance thresholds
    MIN_F1_SCORE = 0.7
    MIN_AUC_PR = 0.75


# Combine all configurations
class Config:
    """Main configuration class combining all settings"""
    
    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.evaluation = EvaluationConfig()
        self.data = DataConfig()
        self.system = SystemConfig()
        self.logging = LoggingConfig()
        self.persistence = ModelPersistence()
        self.validation = ValidationConfig()
    
    def update_model(self, model_name: str):
        """Update model configuration"""
        if model_name not in self.model.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model.DEFAULT_MODEL = model_name
    
    def get_model_config(self, model_name: str = None):
        """Get configuration for a specific model"""
        if model_name is None:
            model_name = self.model.DEFAULT_MODEL
        
        return self.model.AVAILABLE_MODELS[model_name]
    
    def validate_config(self):
        """Validate configuration settings"""
        # Check if model directories exist
        for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
        
        # Validate model name
        if self.model.DEFAULT_MODEL not in self.model.AVAILABLE_MODELS:
            raise ValueError(f"Invalid default model: {self.model.DEFAULT_MODEL}")
        
        # Validate confidence threshold
        if not 0 < self.training.CONFIDENCE_THRESHOLD < 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        
        return True


# Create global configuration instance
config = Config()

# Validate configuration on import
try:
    config.validate_config()
except Exception as e:
    print(f"Configuration validation failed: {e}")
    raise

