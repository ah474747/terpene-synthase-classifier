"""
Unit tests for configuration
"""

import pytest
from config.config import TerpenePredictorConfig, validate_config, TERPENE_PRODUCTS, TERPENE_SMILES, TERPENE_EC_NUMBERS

def test_default_config():
    """Test default configuration creation"""
    config = TerpenePredictorConfig()
    
    # Test data parameters
    assert config.min_sequence_length == 50
    assert config.max_sequence_length == 2000
    assert config.min_confidence == 0.7
    assert config.min_samples_per_class == 10
    assert config.max_samples_per_class == 1000
    
    # Test model parameters
    assert config.saprot_model_name == "nferruz/SaProt_650M"
    assert config.protein_embedding_dim == 1280
    assert config.molecular_fingerprint_dim == 2223
    assert config.hidden_dim == 512
    assert config.num_attention_heads == 8
    assert config.dropout_rate == 0.3
    
    # Test training parameters
    assert config.learning_rate == 1e-4
    assert config.batch_size == 32
    assert config.num_epochs == 100
    assert config.early_stopping_patience == 10
    assert config.weight_decay == 1e-5
    
    # Test validation parameters
    assert config.cv_folds == 5
    assert config.test_size == 0.2
    assert config.val_size == 0.2
    assert config.holdout_fraction == 0.2
    
    # Test system parameters
    assert config.device == "auto"
    assert config.cache_dir == "data/cache"
    assert config.random_state == 42
    assert config.log_level == "INFO"

def test_config_validation_valid():
    """Test configuration validation with valid parameters"""
    config = TerpenePredictorConfig()
    
    # Should not raise any exceptions
    assert validate_config(config) == True

def test_config_validation_invalid_sequence_length():
    """Test configuration validation with invalid sequence length"""
    config = TerpenePredictorConfig()
    
    # Test min_sequence_length too small
    config.min_sequence_length = 5
    with pytest.raises(ValueError, match="min_sequence_length must be >= 10"):
        validate_config(config)
    
    # Test max_sequence_length < min_sequence_length
    config.min_sequence_length = 100
    config.max_sequence_length = 50
    with pytest.raises(ValueError, match="max_sequence_length must be >= min_sequence_length"):
        validate_config(config)

def test_config_validation_invalid_confidence():
    """Test configuration validation with invalid confidence"""
    config = TerpenePredictorConfig()
    
    # Test confidence out of range
    config.min_confidence = 1.5
    with pytest.raises(ValueError, match="min_confidence must be between 0 and 1"):
        validate_config(config)
    
    config.min_confidence = -0.1
    with pytest.raises(ValueError, match="min_confidence must be between 0 and 1"):
        validate_config(config)

def test_config_validation_invalid_samples():
    """Test configuration validation with invalid sample counts"""
    config = TerpenePredictorConfig()
    
    # Test min_samples_per_class too small
    config.min_samples_per_class = 0
    with pytest.raises(ValueError, match="min_samples_per_class must be >= 1"):
        validate_config(config)
    
    # Test max_samples_per_class < min_samples_per_class
    config.min_samples_per_class = 100
    config.max_samples_per_class = 50
    with pytest.raises(ValueError, match="max_samples_per_class must be >= min_samples_per_class"):
        validate_config(config)

def test_config_validation_invalid_model_params():
    """Test configuration validation with invalid model parameters"""
    config = TerpenePredictorConfig()
    
    # Test negative dimensions
    config.protein_embedding_dim = -1
    with pytest.raises(ValueError, match="protein_embedding_dim must be > 0"):
        validate_config(config)
    
    # Test hidden_dim not divisible by num_attention_heads
    config.protein_embedding_dim = 1280
    config.hidden_dim = 500
    config.num_attention_heads = 8
    with pytest.raises(ValueError, match="hidden_dim must be divisible by num_attention_heads"):
        validate_config(config)

def test_config_validation_invalid_training_params():
    """Test configuration validation with invalid training parameters"""
    config = TerpenePredictorConfig()
    
    # Test negative learning rate
    config.learning_rate = -0.001
    with pytest.raises(ValueError, match="learning_rate must be > 0"):
        validate_config(config)
    
    # Test negative batch size
    config.learning_rate = 1e-4
    config.batch_size = -1
    with pytest.raises(ValueError, match="batch_size must be > 0"):
        validate_config(config)
    
    # Test negative epochs
    config.batch_size = 32
    config.num_epochs = -1
    with pytest.raises(ValueError, match="num_epochs must be > 0"):
        validate_config(config)

def test_config_validation_invalid_validation_params():
    """Test configuration validation with invalid validation parameters"""
    config = TerpenePredictorConfig()
    
    # Test test_size out of range
    config.test_size = 1.5
    with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
        validate_config(config)
    
    # Test val_size out of range
    config.test_size = 0.2
    config.val_size = -0.1
    with pytest.raises(ValueError, match="val_size must be between 0 and 1"):
        validate_config(config)
    
    # Test test_size + val_size >= 1
    config.val_size = 0.2
    config.test_size = 0.9
    with pytest.raises(ValueError, match="test_size \\+ val_size must be < 1"):
        validate_config(config)
    
    # Test holdout_fraction out of range
    config.test_size = 0.2
    config.holdout_fraction = 1.5
    with pytest.raises(ValueError, match="holdout_fraction must be between 0 and 1"):
        validate_config(config)

def test_terpene_products():
    """Test terpene product definitions"""
    assert len(TERPENE_PRODUCTS) > 0
    assert "limonene" in TERPENE_PRODUCTS
    assert "pinene" in TERPENE_PRODUCTS
    assert "myrcene" in TERPENE_PRODUCTS
    assert "linalool" in TERPENE_PRODUCTS
    assert "germacrene_a" in TERPENE_PRODUCTS
    assert "germacrene_d" in TERPENE_PRODUCTS

def test_terpene_smiles():
    """Test terpene SMILES definitions"""
    assert len(TERPENE_SMILES) > 0
    assert "limonene" in TERPENE_SMILES
    assert "pinene" in TERPENE_SMILES
    assert "myrcene" in TERPENE_SMILES
    assert "linalool" in TERPENE_SMILES
    assert "germacrene_a" in TERPENE_SMILES
    assert "germacrene_d" in TERPENE_SMILES
    
    # Test that all products have SMILES
    for product in TERPENE_PRODUCTS:
        assert product in TERPENE_SMILES
        assert len(TERPENE_SMILES[product]) > 0

def test_terpene_ec_numbers():
    """Test terpene EC number definitions"""
    assert len(TERPENE_EC_NUMBERS) > 0
    assert "4.2.3.27" in TERPENE_EC_NUMBERS  # Limonene synthase
    assert "4.2.3.20" in TERPENE_EC_NUMBERS  # Pinene synthase
    assert "4.2.3.15" in TERPENE_EC_NUMBERS  # Myrcene synthase
    assert "4.2.3.14" in TERPENE_EC_NUMBERS  # Linalool synthase

if __name__ == "__main__":
    pytest.main([__file__])
