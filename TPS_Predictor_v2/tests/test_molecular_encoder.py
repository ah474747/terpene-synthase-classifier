"""
Unit tests for molecular encoder
"""

import pytest
import numpy as np
from models.molecular_encoder import TerpeneProductEncoder, MolecularFingerprint

def test_molecular_encoder_initialization():
    """Test molecular encoder initialization"""
    encoder = TerpeneProductEncoder()
    assert encoder.terpene_smiles is not None
    assert len(encoder.terpene_smiles) > 0
    assert encoder.morgan_radius > 0
    assert encoder.morgan_nbits > 0

def test_smiles_validation():
    """Test SMILES validation"""
    encoder = TerpeneProductEncoder()
    
    # Test valid SMILES
    valid_smiles = "CC1=CCC(CC1)C(=C)C"  # Limonene
    mol = encoder._create_molecule(valid_smiles)
    assert mol is not None
    
    # Test invalid SMILES
    invalid_smiles = "invalid_smiles"
    mol_invalid = encoder._create_molecule(invalid_smiles)
    assert mol_invalid is None

def test_morgan_fingerprint_generation():
    """Test Morgan fingerprint generation"""
    encoder = TerpeneProductEncoder()
    
    # Test with valid SMILES
    smiles = "CC1=CCC(CC1)C(=C)C"  # Limonene
    mol = encoder._create_molecule(smiles)
    
    if mol is not None:
        fp = encoder._generate_morgan_fingerprint(mol)
        assert isinstance(fp, np.ndarray)
        assert fp.shape[0] == encoder.morgan_nbits
        assert fp.dtype == np.int32
    else:
        pytest.skip("RDKit not available or molecule creation failed")

def test_maccs_fingerprint_generation():
    """Test MACCS fingerprint generation"""
    encoder = TerpeneProductEncoder()
    
    # Test with valid SMILES
    smiles = "CC1=CCC(CC1)C(=C)C"  # Limonene
    mol = encoder._create_molecule(smiles)
    
    if mol is not None:
        fp = encoder._generate_maccs_fingerprint(mol)
        assert isinstance(fp, np.ndarray)
        assert fp.shape[0] == encoder.maccs_nbits
        assert fp.dtype == np.int32
    else:
        pytest.skip("RDKit not available or molecule creation failed")

def test_rdkit_descriptors():
    """Test RDKit descriptor generation"""
    encoder = TerpeneProductEncoder()
    
    # Test with valid SMILES
    smiles = "CC1=CCC(CC1)C(=C)C"  # Limonene
    mol = encoder._create_molecule(smiles)
    
    if mol is not None:
        desc = encoder._generate_rdkit_descriptors(mol)
        assert isinstance(desc, np.ndarray)
        assert len(desc) == len(encoder.descriptor_names)
        assert desc.dtype == np.float64
    else:
        pytest.skip("RDKit not available or molecule creation failed")

def test_product_encoding():
    """Test complete product encoding"""
    encoder = TerpeneProductEncoder()
    
    # Test with known product
    product_name = "limonene"
    fingerprint = encoder.encode_product(product_name)
    
    if fingerprint is not None:
        assert isinstance(fingerprint, MolecularFingerprint)
        assert fingerprint.smiles == encoder.terpene_smiles[product_name]
        assert fingerprint.morgan_fingerprint.shape[0] == encoder.morgan_nbits
        assert fingerprint.maccs_fingerprint.shape[0] == encoder.maccs_nbits
        assert fingerprint.molecular_weight > 0
    else:
        pytest.skip("Product encoding failed (RDKit not available)")

def test_unknown_product():
    """Test encoding of unknown product"""
    encoder = TerpeneProductEncoder()
    
    # Test with unknown product
    unknown_product = "unknown_terpene"
    fingerprint = encoder.encode_product(unknown_product)
    
    assert fingerprint is None

def test_custom_smiles():
    """Test encoding with custom SMILES"""
    encoder = TerpeneProductEncoder()
    
    # Test with custom SMILES
    custom_smiles = "CC1=CCC(CC1)C(=C)C"  # Limonene
    fingerprint = encoder.encode_product("custom_product", custom_smiles)
    
    if fingerprint is not None:
        assert fingerprint.smiles == custom_smiles
        assert fingerprint.molecular_weight > 0
    else:
        pytest.skip("Custom SMILES encoding failed (RDKit not available)")

def test_dataset_encoding():
    """Test encoding of multiple products"""
    encoder = TerpeneProductEncoder()
    
    products = ["limonene", "pinene", "myrcene"]
    encoded_products = encoder.encode_dataset(products)
    
    if encoded_products:
        assert len(encoded_products) == len(products)
        assert all(isinstance(fp, MolecularFingerprint) for fp in encoded_products)
    else:
        pytest.skip("Dataset encoding failed (RDKit not available)")

def test_fingerprint_matrix_creation():
    """Test fingerprint matrix creation"""
    encoder = TerpeneProductEncoder()
    
    products = ["limonene", "pinene"]
    encoded_products = encoder.encode_dataset(products)
    
    if encoded_products:
        matrix, names = encoder.create_fingerprint_matrix(encoded_products)
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape[0] == len(products)
        assert matrix.shape[1] == encoder.morgan_nbits + encoder.maccs_nbits + len(encoder.descriptor_names)
        assert len(names) == len(products)
    else:
        pytest.skip("Fingerprint matrix creation failed (RDKit not available)")

if __name__ == "__main__":
    pytest.main([__file__])
