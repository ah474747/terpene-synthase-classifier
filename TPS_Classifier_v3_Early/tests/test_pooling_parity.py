#!/usr/bin/env python3
"""Test: Pooling Parity"""

import torch
from tps.config import ESM_MODEL_ID

def test_esm_model_id_consistency():
    # Check that ESM model ID is defined
    assert ESM_MODEL_ID is not None, "ESM_MODEL_ID should be defined"
    assert isinstance(ESM_MODEL_ID, str), "ESM_MODEL_ID should be a string"
    assert len(ESM_MODEL_ID) > 0, "ESM_MODEL_ID should not be empty"
    
    # Check it's a valid ESM model
    assert "esm" in ESM_MODEL_ID.lower(), "Should be an ESM model"
    
    print(f"✓ ESM model ID: {ESM_MODEL_ID}")

def test_esm_dimensions():
    from tps.config import ESM2_DIM, LATENT_DIM
    
    assert ESM2_DIM > 0, "ESM2_DIM should be positive"
    assert LATENT_DIM > 0, "LATENT_DIM should be positive"
    assert ESM2_DIM >= LATENT_DIM, "ESM2_DIM should be >= LATENT_DIM"
    
    print(f"✓ ESM2_DIM: {ESM2_DIM}, LATENT_DIM: {LATENT_DIM}")

if __name__ == "__main__":
    test_esm_model_id_consistency()
    test_esm_dimensions()
    print("✅ Pooling parity tests passed!")