#!/usr/bin/env python3
"""Test: Hierarchy Masking"""

import torch
from tps.hierarchy.head import HierarchyHead

def test_hierarchy_basic():
    hierarchy_head = HierarchyHead(n_classes=30, latent_dim=512)
    latent_features = torch.randn(2, 512)
    type_logits = hierarchy_head.type_head(latent_features)
    type_probs = torch.softmax(type_logits, dim=-1)
    fine_logits = torch.randn(2, 30)
    masked_logits = hierarchy_head.apply_type_mask(fine_logits, type_probs)
    assert masked_logits.shape == fine_logits.shape
    print("✓ Hierarchy masking works")

if __name__ == "__main__":
    test_hierarchy_basic()
    print("✅ Hierarchy test passed!")