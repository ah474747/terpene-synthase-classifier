
import numpy as np
from TPS_Predictor_Stabilized import TPSPredictorStabilized

def test_determinism():
    seqs = ["ACDEFGHIKLMNPQRSTVWY", "MSTNQ"]
    labels = [f"class_{i}" for i in range(5)]
    pred = TPSPredictorStabilized(n_classes=5, label_order=labels)
    out1 = pred.predict(seqs, use_knn=False, use_hierarchy=False)
    out2 = pred.predict(seqs, use_knn=False, use_hierarchy=False)
    assert np.allclose(out1["probs"], out2["probs"], atol=1e-8)
