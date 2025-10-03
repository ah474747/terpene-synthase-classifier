
import os
from . import config

class MissingArtifact(Exception):
    pass

def _assert_exists(path, what):
    if not os.path.exists(path):
        raise MissingArtifact(f"{what} not found at: {path}")
    return os.path.abspath(path)

def resolve_checkpoint():
    return _assert_exists(config.CHECKPOINT_PATH, "Checkpoint")

def resolve_thresholds():
    return _assert_exists(config.THRESHOLDS_PATH, "Training results/thresholds JSON")

def resolve_label_order():
    return _assert_exists(config.LABEL_ORDER_PATH, "Label order JSON")

def resolve_calibration_dir():
    if not os.path.exists(config.CALIBRATION_DIR):
        raise MissingArtifact(f"Calibration dir not found at: {config.CALIBRATION_DIR}")
    return os.path.abspath(config.CALIBRATION_DIR)

def resolve_knn_index():
    return _assert_exists(config.KNN_INDEX_PATH, "kNN index file (FAISS or npy)")

def resolve_knn_meta():
    return _assert_exists(config.KNN_META_PATH, "kNN index metadata JSON")
