
import os
ESM_MODEL_ID = os.getenv("TPS_ESM_MODEL_ID", "esm2_t6_8M_UR50D")
STRUCTURE_ENABLED = os.getenv("TPS_STRUCTURE_ENABLED", "1") == "1"
ALPHA_KNN = float(os.getenv("TPS_ALPHA_KNN", "0.7"))
CALIBRATION_MODE = os.getenv("TPS_CALIBRATION_MODE", "platt")
F1_BETA = float(os.getenv("TPS_F1_BETA", "0.7"))
PRECISION_MIN = float(os.getenv("TPS_PRECISION_MIN", "0.6"))
IDENTITY_THRESHOLD = float(os.getenv("TPS_IDENTITY_THRESHOLD", "0.4"))
TOP_K = int(os.getenv("TPS_TOP_K", "3"))
RANDOM_SEED = int(os.getenv("TPS_SEED", "42"))
CHECKPOINT_PATH = os.getenv("TPS_CHECKPOINT_PATH", "models/checkpoints/complete_multimodal_best.pth")
THRESHOLDS_PATH = os.getenv("TPS_THRESHOLDS_PATH", "results/training_results/final_functional_training_results.json")
LABEL_ORDER_PATH = os.getenv("TPS_LABEL_ORDER_PATH", "models/checkpoints/label_order.json")
CALIBRATION_DIR = os.getenv("TPS_CALIBRATION_DIR", "models/calibration/")
KNN_INDEX_PATH = os.getenv("TPS_KNN_INDEX_PATH", "models/knn/index.faiss")
KNN_META_PATH = os.getenv("TPS_KNN_META_PATH", "models/knn/index_meta.json")
