import argparse, json, os, numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import from workspace root
sys.path.append(str(Path(__file__).parent.parent))

from TPS_Predictor_Stabilized import TPSPredictorStabilized
from tps.retrieval.knn_head import KNNSoftClassifier

def read_fasta(path):
    seqs, ids = [], []
    with open(path) as f:
        sid, s = None, []
        for line in f:
            if line.startswith('>'):
                if sid is not None:
                    seqs.append(''.join(s)); s=[]
                sid = line.strip()[1:]; ids.append(sid)
            else:
                s.append(line.strip())
        if sid is not None:
            seqs.append(''.join(s))
    return ids, seqs

def main(args):
    label_order = [c.strip() for c in open(args.class_list).read().splitlines() if c.strip()]
    predictor = TPSPredictorStabilized(n_classes=len(label_order), label_order=label_order)

    if args.knn_index and args.knn_meta and os.path.exists(args.knn_index):
        knn = KNNSoftClassifier(k=args.k)
        knn.load(args.knn_index, args.knn_meta)
        predictor.knn = knn

    calibrators, thresholds = None, None
    if args.calibration_dir:
        cal_path = os.path.join(args.calibration_dir, "calibrators.json")
        thr_path = os.path.join(args.calibration_dir, "thresholds.json")
        if os.path.exists(cal_path): calibrators = json.load(open(cal_path))
        if os.path.exists(thr_path): thresholds = np.array(json.load(open(thr_path)), dtype=np.float32)

    ids, seqs = read_fasta(args.input)
    out = predictor.predict(seqs, use_knn=args.use_knn, alpha=args.alpha,
                            use_hierarchy=args.use_hierarchy, calibrators=calibrators, thresholds=thresholds)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for i, sid in enumerate(ids):
            rec = {"id": sid, "label_order": out["label_order"], "decisions": out["decisions"][i]}
            if args.emit_logits:
                rec["logits"] = out["logits"][i]
            else:
                rec["probs"] = out["probs"][i]
            f.write(json.dumps(rec) + "\n")
    print("Wrote predictions to", args.out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--class_list", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--use_knn", action="store_true")
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--use_hierarchy", action="store_true")
    p.add_argument("--calibration_dir", default=None)
    p.add_argument("--knn_index", default=None)
    p.add_argument("--knn_meta", default=None)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--emit_logits", action="store_true")
    main(p.parse_args())