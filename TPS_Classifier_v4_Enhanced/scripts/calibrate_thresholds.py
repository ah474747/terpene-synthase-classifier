
import argparse, json, os, numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import from workspace root
sys.path.append(str(Path(__file__).parent.parent))

from tps.eval.calibration import fit_platt_per_class, apply_scalers, optimize_thresholds

def load_preds(path):
    arr = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            arr.append(obj.get("logits", obj.get("probs")))
    A = np.array(arr, dtype=np.float32)
    return A

def load_labels(path, n_classes):
    Y = []
    with open(path) as f:
        for line in f:
            vals = [int(x) for x in line.strip().split(',')[:n_classes]]
            Y.append(vals)
    return np.array(Y, dtype=np.int32)

def main(args):
    arr = load_preds(args.preds)
    class_list = [c.strip() for c in open(args.class_list).read().splitlines() if c.strip()]
    y_true = load_labels(args.labels, len(class_list))
    if args.use_logits:
        scalers = fit_platt_per_class(arr, y_true)
        probs = apply_scalers(arr, scalers)
        cal = {str(c): [scalers[c].A, scalers[c].B] for c in scalers}
    else:
        probs = arr
        cal = {str(c): [1.0, 0.0] for c in range(arr.shape[1])}
    thr = optimize_thresholds(probs, y_true, mode=args.mode, beta=args.beta, precision_min=args.prec_floor)
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(cal, open(os.path.join(args.out_dir, "calibrators.json"), "w"), indent=2)
    json.dump(thr.tolist(), open(os.path.join(args.out_dir, "thresholds.json"), "w"), indent=2)
    
    # Print threshold stats
    print("Thresholds: min={:.3f} mean={:.3f} max={:.3f}".format(float(np.min(thr)), float(np.mean(thr)), float(np.max(thr))))
    hi = int(np.sum(thr >= 0.9)); lo = int(np.sum(thr <= 0.1))
    print(f"Threshold extremes: >=0.9 -> {hi}, <=0.1 -> {lo}, total classes -> {len(thr)}")
    print("Saved calibrators & thresholds to", args.out_dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--preds", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--class_list", required=True)
    p.add_argument("--use_logits", action="store_true")
    p.add_argument("--mode", choices=["f1beta","precision_floor"], default="f1beta")
    p.add_argument("--beta", type=float, default=0.7)
    p.add_argument("--prec_floor", type=float, default=0.6)
    p.add_argument("--out_dir", default="models/calibration/")
    main(p.parse_args())
