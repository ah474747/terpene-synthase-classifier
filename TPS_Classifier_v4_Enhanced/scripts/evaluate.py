
import argparse, json, numpy as np, os
import sys
from pathlib import Path
from typing import List

# Add parent directory to path to import from workspace root
sys.path.append(str(Path(__file__).parent.parent))

def load_preds(paths: List[str]):
    outs = []
    for p in paths:
        P = []
        with open(p) as f:
            for line in f:
                rec = json.loads(line)
                P.append(rec.get("probs", rec.get("logits")))
        outs.append(np.array(P, dtype=np.float32))
    return outs

def load_labels(path: str, n_classes: int):
    Y = []
    with open(path) as f:
        for line in f:
            arr = [int(x) for x in line.strip().split(',')[:n_classes]]
            Y.append(arr)
    return np.array(Y, dtype=np.int32)

def macro_f1(probs, y_true, thr=0.5):
    C = probs.shape[1]
    f1s = []
    for c in range(C):
        y = y_true[:, c].astype(bool)
        yhat = probs[:, c] >= thr if np.isscalar(thr) else probs[:, c] >= thr[c]
        tp = (yhat & y).sum()
        fp = (yhat & ~y).sum()
        fn = (~yhat & y).sum()
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-8, prec + rec)
        f1s.append(f1)
    return float(np.mean(f1s))

def bootstrap_ci(metric_fn, probs, y_true, iters=1000, seed=42):
    rng = np.random.default_rng(seed)
    N = probs.shape[0]
    vals = []
    for _ in range(iters):
        idx = rng.integers(0, N, N)
        vals.append(metric_fn(probs[idx], y_true[idx]))
    vals = np.array(vals)
    return float(np.mean(vals)), (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

def main(args):
    preds_list = load_preds(args.preds)
    n_classes = preds_list[0].shape[1]
    y_true = load_labels(args.labels, n_classes)
    report = {}
    for i, probs in enumerate(preds_list):
        m, (lo, hi) = bootstrap_ci(lambda P, Y: macro_f1(P, Y, args.thr), probs, y_true, iters=args.bootstrap)
        report[os.path.basename(args.preds[i])] = {"macro_f1": m, "ci95": [lo, hi]}
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--preds", nargs="+", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--report", required=True)
    main(p.parse_args())
