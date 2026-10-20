"""Random Forest pruning stage: descriptor tables, training and per-structure grid probabilities.

    python training/rf.py features  --features_dir work/features --out_dir work/rf_features
    python training/rf.py train     --ids data/splits/train.txt --rf_features_dir work/rf_features --model work/promet_rf.joblib
    python training/rf.py predict   --ids data/splits/train.txt data/splits/validation.txt \
                                    --rf_features_dir work/rf_features --model work/promet_rf.joblib --out_dir work/rf_probs

The released RF was trained on monomer structures of the training split (multimers excluded for memory).
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from tqdm import tqdm

from promet.featurizer import Features
from promet.rf_features import make_rf_features

LABEL = "label_2.0"


def label_grids(metal_coords, grid_coords, t=2.0):
    """Grid is positive if a metal ion lies within `t` A."""
    if metal_coords is None or metal_coords.size == 0:
        return np.zeros(grid_coords.shape[0], dtype=bool)
    dists = np.linalg.norm(grid_coords[:, None, :] - metal_coords[None, :, :], axis=-1)
    return dists.min(axis=-1) <= t


def read_ids(files):
    return [l.strip().split(".pdb")[0] for f in files for l in open(f) if l.strip()]


def cmd_features(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for npz in tqdm(sorted(Path(args.features_dir).glob("*.npz"))):
        f = Features(**np.load(npz, allow_pickle=True))
        df = make_rf_features(f)
        df[LABEL] = label_grids(f.metal_positions, f.grid_positions)
        df.to_csv(out / f"{npz.stem}.csv.gz", index=False, compression="gzip")


def load_table(ids, rf_dir):
    frames = [pd.read_csv(Path(rf_dir) / f"{i}.csv.gz") for i in tqdm(ids) if (Path(rf_dir) / f"{i}.csv.gz").exists()]
    data = pd.concat(frames, ignore_index=True)
    return data.drop(columns=[LABEL]), data[LABEL]


def cmd_train(args):
    X, y = load_table(read_ids(args.ids), args.rf_features_dir)
    rf = BalancedRandomForestClassifier(random_state=42, n_jobs=-1)
    rf.fit(X, y)
    joblib.dump(rf, args.model)
    print(f"saved {args.model}")


def cmd_predict(args):
    rf = joblib.load(args.model)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i in tqdm(read_ids(args.ids)):
        path = Path(args.rf_features_dir) / f"{i}.csv.gz"
        if path.exists():
            X = pd.read_csv(path).drop(columns=[LABEL])
            np.savez(out / f"{i}.npz", prob=rf.predict_proba(X)[:, 1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("features"); p.add_argument("--features_dir", required=True); p.add_argument("--out_dir", required=True)
    p.set_defaults(func=cmd_features)
    p = sub.add_parser("train"); p.add_argument("--ids", nargs="+", required=True)
    p.add_argument("--rf_features_dir", required=True); p.add_argument("--model", required=True)
    p.set_defaults(func=cmd_train)
    p = sub.add_parser("predict"); p.add_argument("--ids", nargs="+", required=True)
    p.add_argument("--rf_features_dir", required=True); p.add_argument("--model", required=True)
    p.add_argument("--out_dir", required=True)
    p.set_defaults(func=cmd_predict)
    args = parser.parse_args()
    args.func(args)
