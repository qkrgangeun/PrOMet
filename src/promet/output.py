"""Writers for prediction results."""
import csv
from pathlib import Path

import numpy as np

from promet.constants import metals
from promet.predict import Prediction


def write_sites_csv(pred: Prediction, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["site", "metal", "probability", "x", "y", "z", "n_grids"] + [f"p_{m}" for m in metals])
        for i, s in enumerate(pred.sites, start=1):
            w.writerow([i, s.metal, f"{s.probability:.4f}", *(f"{c:.3f}" for c in s.position), s.n_grids]
                       + [f"{p:.4f}" for p in s.type_probs])


def write_sites_pdb(pred: Prediction, path):
    """Predicted sites as HETATM records (B-factor = site probability), to overlay on the input structure."""
    with open(path, "w") as f:
        f.write(f"REMARK   PrOMet predicted metal-binding sites for {pred.name}\n")
        f.write("REMARK   B-factor column = site probability\n")
        for i, s in enumerate(pred.sites, start=1):
            x, y, z = s.position
            el = s.metal
            f.write(
                f"HETATM{i:5d} {el:>2}   {el:>3} Z{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                f"{1.0:6.2f}{s.probability:6.2f}          {el:>2}\n"
            )
        f.write("END\n")


def write_grids_npz(pred: Prediction, path):
    np.savez_compressed(
        path,
        grid_positions=pred.grid_positions,
        rf_probs=pred.rf_probs,
        grid_probs=pred.grid_probs,
        grid_type_probs=pred.grid_type_probs,
        metals=np.array(metals),
    )


def write_all(pred: Prediction, out_dir, save_grids=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_sites_csv(pred, out_dir / f"{pred.name}_sites.csv")
    write_sites_pdb(pred, out_dir / f"{pred.name}_sites.pdb")
    if save_grids:
        write_grids_npz(pred, out_dir / f"{pred.name}_grids.npz")
