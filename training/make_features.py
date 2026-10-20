"""PDB -> per-structure feature file (<pdb_id>.npz) used by both RF and EGNN training.

    python training/make_features.py pdbs/*.pdb -o work/features
"""
import argparse
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np

from promet.featurizer import make_features
from promet.grid import filter_by_clashmap, sasa_grids_thread
from promet.predict import MAX_ATOMS
from promet.structure import StructureWithGrid, read_pdb


def bondmask_to_neighidx(bond_mask):
    rows, cols = np.where(np.triu(bond_mask) > 0)
    return np.stack([rows, cols], axis=0).astype(np.int32)


def optimize_dtype(key, arr):
    if key == "bond_masks":
        return bondmask_to_neighidx(arr)
    if key in ["atom_positions", "metal_positions", "grid_positions", "qs", "sasas"]:
        return arr.astype(np.float32)
    if key == "residue_idxs":
        return arr.astype(np.int32)
    if key in ["sec_structs", "gen_types"]:
        return arr.astype(np.int16)
    if key == "is_ligand":
        return arr.astype(np.bool_)
    if arr.dtype.kind == "U":
        return arr.astype(f"<U{max(len(str(s)) for s in arr)}")
    return arr


def process(pdb_path: Path, out_dir: Path):
    structure = read_pdb(pdb_path)
    if len(structure.atom_positions) > MAX_ATOMS:
        print(f"skip {pdb_path.name}: more than {MAX_ATOMS} atoms")
        return
    grids = filter_by_clashmap(sasa_grids_thread(structure.atom_positions, structure.atom_elements))
    features = make_features(StructureWithGrid(grid_positions=grids, **asdict(structure)), with_bonds=True)
    arrays = {k: optimize_dtype(k, v) for k, v in asdict(features).items() if isinstance(v, np.ndarray)}
    np.savez(out_dir / f"{pdb_path.stem}.npz", **arrays)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdbs", nargs="+")
    parser.add_argument("-o", "--output_dir", default="work/features")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for p in args.pdbs:
        try:
            process(Path(p), out)
            print(f"done {p}")
        except Exception:
            print(f"failed {p}")
            traceback.print_exc()
