"""End-to-end PrOMet inference: PDB -> grid -> RF pruning -> EGNN -> sites."""
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from promet.cluster import Site, cluster_sites
from promet.constants import metals
from promet.featurizer import make_features
from promet.graph import build_graph
from promet.grid import filter_by_clashmap, sasa_grids_thread
from promet.model import Model
from promet.rf_features import make_rf_features
from promet.structure import StructureWithGrid, read_pdb
from promet.weights import EGNN_PATH, default_rf_path

MAX_ATOMS = 50000


@dataclass
class Prediction:
    name: str
    grid_positions: np.ndarray  # [G, 3] grids kept after RF pruning
    rf_probs: np.ndarray  # [G] RF probability of the kept grids
    grid_probs: np.ndarray  # [G] EGNN metal occupancy probability
    grid_type_probs: np.ndarray  # [G, n_metals] EGNN metal-type distribution
    sites: List[Site]
    input_metals: list = field(default_factory=list)  # (element, xyz) of metal ions removed from the input
    n_grids_total: int = 0
    timings: dict = field(default_factory=dict)


class PrOMet:
    """Load the released models once and predict metal-binding sites for many structures.

    >>> model = PrOMet(device="cuda")
    >>> pred = model.predict("examples/1afr.pdb")
    >>> for s in pred.sites: print(s.metal, s.probability, s.position)
    """

    def __init__(self, rf_path=None, egnn_path=None, device=None, rf_threshold=0.5,
                 topk=16, edge_cutoff=15.0, n_jobs=-1):
        import joblib

        rf_path = Path(rf_path) if rf_path else default_rf_path()
        if not rf_path.exists():
            raise FileNotFoundError(
                f"Random Forest parameters not found at {rf_path}. "
                "Run `promet download-weights` (or pass --rf-model)."
            )
        self.rf = joblib.load(rf_path)
        self.rf.n_jobs = n_jobs
        self.rf.verbose = 0

        ckpt = torch.load(egnn_path or EGNN_PATH, map_location="cpu")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.egnn = Model(**ckpt["model_config"])
        self.egnn.load_state_dict(ckpt["state_dict"])
        self.egnn.to(self.device).eval()

        self.rf_threshold = rf_threshold
        self.topk = topk
        self.edge_cutoff = edge_cutoff

    def predict(self, pdb_path, threshold=0.5, eps=2.0, min_samples=2, name: Optional[str] = None) -> Prediction:
        """Predict sites for one structure.

        Metal ions present in the input are removed before prediction and reported in `input_metals`.
        `threshold` is the grid-probability cutoff used before DBSCAN clustering.
        """
        pdb_path = Path(pdb_path)
        timings = {}
        t0 = time.time()

        structure = read_pdb(pdb_path)
        if len(structure.atom_positions) > MAX_ATOMS:
            raise ValueError(f"{pdb_path.name}: {len(structure.atom_positions)} atoms (> {MAX_ATOMS})")
        input_metals = []
        if structure.metal_positions is not None:
            input_metals = list(zip(structure.metal_types.tolist(), structure.metal_positions.tolist()))

        grids = filter_by_clashmap(sasa_grids_thread(structure.atom_positions, structure.atom_elements))
        s = asdict(structure)
        s["metal_positions"], s["metal_types"] = None, None
        features = make_features(StructureWithGrid(grid_positions=grids, **s))
        timings["features"] = time.time() - t0

        t0 = time.time()
        rf_probs = self.rf.predict_proba(make_rf_features(features).values)[:, 1]
        keep = rf_probs >= self.rf_threshold
        features.grid_positions = grids[keep]
        timings["rf"] = time.time() - t0

        t0 = time.time()
        if keep.sum() > 0:
            graph = build_graph(features, topk=self.topk, cutoff=self.edge_cutoff).to(self.device)
            with torch.no_grad():
                logit, type_logit, _ = self.egnn(graph)
            grid_idx = graph.ndata["grid_mask"] > 0
            grid_probs = torch.sigmoid(logit[grid_idx].reshape(-1)).cpu().numpy()
            # last type class is "no metal"; the metal distribution is the softmax over metal classes only
            grid_type_probs = torch.softmax(type_logit[grid_idx][:, : len(metals)], dim=-1).cpu().numpy()
        else:
            grid_probs = np.zeros(0, dtype=np.float32)
            grid_type_probs = np.zeros((0, len(metals)), dtype=np.float32)
        timings["egnn"] = time.time() - t0

        sites = cluster_sites(features.grid_positions, grid_probs, grid_type_probs, metals,
                              threshold=threshold, eps=eps, min_samples=min_samples)
        return Prediction(
            name=name or pdb_path.stem,
            grid_positions=features.grid_positions.astype(np.float32),
            rf_probs=rf_probs[keep].astype(np.float32),
            grid_probs=grid_probs,
            grid_type_probs=grid_type_probs,
            sites=sites,
            input_metals=input_metals,
            n_grids_total=len(grids),
            timings=timings,
        )
