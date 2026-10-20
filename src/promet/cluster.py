"""Grid-level predictions -> site-level calls."""
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN


@dataclass
class Site:
    position: np.ndarray  # [3] probability-weighted mean of member grid coordinates
    probability: float  # max grid occupancy probability in the cluster
    metal: str  # top-1 metal of the highest-probability grid
    type_probs: np.ndarray  # [n_metals] metal-type distribution of the highest-probability grid
    n_grids: int


def cluster_sites(grid_positions, grid_probs, grid_type_probs, metals, threshold=0.5, eps=2.0, min_samples=2):
    """DBSCAN over grids with probability >= `threshold`; one site per cluster, sorted by probability."""
    keep = grid_probs >= threshold
    coords, probs, type_probs = grid_positions[keep], grid_probs[keep], grid_type_probs[keep]
    if len(coords) == 0:
        return []

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(coords).labels_
    sites = []
    for lab in set(labels) - {-1}:
        idx = np.where(labels == lab)[0]
        best = idx[np.argmax(probs[idx])]
        sites.append(Site(
            position=np.average(coords[idx], axis=0, weights=probs[idx]),
            probability=float(probs[best]),
            metal=metals[int(np.argmax(type_probs[best]))],
            type_probs=type_probs[best],
            n_grids=len(idx),
        ))
    return sorted(sites, key=lambda s: -s.probability)
