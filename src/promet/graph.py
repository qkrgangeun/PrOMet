"""Protein-grid graph for the EGNN optimization stage."""
import dgl  # type: ignore
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from promet.constants import ATOMIC_NUMBERS, sec_struct_dict, standard_residues, sybyl_type_dict
from promet.featurizer import Features

EPS = 1e-6


def node_features(features: Features) -> torch.Tensor:
    """One-hot residue / element / chemical type / secondary structure + SASA + charge + node type.

    Grid nodes use dedicated "grid" categories and zeros for real-valued features.
    """
    num_grids = len(features.grid_positions)

    aatype = torch.Tensor([
        standard_residues.index(res) if res in standard_residues else len(standard_residues)
        for res in features.atom_residues
    ])
    aatype = torch.cat((aatype, torch.ones(num_grids) * (len(standard_residues) + 1)))

    atomtype = torch.Tensor([ATOMIC_NUMBERS.get(elem, 119) for elem in features.atom_elements])
    atomtype = torch.cat([atomtype, torch.zeros(num_grids)])

    chemtype = torch.from_numpy(features.gen_types)
    chemtype = torch.cat([chemtype, torch.ones(num_grids) * (max(sybyl_type_dict.values()) + 1)])

    sec_structs = torch.cat([torch.from_numpy(features.sec_structs), torch.ones(num_grids) * len(sec_struct_dict)])
    node_type = torch.cat([torch.from_numpy(features.is_ligand), torch.ones(num_grids) * 2])

    aatype = F.one_hot(aatype.to(torch.int64), num_classes=len(standard_residues) + 2)
    atomtype = F.one_hot(atomtype.to(torch.int64), num_classes=len(ATOMIC_NUMBERS) + 2)
    sec_structs = F.one_hot(sec_structs.to(torch.int64), num_classes=len(sec_struct_dict) + 1)
    chemtype = F.one_hot(chemtype.to(torch.int64), num_classes=max(sybyl_type_dict.values()) + 2)
    node_type = F.one_hot(node_type.to(torch.int64), num_classes=3)

    zeros = torch.zeros(num_grids)
    sasas = torch.cat((torch.from_numpy(features.sasas), zeros)).unsqueeze(-1) + EPS
    qs = torch.cat((torch.from_numpy(features.qs), zeros)).unsqueeze(-1)

    feats = torch.cat([aatype, atomtype, chemtype, sec_structs, sasas, qs, node_type], dim=1)
    return torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def knn_edges(positions: np.ndarray, topk: int, cutoff: float):
    """Directed edges from each node's `topk` nearest neighbours within `cutoff` (A) to the node."""
    num_nodes = len(positions)
    k = min(topk + 1, num_nodes)  # +1: the query point itself
    dd, ii = cKDTree(positions).query(positions, k=k, distance_upper_bound=cutoff)
    src = torch.from_numpy(ii.reshape(-1)).to(torch.int64)
    dst = torch.repeat_interleave(torch.arange(num_nodes, dtype=torch.int64), k)
    keep = (src != dst) & (src != num_nodes)  # drop self loops and missing neighbours
    return src[keep], dst[keep]


def build_graph(features: Features, topk: int = 16, cutoff: float = 15.0) -> dgl.DGLGraph:
    """Joint graph of protein/ligand atoms followed by grid points.

    ndata: "xyz" coordinates, "L0" node features, "grid_mask" (1 for grid nodes).
    """
    xyz = np.concatenate([features.atom_positions, features.grid_positions])
    src, dst = knn_edges(xyz, topk, cutoff)
    g = dgl.graph((src.to(torch.int32), dst.to(torch.int32)), num_nodes=len(xyz))

    grid_mask = torch.ones(len(xyz))
    grid_mask[: len(features.atom_positions)] = 0
    g.ndata["xyz"] = torch.tensor(xyz).to(torch.float32)
    g.ndata["L0"] = node_features(features).to(torch.float32)
    g.ndata["grid_mask"] = grid_mask
    return g
