"""Hand-crafted grid descriptors for the Random Forest pruning stage (Table S6)."""
import numpy as np
import pandas as pd
import scipy.spatial

from promet.constants import (
    ANGLE_PAB,
    DIST_PROBE_ALPHA,
    DIST_PROBE_BETA,
    VAN_DER_WAALS_RADII,
    aliphatic_carbons,
    aromatic_carbons,
    standard_residues,
)

CORE_RESIDUES = ["ASP", "HIS", "GLU", "CYS"]
SHELL_CUTOFFS = [2.5, 2.8, 3.0, 3.2, 5.0]


def near_lig(atom_coords, atom_elems, residues, grids, threshold):
    """Count ligand N/O/S/Se atoms, halide ions and other heteroatoms within `threshold` of each grid."""
    kd_grid = scipy.spatial.cKDTree(grids)
    kd_xyzs = scipy.spatial.cKDTree(atom_coords)
    neighs = kd_grid.query_ball_tree(kd_xyzs, threshold)
    nos_mask = np.isin(atom_elems, ["N", "O", "S", "SE"])
    neg_ion_mask = np.isin(atom_elems, ["BR", "CL", "I"]) & np.isin(residues, ["BR", "CL", "I"])
    etc_mask = np.isin(atom_elems, ["P", "TE", "BR", "CL", "I"]) & ~np.isin(residues, ["P", "TE", "BR", "CL", "I"])
    n_lig_nos, n_lig_nion, n_lig_etc = [], [], []
    for n in neighs:
        n_lig_nos.append(nos_mask[n].sum())
        n_lig_nion.append(neg_ion_mask[n].sum())
        n_lig_etc.append(etc_mask[n].sum())
    return n_lig_nos, n_lig_nion, n_lig_etc


def near_res(residues, atom_names, elems, xyzs, grids, threshold=3.2):
    """Count protein N/O/S atoms, core-residue N/O/S atoms and backbone O within `threshold` of each grid."""
    kd_grid = scipy.spatial.cKDTree(grids)
    kd_xyzs = scipy.spatial.cKDTree(xyzs)
    neighs = kd_grid.query_ball_tree(kd_xyzs, threshold)
    nos_mask = np.isin(elems, ["N", "O", "S"])
    res_mask = np.isin(residues, CORE_RESIDUES)
    bb_mask = (elems == "O") | (atom_names == "O") | (atom_names == "OXT")

    n_coords_res, n_core_res, n_bb_coords = [], [], []
    for n in neighs:
        n_coords_res.append(nos_mask[n].sum())
        n_core_res.append((res_mask[n] & nos_mask[n]).sum())
        n_bb_coords.append(bb_mask[n].sum())
    return n_coords_res, n_core_res, n_bb_coords


def binned_res(residues, atom_names, elems, xyzs, grids, lower=3.0, upper=5.0):
    """Second-shell counts: atoms in the (lower, upper] shell around each grid."""
    kd_grid = scipy.spatial.cKDTree(grids)
    kd_xyzs = scipy.spatial.cKDTree(xyzs)
    inner = kd_grid.query_ball_tree(kd_xyzs, lower)
    outer = kd_grid.query_ball_tree(kd_xyzs, upper)

    nos_mask = np.isin(elems, ["N", "O", "S"])
    res_mask = np.isin(residues, CORE_RESIDUES)
    bb_mask = (elems == "O") | (atom_names == "O") | (atom_names == "OXT") | np.isin(residues, standard_residues)

    n_coords_res, n_core_res, n_bb_coords = [], [], []
    for inner_n, outer_n in zip(inner, outer):
        neighs = np.setdiff1d(outer_n, inner_n)
        n_coords_res.append(nos_mask[neighs].sum())
        n_core_res.append((res_mask[neighs] & nos_mask[neighs]).sum())
        n_bb_coords.append(bb_mask[neighs].sum())
    return n_coords_res, n_core_res, n_bb_coords


def residue_backbone_atoms(features):
    """Collect per-residue CA/CB coordinates and residue names, keyed by (chain, residue number)."""
    res_id_dict = {}
    for chain_id, res_id in zip(features.chain_ids, features.residue_idxs):
        key = f"{chain_id}:{res_id}"
        if key not in res_id_dict:
            res_id_dict[key] = len(res_id_dict)

    n_res = len(res_id_dict)
    alphas = np.zeros((n_res, 3))
    betas = np.zeros((n_res, 3))
    res_names = np.empty((n_res,), dtype=object)
    for chain_id, res_id, res, atom_name, xyz in zip(
        features.chain_ids, features.residue_idxs, features.atom_residues, features.atom_names, features.atom_positions
    ):
        idx = res_id_dict[f"{chain_id}:{res_id}"]
        if atom_name == "CA":
            alphas[idx] = xyz
        elif atom_name == "CB":
            betas[idx] = xyz
        res_names[idx] = res
    return {"alphas": alphas, "betas": betas, "res": res_names}


def filter_by_biometall(grid, atom_dict):
    """Number of core residues whose CA/CB geometry is compatible with a metal at each grid (BioMetAll)."""
    grid = grid.astype(np.float32)
    alphas = atom_dict["alphas"].astype(np.float32)
    betas = atom_dict["betas"].astype(np.float32)
    res = np.array(atom_dict["res"])

    alpha_beta_distances = np.linalg.norm(betas - alphas, axis=1).astype(np.float32)
    alpha_distances = np.linalg.norm(grid[:, np.newaxis, :] - alphas[np.newaxis, :, :], axis=2).astype(np.float32)
    beta_distances = np.linalg.norm(grid[:, np.newaxis, :] - betas[np.newaxis, :, :], axis=2).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        pab_angles = np.arccos(
            (np.square(alpha_distances) + np.square(alpha_beta_distances) - np.square(beta_distances))
            / (2 * alpha_distances * alpha_beta_distances)
        ).astype(np.float32)

    grid_indices = []
    for res_name in ["HIS", "GLU", "ASP", "CYS"]:
        grid_index, res_index = np.where(
            (DIST_PROBE_ALPHA[res_name][0] <= alpha_distances) & (alpha_distances <= DIST_PROBE_ALPHA[res_name][1])
            & (DIST_PROBE_BETA[res_name][0] <= beta_distances) & (beta_distances <= DIST_PROBE_BETA[res_name][1])
            & (ANGLE_PAB[res_name][0] <= pab_angles) & (pab_angles <= ANGLE_PAB[res_name][1])
        )
        grid_indices.append(grid_index[np.isin(res[res_index], res_name)])
    grid_indices = np.concatenate(grid_indices)
    return np.bincount(grid_indices, minlength=len(grid)).astype(np.float32)


def _nearest_dist(points, grids):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    grids = np.asarray(grids, dtype=np.float32)
    _, idx = scipy.spatial.cKDTree(points).query(grids, k=1)
    # recompute in float32 so values match the original dense (grid x atom) implementation
    return np.sqrt(np.sum(np.square(grids - points[idx]), axis=-1), dtype=np.float32)


def nearest_prot_carbon_dist(residues, atom_names, elems, xyzs, grids):
    """Distance from each grid to the nearest nonpolar (aliphatic/aromatic or ligand) carbon."""
    carbons = []
    for res, atom, xyz, elem in zip(residues, atom_names, xyzs, elems):
        if res in aliphatic_carbons and atom in aliphatic_carbons[res]:
            carbons.append(xyz)
        if res in aromatic_carbons and atom in aromatic_carbons[res]:
            carbons.append(xyz)
        if res not in standard_residues and elem == "C":
            carbons.append(xyz)
    return _nearest_dist(carbons, grids)


def nearest_bb_dist(atom_names, xyzs, grids):
    """Distance from each grid to the nearest backbone C or N atom."""
    bb_cn = [xyz for atom, xyz in zip(atom_names, xyzs) if atom in ["C", "N"]]
    return _nearest_dist(bb_cn, grids)


def _probe_sphere_points(grids, n_samples):
    probe_radius = 1.4
    radius = 1.5  # average metal vdW radius
    inc = np.pi * (3 - np.sqrt(5))
    off = 2.0 / n_samples
    pts0 = []
    for k in range(n_samples):
        phi = k * inc
        y = k * off - 1 + (off / 2)
        r = np.sqrt(1 - y * y)
        pts0.append([np.cos(phi) * r, y, np.sin(phi) * r])
    pts0 = np.array(pts0)[np.newaxis, :, :].repeat(len(grids), axis=0)
    return pts0 * (radius + probe_radius) + grids[:, np.newaxis, :]


def _count_exposed(probe_points, elems, xyzs):
    elems = np.array(elems)
    groups = [
        (xyzs[elems == "C"], VAN_DER_WAALS_RADII["C"]),
        (xyzs[elems == "N"], VAN_DER_WAALS_RADII["N"]),
        (xyzs[elems == "O"], VAN_DER_WAALS_RADII["O"]),
        (xyzs[elems == "S"], VAN_DER_WAALS_RADII["S"]),
        (xyzs[~np.isin(elems, ["C", "N", "O", "S"])], VAN_DER_WAALS_RADII["C"]),
    ]
    num_grids, num_samples, _ = probe_points.shape
    flat = probe_points.reshape(-1, 3)
    probe_tree = scipy.spatial.cKDTree(flat)

    clashes = []
    for coords, radius in groups:
        if coords.size > 0:
            clashes += scipy.spatial.cKDTree(coords).query_ball_tree(probe_tree, radius)
    exposed = np.ones(flat.shape[0], dtype=bool)
    if clashes:
        exposed[np.unique(np.concatenate(clashes)).astype(int)] = False
    return exposed.reshape(num_grids, num_samples).sum(axis=-1)


def RSA(grids, xyzs, elems, n_samples=50):
    """Fraction of probe points around each grid that do not clash with any atom."""
    probe_points = _probe_sphere_points(grids, n_samples)
    return _count_exposed(probe_points, elems, xyzs) / n_samples


def make_rf_features(features) -> pd.DataFrame:
    """Build the RF descriptor table (one row per grid point, columns in training order)."""
    protein_mask = features.is_ligand == 0
    ligand_mask = features.is_ligand == 1
    grids = features.grid_positions
    df = pd.DataFrame()

    for t in SHELL_CUTOFFS:
        p_coords, p_core, p_bb = near_res(
            features.atom_residues[protein_mask],
            features.atom_names[protein_mask],
            features.atom_elements[protein_mask],
            features.atom_positions[protein_mask],
            grids,
            t,
        )
        n_nos, n_nion, n_etc = near_lig(
            features.atom_positions[ligand_mask],
            features.atom_elements[ligand_mask],
            features.atom_residues[ligand_mask],
            grids,
            t,
        )
        df[f"p_coords_res_{t}"] = np.array(p_coords).astype(np.int8)
        df[f"p_core_res_{t}"] = np.array(p_core).astype(np.int8)
        df[f"p_bb_coords_{t}"] = np.array(p_bb).astype(np.int8)
        df[f"n_lig_NOS_{t}"] = np.array(n_nos).astype(np.int8)
        df[f"n_lig_nion_{t}"] = np.array(n_nion).astype(np.int8)
        df[f"n_lig_etc_{t}"] = np.array(n_etc).astype(np.int8)

    coords, core, bb = binned_res(
        features.atom_residues, features.atom_names, features.atom_elements, features.atom_positions, grids, 3, 5
    )
    df["n_coords_res_bin"] = np.array(coords).astype(np.int8)
    df["n_core_res_bin"] = np.array(core).astype(np.int8)
    df["n_bb_coords_bin"] = np.array(bb).astype(np.int8)

    df["min_c_dist"] = nearest_prot_carbon_dist(
        features.atom_residues, features.atom_names, features.atom_elements, features.atom_positions, grids
    ).astype(np.float16)
    df["near_bb_dist"] = nearest_bb_dist(
        features.atom_names[protein_mask], features.atom_positions[protein_mask], grids
    ).astype(np.float16)
    df["sasa"] = np.array(RSA(grids, features.atom_positions, features.atom_elements)).astype(np.float16)
    df["biometall"] = filter_by_biometall(grids, residue_backbone_atoms(features)).astype(np.int8)

    # The RF was trained on these tables after a .csv round-trip, i.e. float16 values parsed back
    # from their shortest decimal representation. Reproduce that so borderline grids are pruned identically.
    for col in ["min_c_dist", "near_bb_dist", "sasa"]:
        df[col] = df[col].astype(str).astype(np.float64)
    return df
