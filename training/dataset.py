"""Training dataset: preprocessed per-structure features (.npz) + RF grid probabilities (.npz) -> graphs and labels.

This is the research code used to train the released EGNN, kept close to the original.
Note: true metal positions are appended to the RF-retained grids (positives), as in the original training.
"""
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import dgl  # type: ignore
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

from promet.constants import ATOMIC_NUMBERS, metals, sec_struct_dict, standard_residues, sybyl_type_dict
from promet.featurizer import Features


@dataclass
class Info:
    pdb_id: np.ndarray
    grids_positions: torch.Tensor  # [g, 3]
    metal_positions: torch.Tensor  # [m, 3]
    metal_types: torch.Tensor  # [m]


class PreprocessedDataSet(torch.utils.data.Dataset):
    def __init__(self, data_file: str, features_dir: str, rf_result_dir: str, topk: int, edge_dist_cutoff: float,
                 pocket_dist: float, rf_threshold: float, metal_dir=None, eps=1e-6):
        super().__init__()
        self.data_file=Path(data_file)
        self.features_dir=Path(features_dir)
        self.rf_result_dir=Path(rf_result_dir)
        self.topk = topk
        self.edge_dist_cutoff=edge_dist_cutoff
        self.pocket_dist=pocket_dist
        self.rf_threshold=rf_threshold
        self.pdbid_lists=[pdb.strip().split(".pdb")[0] for pdb in open(data_file) if (self.features_dir / f"{pdb.strip().split('.pdb')[0]}.npz").exists()]
        self.eps = eps
        self.alpha = 4/math.log(2) #5.77078
        self.metal_dir = metal_dir
        self.relpos_embedding = nn.Embedding(65, 8)  # relative position embedding, dim=8
    def __len__(self):
        return len(self.pdbid_lists)
    
    def __getitem__(self, index:int):
        G = []
        L = []
        
        pdb_id = self.pdbid_lists[index]
        feature_path = self.features_dir / f"{pdb_id}.npz"
        rf_result_path = self.rf_result_dir / f"{pdb_id}.npz"
        
        data = np.load(feature_path,allow_pickle=True)

        features = Features(**data)
        if self.metal_dir is not None:
            metal_path = Path(f"{self.metal_dir}/{pdb_id}.npz")
            if metal_path.exists():
                metal = np.load(metal_path, allow_pickle=True)
                features.metal_positions = metal["metal_positions"]
                features.metal_types     = metal["metal_types"]
        
        if len(features.atom_positions) != len(features.atom_elements):
            raise Exception(feature_path)
        if features.bond_masks.shape != (len(features.atom_elements), len(features.atom_elements)):
            bonds_mask = self.neigh_to_bondmask(features)
            features.bond_masks = bonds_mask
        
        grid_positions = features.grid_positions
        grid_data = np.load(rf_result_path)
        grid_probs = grid_data["prob"] 
        grid_mask = grid_probs >= self.rf_threshold
        grids_after_rf = grid_positions[grid_mask]
        if features.metal_positions is not None:
            grids_after_rf = np.concatenate((grids_after_rf,features.metal_positions),axis=0)
        features.grid_positions = grids_after_rf
        
        
        g = self.make_graph(features)
        l_prob, l_type, l_vector = self.make_label(features)
        labels = torch.cat([l_prob.unsqueeze(1), l_type.unsqueeze(1), l_vector], dim=1)  # shape [N,5]
        G.append(g)
        L.append(labels)
        if not G:
            raise AttributeError(f"{pdb_id} have none type graph")
        info = Info(
            pdb_id=np.array(pdb_id),
            grids_positions=torch.tensor(grids_after_rf, dtype=torch.float32),
            metal_positions=(
            torch.tensor(features.metal_positions, dtype=torch.float32)
            if features.metal_positions is not None and len(features.metal_positions) > 0
            else None
            ),
            metal_types=(
            torch.tensor([metals.index(metal) for metal in features.metal_types])
            if features.metal_types is not None and len(features.metal_types) > 0
            else None
            ),
        )

        return G, L, info

    def neigh_to_bondmask(self, features:Features):
        n_atom = len(features.atom_names)
        cov_bonds_mask = np.zeros((n_atom,n_atom))
        neigh = features.bond_masks
        cov_bonds_mask[neigh[0],neigh[1]] = 1
        cov_bonds_mask[neigh[1],neigh[0]] = 1
        return cov_bonds_mask
    
    def find_pocket(self, features: Features, grids: np.ndarray):
        c_grids = grids
        atom_pos = features.atom_positions
        size = len(features.metal_positions)
        perturb_size = np.random.uniform(0,3,size=features.metal_positions.shape[0])
        perturb_direct = np.random.randn(features.metal_positions.shape[0],3)
        normalized_direct = perturb_direct / np.linalg.norm(perturb_direct, axis=-1)[:,None]
        perturbation = normalized_direct * perturb_size[:,None]
        perturbed_metal_positions = features.metal_positions + perturbation
        random_size = np.random.randint(0,2*size)
        random_int = np.random.choice(len(grids), size=random_size, replace=False)
        combined_positions = np.vstack([c_grids[random_int], perturbed_metal_positions])

        gtree = cKDTree(c_grids)
        ptree = cKDTree(atom_pos)
        
        if len(features.metal_positions) > 10:
            mtree = cKDTree(perturbed_metal_positions)
            target = mtree
        else:
            rtree = cKDTree(combined_positions)
            target = rtree

        jj = target.query_ball_tree(gtree, self.pocket_dist)
        jdx = np.unique(np.concatenate([j for j in jj if j], axis=0)).astype(int)

        c_features = Features(
            atom_positions=atom_pos,
            atom_names=features.atom_names,
            atom_elements=features.atom_elements,
            atom_residues=features.atom_residues,
            residue_idxs=features.residue_idxs,
            chain_ids=features.chain_ids,
            is_ligand=features.is_ligand,
            metal_positions=features.metal_positions, 
            metal_types=features.metal_types,
            grid_positions=c_grids[jdx],
            sasas=features.sasas,
            qs=features.qs,
            sec_structs=features.sec_structs,
            gen_types=features.gen_types,
            bond_masks=features.bond_masks, 
        )
        return c_features, True
    
    def make_label(self, features: Features) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grid = np.array(features.grid_positions, dtype=np.float32)
        grids = torch.from_numpy(grid)  # CPU float32
        G = grids.shape[0]
        device = grids.device
        background_class = len(metals)

        no_metal = (
            features.metal_positions is None
            or (isinstance(features.metal_positions, np.ndarray) and features.metal_positions.size == 0)
        )

        if no_metal:
            label_prob   = torch.zeros(G, dtype=torch.float32, device=device)
            label_type   = torch.full((G,), background_class, dtype=torch.long, device=device)
            label_vector = torch.zeros(G, 3, dtype=torch.float32, device=device)
            return label_prob, label_type, label_vector

        metal_pos_np = np.asarray(features.metal_positions, dtype=np.float32)
        M = metal_pos_np.shape[0]
        if M == 0:
            label_prob   = torch.zeros(G, dtype=torch.float32, device=device)
            label_type   = torch.full((G,), background_class, dtype=torch.long, device=device)
            label_vector = torch.zeros(G, 3, dtype=torch.float32, device=device)
            return label_prob, label_type, label_vector

        metal_pos = torch.from_numpy(metal_pos_np).to(device=device)  # float32
        metal_types_idx = torch.tensor([metals.index(m) for m in features.metal_types],
                                    dtype=torch.long, device=device)  # [M]

        diff = grids.unsqueeze(1) - metal_pos.unsqueeze(0)        # [G, M, 3]
        dist = torch.sqrt(torch.sum(diff * diff, dim=-1)).clamp_min(0) + self.eps  # [G, M]

        exp_dist = torch.exp(-(dist * dist) / self.alpha)         # [G, M]
        label_p, _ = exp_dist.max(dim=-1)                         # [G]
        label_prob = torch.where(label_p <= 0.1,
                                torch.zeros_like(label_p),
                                label_p)

        min_dist, min_idx = dist.min(dim=-1)                      # [G], [G]

        label_type = torch.full((G,), background_class, dtype=torch.long, device=device)
        label_vector = torch.zeros(G, 3, dtype=torch.float32, device=device)

        thr = 2.0
        valid = min_dist <= thr
        if valid.any():
            v_idx = min_idx[valid]                                 # [G_valid]
            label_type[valid] = metal_types_idx[v_idx]
            label_vector[valid] = diff[valid, v_idx, :]


        return label_prob, label_type, label_vector


    
    def make_graph(self, features: Features) -> dgl.DGLGraph:
        xyz = torch.tensor(np.concatenate([features.atom_positions, features.grid_positions]))
        grid_mask = torch.ones(len(xyz))
        grid_mask[: len(features.sasas)] = 0
        n_feats, n_polar_vec = self.get_node_features(features)
        num_nodes = xyz.shape[0]
        edge_index_src, edge_index_dst, e_feats, rel_vec = self.make_edge(features)
        G = dgl.graph((edge_index_src.to(torch.int32), edge_index_dst.to(torch.int32)),num_nodes=num_nodes)
        G.ndata["xyz"] = xyz.to(torch.float32)
        G.ndata["L0"] = n_feats.to(torch.float32)
        G.ndata["L1"] = n_polar_vec.to(torch.float32)
        G.ndata["grid_mask"] = grid_mask.to(torch.float32)
        G.edata["L0"] = e_feats.to(torch.float32)
        G.edata["L1"] = rel_vec.to(torch.float32)
        return G
    
    def make_polarity_vector(self, features: Features) -> np.ndarray:
        xyz = torch.from_numpy(features.atom_positions)
        neigh_masks = torch.from_numpy(features.bond_masks)

        self_idx, nei_idx = torch.nonzero(neigh_masks, as_tuple=True)

        xyz_self = xyz * neigh_masks.sum(dim=1, keepdim=True)
        xyz_nei = -xyz[nei_idx].to(xyz_self.dtype)
        xyz_self.scatter_add_(0, self_idx[:, None].expand(-1, 3), xyz_nei)

        polar_vec = F.normalize(xyz_self, dim=1)
        polarity_vectors = torch.cat(
            [polar_vec, torch.zeros(features.grid_positions.shape)], dim=0
        ).numpy()
        return polarity_vectors

    def get_node_features(self, features: Features) -> Tuple[torch.Tensor, torch.Tensor]:
        num_grids = len(features.grid_positions)

        sasas = torch.from_numpy(features.sasas)
        qs = torch.from_numpy(features.qs)
        sec_structs = torch.from_numpy(features.sec_structs)
        atom_gentype = torch.from_numpy(features.gen_types)
        
        aatype = torch.Tensor([
            standard_residues.index(res) if res in standard_residues else len(standard_residues)
            for res in features.atom_residues
        ])
        grids_aatype = torch.ones(num_grids) * (len(standard_residues) + 1)
        aatype = torch.cat((aatype, grids_aatype))

        atomtype = torch.Tensor([ATOMIC_NUMBERS.get(elem, 119) for elem in features.atom_elements])
        grids_atomtype = torch.zeros(num_grids)
        atomtype = torch.cat([atomtype, grids_atomtype], dim=0)

        grids_atomchemtype = torch.ones(num_grids) * (max(sybyl_type_dict.values()) + 1)
        atom_chem_type = torch.cat([atom_gentype, grids_atomchemtype], dim=0)

        grids_2nd = torch.ones(num_grids) * len(sec_struct_dict)
        sec_structs = torch.cat([sec_structs, grids_2nd])
        node_type = torch.cat([torch.from_numpy(features.is_ligand),torch.ones_like(grids_2nd)*2])
        aatype = F.one_hot(aatype.to(torch.int64), num_classes=len(standard_residues) + 2)
        atomtype = F.one_hot(atomtype.to(torch.int64), num_classes=len(ATOMIC_NUMBERS) + 2)
        sec_structs = F.one_hot(sec_structs.to(torch.int64), num_classes=len(sec_struct_dict) + 1)
        atom_chemtype = F.one_hot(atom_chem_type.to(torch.int64), num_classes=max(sybyl_type_dict.values()) + 2)
        node_type = F.one_hot(node_type.to(torch.int64), num_classes=3)
        grids_feat = torch.zeros(num_grids)
        sasas = torch.cat((sasas, grids_feat)).unsqueeze(-1)
        qs = torch.cat((qs, grids_feat)).unsqueeze(-1)
        sasas = sasas + self.eps


        n_feats = torch.cat([aatype, atomtype, atom_chemtype, sec_structs, sasas, qs, node_type], dim=1)
        n_feats = torch.nan_to_num(n_feats, nan=0.0, posinf=0.0, neginf=0.0)

        polarity_vectors = self.make_polarity_vector(features)
        polarity_vectors = torch.tensor(polarity_vectors)
        polarity_vectors = torch.nan_to_num(polarity_vectors, nan=0.0, posinf=0.0, neginf=0.0)

        return n_feats, polarity_vectors


    def onehot_edge_dist(self, dists: torch.Tensor) -> torch.Tensor:
        bin_edges = np.arange(0, self.edge_dist_cutoff + 0.5, 0.1)
        dist_binned = np.digitize(dists, bins=bin_edges) - 1
        one_hot_dist = F.one_hot(
            torch.from_numpy(dist_binned), num_classes=len(bin_edges)
        )
        return one_hot_dist

    def onehot_edge_type(
        self, edge_index_src: torch.Tensor, edge_index_dst: torch.Tensor, num_atom: int
    ) -> torch.Tensor:
        feat = np.zeros_like(edge_index_src)  # p to p :0
        feat[np.where((edge_index_src < num_atom) & (edge_index_dst >= num_atom))] = (
            1  # p to g :1
        )
        feat[np.where((edge_index_src >= num_atom) & (edge_index_dst < num_atom))] = (
            2  # g to p : 2
        )
        feat[np.where((edge_index_src >= num_atom) & (edge_index_dst >= num_atom))] = (
            3  # g to g :3
        )
        one_hot_feat = F.one_hot(torch.from_numpy(feat).to(torch.int64), num_classes=4)
        return one_hot_feat

    def cov_bond(
        self,
        edge_index_src: torch.Tensor,
        edge_index_dst: torch.Tensor,
        num_atom: int,
        features: Features,
    ) -> torch.Tensor:
        cov_bond = np.zeros(len(edge_index_src))
        prot_idx_mask = (edge_index_src < num_atom) & (edge_index_dst < num_atom)
        idx = (edge_index_src[prot_idx_mask], edge_index_dst[prot_idx_mask])
        cov_bond[prot_idx_mask] = features.bond_masks[tuple(idx)]
        cov_bond = torch.from_numpy(cov_bond)
        return cov_bond

    def make_edge(
        self, features: Features) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_atom = len(features.atom_positions)
        num_grids = len(features.grid_positions)
        num_nodes = num_atom + num_grids

        node_pos = np.concatenate([features.atom_positions, features.grid_positions], axis=0)
        k_nearest = min(self.topk + 1, num_nodes)

        tree = cKDTree(node_pos)
        dd, ii = tree.query(
            node_pos, k=k_nearest, distance_upper_bound=self.edge_dist_cutoff
        )
        node_pos = torch.from_numpy(node_pos).to(torch.float32)
        index_tensor = torch.arange(num_nodes, dtype=torch.int32)
        edge_index_src = torch.flatten(torch.from_numpy(ii)).to(torch.int32)
        edge_index_dst = torch.repeat_interleave(index_tensor, k_nearest)
        dists = torch.flatten(torch.from_numpy(dd))

        edge_mask = torch.logical_and(edge_index_src != edge_index_dst, edge_index_src != num_nodes)


        edge_index_src = edge_index_src[edge_mask].long()
        edge_index_dst = edge_index_dst[edge_mask].long()
        dists = dists[edge_mask]

        dist_bin = self.onehot_edge_dist(dists)
        onehot_type = self.onehot_edge_type(edge_index_src, edge_index_dst, num_atom)
        covalent_bond = self.cov_bond(edge_index_src, edge_index_dst, num_atom, features)
        covalent_bond = covalent_bond.unsqueeze(-1)
        e_vec = torch.tensor(
            node_pos[edge_index_dst.long()] - node_pos[edge_index_src.long()]
        )

        polarity_vectors = torch.tensor(
            self.make_polarity_vector(features), dtype=torch.float32
        )
        edge_type_prot_to_prot = (edge_index_src < num_atom) & (
            edge_index_dst < num_atom
        )
        edge_type_grid_to_grid = (edge_index_src >= num_atom) & (
            edge_index_dst >= num_atom
        )
        edge_type_grid_to_prot = (edge_index_src >= num_atom) & (
            edge_index_dst < num_atom
        )
        edge_type_prot_to_grid = (edge_index_src < num_atom) & (
            edge_index_dst >= num_atom
        )

        start = torch.zeros((len(edge_index_src), 3), dtype=torch.float32)
        end = torch.zeros((len(edge_index_src), 3), dtype=torch.float32)

        mask = edge_type_prot_to_prot | edge_type_grid_to_grid
        start[mask] = polarity_vectors[edge_index_dst[mask].long()]
        end[mask] = polarity_vectors[edge_index_src[mask].long()]

        mask = edge_type_grid_to_prot
        start[mask] = (
            node_pos[edge_index_src[mask].long()]
            - node_pos[edge_index_dst[mask].long()]
        )
        end[mask] = polarity_vectors[edge_index_dst[mask].long()]

        mask = edge_type_prot_to_grid
        start[mask] = polarity_vectors[edge_index_src[mask].long()]
        end[mask] = (
            node_pos[edge_index_dst[mask].long()]
            - node_pos[edge_index_src[mask].long()]
        )

        cos = (
            torch.einsum(
                "ij,ij->i",
                start,
                end,
            ).unsqueeze(-1)
            + self.eps
        )
        sin = (
            torch.norm(
                torch.cross(
                    start,
                    end,
                ),
                dim=1,
                keepdim=True,
            )
            + self.eps
        )
        
        residue_idx = torch.tensor(features.residue_idxs, dtype=torch.int64)
        chain_ids = torch.tensor([hash(c) % 997 for c in features.chain_ids], dtype=torch.int64)  # hash to int
        residue_idx_all = torch.cat([
            residue_idx,
            torch.full((len(features.grid_positions),), -999, dtype=torch.int64)
        ])
        chain_ids_all = torch.cat([
            chain_ids,
            torch.full((len(features.grid_positions),), -1, dtype=torch.int64)
        ])

        same_chain = chain_ids_all[edge_index_src] == chain_ids_all[edge_index_dst]
        rel_idx = residue_idx_all[edge_index_src.long()] - residue_idx_all[edge_index_dst.long()]

        rel_idx = torch.clamp(rel_idx, -32, 32) + 32
        rel_idx[~same_chain] = 64  # special index for inter-chain
        rel_emb = self.relpos_embedding(rel_idx).detach()  # [E, 8]

        e_feats = torch.cat([onehot_type, dist_bin, covalent_bond, cos, sin, rel_emb], dim=1)

        return edge_index_src, edge_index_dst, e_feats, e_vec
    
    @staticmethod
    def collate(samples: list) -> Tuple[dgl.DGLGraph, torch.Tensor, Info]:
        graphs, labels, g_pos, m_pos, m_types, pdb_ids = [], [], [], [], [], []

        for G, L, info in samples:
            if G is not None:
                graphs.extend(G)
                labels.extend(L)
                g_pos.append(info.grids_positions)
                if info.metal_positions is not None and info.metal_positions.numel() > 0:
                    m_pos.append(info.metal_positions)
                else:
                    m_pos.append(torch.zeros((1, 3), dtype=torch.float32))
                if info.metal_types is not None and len(info.metal_types) > 0:
                    m_types.append(info.metal_types)
                else:
                    m_types.append(torch.full((1,), -1, dtype=torch.long))
                pdb_ids.append(info.pdb_id)
        batched_graphs = dgl.batch(graphs)  # shape [B*N]
        batched_labels = torch.cat(labels, dim=0)  # shape [B*N,2]
        g_poss = torch.cat(g_pos, dim=0)
        m_poss = torch.cat(m_pos, dim=0) 
        m_typess = torch.cat(m_types, dim=0) 
        pdb_idss = np.array(pdb_ids)
        batched_infos = Info(
            pdb_id=pdb_idss,
            grids_positions=g_poss.detach(),
            metal_positions=m_poss.detach(),
            metal_types=m_typess.detach(),
        )
        return batched_graphs, batched_labels, batched_infos
