"""Per-atom features: SASA, partial charge, secondary structure and chemical atom type."""
import io
import os
import re
import shutil
import tempfile
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import freesasa
import numpy as np
from Bio.PDB import DSSP, PDBParser
from openbabel import openbabel

from promet.constants import (
    atype2num,
    atypes,
    bonds,
    partial_q,
    sec_struct_dict,
    standard_residues,
    sybyl_type_dict,
)
from promet.structure import StructureWithGrid, read_pdb

openbabel.obErrorLog.SetOutputLevel(0)


@dataclass
class Features:
    atom_positions: np.ndarray  # [n_atoms, 3]
    atom_names: np.ndarray  # [n_atoms]
    atom_elements: np.ndarray  # [n_atoms]
    atom_residues: np.ndarray  # [n_atoms]
    residue_idxs: np.ndarray  # [n_atoms]
    chain_ids: np.ndarray  # [n_atoms]
    is_ligand: np.ndarray  # [n_atoms]
    grid_positions: np.ndarray  # [n_grids, 3]
    sasas: np.ndarray  # [n_atoms]
    qs: np.ndarray  # [n_atoms]
    sec_structs: np.ndarray  # [n_atoms]
    gen_types: np.ndarray  # [n_atoms]
    bond_masks: Optional[np.ndarray] = None  # [n_atoms, n_atoms], only needed for training
    metal_positions: Optional[np.ndarray] = None  # [n_metals, 3]
    metal_types: Optional[np.ndarray] = None  # [n_metals]


def make_pdb(protein: StructureWithGrid) -> tuple:
    """Re-write the structure as a clean PDB with residues renumbered per chain.

    Returns (complex, protein-only, ligand-only) PDB strings as StringIO objects.
    """
    pdb_io, protein_io, ligand_io = io.StringIO(), io.StringIO(), io.StringIO()
    unique_residues = {}  # (chain, resnum, icode, resname) -> new resnum
    res_counter = defaultdict(int)

    for idx, (chain_id, res_idx, res_insert, res_name, atom_name, atom_pos, atom_elem, is_lig) in enumerate(zip(
            protein.chain_ids, protein.residue_idxs, protein.residue_inserts, protein.atom_residues,
            protein.atom_names, protein.atom_positions, protein.atom_elements, protein.is_ligand)):

        res_key = (chain_id, str(res_idx), res_insert, res_name)
        if res_key not in unique_residues:
            res_counter[chain_id] += 1
            unique_residues[res_key] = res_counter[chain_id]
        new_res_idx = unique_residues[res_key]

        if len(atom_name) == 4:
            atom_name_fixed = f"{atom_name:<4}"
        elif len(atom_elem) == 1:
            if len(atom_name) > 1:
                atom_name_fixed = f"{atom_elem:>2}{atom_name[1:]:<2}"
            else:
                atom_name_fixed = f" {atom_elem:>1}  "
        elif len(atom_elem) == 2:
            atom_name_fixed = f"{atom_elem:<2}{atom_name[2:]:<2}"
        else:
            raise ValueError(f"Unrecognised element symbol: {atom_elem}")

        pdb_line = (
            f"{'HETATM' if is_lig == 1 else 'ATOM  '}"
            f"{idx + 1:>5} "
            f"{atom_name_fixed:<4}"
            f" "
            f"{res_name:>3} "
            f"{chain_id:>1}"
            f"{new_res_idx:>4d}"
            f" "
            f"   "
            f"{atom_pos[0]:>8.3f}"
            f"{atom_pos[1]:>8.3f}"
            f"{atom_pos[2]:>8.3f}"
            f"{1.00:>6.2f}"
            f"{0.00:>6.2f}"
            f"          "
            f"{atom_elem:>2}"
            f"  "
            f"\n"
        )
        pdb_io.write(pdb_line)
        (ligand_io if is_lig else protein_io).write(pdb_line)

    pdb_io.seek(0)
    protein_io.seek(0)
    ligand_io.seek(0)
    return pdb_io, protein_io, ligand_io


class CustomClassifier(freesasa.Classifier):
    """FreeSASA classifier that also handles ligand atoms (unknown atoms get a 1.5 A radius)."""
    purePython = True

    def classify(self, res_name: str, atom_name: str) -> str:
        if re.match(r"\s*N", atom_name): return "Nitrogen"
        if re.match(r"\s*C", atom_name): return "Carbon"
        if re.match(r"\s*O", atom_name): return "Oxygen"
        if re.match(r"\s*S", atom_name): return "Sulfur"
        return "Unknown"

    def radius(self, res_name: str, atom_name: str) -> float:
        if re.match(r"\s*N", atom_name): return 1.6
        if re.match(r"\s*C", atom_name): return 1.7
        if re.match(r"\s*O", atom_name): return 1.4
        if re.match(r"\s*S", atom_name): return 1.8
        return 1.5


def calculate_sasa(pdb_path: str) -> np.ndarray:
    """Per-atom SASA (Shrake-Rupley via FreeSASA, including HETATM), scaled by 1/50."""
    structure = PDBParser(QUIET=True).get_structure("protein", pdb_path)
    freesasa.setVerbosity(1)
    try:
        result, _ = freesasa.calcBioPDB(
            structure, classifier=CustomClassifier(), options={"hetatm": True, "skip-unknown": False}
        )
    except Exception as e:
        warnings.warn(f"FreeSASA failed ({e}); SASA set to 0")
        return np.zeros(len(list(structure.get_atoms())))
    return np.array([result.atomArea(i) for i in range(result.nAtoms())]) / 50


def _dssp_executable() -> str:
    for name in (os.environ.get("PROMET_DSSP"), "mkdssp", "dssp"):
        if name and shutil.which(name):
            return name
    raise RuntimeError("DSSP executable not found. Install it (e.g. `conda install -c salilab dssp`) "
                       "or point PROMET_DSSP to the binary.")


def secondary_struct(pdb_path: str, structure) -> np.ndarray:
    """Per-atom DSSP secondary structure code; ligand atoms get 'L'."""
    structure_pdb = PDBParser(QUIET=True).get_structure("protein", pdb_path)
    dssp = DSSP(structure_pdb[0], pdb_path, dssp=_dssp_executable())

    per_atom = []
    for chain in structure_pdb.get_chains():
        for residue in chain.get_residues():
            try:
                sec = dssp[(chain.get_id(), residue.get_id())][2]
            except KeyError:
                sec = "-"
            per_atom.extend([sec] * len(list(residue.get_atoms())))

    secs = np.array([sec_struct_dict[s] for s in per_atom])
    ligand_mask = structure.is_ligand == 1
    secs[ligand_mask] = sec_struct_dict["L"]
    assert len(ligand_mask) == len(secs)
    return secs


def q_per_atom(ligand_mol, structure: StructureWithGrid) -> np.ndarray:
    """Partial charges: tabulated values for protein atoms, Gasteiger (Open Babel) for ligands."""
    ligand_mask = structure.is_ligand == 1
    protein_mask = ~ligand_mask
    qs = np.zeros(len(structure.atom_names))

    prot_qs = []
    for res, atom in zip(structure.atom_residues[protein_mask], structure.atom_names[protein_mask]):
        if res in partial_q and atom in partial_q[res]:
            prot_qs.append(partial_q[res][atom])
        elif atom == "OXT":
            prot_qs.append(partial_q[atom])
        else:
            prot_qs.append(0.0)

    ligand_qs = [0.0] * int(np.sum(ligand_mask))
    if ligand_mol is not None:
        try:
            ligand_mol.AddHydrogens()
            charge_model = openbabel.OBChargeModel.FindType("Gasteiger")
            if charge_model is None or not charge_model.ComputeCharges(ligand_mol):
                raise RuntimeError("Gasteiger charge calculation failed")
            ligand_qs = [
                atom.GetPartialCharge() for atom in openbabel.OBMolAtomIter(ligand_mol)
                if atom.GetAtomicNum() != 1
            ]
            ligand_mol.DeleteHydrogens()
        except Exception as e:
            warnings.warn(f"Ligand partial charges set to 0 ({e})")
            ligand_qs = [0.0] * int(np.sum(ligand_mask))

    prot_qs = np.array(prot_qs, dtype=np.float32)
    ligand_qs = np.array(ligand_qs, dtype=np.float32)
    assert ligand_mask.sum() == len(ligand_qs), f"ligand atoms {ligand_mask.sum()} != charges {len(ligand_qs)}"
    assert protein_mask.sum() == len(prot_qs)
    qs[np.where(protein_mask)] = prot_qs
    qs[np.where(ligand_mask)] = ligand_qs
    return qs


def cov_bonds_mask(structure: StructureWithGrid, ligand_mol) -> np.ndarray:
    """Dense covalent-bond adjacency matrix (used by the training data pipeline only)."""
    mask = np.zeros((len(structure.atom_names), len(structure.atom_names)))
    for i, (res_idx, res_name, atom_name) in enumerate(zip(
            structure.residue_idxs, structure.atom_residues, structure.atom_names)):
        if res_name in standard_residues:
            for atom, neigh in bonds[res_name]:
                if atom == atom_name:
                    sel = (structure.residue_idxs == res_idx) & (structure.atom_residues == res_name) & (structure.atom_names == neigh)
                    mask[i][sel] = 1
    ligand_indices = np.where(structure.is_ligand == 1)[0]
    if ligand_mol:
        ligand_mol.DeleteHydrogens()
        for bond in openbabel.OBMolBondIter(ligand_mol):
            a1 = bond.GetBeginAtom().GetIndex()
            a2 = bond.GetEndAtom().GetIndex()
            mask[ligand_indices[a1], ligand_indices[a2]] = 1
            mask[ligand_indices[a2], ligand_indices[a1]] = 1
    return mask


def make_gentype(structure, ligand_mol) -> np.ndarray:
    """Chemical atom category: residue-specific types for protein atoms, SYBYL types for ligand atoms."""
    prot_gentype, lig_gentype = [], []
    gentype = np.ones(len(structure.atom_elements))

    for res, atm, is_lig in zip(structure.atom_residues, structure.atom_names, structure.is_ligand):
        if is_lig == 0:
            if res in standard_residues:
                prot_gentype.append(6 if atm == "OXT" else atype2num[atypes.get((res, atm), "X")])
            else:
                prot_gentype.append(60)

    if ligand_mol:
        ligand_mol.DeleteHydrogens()
        for atom in openbabel.OBMolAtomIter(ligand_mol):
            lig_gentype.append(sybyl_type_dict.get(atom.GetType(), max(sybyl_type_dict.values())))

    gentype[structure.is_ligand == 0] = np.array(prot_gentype)
    gentype[structure.is_ligand == 1] = np.array(lig_gentype)
    assert sum(structure.is_ligand == 1) == len(lig_gentype)
    assert sum(structure.is_ligand == 0) == len(prot_gentype)
    return gentype


def make_features(structure: StructureWithGrid, with_bonds: bool = False) -> Features:
    """Compute all per-atom features for a structure (with grid points already attached)."""
    pdb_io, _, ligand_io = make_pdb(structure)
    ligand_pdb_str = ligand_io.getvalue()

    ligand_mol = None
    if ligand_pdb_str.strip():
        conv = openbabel.OBConversion()
        conv.SetInFormat("pdb")
        ligand_mol = openbabel.OBMol()
        conv.ReadString(ligand_mol, ligand_pdb_str)

    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as tmp:
        tmp.write(pdb_io.getvalue())
        clean_pdb_path = tmp.name
    try:
        new_structure = read_pdb(clean_pdb_path)
        sasas = calculate_sasa(clean_pdb_path)
        qs = q_per_atom(ligand_mol, new_structure)
        sec_structs = secondary_struct(clean_pdb_path, new_structure)
        bond_masks = cov_bonds_mask(new_structure, ligand_mol) if with_bonds else None
        gen_types = make_gentype(new_structure, ligand_mol)
    finally:
        os.remove(clean_pdb_path)

    assert len(structure.is_ligand) == len(new_structure.is_ligand)
    assert sum(structure.is_ligand) == sum(new_structure.is_ligand)

    skip = {"metal_positions", "metal_types", "residue_inserts"}
    return Features(
        **{k: v for k, v in new_structure.__dict__.items() if k not in skip},
        metal_positions=structure.metal_positions,
        metal_types=structure.metal_types,
        grid_positions=structure.grid_positions,
        gen_types=gen_types,
        sasas=sasas,
        qs=qs,
        sec_structs=sec_structs,
        bond_masks=bond_masks,
    )
