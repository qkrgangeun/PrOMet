"""PDB parsing into flat per-atom arrays."""
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
from Bio.PDB.PDBParser import PDBParser

from promet.constants import metals


@dataclass
class Structure:
    atom_positions: np.ndarray  # [n_atoms, 3]
    atom_names: np.ndarray  # [n_atoms]
    atom_elements: np.ndarray  # [n_atoms]
    atom_residues: np.ndarray  # [n_atoms] residue (or ligand) name
    residue_idxs: np.ndarray  # [n_atoms]
    residue_inserts: np.ndarray  # [n_atoms]
    chain_ids: np.ndarray  # [n_atoms]
    is_ligand: np.ndarray  # [n_atoms] 0 = protein, 1 = HETATM ligand/cofactor
    metal_positions: Optional[np.ndarray] = None  # [n_metals, 3]
    metal_types: Optional[np.ndarray] = None  # [n_metals]


@dataclass
class StructureWithGrid(Structure):
    grid_positions: Optional[np.ndarray] = None  # [n_grids, 3]


def read_pdb(pdb_path) -> Structure:
    """Read the first model of a PDB file.

    Hydrogens and waters are dropped. Metal ions listed in `promet.constants.metals`
    are separated into `metal_positions`/`metal_types` and are not part of the atom arrays;
    all other HETATM records are kept as ligand atoms.
    """
    structure = PDBParser(QUIET=True).get_structure("none", str(pdb_path))
    model = next(structure.get_models())

    data = defaultdict(list)

    def add_atom(atom, res, chain, is_ligand):
        data["atom_positions"].append(atom.coord)
        data["atom_elements"].append(atom.element)
        data["atom_residues"].append(res.get_resname())
        data["atom_names"].append(atom.name)
        data["is_ligand"].append(is_ligand)
        data["residue_idxs"].append(res.get_id()[1])
        data["residue_inserts"].append(res.id[2])
        data["chain_ids"].append(chain.get_id())

    for chain in model:
        for res in chain:
            if res.id[0] == " ":
                for atom in res:
                    if atom.element not in ["H", "D"]:
                        add_atom(atom, res, chain, 0)
            elif "H_" in res.id[0] and res.get_resname() not in ["HOH", "DOD"]:
                for atom in res.get_atoms():
                    if atom.element in metals and atom.element in res.get_resname():
                        data["metal_positions"].append(atom.coord)
                        data["metal_types"].append(atom.element)
                    elif atom.element not in ["H", "D"]:
                        add_atom(atom, res, chain, 1)

    return Structure(**{k: np.array(v) for k, v in data.items()})
