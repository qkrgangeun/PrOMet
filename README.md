# PrOMet

**PrOMet** (**Pr**une-and-**O**ptimize **Met**al predictor) predicts metal-binding sites and their metal types from a protein structure.

> G. Park, C. Seok, H. Park. *PrOMet: Accurate and Robust Metal-binding Site and Type Prediction.* (manuscript under review)

PrOMet works in two stages:

1. **Prune.** Candidate points are sampled on the protein surface. A Balanced Random Forest scores each point from hand-crafted descriptors, such as coordinating atoms in the first and second shell, solvent exposure and backbone geometry. Points scoring below 0.5 are removed, which is about 97% of the candidates.
2. **Optimize.** The remaining points and the protein/ligand atoms are joined into one k-NN graph (k = 16). An E(3)-equivariant GNN (EGNN) predicts a metal occupancy probability and a metal type for every point. High-probability points are grouped into sites with DBSCAN (eps = 2 Å, min_samples = 2).

PrOMet uses bound ligands and cofactors (HETATM records) as part of the binding environment. It predicts 10 metal types: Mg, Zn, Mn, Ca, Fe, Ni, Co, Cu, K and Na.

## Installation

PrOMet needs Linux and conda. A CUDA GPU is optional; the EGNN step also runs fast on CPU.

```bash
git clone https://github.com/qkrgangeun/PrOMet.git
cd PrOMet
conda env create -f environment.yml      # creates the `promet` environment
conda activate promet
pip install -e .
```

Download the Random Forest parameters (about 1.8 GB). They are hosted as a GitHub release asset, and the download is checked against a SHA-256 checksum:

```bash
promet download-weights                  # saves to ~/.cache/promet/promet_rf.joblib
```

Set `PROMET_HOME` to store the file somewhere else. The EGNN parameters are small (0.6 MB) and ship with the package (`src/promet/params/promet_egnn.pt`).

> The RF file is a pickled `imblearn` model. Loading it requires the exact `scikit-learn==1.4.2` and `imbalanced-learn==0.12.3` pinned in `environment.yml`.

## Usage

```bash
promet predict examples/6wi8.pdb -o results/
```

```
6wi8: 7 sites | grids 13748 -> 276 after RF | 1.0s feat, 6.7s RF, 0.1s EGNN
  note: 8 metal ion(s) in the input were removed before prediction
  site  1  ZN  p=0.989  (5.47, 8.86, -4.07)
  site  2  ZN  p=0.985  (4.16, 56.47, 15.81)
  ...
```

The input can be one or more PDB files, a directory of `.pdb` files, or a `.txt` file listing PDB paths. The RF and EGNN models are loaded once and reused for every input.

| option | default | description |
|---|---|---|
| `-o, --output` | `promet_output` | output directory |
| `-t, --threshold` | `0.5` | grid probability cutoff applied before clustering. Lower values give more sites (higher recall); higher values give fewer, more confident sites. |
| `--rf-threshold` | `0.5` | RF pruning threshold. The paper found 0.4–0.7 to be stable. |
| `--rf-model` | `$PROMET_HOME/promet_rf.joblib` | path to the RF parameters |
| `--device` | `cuda` if available | e.g. `cpu`, `cuda:1` |
| `--n-jobs` | `-1` | CPU threads used by the RF |
| `--save-grids` | off | also write grid-level predictions |

### Outputs

For each input `<name>.pdb`, PrOMet writes these files:

- `<name>_sites.csv` has one row per predicted site, sorted by probability:

  ```
  site,metal,probability,x,y,z,n_grids,p_MG,p_ZN,p_MN,p_CA,p_FE,p_NI,p_CO,p_CU,p_K,p_NA
  1,ZN,0.9890,5.474,8.860,-4.073,5,0.0002,0.9960,0.0006,0.0000,0.0012,...
  ```

  - `probability` is the highest grid probability in the cluster.
  - `x, y, z` is the probability-weighted mean position of the cluster.
  - `metal` and `p_*` are the metal-type distribution of the cluster's highest-probability grid point.
- `<name>_sites.pdb` contains the predicted sites as HETATM records in chain `Z`, with the site probability in the B-factor column. Load it together with the input structure (e.g. in PyMOL) to view the sites.
- `<name>_grids.npz` is written only with `--save-grids`. It holds the RF-retained grid points (`grid_positions`), their RF and EGNN probabilities (`rf_probs`, `grid_probs`) and their metal-type distributions (`grid_type_probs`). Use it to re-cluster at other thresholds.

### Python API

```python
from promet import PrOMet

model = PrOMet(device="cuda")            # loads RF + EGNN once
pred = model.predict("examples/6wi8.pdb", threshold=0.5)
for site in pred.sites:
    print(site.metal, round(site.probability, 3), site.position)
```

### Notes on the input

- Only the first model is used. Hydrogens and waters are ignored.
- **Metal ions already in the input are removed before prediction.** They are reported as `input_metals`, so existing sites are predicted rather than copied. All other HETATM residues (ligands, cofactors, nucleic acids) are kept as context.
- Structures with more than 50,000 heavy atoms are skipped.
- Runtime depends mostly on feature computation and the RF. A 3,000-atom protein takes about 10 s on 8 CPU cores. Large complexes take longer; the RF scales with the number of surface grid points. Loading the RF takes a few seconds and about 3 GB of RAM.

## Training

`training/` contains the code used to train the released models. `data/splits/` contains the train, validation and test splits. The splits are chronological (cut-offs 2019-01-01 and 2021-03-31), and test proteins share at most 30% sequence identity with training proteins. Per-metal test subsets are in `test_<METAL>.txt`.

```bash
# 1. per-structure features (grids + atom features)                 -> work/features/<id>.npz
python training/make_features.py pdbs/*.pdb -o work/features

# 2. RF descriptor tables, RF training and per-structure RF probabilities
python training/rf.py features --features_dir work/features --out_dir work/rf_features
python training/rf.py train    --ids data/splits/train.txt --rf_features_dir work/rf_features --model work/promet_rf.joblib
python training/rf.py predict  --ids data/splits/train.txt data/splits/validation.txt \
                               --rf_features_dir work/rf_features --model work/promet_rf.joblib --out_dir work/rf_probs

# 3. EGNN
python training/train_egnn.py fit --config training/configs/egnn.yaml
```

The input structures are BioLiP entries from the PDB. Metal labels come from the metal ions in each structure; see the paper for the curation of biologically relevant sites. A trained checkpoint can be converted to the inference format (`{"model_config": ..., "state_dict": ...}`) by removing the `model.` prefix from the Lightning `state_dict` keys.

## Citation

If you use PrOMet, please cite:

```
G. Park, C. Seok, H. Park. PrOMet: Accurate and Robust Metal-binding Site and Type Prediction. (under review)
```

## Contact

Gangeun Park (gangeun@snu.ac.kr). Please open an issue for bugs or questions.
