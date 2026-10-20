"""Command-line interface: `promet predict` and `promet download-weights`."""
import argparse
import sys
import traceback
from pathlib import Path


def _collect_inputs(inputs):
    paths = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            paths += sorted(p.glob("*.pdb"))
        elif p.suffix == ".txt" and p.exists():
            paths += [Path(line.strip()) for line in open(p) if line.strip()]
        else:
            paths.append(p)
    return paths


def cmd_predict(args):
    from promet.output import write_all
    from promet.predict import PrOMet

    pdbs = _collect_inputs(args.input)
    if not pdbs:
        sys.exit("No input PDB files found.")
    print(f"Loading models (device={args.device or 'auto'}) ...", flush=True)
    model = PrOMet(rf_path=args.rf_model, device=args.device, rf_threshold=args.rf_threshold, n_jobs=args.n_jobs)

    failed = []
    for pdb in pdbs:
        try:
            pred = model.predict(pdb, threshold=args.threshold)
        except Exception as e:
            failed.append(pdb)
            print(f"[FAILED] {pdb}: {e}", file=sys.stderr)
            if args.verbose:
                traceback.print_exc()
            continue
        write_all(pred, args.output, save_grids=args.save_grids)
        t = pred.timings
        print(f"{pred.name}: {len(pred.sites)} sites | grids {pred.n_grids_total} -> {len(pred.grid_positions)} after RF"
              f" | {t['features']:.1f}s feat, {t['rf']:.1f}s RF, {t['egnn']:.1f}s EGNN", flush=True)
        if pred.input_metals:
            print(f"  note: {len(pred.input_metals)} metal ion(s) in the input were removed before prediction")
        for i, s in enumerate(pred.sites[: args.show], start=1):
            x, y, z = s.position
            print(f"  site {i:>2}  {s.metal:<2}  p={s.probability:.3f}  ({x:.2f}, {y:.2f}, {z:.2f})")
    print(f"Results written to {Path(args.output).resolve()}")
    if failed:
        sys.exit(f"{len(failed)} structure(s) failed.")


def cmd_download(args):
    from promet.weights import download_rf

    download_rf(dest=args.dest, force=args.force)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="promet", description="PrOMet: protein metal-binding site and type prediction")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predict", help="predict metal-binding sites")
    p.add_argument("input", nargs="+", help="PDB file(s), directories of .pdb files, or .txt lists of PDB paths")
    p.add_argument("-o", "--output", default="promet_output", help="output directory (default: promet_output)")
    p.add_argument("-t", "--threshold", type=float, default=0.5,
                   help="grid probability cutoff applied before clustering into sites (default: 0.5)")
    p.add_argument("--rf-threshold", type=float, default=0.5, help="RF pruning threshold (default: 0.5)")
    p.add_argument("--rf-model", default=None, help="path to the RF parameters (default: $PROMET_HOME/promet_rf.joblib)")
    p.add_argument("--device", default=None, help="torch device, e.g. cpu, cuda, cuda:1 (default: cuda if available)")
    p.add_argument("--n-jobs", type=int, default=-1, help="CPU threads for the RF (default: all)")
    p.add_argument("--save-grids", action="store_true", help="also save grid-level predictions (<name>_grids.npz)")
    p.add_argument("--show", type=int, default=10, help="number of top sites to print per structure")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_predict)

    d = sub.add_parser("download-weights", help="download the Random Forest parameters (~1.8 GB)")
    d.add_argument("--dest", default=None, help="destination file (default: $PROMET_HOME/promet_rf.joblib)")
    d.add_argument("--force", action="store_true")
    d.set_defaults(func=cmd_download)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
