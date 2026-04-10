"""
MS-conditioned sampling vs HDF5 reference: Morgan Tanimoto, optional CSV.

Incremental CSV: each row is written and flushed immediately (safe to tail; partial results if interrupted).

Index scope:
  --split all     : HDF5 rows [start_idx, end_idx) (default full table).
  --split test    : only global indices listed in data/metabolite/raw/split_test.npy (same test fold as training).
  --split train|val : same for other folds.

Run:
  python src/evaluate_metabolite_ms_similarity.py --checkpoint CKPT --split test --output_csv out.csv
  python src/evaluate_metabolite_ms_similarity.py --checkpoint CKPT --split all --full_dataset --output_csv out.csv

Hydra overrides after ``--``. Linux GLIBCXX: auto re-exec with CONDA_PREFIX/lib on LD_LIBRARY_PATH.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
_LD_MARKER = "DIGRESS_CONDA_LIBSTDCXX_OK"


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--global_idx", type=int, default=0, help="Single-molecule mode only.")
    p.add_argument(
        "--full_dataset",
        action="store_true",
        help="With --split all: scan HDF5 rows in [start_idx, end_idx). Ignored if --split is train/val/test.",
    )
    p.add_argument(
        "--split",
        choices=("all", "train", "val", "test"),
        default="all",
        help="all=HDF5 row range; test|val|train=only indices from the corresponding split_*.npy (training splits).",
    )
    p.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="For split=all: HDF5 start (inclusive). For split=train|val|test: start offset into that split list.",
    )
    p.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="For split=all: HDF5 end (exclusive). For split=train|val|test: end offset into split list (exclusive).",
    )
    p.add_argument("--output_csv", default=None)
    p.add_argument("--progress_every", type=int, default=100)
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--use_true_n_nodes", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--morgan_radius", type=int, default=2)
    p.add_argument("--morgan_bits", type=int, default=2048)
    p.add_argument("hydra_overrides", nargs="*", help="Hydra overrides")
    return p.parse_args()


def _maybe_reexec_with_conda_libstdcxx():
    conda = os.environ.get("CONDA_PREFIX")
    if not sys.platform.startswith("linux") or not conda or os.environ.get(_LD_MARKER):
        return
    libdir = os.path.join(conda, "lib")
    prev = os.environ.get("LD_LIBRARY_PATH", "")
    if libdir in prev.split(os.pathsep):
        return
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = (libdir + os.pathsep + prev) if prev else libdir
    env[_LD_MARKER] = "1"
    rc = subprocess.call([sys.executable, os.path.abspath(__file__)] + sys.argv[1:], env=env, cwd=str(ROOT))
    raise SystemExit(rc)


def _load_model_and_cfg(args):
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))

    import h5py
    import torch
    from hydra import compose, initialize_config_dir
    from pytorch_lightning.utilities.warnings import PossibleUserWarning
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, DataStructs
    import warnings

    RDLogger.DisableLog("rdApp.*")
    warnings.filterwarnings("ignore", category=PossibleUserWarning)

    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="config", overrides=list(args.hydra_overrides))

    os.chdir(ROOT)

    from analysis.visualization import MolecularVisualization
    from datasets.metabolite_dataset import MetaboliteDataModule, MetaboliteInfos, get_train_smiles
    from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
    from diffusion.extra_features_molecular import ExtraMolecularFeatures
    from diffusion_model_discrete import DiscreteDenoisingDiffusion
    from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
    from metrics.molecular_metrics import SamplingMolecularMetrics
    from src.analysis.rdkit_functions import build_molecule_with_partial_charges, mol2smiles

    if cfg.model.type != "discrete":
        print("Only discrete diffusion is supported.")
        sys.exit(1)
    if getattr(cfg.dataset, "name", None) != "metabolite":
        print("Use dataset=metabolite.")
        sys.exit(1)

    dm = MetaboliteDataModule(cfg)
    infos = MetaboliteInfos(dm, cfg, recompute_statistics=cfg.general.get("recompute_statistics", False))
    train_smiles = get_train_smiles(
        cfg, dm.train_dataloader(), infos, evaluate_dataset=cfg.general.get("evaluate_dataset", False)
    )

    if cfg.model.extra_features is not None:
        xf = ExtraFeatures(cfg.model.extra_features, dataset_info=infos)
        dom = ExtraMolecularFeatures(dataset_infos=infos)
    else:
        xf, dom = DummyExtraFeatures(), DummyExtraFeatures()
    infos.compute_input_output_dims(datamodule=dm, extra_features=xf, domain_features=dom)

    kwargs = {
        "dataset_infos": infos,
        "train_metrics": TrainMolecularMetricsDiscrete(infos),
        "sampling_metrics": SamplingMolecularMetrics(infos, train_smiles),
        "visualization_tools": MolecularVisualization(cfg.dataset.remove_h, dataset_infos=infos),
        "extra_features": xf,
        "domain_features": dom,
    }
    model = DiscreteDenoisingDiffusion.load_from_checkpoint(args.checkpoint, cfg=cfg, map_location="cpu", **kwargs)
    model.eval()
    if cfg.general.gpus > 0 and torch.cuda.is_available():
        model = model.cuda()

    return (
        cfg,
        model,
        infos,
        h5py,
        torch,
        Chem,
        AllChem,
        DataStructs,
        build_molecule_with_partial_charges,
        mol2smiles,
    )


def _pred_smiles_valid(gsmi) -> bool:
    if gsmi is None:
        return False
    if not isinstance(gsmi, str):
        gsmi = str(gsmi)
    s = gsmi.strip()
    if len(s) < 1 or s in ("-1", "?", "nan", "None"):
        return False
    return True


def _resolve_indices(args, n_h5: int, datadir: str) -> tuple[list[int], str]:
    """Returns (global_idx list, human-readable scope description)."""
    raw_dir = ROOT / datadir / "raw"
    if args.split != "all":
        path = raw_dir / f"split_{args.split}.npy"
        if not path.is_file():
            print(f"Split file not found: {path}. Build dataset once or check dataset.datadir.")
            sys.exit(1)
        arr = np.load(path)
        full = arr.tolist()
        e = args.end_idx if args.end_idx is not None else len(full)
        s = max(0, args.start_idx)
        e = min(e, len(full))
        if s >= e:
            print("Empty index range after slicing split.")
            sys.exit(1)
        chosen = full[s:e]
        return chosen, f"split_{args.split}[{s}:{e}] ({len(chosen)} HDF5 global indices)"

    if args.full_dataset:
        end = n_h5 if args.end_idx is None else min(args.end_idx, n_h5)
        start = max(0, args.start_idx)
        if start >= end:
            print("Empty HDF5 index range.")
            sys.exit(1)
        return list(range(start, end)), f"HDF5 rows [{start}, {end}) ({end - start} rows)"

    return [args.global_idx], f"single global_idx={args.global_idx}"


def main():
    args = _parse_args()
    loaded = _load_model_and_cfg(args)
    cfg, model, infos, h5py, torch, Chem, AllChem, DataStructs, build_molecule_with_partial_charges, mol2smiles = loaded

    device = next(model.parameters()).device
    param_dtype = next(model.parameters()).dtype
    hdf5_path = cfg.dataset.hdf5_path

    with h5py.File(hdf5_path, "r") as f:
        n_h5 = int(f["smiles"].shape[0])

    indices, scope_desc = _resolve_indices(args, n_h5, cfg.dataset.datadir)

    if len(indices) == 1 and args.split == "all" and not args.full_dataset:
        if args.global_idx < 0 or args.global_idx >= n_h5:
            print(f"global_idx out of range [0, {n_h5 - 1}].")
            sys.exit(1)

    for g in indices:
        if g < 0 or g >= n_h5:
            print(f"Index {g} out of HDF5 range [0, {n_h5 - 1}]. Check split npy.")
            sys.exit(1)

    out_csv = args.output_csv
    if out_csv is None and len(indices) > 1:
        out_csv = str(Path.cwd() / "metabolite_ms_similarity.csv")

    fieldnames = [
        "global_idx",
        "status",
        "ref_smiles",
        "pred_smiles",
        "n_atoms",
        "mean_tanimoto",
        "max_tanimoto",
    ]

    def evaluate_one(global_idx: int, raw, emb) -> Optional[dict]:
        smi_raw = raw.decode() if isinstance(raw, bytes) else str(raw)
        ref_mol = Chem.MolFromSmiles(smi_raw)
        if ref_mol is None:
            return None
        ref_mol = Chem.RemoveHs(ref_mol) if cfg.dataset.remove_h else Chem.AddHs(ref_mol)
        n_atoms = ref_mol.GetNumAtoms()
        ref_canonical = Chem.MolToSmiles(ref_mol, canonical=True)
        fp_ref = AllChem.GetMorganFingerprintAsBitVect(ref_mol, args.morgan_radius, nBits=args.morgan_bits)

        y_cond = torch.from_numpy(np.asarray(emb, dtype=np.float32)).view(1, -1).expand(args.num_samples, -1).to(
            device=device, dtype=param_dtype
        )
        num_nodes = None
        if args.use_true_n_nodes:
            num_nodes = n_atoms * torch.ones(args.num_samples, device=device, dtype=torch.int)

        model.visualization_tools = None
        with torch.no_grad():
            mol_list = model.sample_batch(
                batch_id=global_idx,
                batch_size=args.num_samples,
                keep_chain=1,
                number_chain_steps=model.number_chain_steps,
                save_final=args.num_samples,
                num_nodes=num_nodes,
                y_condition=y_cond,
            )

        sims = []
        preds = []
        for atom_types, edge_types in mol_list:
            mol = build_molecule_with_partial_charges(
                atom_types.cpu(), edge_types.cpu(), infos.atom_decoder
            )
            gsmi = mol2smiles(mol)
            preds.append(gsmi if _pred_smiles_valid(gsmi) else "")
            if not _pred_smiles_valid(gsmi):
                sims.append(float("nan"))
                continue
            gm = Chem.MolFromSmiles(gsmi)
            if gm is None:
                sims.append(float("nan"))
                continue
            fp_g = AllChem.GetMorganFingerprintAsBitVect(gm, args.morgan_radius, nBits=args.morgan_bits)
            sims.append(DataStructs.TanimotoSimilarity(fp_ref, fp_g))

        ok = [s for s in sims if s == s]
        mean_sim = float(np.mean(ok)) if ok else float("nan")
        max_sim = float(np.max(ok)) if ok else float("nan")
        pred_joined = "|".join(preds)
        return {
            "global_idx": global_idx,
            "status": "evaluated",
            "ref_smiles": ref_canonical,
            "pred_smiles": pred_joined,
            "n_atoms": n_atoms,
            "mean_tanimoto": mean_sim,
            "max_tanimoto": max_sim,
        }

    t0 = time.time()
    means, maxes = [], []
    n_bad_ref = 0
    n_total_loop = len(indices)

    print(f"Scope: {scope_desc}")
    print(f"Total indices to run: {n_total_loop}")

    csv_fp = None
    writer = None
    if out_csv:
        csv_fp = open(out_csv, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
        writer.writeheader()
        csv_fp.flush()
        print(f"Writing incrementally to: {out_csv}")

    try:
        with h5py.File(hdf5_path, "r") as f:
            for done_count, global_idx in enumerate(indices, start=1):
                raw = f["smiles"][global_idx]
                emb = f["DreaMS_embedding"][global_idx]
                row = evaluate_one(global_idx, raw, emb)
                if row is None:
                    n_bad_ref += 1
                    out = {
                        "global_idx": global_idx,
                        "status": "bad_ref_smiles",
                        "ref_smiles": "",
                        "pred_smiles": "",
                        "n_atoms": "",
                        "mean_tanimoto": "",
                        "max_tanimoto": "",
                    }
                else:
                    means.append(row["mean_tanimoto"])
                    maxes.append(row["max_tanimoto"])
                    out = {k: row[k] for k in fieldnames}

                if writer is not None:
                    writer.writerow(out)
                    csv_fp.flush()

                if args.progress_every > 0 and done_count % args.progress_every == 0:
                    print(f"Progress: {done_count}/{n_total_loop} (idx={global_idx})")
    finally:
        if csv_fp is not None:
            csv_fp.close()

    elapsed = time.time() - t0

    vm = [m for m in means if m == m]
    vx = [x for x in maxes if x == x]
    print(f"Done in {elapsed:.1f}s. bad_ref_smiles skipped: {n_bad_ref} / {n_total_loop}")
    if vm:
        print(f"Mean of mean_tanimoto (evaluated rows): {float(np.mean(vm)):.4f}")
    if vx:
        print(f"Mean of max_tanimoto (evaluated rows): {float(np.mean(vx)):.4f}")

    return 0


if __name__ == "__main__":
    _maybe_reexec_with_conda_libstdcxx()
    raise SystemExit(main())
