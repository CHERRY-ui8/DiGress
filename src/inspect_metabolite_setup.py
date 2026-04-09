"""
Lightweight checks for metabolite training setup (no training loop).

Run from the digress repo root (same as main.py):
    python src/inspect_metabolite_setup.py

Optional overrides (Hydra), e.g. different HDF5 path:
    python src/inspect_metabolite_setup.py dataset.hdf5_path=/path/to/file.hdf5

Exit code 1 if batch.y looks empty (common with stale proc_*.pt caches).

RDKit + PyTorch: GLIBCXX errors happen when the system libstdc++ is too old.
Setting LD_LIBRARY_PATH inside Python is too late once PyTorch has loaded; this
script re-executes itself once with CONDA_PREFIX/lib on LD_LIBRARY_PATH (Linux).

You can instead export before any Python command (including training):
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # digress repo root

_LD_MARKER = "DIGRESS_CONDA_LIBSTDCXX_OK"


if __name__ == "__main__":
    # Re-exec so the dynamic linker sees conda libstdc++ before PyTorch/RDKit load.
    _conda = os.environ.get("CONDA_PREFIX")
    if sys.platform.startswith("linux") and _conda and not os.environ.get(_LD_MARKER):
        _libdir = os.path.join(_conda, "lib")
        _prev = os.environ.get("LD_LIBRARY_PATH", "")
        if _libdir not in _prev.split(os.pathsep):
            _env = os.environ.copy()
            _env["LD_LIBRARY_PATH"] = (_libdir + os.pathsep + _prev) if _prev else _libdir
            _env[_LD_MARKER] = "1"
            _rc = subprocess.call(
                [sys.executable, os.path.abspath(__file__)] + sys.argv[1:],
                env=_env,
                cwd=str(ROOT),
            )
            raise SystemExit(_rc)

    # Resolve imports like main.py: `src.*` from repo root, top-level modules under src/.
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    cli = sys.argv[1:]
    # Root config.yaml already defaults to dataset=metabolite; extra CLI only when needed.
    overrides = cli

    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="config", overrides=overrides)

    os.chdir(ROOT)

    from datasets.metabolite_dataset import MetaboliteDataModule, MetaboliteInfos
    from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
    from diffusion.extra_features_molecular import ExtraMolecularFeatures

    print("=== Hydra overrides used ===")
    print("dataset.name:", cfg.dataset.name)
    print("dataset.datadir:", cfg.dataset.datadir)
    print("dataset.hdf5_path:", cfg.dataset.hdf5_path)
    print("dataset.metabolite_use_data_statistics:", cfg.dataset.get("metabolite_use_data_statistics", True))
    print("general.recompute_statistics:", cfg.general.get("recompute_statistics", False))
    print()

    dm = MetaboliteDataModule(cfg)
    infos = MetaboliteInfos(
        dm,
        cfg,
        recompute_statistics=cfg.general.get("recompute_statistics", False),
    )

    # Match training: build extra / domain features before dims are final.
    if cfg.model.type == "discrete" and cfg.model.extra_features is not None:
        extra = ExtraFeatures(cfg.model.extra_features, dataset_info=infos)
        domain = ExtraMolecularFeatures(dataset_infos=infos)
    else:
        extra = DummyExtraFeatures()
        domain = DummyExtraFeatures()
    infos.compute_input_output_dims(datamodule=dm, extra_features=extra, domain_features=domain)

    print("=== MetaboliteInfos (used by marginal transition) ===")
    print("node_types (sum should be ~1.0):", infos.node_types.sum().item())
    print("node_types:", infos.node_types)
    print("edge_types (sum should be ~1.0):", infos.edge_types.sum().item())
    print("edge_types:", infos.edge_types)
    nz = (infos.n_nodes > 0).sum().item()
    print(f"n_nodes: len={len(infos.n_nodes)}, nonzero bins={nz}")
    if nz <= 1:
        print("  WARNING: n_nodes is mostly zeros — still placeholder? Run with general.recompute_statistics=True.")
    print("valency_distribution nonzero bins:", (infos.valency_distribution > 0).sum().item())
    print()

    batch = next(iter(dm.train_dataloader()))
    yshape = tuple(batch.y.shape)
    ynum = batch.y.numel()
    print("=== First training batch ===")
    print("batch.y shape:", yshape, " numel:", ynum)
    if ynum == 0:
        print("FAIL: batch.y is empty — DreaMS conditioning is not in the graph batch.")
        print("      Delete data/metabolite/processed/proc_* .pt and rerun to rebuild from HDF5.")
        sys.exit(1)
    print("input_dims['y'] (incl. time + extra):", infos.input_dims["y"])
    print("output_dims['y'] (discrete CE head, often 0):", infos.output_dims["y"])
    print()
    print("OK — batch.y present; marginal placeholders: compare node_types/edge_types/n_nodes above with recompute output.")
