# FreeEagle (a forked and adapted version)

This repository is a fork adapted to support batch model detection and a reproducible Poetry-managed Python environment.

Original project: [Free Eagle Official Repository](https://github.com/FuChong-cyber/Data-Free-Neural-Backdoor-Detector-FreeEagle)
Original README: [readme-origin.md](./README-origin.md)

Purpose
-------
Run FreeEagle-style detection over many saved models, collect results into CSV files, and compute consolidated evaluation metrics (AUROC, F1, TPR, FPR).

Modifications of this fork
----------------------
- Poetry-based dependency management and a committed lockfile for reproducibility.
- Scripts and helpers for batch execution, log parsing, and result aggregation.
- Minor fixes to model configuration (kernel-size).

Repository layout (key items)
----------------------------
- `MLBackdoorDetection/` — core detection and analysis code.
- `batch_free_eagle.py` — run FreeEagle over a model set and write logs.
- `analyse_log.py` — extract metrics from logs into CSV.
- `final_results.py` — compute final evaluation metrics from benign/trojan CSVs.
- `readme-origin.md` — original README from upstream.

Environment and installation (recommended)
-----------------------------------------
Two common flows:

1) Install into an existing conda environment (recommended for binary/CUDA packages):

```bash
conda create -n freeeagle python=3.8 -y
conda activate freeeagle
pip install --user poetry   # or pip install poetry

# Make Poetry install into the current env (do not create a separate venv)
poetry config virtualenvs.create false --local
poetry install --no-root
```

2) Let Poetry manage a virtualenv for the project:

```bash
poetry install
poetry run python batch_free_eagle.py --help
```

Notes
- Install heavy binary dependencies (MKL, Intel libs, CUDA-specific PyTorch builds, tbb, triton) with `conda` to ensure ABI/CUDA compatibility. Keep those out of Poetry-managed dependencies when appropriate.
- If you do not want Poetry to install the project package itself, use `poetry install --no-root` or set `package-mode = false` in `pyproject.toml`.

Quick usage
-----------
Run batch detection:

```bash
poetry run python batch_free_eagle.py --dataset cifar10 --model resnet18 --type benign > batch_bengin.log
```

Extract CSV from logs:

```bash
poetry run python analyse_log.py logs batch_benign.log -o ./results/benign.csv
```

Compute final metrics:

```bash
poetry run python final_results.py --benign_csv ./results/benign.csv --trojan_csv ./results/badnet.csv --seed 42
```

Recent changes
--------------
- Unified kernel-size inconsistencies in model configuration; ensure custom models match repository defaults.
- Updated Poetry guidance: prefer `poetry install --no-root` when installing into an existing conda env to avoid modifying system-level packaging tools.



License
-------
See the `LICENSE` file in the repository root.
