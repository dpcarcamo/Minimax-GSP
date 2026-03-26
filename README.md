# Minimax GSP (Python)

Python translation of the Matlab GSP Ising model utilities. The main entry
points are in `minimax_gsp/gsp.py`.

## Setup

Create and activate the Conda environment:
```
conda env create -f environment.yml
conda activate minimax-gsp
```

## Quick Start

From the project folder:
```
python - <<'PY'
import numpy as np
from minimax_gsp import random_gsp_ising, exact_ising, find_gsp_update

np.random.seed(0)
n = 5
J_true, h_true = random_gsp_ising(n)
mean, corr, _, _ = exact_ising(J_true, h_true, kT=1.0)
h_hat, J_hat, ent, ents = find_gsp_update(mean, corr)

print("J diff:", np.linalg.norm(J_hat - J_true))
print("h diff:", np.linalg.norm(h_hat - h_true))
print("Entropy:", ent)
PY
```

Or open the notebook:
```
jupyter notebook gsp_test.ipynb
```

## Key Functions

- `find_gsp_update(mean, corr)`: greedy GSP construction by entropy reduction.
- `correlations_gsp_01/02/03(J, h)`: GSP correlation routines (0/1 spins).
- `inverse_ising_gsp_01_helper(m, C, step_size)`: 3‑node inverse solver.
- `gsp_fit`, `gsp_fit_topology`, `random_gsp_fit`, `random_tree`.

## Notes

- The code assumes 0/1 spin convention (not ±1).
- `Helper Function/` (Matlab originals) is intentionally ignored by git.
