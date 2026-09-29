#!/usr/bin/env python
"""Small null calibration using the original stochastic/time-average simulator.

Each replicate group has four independent unperturbed trajectories treated as
WT, A, B and AB. This checks the noise floor, not perturbation-specific basin
switching, which would require a larger stochastic screen.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from modality_analysis import load_network

rows = []
for version, seed in [('V1',42), ('V3',5)]:
    G = load_network(ROOT / 'results' / 'modalities_256' / f'{version}_seed{seed}' / 'network.npz')
    for replicate in range(3):
        np.random.seed(12345 + replicate)
        x = G.simulate_rna(x0=G.rna, n=4, s=1e-4, dt=.01,
                           tmax=20000, burnin=5000, save=False)
        mask = (G.rna >= 1e-4) & (x >= 1e-4).all(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            residual = np.log2(x[3])+np.log2(x[0])-np.log2(x[1])-np.log2(x[2])
        errors = np.abs(residual[mask])
        rows.append(dict(network=version, replicate=replicate, eligible_genes=int(mask.sum()),
                         p99_abs_residual=np.quantile(errors,.99),
                         max_abs_residual=errors.max(), fraction_gt_0p1=(errors>.1).mean()))
pd.DataFrame(rows).to_csv(ROOT/'results'/'modalities_256'/'noise_null.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
