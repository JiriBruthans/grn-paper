#!/usr/bin/env python
"""Repeat the stability scan and the predefined additional-seed sensitivity checks."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from modality_analysis import make_network, choose_pairs, run_conditions, CONDITIONS

rows = []
for version, seeds in [('V1', [42, 5, 7]), ('V3', list(range(10)))]:
    for seed in seeds:
        G = make_network(version, seed)
        p = G.link(G.alpha.ravel() + G.rna @ G.beta)
        J = (p*(1-p))[:, None] * G.beta.T - np.diag(G.l.ravel())
        rows.append(dict(version=version, seed=seed, converged=G.converged,
            max_real_eigenvalue=np.linalg.eigvals(J).real.max(),
            functional_edges=int((G.beta != 0).sum()),
            regulators=int((G.beta != 0).any(axis=1).sum()),
            wt_above_floor=int((G.rna >= 1e-4).sum())))
out = ROOT / 'results' / 'modalities_256'
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out / 'network_diagnostics.csv', index=False)

conditions = [c for c in CONDITIONS if c[0] in ('KO','KD_90pct','CRISPRa_10x','OE_30x')]
for version, seeds, count, filename in [('V1', [5,7], 1024, 'summary.csv'),
                                       ('V3', [3,7,9], 512, 'summary_V3.csv')]:
    summaries = []
    for seed in seeds:
        G = make_network(version, seed)
        assert G.converged, 'Stability scan differs from the stored pilot; inspect before proceeding.'
        folder = ROOT / 'results' / 'seed_sensitivity' / f'{version}_seed{seed}'
        summaries.append(run_conditions(G, choose_pairs(G.n, count), folder, conditions))
    pd.concat(summaries).to_csv(ROOT / 'results' / 'seed_sensitivity' / filename, index=False)
