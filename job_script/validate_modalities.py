#!/usr/bin/env python
"""Numerical sensitivity on the actual 256-gene screens (not a second full run).

Checks KO, KD90%, CRISPRa10x and OE30x on eight uniformly selected pairs plus two
largest residual pairs and up to two failed pairs per screen. Reintegrates with half
the step size, twice the time horizon, and tenfold tighter tolerances. A smaller
set is also checked against SciPy's independent adaptive DOP853 integrator.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from modality_analysis import make_network, load_screen, residuals, CONDITIONS


def main():
    results = ROOT / 'results' / 'modalities_256'
    rng = np.random.default_rng(20260929)
    rows, adaptive_rows = [], []
    for version in ('V1', 'V3'):
        G = make_network(version)
        folder = results / f'{version}_seed{G.screen_metadata["seed"]}'
        for name, mechanism, fold in CONDITIONS:
            if name not in ('KO', 'KD_90pct', 'CRISPRa_10x', 'OE_30x'):
                continue
            path = folder / name / (name + '.npz')
            data = load_screen(path)
            r = residuals(data)
            score = np.max(np.where(r['mask'], np.abs(r['log']), 0.), axis=1)
            selected = np.unique(np.r_[rng.choice(len(score), 8, replace=False),
                                       np.argsort(score)[-2:],
                                       np.flatnonzero(~data['double_converged'])[:2]])
            for label, ids, targets in [
                ('double', selected, data['pairs'][selected]),
                ('single', np.unique(data['pairs'][selected]), np.unique(data['pairs'][selected])[:, None])]:
                max_diff, max_logdiff = [], []
                comparable, agreement, recovered = 0, 0, 0
                for idx, genes in zip(ids, targets):
                    pars, _ = G.expression_parameters(genes, fold, mechanism, saturation='clip')
                    refined = G.simulate_steady_state(x0=G.rna, **pars, dt=.1,
                                max_steps=20000, atol=1e-11, rtol=1e-9)
                    old_ok = bool(data[label + '_converged'][idx])
                    new_ok = bool(refined['convergence'][0])
                    agreement += old_ok == new_ok
                    recovered += not old_ok and new_ok
                    if old_ok and new_ok:
                        comparable += 1
                        original, new = data[label][idx], refined['new_rna'][0]
                        mask = (original >= 1e-4) & (new >= 1e-4)
                        mask[genes] = False
                        max_diff.append(float(np.max(np.abs(original-new))))
                        max_logdiff.append(float(np.max(np.abs(np.log2(original[mask]/new[mask])))))
                rows.append(dict(network=version, condition=name, kind=label, checked=len(ids),
                    comparable=comparable, convergence_agreement=agreement,
                    recovered_with_longer_run=recovered, max_expression_difference=max(max_diff,default=np.nan),
                    max_log2_difference_above_floor=max(max_logdiff,default=np.nan)))
            # Independent adaptive integration of two high-residual, converged doubles.
            if name in ('KO','KD_90pct','CRISPRa_10x','OE_30x'):
                for idx in np.argsort(score)[-2:]:
                    genes = data['pairs'][idx]
                    pars, _ = G.expression_parameters(genes, fold, mechanism, saturation='clip')
                    fixed = np.isfinite(pars['clamp'])
                    initial = G.rna.copy(); initial[fixed] = pars['clamp'][fixed]
                    def rhs(t, x):
                        drift = pars['production_scale'] * G.link(G.alpha.ravel()+pars['alpha_shift']+x@G.beta)
                        drift += pars['extra_production']-G.l.ravel()*x
                        drift[fixed] = 0.
                        return drift
                    out = solve_ivp(rhs, (0, 1000), initial, method='DOP853', rtol=1e-10, atol=1e-12)
                    ref = data['double'][idx]
                    observed = out.y[:, -1]
                    mask = (ref >= 1e-4) & (observed >= 1e-4); mask[genes] = False
                    adaptive_rows.append(dict(network=version, condition=name, pair_index=int(idx),
                        success=out.success, final_max_drift=float(np.max(np.abs(rhs(1000,observed)))),
                        max_log2_difference_above_floor=float(np.max(np.abs(np.log2(ref[mask]/observed[mask]))))))
            pd.DataFrame(rows).to_csv(results / 'numerical_validation.csv', index=False)
            pd.DataFrame(adaptive_rows).to_csv(results / 'adaptive_ode_validation.csv', index=False)
            print(version, name, 'validated', flush=True)


if __name__ == '__main__':
    main()
