"""Small reusable helpers for the root-level perturbation notebooks.

The model/API lives in grn.py. This file only batches screens, records their
provenance, and computes the same logFC residual used in the original notebooks.
"""
from pathlib import Path
import hashlib
import json
import platform
import time

import networkx as nx
import numpy as np
import pandas as pd
import scipy
from grn import grn


NETWORKS = {
    'V1': dict(beta=.75, gamma=.25, delta_out=1),
    'V3': dict(beta=.99, gamma=.01, delta_out=100),
}
DEFAULT_SEEDS = {'V1': 42, 'V3': 5}
SOLVER = dict(dt=.2, max_steps=5000, atol=1e-10, rtol=1e-8, check_every=25)
# Doses are a sensitivity grid, not a fitted distribution of experimental efficacy.
CONDITIONS = [
    ('KO', 'clamp', 0.),
    ('KD_99pct', 'production', .01),
    ('KD_90pct', 'production', .1),
    ('KD_50pct', 'production', .5),
    ('KD_25pct', 'production', .75),
    ('CRISPRa_1p25x', 'production', 1.25),
    ('CRISPRa_2x', 'production', 2.),
    ('CRISPRa_5x', 'production', 5.),
    ('CRISPRa_10x', 'production', 10.),
    ('OE_2x', 'transgene', 2.),
    ('OE_10x', 'transgene', 10.),
    ('OE_30x', 'transgene', 30.),
    ('clamp_0p1x', 'clamp', .1),
    ('clamp_0p5x', 'clamp', .5),
    ('clamp_2x', 'clamp', 2.),
    ('clamp_10x', 'clamp', 10.),
    ('clamp_30x', 'clamp', 30.),
    ('promoter_2x', 'promoter', 2.),
    ('promoter_10x', 'promoter', 10.),
]


def make_network(version='V1', seed=None, n=256, solver=None):
    """Recreate notebook graph/weights exactly for a given NumPy seed.

    WT is recomputed deterministically; it is not the old noisy time average.
    All variants retain k=1, kappa=10 and delta_in=100 from the toy notebooks.
    """
    seed = DEFAULT_SEEDS[version] if seed is None else seed
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        G = grn().add_structure(n=n, k=1, alpha=1e-99, kappa=10,
                               delta_in=100, **NETWORKS[version])
    finally:
        np.random.set_state(state)
    settings = {**SOLVER, **(solver or {})}
    baseline = G.simulate_steady_state(save=True, **settings)
    G.screen_metadata = dict(version=version, seed=seed, n=n,
                             baseline_converged=bool(baseline['convergence'][0]),
                             baseline_steps=int(baseline['steps'][0]))
    return G


def choose_pairs(n, count=None, seed=20260928):
    """Uniform unordered pairs without replacement; count=None is a census.

    Sampling avoids allocating n*(n-1)/2 pairs for larger networks.
    """
    total = n * (n - 1) // 2
    if count is None or count >= total:
        return np.column_stack(np.triu_indices(n, 1)).astype(np.int32)
    if count < 1:
        raise ValueError('count must be positive.')
    rng = np.random.default_rng(seed)
    # Uniform ranks mapped into the upper triangle without materializing it.
    ranks = np.sort(rng.choice(total, count, replace=False))
    starts = np.arange(n, dtype=np.int64)
    starts = starts * (2 * n - starts - 1) // 2
    a = np.searchsorted(starts, ranks, side='right') - 1
    b = a + 1 + ranks - starts[a]
    return np.column_stack([a, b]).astype(np.int32)


def graph_hash(G):
    h = hashlib.sha256()
    for a in (G.beta, G.alpha, G.l, G.rna):
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    return h.hexdigest()


def save_network(G, path):
    """Portable numerical snapshot: no pickle and no reliance on future RNG versions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, beta=G.beta, alpha=G.alpha, decay=G.l, wt=G.rna,
                        groups=np.array([G.groups[i] for i in range(G.n)]),
                        metadata_json=json.dumps(G.screen_metadata, sort_keys=True))


def load_network(path):
    with np.load(path, allow_pickle=False) as z:
        state = np.random.get_state()
        try:
            G = grn(z['beta'])
        finally:
            np.random.set_state(state)
        G.alpha, G.l, G.rna = z['alpha'], z['decay'], z['wt']
        G.groups = dict(enumerate(z['groups'].tolist()))
        G.screen_metadata = json.loads(str(z['metadata_json']))
        G.converged = G.screen_metadata['baseline_converged']
    return G


def screen(G, pairs, mechanism, fold_change, batch_size=256, solver=None):
    """All singles plus selected doubles, retaining convergence per condition.

    Each intervention starts at the same WT. Failed runs are saved but never
    silently reinterpreted as equilibria. Memory is O(batch_size*n + pairs*n),
    without a time axis. Persist each returned screen before starting the next.
    """
    if not G.converged:
        raise ValueError('WT did not converge: do not analyze an equilibrium screen.')
    pairs = np.asarray(pairs, dtype=int)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or np.any(pairs[:, 0] >= pairs[:, 1]):
        raise ValueError('pairs must have shape (m, 2), with a < b.')
    if np.any(pairs < 0) or np.any(pairs >= G.n) or len(np.unique(pairs, axis=0)) != len(pairs):
        raise ValueError('Pairs must be unique and within the network.')
    if batch_size < 1:
        raise ValueError('batch_size must be positive.')
    settings = {**SOLVER, **(solver or {})}
    wt = G.rna.copy()
    capped = np.zeros(G.n, dtype=bool)
    single_parameters = []
    for gene in range(G.n):
        pars, info = G.expression_parameters([gene], fold_change, mechanism, saturation='clip')
        single_parameters.append(pars)
        capped[gene] = info['capped'][0]
    by_key = {key: np.stack([p[key] for p in single_parameters]) for key in single_parameters[0]}
    result = dict(wt=wt, pairs=pairs, capped=capped)
    for label, targets in [('single', np.arange(G.n)[:, None]), ('double', pairs)]:
        values = np.empty((len(targets), G.n), dtype=np.float64)
        converged = np.zeros(len(targets), dtype=bool)
        steps = np.empty(len(targets), dtype=int)
        drift = np.empty(len(targets))
        for start in range(0, len(targets), batch_size):
            t = targets[start:start + batch_size]
            pars = {}
            for key in ('production_scale', 'extra_production', 'alpha_shift'):
                pars[key] = by_key[key][t].sum(axis=1)
                if key == 'production_scale':
                    pars[key] -= t.shape[1] - 1
            pars['clamp'] = np.full((len(t), G.n), np.nan)
            if mechanism == 'clamp':
                pars['clamp'][np.arange(len(t))[:, None], t] = fold_change * wt[t]
            out = G.simulate_steady_state(x0=np.tile(wt, (len(t), 1)), **pars, **settings)
            sl = slice(start, start + len(t))
            values[sl] = out['new_rna']
            converged[sl] = out['convergence']
            steps[sl] = out['steps']
            drift[sl] = out['max_abs_drift']
        result[label] = values
        result[label + '_converged'] = converged
        result[label + '_steps'] = steps
        result[label + '_drift'] = drift
    result['metadata'] = dict(network=getattr(G, 'screen_metadata', {}),
        graph_sha256=graph_hash(G), mechanism=mechanism, fold_change=float(fold_change),
        solver=settings, python=platform.python_version(), numpy=np.__version__,
        scipy=scipy.__version__, networkx=nx.__version__)
    return result


def save_screen(data, path):
    """Float64 NPZ, readable without pickle; atomic completion for safe resumes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in data.items() if k != 'metadata'}
    payload['metadata_json'] = json.dumps(data['metadata'], sort_keys=True)
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def load_screen(path):
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files if k != 'metadata_json'}
        out['metadata'] = json.loads(str(z['metadata_json']))
    return out


def residuals(data, expression_floor=1e-4, floor_policy='all', exclude_targets=True):
    """Log-additivity, expression-additivity, and exact denominator masks.

    LFC residual = log2(xAB*xWT/(xA*xB)); zero means multiplicative expression.
    Raw residual = (xAB-xA-xB+xWT)/xWT; zero means additive expression changes.
    Primary filtering requires WT, both singles, and double >= expression_floor.
    Also inspect floor_policy='wt' and floor=0 as sensitivity analyses.
    """
    wt, singles, doubles, pairs = (data[k] for k in ('wt', 'single', 'double', 'pairs'))
    a, b = singles[pairs[:, 0]], singles[pairs[:, 1]]
    valid_pairs = data['double_converged'] & data['single_converged'][pairs].all(axis=1)
    mask = np.broadcast_to((wt >= expression_floor) & (wt > 0), doubles.shape).copy()
    mask &= valid_pairs[:, None]
    if floor_policy == 'all':
        mask &= (a >= expression_floor) & (b >= expression_floor) & (doubles >= expression_floor)
    elif floor_policy != 'wt':
        raise ValueError("floor_policy must be 'all' or 'wt'.")
    mask &= (a > 0) & (b > 0) & (doubles > 0)
    if exclude_targets:
        mask[np.arange(len(pairs))[:, None], pairs] = False
    with np.errstate(divide='ignore', invalid='ignore'):
        log_single = np.log2(singles) - np.log2(wt)
        log_double = np.log2(doubles) - np.log2(wt)
        log_residual = log_double - log_single[pairs[:, 0]] - log_single[pairs[:, 1]]
        linear_residual = (doubles - a - b + wt) / wt
    mask &= np.isfinite(log_residual) & np.isfinite(linear_residual)
    return dict(log=log_residual, linear=linear_residual, mask=mask,
                valid_pairs=valid_pairs, single_lfc=log_single, double_lfc=log_double)


def shared_downstream(G, pairs):
    """Reachability of the functional beta matrix, including feedback paths."""
    graph = nx.from_numpy_array((G.beta != 0).astype(int), create_using=nx.DiGraph)
    descendants = np.zeros((G.n, G.n), dtype=bool)
    for i in range(G.n):
        descendants[i, list(nx.descendants(graph, i))] = True
    common = descendants[pairs[:, 0]] & descendants[pairs[:, 1]]
    common[np.arange(len(pairs))[:, None], pairs] = False
    return common


def summarize(data, G=None, expression_floor=1e-4, floor_policy='all', exclude_targets=True):
    r = residuals(data, expression_floor, floor_policy, exclude_targets)
    mask, errors = r['mask'], r['log'][r['mask']]
    pairs = data['pairs']
    eligible_pairs = mask.any(axis=1)
    meta = data['metadata']
    row = dict(network=meta['network'].get('version', 'custom'),
        seed=meta['network'].get('seed'), mechanism=meta['mechanism'], dose=meta['fold_change'],
        n_pairs=len(pairs), n_single_converged=int(data['single_converged'].sum()),
        n_double_converged=int(data['double_converged'].sum()),
        n_valid_pairs=int(r['valid_pairs'].sum()), n_eligible_pairs=int(eligible_pairs.sum()),
        n_readouts=int(mask.sum()), expression_floor=expression_floor,
        floor_policy=floor_policy, exclude_targets=exclude_targets,
        target_cap_fraction=float(data['capped'].mean()))
    target_folds = np.diag(data['single']) / data['wt']
    target_folds = target_folds[data['single_converged']]
    for q in (10, 50, 90):
        row[f'achieved_fold_p{q}'] = float(np.percentile(target_folds, q)) if len(target_folds) else np.nan
    for q in (.1, 1, 5, 50, 95, 99, 99.9):
        row[f'residual_p{q:g}'] = float(np.percentile(errors, q)) if len(errors) else np.nan
    for q in (50, 90, 95, 99, 99.9):
        row[f'abs_residual_p{q:g}'] = float(np.percentile(np.abs(errors), q)) if len(errors) else np.nan
    for threshold in (.01, .1, .5, 1.):
        hit = (np.abs(r['log']) > threshold) & mask
        row[f'readout_gt_{threshold:g}'] = float(hit.sum() / mask.sum()) if mask.any() else np.nan
        row[f'pair_any_gt_{threshold:g}'] = float(hit.any(axis=1)[eligible_pairs].mean()) if eligible_pairs.any() else np.nan
    if mask.any():
        row['linear_readout_gt_0.1'] = float((np.abs(r['linear'][mask]) > .1).mean())
        row['log_additive_rmse'] = float(np.sqrt(np.mean(errors ** 2)))
    if G is not None:
        shared = shared_downstream(G, pairs) & mask
        effective = (np.abs(r['single_lfc'][pairs[:, 0]]) > .1) & (np.abs(r['single_lfc'][pairs[:, 1]]) > .1) & mask
        for name, subset in [('shared_downstream', shared), ('shared_effective', effective)]:
            row[name + '_n'] = int(subset.sum())
            row[name + '_gt_0.1'] = float((np.abs(r['log'][subset]) > .1).mean()) if subset.any() else np.nan
        row['no_shared_path_any_gt_0.1'] = int((((np.abs(r['log']) > .1) & mask).any(axis=1) & ~shared_downstream(G, pairs).any(axis=1)).sum())
    return row


def run_conditions(G, pairs, output, conditions=CONDITIONS, batch_size=256, solver=None):
    """Checkpoint one condition at a time; resume only matching cached settings."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, mechanism, fold in conditions:
        path = output / (name + '.npz')
        start = time.perf_counter()
        if path.exists():
            data = load_screen(path)
            assert data['metadata']['graph_sha256'] == graph_hash(G), 'Cached graph mismatch'
            assert np.array_equal(data['pairs'], pairs), 'Cached pair mismatch'
            assert data['metadata']['solver'] == {**SOLVER, **(solver or {})}, 'Cached solver mismatch'
            assert data['metadata']['mechanism'] == mechanism and data['metadata']['fold_change'] == fold
        else:
            data = screen(G, pairs, mechanism, fold, batch_size, solver)
            data['metadata']['condition'] = name
            save_screen(data, path)
        row = summarize(data, G)
        row['condition'] = name
        rows.append(row)
        pd.DataFrame(rows).to_csv(output / 'summary.csv', index=False)
        print(f'{name}: {row["n_valid_pairs"]}/{len(pairs)} valid pairs; '
              f'{100*row["readout_gt_0.1"]:.3f}% readouts, '
              f'{100*row["pair_any_gt_0.1"]:.2f}% pairs; {time.perf_counter()-start:.1f}s', flush=True)
    return pd.DataFrame(rows)
