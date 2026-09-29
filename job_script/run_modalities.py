#!/usr/bin/env python
"""Reproducible CPU pilot; run from anywhere, results beside the notebooks.

python job_script/run_modalities.py --pairs 4096 --workers 4
python job_script/run_modalities.py --pairs all --networks V1 --workers 4
Use --genes 2000 only when intentionally expanding beyond the toy pilot.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from modality_analysis import (CONDITIONS, SOLVER, make_network, choose_pairs,
                               run_conditions, save_network)


def run_one(version, seed, genes, pair_count, condition, output):
    G = make_network(version, seed, genes)
    pairs = choose_pairs(genes, pair_count)
    path = Path(output) / f'{version}_seed{G.screen_metadata["seed"]}'
    # Each worker writes only its own screen and condition summary.
    start = time.time()
    df = run_conditions(G, pairs, path / condition[0], conditions=[condition])
    row = df.iloc[0].to_dict()
    row['elapsed_seconds'] = time.time() - start
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', default='4096')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--genes', type=int, default=256)
    parser.add_argument('--networks', nargs='+', choices=['V1', 'V3'], default=['V1', 'V3'])
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--conditions', nargs='+', default=None)
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'modalities_256')
    args = parser.parse_args()
    count = None if args.pairs == 'all' else int(args.pairs)
    selected = [c for c in CONDITIONS if args.conditions is None or c[0] in args.conditions]
    if args.conditions and set(args.conditions) - {c[0] for c in selected}:
        parser.error('Unknown condition name')
    args.output.mkdir(parents=True, exist_ok=True)
    for version in args.networks:
        graph = make_network(version, args.seed, args.genes)
        save_network(graph, args.output / f'{version}_seed{graph.screen_metadata["seed"]}' / 'network.npz')
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, v, args.seed, args.genes, count, c, args.output)
                   for v in args.networks for c in selected]
        for task in as_completed(futures):
            rows.append(task.result())
            pd.DataFrame(rows).sort_values(['network', 'condition']).to_csv(args.output / 'summary.csv', index=False)
    manifest = dict(genes=args.genes, pairs=count, pair_seed=20260928,
                    networks=args.networks, seed=args.seed, conditions=selected,
                    solver=SOLVER, command=sys.argv,
                    note='Deterministic equilibria; all failed singles/doubles are retained with flags.')
    (args.output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
