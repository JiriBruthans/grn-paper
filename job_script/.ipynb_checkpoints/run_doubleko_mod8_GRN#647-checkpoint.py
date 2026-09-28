from joblib import Parallel, delayed
import sys
import pickle
import numpy as np
import pandas as pd
import networkx as nx

np.random.seed(5)

sys.path.insert(0, "src")

from grn import grn

with open("../graph.647.gpickle", "rb") as f:
    G = pickle.load(f)

module_id = 8

genes_in_module = [
    gene
    for gene, group in G.groups.items()
    if group == module_id
]

genes_in_module = [int(g) for g in genes_in_module]

print(len(genes_in_module))
print(genes_in_module[:20])







n_jobs = 96

def run_single_ko(i):

    result = G.ko_nodes(
        genes=[i],
        stats=("new_rna", "logfc", 'convergence'),
        s=1e-4,
        dt=1e-2,
        tmax=20000,
        burnin=5000,
        tol=1e-3,
    )

    return (
        np.asarray(result["new_rna"]).flatten(),
        np.asarray(result["logfc"]).flatten(),
        np.asarray(result['convergence'])
    )


results = Parallel(
    n_jobs=n_jobs,
    pre_dispatch="n_jobs",
    prefer="processes",
    verbose=10,
)(
    delayed(run_single_ko)(gene)
    for gene in genes_in_module
)

single_expression = np.asarray(
    [r[0] for r in results],
    dtype=np.float32
)

single_logfc = np.asarray(
    [r[1] for r in results],
    dtype=np.float32
)
single_convergence = np.asarray(
    [r[2] for r in results],
    dtype=np.float32
)




np.savez_compressed(
    "../module8_GRN#647_single_ko.npz",
    expression=single_expression,
    logfc=single_logfc,
    convergence=single_convergence,
)


from itertools import combinations

# Generate all unique pairs of genes in the module
all_pairs_module = list(combinations(genes_in_module, 2))


def run_double_ko(pair):
    a, b = pair

    result = G.ko_nodes(
        genes=[a, b],
        stats=("new_rna", "logfc", 'convergence'),
        s=1e-4,
        dt=1e-2,
        tmax=20000,
        burnin=5000,
        tol=1e-3,
    )

    return (
        np.asarray(result["new_rna"]).flatten(),
        np.asarray(result["logfc"]).flatten(),
        np.asarray(result['convergence'])
    )


results = Parallel(
    n_jobs=n_jobs,
    pre_dispatch="n_jobs",
    prefer="processes",
    verbose=10,
)(
    delayed(run_double_ko)(pair)
    for pair in all_pairs_module
)

double_expression = np.asarray(
    [r[0] for r in results],
    dtype=np.float32
)

double_logfc = np.asarray(
    [r[1] for r in results],
    dtype=np.float32
)
double_convergence = np.asarray(
    [r[2] for r in results],
    dtype=np.float32
)




np.savez_compressed(
    "../module8_GRN#647_double_ko.npz",
    expression=double_expression,
    logfc=double_logfc,
    convergence=double_convergence,
)









