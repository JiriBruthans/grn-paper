#!/usr/bin/env python3

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

# Important for unpickling grn objects created by grn.py
from grn import grn


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

INDIR = Path("/storage/brno2/home/jiribruthans/grn_files/sweep")

# Fine to write the combined files into the same directory.
# You can change this if you prefer a separate directory.
OUTDIR = INDIR
OUTDIR.mkdir(parents=True, exist_ok=True)

N_GENES = 256
N_REPS = 5

# These reproduce the loops in your PBS script exactly
R_VALUES = [2, 3, 4, 5, 8, 16]
DOUT_VALUES = [1, 3, 10, 30, 100, 300]
DIN_VALUES = [10, 30, 50, 100, 200, 300]

# Fixed in your sweep
N_GROUPS = 1
W = 1


# ---------------------------------------------------------------------
# Reconstruct all runs that the PBS script is supposed to generate
# ---------------------------------------------------------------------

expected_runs = []

config = 0

for r in R_VALUES:
    for delta_out in DOUT_VALUES:
        for delta_in in DIN_VALUES:

            config += 1

            for rep in range(1, N_REPS + 1):

                seed = config * 100 + rep

                prefix = (
                    INDIR
                    / f"graph.cfg{config}"
                      f".rep{rep}"
                      f".r{r}"
                      f".dout{delta_out}"
                      f".din{delta_in}"
                      f".seed{seed}"
                )

                expected_runs.append({
                    "config": config,
                    "rep": rep,

                    # Stable ID across repeated calls to combine.py.
                    # 1 ... 1080
                    "sweep_id": (config - 1) * N_REPS + rep,

                    "r": r,
                    "delta_out": delta_out,
                    "delta_in": delta_in,
                    "k": N_GROUPS,
                    "w": W,
                    "seed": seed,

                    "prefix": str(prefix),
                    "gpickle": str(prefix) + ".gpickle",
                    "log": str(prefix) + ".log",
                })


assert config == 216
assert len(expected_runs) == 216 * N_REPS


# ---------------------------------------------------------------------
# Load only completed / valid GRNs
# ---------------------------------------------------------------------

network_rows = []
ko_arrays = []
rna_arrays = []

manifest = []


for run in tqdm(expected_runs, desc="Reading GRNs"):

    gpickle_path = Path(run["gpickle"])

    manifest_row = {
        **run,
        "status": None,
        "network_idx": None,
        "error": "",
    }

    # Run has not completed yet
    if not gpickle_path.exists():
        manifest_row["status"] = "missing"
        manifest.append(manifest_row)
        continue

    try:
        with open(gpickle_path, "rb") as f:
            G = pickle.load(f)

        # A successfully usable GRN must contain both of these
        if not hasattr(G, "rna"):
            raise ValueError("GRN has no 'rna' attribute")

        if not hasattr(G, "ko"):
            raise ValueError("GRN has no 'ko' attribute")

        rna_i = np.asarray(G.rna).reshape(-1)
        ko_i = np.asarray(G.ko)

        # Sanity checks
        if rna_i.shape != (N_GENES,):
            raise ValueError(
                f"Unexpected rna shape {rna_i.shape}; "
                f"expected {(N_GENES,)}"
            )

        if ko_i.shape != (N_GENES, N_GENES):
            raise ValueError(
                f"Unexpected ko shape {ko_i.shape}; "
                f"expected {(N_GENES, N_GENES)}"
            )

        # This is the POSITION this GRN will occupy in ko.npy/rna.npy.
        network_idx = len(network_rows)

        # Match the columns expected by the Figure 5 notebook.
        network_rows.append({
            "files": run["prefix"],
            "n": N_GENES,
            "k": run["k"],
            "r": run["r"],
            "delta_in": run["delta_in"],
            "delta_out": run["delta_out"],
            "w": run["w"],

            # Extra columns useful for tracking your sweep
            "config": run["config"],
            "rep": run["rep"],
            "seed": run["seed"],
            "sweep_id": run["sweep_id"],
        })

        # Copy so G itself can be released from memory
        rna_arrays.append(rna_i.copy())
        ko_arrays.append(ko_i.copy())

        manifest_row["status"] = "combined"
        manifest_row["network_idx"] = network_idx

    except Exception as e:

        # This also catches partially written/corrupted pickle files
        manifest_row["status"] = "invalid"
        manifest_row["error"] = repr(e)

    manifest.append(manifest_row)


# ---------------------------------------------------------------------
# Save combined arrays
# ---------------------------------------------------------------------

if len(network_rows) == 0:
    raise RuntimeError(
        "No complete usable GRNs were found. "
        "Nothing to combine."
    )


networks = pd.DataFrame(network_rows)

# Make the DataFrame index correspond exactly to the first dimension
# of ko.npy and rna.npy:
#
#     networks.loc[0] <-> ko[0] <-> rna[0]
#     networks.loc[1] <-> ko[1] <-> rna[1]
#     ...
#
networks.index = np.arange(len(networks))
networks.index.name = "network_idx"

ko = np.stack(ko_arrays, axis=0)
rna = np.stack(rna_arrays, axis=0)


# Final consistency checks
assert ko.shape[0] == len(networks)
assert rna.shape[0] == len(networks)

assert ko.shape[1:] == (N_GENES, N_GENES)
assert rna.shape[1:] == (N_GENES,)


networks.to_csv(
    OUTDIR / "networks.tsv",
    sep="\t"
)

np.save(
    OUTDIR / "ko.npy",
    ko
)

np.save(
    OUTDIR / "rna.npy",
    rna
)


# ---------------------------------------------------------------------
# Save a manifest showing what has / has not finished
# ---------------------------------------------------------------------

manifest = pd.DataFrame(manifest)

manifest.to_csv(
    OUTDIR / "sweep_manifest.tsv",
    sep="\t",
    index=False
)


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

print()
print("Combined dataset written successfully")
print("-------------------------------------")
print(f"Completed GRNs: {len(networks)}")
print(f"Expected GRNs:  {len(expected_runs)}")
print()
print(f"networks.tsv: {networks.shape}")
print(f"rna.npy:      {rna.shape}")
print(f"ko.npy:       {ko.shape}")
print()
print(manifest["status"].value_counts())