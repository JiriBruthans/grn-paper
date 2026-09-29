# Gene regulatory network structure informs the distribution of perturbation effects

This is the repository for "[Gene regulatory network structure informs the distribution of perturbation effects](https://doi.org/10.1101/2024.07.04.602130)" (Aguirre MA, Spence JP, Sella G, and Pritchard JK; *bioRxiv* 2024). 

Source code can be found in the `src` directory, and code to replicate primary analysis and figures can be found in the `figures` directory. Please refer to the publication for a description of methods and information about correspondence. 

## Knockdown, CRISPRa and overexpression extension

Start with **[the pilot results](MODALITY_RESULTS.md)** and the executed notebooks:

- [small_GRN-modalities.ipynb](small_GRN-modalities.ipynb): APIs, dose sweeps, percentiles and histograms.
- [small_GRN-modalities-robustness.ipynb](small_GRN-modalities-robustness.ipynb): masks, matched denominators, mechanisms, convergence, and additional seeds.

The pilot uses the original 256-gene V1/V3 settings, all singles, and 4,096 matched sampled
pairs per condition. Original notebooks are preserved. No full 2,000-gene run is included.

### Quick start

```bash
python -m pip install -r requirements-modalities.txt
# Download the companion grn-modalities-data.zip supplied with this analysis,
# then extract it into this repository root (it contains a results/ folder):
unzip -o grn-modalities-data.zip
jupyter lab small_GRN-modalities.ipynb
```

The notebooks already contain executed tables and plots and can be read without the archive.
The archive is needed to rerun analysis cells immediately. To regenerate raw data instead:

```bash
python job_script/run_modalities.py --pairs 4096 --workers 4
python job_script/seed_sensitivity.py
python job_script/validate_modalities.py
python job_script/noise_control.py
```

`--pairs all` runs a complete pair census. `--genes 2000 --output results/my_2000_run`
starts a larger sampled screen; use a new output directory when changing settings.
Each condition is checkpointed independently and matching completed checkpoints are reused.
Raw NPZ arrays are kept outside git; summary tables, plots and notebooks are versioned.

### Methods on an existing GRN

```python
# G.rna must already contain a WT baseline.
G.knockdown_nodes([3, 8], remaining=0.1)      # 90% lower synthesis capacity
G.crispri_nodes([3, 8], remaining=0.1)        # alias
G.crispra_nodes([3, 8], fold_change=2)        # endogenous capacity multiplier
G.overexpress_nodes([3, 8], fold_change=30)   # additional constitutive synthesis
G.expression_nodes([3, 8], [0.5, 5], mechanism='clamp')  # exact target doses
```

These methods return dictionaries like `ko_nodes`; use `stats=('new_rna','logfc',
'convergence','diagnostics')` to include achieved doses. They do not mutate the network
or WT baseline. Default simulation is deterministic and checks every condition's drift.
`solver='trajectory'` accepts the original stochastic `simulate_rna` arguments.
`G.simulate_steady_state(save=True)` computes a deterministic WT with no stored trajectory.

The `crispra_nodes` default changes production capacity. For the alternative bounded-promoter
model, pass `mechanism='promoter'`; unattainable requested folds raise unless
`saturation='clip'` is explicitly requested. Exact-dose comparisons use `mechanism='clamp'`.
These are transparent modeling assumptions, not fitted CRISPR efficacy predictions.

Tests: `python -m unittest discover -s tests -v`.
