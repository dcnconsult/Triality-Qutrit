# Triality Qutrit Hotspot — Triad Resonance Spectroscopy

Concrete, falsifiable experiment and analysis code to test the **Triality** prediction of an **asymmetric coherence hotspot** in a driven qutrit (transmon) under a strong pump (|g⟩↔|f⟩ two-photon) and two weaker signal drives (|g⟩↔|e⟩ and |e⟩↔|f⟩).

## TL;DR
- Sweep signal powers (P_ge, P_ef) with pump P_gf held high and constant.
- Measure coherence (T2* via Ramsey on e↔f) at each point; build a 2D map.
- Success = a reproducible, localized coherence island in the **asymmetric** region (both signals ≪ pump).
- Repo ships synthetic dataset + analysis pipeline, Bayesian hotspot search, preregistered statistics, and falsification checks.

## Repo Layout
```
.
├── LICENSE
├── README.md
├── environment.yml
├── requirements.txt
├── config/
│   └── experiment.default.yaml
├── data/
│   ├── synthetic/
│   │   ├── sweep_metadata.json
│   │   └── triad_sweep.csv
│   └── schema/
│       └── dataset_schema.json
├── notebooks/
│   ├── 01_visualize_heatmap.ipynb
│   ├── 02_hotspot_significance.ipynb
│   └── 03_bayes_optimization_demo.ipynb
└── src/
    └── triality_qutrit/
        ├── __init__.py
        ├── io.py
        ├── simulate.py
        ├── stats.py
        ├── analyze.py
        ├── plotting.py
        └── optimize.py
```

## Quickstart
```bash
# (Optional) Conda
conda env create -f environment.yml
conda activate triality-qutrit

# Or pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run basic analysis
python -m triality_qutrit.analyze --data data/synthetic/triad_sweep.csv --out out/
```

## Falsification Gates (pre-registered)
1. No significant spatial structure vs shuffled labels (p ≥ 0.05) → **Fail**.
2. Maximum T2* aligned with symmetry diagonal P_ge≈P_ef or near-equal to pump → **Fail**.
3. Hotspot not reproducible across seeds/batches/bootstraps → **Fail**.
4. Hotspot vanishes when phases are randomized in control runs → **Fail**.

## Data Standard
Rows = one Ramsey measurement at a specific (P_ge, P_ef) with fixed P_gf, fixed phases unless otherwise noted.
Columns:
- run_id, batch_id, timestamp
- f_ge_Hz, f_ef_Hz, f_gf_Hz
- P_pump_dBm, P_ge_dBm, P_ef_dBm
- phi_ge_rad, phi_ef_rad, phi_gf_rad
- T2star_ns (primary endpoint), T1_ns (optional), contrast, fit_ok (bool)
- temp_mK, fridge_cycle_id, resonator_Q, notes

See `data/schema/dataset_schema.json` for full validation.

## What To Do On Real Hardware
- Replace synthetic CSV with instrument stream (same columns).
- Keep pump fixed high; raster-scan P_ge, P_ef; measure T2* via Ramsey on e↔f.
- Use `src/triality_qutrit/optimize.py` to switch to bandit/BO for faster hotspot discovery.

**This is a research scaffold, not medical/mission-critical software.**
