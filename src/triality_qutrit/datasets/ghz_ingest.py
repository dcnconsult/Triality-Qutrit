from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json

def parse_args():
    ap = argparse.ArgumentParser(description="Ingest three-qutrit GHZ counts into unified schema.")
    ap.add_argument("--in", required=True, help="Path to raw counts CSV from GHZ repo")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--map", default=None, help="JSON mapping from raw->standard columns")
    ap.add_argument("--device", default="unknown", help="Device identifier")
    ap.add_argument("--run_id", default="ghz_public_001", help="Run identifier")
    return ap.parse_args()

def base3_to_tuple(s: str):
    return tuple(int(c) for c in str(s))

def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.in)
    # Standardize columns
    colmap = {"shot_count":"shot_count","setting_id":"setting_id","outcome":"outcome","count":"count"}
    if args.map:
        user_map = json.loads(args.map)
        df = df.rename(columns=user_map)
    # Normalize per setting
    probs = (df.groupby(["setting_id","outcome"])["count"].sum()
               .groupby(level=0).apply(lambda s: s/s.sum())
               .reset_index(name="prob"))
    # Expand outcomes
    splits = probs["outcome"].astype(str).apply(base3_to_tuple)
    probs[["q0","q1","q2"]] = pd.DataFrame(splits.tolist(), index=probs.index)
    probs.to_csv(outdir/"ghz_probs.csv", index=False)

    # Minimal schema-like events CSV
    rows = []
    for sid, grp in probs.groupby("setting_id"):
        p = grp["prob"].values
        entropy = float(-(p*np.log(p+1e-15)).sum())
        rows.append(dict(
            run_id=f"{args.run_id}_{sid}",
            batch_id="ghz_ingest",
            timestamp="",
            f_ge_Hz=0.0, f_ef_Hz=0.0, f_gf_Hz=0.0,
            P_pump_dBm=0.0, P_ge_dBm=0.0, P_ef_dBm=0.0,
            phi_ge_rad=0.0, phi_ef_rad=0.0, phi_gf_rad=0.0,
            T2star_ns=0.0, T1_ns=0.0, contrast=0.0, fit_ok=True,
            temp_mK=0.0, fridge_cycle_id=args.device, resonator_Q=0.0,
            notes=f"GHZ setting {sid}; entropy={entropy:.3f}"
        ))
    pd.DataFrame(rows).to_csv(outdir/"ghz_schema_events.csv", index=False)
    print("Wrote:", outdir/"ghz_probs.csv", "and", outdir/"ghz_schema_events.csv")

if __name__ == "__main__":
    main()
