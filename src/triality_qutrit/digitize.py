
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import yaml

def load_config(cfg_path: Path):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def build_rows(df: pd.DataFrame, cfg: dict, meta: dict) -> pd.DataFrame:
    xcol = cfg['columns']['x']
    ycol = cfg['columns']['y']
    zcol = cfg['columns']['z']
    xscale = cfg.get('scales', {}).get('x', 1.0)
    yscale = cfg.get('scales', {}).get('y', 1.0)
    zscale = cfg.get('scales', {}).get('z', 1.0)
    xoff = cfg.get('offsets', {}).get('x', 0.0)
    yoff = cfg.get('offsets', {}).get('y', 0.0)
    zoff = cfg.get('offsets', {}).get('z', 0.0)

    use = df[[xcol, ycol, zcol]].dropna().copy()
    use['P_ge_dBm'] = use[xcol] * xscale + xoff
    use['P_ef_dBm'] = use[ycol] * yscale + yoff
    use['T2star_ns'] = use[zcol] * zscale + zoff

    out = pd.DataFrame({
        "run_id": [f"wdp_{i:06d}" for i in range(len(use))],
        "batch_id": meta.get('batch_id', 'wdp_import'),
        "timestamp": meta.get('timestamp', ''),
        "f_ge_Hz": meta.get('f_ge_Hz', 0.0),
        "f_ef_Hz": meta.get('f_ef_Hz', 0.0),
        "f_gf_Hz": meta.get('f_gf_Hz', 0.0),
        "P_pump_dBm": meta.get('P_pump_dBm', 0.0),
        "P_ge_dBm": use['P_ge_dBm'].values,
        "P_ef_dBm": use['P_ef_dBm'].values,
        "phi_ge_rad": meta.get('phi_ge_rad', 0.0),
        "phi_ef_rad": meta.get('phi_ef_rad', 0.0),
        "phi_gf_rad": meta.get('phi_gf_rad', 0.0),
        "T2star_ns": use['T2star_ns'].values,
        "T1_ns": meta.get('T1_ns', 0.0),
        "contrast": meta.get('contrast', 0.0),
        "fit_ok": True,
        "temp_mK": meta.get('temp_mK', 0.0),
        "fridge_cycle_id": meta.get('fridge_cycle_id', 'wdp'),
        "resonator_Q": meta.get('resonator_Q', 0.0),
        "notes": meta.get('notes', 'WPD imported')
    })
    return out

def main():
    ap = argparse.ArgumentParser(description="Import WebPlotDigitizer CSV into Triality schema CSV.")
    ap.add_argument('--in', dest='inp', required=True, help='Path to WPD-exported CSV')
    ap.add_argument('--config', required=True, help='YAML mapping config')
    ap.add_argument('--out', required=True, help='Output CSV path (schema-compliant)')
    ap.add_argument('--meta', default=None, help='Optional YAML with metadata (frequencies, pump, etc.)')
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    df = pd.read_csv(args.inp)
    meta = {}
    if args.meta:
        meta = load_config(Path(args.meta))
    out = build_rows(df, cfg, meta)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} rows to {args.out}")

if __name__ == '__main__':
    main()
