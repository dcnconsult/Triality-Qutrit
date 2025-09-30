from __future__ import annotations
import argparse
from pathlib import Path
from .io import load_csv, save_csv
from .plotting import heatmap
from .stats import spatial_signal_to_noise, hotspot_peak

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='CSV path with sweep data')
    ap.add_argument('--out', default='out/', help='Output directory')
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    df = load_csv(args.data)
    heatmap(df, out=outdir/'heatmap_T2star.png')
    stats = spatial_signal_to_noise(df)
    peak = hotspot_peak(df)
    with open(outdir/'summary.txt', 'w', encoding='utf-8') as f:
        f.write(f"SNR_Z: {stats['snr_z']:.3f}\nP-value (shuffle): {stats['p_value']:.4f}\n")
        f.write(f"Peak T2*: {peak['peak_value']:.2f} ns at (P_ge={peak['peak_P_ge']:.1f} dBm, P_ef={peak['peak_P_ef']:.1f} dBm)\n")
    print('Analysis complete. See', outdir)

if __name__ == '__main__':
    main()
