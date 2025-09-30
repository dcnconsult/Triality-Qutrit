from __future__ import annotations
import argparse
from pathlib import Path
from .io import load_csv, save_csv
from .plotting import heatmap
from .stats import spatial_signal_to_noise, hotspot_peak, symmetry_distance, bootstrap_hotspot, symmetry_line_p_test

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_boot', type=int, default=500)
    ap.add_argument('--boot_frac', type=float, default=0.8)
    ap.add_argument('--n_perm', type=int, default=1000)
    ap.add_argument('--data', required=True, help='CSV path with sweep data')
    ap.add_argument('--out', default='out/', help='Output directory')
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    df = load_csv(args.data)
    heatmap(df, out=outdir/'heatmap_T2star.png')
    stats = spatial_signal_to_noise(df)
    peak = hotspot_peak(df)
    sym = symmetry_distance(df)
    boot_df, boot_sum = bootstrap_hotspot(df, n_boot=args.n_boot, frac=args.boot_frac)
    boot_df.to_csv(outdir/'bootstrap_peaks.csv', index=False)
    sym_p, d_null = symmetry_line_p_test(df, n_perm=args.n_perm)
    import numpy as _np
    _np.savetxt(outdir/'symmetry_null_dist.csv', d_null, delimiter=',', header='sym_dist_db', comments='')

# Plots: null histogram with observed line, and empirical CDF
import matplotlib.pyplot as plt
import numpy as _np

# Histogram
fig1 = plt.figure(figsize=(7,6))
plt.hist(d_null, bins=40, density=True)
plt.axvline(sym_p['d_obs'], linestyle='--')
plt.xlabel('Symmetry-line distance |d| [dB]')
plt.ylabel('Density')
plt.title('Permutation Null: |d| histogram with observed')
fig1.savefig(outdir/'symmetry_null_hist.png', dpi=150, bbox_inches='tight')

# Empirical CDF
fig2 = plt.figure(figsize=(7,6))
xs = _np.sort(d_null)
ys = _np.arange(1, len(xs)+1)/len(xs)
plt.plot(xs, ys)
# Mark observed
obs = sym_p['d_obs']
# y at observed
y_obs = (xs <= obs).sum()/len(xs)
plt.axvline(obs, linestyle='--')
plt.axhline(y_obs, linestyle=':')
plt.xlabel('Symmetry-line distance |d| [dB]')
plt.ylabel('Empirical CDF')
plt.title('Permutation Null: CDF with observed marker')
fig2.savefig(outdir/'symmetry_null_cdf.png', dpi=150, bbox_inches='tight')

    with open(outdir/'summary.txt', 'w', encoding='utf-8') as f:
        f.write(f"SNR_Z: {stats['snr_z']:.3f}\nP-value (shuffle): {stats['p_value']:.4f}\n")
        f.write(f"Peak T2*: {peak['peak_value']:.2f} ns at (P_ge={peak['peak_P_ge']:.1f} dBm, P_ef={peak['peak_P_ef']:.1f} dBm)\n")
        f.write(f"Symmetry-line distance (dB): {sym['sym_dist_db']:.3f}\n")
        f.write(f"Bootstrap weak-weak concentration: {boot_sum['concentration_weakweak']:.3f}\n")
        f.write(f"Bootstrap mean peak @ (P_ge={boot_sum['mu_P_ge']:.2f}±{boot_sum['sd_P_ge']:.2f}, P_ef={boot_sum['mu_P_ef']:.2f}±{boot_sum['sd_P_ef']:.2f}) dBm\n")
        f.write(f"Bootstrap mean peak value: {boot_sum['mu_peak']:.2f}±{boot_sum['sd_peak']:.2f} ns\n")
        f.write(f"Symmetry-line permutation p-value: {sym_p['p_value']:.4g} (d_obs={sym_p['d_obs']:.3f}, null μ={sym_p['d_null_mean']:.3f}±{sym_p['d_null_std']:.3f}, n={sym_p['n_perm']})\n")
    print('Analysis complete. See', outdir)

if __name__ == '__main__':
    main()
