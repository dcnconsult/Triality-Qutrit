from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .io import load_csv
from .plotting import heatmap, side_by_side_heatmap, plot_detuning_sweep
from .stats import spatial_signal_to_noise, hotspot_peak, symmetry_distance, bootstrap_hotspot, symmetry_line_p_test
from .simulate import synthesize

def run_analysis(df: pd.DataFrame, outdir: Path, n_boot=500, n_perm=1000):
    """Runs the full analysis suite on a given dataframe and saves the outputs."""
    outdir.mkdir(parents=True, exist_ok=True)
    heatmap(df, out=outdir / 'heatmap_T2star.png')
    stats = spatial_signal_to_noise(df)
    peak = hotspot_peak(df)
    sym = symmetry_distance(df)

    boot_df, boot_sum = bootstrap_hotspot(df, n_boot=n_boot, frac=0.8)
    boot_df.to_csv(outdir/'bootstrap_peaks.csv', index=False)
    sym_p, d_null = symmetry_line_p_test(df, n_perm=n_perm)

    _np = np
    _np.savetxt(outdir/'symmetry_null_dist.csv', d_null, delimiter=',', header='sym_dist_db', comments='')

    summary = {
        "SNR_Z": f"{stats['snr_z']:.3f}",
        "P-value (shuffle)": f"{stats['p_value']:.4f}",
        "Peak T2*": f"{peak['peak_value']:.2f} ns at (P_ge={peak['peak_P_ge']:.1f} dBm, P_ef={peak['peak_P_ef']:.1f} dBm)",
        "Symmetry-line distance (dB)": f"{sym['sym_dist_db']:.3f}",
        "Bootstrap weak-weak concentration": f"{boot_sum['concentration_weakweak']:.3f}",
        "Bootstrap mean peak": f"(P_ge={boot_sum['mu_P_ge']:.2f}±{boot_sum['sd_P_ge']:.2f}, P_ef={boot_sum['mu_P_ef']:.2f}±{boot_sum['sd_P_ef']:.2f}) dBm",
        "Bootstrap mean peak value": f"{boot_sum['mu_peak']:.2f}±{boot_sum['sd_peak']:.2f} ns",
        "Symmetry-line p-value": f"{sym_p['p_value']:.4g} (d_obs={sym_p['d_obs']:.3f}, null μ={sym_p['d_null_mean']:.3f}±{sym_p['d_null_std']:.3f}, n={sym_p['n_perm']})"
    }
    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', help='CSV path with sweep data')
    ap.add_argument('--out', default='out/', help='Output directory')
    ap.add_argument('--run-phase-control', action='store_true', help='Run simulation with phase randomization control.')
    ap.add_argument('--run-detuning-sweep', action='store_true', help='Run simulation with a pump detuning sweep.')
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.run_phase_control:
        # Generate and analyze datasets for phase control
        df_hotspot = synthesize(outdir / 'triad_sweep_hotspot.csv', randomize_phases=False)
        df_control = synthesize(outdir / 'triad_sweep_control.csv', randomize_phases=True)
        side_by_side_heatmap(df_hotspot, df_control, out=outdir / 'Gate4_PhaseRandomization.png',
                             title1='Coherent Phases (Hotspot)', title2='Random Phases (Control)')
        summary_hotspot = run_analysis(df_hotspot, outdir / 'hotspot_analysis', n_boot=50, n_perm=100)
        summary_control = run_analysis(df_control, outdir / 'control_analysis', n_boot=50, n_perm=100)
        
        with open(outdir / 'summary.txt', 'w', encoding='utf-8') as f:
            f.write("--- ANALYSIS OF COHERENT HOTSPOT RUN ---\n")
            for key, val in summary_hotspot.items(): f.write(f"{key}: {val}\n")
            f.write("\n--- ANALYSIS OF PHASE-RANDOMIZED CONTROL RUN ---\n")
            for key, val in summary_control.items(): f.write(f"{key}: {val}\n")

    elif args.run_detuning_sweep:
        # Perform the detuning sweep
        detunings = np.linspace(-10, 10, 21)
        results = []
        for detuning in detunings:
            print(f"Simulating for detuning: {detuning:.1f} MHz")
            df = synthesize(outdir / f'detuning_{detuning:.1f}MHz.csv', pump_detuning_MHz=detuning)
            peak = hotspot_peak(df)
            results.append({'detuning_MHz': detuning, 'peak_T2star_ns': peak['peak_value']})
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(outdir / 'detuning_sweep_results.csv', index=False)
        plot_detuning_sweep(results_df, out=outdir / 'detuning_response.png')

    else:
        # Standard analysis on a single file
        if not args.data:
            raise ValueError("--data argument is required for standard analysis.")
        df = load_csv(args.data)
        summary = run_analysis(df, outdir / 'analysis')
        with open(outdir / 'summary.txt', 'w', encoding='utf-8') as f:
            for key, val in summary.items(): f.write(f"{key}: {val}\n")

    print('Analysis complete. See', outdir)

if __name__ == '__main__':
    main()