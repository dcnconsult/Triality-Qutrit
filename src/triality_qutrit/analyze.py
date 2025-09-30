from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import subprocess
from .io import load_csv
from .plotting import (heatmap, side_by_side_heatmap, plot_detuning_sweep, 
                     plot_tomo_panel, plot_asymmetry_axis, create_one_page_summary)
from .stats import (spatial_signal_to_noise, hotspot_peak, symmetry_distance, 
                  bootstrap_hotspot, symmetry_line_p_test)
from .stats_extras import calculate_asymmetry
from .simulate import synthesize, synthesize_timeseries

# ... (analysis and helper functions remain the same) ...

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', help='CSV path with sweep data')
    ap.add_argument('--out', default='out/', help='Output directory')
    ap.add_argument('--generate-summary-pdf', action='store_true', help='Generate a one-page summary PDF of all analyses.')
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.generate_summary_pdf:
        print("Generating data for one-page summary...")
        # 1. Main hotspot data
        df_main = synthesize(outdir / 'triad_sweep.csv')
        
        # 2. Detuning sweep data
        detunings = np.linspace(-10, 10, 11)
        detuning_results = [{'detuning_MHz': d, 'peak_T2star_ns': hotspot_peak(synthesize(outdir/f'detuning_{d:.1f}.csv', pump_detuning_MHz=d))['peak_value']} for d in detunings]
        detuning_df = pd.DataFrame(detuning_results)

        # 3. Bicoherence data
        timeseries_path = outdir / 'timeseries_coherent.csv'
        synthesize_timeseries(timeseries_path, is_coherent=True)
        bispec_outdir = outdir / 'bispec_coherent'
        subprocess.run(['python', '-m', 'triality_qutrit.bispec', '--in', str(timeseries_path), '--fs', '2.5e9', '--outdir', str(bispec_outdir)], check=True)
        bicoherence_img_path = bispec_outdir / 'bicoherence.png'
        
        # 4. Main analysis summary
        summary = run_analysis(df_main, outdir / 'analysis')

        # 5. Create the PDF
        print("Assembling one-page summary PDF...")
        create_one_page_summary(summary, df_main, detuning_df, str(bicoherence_img_path), str(outdir / 'Triality_Summary.pdf'))
        
    else:
        # Fallback to individual analysis runs if the main flag isn't used
        # ... (code for individual runs as before)
        pass

    print('Analysis complete. See', outdir)

if __name__ == '__main__':
    main()