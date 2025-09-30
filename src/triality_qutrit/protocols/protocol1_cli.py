from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from .spam import probs_from_counts, ghz_fidelity_proxy, bootstrap_fidelity, leakage_estimate

def main():
    ap = argparse.ArgumentParser(description="Run Protocol 1 (SPAM) on GHZ counts and produce summary artifacts.")
    ap.add_argument("--counts", required=True, help="Counts CSV (setting_id,outcome,count,shot_count)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--shots", type=int, default=4096, help="Shots per setting (for bootstrap)")
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.counts)
    probs = probs_from_counts(raw)
    probs.to_csv(outdir/'protocol1_probs.csv', index=False)

    fid = ghz_fidelity_proxy(probs)
    bdf, ci = bootstrap_fidelity(probs, shots_per_setting=args.shots, n_boot=args.n_boot)
    leak = leakage_estimate(probs)

    fid.to_csv(outdir/'protocol1_fidelity_proxy.csv', index=False)
    ci.to_csv(outdir/'protocol1_fidelity_bootstrap_ci.csv', index=False)
    leak.to_csv(outdir/'protocol1_leakage.csv', index=False)

    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        pdf_path = outdir/'protocol1_one_pager.pdf'
        with PdfPages(pdf_path) as pdf:
            fig = plt.figure(figsize=(8.5,11))
            fig.suptitle("Protocol 1 — SPAM Validation (GHZ, qutrits)", y=0.98)
            ax1 = fig.add_axes([0.08,0.75,0.84,0.18]); ax1.axis('off')
            s_lines = []
            if not ci.empty:
                row = ci.iloc[0]
                s_lines.append(f"Fidelity proxy (GHZ): median={row['median']:.3f} (95% CI [{row['ci_lo']:.3f}, {row['ci_hi']:.3f}])")
            s_lines.append(f"Bootstraps: n={args.n_boot}, shots/setting={args.shots}")
            ax1.text(0.0,1.0,"\n".join(s_lines), va='top')

            ax2 = fig.add_axes([0.08,0.52,0.84,0.18])
            ax2.bar(fid['setting_id'].astype(str), fid['f_proxy'])
            ax2.set_ylabel("Fidelity Proxy"); ax2.set_title("Fidelity by Setting")

            ax3 = fig.add_axes([0.08,0.30,0.84,0.18])
            x = np.arange(len(ci))
            ax3.errorbar(x, ci['median'], yerr=[ci['median']-ci['ci_lo'], ci['ci_hi']-ci['median']], fmt='o')
            ax3.set_xticks(x); ax3.set_xticklabels(ci['setting_id'].astype(str))
            ax3.set_ylabel("Fidelity (CI)"); ax3.set_title("Bootstrap CI (95%)")

            ax4 = fig.add_axes([0.08,0.08,0.84,0.18])
            ax4.bar(leak['setting_id'].astype(str), leak['leakage'])
            ax4.set_ylabel("Leakage"); ax4.set_title("Leakage per Setting")

            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
    except Exception as e:
        print("PDF composition failed:", e)

    print("Protocol 1 artifacts written to", outdir)

if __name__ == "__main__":
    main()
