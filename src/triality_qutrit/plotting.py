from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt

def heatmap(df: pd.DataFrame, value='T2star_ns', x='P_ge_dBm', y='P_ef_dBm', title='Coherence Map', out=None, ax=None):
    piv = df.pivot_table(index=y, columns=x, values=value, aggfunc='mean')
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.get_figure()

    im = ax.imshow(piv.values, origin='lower', aspect='auto',
                   extent=[piv.columns.min(), piv.columns.max(), piv.index.min(), piv.index.max()])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value)

    if out and ax is None: # Save only if it's a single plot
        fig.savefig(out, dpi=150, bbox_inches='tight')
    return fig, ax

def side_by_side_heatmap(df1: pd.DataFrame, df2: pd.DataFrame, out: str, title1='Hotspot', title2='Control'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    heatmap(df1, title=title1, ax=ax1)
    heatmap(df2, title=title2, ax=ax2)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    return fig

def plot_detuning_sweep(sweep_results: pd.DataFrame, out: str):
    """Plots the results of a detuning sweep."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sweep_results['detuning_MHz'], sweep_results['peak_T2star_ns'], 'o-')
    ax.set_xlabel('Pump Detuning [MHz]')
    ax.set_ylabel('Peak T2* [ns]')
    ax.set_title('Hotspot Coherence vs. Pump Detuning')
    ax.grid(True, linestyle=':')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    return fig