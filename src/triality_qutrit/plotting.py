from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

    if out and ax is None:
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

def plot_tomo_panel(tomo_results: dict, out: str):
    """Creates a bar chart comparing tomography results."""
    labels = ['|g⟩', '|e⟩', '|f⟩', '|h⟩']
    peak_pops = [tomo_results['peak']['pop_g'], tomo_results['peak']['pop_e'], tomo_results['peak']['pop_f'], tomo_results['peak']['pop_h']]
    control_pops = [tomo_results['control']['pop_g'], tomo_results['control']['pop_e'], tomo_results['control']['pop_f'], tomo_results['control']['pop_h']]

    x = np.arange(len(labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    rects1 = ax1.bar(x - width/2, peak_pops, width, label=f"Hotspot Peak (Fid: {tomo_results['peak']['fidelity']:.2f})")
    rects2 = ax1.bar(x + width/2, control_pops, width, label=f"Control Region (Fid: {tomo_results['control']['fidelity']:.2f})")
    ax1.set_ylabel('Population')
    ax1.set_title('Qutrit State Populations')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.set_ylim(0, 1)

    metrics = ['Fidelity', 'Purity']
    peak_vals = [tomo_results['peak']['fidelity'], tomo_results['peak']['purity']]
    control_vals = [tomo_results['control']['fidelity'], tomo_results['control']['purity']]
    
    x_metrics = np.arange(len(metrics))
    ax2.bar(x_metrics - width/2, peak_vals, width, label='Hotspot Peak')
    ax2.bar(x_metrics + width/2, control_vals, width, label='Control Region')
    ax2.set_ylabel('Value')
    ax2.set_title('State Fidelity and Purity')
    ax2.set_xticks(x_metrics)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    ax2.set_ylim(0, 1.05)

    fig.suptitle('Tomography Analysis')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out, dpi=150, bbox_inches='tight')
    return fig

def plot_asymmetry_axis(df: pd.DataFrame, out: str):
    """Plots T2* vs the asymmetry parameter A."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Bin the data for a clearer plot
    bins = np.linspace(df['asymmetry_A'].min(), df['asymmetry_A'].max(), 30)
    binned = df.groupby(pd.cut(df['asymmetry_A'], bins=bins))
    
    mean = binned['T2star_ns'].mean()
    std = binned['T2star_ns'].std()
    
    ax.errorbar(mean.index.map(lambda x: x.mid), mean.values, yerr=std.values, fmt='o-', capsize=5)
    
    ax.axvline(0, color='r', linestyle='--', label='Symmetric (A=0)')
    ax.set_xlabel('Asymmetry Parameter A [dB]')
    ax.set_ylabel('Mean T2* [ns]')
    ax.set_title('Coherence vs. Drive Asymmetry')
    ax.legend()
    ax.grid(True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    return fig