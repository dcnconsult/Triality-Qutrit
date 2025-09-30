from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

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

def plot_detuning_sweep(sweep_results: pd.DataFrame, out=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    ax.plot(sweep_results['detuning_MHz'], sweep_results['peak_T2star_ns'], 'o-')
    ax.set_xlabel('Pump Detuning [MHz]')
    ax.set_ylabel('Peak T2* [ns]')
    ax.set_title('Hotspot vs. Pump Detuning')
    ax.grid(True, linestyle=':')
    if out and ax is None:
        fig.savefig(out, dpi=150, bbox_inches='tight')
    return fig, ax

def plot_tomo_panel(tomo_results: dict, out: str):
    # ... (code remains the same)
    pass

def plot_asymmetry_axis(df: pd.DataFrame, out=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()
    
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
    if out and ax is None:
        fig.savefig(out, dpi=150, bbox_inches='tight')
    return fig, ax

def create_one_page_summary(summary_data: dict, df: pd.DataFrame, detuning_df: pd.DataFrame, bicoherence_img_path: str, out_pdf: str):
    """Creates a one-page summary PDF of the entire analysis."""
    fig = plt.figure(figsize=(16, 20))
    gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[1, 0.7, 0.7, 1.2])

    # A) Main Heatmap
    ax_heatmap = fig.add_subplot(gs[0, 0])
    heatmap(df, ax=ax_heatmap, title="A) Coherence Hotspot Map")

    # B) Detuning Response
    ax_detuning = fig.add_subplot(gs[0, 1])
    plot_detuning_sweep(detuning_df, ax=ax_detuning)

    # C) Asymmetry Axis Plot
    from .stats_extras import calculate_asymmetry
    df_asym = calculate_asymmetry(df.copy())
    ax_asymmetry = fig.add_subplot(gs[1, 0])
    plot_asymmetry_axis(df_asym, ax=ax_asymmetry)

    # D) Bicoherence Plot
    ax_bicoherence = fig.add_subplot(gs[1, 1])
    try:
        img = plt.imread(bicoherence_img_path)
        ax_bicoherence.imshow(img)
        ax_bicoherence.set_title("D) Bicoherence at Hotspot Peak")
        ax_bicoherence.axis('off')
    except FileNotFoundError:
        ax_bicoherence.text(0.5, 0.5, 'Bicoherence plot not found.', ha='center', va='center')
        ax_bicoherence.set_title("D) Bicoherence Analysis")


    # E) Summary Text
    ax_text = fig.add_subplot(gs[2, :])
    ax_text.axis('off')
    text_content = "E) Key Statistical Findings:\n\n"
    for key, val in summary_data.items():
        text_content += f"- {key}: {val}\n"
    
    ax_text.text(0.02, 0.95, text_content, ha='left', va='top', fontsize=12, family='monospace')

    fig.tight_layout(pad=3.0)
    fig.suptitle("Triality Qutrit Hotspot Analysis Summary", fontsize=24, y=0.98)
    fig.savefig(out_pdf, format='pdf')
    return fig