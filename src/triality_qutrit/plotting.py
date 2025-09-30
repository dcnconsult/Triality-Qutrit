from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt

def heatmap(df: pd.DataFrame, value='T2star_ns', x='P_ge_dBm', y='P_ef_dBm', title='Coherence Map', out=None):
    piv = df.pivot_table(index=y, columns=x, values=value, aggfunc='mean')
    fig = plt.figure(figsize=(7,6))
    plt.imshow(piv.values, origin='lower', aspect='auto',
               extent=[piv.columns.min(), piv.columns.max(), piv.index.min(), piv.index.max()])
    plt.xlabel(x); plt.ylabel(y); plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label(value)
    if out:
        fig.savefig(out, dpi=150, bbox_inches='tight')
    return fig
