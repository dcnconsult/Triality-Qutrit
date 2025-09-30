from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from scipy.ndimage import gaussian_filter
from scipy.stats import ttest_ind

def spatial_signal_to_noise(df: pd.DataFrame, value='T2star_ns', x='P_ge_dBm', y='P_ef_dBm', sigma_px=1.0, n_shuffle=500, random_state=0):
    piv = df.pivot_table(index=y, columns=x, values=value, aggfunc='mean')
    arr = piv.values
    sm = gaussian_filter(arr, sigma=sigma_px)
    s_real = sm.std()

    rng = np.random.default_rng(random_state)
    null = []
    flat = arr.flatten()
    for _ in range(n_shuffle):
        rng.shuffle(flat)
        arr_shuff = flat.reshape(arr.shape)
        sm_shuff = gaussian_filter(arr_shuff, sigma=sigma_px)
        null.append(sm_shuff.std())
    null = np.array(null)
    z = (s_real - null.mean()) / (null.std() + 1e-9)
    p = (null >= s_real).mean()
    return {'snr_z': float(z), 'p_value': float(p), 's_real': float(s_real), 's_null_mean': float(null.mean())}

def hotspot_peak(df: pd.DataFrame, value='T2star_ns', x='P_ge_dBm', y='P_ef_dBm'):
    piv = df.pivot_table(index=y, columns=x, values=value, aggfunc='mean')
    idx = np.unravel_index(np.argmax(piv.values), piv.shape)
    peak_val = float(piv.values[idx])
    peak_x = float(piv.columns[idx[1]])
    peak_y = float(list(piv.index)[idx[0]])
    return {'peak_value': peak_val, 'peak_P_ge': peak_x, 'peak_P_ef': peak_y}


# --- v3 additions: bootstrap and symmetry metrics ---
def _pivot(df: pd.DataFrame, value='T2star_ns', x='P_ge_dBm', y='P_ef_dBm'):
    piv = df.pivot_table(index=y, columns=x, values=value, aggfunc='mean')
    return piv.sort_index(), np.array(sorted(df[x].unique())), np.array(sorted(df[y].unique()))

def symmetry_distance(df: pd.DataFrame, value='T2star_ns', x='P_ge_dBm', y='P_ef_dBm'):
    """Return distance of peak to the symmetry line x=y, in dB, along with coords."""
    piv, xs, ys = _pivot(df, value=value, x=x, y=y)
    arr = piv.values
    i,j = np.unravel_index(np.nanargmax(arr), arr.shape)
    peak_y = float(piv.index[i])
    peak_x = float(piv.columns[j])
    # Signed distance to x=y in (x,y) plane (Euclidean in dB space)
    d = (peak_x - peak_y)/np.sqrt(2.0)
    return {'peak_P_ge': peak_x, 'peak_P_ef': peak_y, 'sym_dist_db': float(d)}

def bootstrap_hotspot(df: pd.DataFrame, value='T2star_ns', x='P_ge_dBm', y='P_ef_dBm', n_boot=500, frac=0.8, random_state=0):
    """Bootstrap stability of hotspot location and value.
    Returns:
      - peak_samples.csv rows with peak (x,y,val) per replicate
      - summary dict with mean/std and concentration in weak-weak quadrant
    """
    rng = np.random.default_rng(random_state)
    rows = []
    piv_full = df.pivot_table(index=y, columns=x, values=value, aggfunc='mean')
    xvals = np.array(sorted(df[x].unique()))
    yvals = np.array(sorted(df[y].unique()))
    x_med = np.median(xvals)
    y_med = np.median(yvals)
    for b in range(n_boot):
        samp = df.sample(frac=frac, replace=True, random_state=rng.integers(0,1<<32))
        piv = samp.pivot_table(index=y, columns=x, values=value, aggfunc='mean')
        arr = piv.values
        i,j = np.unravel_index(np.nanargmax(arr), arr.shape)
        rows.append({
            'boot': b,
            'peak_P_ge': float(piv.columns[j]),
            'peak_P_ef': float(list(piv.index)[i]),
            'peak_value': float(arr[i,j]),
            'weakweak': bool(piv.columns[j] < x_med and list(piv.index)[i] < y_med)
        })
    boot_df = pd.DataFrame(rows)
    conc = float(boot_df['weakweak'].mean())
    mu_x, mu_y = float(boot_df['peak_P_ge'].mean()), float(boot_df['peak_P_ef'].mean())
    sd_x, sd_y = float(boot_df['peak_P_ge'].std(ddof=1)), float(boot_df['peak_P_ef'].std(ddof=1))
    mu_v, sd_v = float(boot_df['peak_value'].mean()), float(boot_df['peak_value'].std(ddof=1))
    return boot_df, {'concentration_weakweak': conc, 'mu_P_ge': mu_x, 'sd_P_ge': sd_x, 'mu_P_ef': mu_y, 'sd_P_ef': sd_y, 'mu_peak': mu_v, 'sd_peak': sd_v}
