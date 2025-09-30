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
