from __future__ import annotations
import pandas as pd
import numpy as np

def calculate_asymmetry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the asymmetry parameter A = 10 * log10(P_ge / P_ef)
    and adds it as a column to the dataframe.
    """
    # Power in linear scale is 10^(P_dBm / 10)
    p_ge_linear = 10**(df['P_ge_dBm'] / 10)
    p_ef_linear = 10**(df['P_ef_dBm'] / 10)
    
    # Avoid division by zero, though powers are typically negative dBm (so > 0 linear)
    ratio = p_ge_linear / (p_ef_linear + 1e-12)
    
    df['asymmetry_A'] = 10 * np.log10(ratio)
    return df