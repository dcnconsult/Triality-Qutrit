from __future__ import annotations
import numpy as np
import pandas as pd

def probs_from_counts(df: pd.DataFrame, outcome_col="outcome", count_col="count", group_col="setting_id"):
    grp = df.groupby([group_col, outcome_col])[count_col].sum().reset_index()
    grp["prob"] = grp.groupby(group_col)[count_col].transform(lambda s: s/s.sum())
    return grp

def ghz_fidelity_proxy(probs: pd.DataFrame, outcome_col="outcome", prob_col="prob", group_col="setting_id"):
    res = []
    for sid, g in probs.groupby(group_col):
        d = dict(zip(g[outcome_col].astype(str), g[prob_col]))
        f = 0.5*(d.get("000",0.0)+d.get("222",0.0))
        res.append({"setting_id": sid, "f_proxy": f})
    return pd.DataFrame(res)

def bootstrap_fidelity(probs: pd.DataFrame, shots_per_setting: int, n_boot=1000, random_state=0,
                       outcome_col="outcome", prob_col="prob", group_col="setting_id"):
    rng = np.random.default_rng(random_state)
    settings = sorted(probs[group_col].unique())
    out_vals = sorted(probs[outcome_col].astype(str).unique())
    P = {sid: probs[probs[group_col]==sid].set_index(outcome_col)[prob_col].reindex(out_vals).fillna(0.0).values
         for sid in settings}
    boots = []
    for sid in settings:
        p = P[sid]
        for _ in range(n_boot):
            counts = rng.multinomial(shots_per_setting, p)
            emp = counts / counts.sum()
            idx0 = out_vals.index("000") if "000" in out_vals else None
            idx2 = out_vals.index("222") if "222" in out_vals else None
            f = np.nan
            if idx0 is not None and idx2 is not None:
                f = 0.5*(emp[idx0] + emp[idx2])
            boots.append({"setting_id":sid, "f_proxy": f})
    bdf = pd.DataFrame(boots)
    ci = bdf.groupby("setting_id")["f_proxy"].quantile([0.025,0.5,0.975]).unstack()
    ci.columns = ["ci_lo","median","ci_hi"]
    return bdf, ci.reset_index()

def leakage_estimate(probs: pd.DataFrame, outcome_col="outcome", prob_col="prob", group_col="setting_id"):
    return probs.groupby(group_col).apply(lambda g: 0.0).reset_index(name="leakage")
