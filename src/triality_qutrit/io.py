from __future__ import annotations
import json
import pandas as pd
from pathlib import Path

def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)

def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def load_metadata(path: str | Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
