import numpy as np
import pandas as pd


def nan_audit(df: pd.DataFrame, label: str) -> pd.DataFrame:
    n = len(df)
    audit = df.isnull().sum().rename('null_count').to_frame()
    audit['null_pct'] = (audit['null_count'] / n * 100).round(2)
    audit.index.name = 'column'
    audit = audit[audit['null_count'] > 0].sort_values('null_pct', ascending=False)
    print(f'\n--- NaN audit: {label} ({n} rows) ---')
    print(audit.to_string())
    return audit


def vif_from_corr_matrix(corr_matrix: pd.DataFrame) -> dict:
    """VIF_i = diagonal of (C^-1) where C is the correlation matrix."""
    cols = corr_matrix.columns.tolist()
    try:
        C_inv = np.linalg.inv(corr_matrix.values.astype(float))
        return {col: round(float(C_inv[i, i]), 2) for i, col in enumerate(cols)}
    except np.linalg.LinAlgError:
        return {col: np.nan for col in cols}
