"""
Step 7: Build final training CSV.

Concatenates StudentLife + LifeSnaps + NHANES discretized node-state CSVs.

NHANES is cross-sectional (no time-series), so its rows have NaN for the
structural columns (user_id, date, hour).  They contribute to intra-slice
structure learning in the Structural EM loop but do NOT form consecutive
hourly pairs for DBN inter-slice fitting — that is handled automatically
by train_dbn.py which only pairs rows sharing the same user_id+date.

Without NHANES, Cat2 nodes (age, sex, education_level, marital_status,
general_health, diabetes_status, chronic_condition, alcohol_use, smoking)
are 100 % NaN and get excluded by the null-rate filter in train_dbn.py.

Output:
  datasets/final_dataset/training_final.csv
"""

import os
import pandas as pd

_ROOT   = os.path.join(os.path.dirname(__file__), '..', '..', '..')
_PROC   = os.path.join(_ROOT, 'datasets', 'data_preprocessed')
_OUTDIR = os.path.join(_ROOT, 'datasets', 'final_dataset')


def build_training_set() -> pd.DataFrame:
    sl = pd.read_csv(os.path.join(_PROC, 'studentlife_discretized.csv'))
    ls = pd.read_csv(os.path.join(_PROC, 'lifesnaps_discretized.csv'))
    nh = pd.read_csv(os.path.join(_PROC, 'nhanes_discretized.csv'))

    # NHANES has no structural columns — add them as NaN so concat aligns
    for col in ('user_id', 'date', 'hour'):
        nh[col] = None
    nh['dataset'] = 'nhanes'

    df = pd.concat([sl, ls, nh], ignore_index=True)

    os.makedirs(_OUTDIR, exist_ok=True)
    out_path = os.path.join(_OUTDIR, 'training_final.csv')
    df.to_csv(out_path, index=False)

    print(f'Training set: {len(df)} rows, {len(df.columns)} cols')
    print(f'  StudentLife: {len(sl)} rows  |  LifeSnaps: {len(ls)} rows  |  NHANES: {len(nh)} rows')
    print(f'Saved -> {out_path}')
    return df


if __name__ == '__main__':
    build_training_set()
