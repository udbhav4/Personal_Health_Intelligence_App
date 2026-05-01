"""
Step 7: Build final training + validation CSVs.

Concatenates StudentLife + LifeSnaps + NHANES discretized node-state CSVs.
Splits:
  StudentLife + LifeSnaps — uid-level 80/20 (seed 42)
  NHANES                  — row-level 80/20 (seed 42); no uids available

20% of NHANES goes to val so Cat2 nodes (age, sex, education_level,
marital_status, general_health, diabetes_status, chronic_condition,
alcohol_use, smoking) can be validated via cross-sectional LBP inference
in validate_dbn.py.  Without this, Cat2 nodes are 100% NaN in val and
cannot be evaluated at all.

NHANES rows have NaN for structural columns (user_id, date, hour) so they
never form consecutive hourly pairs for DBN inter-slice fitting.

Outputs:
  datasets/final_dataset/training_final.csv
  datasets/final_dataset/validation_final.csv
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from utils import get_uid_split

_ROOT   = os.path.join(os.path.dirname(__file__), '..', '..', '..')
_PROC   = os.path.join(_ROOT, 'datasets', 'data_preprocessed')
_OUTDIR = os.path.join(_ROOT, 'datasets', 'final_dataset')


def build_training_set():
    sl = pd.read_csv(os.path.join(_PROC, 'studentlife_discretized.csv'))
    ls = pd.read_csv(os.path.join(_PROC, 'lifesnaps_discretized.csv'))
    nh = pd.read_csv(os.path.join(_PROC, 'nhanes_discretized.csv'))

    # NHANES has no structural columns — add them as NaN so concat aligns
    for col in ('user_id', 'date', 'hour'):
        nh[col] = None
    nh['dataset'] = 'nhanes'

    sl_train_uids, sl_val_uids = get_uid_split(sl, uid_col='user_id')
    ls_train_uids, ls_val_uids = get_uid_split(ls, uid_col='user_id')

    sl_train = sl[sl['user_id'].isin(sl_train_uids)]
    sl_val   = sl[sl['user_id'].isin(sl_val_uids)]
    ls_train = ls[ls['user_id'].isin(ls_train_uids)] if ls_train_uids else ls
    ls_val   = ls[ls['user_id'].isin(ls_val_uids)]   if ls_val_uids   else ls.iloc[0:0]

    # NHANES: row-level 80/20 (no uids)
    nh_shuf  = nh.sample(frac=1, random_state=42).reset_index(drop=True)
    n_nh_val = max(1, int(len(nh_shuf) * 0.2))
    nh_val   = nh_shuf.iloc[:n_nh_val]
    nh_train = nh_shuf.iloc[n_nh_val:]

    train_df = pd.concat([sl_train, ls_train, nh_train], ignore_index=True)
    val_df   = pd.concat([sl_val,   ls_val,   nh_val],   ignore_index=True)

    os.makedirs(_OUTDIR, exist_ok=True)
    train_path = os.path.join(_OUTDIR, 'training_final.csv')
    val_path   = os.path.join(_OUTDIR, 'validation_final.csv')
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f'Train set: {len(train_df)} rows, {len(train_df.columns)} cols')
    print(f'  SL train: {len(sl_train)} rows ({len(sl_train_uids)} uids)  '
          f'|  LS train: {len(ls_train)} rows ({len(ls_train_uids)} uids)  '
          f'|  NHANES train: {len(nh_train)} rows')
    print(f'Val set: {len(val_df)} rows')
    print(f'  SL val: {len(sl_val)} rows ({len(sl_val_uids)} uids)  '
          f'|  LS val: {len(ls_val)} rows ({len(ls_val_uids)} uids)  '
          f'|  NHANES val: {len(nh_val)} rows')
    print(f'Saved -> {train_path}')
    print(f'Saved -> {val_path}')
    return train_df, val_df


if __name__ == '__main__':
    build_training_set()
