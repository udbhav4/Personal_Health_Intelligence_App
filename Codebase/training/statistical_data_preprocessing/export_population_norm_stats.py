"""
Export population-level mean and std for sensor cols to
configs/population_norm_stats.json.

Must be run AFTER harmonize and BEFORE normalize_per_user in the pipeline.
Called automatically by main.py between harmonize and normalize steps.
Can also be run standalone:
    python export_population_norm_stats.py

Output format:
    {
        "col_name": { "mean": float, "std": float },
        ...
    }

Population std uses ddof=0 (not ddof=1): we are computing the full-dataset
parameter, not estimating from a sample.  The resulting stats are used as fixed
scalars at inference time, so population (not sample) std is correct here.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

# Allow imports from the same package directory when run as a standalone script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_cleaning_pruning_nhanes      import load_nhanes, clean_nhanes, prune_nhanes
from data_cleaning_pruning_studentlife import load_studentlife, clean_studentlife, prune_studentlife
from data_cleaning_pruning_lifesnaps   import load_lifesnaps, clean_lifesnaps, prune_lifesnaps
from data_harmonization                import harmonize_nhanes, harmonize_studentlife, harmonize_lifesnaps

_ROOT           = os.path.join(os.path.dirname(__file__), '..', '..', '..')
_OUT_PATH       = os.path.join(_ROOT, 'configs', 'population_norm_stats.json')

# Sensor cols for which population z-score replaces per-user z-score.
# Must stay in sync with _POPULATION_ZSCORE_COLS in data_normalization.py.
_SENSOR_COLS = [
    'screen_time_window_minutes',
    'dark_window_minutes',
    'unlocked_window_minutes',
    'yesterday_nighttime_screen_minutes',
    'prev_day_total_screen_time_minutes',
    'call_count',
    'call_duration_total',
    'sms_count',
    'prev_day_call_count',
    'prev_day_call_duration_total',
    'prev_day_sms_count',
    'avg_running_tasks_window',
    'prev_day_avg_running_tasks',
    'prev_day_peak_running_tasks',
    'hourly_steps',
    'prev_day_steps',
    'prev_night_resting_hr',
    'prev_night_temperature',
]


def _harmonize_all() -> pd.DataFrame:
    """
    Run load → clean → prune → harmonize for all three datasets.
    Returns concatenated harmonized DataFrame (before normalization).

    Replicates the exact call order used in main.py — do not reorder.
    """
    print('Loading and harmonizing NHANES...')
    df_nh = load_nhanes()
    df_nh = clean_nhanes(df_nh)
    df_nh = prune_nhanes(df_nh)
    df_nh = harmonize_nhanes(df_nh)

    print('Loading and harmonizing StudentLife...')
    df_sl = load_studentlife()
    df_sl = clean_studentlife(df_sl)
    df_sl = prune_studentlife(df_sl)
    df_sl = harmonize_studentlife(df_sl)

    print('Loading and harmonizing LifeSnaps...')
    df_ls = load_lifesnaps()
    df_ls = clean_lifesnaps(df_ls)
    df_ls = prune_lifesnaps(df_ls)
    df_ls = harmonize_lifesnaps(df_ls)

    # Concat with ignore_index so original dataset-level indices don't collide.
    combined = pd.concat([df_nh, df_sl, df_ls], ignore_index=True)
    print(f'Combined harmonized shape: {combined.shape}')
    return combined


def export_population_norm_stats(df: pd.DataFrame | None = None) -> dict:
    """
    Compute per-column population mean and std (ddof=0) across ALL users and
    ALL datasets for the sensor cols in _SENSOR_COLS, then write the result to
    configs/population_norm_stats.json.

    Parameters
    ----------
    df : pd.DataFrame or None
        Pre-harmonized DataFrame (before normalization).  If None, the full
        load → clean → prune → harmonize pipeline is run internally.

    Returns
    -------
    dict
        { "col_name": { "mean": float, "std": float }, ... }
    """
    if df is None:
        df = _harmonize_all()

    stats: dict = {}
    found:   list[str] = []
    skipped: list[str] = []

    for col in _SENSOR_COLS:
        if col not in df.columns:
            skipped.append(col)
            continue

        values = df[col].dropna()
        if len(values) == 0:
            print(f'  [WARN] {col}: all values are NaN — skipping')
            skipped.append(col)
            continue

        # ddof=0: population std (not sample std).  We are computing the
        # fixed scalar used at inference, not estimating a sample parameter.
        col_mean = float(np.mean(values))
        col_std  = float(np.std(values, ddof=0))

        stats[col] = {'mean': col_mean, 'std': col_std}
        found.append(col)

    # Write output
    os.makedirs(os.path.dirname(_OUT_PATH), exist_ok=True)
    with open(_OUT_PATH, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f'\n=== Population norm stats exported ===')
    print(f'  Output : {os.path.abspath(_OUT_PATH)}')
    print(f'  Found  : {len(found)} col(s): {found}')
    if skipped:
        print(f'  Skipped: {len(skipped)} col(s) not in DataFrame: {skipped}')

    return stats


if __name__ == '__main__':
    export_population_norm_stats()
