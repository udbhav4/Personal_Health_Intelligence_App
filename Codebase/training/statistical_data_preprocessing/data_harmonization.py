import pandas as pd
import numpy as np


def _hour_to_time_of_day(hour_series: pd.Series) -> pd.Series:
    # night: 8pm–midnight + midnight–4am (non-contiguous, needs np.select)
    h = hour_series
    return pd.Series(
        np.select(
            [(h < 4) | (h >= 20), (h >= 4) & (h < 12), (h >= 12) & (h < 17), (h >= 17) & (h < 20)],
            ['night', 'morning', 'afternoon', 'evening'],
            default=None,
        ),
        index=hour_series.index,
    )


# ---------------------------------------------------------------------------
# StudentLife
# ---------------------------------------------------------------------------

def harmonize_studentlife(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.rename(columns={'uid': 'user_id'})
    df['dataset'] = 'studentlife'
    if 'hour' in df.columns:
        df['time_of_day'] = _hour_to_time_of_day(df['hour'])
    return df


# ---------------------------------------------------------------------------
# LifeSnaps
# ---------------------------------------------------------------------------

def harmonize_lifesnaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename/recode LifeSnaps columns to harmonized node column names:

      id                      → user_id
      gender (MALE/FEMALE)    → sex (male/female)
      daily_active_ratio      → active_ratio
      prev_night_minutesasleep → sleep_hours  (÷60 unit conversion)
      age                     → DROPPED (binary 0/1 too coarse for 4-state
                                clinical discretization; source_columns.lifesnaps=[]
                                for age node in config)
    """
    df = df.copy()

    df = df.rename(columns={
        'id':                   'user_id',
        'gender':               'sex',
        'daily_active_ratio':   'active_ratio',
    })

    df['sex'] = df['sex'].str.lower()

    if 'prev_night_minutesasleep' in df.columns:
        df['sleep_hours'] = df['prev_night_minutesasleep'] / 60.0
        df = df.drop(columns=['prev_night_minutesasleep'])

    df = df.drop(columns=['age'], errors='ignore')

    df['dataset'] = 'lifesnaps'
    if 'hour' in df.columns:
        df['time_of_day'] = _hour_to_time_of_day(df['hour'])
    return df


# ---------------------------------------------------------------------------
# NHANES
# ---------------------------------------------------------------------------

def harmonize_nhanes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename NHANES columns to harmonized node column names.

    Single-source nodes get direct renames.
    Multi-source nodes (CDQ001/CDQ010, ALQ121/130/142, PAD/PAQ cols) keep
    their raw names — discretization handles the merge/k-means step.
    phq_total, diabetes_status, chronic_condition already computed by prune_nhanes.
    """
    df = df.copy()

    df = df.rename(columns={
        'RIDAGEYR':  'age',
        'RIAGENDR':  'sex',
        'BMI':       'bmi',
        'EDUCATION': 'education_level',
        'DMDMARTL':  'marital_status',
        'HSD010':    'general_health',
        'SMQ040':    'smoking',
    })

    df['sex'] = df['sex'].map({1: 'male', 2: 'female'})

    # PAD680 = sedentary min/day → invert to active_ratio (0–1) so it pools
    # with StudentLife/LifeSnaps active_ratio for k-means discretization.
    if 'PAD680' in df.columns:
        df['active_ratio'] = (1 - df['PAD680'] / 1440).clip(0, 1)
        df = df.drop(columns=['PAD680'])

    return df
