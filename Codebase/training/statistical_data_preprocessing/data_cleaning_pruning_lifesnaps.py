import os
import pandas as pd
import numpy as np
from utils import nan_audit, vif_from_corr_matrix

BASE = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'datasets')
LIFESNAPS_PATH = os.path.join(BASE, 'data_loaded_cleaned', 'lifesnaps_self', 'lifesnaps_final.csv')


def load_lifesnaps() -> pd.DataFrame:
    if not os.path.exists(LIFESNAPS_PATH):
        raise FileNotFoundError(f'LifeSnaps file not found: {LIFESNAPS_PATH}')
    return pd.read_csv(LIFESNAPS_PATH, low_memory=False)



# ---------------------------------------------------------------------------
# LifeSnaps — cleaning
# ---------------------------------------------------------------------------

# BMI has three categorical bracket strings alongside numeric values.
# Mapped to boundary values that fall in the correct clinical discretization bin:
#   '<19'  → 18.0  (< 18.5 underweight threshold)
#   '>=25' → 25.0  (overweight range start; '>=30' is a separate category so this means 25–29.9)
#   '>=30' → 30.0  (obese threshold)
_BMI_CAT_MAP = {'<19': 18.0, '>=25': 25.0, '>=30': 30.0}


def clean_lifesnaps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Physiological range clamp
    df.loc[(df['hourly_bpm'] < 40) | (df['hourly_bpm'] > 180), 'hourly_bpm'] = np.nan

    # BMI: replace categorical brackets then convert to float
    df['bmi'] = pd.to_numeric(df['bmi'].replace(_BMI_CAT_MAP), errors='coerce')

    # age '<30' / '>=30' → binary 0 / 1
    df['age'] = (df['age'] == '>=30').astype(float)

    # Normalise column names: lowercase, replace '/' and spaces with '_'
    df.columns = [c.replace('/', '_').replace(' ', '_').lower() for c in df.columns]

    nan_audit(df, 'LifeSnaps post-clean')
    return df


# ---------------------------------------------------------------------------
# LifeSnaps — redundancy analysis
# Node groups include only columns being considered for retention so the
# report surfaces within-node correlations and VIF before any aggregation.
# Columns dropped for data-quality reasons (null rate) are excluded — their
# drop justification is documented in prune_lifesnaps below.
# ---------------------------------------------------------------------------

_LS_NODE_GROUPS = {
    # daily_sedentary_ratio included to show r=1.0 with daily_active_ratio before drop
    'activity': [
        'daily_active_ratio', 'daily_sedentary_ratio',
    ],
    # hourly vs prev_day: same signal at different temporal resolutions
    'heart_rate': ['hourly_bpm', 'prev_day_bpm'],
    'steps':      ['hourly_steps', 'prev_day_steps'],
    # three remaining sleep physio cols after spo2 + temp_variation dropped
    'sleep_physio': [
        'prev_night_resting_hr', 'prev_night_temperature', 'prev_night_minutesasleep',
    ],
}

_CORR_FLAG  = 0.70
_CORR_PRUNE = 0.85



def lifesnaps_redundancy_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Print per-node VIF + high-r pairs. Returns a DataFrame with columns
    [node, column, vif, high_vif] for further inspection.
    """
    rows = []
    print(f'\n{"="*65}')
    print('LifeSnaps Node Redundancy Report')
    print(f'Flag: r > {_CORR_FLAG}  |  Prune candidate: r > {_CORR_PRUNE}  |  High VIF: > 5')
    print(f'{"="*65}')
    for node, cols in _LS_NODE_GROUPS.items():
        present = [c for c in cols if c in df.columns]
        if len(present) < 2:
            continue
        sub  = df[present].apply(pd.to_numeric, errors='coerce')
        corr = sub.corr().abs()
        vifs = vif_from_corr_matrix(corr)
        print(f'\n[{node}]  ({len(present)} cols)')
        for col, v in vifs.items():
            flag = '  *** HIGH' if isinstance(v, float) and v > 5 else ''
            print(f'    {col:<42}  VIF={v}{flag}')
            rows.append({'node': node, 'column': col, 'vif': v,
                         'high_vif': bool(isinstance(v, float) and v > 5)})
        for i, c1 in enumerate(present):
            for c2 in present[i + 1:]:
                r = corr.loc[c1, c2]
                if r > _CORR_FLAG:
                    tag = '[PRUNE CANDIDATE]' if r > _CORR_PRUNE else '[FLAG]'
                    print(f'    {c1} ~ {c2}:  r={r:.3f}  {tag}')
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

_LS_MOOD_COLS = [
    'today_alert', 'today_happy', 'today_neutral', 'today_rested_relaxed',
    'today_sad', 'today_tense_anxious', 'today_tired',
    'prev_day_alert', 'prev_day_happy', 'prev_day_neutral', 'prev_day_rested_relaxed',
    'prev_day_sad', 'prev_day_tense_anxious', 'prev_day_tired',
]


def prune_lifesnaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 1 — Redundancy report (correlation + VIF on retained columns).

    Step 2 — Drops:

      Correlation:
        daily_sedentary_ratio       r=1.0 with daily_active_ratio

      Data quality (null rate):
        prev_day_scl_avg            97.9% null
        prev_night_spo2             81.0% null
        prev_day_stress_score       72.1% null
        mood flags (14 cols)        66.2% null + self-reported, not sensor
        prev_night_temp_variation   50.4% null + redundant with prev_night_temperature
    """
    lifesnaps_redundancy_report(df)

    drops = [
        'daily_sedentary_ratio',
        'prev_day_scl_avg',
        'prev_night_spo2',
        'prev_day_stress_score',
        'prev_night_temp_variation',
    ] + _LS_MOOD_COLS

    df = df.drop(columns=[c for c in drops if c in df.columns])
    nan_audit(df, 'LifeSnaps post-prune')
    return df


if __name__ == '__main__':
    print('Loading LifeSnaps data...')
    _df = clean_lifesnaps(load_lifesnaps())
    print(f'Loaded {len(_df)} rows, {len(_df.columns)} columns\n')
    vif_df = lifesnaps_redundancy_report(_df)
    print('\n--- VIF summary ---')
    print(vif_df.to_string(index=False))
