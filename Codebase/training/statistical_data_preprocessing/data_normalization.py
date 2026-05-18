import json
import pathlib

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Population norm stats (Step 2 output)
# ---------------------------------------------------------------------------
_POP_STATS_PATH = (
    pathlib.Path(__file__).parent.parent.parent.parent / 'configs' / 'population_norm_stats.json'
)

# Sensor cols that use population-wide z-score instead of per-user z-score.
# Per-user z-score for these cols means a chronically high user always appears
# neutral to the DBN — chronic harm invisible to the model.
_POPULATION_ZSCORE_COLS = [
    'screen_time_window_minutes', 'dark_window_minutes', 'unlocked_window_minutes',
    'yesterday_nighttime_screen_minutes', 'prev_day_total_screen_time_minutes',
    'call_count', 'call_duration_total', 'sms_count',
    'prev_day_call_count', 'prev_day_call_duration_total', 'prev_day_sms_count',
    'avg_running_tasks_window', 'prev_day_avg_running_tasks', 'prev_day_peak_running_tasks',
    'hourly_steps', 'prev_day_steps',
    'prev_night_resting_hr', 'prev_night_temperature',
]


def _load_pop_stats() -> dict:
    """Load population norm stats from configs/population_norm_stats.json.

    Raises FileNotFoundError (with user-facing instructions) if not present.
    """
    if not _POP_STATS_PATH.exists():
        raise FileNotFoundError(
            f'Population norm stats not found at {_POP_STATS_PATH}.\n'
            'Run export_population_norm_stats.py first to generate this file:\n'
            '  python export_population_norm_stats.py'
        )
    return json.loads(_POP_STATS_PATH.read_text())


# ---------------------------------------------------------------------------
# Cols to z-score
# ---------------------------------------------------------------------------
# Sensor + daily-EMA cols with k-means discretization.
# Uses post-harmonization column names (active_ratio not daily_active_ratio,
# sleep_hours excluded — clinical threshold node).
#
# NOT normalized:
#   sleep_hours           — clinical threshold (<6 / 6–9 / >9 hrs)
#   phq_total             — clinical threshold (PHQ-9 sum)
#   pss_helplessness      — between-person survey scale
#   pss_self_efficacy     — between-person survey scale
#   panas_pa / panas_na   — between-person survey scale
#   lonely_total          — clinical threshold
#   neuroticism           — stable trait, between-person variance meaningful
#   extraversion          — stable trait, between-person variance meaningful
#   vr_physical_health    — monthly survey
#   vr_mental_health      — monthly survey
#   sleep_disturbances    — monthly PSQI
#   exercise_type/walk    — passthrough ordinal
#   bmi / sex / age       — clinical / categorical

_NORMALIZE_COLS = [
    # ── screen (StudentLife) ─────────────────────────────────────────────────
    'screen_time_window_minutes', 'dark_window_minutes', 'unlocked_window_minutes',
    'yesterday_nighttime_screen_minutes', 'prev_day_total_screen_time_minutes',
    # ── calls / sms (StudentLife) ────────────────────────────────────────────
    'call_count', 'call_duration_total', 'sms_count',
    'prev_day_call_count', 'prev_day_call_duration_total', 'prev_day_sms_count',
    # ── activity: active_ratio REMOVED — bounded 0–1, WHO norms exist,
    #   per-user z-score destroys absolute meaning; clinical threshold used instead
    # ── running tasks (StudentLife) ──────────────────────────────────────────
    'avg_running_tasks_window', 'prev_day_avg_running_tasks', 'prev_day_peak_running_tasks',
    # ── sleep supplementaries (StudentLife) ──────────────────────────────────
    # sleep_latency_mins excluded — clinical threshold (<20 / 20-45 / ≥45 min)
    # prev_day_sleep_ema_hours excluded — clinical threshold (<6 / 6-9 / ≥9 hrs)
    # sleep_quality_rating excluded — PSQI survey (once per semester period);
    #   every daily row in a period has the identical value → within-user std=0
    # prev_day_sleep_ema_rating REMOVED — bounded ordinal 1–5; clinical threshold used
    # prev_day_sleep_ema_alertness REMOVED — bounded ordinal 1–5; clinical threshold used
    # ── heart rate + steps (LifeSnaps) ───────────────────────────────────────
    # hourly_bpm / prev_day_bpm excluded — clinical thresholds on raw BPM values
    'hourly_steps', 'prev_day_steps',
    # ── physio sleep (LifeSnaps) ─────────────────────────────────────────────
    'prev_night_resting_hr', 'prev_night_temperature',
    # ── daily EMA cols REMOVED — bounded ordinal scales; clinical thresholds used ─
    # mood_how, mood_happy, mood_sad (1–5)
    # prev_day_mood_how, prev_day_mood_happy, prev_day_mood_sad (1–5)
    # productivity, prev_day_productivity (1–5)
    # stress_ema_level, prev_day_stress_ema_level (1–4)
    # prev_day_events_positive, prev_day_events_negative (0–N count)
    # active_ratio, prev_day_active_ratio (0.0–1.0 fraction)
]


def normalize_per_user(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize continuous sensor/EMA columns before discretization.

    Two regimes — determined by whether the column appears in
    _POPULATION_ZSCORE_COLS:

    1. Population z-score (sensor cols):
       z = (value - pop_mean) / pop_std  using scalars from
       population_norm_stats.json (computed once across all users by
       export_population_norm_stats.py).  A chronically high user then
       correctly registers as high rather than being collapsed to zero.
       pop_std == 0 or NaN → column left unscaled (fallback, logged).

    2. Per-user z-score (remaining cols in _NORMALIZE_COLS):
       z = (value - user_mean) / user_std  grouped by user_id.
       std == 0  (user has constant value for a col) → z = 0 for all rows.
       Single-obs user (std = NaN)                   → z = NaN (treated as missing).

    Bounded/ordinal EMA cols (removed from _NORMALIZE_COLS in Step 1) are
    skipped entirely — their raw integer values feed clinical-threshold
    discretization directly.

    Requires harmonized DataFrame (uid/id already renamed to user_id).
    Called separately per dataset before concat — prevents user_id collisions
    between StudentLife and LifeSnaps.

    NaN input values remain NaN throughout.
    """
    df = df.copy()
    present = [c for c in _NORMALIZE_COLS if c in df.columns]

    # Split present cols into population-z and per-user-z buckets
    pop_cols  = [c for c in present if c in _POPULATION_ZSCORE_COLS]
    user_cols = [c for c in present if c not in _POPULATION_ZSCORE_COLS]

    # ── 1. Population z-score ────────────────────────────────────────────────
    if pop_cols:
        pop_stats = _load_pop_stats()
        for col in pop_cols:
            if col not in pop_stats:
                print(f'  [WARN] {col} not in population_norm_stats.json — skipping normalization')
                continue
            pop_mean = pop_stats[col]['mean']
            pop_std  = pop_stats[col]['std']
            if pop_std == 0 or np.isnan(pop_std):
                print(f'  [WARN] {col}: pop_std={pop_std!r}, leaving column unscaled')
                continue
            df[col] = (df[col] - pop_mean) / pop_std
        print(f'Population z-scored {len(pop_cols)} sensor col(s).')

    # ── 2. Per-user z-score ──────────────────────────────────────────────────
    for col in user_cols:
        grp      = df.groupby('user_id')[col]
        mean     = grp.transform('mean')
        std      = grp.transform('std', ddof=1)
        safe_std = std.where(std != 0, np.nan)   # std=0 → NaN so division is safe on any dtype
        z        = (df[col] - mean) / safe_std   # std=0 rows produce NaN (not inf, not exception)
        constant = (std == 0) & df[col].notna()
        df[col]  = z.where(~constant, other=0.0)
    if user_cols:
        print(f'Per-user z-scored {len(user_cols)} col(s) '
              f'({len(df["user_id"].unique())} users).')

    print(f'Normalized {len(present)} cols total '
          f'({len(pop_cols)} population, {len(user_cols)} per-user).')
    return df
