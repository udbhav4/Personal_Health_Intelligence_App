import pandas as pd
import numpy as np


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
    # ── activity (both datasets, harmonized to active_ratio) ─────────────────
    'active_ratio', 'prev_day_active_ratio',
    # ── running tasks (StudentLife) ──────────────────────────────────────────
    'avg_running_tasks_window', 'prev_day_avg_running_tasks', 'prev_day_peak_running_tasks',
    # ── sleep supplementaries (StudentLife) ──────────────────────────────────
    # sleep_latency_mins excluded — clinical threshold (<20 / 20-45 / ≥45 min)
    # prev_day_sleep_ema_hours excluded — clinical threshold (<6 / 6-9 / ≥9 hrs)
    # sleep_quality_rating excluded — PSQI survey (once per semester period);
    #   every daily row in a period has the identical value → within-user std=0
    'prev_day_sleep_ema_rating', 'prev_day_sleep_ema_alertness',
    # ── heart rate + steps (LifeSnaps) ───────────────────────────────────────
    # hourly_bpm / prev_day_bpm excluded — clinical thresholds on raw BPM values
    'hourly_steps', 'prev_day_steps',
    # ── physio sleep (LifeSnaps) ─────────────────────────────────────────────
    'prev_night_resting_hr', 'prev_night_temperature',
    # ── daily EMA (StudentLife) ──────────────────────────────────────────────
    'mood_how', 'mood_happy', 'mood_sad',
    'prev_day_mood_how', 'prev_day_mood_happy', 'prev_day_mood_sad',
    'productivity', 'prev_day_productivity',
    'stress_ema_level', 'prev_day_stress_ema_level',
    'prev_day_events_positive', 'prev_day_events_negative',
]


def normalize_per_user(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-user z-score normalization grouped by user_id.

    Requires harmonized DataFrame (uid/id already renamed to user_id).
    Called separately per dataset before concat — prevents user_id collisions
    between StudentLife and LifeSnaps.

    std == 0  (user has constant value for a col) → z = 0 for all rows.
    Single-obs user (std = NaN)                   → z = NaN (treated as missing).
    NaN input values remain NaN.
    """
    df = df.copy()
    present = [c for c in _NORMALIZE_COLS if c in df.columns]
    for col in present:
        grp      = df.groupby('user_id')[col]
        mean     = grp.transform('mean')
        std      = grp.transform('std', ddof=1)
        safe_std = std.where(std != 0, np.nan)   # std=0 → NaN so division is safe on any dtype
        z        = (df[col] - mean) / safe_std   # std=0 rows produce NaN (not inf, not exception)
        constant = (std == 0) & df[col].notna()
        df[col]  = z.where(~constant, other=0.0)
    print(f'Normalized {len(present)} cols per-user ({len(df["user_id"].unique())} users).')
    return df
