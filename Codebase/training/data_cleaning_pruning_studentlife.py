import os
import pandas as pd
import numpy as np
from utils import nan_audit, vif_from_corr_matrix

BASE = r'C:\Users\udbha\Documents\VS Code\MedApp\datasets'
STUDENTLIFE_PATH = os.path.join(BASE, 'data_loaded_cleaned', 'studentlife_self', 'studentlife_daily_and_surveys.csv')


def load_studentlife() -> pd.DataFrame:
    if not os.path.exists(STUDENTLIFE_PATH):
        raise FileNotFoundError(f'StudentLife file not found: {STUDENTLIFE_PATH}')
    return pd.read_csv(STUDENTLIFE_PATH)



# ---------------------------------------------------------------------------
# StudentLife — cleaning
# ---------------------------------------------------------------------------

def clean_studentlife(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.loc[df['sleep_hours'] > 24, 'sleep_hours'] = np.nan
    nan_audit(df, 'StudentLife post-clean')
    return df


# ---------------------------------------------------------------------------
# StudentLife — redundancy analysis
# Node groups defined over RAW columns (pre-merge/pre-drop) so the report
# surfaces within-node correlations and VIF before any aggregation.
# ---------------------------------------------------------------------------

# PSQI disturbance/dysfunction items (periodic monthly survey) → sleep_disturbances.
# Includes low_enthusiasm (PSQI component 7: daytime dysfunction from poor sleep).
_SL_SLEEP_DISTURBANCE_ITEMS = [
    'sleep_trouble_30min', 'sleep_wakeup', 'sleep_cough_snore',
    'sleep_bad_dreams', 'pain_during_sleep', 'low_enthusiasm',
]

_SL_NODE_GROUPS = {
    # ── Survey scales ────────────────────────────────────────────────────────
    'depression': [
        'phq_interest', 'phq_depressed', 'phq_sleep', 'phq_tired',
        'phq_appetite', 'phq_failure', 'phq_concentrate', 'phq_psychomotor', 'phq_death',
    ],
    'pss_helplessness':  ['pss_1', 'pss_2', 'pss_3', 'pss_6', 'pss_9', 'pss_10'],
    'pss_self_efficacy': ['pss_4', 'pss_5', 'pss_7', 'pss_8'],
    'positive_affect': [
        'panas_interested', 'panas_strong', 'panas_enthusiastic', 'panas_active',
        'panas_alert', 'panas_inspired', 'panas_attentive', 'panas_proud', 'panas_determined',
    ],
    # panas_afraid included to show r=0.901 with panas_scared before statistical drop
    'negative_affect': [
        'panas_distressed', 'panas_upset', 'panas_scared', 'panas_irritable',
        'panas_nervous', 'panas_jittery', 'panas_afraid', 'panas_guilty', 'panas_hostile',
    ],
    # lonely_20 included to show r=0.903 with lonely_19 before statistical drop
    'loneliness': [f'lonely_{i}' for i in range(1, 21)],
    'extraversion':  ['e_talkative', 'e_reserved_r', 'e_quiet_r', 'e_sociable'],
    'neuroticism':   ['n_depressed', 'n_tense', 'n_worries', 'n_stable_r', 'n_moody', 'n_nervous'],
    # vr_climb_stairs and vr_physical_limit_kind included to show FLAG pairs before logical drop
    'vr_physical_health': [
        'vr_general_health', 'vr_moderate_activity', 'vr_climb_stairs',
        'vr_physical_limit_work', 'vr_physical_limit_kind', 'vr_pain_interference',
    ],
    'vr_mental_health': [
        'vr_energy', 'vr_emotional_limit_work', 'vr_emotional_limit_care',
        'vr_downhearted', 'vr_social_interference',
    ],

    # ── Sensor / EMA / PSQI ─────────────────────────────────────────────────
    'sleep_quality': [
        'sleep_hours', 'sleep_quality_rating', 'sleep_latency_mins',
        'prev_day_sleep_ema_hours', 'prev_day_sleep_ema_rating', 'prev_day_sleep_ema_alertness',
    ] + _SL_SLEEP_DISTURBANCE_ITEMS,

    # sedentary cols included to show r=1.0 before drop
    'activity': [
        'sedentary_ratio', 'active_ratio',
        'prev_day_sedentary_ratio', 'prev_day_active_ratio',
    ],
    'screen_usage': [
        'dark_window_minutes', 'unlocked_window_minutes',
        'screen_time_window_minutes', 'yesterday_nighttime_screen_minutes',
        'prev_day_total_screen_time_minutes',
    ],
    # evening cols included to show high-r with daily cols before drop
    'running_tasks': [
        'avg_running_tasks_window',
        'prev_evening_running_tasks', 'prev_evening_peak_running_tasks',
        'prev_day_avg_running_tasks', 'prev_day_peak_running_tasks',
    ],
    'communication': [
        'call_count', 'call_duration_total', 'sms_count',
        'prev_day_call_count', 'prev_day_call_duration_total', 'prev_day_sms_count',
    ],
    'mood': [
        'mood_happy', 'mood_sad', 'mood_how',
        'prev_day_mood_happy', 'prev_day_mood_sad', 'prev_day_mood_how',
    ],
    # exercise_have cols included to show r=0.833 before drop
    'exercise': [
        'exercise_type', 'exercise_have', 'exercise_walk',
        'prev_day_exercise_type', 'prev_day_exercise_have', 'prev_day_exercise_walk',
    ],
    'productivity':  ['productivity', 'prev_day_productivity'],
    'stress_ema':    ['stress_ema_level', 'prev_day_stress_ema_level'],
    'social_events': ['prev_day_events_positive', 'prev_day_events_negative'],
}

_CORR_FLAG  = 0.70
_CORR_PRUNE = 0.85



def studentlife_redundancy_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Print per-node VIF + high-r pairs. Returns a DataFrame with columns
    [node, column, vif, high_vif] for further inspection.
    """
    rows = []
    print(f'\n{"="*65}')
    print('StudentLife Node Redundancy Report')
    print(f'Flag: r > {_CORR_FLAG}  |  Prune candidate: r > {_CORR_PRUNE}  |  High VIF: > 5')
    print(f'{"="*65}')
    for node, cols in _SL_NODE_GROUPS.items():
        present = [c for c in cols if c in df.columns]
        if len(present) < 2:
            continue
        sub  = df[present].apply(pd.to_numeric, errors='coerce')
        corr = sub.corr().abs()
        vifs = vif_from_corr_matrix(corr)
        print(f'\n[{node}]')
        for col, v in vifs.items():
            flag = '  *** HIGH' if isinstance(v, float) and v > 5 else ''
            print(f'    {col:<42}  VIF={v}{flag}')
            rows.append({'node': node, 'column': col, 'vif': v, 'high_vif': bool(isinstance(v, float) and v > 5)})
        for i, c1 in enumerate(present):
            for c2 in present[i + 1:]:
                r = corr.loc[c1, c2]
                if r > _CORR_FLAG:
                    tag = '[PRUNE CANDIDATE]' if r > _CORR_PRUNE else '[FLAG]'
                    print(f'    {c1} ~ {c2}:  r={r:.3f}  {tag}')
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Scale merge helpers
# ---------------------------------------------------------------------------

_SL_PHQ_ITEMS         = _SL_NODE_GROUPS['depression']
# PSS-4 validated short form (Cohen & Williamson 1988): items 2,4,5,10 from PSS-10.
# Split by subfactor — 2 items each keeps subscale structure with minimum question burden.
_SL_PSS_HELPLESSNESS  = ['pss_2', 'pss_10']   # unable to control + difficulties piling up
_SL_PSS_SELF_EFFICACY = ['pss_4', 'pss_5']    # confident handling problems + things going your way
# PA: keep only items with no equivalent in PHQ/PSS/mood-EMA
_SL_PANAS_PA          = ['panas_enthusiastic', 'panas_inspired']
# NA: keep only fear/guilt/hostility — absent from PHQ, PSS-4, stress_ema, mood_sad
# panas_afraid already dropped (r=0.901 with panas_scared)
_SL_PANAS_NA          = ['panas_scared', 'panas_guilty', 'panas_hostile']
# UCLA-3 validated short form (Hughes et al. 2004): lack companionship / left out / isolated.
# Mapped to column names by exact item text match against LonelinessScale.csv header.
_SL_LONELY_ITEMS      = ['lonely_2', 'lonely_11', 'lonely_14']
_SL_EXTRA_ITEMS       = _SL_NODE_GROUPS['extraversion']
_SL_NEURO_ITEMS       = _SL_NODE_GROUPS['neuroticism']
# vr_climb_stairs dropped: r=0.751 with vr_moderate_activity (moderate_activity is more general)
# vr_physical_limit_kind dropped: r=0.703 with vr_physical_limit_work (same role-limitation construct)
_SL_VR12_PHYSICAL     = ['vr_general_health', 'vr_moderate_activity',
                          'vr_physical_limit_work', 'vr_pain_interference']
_SL_VR12_MENTAL       = _SL_NODE_GROUPS['vr_mental_health']


def _merge_sl_phq_total(df: pd.DataFrame) -> pd.DataFrame:
    """
    9 PHQ-9 items (0–3 each) → phq_total (0–27).
    Logical merge: PHQ-9 designed for summation; all 9 items cover distinct
    DSM-5 symptom domains. No reversals needed.
    """
    df = df.copy()
    df['phq_total'] = df[_SL_PHQ_ITEMS].sum(axis=1, min_count=1)
    return df.drop(columns=[c for c in _SL_PHQ_ITEMS if c in df.columns])


def _merge_sl_pss(df: pd.DataFrame) -> pd.DataFrame:
    """
    PSS-4 validated short form (Cohen & Williamson 1988) → two subscales:
      pss_helplessness  (pss_2 + pss_10; range 0–8):  higher = more helpless/distressed.
      pss_self_efficacy (pss_4 + pss_5;  range 0–8):  higher = more in control/coping.
    All 10 raw items dropped after merge.
    """
    df = df.copy()
    df['pss_helplessness']  = df[_SL_PSS_HELPLESSNESS].sum(axis=1, min_count=1)
    df['pss_self_efficacy'] = df[_SL_PSS_SELF_EFFICACY].sum(axis=1, min_count=1)
    # Drop all 10 raw PSS items; composites above replace them
    all_pss = _SL_NODE_GROUPS['pss_helplessness'] + _SL_NODE_GROUPS['pss_self_efficacy']
    return df.drop(columns=[c for c in all_pss if c in df.columns])


def _merge_sl_panas(df: pd.DataFrame) -> pd.DataFrame:
    """
    PANAS → two subscales, minimal unique items not covered by PHQ/PSS/mood-EMA:
      panas_pa (2 items): enthusiastic (positive activation), inspired (motivation).
      panas_na (3 items): scared (fear), guilty (guilt), hostile (hostility/aggression).
    All 18 raw PANAS items dropped after merge.
    """
    df = df.copy()
    df['panas_pa'] = df[_SL_PANAS_PA].sum(axis=1, min_count=1)
    df['panas_na'] = df[_SL_PANAS_NA].sum(axis=1, min_count=1)
    all_panas = _SL_NODE_GROUPS['positive_affect'] + _SL_NODE_GROUPS['negative_affect']
    return df.drop(columns=[c for c in all_panas if c in df.columns])


def _merge_sl_lonely_total(df: pd.DataFrame) -> pd.DataFrame:
    """
    UCLA-3 validated short form (Hughes et al. 2004) → lonely_total (3–12).
    Items: lonely_2 (lack companionship), lonely_11 (left out), lonely_14 (isolated).
    Positively worded items already reverse-scored in data_loading.py (5 − value).
    Higher = more lonely.
    """
    df = df.copy()
    df['lonely_total'] = df[_SL_LONELY_ITEMS].sum(axis=1, min_count=1)
    all_lonely = _SL_NODE_GROUPS['loneliness']
    return df.drop(columns=[c for c in all_lonely if c in df.columns])


def _merge_sl_big_five(df: pd.DataFrame) -> pd.DataFrame:
    """
    BFI items (1–5 Likert) → two subscale sums:
      extraversion (4 items, range 4–20).
      neuroticism  (6 items, range 6–30).
    _r items already reverse-scored in data_loading.py (6 − value).
    Higher = more extraverted / more neurotic.
    """
    df = df.copy()
    df['extraversion'] = df[_SL_EXTRA_ITEMS].sum(axis=1, min_count=1)
    df['neuroticism']  = df[_SL_NEURO_ITEMS].sum(axis=1, min_count=1)
    all_bf = _SL_EXTRA_ITEMS + _SL_NEURO_ITEMS
    return df.drop(columns=[c for c in all_bf if c in df.columns])


def _merge_sl_vr12(df: pd.DataFrame) -> pd.DataFrame:
    """
    VR-12 → two component scores (all items already reverse-scored in data_loading.py):

      vr_physical_health (4 items after logical drops):
        vr_general_health, vr_moderate_activity, vr_physical_limit_work, vr_pain_interference.
        Logical drops: vr_climb_stairs (r=0.751 with vr_moderate_activity; moderate_activity
        is more general), vr_physical_limit_kind (r=0.703 with vr_physical_limit_work;
        same role-limitation construct — amount of work is clearer signal).

      vr_mental_health (5 items, max within-component r=0.626):
        vr_energy, vr_emotional_limit_work, vr_emotional_limit_care,
        vr_downhearted, vr_social_interference.

    Higher = worse physical / mental health.
    """
    df = df.copy()
    df['vr_physical_health'] = df[_SL_VR12_PHYSICAL].sum(axis=1, min_count=1)
    df['vr_mental_health']   = df[_SL_VR12_MENTAL].sum(axis=1, min_count=1)
    all_vr = _SL_NODE_GROUPS['vr_physical_health'] + _SL_VR12_MENTAL
    return df.drop(columns=[c for c in all_vr if c in df.columns])


def _merge_sl_sleep_disturbances(df: pd.DataFrame) -> pd.DataFrame:
    """
    6 PSQI monthly-survey items → sleep_disturbances (sum).
    All items measure reasons sleep was disrupted or impaired in the past month:
      sleep_trouble_30min  — trouble falling asleep (>30 min)
      sleep_wakeup         — waking in the middle of the night / early morning
      sleep_cough_snore    — coughing or snoring
      sleep_bad_dreams     — bad dreams
      pain_during_sleep    — pain
      low_enthusiasm       — PSQI component 7: daytime dysfunction (trouble keeping
                             up enthusiasm) caused by poor sleep quality
    Merging reduces 6 periodic survey items to one sleep-problem burden score.
    Higher = more sleep disruption / daytime impairment.
    """
    df = df.copy()
    df['sleep_disturbances'] = df[_SL_SLEEP_DISTURBANCE_ITEMS].sum(axis=1, min_count=1)
    return df.drop(columns=[c for c in _SL_SLEEP_DISTURBANCE_ITEMS if c in df.columns])


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

def prune_studentlife(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 1 — Redundancy report: prints all within-node r and VIF values.

    Step 2 — Statistical drops (r ≥ 0.80 from report):
      sedentary_ratio              r=1.000 with active_ratio            (sum to 1 by construction)
      prev_day_sedentary_ratio     r=1.000 with prev_day_active_ratio   (same identity, t-1)
      exercise_have                r=0.833 with exercise_type           (binary gate subsumed by ordinal)
      prev_day_exercise_have       r=0.835 with prev_day_exercise_type  (same at t-1)
      panas_afraid                 r=0.901 with panas_scared            (dropped before panas_na merge)
      lonely_20                    r=0.903 with lonely_19               (dropped before lonely_total merge)
      prev_evening_running_tasks   r=0.845 with prev_day_avg_running_tasks
      prev_evening_peak_running_tasks r=0.802 with prev_day_peak_running_tasks

    Step 3 — Logical drops within scale groups (FLAG pairs + semantic overlap):
      vr_climb_stairs          r=0.751 with vr_moderate_activity   → vr_moderate_activity is more general
      vr_physical_limit_kind   r=0.703 with vr_physical_limit_work → same role-limitation construct

    Step 4 — Scale merges (validated scoring + question reduction):
      Survey:  phq_total, pss_helplessness, pss_self_efficacy,
               panas_pa (9 items), panas_na (8 items), lonely_total (19 items),
               extraversion, neuroticism, vr_physical_health (4 items), vr_mental_health (5 items)
      PSQI:    sleep_disturbances (6 monthly items → 1 burden score)
    """
    studentlife_redundancy_report(df)

    # Step 2 — statistical drops
    stat_drops = [
        'sedentary_ratio',
        'prev_day_sedentary_ratio',
        'exercise_have',
        'prev_day_exercise_have',
        'panas_afraid',
        'lonely_20',
        'prev_evening_running_tasks',
        'prev_evening_peak_running_tasks',
    ]
    df = df.drop(columns=[c for c in stat_drops if c in df.columns])

    # Step 3+4 — logical drops embedded in merge helpers (see each helper's docstring)
    df = _merge_sl_phq_total(df)
    df = _merge_sl_pss(df)
    df = _merge_sl_panas(df)
    df = _merge_sl_lonely_total(df)
    df = _merge_sl_big_five(df)
    df = _merge_sl_vr12(df)
    df = _merge_sl_sleep_disturbances(df)

    nan_audit(df, 'StudentLife post-prune')
    return df


if __name__ == '__main__':
    _df = clean_studentlife(load_studentlife())
    vif_df = studentlife_redundancy_report(_df)
    print('\n--- VIF summary ---')
    print(vif_df.to_string(index=False))
