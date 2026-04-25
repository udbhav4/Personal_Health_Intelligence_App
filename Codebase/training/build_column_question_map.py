"""
Generate configs/column_question_map.csv from source label files.

Sources:
  datasets/data_loaded_cleaned/studentlife_self/studentlife_column_labels.csv
  datasets/data_loaded_cleaned/nhanes_self/nhanes_column_labels.csv
  LifeSnaps labels — hardcoded (no source label file exists)

Output columns:
  original_column   — raw column name in the source dataset
  harmonized_column — post-harmonization name (empty for composite sub-items)
  composite_column  — target composite column (empty for surviving columns)
  dataset           — studentlife | lifesnaps | nhanes
  question_label    — exact question / description text
"""

import os
import pandas as pd

_ROOT      = os.path.join(os.path.dirname(__file__), '..', '..')
_SL_LABELS = os.path.join(_ROOT, 'datasets', 'data_loaded_cleaned', 'studentlife_self', 'studentlife_column_labels.csv')
_NH_LABELS = os.path.join(_ROOT, 'datasets', 'data_loaded_cleaned', 'nhanes_self',      'nhanes_column_labels.csv')
_OUT       = os.path.join(_ROOT, 'configs', 'column_question_map.csv')

# ── StudentLife renames (original → harmonized) ───────────────────────────────
# Only columns whose name changes after harmonization.

_SL_RENAME = {
    'uid': 'user_id',
}

# ── StudentLife composite sub-items (original → composite_column) ─────────────

_SL_COMPOSITE = {
    'phq_interest':      'phq_total',
    'phq_depressed':     'phq_total',
    'phq_sleep':         'phq_total',
    'phq_tired':         'phq_total',
    'phq_appetite':      'phq_total',
    'phq_failure':       'phq_total',
    'phq_concentrate':   'phq_total',
    'phq_psychomotor':   'phq_total',
    'phq_death':         'phq_total',
    'pss_2':             'pss_helplessness',
    'pss_10':            'pss_helplessness',
    'pss_4':             'pss_self_efficacy',
    'pss_5':             'pss_self_efficacy',
    'panas_enthusiastic': 'panas_pa',
    'panas_inspired':    'panas_pa',
    'panas_scared':      'panas_na',
    'panas_guilty':      'panas_na',
    'panas_hostile':     'panas_na',
    'lonely_2':          'lonely_total',
    'lonely_11':         'lonely_total',
    'lonely_14':         'lonely_total',
    'e_talkative':       'extraversion',
    'e_reserved_r':      'extraversion',
    'e_quiet_r':         'extraversion',
    'e_sociable':        'extraversion',
    'n_depressed':       'neuroticism',
    'n_tense':           'neuroticism',
    'n_worries':         'neuroticism',
    'n_stable_r':        'neuroticism',
    'n_moody':           'neuroticism',
    'n_nervous':         'neuroticism',
    'vr_general_health':       'vr_physical_health',
    'vr_moderate_activity':    'vr_physical_health',
    'vr_physical_limit_work':  'vr_physical_health',
    'vr_pain_interference':    'vr_physical_health',
    'vr_energy':               'vr_mental_health',
    'vr_emotional_limit_work': 'vr_mental_health',
    'vr_emotional_limit_care': 'vr_mental_health',
    'vr_downhearted':          'vr_mental_health',
    'vr_social_interference':  'vr_mental_health',
    'sleep_trouble_30min': 'sleep_disturbances',
    'sleep_wakeup':        'sleep_disturbances',
    'sleep_cough_snore':   'sleep_disturbances',
    'sleep_bad_dreams':    'sleep_disturbances',
    'pain_during_sleep':   'sleep_disturbances',
    'low_enthusiasm':      'sleep_disturbances',
}

# ── StudentLife columns to exclude entirely ───────────────────────────────────
# Structural identifiers, redundant features, and dropped survey items.

_SL_DROP = {
    'uid', 'date', 'hour',
    'sedentary_ratio', 'prev_day_sedentary_ratio',
    'exercise_have', 'prev_day_exercise_have',
    'prev_evening_running_tasks', 'prev_evening_peak_running_tasks',
    'pss_1', 'pss_3', 'pss_6', 'pss_7', 'pss_8', 'pss_9',
    'panas_interested', 'panas_distressed', 'panas_upset', 'panas_strong',
    'panas_active', 'panas_irritable', 'panas_alert', 'panas_nervous',
    'panas_attentive', 'panas_jittery', 'panas_afraid', 'panas_proud',
    'panas_determined',
    'lonely_1', 'lonely_3', 'lonely_4', 'lonely_5', 'lonely_6',
    'lonely_7', 'lonely_8', 'lonely_9', 'lonely_10', 'lonely_12',
    'lonely_13', 'lonely_15', 'lonely_16', 'lonely_17', 'lonely_18',
    'lonely_19', 'lonely_20',
    'vr_climb_stairs', 'vr_physical_limit_kind',
}

# ── NHANES renames (original → harmonized) ───────────────────────────────────

_NH_RENAME = {
    'RIDAGEYR':  'age',
    'RIAGENDR':  'sex',
    'BMI':       'bmi',
    'EDUCATION': 'education_level',
    'DMDMARTL':  'marital_status',
    'HSD010':    'general_health',
    'SMQ040':    'smoking',
    'PAD680':    'active_ratio',
}

# ── NHANES composite sub-items ────────────────────────────────────────────────

_NH_COMPOSITE = {
    'DPQ010': 'phq_total', 'DPQ020': 'phq_total', 'DPQ030': 'phq_total',
    'DPQ040': 'phq_total', 'DPQ050': 'phq_total', 'DPQ060': 'phq_total',
    'DPQ070': 'phq_total', 'DPQ080': 'phq_total', 'DPQ090': 'phq_total',
    'DIQ010': 'diabetes_status', 'DIQ160': 'diabetes_status', 'DIQ050': 'diabetes_status',
    'MCQ160A': 'chronic_condition', 'MCQ160N': 'chronic_condition',
    'MCQ160C': 'chronic_condition', 'MCQ160E': 'chronic_condition',
    'MCQ160M': 'chronic_condition', 'MCQ160O': 'chronic_condition',
    'MCQ520':  'chronic_condition', 'MCQ080':  'chronic_condition',
    'CDQ001': 'pain_level', 'CDQ010': 'pain_level',
}

# ── NHANES columns to exclude entirely ───────────────────────────────────────

_NH_DROP = {
    'SEQN', 'RIDRETH3', 'INDHHIN2',
    'DMDEDUC2', 'DMDEDUC3',
    'SMQ020', 'SMD641', 'ALQ111',
    'WHD010', 'WHD020',
    'PAQ605', 'PAQ620', 'PAQ635', 'PAQ650',
}

# ── LifeSnaps rows (hardcoded — no source label file) ─────────────────────────
# Tuple: (original_column, harmonized_column, composite_column, question_label)

_LS_ROWS = [
    ('gender',                'sex',               '', 'Biological sex (recorded at account setup)'),
    ('bmi',                   'bmi',               '', 'Body Mass Index (kg/m²)'),
    ('daily_active_ratio',    'active_ratio',      '', 'Proportion of day spent active (Fitbit accelerometer)'),
    ('prev_day_active_ratio', 'prev_day_active_ratio', '', 'Proportion of previous day spent active (Fitbit — carry-forward from prior day)'),
    ('hourly_bpm',            'hourly_bpm',        '', 'Hourly heart rate from Fitbit (beats per minute) — passive sensor'),
    ('prev_day_bpm',          'prev_day_bpm',      '', 'Average heart rate on previous day from Fitbit (bpm) — passive sensor'),
    ('hourly_steps',          'hourly_steps',      '', 'Hourly step count from Fitbit — passive sensor'),
    ('prev_day_steps',        'prev_day_steps',    '', 'Total steps on previous day from Fitbit — passive sensor'),
    ('prev_night_resting_hr', 'prev_night_resting_hr', '', 'Resting heart rate last night from Fitbit (bpm) — passive sensor'),
    ('prev_night_temperature','prev_night_temperature', '', 'Skin temperature last night from Fitbit (°C) — passive sensor'),
    ('prev_night_minutesasleep', 'sleep_hours',   '', 'Hours of sleep last night from Fitbit tracking (converted from minutes ÷ 60)'),
]


# ── Builders ──────────────────────────────────────────────────────────────────

def _build_studentlife_rows():
    labels = pd.read_csv(_SL_LABELS)
    rows = []
    for _, r in labels.iterrows():
        col   = r['column']
        label = r['label']
        if col in _SL_DROP:
            continue
        if col in _SL_COMPOSITE:
            rows.append((col, '', _SL_COMPOSITE[col], 'studentlife', label))
        elif col in _SL_RENAME:
            rows.append((col, _SL_RENAME[col], '', 'studentlife', label))
        else:
            rows.append((col, col, '', 'studentlife', label))
    return rows


def _build_nhanes_rows():
    labels = pd.read_csv(_NH_LABELS)
    rows = []
    for _, r in labels.iterrows():
        col   = r['column']
        label = r['label']
        if col in _NH_DROP:
            continue
        if col in _NH_COMPOSITE:
            rows.append((col, '', _NH_COMPOSITE[col], 'nhanes', label))
        elif col in _NH_RENAME:
            rows.append((col, _NH_RENAME[col], '', 'nhanes', label))
        else:
            rows.append((col, col, '', 'nhanes', label))
    return rows


def _build_lifesnaps_rows():
    return [
        (orig, harm, comp, 'lifesnaps', qlabel)
        for orig, harm, comp, qlabel in _LS_ROWS
    ]


# ── Main ──────────────────────────────────────────────────────────────────────

def build_map():
    rows = (
        _build_studentlife_rows()
        + _build_lifesnaps_rows()
        + _build_nhanes_rows()
    )
    df = pd.DataFrame(rows, columns=[
        'original_column', 'harmonized_column',
        'composite_column', 'dataset', 'question_label',
    ])
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    df.to_csv(_OUT, index=False)
    print(f'Saved {len(df)} rows → {_OUT}')


if __name__ == '__main__':
    build_map()
