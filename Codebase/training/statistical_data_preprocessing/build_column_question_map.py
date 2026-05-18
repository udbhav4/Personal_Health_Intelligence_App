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
  options_json      — JSON string: [[v, "label"], ...] for discrete opts,
                      {"min":n,"max":n,"unit":"str"} for continuous range,
                      "" if no options defined
"""

import json
import os
import re

import pandas as pd

_ROOT      = os.path.join(os.path.dirname(__file__), '..', '..', '..')
_SL_LABELS = os.path.join(_ROOT, 'datasets', 'data_loaded_cleaned', 'studentlife_self', 'studentlife_column_labels.csv')
_NH_LABELS = os.path.join(_ROOT, 'datasets', 'data_loaded_cleaned', 'nhanes_self',      'nhanes_column_labels.csv')
_OUT       = os.path.join(_ROOT, 'configs', 'column_question_map.csv')

# ── StudentLife renames (original → harmonized) ───────────────────────────────

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

# ── Options / ranges ──────────────────────────────────────────────────────────
# Verified against unique values in nhanes_merged_cleaned.csv and
# studentlife_daily_and_surveys.csv. Sentinel codes (77/99/999 etc.) excluded.
#
# _OPTIONS: col → {int_value: label_str}  (discrete; chips or scrollable picker)
# _RANGES:  col → {"min": n, "max": n, "unit": str}  (continuous slider)
# StudentLife cols with "Options: [n]..." embedded in the label are parsed
# automatically; explicit _SL_OPTIONS entries take precedence.

_PHQ_OPTS    = {0: 'Not at all', 1: 'Several days', 2: 'More than half the days', 3: 'Nearly every day'}
_PSQI_FREQ   = {0: 'Not in past month', 1: 'Less than once/week', 2: 'Once or twice/week', 3: '3+ times/week'}
_PSS_OPTS    = {0: 'Never', 1: 'Almost never', 2: 'Sometimes', 3: 'Fairly often', 4: 'Very often'}
_PANAS_OPTS  = {1: 'Very slightly/Not at all', 2: 'A little', 3: 'Moderately', 4: 'Quite a bit', 5: 'Extremely'}
_BIG5_OPTS   = {1: 'Disagree strongly', 2: 'Disagree a little', 3: 'Neither', 4: 'Agree a little', 5: 'Agree strongly'}
_UCLA_OPTS   = {1: 'Never', 2: 'Rarely', 3: 'Sometimes', 4: 'Often'}
_ALQ_FREQ    = {
    0: 'Never', 1: 'Every day', 2: 'Nearly every day',
    3: '3-4 times/week', 4: '2 times/week', 5: 'Once/week',
    6: '2-3 times/month', 7: 'Once/month',
    8: '7-11 times/year', 9: '3-6 times/year', 10: '1-2 times/year',
}
_VR_FREQ5    = {1: 'All the time', 2: 'Most of the time', 3: 'Some of the time', 4: 'A little of the time', 5: 'None of the time'}

_NHANES_OPTIONS: dict[str, dict[int, str]] = {
    'ALQ121':   _ALQ_FREQ,
    'ALQ142':   _ALQ_FREQ,
    'SMQ040':   {1: 'Every day', 2: 'Some days', 3: 'Not at all'},
    'CDQ001':   {1: 'Yes', 2: 'No'},
    'CDQ010':   {1: 'Yes', 2: 'No'},
    'HSD010':   {1: 'Excellent', 2: 'Very good', 3: 'Good', 4: 'Fair', 5: 'Poor'},
    'DMDMARTL': {1: 'Married', 2: 'Widowed', 3: 'Divorced', 4: 'Separated', 5: 'Never married', 6: 'Living with partner'},
    'DIQ010':   {1: 'Yes', 2: 'No', 3: 'Borderline'},
    'DIQ160':   {1: 'Yes', 2: 'No'},
    'DIQ050':   {1: 'Yes', 2: 'No'},
    'MCQ160A':  {1: 'Yes', 2: 'No'},
    'MCQ160N':  {1: 'Yes', 2: 'No'},
    'MCQ160C':  {1: 'Yes', 2: 'No'},
    'MCQ160E':  {1: 'Yes', 2: 'No'},
    'MCQ160M':  {1: 'Yes', 2: 'No'},
    'MCQ160O':  {1: 'Yes', 2: 'No'},
    'MCQ520':   {1: 'Yes', 2: 'No'},
    'MCQ080':   {1: 'Yes', 2: 'No'},
    'DPQ010': _PHQ_OPTS, 'DPQ020': _PHQ_OPTS, 'DPQ030': _PHQ_OPTS,
    'DPQ040': _PHQ_OPTS, 'DPQ050': _PHQ_OPTS, 'DPQ060': _PHQ_OPTS,
    'DPQ070': _PHQ_OPTS, 'DPQ080': _PHQ_OPTS, 'DPQ090': _PHQ_OPTS,
    'DPQ100': {0: 'Not difficult at all', 1: 'Somewhat difficult', 2: 'Very difficult', 3: 'Extremely difficult'},
    # EDUCATION: adult categories (DMDEDUC2) — values 1-5 present in merged col
    'EDUCATION': {
        1: 'Less than 9th grade',
        2: '9-11th grade',
        3: 'High school graduate / GED',
        4: 'Some college or AA degree',
        5: 'College graduate or above',
    },
}

_SL_OPTIONS: dict[str, dict[int, str]] = {
    'sleep_quality_rating': {0: 'Very good', 1: 'Fairly good', 2: 'Fairly bad', 3: 'Very bad'},
    'sleep_trouble_30min':  _PSQI_FREQ,
    'sleep_wakeup':         _PSQI_FREQ,
    'sleep_cough_snore':    _PSQI_FREQ,
    'sleep_bad_dreams':     _PSQI_FREQ,
    'pain_during_sleep':    _PSQI_FREQ,
    'low_enthusiasm':       _PSQI_FREQ,
    'phq_interest':      _PHQ_OPTS, 'phq_depressed':   _PHQ_OPTS,
    'phq_sleep':         _PHQ_OPTS, 'phq_tired':        _PHQ_OPTS,
    'phq_appetite':      _PHQ_OPTS, 'phq_failure':      _PHQ_OPTS,
    'phq_concentrate':   _PHQ_OPTS, 'phq_psychomotor':  _PHQ_OPTS,
    'phq_death':         _PHQ_OPTS,
    'pss_2':  _PSS_OPTS, 'pss_10': _PSS_OPTS,
    'pss_4':  _PSS_OPTS, 'pss_5':  _PSS_OPTS,
    'panas_enthusiastic': _PANAS_OPTS, 'panas_inspired': _PANAS_OPTS,
    'panas_scared':       _PANAS_OPTS, 'panas_guilty':   _PANAS_OPTS,
    'panas_hostile':      _PANAS_OPTS,
    'lonely_2': _UCLA_OPTS, 'lonely_11': _UCLA_OPTS, 'lonely_14': _UCLA_OPTS,
    'e_talkative':  _BIG5_OPTS, 'e_reserved_r': _BIG5_OPTS,
    'e_quiet_r':    _BIG5_OPTS, 'e_sociable':    _BIG5_OPTS,
    'n_depressed':  _BIG5_OPTS, 'n_tense':       _BIG5_OPTS,
    'n_worries':    _BIG5_OPTS, 'n_stable_r':    _BIG5_OPTS,
    'n_moody':      _BIG5_OPTS, 'n_nervous':     _BIG5_OPTS,
    'vr_general_health':      {1: 'Excellent', 2: 'Very good', 3: 'Good', 4: 'Fair'},
    'vr_moderate_activity':   {1: 'Yes, limited a lot', 2: 'Yes, limited a little', 3: 'No, not limited'},
    'vr_physical_limit_work': _VR_FREQ5,
    'vr_pain_interference':   {1: 'Not at all', 2: 'A little bit', 3: 'Moderately', 4: 'Quite a bit', 5: 'Extremely'},
    'vr_energy':              {1: 'All the time', 2: 'Most of the time', 3: 'A good bit', 4: 'Some of the time', 5: 'A little', 6: 'None of the time'},
    'vr_emotional_limit_work': _VR_FREQ5,
    'vr_emotional_limit_care': _VR_FREQ5,
    'vr_downhearted':         {0: 'All the time', 1: 'Most of the time', 2: 'Some of the time', 3: 'A little of the time', 4: 'None of the time'},
    'vr_social_interference': _VR_FREQ5,
}

# Continuous columns — rendered as sliders. Max values are dataset maxima
# (sentinel codes like 9999 already cleaned out by data_cleaning step).
_RANGES: dict[str, dict] = {
    'sleep_hours':             {'min': 0, 'max': 12,  'unit': 'hours'},
    'prev_day_sleep_ema_hours': {'min': 0, 'max': 12,  'unit': 'hours'},
    'prev_day_events_positive': {'min': 1, 'max': 7,   'unit': 'intensity'},
    'prev_day_events_negative': {'min': 1, 'max': 7,   'unit': 'intensity'},
    'sleep_latency_mins':      {'min': 0, 'max': 120, 'unit': 'min'},
    'SMD650':             {'min': 0, 'max': 60,  'unit': 'cigarettes/day'},
    'ALQ130':             {'min': 0, 'max': 15,  'unit': 'drinks/day'},
    'PAQ610':             {'min': 0, 'max': 7,   'unit': 'days/week'},
    'PAD615':             {'min': 0, 'max': 840, 'unit': 'min/week'},
    'PAD630':             {'min': 0, 'max': 840, 'unit': 'min/week'},
    'PAD645':             {'min': 0, 'max': 840, 'unit': 'min/week'},
    'PAD660':             {'min': 0, 'max': 840, 'unit': 'min/week'},
}


def _parse_embedded_opts(label: str) -> dict[int, str] | None:
    """Parse 'Options: [1]label1, [2]label2, ...' from a label string."""
    m = re.search(r'[Oo]ptions:\s*(.+)$', label)
    if not m:
        return None
    pairs = re.findall(r'\[(\d+)\]([^,\[\]]+)', m.group(1))
    if not pairs:
        return None
    return {int(v): lbl.strip().rstrip(',') for v, lbl in pairs}


def _opts_json(col: str, label: str, dataset: str) -> str:
    if col in _RANGES:
        return json.dumps(_RANGES[col], separators=(',', ':'))

    # Explicit dict takes precedence over embedded parsing
    opts_dict: dict[int, str] | None = None
    if dataset == 'nhanes':
        opts_dict = _NHANES_OPTIONS.get(col)
    elif dataset == 'studentlife':
        opts_dict = _SL_OPTIONS.get(col) or _parse_embedded_opts(label)

    if not opts_dict:
        return ''
    return json.dumps([[v, lbl] for v, lbl in sorted(opts_dict.items())], separators=(',', ':'))


# ── Builders ──────────────────────────────────────────────────────────────────

def _build_studentlife_rows():
    labels = pd.read_csv(_SL_LABELS)
    rows = []
    for _, r in labels.iterrows():
        col   = r['column']
        label = r['label']
        if col in _SL_DROP:
            continue
        opts = _opts_json(col, label, 'studentlife')
        if col in _SL_COMPOSITE:
            rows.append((col, '', _SL_COMPOSITE[col], 'studentlife', label, opts))
        elif col in _SL_RENAME:
            rows.append((col, _SL_RENAME[col], '', 'studentlife', label, opts))
        else:
            rows.append((col, col, '', 'studentlife', label, opts))
    return rows


def _build_nhanes_rows():
    labels = pd.read_csv(_NH_LABELS)
    rows = []
    for _, r in labels.iterrows():
        col   = r['column']
        label = r['label']
        if col in _NH_DROP:
            continue
        opts = _opts_json(col, label, 'nhanes')
        if col in _NH_COMPOSITE:
            rows.append((col, '', _NH_COMPOSITE[col], 'nhanes', label, opts))
        elif col in _NH_RENAME:
            rows.append((col, _NH_RENAME[col], '', 'nhanes', label, opts))
        else:
            rows.append((col, col, '', 'nhanes', label, opts))
    return rows


def _build_lifesnaps_rows():
    return [
        (orig, harm, comp, 'lifesnaps', qlabel, '')
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
        'composite_column', 'dataset', 'question_label', 'options_json',
    ])
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    df.to_csv(_OUT, index=False)
    print(f'Saved {len(df)} rows -> {_OUT}')


if __name__ == '__main__':
    build_map()
