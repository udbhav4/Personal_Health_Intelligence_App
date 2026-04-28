import os
import pandas as pd
import numpy as np
from utils import nan_audit, vif_from_corr_matrix

BASE = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'datasets')

NHANES_PATH = os.path.join(BASE, 'data_loaded_cleaned', 'nhanes_self', 'nhanes_merged_cleaned.csv')


def load_nhanes() -> pd.DataFrame:
    if not os.path.exists(NHANES_PATH):
        raise FileNotFoundError(f'NHANES file not found: {NHANES_PATH}')
    return pd.read_csv(NHANES_PATH)


# ---------------------------------------------------------------------------
# NHANES CLEANING
# ---------------------------------------------------------------------------

_NHANES_DPQ_COLS = [f'DPQ0{i}0' for i in range(1, 10)] + ['DPQ100']

# Columns with 7=Refused / 9=Don't know pattern
_NHANES_7_9_COLS = [
    'CDQ001', 'CDQ010',
    'HSD010',
    'DIQ010', 'DIQ160', 'DIQ050',
    'MCQ160A', 'MCQ160N', 'MCQ160C', 'MCQ160E', 'MCQ160M', 'MCQ160O',
    'MCQ520', 'MCQ080',
    'PAQ605', 'PAQ620',
]

# Columns with 99=Don't know (PAQ610 uses 99)
_NHANES_99_COLS = ['PAQ610']

# Columns with 9999=Don't know (PAD duration cols)
_NHANES_9999_COLS = ['PAD615', 'PAD630', 'PAD645', 'PAD660']

_NHANES_SENTINEL_MAP = {
    'PAD680':   [9999, 7777],
    'ALQ130':   [999,  777],
    'ALQ121':   [77,   99],
    'ALQ142':   [77,   99],
    'SMQ040':   [7,    9],
    'DMDMARTL': [77],
    'INDHHIN2': [77,   99],
    'WHD010':   [7777, 9999],
    'WHD020':   [7777, 9999],
    'SMD641':   [99],
    'SMD650':   [777,  999],
    'EDUCATION':[7,    9,   66],
}


def _replace_nhanes_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, codes in _NHANES_SENTINEL_MAP.items():
        if col in df.columns:
            df[col] = df[col].replace(codes, np.nan)
    for col in _NHANES_DPQ_COLS + _NHANES_7_9_COLS:
        if col in df.columns:
            df[col] = df[col].replace([7, 9], np.nan)
    for col in _NHANES_99_COLS:
        if col in df.columns:
            df[col] = df[col].replace([99], np.nan)
    for col in _NHANES_9999_COLS:
        if col in df.columns:
            df[col] = df[col].replace([9999], np.nan)
    return df


def _remap_education(df: pd.DataFrame) -> pd.DataFrame:
    """
    EDUCATION = DMDEDUC2.fillna(DMDEDUC3) from data_loading.
    DMDEDUC2 (adults 20+): 1-5 scale.
    DMDEDUC3 (youth 6-19): grade-level codes 0-15 — 18-19 year-olds carry these.
    Remap DMDEDUC3 grade codes to DMDEDUC2 scale so all rows share one scale.
      0-8  → 1  (less than 9th grade)
      9-12 → 2  (some high school)
      13   → 3  (HS graduate)
      14   → 3  (GED equivalent)
      15   → 4  (more than HS / some college)
    Values already in 1-5 range are untouched.
    """
    if 'EDUCATION' not in df.columns:
        return df
    df = df.copy()
    edu = df['EDUCATION']
    remap = {
        0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1,
        9: 2, 10: 2, 11: 2, 12: 2,
        13: 3, 14: 3,
        15: 4,
    }
    mask = edu > 5
    df.loc[mask, 'EDUCATION'] = edu[mask].map(remap)
    return df


def _nullify_impossible_nhanes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.loc[(df['BMI'] < 10) | (df['BMI'] > 80), 'BMI'] = np.nan
    df.loc[df['RIDAGEYR'] < 0, 'RIDAGEYR'] = np.nan
    return df


def _nullify_negatives(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        df.loc[df[col] < 0, col] = np.nan
    return df




def clean_nhanes(df: pd.DataFrame) -> pd.DataFrame:
    df = _replace_nhanes_sentinels(df)
    df = _remap_education(df)
    df = _nullify_impossible_nhanes(df)
    df = df[df['RIDAGEYR'] >= 18].reset_index(drop=True)
    df = _nullify_negatives(df)
    nan_audit(df, 'NHANES post-clean')
    return df


# ---------------------------------------------------------------------------
# NHANES — redundancy analysis + column pruning
# ---------------------------------------------------------------------------

# Node → column groups for statistical redundancy analysis.
# Excludes up front: SEQN (row ID), RIDRETH3 (ethnicity), INDHHIN2 (income),
# WHD010/WHD020 (raw weight/height) — scope decisions, no stats needed.
_NHANES_NODE_GROUPS = {
    'depression':        [f'DPQ0{i}0' for i in range(1, 10)] + ['DPQ100'],
    'alcohol_use':       ['ALQ111', 'ALQ121', 'ALQ130', 'ALQ142'],
    'smoking':           ['SMQ020', 'SMQ040', 'SMD641', 'SMD650'],
    'pain_level':        ['CDQ001', 'CDQ010'],
    'physical_exercise': ['PAQ605', 'PAQ610', 'PAD615',
                          'PAQ620', 'PAD630',
                          'PAQ635', 'PAD645',
                          'PAQ650', 'PAD660', 'PAD680'],
    'diabetes_status':   ['DIQ010', 'DIQ160', 'DIQ050'],
    'chronic_condition': ['MCQ160A', 'MCQ160N', 'MCQ160C', 'MCQ160E',
                          'MCQ160M', 'MCQ160O', 'MCQ520',  'MCQ080'],
    'general_health':    ['HSD010'],
    'bmi':               ['BMI'],
    'education_level':   ['EDUCATION'],
    'marital_status':    ['DMDMARTL'],
}

_NHANES_PHQ_ITEMS     = [f'DPQ0{i}0' for i in range(1, 10)]
_NHANES_CHRONIC_FLAGS = [
    'MCQ160A', 'MCQ160N', 'MCQ160C', 'MCQ160E',
    'MCQ160M', 'MCQ160O', 'MCQ520',  'MCQ080',
]

_CORR_FLAG  = 0.70   # flag for attention
_CORR_PRUNE = 0.85   # strong statistical redundancy candidate


def nhanes_redundancy_report(df: pd.DataFrame) -> None:
    """Print per-node Pearson r matrix + VIF for all NHANES node groups."""
    print(f'\n{"="*65}')
    print('NHANES Node Redundancy Report')
    print(f'Flag: r > {_CORR_FLAG}  |  Prune candidate: r > {_CORR_PRUNE}  |  High VIF: > 5')
    print(f'{"="*65}')
    for node, cols in _NHANES_NODE_GROUPS.items():
        present = [c for c in cols if c in df.columns]
        if len(present) < 2:
            print(f'\n[{node}] — {len(present)} col, skip')
            continue
        sub = df[present].apply(pd.to_numeric, errors='coerce')
        complete = sub.dropna()
        corr = sub.corr().abs()
        vifs = vif_from_corr_matrix(corr)
        print(f'\n[{node}]  ({len(present)} cols, {len(complete)} fully-complete rows)')
        print('  VIF:')
        for col, v in vifs.items():
            flag = '  *** HIGH' if isinstance(v, float) and v > 5 else ''
            print(f'    {col:<12}  {v}{flag}')
        print('  High-r pairs:')
        found = False
        for i, c1 in enumerate(present):
            for c2 in present[i + 1:]:
                r = corr.loc[c1, c2]
                if r > _CORR_FLAG:
                    tag = '[PRUNE CANDIDATE]' if r > _CORR_PRUNE else '[FLAG]'
                    print(f'    {c1} ~ {c2}:  r={r:.3f}  {tag}')
                    found = True
        if not found:
            print('    none above threshold')


def _merge_diabetes_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    DIQ010/DIQ160/DIQ050 → ordinal diabetes_status (0–3).
      0 = no diabetes, no prediabetes
      1 = prediabetes  (DIQ160=1, or DIQ010=3 borderline)
      2 = diabetes, no insulin
      3 = diabetes + insulin

    NHANES skip pattern: DIQ160 is not asked when DIQ010=1 (already diabetic),
    so DIQ160=NaN for all diabetics. Previous logic required both non-null →
    878 diabetics incorrectly became NaN. Fixed: diabetes branch checks only
    DIQ010; prediabetes/none branch checks DIQ010 + DIQ160.
    DIQ010=3 (borderline) treated as prediabetes.
    """
    s = pd.Series(np.nan, index=df.index)
    d10 = df['DIQ010']
    d16 = df['DIQ160']
    d50 = df['DIQ050']

    # Diabetes branch — DIQ160 is NaN here (skipped), don't require it
    s.loc[d10.notna() & (d10 == 1) & d50.notna() & (d50 == 2)] = 2
    s.loc[d10.notna() & (d10 == 1) & d50.notna() & (d50 == 1)] = 3

    # No-diabetes branch — DIQ160 is answered
    has_both = d10.notna() & d16.notna()
    s.loc[has_both & (d10 == 2) & (d16 == 2)] = 0   # no diabetes, no prediabetes
    s.loc[has_both & (d10 == 2) & (d16 == 1)] = 1   # prediabetes
    s.loc[has_both & (d10 == 3)]               = 1   # borderline = prediabetes

    df = df.copy()
    df['diabetes_status'] = s
    return df.drop(columns=[c for c in ['DIQ010', 'DIQ160', 'DIQ050'] if c in df.columns])


def _merge_chronic_condition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Eight MCQ binary flags → binary chronic_condition (0/1).
    Logical merge: individual conditions share no meaningful stratum with age/sex
    for prior computation. Union = 'has any chronic condition' is the relevant signal.
    Stats: all pairwise r < 0.7 — conditions are largely independent comorbidities.
    """
    present = [c for c in _NHANES_CHRONIC_FLAGS if c in df.columns]
    flags = df[present].replace({2: 0})   # NHANES: 1=Yes, 2=No → 0
    df = df.copy()
    df['chronic_condition'] = flags.max(axis=1)
    return df.drop(columns=present)


def _merge_phq_total(df: pd.DataFrame) -> pd.DataFrame:
    """
    DPQ010–DPQ090 (PHQ-9 items) → phq_total (0–27). DPQ100 kept as second
    depression-node column (functional impairment from symptoms).
    Logical merge: 9 symptom items designed for summation into total score.
    Stats: all pairwise r < 0.7 — items cover distinct symptom domains.
    """
    present = [c for c in _NHANES_PHQ_ITEMS if c in df.columns]
    df = df.copy()
    df['phq_total'] = df[present].sum(axis=1, min_count=1)
    return df.drop(columns=present)


def prune_nhanes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run redundancy report, then apply drops and merges.

    Statistical drops (evidence in redundancy_report output):
      SMD641 ~ SMD650: r=0.816 on cleaned data → same construct (cigs/day),
        different recall windows. Keep SMD650 (standard 30-day window).
      PAQ605 ~ PAQ610: r=0.762 → vigorous-work gate subsumed by count col.
      PAQ635, PAQ650: gate cols whose duration pairs (PAD645, PAD660)
        carry the informative signal; gates add no independent variance.
      SMQ020, ALQ111: ever-smoked / ever-drank gates subsumed by
        SMQ040 (current status) and ALQ121 (frequency).

    Logical merges (clinical rationale, not statistically redundant):
      diabetes_status, chronic_condition, phq_total — see helpers above.

    Scope drops (user decision, no stats required):
      SEQN, RIDRETH3, INDHHIN2, WHD010, WHD020.
    """
    nhanes_redundancy_report(df)

    df = _merge_diabetes_status(df)
    df = _merge_chronic_condition(df)
    df = _merge_phq_total(df)

    # PAQ620 added: binary gate (Yes=1/No=2), same type as PAQ605/PAQ635/PAQ650 (all dropped).
    stat_drops  = ['SMD641', 'PAQ605', 'PAQ620', 'PAQ635', 'PAQ650', 'SMQ020', 'ALQ111']
    scope_drops = ['SEQN', 'RIDRETH3', 'INDHHIN2', 'WHD010', 'WHD020']
    df = df.drop(columns=[c for c in stat_drops + scope_drops if c in df.columns])

    nan_audit(df, 'NHANES post-prune')
    return df


# ---------------------------------------------------------------------------
# NHANES — final column labels map
# ---------------------------------------------------------------------------

_NHANES_LABELS = [
    # column, node, origin, description, raw_scale, state_labels
    ('RIDAGEYR',          'age',               'raw',    'Age in years (adults 18+ kept)',
     'continuous 18–85+',
     'continuous — no discretization'),

    ('RIAGENDR',          'sex',               'raw',    'Biological sex',
     '1=Male, 2=Female',
     '1=male | 2=female'),

    ('BMI',               'bmi',               'raw',    'Body Mass Index (kg/m²)',
     'continuous 10–80',
     '<18.5=underweight | 18.5–24.9=normal | 25–29.9=overweight | ≥30=obese'),

    ('EDUCATION',         'education_level',   'raw',    'Education level (DMDEDUC2 for 20+; DMDEDUC3 grade codes remapped for 18–19 yr olds)',
     '1=<9th grade | 2=9–11th grade | 3=HS grad/GED | 4=some college/AA | 5=college grad',
     '1–2=less_than_HS | 3=HS_grad | 4=some_college | 5=college_grad'),

    ('DMDMARTL',          'marital_status',    'raw',    'Marital status',
     '1=Married | 2=Widowed | 3=Divorced | 4=Separated | 5=Never married | 6=Living with partner',
     'married_partnered (1 or 6) | single_separated_divorced_widowed (2–5)'),

    ('HSD010',            'general_health',    'raw',    'Self-reported general health status',
     '1=Excellent | 2=Very Good | 3=Good | 4=Fair | 5=Poor',
     '1=excellent | 2=very_good | 3=good | 4=fair | 5=poor'),

    ('phq_total',         'depression',        'merged', 'PHQ-9 total score — sum of DPQ010–DPQ090; DPQ100 kept separately',
     '0–27 (sum of 9 items, each 0–3)',
     '0–4=none | 5–9=mild | 10–14=moderate | 15–19=mod_severe | 20–27=severe'),

    ('DPQ100',            'depression',        'raw',    'Functional impairment from depressive symptoms (PHQ item 10)',
     '0=Not at all | 1=Several days | 2=More than half the days | 3=Nearly every day',
     '0=none | 1=several_days | 2=more_than_half | 3=nearly_every_day'),

    ('diabetes_status',   'diabetes_status',   'merged', 'Ordinal diabetes status — merged from DIQ010, DIQ160, DIQ050',
     '0=none | 1=prediabetes/borderline | 2=diabetes no insulin | 3=diabetes+insulin',
     '0=none | 1=prediabetes | 2=diabetes | 3=diabetes_insulin'),

    ('chronic_condition', 'chronic_condition', 'merged', 'Binary chronic condition flag — union of 8 MCQ flags (arthritis, stroke, CHD, heart failure, angina, other heart, COPD, obesity-related)',
     '0=no chronic condition | 1=has at least one',
     '0=no | 1=yes'),

    ('CDQ001',            'pain_level',        'raw',    'Chest pain/tightness/discomfort on exertion',
     '1=Yes | 2=No',
     '1=yes | 2=no'),

    ('CDQ010',            'pain_level',        'raw',    'Shortness of breath on level ground',
     '1=Yes | 2=No',
     '1=yes | 2=no'),

    ('ALQ121',            'alcohol_use',       'raw',    'How often drank alcohol in past 12 months',
     '0=Never | 1=Every day | 2=Nearly every day | 3=3–4×/wk | 4=2×/wk | 5=1×/wk | 6=2–3×/mo | 7=1×/mo | 8=7–11×/yr | 9=3–6×/yr | 10=1–2×/yr',
     'k-means discretization (k=3 or 4)'),

    ('ALQ130',            'alcohol_use',       'raw',    'Avg drinks per day on drinking days (past 12 months)',
     '1–15+ (15=15 or more)',
     'k-means discretization (k=3 or 4)'),

    ('ALQ142',            'alcohol_use',       'raw',    'Drinks per day on drinking days in past 30 days',
     '1–15+ (15=15 or more)',
     'k-means discretization (k=3 or 4)'),

    ('SMQ040',            'smoking',           'raw',    'Current cigarette smoking status (never-smokers assigned state 3 in data_loading.py)',
     '1=Every day | 2=Some days | 3=Not at all',
     '1=daily | 2=some_days | 3=not_at_all'),

    ('SMD650',            'smoking',           'raw',    'Avg cigarettes per day in past 30 days',
     '1–95',
     'k-means discretization (k=3 or 4)'),

    ('PAQ610',            'physical_exercise', 'raw',    'Days/week with vigorous work-related physical activity',
     '0=None | 1–7=days per week',
     'k-means discretization (k=3 or 4)'),

    ('PAD615',            'physical_exercise', 'raw',    'Minutes of vigorous work-related activity per day',
     'continuous minutes',
     'k-means discretization (k=3 or 4)'),

    ('PAD630',            'physical_exercise', 'raw',    'Minutes of moderate work-related activity per day',
     'continuous minutes',
     'k-means discretization (k=3 or 4)'),

    ('PAD645',            'physical_exercise', 'raw',    'Minutes of walking or bicycling per day',
     'continuous minutes',
     'k-means discretization (k=3 or 4)'),

    ('PAD660',            'physical_exercise', 'raw',    'Minutes of vigorous recreational activity per day',
     'continuous minutes',
     'k-means discretization (k=3 or 4)'),

    ('PAD680',            'physical_exercise', 'raw',    'Minutes of sedentary activity per day (sitting/lying excl. sleep)',
     'continuous minutes',
     'k-means discretization (k=3 or 4)'),
]

_NHANES_LABELS_COLS = ['column', 'node', 'origin', 'description', 'raw_scale', 'state_labels']

NHANES_LABELS_PATH = os.path.join(BASE, 'data_loaded_cleaned', 'nhanes_self', 'nhanes_final_labels.csv')


def save_nhanes_labels() -> pd.DataFrame:
    df = pd.DataFrame(_NHANES_LABELS, columns=_NHANES_LABELS_COLS)
    df.to_csv(NHANES_LABELS_PATH, index=False)
    print(f'Saved {len(df)} column labels → {NHANES_LABELS_PATH}')
    return df

