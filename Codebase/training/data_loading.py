import pandas as pd
import numpy as np
import pyreadstat
import os
import re
import json


BASE             = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NHANES_PATH      = os.path.join(BASE, 'datasets', 'NHANES')
STUDENTLIFE_PATH = os.path.join(BASE, 'datasets', 'studentlife', 'dataset')
LIFESNAPS_PATH   = os.path.join(BASE, 'datasets', 'lifesnaps', 'rais_anonymized', 'csv_rais_anonymized')

# ── ROLLING WINDOW CONFIG ─────────────────────────────────────────────────────
# W controls the granularity of all sensor aggregations.
# Valid values (minutes): 15, 60, 360, 1440
# 1440 = daily (reproduces original row count with one extra window_start column)
# Change this one value to switch the entire pipeline to a different granularity.
WINDOW_MINUTES = 60

# EMA definition JSON — maps question_id to question_text for all EMA folders.
EMA_DEF_PATH = r'C:\Users\udbha\Documents\VS Code\MedApp\datasets\studentlife\dataset\EMA\EMA_definition.json'


## HELPER FUNCTIONS ##

def extract_uid(filename):
    # Extracts participant ID (e.g. u00, u01) from filename.
    match = re.search(r'(u\d+)', filename)
    return match.group(1) if match else None

def unix_to_date(ts):
    # Converts unix timestamp (seconds) to date object.
    try:
        return pd.to_datetime(float(ts), unit='s').date()
    except:
        return np.nan

def parse_numeric(val):
    # Handles messy numeric entries like "6-8 hours", "approx 5", "6", etc.
    if pd.isna(val):
        return np.nan
    val = str(val).lower().strip()
    val = re.sub(r'[a-zA-Z]', '', val).strip()
    # For ranges like "6-8" - take mean.
    if '-' in val:
        parts = val.split('-')
        try:
            return np.mean([float(p.strip()) for p in parts if p.strip()])
        except:
            return np.nan
    try:
        return float(val)
    except:
        return np.nan

def deduplicate_intervals(group, start_col, end_col):
    # Merges overlapping time intervals and returns total non-overlapping minutes.
    # Prevents double-counting when sensor logs overlapping dark/lock periods.
    intervals = sorted(zip(group[start_col].values, group[end_col].values))
    merged = []
    for start, end in intervals:
        if merged and start < merged[-1][1]:
            merged[-1] = [merged[-1][0], max(merged[-1][1], end)]
        else:
            merged.append([start, end])
    total_mins = sum((e - s) / 60 for s, e in merged)
    # Safety cap — total cannot exceed minutes in a day
    return min(total_mins, 1440)

def clip_intervals_to_window(intervals_unix, window_start_unix, window_end_unix):
    """
    Clips a list of (start, end) unix-second intervals to a specific time window,
    deduplicates overlaps within that window, and returns total minutes covered.

    Works for any window size — 15 min, 60 min, 6 hours, or 1 day.
    Used for dark sensing and phonelock sensing (interval-based sensors).

    Args:
        intervals_unix    : list of (start_unix, end_unix) tuples in seconds
        window_start_unix : window open boundary in unix seconds (inclusive)
        window_end_unix   : window close boundary in unix seconds (exclusive)

    Returns:
        float: total non-overlapping minutes covered within [window_start, window_end)
    """

    # Checks whether there is any sensor interval that lies within this specific window. If not, returns 0 immediately to avoid unnecessary processing.
    clipped = []
    for s, e in intervals_unix:
        cs = max(s, window_start_unix)
        ce = min(e, window_end_unix)
        if ce > cs:
            clipped.append((cs, ce))

    if not clipped:
        return 0.0

    # This part makes sure that within a window, there is no overlap between the sensor intervals that would cause double-counting.
    # For example, if a phone is locked from 1:50-2:10 and 2:05-2:20, the total locked time in the 1:00-2:00 window should be 10 minutes (1:50-2:00), not 15 minutes.
    clipped.sort()
    merged = [list(clipped[0])] # mergred is a list which is initiated using the first interval in clipped.
    for s,e in clipped[1:]:
        if s < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    total_secs = sum((e - s) for s, e in merged)
    window_secs = window_end_unix - window_start_unix
    return min(total_secs, window_secs) / 60.0


def get_window_slots(date_obj, window_minutes):
    """
    Generates all W-minute window boundaries for a given calendar date.
    Windows are FIXED and anchored to midnight — they never shift based on data.

    Returns a list of (window_start_unix, window_end_unix, window_start_dt) tuples.

    For window_minutes=1440 : 1 slot  (00:00 → midnight next day)
    For window_minutes=60   : 24 slots (00:00–01:00, 01:00–02:00, ...)
    For window_minutes=15   : 96 slots (00:00–00:15, 00:15–00:30, ...)

    window_start_dt is a pandas Timestamp (UTC-consistent) — becomes the 'window_start' column.
    Uses UTC-consistent midnight arithmetic via pd.Timestamp to match date extraction
    throughout the pipeline (which also uses unit='s' UTC-based conversion).
    """
    base_unix = int(pd.Timestamp(date_obj).value // 10**9)  # UTC midnight unix seconds
    n_slots = 1440 // window_minutes
    slots = []
    for i in range(n_slots):
        w_start_unix = base_unix + i * window_minutes * 60
        w_end_unix   = w_start_unix + window_minutes * 60
        w_start_dt   = pd.Timestamp(w_start_unix, unit='s')
        slots.append((w_start_unix, w_end_unix, w_start_dt))
    return slots


def load_per_user_json(folder_path, keys_keep):
    # Loads per-user JSON files from a folder, keeps only specified keys.
    dfs = []
    for fname in os.listdir(folder_path):
        if not fname.endswith('.json'):
            continue
        uid = extract_uid(fname)
        if uid is None:
            continue
        fpath = os.path.join(folder_path, fname)
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                if any(isinstance(v, list) for v in data.values()):
                    df = pd.DataFrame(data)
                else:
                    df = pd.DataFrame([data])
            existing = [k for k in keys_keep if k in df.columns]
            if not existing:
                continue
            df = df[existing].copy()
            df['uid'] = uid
            dfs.append(df)
        except Exception as e:
            print(f"  Error loading {fname}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def build_ema_label_map(ema_def_path):
    # Builds {question_id: question_text} from EMA_definition.json.
    # Skips location fields. Stores qid as fallback if question_text is missing.
    try:
        with open(ema_def_path, 'r') as f:
            ema_def = json.load(f)
        label_map = {}
        for folder in ema_def:
            for q in folder['questions']:
                qid   = q.get('question_id', '')
                qtext = q.get('question_text', '').strip()
                if qid and qid != 'location':
                    label_map[qid] = qtext if qtext else f'EMA question_id: {qid}'
        return label_map
    except Exception as e:
        print(f"  Warning: could not load EMA definition JSON: {e}")
        return {}
# -----------------------------------------------------------------------------


## NHANES CLEANING & HARMONIZATION ##

# All useful columns from NHANES.
# WHD010 (height inches) and WHD020 (weight pounds) kept alongside computed BMI.
# At runtime the user is asked height and weight separately; BMI is computed from those answers.
COLUMNS = {
    'demographic':                       ['SEQN','RIAGENDR','RIDAGEYR','RIDRETH3','DMDEDUC2','DMDEDUC3','DMDMARTL','INDHHIN2'],
    'smoking_use':                       ['SEQN','SMQ020','SMQ040','SMD641','SMD650'],
    'alcohol_use':                       ['SEQN','ALQ111','ALQ121','ALQ130','ALQ142'],
    'cardiovascular':                    ['SEQN','CDQ001','CDQ010'],
    'current_health_status':             ['SEQN','HSD010'],
    'diabetes':                          ['SEQN','DIQ010','DIQ160','DIQ050'],
    'medical_conditions':                ['SEQN','MCQ160A','MCQ160N','MCQ160C','MCQ160E','MCQ160M','MCQ160O','MCQ520','MCQ080'],
    'mental_health_depression_screener': ['SEQN','DPQ010','DPQ020','DPQ030','DPQ040','DPQ050','DPQ060','DPQ070','DPQ080','DPQ090','DPQ100'],
    'physical_activity':                 ['SEQN','PAQ605','PAQ610','PAD615','PAQ620','PAD630','PAQ635','PAD645','PAQ650','PAD660','PAD680'],
    'weight_history':                    ['SEQN','WHD010','WHD020'],
}


def merge_nhanes():
    merged = None
    all_labels = {}

    for fname, cols in COLUMNS.items():
        path = os.path.join(NHANES_PATH, f'{fname}.xpt')
        df, meta = pyreadstat.read_xport(path)
        for col in cols:
            if col in meta.column_names_to_labels:
                all_labels[col] = meta.column_names_to_labels[col]
        df = df[cols]
        merged = df if merged is None else merged.merge(df, on='SEQN', how='outer')
        print(f"Loaded {fname}: {df.shape}")

    print(f"\nFinal merged shape: {merged.shape}")
    return merged, all_labels


def clean_nhanes(df, all_labels):

    # Merge education columns
    df['EDUCATION'] = df['DMDEDUC2'].fillna(df['DMDEDUC3'])
    df.drop(columns=['DMDEDUC2','DMDEDUC3'], inplace=True)
    all_labels['EDUCATION'] = 'Education level — merged from DMDEDUC2 (adults 20+) and DMDEDUC3 (youth 6-19)'
    print(f"Education columns merged into EDUCATION.")

    # Under-20 marital status = 0 (not applicable for minors)
    df['DMDMARTL'] = df.apply(
        lambda row: 0 if pd.isna(row['DMDMARTL']) and row['RIDAGEYR'] < 20
        else row['DMDMARTL'], axis=1
    )
    print(f"Under-20 marital status filled with 0.")

    # INDHHIN2 is integer category 1-12 representing USD annual income ranges — not a raw dollar value.
    all_labels['INDHHIN2'] = 'Annual household income — integer category 1-12 representing USD ranges (1=under $5,000, 12=$100,000+)'

    # Condition-based zero filling for conditional skip columns.
    # Only fill 0 when the parent question is explicitly answered as No (coded as 2 in NHANES).
    # If parent is NaN (not asked), child stays NaN for EM — do NOT fill 0 in that case.
    df.loc[df['SMQ020'] == 2, ['SMQ040','SMD641','SMD650']] = 0   # never smoked
    df.loc[df['SMQ040'] == 3, ['SMD641','SMD650']]          = 0   # smokes not at all currently
    df.loc[df['ALQ111'] == 2, ['ALQ121','ALQ130','ALQ142']] = 0   # never had a drink
    df.loc[df['PAQ605'] == 2, ['PAQ610','PAD615']]          = 0   # no vigorous work activity
    df.loc[df['PAQ620'] == 2, ['PAD630']]                   = 0   # no moderate work activity
    df.loc[df['PAQ635'] == 2, ['PAD645']]                   = 0   # no walking or cycling
    df.loc[df['PAQ650'] == 2, ['PAD660']]                   = 0   # no vigorous recreational activity
    print(f"Conditional skip columns filled with 0 only where parent confirmed No (coded as 2).")

    # Compute BMI from height (inches) and weight (pounds).
    # WHD010 and WHD020 retained — at runtime user is asked height and weight separately.
    # BMI is computed from those answers. Do not ask BMI directly.
    df['BMI'] = (df['WHD020'] * 0.453592) / ((df['WHD010'] * 0.0254) ** 2)
    all_labels['BMI']    = 'Body Mass Index — computed as weight(kg)/height(m)^2. Ask height and weight separately at runtime.'
    all_labels['WHD010'] = 'Self-reported height in inches. Ask user: What is your height?'
    all_labels['WHD020'] = 'Self-reported weight in pounds. Ask user: What is your weight?'
    print(f"BMI computed. WHD010 and WHD020 retained alongside BMI.")

    # Leave all other NaNs for the EM algorithm.
    remaining = df.isnull().sum()
    remaining_pct = (remaining / len(df) * 100).round(1)
    report = pd.DataFrame({'remaining_count': remaining, 'remaining_pct': remaining_pct})
    print(report[report['remaining_count'] > 0].to_string())

    labels_df = pd.DataFrame(list(all_labels.items()), columns=['column','label'])
    labels_df.to_csv(r'C:\Users\udbha\Documents\VS Code\MedApp\datasets\nhanes_self\nhanes_column_labels.csv', index=False)

    df.to_csv(r'C:\Users\udbha\Documents\VS Code\MedApp\datasets\nhanes_self\nhanes_merged_cleaned.csv', index=False)
    print(f"\nSaved nhanes_merged_cleaned.csv | Shape: {df.shape}")
    return df
# -----------------------------------------------------------------------------


## STUDENTLIFE CLEANING AND HARMONIZATION ##

# ── SURVEY STRING → INT ENCODINGS ─────────────────────────────────────────────
# Starts from 1:  BigFive, LonelinessScale, VR-12
# Starts from 0:  PHQ-9, PerceivedStressScale, PSQI
# Already numeric: PANAS

BIGFIVE_MAP = {
    "Disagree Strongly":          1,
    "Disagree a little":          2,
    "Neither agree nor disagree": 3,
    "Agree a little":             4,
    "Agree strongly":             5,
}

LONELINESS_MAP = {
    "Never":     1,
    "Rarely":    2,
    "Sometimes": 3,
    "Often":     4,
}

PHQ9_ITEMS_MAP = {
    "Not at all":              0,
    "Several days":            1,
    "More than half the days": 2,
    "Nearly every day":        3,
}
PHQ9_RESPONSE_MAP = {
    "Not difficult at all": 0,
    "Somewhat difficult":   1,
    "Very difficult":       2,
    "Extremely difficult":  3,
}

PSS_MAP = {
    "Never":        0,
    "Almost never": 1,
    "Sometime":     2,
    "Fairly often": 3,
    "Very often":   4,
}

PSQI_FREQ_MAP = {
    "Not during the past month":  0,
    "Less than once week":        1,
    "Once or a twice week":       2,
    "Three or a more times week": 3,
}
PSQI_QUALITY_MAP = {
    "Very good":   0,
    "Fairly good": 1,
    "Fairly bad":  2,
    "Very bad":    3,
}

# VR-12: different sub-scales per column — applied positionally.
# "Some of the time" means 4 on the 6-pt MH scale but 3 on the 5-pt social scale,
# so a global replace would be wrong — each column gets its own dict.
VR12_HEALTH_MAP  = {"Excellent": 1, "Very good": 2, "Good": 3, "Fair": 4, "Poor": 5}
VR12_LIMIT_MAP   = {"Yes, limited a lot": 1, "Yes, limited a little": 2, "No, not limited at all": 3}
VR12_ROLE_MAP    = {
    "Yes, all of the time": 1, "Yes, most of the time": 2, "Yes, some of the time": 3,
    "Yes, a little of the time": 4, "No, none of the time": 5,
}
VR12_PAIN_MAP    = {"Not at all": 1, "A little bit": 2, "Moderately": 3, "Quite a bit": 4, "Extremely": 5}
VR12_MH6_MAP     = {
    "All of the time": 1, "Most of the time": 2, "A good bit of the time": 3,
    "Some of the time": 4, "A little of the time": 5, "None of the time": 6,
}
VR12_SOCIAL5_MAP = {
    "All of the time": 1, "Most of the time": 2, "Some of the time": 3,
    "A little of the time": 4, "None of the time": 5,
}
VR12_CHANGE_MAP  = {
    "Much better": 1, "Slightly better": 2, "About the same": 3, "Slightly worse": 4, "Much worse": 5,
}

# PSQI free-text columns — left as raw strings (handled by parse_numeric elsewhere)
_PSQI_FREETEXT = {
    "During the past month, what time have you usually gone to bed at night? ",
    "During the past month, how long (in minutes) has it usually taken you to fall asleep each night?",
    "When have you usually gotten up in the morning?",
    "During the past month, how many hours of actual sleep did you get at night? (This may be different than the number of hours you spent in bed.)",
    "Other reason(s), please describe, including how often you have had trouble sleeping because of this reason(s):",
}
_PSQI_QUALITY_COL = "During the past month, how would you rate your sleep quality overall?"

# VR-12 columns in CSV order (positional after uid/type)
_VR12_COL_MAPS = [
    VR12_HEALTH_MAP,   # general health
    VR12_LIMIT_MAP,    # moderate activity
    VR12_LIMIT_MAP,    # climb stairs
    VR12_ROLE_MAP,     # accomplished less (physical)
    VR12_ROLE_MAP,     # limited in kind (physical)
    VR12_ROLE_MAP,     # accomplished less (emotional)
    VR12_ROLE_MAP,     # didn't do carefully (emotional)
    VR12_PAIN_MAP,     # pain interference
    VR12_MH6_MAP,      # calm and peaceful (6-pt)
    VR12_MH6_MAP,      # a lot of energy (6-pt)
    VR12_MH6_MAP,      # downhearted and blue (6-pt)
    VR12_SOCIAL5_MAP,  # social activities interference (5-pt)
    VR12_CHANGE_MAP,   # physical health vs last year
    VR12_CHANGE_MAP,   # emotional problems vs last year
]


def encode_survey_strings(survey_dir):
    """
    Loads all 8 StudentLife survey CSVs and encodes string responses to integers.
    uid and type columns are preserved. FlourishingScale and PANAS are already
    numeric and returned unchanged. PSQI free-text columns are left as-is.

    Returns dict[str, pd.DataFrame] keyed by survey name.
    """
    p    = lambda f: os.path.join(survey_dir, f)
    meta = ('uid', 'type')
    out  = {}

    # BigFive — 1-5
    df = pd.read_csv(p('BigFive.csv'))
    cols = [c for c in df.columns if c not in meta]
    df[cols] = df[cols].replace(BIGFIVE_MAP)
    out['BigFive'] = df

    # FlourishingScale — already numeric 1-7
    out['FlourishingScale'] = pd.read_csv(p('FlourishingScale.csv'))

    # LonelinessScale — 1-4
    df = pd.read_csv(p('LonelinessScale.csv'))
    cols = [c for c in df.columns if c not in meta]
    df[cols] = df[cols].replace(LONELINESS_MAP)
    out['LonelinessScale'] = df

    # PHQ-9 — items 0-3, Response 0-3
    df = pd.read_csv(p('PHQ-9.csv'))
    item_cols = [c for c in df.columns if c not in (*meta, 'Response')]
    df[item_cols]  = df[item_cols].replace(PHQ9_ITEMS_MAP)
    df['Response'] = df['Response'].replace(PHQ9_RESPONSE_MAP)
    out['PHQ-9'] = df

    # PSS — 0-4
    df = pd.read_csv(p('PerceivedStressScale.csv'))
    cols = [c for c in df.columns if c not in meta]
    df[cols] = df[cols].replace(PSS_MAP)
    out['PerceivedStressScale'] = df

    # PANAS — already numeric 1-5
    out['panas'] = pd.read_csv(p('panas.csv'))

    # PSQI — frequency 0-3, sleep quality 0-3, free-text left as-is
    df = pd.read_csv(p('psqi.csv'))
    freq_cols = [c for c in df.columns if c not in meta
                 and c not in _PSQI_FREETEXT and c != _PSQI_QUALITY_COL]
    df[freq_cols]         = df[freq_cols].replace(PSQI_FREQ_MAP)
    df[_PSQI_QUALITY_COL] = df[_PSQI_QUALITY_COL].replace(PSQI_QUALITY_MAP)
    out['psqi'] = df

    # VR-12 — column-specific mappings, all start from 1
    df = pd.read_csv(p('vr_12.csv'))
    data_cols = [c for c in df.columns if c not in meta]
    for col, mapping in zip(data_cols, _VR12_COL_MAPS):
        df[col] = df[col].replace(mapping)
    out['vr_12'] = df

    return out


def load_surveys(survey_path):
    print("\nLoading surveys...")

    # survey_labels: {renamed_column: original_question_text}
    # Used for label CSV and for deciding sub-dimension attribute node names.
    # All items restored — EM decides CPT redundancy; sub-dimension layer decides runtime usage.
    survey_labels = {}

    encoded = encode_survey_strings(survey_path)

    # PHQ-9: all 9 items kept. phq_total removed — it is a perfect linear combination of the 9
    # items and causes multicollinearity in HillClimbSearch BIC scoring. Clinical threshold
    # (PHQ >= 10) is validated separately after training, not during it.
    # PHQ items scored 0-3 (not at all → nearly every day) — higher = worse. No reversal needed.
    # Each item becomes a sub-dimension attribute at runtime for pointed questions.
    # phq = pd.read_csv(os.path.join(survey_path, 'PHQ-9.csv'))
    phq = encoded['PHQ-9']
    phq_rename = {
        'Little interest or pleasure in doing things':                                                                                                                                           'phq_interest',
        'Feeling down, depressed, hopeless.':                                                                                                                                                    'phq_depressed',
        'Trouble falling or staying asleep, or sleeping too much.':                                                                                                                              'phq_sleep',
        'Feeling tired or having little energy':                                                                                                                                                 'phq_tired',
        'Poor appetite or overeating':                                                                                                                                                           'phq_appetite',
        'Feeling bad about yourself or that you are a failure or have let yourself or your family down':                                                                                         'phq_failure',
        'Trouble concentrating on things, such as reading the newspaper or watching television':                                                                                                 'phq_concentrate',
        'Moving or speaking so slowly that other people could have noticed. Or the opposite being so figety or restless that you have been moving around a lot more than usual':                 'phq_psychomotor',
        'Thoughts that you would be better off dead, or of hurting yourself':                                                                                                                    'phq_death',
    }
    phq = phq.rename(columns=phq_rename)
    phq = phq[['uid'] + list(phq_rename.values())].groupby('uid').mean().reset_index()
    survey_labels.update({v: k for k, v in phq_rename.items()})
    print(f"  PHQ-9: {phq.shape}")

    # PSS: all 10 items kept. pss_total removed — same multicollinearity reason as phq_total.
    # PSS items scored 0-4 (never → very often).
    # Items 4, 5, 7, 8 are positively worded — higher score = LESS stressed. Must reverse.
    # Reversal formula on 0-4 scale: reversed = 4 - value.
    # After reversal all items consistently read: higher = more stressed.
    # pss = pd.read_csv(os.path.join(survey_path, 'PerceivedStressScale.csv'))
    pss = encoded['PerceivedStressScale']
    non_meta   = [c for c in pss.columns if c not in ['uid','type']]
    pss_rename = {col: f'pss_{i+1}' for i, col in enumerate(non_meta)}
    pss = pss.rename(columns=pss_rename)
    # Reverse score positively worded items (4, 5, 7, 8) on 0-4 scale
    for col in ['pss_4','pss_5','pss_7','pss_8']:
        if col in pss.columns:
            pss[col] = 4 - pss[col]
    pss_item_cols = list(pss_rename.values())
    pss = pss[['uid'] + pss_item_cols].groupby('uid').mean().reset_index()
    survey_labels.update({f'pss_{i+1}': col for i, col in enumerate(non_meta)})
    print(f"  PSS: {pss.shape}")

    # PSQI: sleep quality items.
    # Removed: sleep_too_cold (f), sleep_too_hot (g) — environmental factors, not health signals.
    # Kept: sleep_cough_snore (e) — respiratory/airway signal; shapes physical_stress + sleep_quality CPTs.
    # Kept: sleep_bad_dreams (h) — distinct mental_stress signal not elsewhere in dataset.
    # psqi = pd.read_csv(os.path.join(survey_path, 'psqi.csv'))
    psqi = encoded['psqi']
    psqi_rename = {
        'During the past month, how many hours of actual sleep did you get at night? (This may be different than the number of hours you spent in bed.)': 'sleep_hours',
        'During the past month, how would you rate your sleep quality overall?':                                                                          'sleep_quality_rating',
        'During the past month, how long (in minutes) has it usually taken you to fall asleep each night?':                                               'sleep_latency_mins',
        'a. Cannot get to sleep within 30 minutes':                                                                                                       'sleep_trouble_30min',
        'b. Wake up in the middle of the night or early morning':                                                                                         'sleep_wakeup',
        'e. Cough or snore loudly':                                                                                                                       'sleep_cough_snore',
        'h. Have bad dreams':                                                                                                                             'sleep_bad_dreams',
        'i. Have pain':                                                                                                                                   'pain_during_sleep',
        'During the past month, how much of a problem has it been for you to keep up enthusiasm to get things done?':                                     'low_enthusiasm'
    }
    psqi = psqi.rename(columns=psqi_rename)

    for col in ['sleep_hours','sleep_latency_mins']:
        if col in psqi.columns:
            psqi[col] = psqi[col].apply(parse_numeric)
    existing_psqi_cols = [c for c in psqi_rename.values() if c in psqi.columns]
    psqi = psqi[['uid'] + existing_psqi_cols].groupby('uid').mean().reset_index()
    survey_labels.update({v: k for k, v in psqi_rename.items()})
    print(f"  PSQI: {psqi.shape}")

    # PANAS: 14 items kept (removed Proud, Determined, Guilty, Hostile — no DBN node mapping).
    # All other 14 items map to at least one DBN node dimension and can form pointed questions.
    # EM decides CPT redundancy. Sub-dimension layer decides which to ask at runtime.
    # Two-factor structure (PA/NA) respected — both poles covered without bias.
    # panas = pd.read_csv(os.path.join(survey_path, 'panas.csv'))
    panas = encoded['panas']
    panas.columns = panas.columns.str.strip()
    panas_rename = {
        'Interested':   'panas_interested',    # engagement/anhedonia — phq_interest dimension
        'Distressed':   'panas_distressed',    # mental distress — mental_stress node
        'Upset':        'panas_upset',         # negative affect — mood node
        'Strong':       'panas_strong',        # physical vigor — energy_level node
        'Scared':       'panas_scared',        # fear/anxiety — mental_stress node
        'Enthusiastic': 'panas_enthusiastic',  # positive energy — mood + energy_level
        'Active':       'panas_active',        # behavioural energy — energy_level + physical_exercise
        'Irritable':    'panas_irritable',     # mood negative — mood node
        'Alert':        'panas_alert',         # cognitive readiness — energy_level node
        'Inspired':     'panas_inspired',      # positive engagement — mood node
        'Nervous':      'panas_nervous',       # anxiety — mental_stress node
        'Attentive':    'panas_attentive',     # focus — energy_level node
        'Jittery':      'panas_jittery',       # anxious arousal — mental_stress node
        'Afraid':       'panas_afraid',        # fear — mental_stress node
        'Proud':        'panas_proud',         # pride — mental_stress node
        'Determined':    'panas_determined',    # determination — mood node
        'Guilty':       'panas_guilty',        # guilt — mental_stress node
        'Hostile':      'panas_hostile',       # hostility — mood node
    }
    # Clean up trailing space variant
    panas_rename = {k.strip(): v for k, v in panas_rename.items()}
    existing_panas = {k: v for k, v in panas_rename.items() if k in panas.columns}
    panas = panas.rename(columns=existing_panas)
    kept_panas_cols = list(dict.fromkeys([v for v in existing_panas.values() if v in panas.columns]))
    panas = panas[['uid'] + kept_panas_cols].groupby('uid').mean().reset_index()
    survey_labels.update({v: k for k, v in existing_panas.items()})
    print(f"  PANAS: {panas.shape}")

    # LONELINESS: all 20 items kept.
    # UCLA Loneliness Scale has three validated sub-dimensions:
    # intimate (close relationships), relational (friendships), collective (community belonging).
    # All 20 items kept — EM discovers factor structure; sub-dimension layer aggregates into
    # lonely_intimate, lonely_relational, lonely_collective attributes at runtime.
    # lonely = pd.read_csv(os.path.join(survey_path, 'LonelinessScale.csv'))
    lonely = encoded['LonelinessScale']
    lonely_cols = [c for c in lonely.columns if c not in ['uid','type']]
    lonely_rename = {col: f'lonely_{i+1}' for i, col in enumerate(lonely_cols)}
    lonely = lonely.rename(columns=lonely_rename)
    lonely = lonely[['uid'] + list(lonely_rename.values())].groupby('uid').mean().reset_index()
    # UCLA Loneliness Scale — positively worded items need reversal on 1-4 scale (reversed = 5 - value).
    # Positively worded items (higher original score = LESS lonely — must flip):
    # Items 1,4,5,6,9,10,15,16,19,20 based on standard UCLA-R scoring manual.
    # After reversal: higher = more lonely consistently across all 20 items.
    lonely_reverse_items = ['lonely_1','lonely_4','lonely_5','lonely_6','lonely_9',
                            'lonely_10','lonely_15','lonely_16','lonely_19','lonely_20']
    for col in lonely_reverse_items:
        if col in lonely.columns:
            lonely[col] = 5 - lonely[col]
    survey_labels.update({f'lonely_{i+1}': col for i, col in enumerate(lonely_cols)})
    print(f"  Loneliness: {lonely.shape}")

    # VR-12: 11 items kept (removed vr_calm — inverse duplicate of vr_downhearted;
    # removed vr_physical_vs_lastyear, vr_emotional_vs_lastyear — trajectory items,
    # unreliable as pointed questions requiring memory of past state).
    # Physical functioning items (moderate_activity, climb_stairs) kept — distinct from
    # active_ratio which measures what user DID, not what their health LIMITS them to do.
    # Emotional role limitation items kept — equivalent treatment to physical role items.
    # vr = pd.read_csv(os.path.join(survey_path, 'vr_12.csv'))
    vr = encoded['vr_12']
    vr_rename = {
        'In general, would you say your health is':                                                                                                                           'vr_general_health',
        'Moderate activities, such as moving a table, pushing a vacuum cleaner, bowling or playing golf?':                                                                    'vr_moderate_activity',
        'Climbing several flights of stairs?':                                                                                                                                'vr_climb_stairs',
        'Accomplished less than you would like.':                                                                                                                             'vr_physical_limit_work',
        'Were limited in the kind of work or other activities.':                                                                                                              'vr_physical_limit_kind',
        'Accomplished less than you would like..1':                                                                                                                           'vr_emotional_limit_work',
        "Didn't do work or other activities as carefully as usual.":                                                                                                          'vr_emotional_limit_care',
        'During the past 4 weeks, how much did pain interfere with your normal work (including both work outside the home and housework)?':                                    'vr_pain_interference',
        'How much of the time during the past 4 weeks: Did you have a lot of energy?':                                                                                        'vr_energy',
        'How much of the time during the past 4 weeks: Have you felt downhearted and blue?':                                                                                  'vr_downhearted',
        'During the past 4 weeks, how much of the time has your physical health or emotional problems interfered with your social activities (like visiting with friends, relatives, etc.)?': 'vr_social_interference',
    }
    existing_vr = {k: v for k, v in vr_rename.items() if k in vr.columns}
    vr = vr.rename(columns=existing_vr)
    kept_vr_cols = [v for v in existing_vr.values() if v in vr.columns]
    vr = vr[['uid'] + kept_vr_cols].groupby('uid').mean().reset_index()
    # VR-12 reverse scoring — make all items consistent: higher = worse health/more limitation.
    # vr_moderate_activity, vr_climb_stairs: 1=limited a lot, 3=not limited → higher=better → reverse (4 - value)
    for col in ['vr_moderate_activity','vr_climb_stairs']:
        if col in vr.columns:
            vr[col] = 4 - vr[col]
    # vr_physical_limit_work, vr_physical_limit_kind,
    # vr_emotional_limit_work, vr_emotional_limit_care: 1=Yes (limited), 5=No → higher=better → reverse (6 - value)
    for col in ['vr_physical_limit_work','vr_physical_limit_kind',
                'vr_emotional_limit_work','vr_emotional_limit_care']:
        if col in vr.columns:
            vr[col] = 6 - vr[col]
    # vr_downhearted, vr_social_interference: 1=All of the time, 5=None of the time → higher=better → reverse (6 - value)
    for col in ['vr_downhearted','vr_social_interference']:
        if col in vr.columns:
            vr[col] = 6 - vr[col]
    # vr_energy: 1=All of the time, 5=None → higher=less energy=worse → already consistent, no reversal
    # vr_general_health: 1=Excellent, 5=Poor → higher=worse → already consistent, no reversal
    # vr_pain_interference: 1=Not at all, 5=Extremely → higher=worse → already consistent, no reversal
    survey_labels.update({v: k for k, v in existing_vr.items()})
    print(f"  VR-12: {vr.shape}")

    # BIG FIVE (neuroticism + extraversion only)
    # Other 3 dimensions (openness, conscientiousness, agreeableness) have no DBN node mapping.
    # At runtime: neuroticism_score and extraversion_score computed as sub-dimension attributes
    # from individual item answers collected progressively over weeks.
    # bigfive = pd.read_csv(os.path.join(survey_path, 'BigFive.csv'))
    bigfive = encoded['BigFive']
    bigfive_rename = {
        'I see myself as someone who...   - 1. Is talkative':                             'e_talkative',
        'I see myself as someone who...   - 4. Is depressed, blue':                       'n_depressed',
        'I see myself as someone who...   - 6. Is reserved':                              'e_reserved_r',
        'I see myself as someone who...   - 14. Can be tense':                            'n_tense',
        'I see myself as someone who...   - 19. Worries a lot':                           'n_worries',
        'I see myself as someone who...   - 21. Tends to be quiet':                       'e_quiet_r',
        'I see myself as someone who...   - 24. Is emotionally stable, not easily upset': 'n_stable_r',
        'I see myself as someone who...   - 29. Can be moody':                            'n_moody',
        'I see myself as someone who...   - 36. Is outgoing, sociable':                   'e_sociable',
        'I see myself as someone who...   - 39. Gets nervous easily':                     'n_nervous'
    }
    bigfive = bigfive.rename(columns=bigfive_rename)
    bigfive = bigfive[['uid'] + list(bigfive_rename.values())].groupby('uid').mean().reset_index()
    # Reverse score items — on 1-5 scale, reversed = 6 - value
    for col in ['n_stable_r','e_reserved_r','e_quiet_r']:
        if col in bigfive.columns:
            bigfive[col] = 6 - bigfive[col]
    survey_labels.update({v: k for k, v in bigfive_rename.items()})
    print(f"  BigFive: {bigfive.shape}")

    # ── MERGE ALL SURVEYS on uid ──────────────────────────────
    survey_merged = phq
    for df in [pss, psqi, panas, lonely, vr, bigfive]:
        survey_merged = survey_merged.merge(df, on='uid', how='outer')

    survey_merged.to_csv(r'C:\Users\udbha\Documents\VS Code\MedApp\datasets\studentlife_self\studentlife_surveys.csv', index=False)
    print(f"\nSurvey merged shape: {survey_merged.shape}")
    print("Saved to studentlife_surveys.csv")
    return survey_merged, survey_labels


# Loading studentlife window-level data.
# Output: one row per (uid, date, window_start).
# WINDOW_MINUTES controls granularity; 1440 = daily (reproduces original row count).
def load_studentlife():
    import bisect

    sensing_path  = os.path.join(STUDENTLIFE_PATH, 'sensing')
    ema_path      = os.path.join(STUDENTLIFE_PATH, 'EMA', 'response')
    survey_path   = os.path.join(STUDENTLIFE_PATH, 'survey')
    call_path     = os.path.join(STUDENTLIFE_PATH, 'call_log')
    sms_path      = os.path.join(STUDENTLIFE_PATH, 'sms')
    app_path      = os.path.join(STUDENTLIFE_PATH, 'app_usage')

    daily_dfs = []

    # ── PHASE 0: Build validator_alive_windows ────────────────────────────────
    # Validator sensors poll on a fixed schedule regardless of user behaviour.
    # Their timestamps confirm a window is alive even if behavioural sensors are silent.
    # Activity is also a validator: its timestamps feed alive_windows AND produce features.
    print("Building alive windows from validator sensors...")
    validator_timestamps = {}   # uid → list of int unix timestamps
    validator_folders = ['activity', 'wifi', 'bluetooth', 'gps', 'audio', 'conversation']
    for folder_name in validator_folders:
        folder_path = os.path.join(sensing_path, folder_name)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if not fname.endswith('.csv'):
                continue
            uid = extract_uid(fname)
            if not uid:
                continue
            try:
                df = pd.read_csv(os.path.join(folder_path, fname))
                ts_col = next(
                    (c for c in ['timestamp', 'time', 'start', 'start_timestamp'] if c in df.columns), None
                )
                if ts_col is None:
                    print(f"  Warning: no timestamp column in {folder_name}/{fname}")
                    continue
                timestamps = (
                    pd.to_numeric(df[ts_col], errors='coerce')
                    .dropna().astype(int).tolist()
                )
                validator_timestamps.setdefault(uid, []).extend(timestamps) # This is the ewhole motive of this block.
            except Exception as e:
                print(f"  Error loading validator {folder_name}/{fname}: {e}")

    for uid in validator_timestamps:
        validator_timestamps[uid].sort()

    validator_alive_windows = set()   # (uid, window_start_unix)
    for uid, sorted_ts in validator_timestamps.items():
        dates = set(pd.to_datetime(pd.Series(sorted_ts), unit='s').dt.date.tolist())
        for date_obj in dates:
            for w_start_unix, w_end_unix, _ in get_window_slots(date_obj, WINDOW_MINUTES):
                lo = bisect.bisect_left(sorted_ts, w_start_unix)
                if lo < len(sorted_ts) and sorted_ts[lo] < w_end_unix:
                    validator_alive_windows.add((uid, w_start_unix))
    print(f"  Validator alive windows: {len(validator_alive_windows)}")

    # ── PHASE 1: Load raw behavioural sensor data ─────────────────────────────
    # All behavioural sensors loaded into memory first so their timestamps can
    # collectively build behavioural_alive_windows before any aggregation pass.

    # DARK SENSING
    print("Loading dark sensing...")
    dark_dfs = []
    for fname in os.listdir(os.path.join(sensing_path, 'dark')):
        if not fname.endswith('.csv'): continue
        uid = extract_uid(fname)
        if not uid: continue
        df = pd.read_csv(os.path.join(sensing_path, 'dark', fname))
        df['uid']  = uid
        df['date'] = pd.to_datetime(
            pd.to_numeric(df['start'], errors='coerce'), unit='s'
        ).dt.date
        df = df.dropna(subset=['date'])
        dark_dfs.append(df[['uid', 'date', 'start', 'end']])
    dark_df_raw = pd.concat(dark_dfs, ignore_index=True) if dark_dfs else pd.DataFrame(columns=['uid','date','start','end'])

    # PHONELOCK SENSING: start = unlock time, end = lock time → intervals are unlocked sessions.
    print("Loading phonelock sensing...")
    lock_dfs = []
    for fname in os.listdir(os.path.join(sensing_path, 'phonelock')):
        if not fname.endswith('.csv'): continue
        uid = extract_uid(fname)
        if not uid: continue
        df = pd.read_csv(os.path.join(sensing_path, 'phonelock', fname))
        df['uid']  = uid
        df['date'] = pd.to_datetime(
            pd.to_numeric(df['start'], errors='coerce'), unit='s'
        ).dt.date
        df = df.dropna(subset=['date'])
        lock_dfs.append(df[['uid', 'date', 'start', 'end']])
    lock_df_raw = pd.concat(lock_dfs, ignore_index=True) if lock_dfs else pd.DataFrame(columns=['uid','date','start','end'])

    # APP USAGE: cognitive load proxy
    # RUNNING_TASKS_numRunning = number of actively running app tasks.
    # High numRunning signals mental busyness.
    # Cross-midnight fix: adj_dt = dt - 2h so 00:00–01:59 maps to the previous day.
    # Peak hour definitions (invariant):
    #   Morning peak: 07:00-09:59 — morning routine and start-of-day cognitive load
    #   Evening peak: 20:00-01:59 — includes usage past midnight (cross-midnight handled)
    print("Loading app usage sensing...")
    app_dfs = []
    if os.path.isdir(app_path):
        for fname in os.listdir(app_path):
            if not fname.endswith('.csv'): continue
            uid = extract_uid(fname)
            if not uid: continue
            try:
                df = pd.read_csv(os.path.join(app_path, fname))
                if 'RUNNING_TASKS_numRunning' not in df.columns: continue
                df['uid'] = uid
                df['timestamp_unix'] = pd.to_numeric(df['timestamp'], errors='coerce')
                df = df.dropna(subset=['timestamp_unix'])
                df['dt']     = pd.to_datetime(df['timestamp_unix'], unit='s')
                df['hour']   = df['dt'].dt.hour
                df['adj_dt'] = df['dt'] - pd.Timedelta(hours=2)
                df['date']   = df['adj_dt'].dt.date
                # Evening peak: 20:00-23:59 OR 00:00-01:59 (both map to same adj_date)
                df['is_evening'] = df['hour'].between(20, 23) | df['hour'].between(0, 1)
                app_dfs.append(df[['uid','date','timestamp_unix','RUNNING_TASKS_numRunning','is_evening']])
            except Exception as e:
                print(f"  Error loading {fname}: {e}")
    app_df_raw = pd.concat(app_dfs, ignore_index=True) if app_dfs else pd.DataFrame(columns=['uid','date','timestamp_unix','RUNNING_TASKS_numRunning','is_evening'])

    # CALL LOG: social_activity
    # Privacy columns dropped: CALLS_name, CALLS_number, CALLS_numberlabel, CALLS_numbertype, id, device.
    # CALLS_date is in milliseconds → divide by 1000 to get seconds.
    print("Loading call log...")
    call_dfs = []
    for fname in os.listdir(call_path):
        if not fname.endswith('.csv'): continue
        uid = extract_uid(fname)
        if not uid: continue
        df = pd.read_csv(os.path.join(call_path, fname),
                 header=None,
                 names=['id','device','timestamp','CALLS__id','CALLS_date','CALLS_duration','CALLS_name','CALLS_number','CALLS_numberlabel','CALLS_numbertype','CALLS_type']
                 )
        df['uid']          = uid
        df['call_ts_unix'] = pd.to_numeric(df['CALLS_date'], errors='coerce') / 1000
        df['date']         = pd.to_datetime(df['call_ts_unix'], unit='s').dt.date
        df = df.dropna(subset=['date'])
        df["CALLS_duration"] = pd.to_numeric(df["CALLS_duration"], errors='coerce')
        call_dfs.append(df[['uid', 'date', 'call_ts_unix', 'CALLS_duration']])
    call_df_raw = pd.concat(call_dfs, ignore_index=True) if call_dfs else pd.DataFrame(columns=['uid','date','call_ts_unix','CALLS_duration'])

    # SMS: social_activity
    # Privacy columns dropped: MESSAGES_body, MESSAGES_address, MESSAGES_person, all metadata.
    # MESSAGES_date is in milliseconds → divide by 1000 to get seconds.
    print("Loading SMS...")
    sms_dfs = []
    for fname in os.listdir(sms_path):
        if not fname.endswith('.csv'): continue
        uid = extract_uid(fname)
        if not uid: continue
        df = pd.read_csv(os.path.join(sms_path, fname),
                 header=None,
                 names=['id','device','timestamp','MESSAGES_address','MESSAGES_body','MESSAGES_date','MESSAGES_locked','MESSAGES_person','MESSAGES_protocol','MESSAGES_read','MESSAGES_reply_path_present','MESSAGES_service_center','MESSAGES_status','MESSAGES_subject','MESSAGES_thread_id','MESSAGES_type'])
        df['uid']         = uid
        df['sms_ts_unix'] = pd.to_numeric(df['MESSAGES_date'], errors='coerce') / 1000
        df['date']        = pd.to_datetime(df['sms_ts_unix'], unit='s').dt.date
        df = df.dropna(subset=['date'])
        sms_dfs.append(df[['uid', 'date', 'sms_ts_unix', 'MESSAGES_type']])
    sms_df_raw = pd.concat(sms_dfs, ignore_index=True) if sms_dfs else pd.DataFrame(columns=['uid','date','sms_ts_unix','MESSAGES_type'])

    # Build behavioural_alive_windows by scanning ALL raw behavioural timestamps together.
    # A window is behaviourally alive if ANY behavioural sensor fired in it.
    print("Building behavioural alive windows...")
    behavioural_ts = {}   # uid → list of int unix timestamps (all sensors combined)

    def _collect_ts(df, ts_col):
        if df.empty: return
        for uid, grp in df.groupby('uid'):
            ts_list = pd.to_numeric(grp[ts_col], errors='coerce').dropna().astype(int).tolist()
            behavioural_ts.setdefault(uid, []).extend(ts_list)

    _collect_ts(dark_df_raw,  'start')
    _collect_ts(lock_df_raw,  'start')
    _collect_ts(app_df_raw,   'timestamp_unix')
    _collect_ts(call_df_raw,  'call_ts_unix')
    _collect_ts(sms_df_raw,   'sms_ts_unix')

    for uid in behavioural_ts:
        behavioural_ts[uid].sort()

    behavioural_alive_windows = set()
    for uid, sorted_ts in behavioural_ts.items():
        dates = set(pd.to_datetime(pd.Series(sorted_ts), unit='s').dt.date.tolist())
        for date_obj in dates:
            for w_start_unix, w_end_unix, _ in get_window_slots(date_obj, WINDOW_MINUTES):
                lo = bisect.bisect_left(sorted_ts, w_start_unix)
                if lo < len(sorted_ts) and sorted_ts[lo] < w_end_unix:
                    behavioural_alive_windows.add((uid, w_start_unix))

    all_alive_windows = validator_alive_windows | behavioural_alive_windows
    print(f"  All alive windows: {len(all_alive_windows)}")

    # Build all_windows_base — every W slot for every (uid, date) observed in any sensor.
    # All window slots appear in the final output. Dead windows → NaN throughout.
    # all_alive_windows still used for conditional fill: alive+no-readings → 0, dead → NaN.
    all_uid_dates = set()
    for uid, ts_list in validator_timestamps.items():
        for d in set(pd.to_datetime(pd.Series(ts_list), unit='s').dt.date.tolist()):
            all_uid_dates.add((uid, d))
    for uid, ts_list in behavioural_ts.items():
        for d in set(pd.to_datetime(pd.Series(ts_list), unit='s').dt.date.tolist()):
            all_uid_dates.add((uid, d))

    all_window_rows = [
        {'uid': uid, 'date': date_obj, 'window_start': w_start_dt, 'w_start_unix': w_start_unix}
        for uid, date_obj in all_uid_dates
        for w_start_unix, _, w_start_dt in get_window_slots(date_obj, WINDOW_MINUTES)
    ]
    all_windows_base = pd.DataFrame(all_window_rows) if all_window_rows else pd.DataFrame(columns=['uid','date','window_start','w_start_unix'])
    print(f"  Total windows (all days): {len(all_windows_base)}")

    # Combined sorted timestamps per uid — for nighttime alive minute computation.
    all_sensor_ts = {}
    for uid, ts_list in validator_timestamps.items():
        all_sensor_ts.setdefault(uid, []).extend(ts_list)
    for uid, ts_list in behavioural_ts.items():
        all_sensor_ts.setdefault(uid, []).extend(ts_list)
    for uid in all_sensor_ts:
        all_sensor_ts[uid].sort()

    def _night_alive_mins(uid, night_date):
        """Count distinct sensor-occupied minutes within 20:00-02:00 of night_date."""
        night_start = int(pd.Timestamp(night_date).value // 10**9) + 20 * 3600
        night_end   = night_start + 6 * 3600  # 02:00 next day = 20:00 + 6h
        ts_list = all_sensor_ts.get(uid, [])
        if not ts_list:
            return 0
        lo = bisect.bisect_left(ts_list, night_start)
        hi = bisect.bisect_right(ts_list, night_end)
        night_ts = ts_list[lo:hi]
        return len(set((t - night_start) // 60 for t in night_ts)) if night_ts else 0

    # Alive nights base — reused by dark and phonelock phases.
    # Only nights where any sensor fired within 20:00–02:00 appear here.
    _alive_night_rows = [
        {'uid': uid, 'date': date_obj, '_alive_night_mins': _night_alive_mins(uid, date_obj)}
        for uid, date_obj in all_uid_dates
    ]
    all_alive_nights_df = (
        pd.DataFrame([r for r in _alive_night_rows if r['_alive_night_mins'] > 0])
        if _alive_night_rows else
        pd.DataFrame(columns=['uid', 'date', '_alive_night_mins'])
    )

    # ── PHASE 2: Activity sensing ─────────────────────────────────────────────
    # Category: Validator — always-on fixed-schedule polling.
    # Timestamps already fed into validator_alive_windows above.
    # No readings in a window → NaN (activity absence is itself the dead signal).
    # Output: sedentary_ratio, active_ratio per window.
    # activity inference values: 0=stationary, 1=walking, 2=running, 3=unknown
    # "unknown" readings excluded from valid_count to prevent ratio dilution.
    print("Computing activity per window...")
    act_dfs = []
    for fname in os.listdir(os.path.join(sensing_path, 'activity')):
        if not fname.endswith('.csv'): continue
        uid = extract_uid(fname)
        if not uid: continue
        df = pd.read_csv(os.path.join(sensing_path, 'activity', fname))
        df.columns = df.columns.str.strip()
        df['uid']           = uid
        df['timestamp_unix'] = pd.to_numeric(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp_unix'])
        df['date'] = pd.to_datetime(df['timestamp_unix'], unit='s').dt.date
        act_dfs.append(df[['uid', 'date', 'timestamp_unix', 'activity inference']])

    if act_dfs:
        act_df = pd.concat(act_dfs, ignore_index=True)
        # Assign each reading to its midnight-anchored window slot
        act_dt        = pd.to_datetime(act_df['timestamp_unix'], unit='s')
        act_day_start = act_dt.dt.normalize()
        act_mins      = (act_dt - act_day_start).dt.total_seconds() / 60
        act_slot_idx  = (act_mins // WINDOW_MINUTES).astype(int).clip(lower=0, upper=(1440 // WINDOW_MINUTES) - 1)
        act_df['window_start'] = act_day_start + pd.to_timedelta(act_slot_idx * WINDOW_MINUTES, unit='m')

        act_win = act_df.groupby(['uid', 'date', 'window_start']).agg(
            stationary_count = ('activity inference', lambda x: (x == 0).sum()),
            walking_count    = ('activity inference', lambda x: (x == 1).sum()),
            running_count    = ('activity inference', lambda x: (x == 2).sum()),
            valid_count      = ('activity inference', lambda x: (x != 3).sum())
        ).reset_index()
        act_win['valid_count']    = act_win['valid_count'].replace(0, np.nan)
        act_win['sedentary_ratio'] = act_win['stationary_count'] / act_win['valid_count']
        act_win['active_ratio']    = (act_win['walking_count'] + act_win['running_count']) / act_win['valid_count']
        act_win.drop(columns=['stationary_count', 'walking_count', 'running_count', 'valid_count'], inplace=True)
        # Daily ratios — recomputed from all raw readings for the day (not mean of window ratios)
        act_daily = act_df.groupby(['uid', 'date']).agg(
            _stat  = ('activity inference', lambda x: (x == 0).sum()),
            _walk  = ('activity inference', lambda x: (x == 1).sum()),
            _run   = ('activity inference', lambda x: (x == 2).sum()),
            _valid = ('activity inference', lambda x: (x != 3).sum()),
        ).reset_index()
        act_daily['_valid'] = act_daily['_valid'].replace(0, np.nan)
        act_daily['daily_sedentary_ratio'] = act_daily['_stat'] / act_daily['_valid']
        act_daily['daily_active_ratio']    = (act_daily['_walk'] + act_daily['_run']) / act_daily['_valid']
        act_daily.drop(columns=['_stat', '_walk', '_run', '_valid'], inplace=True)
        act_win = act_win.merge(act_daily, on=['uid', 'date'], how='left')
        # Activity: no readings in window → NaN (validator; dead = NaN, not 0).
        act_full = all_windows_base[['uid','date','window_start']].merge(act_win, on=['uid','date','window_start'], how='left')
        daily_dfs.append(act_full)
        print(f"  Activity: {act_full.shape}")

    # ── PHASE 3: Dark sensing (behavioural interval) ──────────────────────────
    # Alive architecture used.
    # dark_minutes     : daily total dark time (deduplicated), no quality filter.
    # nighttime_active_minutes : 360-min window 20:00–02:00, dark subtracted.
    # dark_window_minutes : per window slot, with alive rule (0 if alive + no dark).
    # Nighttime window and cross-midnight shift (2h) are invariant — do not change.
    dark_full = None
    if not dark_df_raw.empty:
        dark_df = dark_df_raw.copy()
        dark_df['start_num'] = pd.to_numeric(dark_df['start'], errors='coerce')
        dark_df['end_num']   = pd.to_numeric(dark_df['end'],   errors='coerce')
        dark_df['start_dt']  = pd.to_datetime(dark_df['start_num'], unit='s')

        # Daily dark_minutes
        dark_daily = dark_df.groupby(['uid', 'date']).apply(
            lambda g: deduplicate_intervals(g, 'start', 'end')
        ).reset_index()
        dark_daily.columns = ['uid', 'date', 'dark_minutes']

        # Nighttime window: 8 PM on day D → 2 AM on day D+1, treated as day D.
        # Shift by 2h so 00:00–01:59 maps to the previous calendar date (invariant).
        dark_df['night_date'] = (dark_df['start_dt'] - pd.Timedelta(hours=2)).dt.normalize().dt.date
        dark_df['win_s'] = dark_df['night_date'].map(
            lambda d: (pd.Timestamp(d) + pd.Timedelta(hours=20)).timestamp()
        )
        dark_df['win_e'] = dark_df['night_date'].map(
            lambda d: (pd.Timestamp(d) + pd.Timedelta(days=1, hours=2)).timestamp()
        )
        dark_df['clip_s'] = dark_df[['start_num', 'win_s']].max(axis=1)
        dark_df['clip_e'] = dark_df[['end_num',   'win_e']].min(axis=1)
        night_df = dark_df[dark_df['clip_s'] < dark_df['clip_e']].copy()

        night_dark = (
            night_df.groupby(['uid', 'night_date'])
            .apply(lambda g: deduplicate_intervals(g, 'clip_s', 'clip_e'))
            .reset_index()
        )
        night_dark.columns = ['uid', 'date', '_nighttime_dark_min']
        # nighttime_active = alive_mins_in_night − dark_mins_in_night (alive nights only)
        night_active_df = all_alive_nights_df.merge(
            night_dark[['uid', 'date', '_nighttime_dark_min']], on=['uid', 'date'], how='left'
        )
        night_active_df['_nighttime_dark_min'] = night_active_df['_nighttime_dark_min'].fillna(0.0)
        night_active_df['nighttime_active_minutes'] = (
            night_active_df['_alive_night_mins'] - night_active_df['_nighttime_dark_min']
        ).clip(lower=0)
        night_active_df = night_active_df[['uid', 'date', 'nighttime_active_minutes']]

        # Per-window dark_window_minutes using clip_intervals_to_window
        # Dark sensor stores real readings for all slots — day and night.
        dark_window_rows = []
        for (uid, date_obj), grp in dark_df.groupby(['uid', 'date']):
            intervals = [
                (s, e) for s, e in zip(
                    pd.to_numeric(grp['start'], errors='coerce').tolist(),
                    pd.to_numeric(grp['end'],   errors='coerce').tolist()
                )
                if not (pd.isna(s) or pd.isna(e))
            ]
            for w_start_unix, w_end_unix, w_start_dt in get_window_slots(date_obj, WINDOW_MINUTES):
                mins = clip_intervals_to_window(intervals, w_start_unix, w_end_unix)
                dark_window_rows.append({
                    'uid': uid, 'date': date_obj,
                    'window_start': w_start_dt,
                    'dark_window_minutes': mins
                })
        dark_window_df = pd.DataFrame(dark_window_rows)

        # Merge with all_windows_base: dead windows → NaN, alive + no-dark → 0.
        dark_full = all_windows_base.merge(dark_window_df, on=['uid', 'date', 'window_start'], how='left')
        _alive_mask = pd.Series(
            [(u, int(w)) in all_alive_windows for u, w in zip(dark_full['uid'], dark_full['w_start_unix'])],
            index=dark_full.index
        )
        dark_full.loc[_alive_mask & dark_full['dark_window_minutes'].isna(), 'dark_window_minutes'] = 0.0
        dark_full.drop(columns=['w_start_unix'], inplace=True)
        # dark_minutes: every row in dark_full is an alive date → fill 0 for dates with no dark data
        dark_full = dark_full.merge(dark_daily[['uid', 'date', 'dark_minutes']], on=['uid', 'date'], how='left')
        dark_full['dark_minutes'] = dark_full['dark_minutes'].fillna(0.0) # This is safe here since the all_windows_base is made up of only those dates where there is at least one sensor active.
        # nighttime_active_minutes: alive nights only; dead nights stay NaN
        dark_full = dark_full.merge(night_active_df, on=['uid', 'date'], how='left')

        daily_dfs.append(dark_full[['uid', 'date', 'window_start', 'dark_window_minutes', 'dark_minutes', 'nighttime_active_minutes']])
        print(f"  Dark: {dark_full.shape}")

    # ── PHASE 4: Phonelock sensing (behavioural interval) ─────────────────────
    # Quality filters REMOVED — replaced by alive architecture.
    # start = unlock time, end = lock time → intervals are unlocked sessions.
    # unlocked_minutes          : daily total deduplicated unlocked time.
    # nighttime_unlock_minutes  : unlocked time within 20:00–02:00 (invariant window).
    # unlocked_window_minutes   : per window slot, with alive rule (0 if alive + no unlock).
    lock_full = None
    if not lock_df_raw.empty:
        lock_df = lock_df_raw.copy()
        lock_df['start_num'] = pd.to_numeric(lock_df['start'], errors='coerce')
        lock_df['end_num']   = pd.to_numeric(lock_df['end'],   errors='coerce')
        lock_df['start_dt']  = pd.to_datetime(lock_df['start_num'], unit='s')

        # Daily unlocked_minutes — no quality filters
        lock_daily = lock_df.groupby(['uid', 'date']).apply(
            lambda g: deduplicate_intervals(g, 'start_num', 'end_num')
        ).reset_index()
        lock_daily.columns = ['uid', 'date', 'unlocked_minutes']

        # Nighttime unlock: 8 PM → 2 AM, shift by 2h (invariant cross-midnight shift)
        lock_df['night_date'] = (lock_df['start_dt'] - pd.Timedelta(hours=2)).dt.normalize().dt.date
        lock_df['win_s'] = lock_df['night_date'].map(
            lambda d: (pd.Timestamp(d) + pd.Timedelta(hours=20)).timestamp()
        )
        lock_df['win_e'] = lock_df['night_date'].map(
            lambda d: (pd.Timestamp(d) + pd.Timedelta(days=1, hours=2)).timestamp()
        )
        lock_df['clip_s'] = lock_df[['start_num', 'win_s']].max(axis=1)
        lock_df['clip_e'] = lock_df[['end_num',   'win_e']].min(axis=1)
        night_lock_df = lock_df[lock_df['clip_s'] < lock_df['clip_e']].copy()

        night_unlock = (
            night_lock_df.groupby(['uid', 'night_date'])
            .apply(lambda g: deduplicate_intervals(g, 'clip_s', 'clip_e'))
            .reset_index()
        )
        night_unlock.columns = ['uid', 'date', 'nighttime_unlock_minutes']
        # nighttime_unlock_minutes: 0 for alive nights with no unlock, NaN for dead nights
        night_unlock_full = all_alive_nights_df[['uid', 'date']].merge(
            night_unlock, on=['uid', 'date'], how='left'
        )
        night_unlock_full['nighttime_unlock_minutes'] = night_unlock_full['nighttime_unlock_minutes'].fillna(0.0)

        # Per-window unlocked_window_minutes using clip_intervals_to_window (valid at ALL hours)
        lock_window_rows = []
        for (uid, date_obj), grp in lock_df.groupby(['uid', 'date']):
            intervals = [
                (s, e) for s, e in zip(
                    pd.to_numeric(grp['start'], errors='coerce').tolist(),
                    pd.to_numeric(grp['end'],   errors='coerce').tolist()
                )
                if not (pd.isna(s) or pd.isna(e))
            ]
            for w_start_unix, w_end_unix, w_start_dt in get_window_slots(date_obj, WINDOW_MINUTES):
                mins = clip_intervals_to_window(intervals, w_start_unix, w_end_unix)
                lock_window_rows.append({
                    'uid': uid, 'date': date_obj,
                    'window_start': w_start_dt,
                    'unlocked_window_minutes': mins
                })
        lock_window_df = pd.DataFrame(lock_window_rows)

        # Merge with all_windows_base: dead windows → NaN, alive + no-unlock → 0.
        lock_full = all_windows_base.merge(lock_window_df, on=['uid', 'date', 'window_start'], how='left')
        _alive_mask = pd.Series(
            [(u, int(w)) in all_alive_windows for u, w in zip(lock_full['uid'], lock_full['w_start_unix'])],
            index=lock_full.index
        )
        lock_full.loc[_alive_mask & lock_full['unlocked_window_minutes'].isna(), 'unlocked_window_minutes'] = 0.0
        lock_full.drop(columns=['w_start_unix'], inplace=True)
        # unlocked_minutes: every row in lock_full is an alive date → fill 0 for dates with no lock data
        lock_full = lock_full.merge(lock_daily[['uid', 'date', 'unlocked_minutes']], on=['uid', 'date'], how='left')
        lock_full['unlocked_minutes'] = lock_full['unlocked_minutes'].fillna(0.0)
        # nighttime_unlock_minutes: alive nights → 0 or actual value, dead nights → NaN
        lock_full = lock_full.merge(night_unlock_full, on=['uid', 'date'], how='left')

        daily_dfs.append(lock_full[['uid', 'date', 'window_start', 'unlocked_window_minutes', 'unlocked_minutes', 'nighttime_unlock_minutes']])
        print(f"  Phonelock: {lock_full.shape}")

    # ── PHASE 5: Screen time (derived per window) ─────────────────────────────
    # Derived from dark_window_minutes and unlocked_window_minutes after both are computed.
    # Night slots (21:00–07:00): use phonelock only (dark unreliable at night).
    # Day slots: cross-validate dark and phonelock.
    # coalesce(x, default) = x if not NaN, else default.
    if dark_full is not None or lock_full is not None:
        screen_base = all_windows_base[['uid','date','window_start']].copy()
        if dark_full is not None:
            screen_base = screen_base.merge(
                dark_full[['uid', 'date', 'window_start', 'dark_window_minutes']],
                on=['uid', 'date', 'window_start'], how='left'
            )
        else:
            screen_base['dark_window_minutes'] = np.nan

        if lock_full is not None:
            screen_base = screen_base.merge(
                lock_full[['uid', 'date', 'window_start', 'unlocked_window_minutes']],
                on=['uid', 'date', 'window_start'], how='left'
            )
        else:
            screen_base['unlocked_window_minutes'] = np.nan

        hour_series = screen_base['window_start'].dt.hour
        is_night    = (hour_series >= 20) | (hour_series < 4)

        def _screen_time(row):
            dark_m   = row['dark_window_minutes']
            unlock_m = row['unlocked_window_minutes']
            if pd.isna(dark_m) and pd.isna(unlock_m):
                return np.nan  # both unknown — dead window
            if row['_is_night']:
                # Night: dark unreliable — use phonelock unlocked time directly
                return unlock_m
            else:
                # Day: cross-validate
                locked_m   = WINDOW_MINUTES - (unlock_m if not pd.isna(unlock_m) else WINDOW_MINUTES)
                screen_off = max(dark_m if not pd.isna(dark_m) else 0.0, locked_m)
                return max(0.0, WINDOW_MINUTES - screen_off)

        screen_base['_is_night'] = is_night
        screen_base['screen_time_window_minutes'] = screen_base.apply(_screen_time, axis=1)
        screen_base.drop(columns=['_is_night', 'dark_window_minutes', 'unlocked_window_minutes'], inplace=True)
        screen_daily = screen_base.groupby(['uid', 'date'])['screen_time_window_minutes'].sum().reset_index(name='daily_screen_time_minutes')
        screen_base = screen_base.merge(screen_daily, on=['uid', 'date'], how='left')
        daily_dfs.append(screen_base[['uid', 'date', 'window_start', 'screen_time_window_minutes', 'daily_screen_time_minutes']])
        print(f"  Screen time: {screen_base.shape}")

    # ── PHASE 6: App usage ────────────────────────────────────────────────────
    # avg_running_tasks_window: per window slot, alive rule applied.
    # Daily aggregates computed first (no window math needed), then per-window.
    # Window assignment uses actual timestamp — no shift — consistent with all sensors.
    # is_evening uses raw clock hour, unaffected by date assignment.
    if not app_df_raw.empty:
        app_df = app_df_raw.copy()
        app_df['dt']       = pd.to_datetime(app_df['timestamp_unix'], unit='s')
        app_df['adj_date'] = app_df['date']        # preserve −2h shifted date for evening aggregation
        app_df['date']     = app_df['dt'].dt.date  # raw date for window/alive alignment

        # ── Window assignment — actual timestamp, no shift ──
        app_df['day_start'] = app_df['dt'].dt.normalize()
        day_mins = (app_df['dt'] - app_df['day_start']).dt.total_seconds() / 60
        app_slot_idx = (day_mins // WINDOW_MINUTES).astype(int).clip(lower=0, upper=(1440 // WINDOW_MINUTES) - 1)
        app_df['window_start'] = app_df['day_start'] + pd.to_timedelta(app_slot_idx * WINDOW_MINUTES, unit='m')
        app_df['w_start_unix'] = app_df['day_start'].astype('datetime64[s]').astype('int64') + app_slot_idx * WINDOW_MINUTES * 60

        app_alive_win = app_df[
            pd.Series([(u, int(w)) in all_alive_windows for u, w in zip(app_df['uid'], app_df['w_start_unix'])], index=app_df.index)
        ]

        # ── Per-window avg_running_tasks_window ──
        app_win = app_alive_win.groupby(['uid', 'date', 'window_start']).agg(
            avg_running_tasks_window = ('RUNNING_TASKS_numRunning', 'mean')
        ).reset_index()
        app_win_full = all_windows_base.merge(app_win, on=['uid', 'date', 'window_start'], how='left')
        _alive_mask = pd.Series(
            [(u, int(w)) in all_alive_windows for u, w in zip(app_win_full['uid'], app_win_full['w_start_unix'])],
            index=app_win_full.index
        )
        app_win_full.loc[_alive_mask & app_win_full['avg_running_tasks_window'].isna(), 'avg_running_tasks_window'] = 0.0
        app_win_full.drop(columns=['w_start_unix'], inplace=True)

        # ── Daily aggregates ──
        # avg/peak: day-level (all alive-window readings for the day)
        # morning/evening: window-level alive filter — only readings from alive morning/evening slots
        if not app_alive_win.empty:
            app_daily = app_alive_win.groupby(['uid', 'date']).agg(
                avg_running_tasks  = ('RUNNING_TASKS_numRunning', 'mean'),
                peak_running_tasks = ('RUNNING_TASKS_numRunning', 'max'),
            ).reset_index()
            evening_agg = (
                app_alive_win[app_alive_win['is_evening']]
                .groupby(['uid', 'adj_date'])['RUNNING_TASKS_numRunning']
                .agg(evening_running_tasks='mean', evening_peak_running_tasks='max')
                .reset_index()
                .rename(columns={'adj_date': 'date'})
            )
            app_daily = app_daily.merge(evening_agg, on=['uid', 'date'], how='left')
        else:
            app_daily = pd.DataFrame(columns=['uid','date','avg_running_tasks','peak_running_tasks','evening_running_tasks','evening_peak_running_tasks'])

        app_full = app_win_full.merge(app_daily, on=['uid', 'date'], how='left')
        daily_dfs.append(app_full[['uid', 'date', 'window_start', 'avg_running_tasks_window',
                                    'avg_running_tasks', 'peak_running_tasks',
                                    'evening_running_tasks', 'evening_peak_running_tasks']])
        print(f"  App usage: {app_full.shape}")
    else:
        print("  App usage: folder not found or no valid files.")

    # ── PHASE 7: Call log ─────────────────────────────────────────────────────
    # call_count, call_duration_total: per window, alive rule (0 if alive + no calls).
    if not call_df_raw.empty:
        call_df = call_df_raw.copy()
        call_dt        = pd.to_datetime(call_df['call_ts_unix'], unit='s')
        call_day_start = call_dt.dt.normalize()
        call_mins      = (call_dt - call_day_start).dt.total_seconds() / 60
        call_slot_idx  = (call_mins // WINDOW_MINUTES).astype(int).clip(lower=0, upper=(1440 // WINDOW_MINUTES) - 1)
        call_df['window_start'] = call_day_start + pd.to_timedelta(call_slot_idx * WINDOW_MINUTES, unit='m')

        call_win = call_df.groupby(['uid', 'date', 'window_start']).agg(
            call_count          = ('CALLS_duration', 'count'),
            call_duration_total = ('CALLS_duration', 'sum')
        ).reset_index()

        call_full = all_windows_base.merge(call_win, on=['uid', 'date', 'window_start'], how='left')
        _alive_mask = pd.Series(
            [(u, int(w)) in all_alive_windows for u, w in zip(call_full['uid'], call_full['w_start_unix'])],
            index=call_full.index
        )
        for _col in ['call_count', 'call_duration_total']:
            call_full.loc[_alive_mask & call_full[_col].isna(), _col] = 0.0
        call_full.drop(columns=['w_start_unix'], inplace=True)
        call_daily = call_df.groupby(['uid', 'date']).agg(
            daily_call_count          = ('CALLS_duration', 'count'),
            daily_call_duration_total = ('CALLS_duration', 'sum'),
        ).reset_index()
        call_full = call_full.merge(call_daily, on=['uid', 'date'], how='left')
        call_full['daily_call_count']          = call_full['daily_call_count'].fillna(0.0)
        call_full['daily_call_duration_total'] = call_full['daily_call_duration_total'].fillna(0.0)
        daily_dfs.append(call_full[['uid', 'date', 'window_start', 'call_count', 'call_duration_total', 'daily_call_count', 'daily_call_duration_total']])
        print(f"  Call log: {call_full.shape}")

    # ── PHASE 8: SMS ──────────────────────────────────────────────────────────
    # sms_count: per window, alive rule (0 if alive + no messages).
    if not sms_df_raw.empty:
        sms_df = sms_df_raw.copy()
        sms_dt        = pd.to_datetime(sms_df['sms_ts_unix'], unit='s')
        sms_day_start = sms_dt.dt.normalize()
        sms_mins      = (sms_dt - sms_day_start).dt.total_seconds() / 60
        sms_slot_idx  = (sms_mins // WINDOW_MINUTES).astype(int).clip(lower=0, upper=(1440 // WINDOW_MINUTES) - 1)
        sms_df['window_start'] = sms_day_start + pd.to_timedelta(sms_slot_idx * WINDOW_MINUTES, unit='m')

        sms_win = sms_df.groupby(['uid', 'date', 'window_start']).agg(
            sms_count = ('MESSAGES_type', 'count')
        ).reset_index()

        sms_full = all_windows_base.merge(sms_win, on=['uid', 'date', 'window_start'], how='left')
        _alive_mask = pd.Series(
            [(u, int(w)) in all_alive_windows for u, w in zip(sms_full['uid'], sms_full['w_start_unix'])],
            index=sms_full.index
        )
        sms_full.loc[_alive_mask & sms_full['sms_count'].isna(), 'sms_count'] = 0.0
        sms_full.drop(columns=['w_start_unix'], inplace=True)
        sms_daily = sms_df.groupby(['uid', 'date']).agg(
            daily_sms_count = ('MESSAGES_type', 'count')
        ).reset_index()
        sms_full = sms_full.merge(sms_daily, on=['uid', 'date'], how='left')
        sms_full['daily_sms_count'] = sms_full['daily_sms_count'].fillna(0.0)
        daily_dfs.append(sms_full[['uid', 'date', 'window_start', 'sms_count', 'daily_sms_count']])
        print(f"  SMS: {sms_full.shape}")

    # ── PHASE 9: EMA loaders (carry-forward, NO alive rule) ───────────────────
    # EMA responses are user-submitted state declarations — valid regardless of sensor state.
    # Each response fills its window slot AND carries forward into subsequent slots for that day.
    # Day boundary: fall-through does NOT cross midnight. Each day is independent.
    # Multiple responses in same slot: mean is taken.
    # Slots before the first response of the day → NaN (genuinely unknown, never fill 0).

    def _ema_fall_through(ema_df, resp_col, value_cols):
        """
        For each (uid, date), assigns EMA values to all window slots with carry-forward.
        Returns a DataFrame with ['uid', 'date', 'window_start'] + value_cols.
        """
        if ema_df.empty:
            return pd.DataFrame(columns=['uid', 'date', 'window_start'] + value_cols)

        result_rows = []
        for (uid, date_obj), grp in ema_df.groupby(['uid', 'date']):
            grp = grp.copy()
            grp['_rts'] = pd.to_numeric(grp[resp_col], errors='coerce')
            grp = grp.sort_values('_rts').reset_index(drop=True)
            resp_list = grp[['_rts'] + value_cols].to_dict('records')
            slots = get_window_slots(date_obj, WINDOW_MINUTES)

            current = {col: np.nan for col in value_cols}
            resp_idx = 0

            for w_start_unix, w_end_unix, w_start_dt in slots:
                # Collect all responses in [w_start, w_end); mean if multiple
                window_vals = {col: [] for col in value_cols}
                while resp_idx < len(resp_list):
                    r_ts = resp_list[resp_idx]['_rts']
                    if pd.isna(r_ts) or r_ts >= w_end_unix:
                        break
                    if r_ts >= w_start_unix:
                        for col in value_cols:
                            v = resp_list[resp_idx][col]
                            if not pd.isna(v):
                                window_vals[col].append(v)
                    resp_idx += 1
                for col in value_cols:
                    if window_vals[col]:
                        current[col] = np.mean(window_vals[col])

                row = {'uid': uid, 'date': date_obj, 'window_start': w_start_dt}
                row.update(current)
                result_rows.append(row)

        return pd.DataFrame(result_rows)

    # EMA: mood
    # happy/sad continuous ratings kept. happyornot/sadornot are binary duplicates — dropped.
    print("Loading EMA Mood...")
    mood_df = load_per_user_json(os.path.join(ema_path, 'Mood'), ['happy', 'sad', 'resp_time'])
    if not mood_df.empty:
        mood_df['date']  = pd.to_datetime(pd.to_numeric(mood_df['resp_time'], errors='coerce'), unit='s').dt.date
        mood_df = mood_df.dropna(subset=['date'])
        mood_df['happy'] = pd.to_numeric(mood_df['happy'], errors='coerce')
        mood_df['sad']   = pd.to_numeric(mood_df['sad'],   errors='coerce')
        mood_win = _ema_fall_through(mood_df, 'resp_time', ['happy', 'sad'])
        mood_win = mood_win.rename(columns={'happy': 'mood_happy', 'sad': 'mood_sad'})
        daily_dfs.append(mood_win)
        print(f"  Mood: {mood_win.shape}")

    # EMA: mood2
    print("Loading EMA Mood 2...")
    mood2_df = load_per_user_json(os.path.join(ema_path, 'Mood 2'), ['how', 'resp_time'])
    if not mood2_df.empty:
        mood2_df['date'] = pd.to_datetime(pd.to_numeric(mood2_df['resp_time'], errors='coerce'), unit='s').dt.date
        mood2_df = mood2_df.dropna(subset=['date'])
        mood2_df['how']  = pd.to_numeric(mood2_df['how'], errors='coerce')
        mood2_win = _ema_fall_through(mood2_df, 'resp_time', ['how'])
        mood2_win = mood2_win.rename(columns={'how': 'mood_how'})
        daily_dfs.append(mood2_win)
        print(f"  Mood 2: {mood2_win.shape}")

    # EMA- Exercise: physical_exercise node
    # exercise = duration bucket. have = binary flag. walk = deliberate walking.
    # schedule dropped — conditional follow-up, no direct node mapping.
    print("Loading EMA Exercise...")
    ex_df = load_per_user_json(os.path.join(ema_path, 'Exercise'), ['exercise', 'have', 'walk', 'resp_time'])
    if not ex_df.empty:
        ex_df['date']     = pd.to_datetime(pd.to_numeric(ex_df['resp_time'], errors='coerce'), unit='s').dt.date
        ex_df = ex_df.dropna(subset=['date'])
        ex_df['exercise'] = pd.to_numeric(ex_df['exercise'], errors='coerce')
        ex_df['have']     = pd.to_numeric(ex_df['have'],     errors='coerce')
        ex_df['walk']     = pd.to_numeric(ex_df['walk'],     errors='coerce')
        ex_win = _ema_fall_through(ex_df, 'resp_time', ['exercise', 'have', 'walk'])
        ex_win = ex_win.rename(columns={'exercise': 'exercise_type', 'have': 'exercise_have', 'walk': 'exercise_walk'})
        daily_dfs.append(ex_win)
        print(f"  Exercise: {ex_win.shape}")

    # EMA- Events: positive/negative event intensity (1-7 scale).
    # pevent/nevent = free text descriptions — dropped (unstructured).
    # Events are inherently retrospective — no window propagation.
    # One row per (uid, date): the latest entry for that day.
    print("Loading EMA Events...")
    ev_df = load_per_user_json(os.path.join(ema_path, 'Events'), ['positive', 'negative', 'resp_time'])
    if not ev_df.empty:
        ev_df['date']     = pd.to_datetime(pd.to_numeric(ev_df['resp_time'], errors='coerce'), unit='s').dt.date
        ev_df = ev_df.dropna(subset=['date'])
        ev_df['_rts']     = pd.to_numeric(ev_df['resp_time'], errors='coerce')
        ev_df['positive'] = pd.to_numeric(ev_df['positive'], errors='coerce')
        ev_df['negative'] = pd.to_numeric(ev_df['negative'], errors='coerce')
        ev_daily = (
            ev_df.sort_values('_rts')
            .groupby(['uid', 'date'])[['positive', 'negative']]
            .last()
            .reset_index()
            .rename(columns={'positive': 'prev_day_events_positive', 'negative': 'prev_day_events_negative'})
        )
        daily_dfs.append(ev_daily)
        print(f"  Events: {ev_daily.shape}")

    # EMA- Study Spaces: energy_level (productivity 1-4 scale)
    # place dropped (location text). noise dropped (no direct node).
    print("Loading EMA Study Spaces...")
    study_df = load_per_user_json(os.path.join(ema_path, 'Study Spaces'), ['productivity', 'resp_time'])
    if not study_df.empty:
        study_df['date']         = pd.to_datetime(pd.to_numeric(study_df['resp_time'], errors='coerce'), unit='s').dt.date
        study_df = study_df.dropna(subset=['date'])
        study_df['productivity'] = pd.to_numeric(study_df['productivity'], errors='coerce')
        study_win = _ema_fall_through(study_df, 'resp_time', ['productivity'])
        daily_dfs.append(study_win)
        print(f"  Study Spaces: {study_win.shape}")

    # EMA- Sleep: sleep_duration, sleep_quality, energy_level nodes
    # hour = hours slept, rate = quality rating, social = daytime alertness proxy.
    # Sleep is inherently retrospective — no window propagation.
    # One row per (uid, date): the latest entry for that day.
    # Graceful empty handling: source files may contain only null keys.
    print("Loading EMA Sleep...")
    sleep_ema_df = load_per_user_json(os.path.join(ema_path, 'Sleep'), ['hour', 'rate', 'social', 'resp_time'])
    if not sleep_ema_df.empty:
        sleep_ema_df['date']   = pd.to_datetime(pd.to_numeric(sleep_ema_df['resp_time'], errors='coerce'), unit='s').dt.date
        sleep_ema_df = sleep_ema_df.dropna(subset=['date'])
        sleep_ema_df['_rts']   = pd.to_numeric(sleep_ema_df['resp_time'], errors='coerce')
        sleep_ema_df['hour']   = pd.to_numeric(sleep_ema_df['hour'],   errors='coerce')
        sleep_ema_df['rate']   = pd.to_numeric(sleep_ema_df['rate'],   errors='coerce')
        sleep_ema_df['social'] = pd.to_numeric(sleep_ema_df['social'], errors='coerce')
        sleep_daily = (
            sleep_ema_df.sort_values('_rts')
            .groupby(['uid', 'date'])[['hour', 'rate', 'social']]
            .last()
            .reset_index()
            .rename(columns={'hour': 'prev_day_sleep_ema_hours', 'rate': 'prev_day_sleep_ema_rating', 'social': 'prev_day_sleep_ema_alertness'})
        )
        daily_dfs.append(sleep_daily)
        print(f"  Sleep EMA: {sleep_daily.shape}")
    else:
        print("  Sleep EMA: no data found (null keys in source files).")

    # EMA- Stress: mental_stress node
    # level = self-reported stress level (1=a little stressed to 5=feeling great).
    # Graceful empty handling: source files may contain only null keys.
    print("Loading EMA Stress...")
    stress_ema_df = load_per_user_json(os.path.join(ema_path, 'Stress'), ['level', 'resp_time'])
    if not stress_ema_df.empty:
        stress_ema_df['date']  = pd.to_datetime(pd.to_numeric(stress_ema_df['resp_time'], errors='coerce'), unit='s').dt.date
        stress_ema_df = stress_ema_df.dropna(subset=['date'])
        stress_ema_df['level'] = pd.to_numeric(stress_ema_df['level'], errors='coerce')
        stress_win = _ema_fall_through(stress_ema_df, 'resp_time', ['level'])
        stress_win = stress_win.rename(columns={'level': 'stress_ema_level'})
        daily_dfs.append(stress_win)
        print(f"  Stress EMA: {stress_win.shape}")
    else:
        print("  Stress EMA: no data found (null keys in source files).")


    # ── PHASE 10: Merge all windowed data ─────────────────────────────────────
    # DataFrames with window_start join on ['uid','date','window_start'].
    # DataFrames without window_start (surveys) join on ['uid'].
    print("\nMerging all windowed data...")
    merged = all_windows_base[['uid','date','window_start']].copy()
    for df in daily_dfs:
        if 'window_start' in df.columns:
            join_keys = ['uid', 'date', 'window_start']
        elif 'date' in df.columns:
            join_keys = ['uid', 'date']   # daily-level df — broadcasts across all windows of that day
        else:
            join_keys = ['uid']           # survey-level df — broadcasts across all days for that uid
        merged = merged.merge(df, on=join_keys, how='left')
    print(f"Windowed merged shape: {merged.shape}")

    # ── window_start → hour (0-23) ───────────────────────────────────────────
    merged['window_start'] = pd.to_datetime(merged['window_start']).dt.hour
    merged = merged.rename(columns={'window_start': 'hour'})

    # ── PHASE 11: Next day Carry-forward ─────────────────────────────────────
    # Daily values from day D shifted to day D+1 as prev_* columns.
    # These broadcast as constants across all window rows for a given date.
    # Logic byte-for-byte identical to original — only the join target date changes.
    # nighttime_screen_minutes: at night, screen time = unlock time (dark unreliable).
    if 'nighttime_unlock_minutes' in merged.columns:
        merged['nighttime_screen_minutes'] = merged['nighttime_unlock_minutes']
    
    prev_day_src_cols = [c for c in [
        'nighttime_screen_minutes',
        'evening_running_tasks', 'evening_peak_running_tasks',
        'avg_running_tasks', 'peak_running_tasks',
        'daily_sedentary_ratio', 'daily_active_ratio',
        'daily_call_count', 'daily_call_duration_total',
        'daily_sms_count', 'daily_screen_time_minutes',
    ] if c in merged.columns]

    if prev_day_src_cols:
        prev_daily = (
            merged[['uid', 'date'] + prev_day_src_cols]
            .drop_duplicates(subset=['uid', 'date'])
            .copy()
        )
        prev_daily['date'] = (
            pd.to_datetime(prev_daily['date']) + pd.Timedelta(days=1)
        ).dt.date
        rename_map = {
            'nighttime_screen_minutes':    'yesterday_nighttime_screen_minutes',
            'evening_running_tasks':       'prev_evening_running_tasks',
            'evening_peak_running_tasks':  'prev_evening_peak_running_tasks',
            'avg_running_tasks':           'prev_day_avg_running_tasks',
            'peak_running_tasks':          'prev_day_peak_running_tasks',
            'daily_sedentary_ratio':       'prev_day_sedentary_ratio',
            'daily_active_ratio':          'prev_day_active_ratio',
            'daily_call_count':            'prev_day_call_count',
            'daily_call_duration_total':   'prev_day_call_duration_total',
            'daily_sms_count':             'prev_day_sms_count',
            'daily_screen_time_minutes':   'prev_day_total_screen_time_minutes',
        }
        rename_map = {k: v for k, v in rename_map.items() if k in prev_day_src_cols}
        prev_daily = prev_daily.rename(columns=rename_map)
        merged = merged.merge(prev_daily, on=['uid', 'date'], how='left')

    # Mood carry-forward — last recorded value of the day (not first).
    # groupby.last() returns last non-NaN per group, matching EMA fall-through semantics.
    mood_cf_cols = [c for c in ['mood_happy', 'mood_sad', 'mood_how'] if c in merged.columns]
    if mood_cf_cols:
        mood_last = (
            merged[['uid', 'date', 'hour'] + mood_cf_cols]
            .sort_values(['uid', 'date', 'hour'])
            .groupby(['uid', 'date'])[mood_cf_cols]
            .last()
            .reset_index()
        )
        mood_last['date'] = (pd.to_datetime(mood_last['date']) + pd.Timedelta(days=1)).dt.date
        mood_last = mood_last.rename(columns={c: f'prev_day_{c}' for c in mood_cf_cols})
        merged = merged.merge(mood_last, on=['uid', 'date'], how='left')

    # Exercise carry-forward — last recorded value of the day shifted to next day.
    exercise_cf_cols = [c for c in ['exercise_type', 'exercise_have', 'exercise_walk'] if c in merged.columns]
    if exercise_cf_cols:
        exercise_last = (
            merged[['uid', 'date', 'hour'] + exercise_cf_cols]
            .sort_values(['uid', 'date', 'hour'])
            .groupby(['uid', 'date'])[exercise_cf_cols]
            .last()
            .reset_index()
        )
        exercise_last['date'] = (pd.to_datetime(exercise_last['date']) + pd.Timedelta(days=1)).dt.date
        exercise_last = exercise_last.rename(columns={c: f'prev_day_{c}' for c in exercise_cf_cols})
        merged = merged.merge(exercise_last, on=['uid', 'date'], how='left')

    # Productivity carry-forward — last recorded value of the day shifted to next day.
    if 'productivity' in merged.columns:
        prod_last = (
            merged[['uid', 'date', 'hour', 'productivity']]
            .sort_values(['uid', 'date', 'hour'])
            .groupby(['uid', 'date'])[['productivity']]
            .last()
            .reset_index()
        )
        prod_last['date'] = (pd.to_datetime(prod_last['date']) + pd.Timedelta(days=1)).dt.date
        prod_last = prod_last.rename(columns={'productivity': 'prev_day_productivity'})
        merged = merged.merge(prod_last, on=['uid', 'date'], how='left')

    # Stress EMA carry-forward — last recorded value of the day shifted to next day.
    if 'stress_ema_level' in merged.columns:
        stress_last = (
            merged[['uid', 'date', 'hour', 'stress_ema_level']]
            .sort_values(['uid', 'date', 'hour'])
            .groupby(['uid', 'date'])[['stress_ema_level']]
            .last()
            .reset_index()
        )
        stress_last['date'] = (pd.to_datetime(stress_last['date']) + pd.Timedelta(days=1)).dt.date
        stress_last = stress_last.rename(columns={'stress_ema_level': 'prev_day_stress_ema_level'})
        merged = merged.merge(stress_last, on=['uid', 'date'], how='left')

    # Drop columns that cannot be known at runtime on the current day.
    # prev_* carry-forwards are already computed above and are retained.
    _drop_cols = [c for c in [
        'dark_minutes', 'nighttime_active_minutes', 'nighttime_unlock_minutes',
        'nighttime_screen_minutes',
        'unlocked_minutes',
        'evening_running_tasks', 'evening_peak_running_tasks',
        'avg_running_tasks', 'peak_running_tasks',
        'daily_sedentary_ratio', 'daily_active_ratio',
        'daily_call_count', 'daily_call_duration_total',
        'daily_sms_count', 'daily_screen_time_minutes',
    ] if c in merged.columns]
    merged.drop(columns=_drop_cols, inplace=True)

    # ── PHASE 12: Surveys (one-time baseline per uid) ─────────────────────────
    # load_surveys() and encode_survey_strings() are completely unchanged.
    # Surveys join on uid and broadcast across all window rows for that uid.
    surveys, survey_labels = load_surveys(survey_path)
    merged = merged.merge(surveys, on='uid', how='left')
    print(f"Final shape with surveys: {merged.shape}")

    # ── PHASE 13: Build and save complete column labels ───────────────────────
    # This file is the master reference for sub-dimension attribute nodes,
    # questions to ask at runtime, and column measurement definitions.
    ema_label_map = build_ema_label_map(EMA_DEF_PATH)

    ema_col_to_qid = {
        'mood_happy':                   'happy',
        'mood_sad':                     'sad',
        'mood_how':                     'how',
        'prev_day_mood_happy':          'happy',
        'prev_day_mood_sad':            'sad',
        'prev_day_mood_how':            'how',
        'exercise_type':                'exercise',
        'exercise_have':                'have',
        'exercise_walk':                'walk',
        'prev_day_exercise_type':       'exercise',
        'prev_day_exercise_have':       'have',
        'prev_day_exercise_walk':       'walk',
        'prev_day_events_positive':     'positive',
        'prev_day_events_negative':     'negative',
        'productivity':                 'productivity',
        'prev_day_productivity':        'productivity',
        'prev_day_sleep_ema_hours':     'hour',
        'prev_day_sleep_ema_rating':    'rate',
        'prev_day_sleep_ema_alertness': 'social',
        'stress_ema_level':             'level',
        'prev_day_stress_ema_level':    'level',
    }

    sensing_labels = {
        'hour':                                'Beginning hour of the window (0-23, midnight-anchored)',
        'sedentary_ratio':                     'Proportion of window spent stationary (accelerometer — unknowns excluded)',
        'active_ratio':                        'Proportion of window spent walking or running (accelerometer — unknowns excluded)',
        'dark_window_minutes':                 'Dark sensor minutes within this window (all hours; NaN only in dead windows)',
        'unlocked_window_minutes':             'Phone unlocked minutes within this window (all hours; 0 in alive windows with no unlock activity)',
        'screen_time_window_minutes':          'Estimated screen-on minutes within this window (derived from dark + phonelock)',
        'call_count':                          'Phone calls within this window slot (call log — 0 in alive windows with no calls)',
        'call_duration_total':                 'Total call duration within this window slot in seconds (call log)',
        'sms_count':                           'SMS messages within this window slot — sent + received (0 in alive windows with no messages)',
        'avg_running_tasks_window':            'Mean running app tasks within this window slot (0 in alive windows with no readings)',
        'yesterday_nighttime_screen_minutes':  'Estimated screen-on minutes during previous night 20:00-02:00 (carry-forward from prior day)',
        'prev_evening_running_tasks':          'Mean running app tasks during previous evening 20:00-01:59 (carry-forward from prior day)',
        'prev_evening_peak_running_tasks':     'Peak running app tasks during previous evening 20:00-01:59 (carry-forward from prior day)',
        'prev_day_avg_running_tasks':          'Mean running app tasks on previous day (carry-forward from prior day)',
        'prev_day_peak_running_tasks':         'Peak running app tasks on previous day (carry-forward from prior day)',
        'prev_day_sedentary_ratio':            'Proportion of previous day spent stationary (carry-forward from prior day)',
        'prev_day_active_ratio':               'Proportion of previous day spent walking or running (carry-forward from prior day)',
        'prev_day_call_count':                 'Total phone calls on previous day (carry-forward from prior day)',
        'prev_day_call_duration_total':        'Total call duration in seconds on previous day (carry-forward from prior day)',
        'prev_day_sms_count':                  'Total SMS messages on previous day — sent + received (carry-forward from prior day)',
        'prev_day_total_screen_time_minutes':  'Estimated total screen-on minutes on previous day (carry-forward from prior day)',
        'prev_day_mood_happy':                 'Last recorded happy rating on previous day (carry-forward from prior day)',
        'prev_day_mood_sad':                   'Last recorded sad rating on previous day (carry-forward from prior day)',
        'prev_day_mood_how':                   'Last recorded mood_how rating on previous day (carry-forward from prior day)',
        'prev_day_exercise_type':              'Last recorded exercise duration bucket on previous day (carry-forward from prior day)',
        'prev_day_exercise_have':              'Last recorded exercise binary flag on previous day (carry-forward from prior day)',
        'prev_day_exercise_walk':              'Last recorded deliberate walking flag on previous day (carry-forward from prior day)',
        'prev_day_productivity':               'Last recorded productivity rating on previous day (carry-forward from prior day)',
        'prev_day_stress_ema_level':           'Last recorded stress level on previous day (carry-forward from prior day)',
    }

    all_studentlife_labels = {}
    for col in merged.columns:
        if col in ['uid', 'date']:
            all_studentlife_labels[col] = 'Participant / date identifier'
        elif col in sensing_labels:
            all_studentlife_labels[col] = sensing_labels[col]
        elif col in ema_col_to_qid:
            qid = ema_col_to_qid[col]
            # Returns question text from EMA JSON; fallback stored if question_id not found
            all_studentlife_labels[col] = ema_label_map.get(qid, f'EMA question_id: {qid}')
        elif col in survey_labels:
            all_studentlife_labels[col] = survey_labels[col]
        else:
            all_studentlife_labels[col] = col

    labels_df = pd.DataFrame(list(all_studentlife_labels.items()), columns=['column', 'label'])
    labels_df.to_csv(r'C:\Users\udbha\Documents\VS Code\MedApp\datasets\studentlife_self\studentlife_column_labels.csv', index=False)
    print("Saved studentlife_column_labels.csv")

    merged.to_csv(r'C:\Users\udbha\Documents\VS Code\MedApp\datasets\studentlife_self\studentlife_daily_and_surveys.csv', index=False)
    print("Saved to studentlife_daily_and_surveys.csv")
    return merged
# -----------------------------------------------------------------------------


## LIFESNAPS ##

LIFESNAPS_COLS = [
    'id', 'date', 'hour',
    'steps', 'bpm',
    'age', 'gender', 'bmi',
    'ALERT', 'HAPPY', 'NEUTRAL', 'RESTED/RELAXED',
    'SAD', 'TENSE/ANXIOUS', 'TIRED',
]

LIFESNAPS_EMA_COLS   = ['ALERT', 'HAPPY', 'NEUTRAL', 'RESTED/RELAXED', 'SAD', 'TENSE/ANXIOUS', 'TIRED']
LIFESNAPS_CARRY_COLS = ['hourly_ALERT', 'hourly_HAPPY', 'hourly_NEUTRAL', 'hourly_RESTED/RELAXED',
                        'hourly_SAD', 'hourly_TENSE/ANXIOUS', 'hourly_TIRED']

LIFESNAPS_RENAME = {
    'steps'          : 'hourly_steps',
    'bpm'            : 'hourly_bpm',
    'age'            : 'hourly_age',
    'gender'         : 'hourly_gender',
    'bmi'            : 'hourly_bmi',
    'ALERT'          : 'hourly_ALERT',
    'HAPPY'          : 'hourly_HAPPY',
    'NEUTRAL'        : 'hourly_NEUTRAL',
    'RESTED/RELAXED' : 'hourly_RESTED/RELAXED',
    'SAD'            : 'hourly_SAD',
    'TENSE/ANXIOUS'  : 'hourly_TENSE/ANXIOUS',
    'TIRED'          : 'hourly_TIRED',
}

def load_lifesnaps_hourly():
    fpath = os.path.join(LIFESNAPS_PATH, 'hourly_fitbit_sema_df_unprocessed.csv')
    hourly = pd.read_csv(fpath, low_memory=False)[LIFESNAPS_COLS]

    hourly['date'] = pd.to_datetime(hourly['date']).dt.date
    hourly = hourly.rename(columns=LIFESNAPS_RENAME)
    hourly = hourly.sort_values(['id', 'date', 'hour']).reset_index(drop=True)

    hourly[LIFESNAPS_CARRY_COLS] = hourly.groupby(['id', 'date'])[LIFESNAPS_CARRY_COLS].ffill()

    return hourly


LIFESNAPS_DAILY_COLS = [
    'id', 'date',
    'nightly_temperature', 'spo2', 'stress_score', 'daily_temperature_variation',
    'bpm', 'lightly_active_minutes', 'moderately_active_minutes',
    'very_active_minutes', 'sedentary_minutes', 'scl_avg', 'resting_hr',
    'minutesAsleep', 'steps', 'age', 'gender', 'bmi',
    'ALERT', 'HAPPY', 'NEUTRAL', 'RESTED/RELAXED', 'SAD', 'TENSE/ANXIOUS', 'TIRED',
]

LIFESNAPS_DAILY_RENAME = {
    'nightly_temperature'        : 'prev_night_temperature',
    'spo2'                       : 'prev_night_spo2',
    'stress_score'               : 'daily_stress_score',
    'daily_temperature_variation': 'prev_night_temp_variation',
    'bpm'                        : 'daily_bpm',
    'scl_avg'                    : 'daily_scl_avg',
    'resting_hr'                 : 'prev_night_resting_hr',
    'minutesAsleep'              : 'prev_night_minutesAsleep',
    'steps'                      : 'daily_steps',
    'age'                        : 'daily_age',
    'gender'                     : 'daily_gender',
    'bmi'                        : 'daily_bmi',
    'ALERT'                      : 'daily_ALERT',
    'HAPPY'                      : 'daily_HAPPY',
    'NEUTRAL'                    : 'daily_NEUTRAL',
    'RESTED/RELAXED'             : 'daily_RESTED/RELAXED',
    'SAD'                        : 'daily_SAD',
    'TENSE/ANXIOUS'              : 'daily_TENSE/ANXIOUS',
    'TIRED'                      : 'daily_TIRED',
}


def load_lifesnaps_daily():
    fpath = os.path.join(LIFESNAPS_PATH, 'daily_fitbit_sema_df_unprocessed.csv')
    daily = pd.read_csv(fpath, usecols=LIFESNAPS_DAILY_COLS)

    daily['date'] = pd.to_datetime(daily['date']).dt.date

    activity_cols = ['lightly_active_minutes', 'moderately_active_minutes',
                     'very_active_minutes', 'sedentary_minutes']
    daily[activity_cols] = daily[activity_cols].fillna(0)
    daily['daily_active_ratio']    = (
        (daily['lightly_active_minutes'] + daily['moderately_active_minutes'] + daily['very_active_minutes'])
        / 1440
    ).clip(0, 1)
    daily['daily_sedentary_ratio'] = 1 - daily['daily_active_ratio'] # Done because when a person removes the fitbit, they generally remove it when they are not moving much (sedentary time).
    daily.drop(columns=activity_cols, inplace=True)

    daily = daily.rename(columns=LIFESNAPS_DAILY_RENAME)
    daily = daily.sort_values(['id', 'date']).reset_index(drop=True)

    return daily


# Columns to coalesce: daily wins, hourly fills gaps → renamed to today_*
LIFESNAPS_COALESCE_COLS = ['age', 'gender', 'bmi',
                            'ALERT', 'HAPPY', 'NEUTRAL', 'RESTED/RELAXED',
                            'SAD', 'TENSE/ANXIOUS', 'TIRED']
LIFESNAPS_PREV_DAY_COLS     = ['daily_stress_score', 'daily_bpm', 'daily_scl_avg', 'daily_steps']
LIFESNAPS_PREV_DAY_EMA_COLS = ['daily_ALERT', 'daily_HAPPY', 'daily_NEUTRAL', 'daily_RESTED/RELAXED',
                                'daily_SAD', 'daily_TENSE/ANXIOUS', 'daily_TIRED']

def load_lifesnaps():
    hourly = load_lifesnaps_hourly()
    daily  = load_lifesnaps_daily()

    # Build prev_day_* by shifting each id's daily values forward by one day
    all_shift_cols = LIFESNAPS_PREV_DAY_COLS + LIFESNAPS_PREV_DAY_EMA_COLS
    prev_day = daily[['id', 'date'] + all_shift_cols].copy()
    prev_day = prev_day.sort_values(['id', 'date'])
    prev_day[all_shift_cols] = prev_day.groupby('id')[all_shift_cols].shift(1)
    prev_day = prev_day.rename(columns={c: c.replace('daily_', 'prev_day_') for c in all_shift_cols})

    # Broadcast daily columns onto every hourly row; join prev_day separately
    merged = hourly.merge(daily, on=['id', 'date'], how='left')
    merged = merged.merge(prev_day, on=['id', 'date'], how='left')

    # Drop only cols that were shifted + replaced (not EMA — those stay as daily_*)
    merged.drop(columns=LIFESNAPS_PREV_DAY_COLS, inplace=True)

    # Coalesce: daily wins, hourly fills gaps → today_*
    for col in LIFESNAPS_COALESCE_COLS:
        if col not in ['age', 'gender', 'bmi']:
            daily_col  = f'daily_{col}'
            hourly_col = f'hourly_{col}'
            merged[f'today_{col}'] = merged[daily_col].combine_first(merged[hourly_col])
            merged.drop(columns=[daily_col, hourly_col], inplace=True)
        # age, gender and bmi should not be named like today_*.
        else:
            daily_col  = f'daily_{col}'
            hourly_col = f'hourly_{col}'
            merged[col] = merged[daily_col].combine_first(merged[hourly_col])
            merged.drop(columns=[daily_col, hourly_col], inplace=True)

    merged = merged.sort_values(['id', 'date', 'hour']).reset_index(drop=True)
    merged.to_csv(r'C:\Users\udbha\Documents\VS Code\MedApp\datasets\lifesnaps_self\lifesnaps_final.csv', index=False)
    return merged


if __name__ == '__main__':
    # NHANES
    merged_nhanes, all_labels = merge_nhanes()
    nhanes_final = clean_nhanes(merged_nhanes, all_labels)

    # StudentLife
    studentlife_final = load_studentlife()

    # LifeSnaps merged
    lifesnaps_final = load_lifesnaps()

    # Print final column list for both datasets
    print("\n" + "="*60)
    print("FINAL COLUMN LIST")
    print("="*60)
    print(f"\nNHANES columns ({len(nhanes_final.columns)}):")
    for col in nhanes_final.columns:
        print(f"  {col}")
    print(f"\nStudentLife columns ({len(studentlife_final.columns)}):")
    for col in studentlife_final.columns:
        print(f"  {col}")
    print(f"\nLifeSnaps columns ({len(lifesnaps_final.columns)}):")
    for col in lifesnaps_final.columns:
        print(f"  {col}")