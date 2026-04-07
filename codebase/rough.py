import pandas as pd
import numpy as np
import pyreadstat
import os
import re
import json


BASE             = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NHANES_PATH      = os.path.join(BASE, 'datasets', 'NHANES')
STUDENTLIFE_PATH = os.path.join(BASE, 'datasets', 'studentlife', 'dataset')
LIFESNAPS_PATH = os.path.join(BASE, 'datasets', 'lifesnaps', 'rais_anonymized', 'csv_rais_anonymized')

# EMA definition JSON — maps question_id to question_text for all EMA folders.
EMA_DEF_PATH = r'C:\Users\udbha\OneDrive\Documents\VS Code\MedApp\datasets\studentlife\dataset\EMA\EMA_definition.json'

WINDOW_MINUTES = 1440

# Output paths
# NHANES_OUT_PATH      = r'C:\Users\udbha\OneDrive\Documents\VS Code\MedApp\datasets\nhanes_self'
# STUDENTLIFE_OUT_PATH = r'C:\Users\udbha\OneDrive\Documents\VS Code\MedApp\datasets\studentlife_self'


# HELPER FUNCTIONS #

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


def test():
    files = {
        'daily':  'daily_fitbit_sema_df_unprocessed.csv',
        'hourly': 'hourly_fitbit_sema_df_unprocessed.csv',
    }

    hourly_features = [
    # --- Structural & Temporal ---
    'id',               # Rename to 'uid' in your script
    'date',             # YYYY-MM-DD
    'hour',             # Used for window_start and Sin/Cos encoding
    
    # --- Physiological Signals (Continuous) ---
    'bpm',              # Heart Rate
    'temperature',      # Skin Temperature
    'calories',         # Metabolic Burn
    
    # --- Behavioral Signals (Continuous/Boolean) ---
    'steps',            # Physical Movement
    'distance',         # Distance Traveled
    'mindfulness_session', # Boolean (Intervention Node)
    
    # --- Mental Health & Context Labels (Sparse/EMA) ---
    'HAPPY',            # Mood Label
    'SAD',              # Mood Label
    'NEUTRAL',          # Mood Label
    'ALERT',            # Mood Label
    'TENSE/ANXIOUS',    # Mood Label
    'TIRED',            # Mood Label
    'HOME',             # Context Label
    'WORK/SCHOOL',      # Context Label
    'OUTDOORS',         # Context Label
    
    # --- Demographic Priors (For NHANES Mapping) ---
    'age',              # Demographic Anchor
    'gender',           # Demographic Anchor
    'bmi'               # Demographic Anchor
    ]

    columns = {
        'daily':  ['uid', 'window_start', 'sleep_duration', 'steps', 'active_minutes', 'calories_burned'],
        'hourly': hourly_features
    }

    # Fetching and adjusting the hourly lifesnaps data.
    df_hourly = pd.read_csv(os.path.join(LIFESNAPS_PATH, files['hourly']))
    df_hourly = df_hourly[columns['hourly']].copy()
    df_hourly["date"] = pd.to_datetime(df_hourly["date"]).dt.date




if __name__ == '__main__':
    test()