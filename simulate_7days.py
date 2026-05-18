"""
7-day DBN simulation.

Inserts fake sensorless evidence into the DB day-by-day (worsening pattern),
calls POST /infer after each day, and prints a belief table for key nodes.

Run with server already started:
  python simulate_7days.py
"""

import json
import sqlite3
import sys
import time
from datetime import datetime, timezone

import httpx

SERVER   = 'http://localhost:8000'
DB_PATH  = 'database/medapp.db'
KEY_NODES = ['mental_stress', 'depression', 'mood', 'stress_ema', 'sleep_quality']

# ── Per-day evidence: worsening over 7 days ───────────────────────────────────
# Each entry: (node_name, node_value, confidence, data_source, merge_mode, temporal_flag)
DAY_EVIDENCE = {
    1: [
        ('mood',            'high',          0.90, 'self_report', 'latest', 'decaying'),
        ('sleep_quality',   'good',          0.90, 'self_report', 'latest', 'decaying'),
        ('stress_ema',      'low',           0.90, 'self_report', 'latest', 'decaying'),
        ('depression',      'none',          0.90, 'self_report', 'latest', 'decaying'),
        ('neuroticism',     'low',           0.88, 'onboarding',  'latest', 'persistent'),
        ('loneliness',      'low',           0.88, 'onboarding',  'latest', 'persistent'),
        ('negative_affect', 'low',           0.88, 'self_report', 'latest', 'decaying'),
        ('positive_affect', 'high',          0.88, 'self_report', 'latest', 'decaying'),
    ],
    2: [
        ('mood',            'low',           0.90, 'self_report', 'latest', 'decaying'),
        ('sleep_quality',   'fair',          0.90, 'self_report', 'latest', 'decaying'),
        ('stress_ema',      'moderate_low',  0.90, 'self_report', 'latest', 'decaying'),
        ('depression',      'mild',          0.90, 'self_report', 'latest', 'decaying'),
    ],
    3: [
        ('mood',            'low',           0.92, 'self_report', 'latest', 'decaying'),
        ('sleep_quality',   'poor',          0.92, 'self_report', 'latest', 'decaying'),
        ('stress_ema',      'moderate_high', 0.92, 'self_report', 'latest', 'decaying'),
        ('depression',      'moderate',      0.92, 'self_report', 'latest', 'decaying'),
        ('negative_affect', 'moderate_high', 0.90, 'self_report', 'latest', 'decaying'),
        ('positive_affect', 'low',           0.90, 'self_report', 'latest', 'decaying'),
        ('loneliness',      'moderate',      0.88, 'self_report', 'latest', 'persistent'),
    ],
    4: [
        ('mood',            'low',           0.93, 'self_report', 'latest', 'decaying'),
        ('sleep_quality',   'poor',          0.93, 'self_report', 'latest', 'decaying'),
        ('stress_ema',      'high',          0.93, 'self_report', 'latest', 'decaying'),
        ('depression',      'moderate_severe', 0.93, 'self_report', 'latest', 'decaying'),
    ],
    5: [
        ('mood',            'low',           0.95, 'self_report', 'latest', 'decaying'),
        ('sleep_quality',   'poor',          0.95, 'self_report', 'latest', 'decaying'),
        ('stress_ema',      'high',          0.95, 'self_report', 'latest', 'decaying'),
        ('depression',      'severe',        0.95, 'self_report', 'latest', 'decaying'),
        ('negative_affect', 'high',          0.93, 'self_report', 'latest', 'decaying'),
        ('positive_affect', 'low',           0.93, 'self_report', 'latest', 'decaying'),
        ('loneliness',      'high',          0.90, 'self_report', 'latest', 'persistent'),
    ],
    6: [
        ('mood',            'low',           0.95, 'self_report', 'latest', 'decaying'),
        ('sleep_quality',   'poor',          0.95, 'self_report', 'latest', 'decaying'),
        ('stress_ema',      'high',          0.95, 'self_report', 'latest', 'decaying'),
        ('depression',      'severe',        0.95, 'self_report', 'latest', 'decaying'),
        ('neuroticism',     'high',          0.90, 'onboarding',  'latest', 'persistent'),
    ],
    7: [
        ('mood',            'low',           0.95, 'self_report', 'latest', 'decaying'),
        ('sleep_quality',   'poor',          0.95, 'self_report', 'latest', 'decaying'),
        ('stress_ema',      'high',          0.95, 'self_report', 'latest', 'decaying'),
        ('depression',      'severe',        0.95, 'self_report', 'latest', 'decaying'),
        ('negative_affect', 'high',          0.95, 'self_report', 'latest', 'decaying'),
        ('positive_affect', 'low',           0.95, 'self_report', 'latest', 'decaying'),
        ('loneliness',      'high',          0.93, 'self_report', 'latest', 'persistent'),
        ('neuroticism',     'high',          0.93, 'onboarding',  'latest', 'persistent'),
    ],
}


def _wipe_test_data(conn):
    conn.execute("DELETE FROM user_data_sensorless WHERE data_source IN ('self_report','onboarding','ema')")
    conn.execute("DELETE FROM inference_snapshots")
    conn.commit()
    print('Cleared existing test data.\n')


def _insert_day(conn, day: int):
    # Always use "now" so no row is stale regardless of node staleness window.
    # Deactivate previous rows for the same node so only the latest counts.
    now_ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    rows   = DAY_EVIDENCE.get(day, [])
    for node_name, node_value, confidence, data_source, merge_mode, temporal_flag in rows:
        conn.execute(
            "UPDATE user_data_sensorless SET is_active = 0 WHERE node_name = ?",
            (node_name,),
        )
        conn.execute(
            """
            INSERT INTO user_data_sensorless
              (timestamp, node_name, node_value, confidence, data_source,
               merge_mode, temporal_flag, is_active, was_proactive, answered, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1, 0, 1, ?)
            """,
            (now_ts, node_name, node_value, confidence, data_source,
             merge_mode, temporal_flag, now_ts),
        )
    conn.commit()


def _call_infer() -> dict:
    r = httpx.post(f'{SERVER}/infer', json={'trigger_type': 'scheduled'}, timeout=30)
    r.raise_for_status()
    return r.json()


def _fmt_belief(beliefs: dict, node: str) -> str:
    b = beliefs.get(node, {})
    if not b:
        return '—'
    top = max(b, key=b.get)
    return f'{top}({b[top]:.2f})'


def main():
    print('=== 7-Day DBN Simulation ===\n')

    # Check server
    try:
        httpx.get(f'{SERVER}/health', timeout=5).raise_for_status()
    except Exception as e:
        sys.exit(f'Server not reachable at {SERVER}: {e}')

    conn = sqlite3.connect(DB_PATH)
    _wipe_test_data(conn)

    header = f"{'Day':<5}" + ''.join(f'{n:<30}' for n in KEY_NODES)
    print(header)
    print('-' * len(header))

    for day in range(1, 8):
        _insert_day(conn, day)
        time.sleep(1)          # ensure unique snapshot_time per call
        result  = _call_infer()
        beliefs = result.get('beliefs', {})

        row = f'{day:<5}' + ''.join(f'{_fmt_belief(beliefs, n):<30}' for n in KEY_NODES)
        print(row)

    conn.close()
    print('\nDone. Check inference_snapshots table for full posteriors.')


if __name__ == '__main__':
    main()
