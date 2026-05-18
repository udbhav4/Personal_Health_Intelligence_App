"""
Convert configs/column_question_map.csv → configs/column_question_map.json
for injection into the on-device SLM NER prompt (Level 1 routing).

Filters to self_report=True nodes only. Drops prev_day_* / prev_night_*
carry-forward rows and passive sensor columns (no user-facing question).

Run once after any CSV change:
  python -m codebase.backend.export_column_question_map

Output format:
  {
    "original_column": {
      "c": "composite_or_direct_source_col",
      "n": "dbn_node_name",
      "q": "question label shown to participant",
      "opts": [{"v": 0, "l": "Label"}, ...],   // discrete — chips or picker
      // OR
      "range": {"min": 0, "max": 12, "unit": "hours"}  // continuous — slider
    },
    ...
  }
"""

import csv
import json
import re
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / 'configs'

# Range overrides — force slider for these columns regardless of CSV options_json.
_RANGE_OVERRIDES: dict[str, dict] = {
    'prev_day_events_positive': {'min': 1, 'max': 7, 'unit': 'intensity'},
    'prev_day_events_negative': {'min': 1, 'max': 7, 'unit': 'intensity'},
}

# Passive-sensor label fragments — rows whose question_label contains any of
# these are training annotations, not user-facing questions, and are excluded.
_SENSOR_FRAGMENTS = (
    'accelerometer', 'fitbit', 'carry-forward',
    'minutes within', 'proportion of', 'sensor',
    'beats per minute', 'step count', 'heart rate',
    'skin temperature', 'resting heart rate',
)


def _build_source_col_to_node(node_cfg: dict) -> dict[str, str]:
    """Reverse-map source_column_bins keys → node_name for self_report nodes."""
    mapping: dict[str, str] = {}
    for node_name, node in node_cfg.items():
        if not node.get('self_report', False):
            continue
        for col in node.get('source_column_bins', {}).keys():
            mapping[col] = node_name
    return mapping


def main() -> None:
    with open(CONFIGS / 'feature_node_config.json', encoding='utf-8') as f:
        node_cfg: dict = json.load(f)['nodes']

    source_col_to_node = _build_source_col_to_node(node_cfg)

    result: dict[str, dict] = {}

    with open(CONFIGS / 'column_question_map.csv', newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            orig     = row['original_column'].strip()
            comp     = row['composite_column'].strip()
            harm     = row['harmonized_column'].strip()
            question = row['question_label'].strip()

            # Drop empty or sensor-only question labels (carry-forward labels
            # contain "carry-forward"; passive sensor labels contain "sensor",
            # "accelerometer", "Fitbit", etc. — this replaces a brittle prefix
            # filter so valid prev_day_* EMA questions are not lost).
            q_lower = question.lower()
            if not question or any(frag in q_lower for frag in _SENSOR_FRAGMENTS):
                continue

            # Determine source column: composite takes priority over harmonized
            source_col = comp if comp else harm

            # Primary lookup: source_column_bins reverse map
            node_name = source_col_to_node.get(source_col)

            # Fallback: composite column name IS the node name directly
            # (e.g. pain_level has source_column_bins keyed by CDQ001/CDQ010,
            # not by "pain_level", so the reverse map misses it).
            if not node_name and node_cfg.get(source_col, {}).get('self_report'):
                node_name = source_col

            if not node_name:
                continue

            clean_q = re.sub(r'\s*[Oo]ptions:.*$', '', question).strip()

            entry: dict = {
                'c': source_col,
                'n': node_name,
                'q': clean_q,
            }

            if orig in _RANGE_OVERRIDES:
                entry['range'] = _RANGE_OVERRIDES[orig]
            else:
                opts_raw = row.get('options_json', '').strip()
                if opts_raw.startswith('['):
                    try:
                        pairs = json.loads(opts_raw)
                        entry['opts'] = [{'v': v, 'l': lbl} for v, lbl in pairs]
                    except (json.JSONDecodeError, ValueError, TypeError):
                        pass
                elif opts_raw.startswith('{'):
                    try:
                        entry['range'] = json.loads(opts_raw)
                    except json.JSONDecodeError:
                        pass

            result[orig] = entry

    out_path = CONFIGS / 'column_question_map.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f'Wrote {len(result)} entries to {out_path}')


if __name__ == '__main__':
    main()
