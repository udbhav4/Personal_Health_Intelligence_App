---
name: DBN Retraining Action Plan
description: Normalization overhaul — remove z-score for bounded/ordinal cols, switch sensor cols to population z-score, retrain DBN
type: project
originSessionId: ca1acd62-ff62-45a4-86c7-72acedf97ea9
---
## Decision

Training used per-user z-score for all `_NORMALIZE_COLS`. This is wrong for two categories:
- **Bounded ordinal self-report EMA** — scale already IS the normalization; per-user z-score destroys absolute meaning
- **Sensor cols** — need population z-score (not per-user) because population norms exist and chronic high values must register as high

**Why:** Per-user z-score for screen/call cols means a chronically high user always appears neutral to the DBN — chronic harm invisible to the model.

---

## Action Plan (execute in order)

### Step 1 — Remove normalization for bounded/ordinal cols, switch to clinical thresholds

Cols to remove from `_NORMALIZE_COLS` in `data_normalization.py`:
- `stress_ema_level`, `prev_day_stress_ema_level`
- `mood_how`, `mood_happy`, `mood_sad`
- `prev_day_mood_how`, `prev_day_mood_happy`, `prev_day_mood_sad`
- `productivity`, `prev_day_productivity`
- `prev_day_events_positive`, `prev_day_events_negative`
- `active_ratio`, `prev_day_active_ratio` (0–1 bounded ratio, WHO norms exist)
- `prev_day_sleep_ema_rating`, `prev_day_sleep_ema_alertness` (bounded ordinal)

In `data_discretization.py`: change method from `kmeans` to `clinical` for these cols.
Define ordinal thresholds based on scale ranges (e.g. stress_ema_level 1–4 → thresholds [1,2,3,4] → low/moderate_low/moderate_high/high).
Bin_edges in `feature-node-config.json` replaced with `thresholds` for these cols.

### Step 2 — Population z-score script for sensor cols

Write a new script: `Codebase/training/statistical_data_preprocessing/export_population_norm_stats.py`

Hook into pipeline AFTER harmonize, BEFORE normalize_per_user:
```python
# Call existing functions:
load_* → clean_* → prune_* → harmonize_*
# On the raw harmonized df (before normalize_per_user):
# compute mean + std per column across ALL users
# export to configs/population_norm_stats.json
```

Cols to compute stats for (keep z-score, switch to population):
- `screen_time_window_minutes`, `dark_window_minutes`, `unlocked_window_minutes`
- `yesterday_nighttime_screen_minutes`, `prev_day_total_screen_time_minutes`
- `call_count`, `call_duration_total`, `sms_count`
- `prev_day_call_count`, `prev_day_call_duration_total`, `prev_day_sms_count`
- `avg_running_tasks_window`, `prev_day_avg_running_tasks`, `prev_day_peak_running_tasks`
- `hourly_steps`, `prev_day_steps`
- `prev_night_resting_hr`, `prev_night_temperature`

Output format: `{ "col_name": { "mean": float, "std": float }, ... }`

Also add step 2.5 in `main.py`: run this script between harmonize and normalize_per_user.

### Step 3 — Change normalize_per_user to population z-score for kept sensor cols

In `data_normalization.py`: for the sensor cols above, replace per-user groupby z-score with population-wide mean/std (computed in Step 2). These cols still get z-scored in training data — just population-wide instead of per-user. Bin_edges from k-means will then be in population z-score space.

### Step 4 — Re-run full pipeline and retrain DBN

Run `main.py` → new bin_edges, new CPDs, new `feature-node-config.json`.

### Step 5 — Runtime fix for sensor cols

At runtime in sensor collectors: `z = (raw_value - pop_mean) / pop_std` using stats from `population_norm_stats.json` → compare against bin_edge. Replace current cold-start absolute threshold with population z-score from day 1. Personal history z-score used only as confidence boost (not for discretization).

**Why:** population z-score = consistent with how the retrained model was trained.

---

## Runtime confidence assignment (Step 3 of original plan)

`toPriorVector` in `evidenceLayer.ts` already handles this correctly.
- confidence < 0.85 → soft prior vector (interpolation between observed state and marginal prior)
- confidence >= 0.85 → hard evidence

Assign confidence based on:
- Population z-score classification only (no personal history): confidence = 0.65 (cold start)
- Population z-score + personal history agrees: confidence = 0.90
- Population z-score + personal history disagrees: confidence = 0.75 (population wins, reduced certainty)

---

## Files to change

| File | Change |
|------|--------|
| `data_normalization.py` | Remove bounded/ordinal cols from `_NORMALIZE_COLS`; switch sensor cols to population z-score |
| `data_discretization.py` | Add clinical thresholds for removed cols; update method field |
| `main.py` | Add Step 2.5 — run export_population_norm_stats before normalize |
| `export_population_norm_stats.py` | New script — compute + export population mean/std |
| `feature-node-config.json` | Auto-updated by retraining pipeline |
| `evidenceLayer.ts` / sensor collectors | Use population_norm_stats.json for runtime z-score |
