# Backend Fix Plan 2 — Post-Fix Regressions & Remaining Issues

---

## CRITICAL

### C1-REG — `initModels.ts` never imported; embed never initialized
**File:** `Codebase/mobile/App.tsx`
**Problem:** `initModels.ts` was created but `App.tsx` still calls `initNlu` and `initAgent`
directly using old pattern. `initEmbed` is never called anywhere. `_modelPath` in `embed.ts`
stays null. `ensureEmbed()` throws `'embed: initEmbed() has not been called'`. Every `startTurn`
crashes on `get_user_memory` → `searchMemory` → `embedText`.
**Fix:** Replace old init calls in `App.tsx` with import + call to `initModels(nluPath, embedPath, agentPath)`.
Add `EMBED_MODEL_PATH` constant. Guard init behind path validity check with user-visible error if paths missing.

### C1-B — `inter_trans` wrapped as `[1×k]` not `[k×k]` — temporal priors wrong for 20 nodes
**File:** `Codebase/backend/export_cpds.py`, `configs/cpd_tables.json`
**Problem:** C1 fix wraps 1D result as `[result]` producing `[1×k]` matrix. `applyInterSlice`
expects `[k×k]`. `trans.map(row => row.reduce(...))` on `[1×k]` produces `[scalar]` →
`normalize([scalar]) = [1.0]` → `safeVec([1.0], k)` returns uniform `Array(k).fill(1/k)`.
6 nodes lose real distributional data: `mental_health`, `loneliness`, `extraversion`,
`neuroticism`, `sleep_disturbances`, `social_events_negative`.
**Fix:** In `export_cpds.py`, for nodes with no t=1 parent, construct a proper `[k×k]` identity
matrix (or diagonal from the node's marginal prior) instead of wrapping 1D as `[1×k]`.
Re-run `export_cpds.py` + `sync_configs.py`.

---

## HIGH

### H1 — `fetchPrevDaySnapshot` queries wrong column — always returns empty
**File:** `core/evidenceLayer.ts` `fetchPrevDaySnapshot()`
**Problem:** Queries `WHERE node_name LIKE 'prev_day_%'`. But `computePrevDayAggregates`
writes `node_name = original_node_name` (e.g. `'activity'`) and prefixes only `source_column`
with `prev_day_`. No row ever has `node_name LIKE 'prev_day_%'`. Entire M15 prev-day fix dead.
**Fix:** Change query to `WHERE source_column LIKE 'prev_day_%'` to correctly find prev_day rows.

### H2 — `completeTurn` passes item column to `maybeResolveComposite` instead of composite source column
**File:** `core/agent.ts` `completeTurn()` ~line 889
**Problem:** Calls `maybeResolveComposite(db, ans.node_name, ans.original_col, turnId)`.
Third arg must be the composite source column (e.g. `'phq_total'`), but `ans.original_col`
is the item column (e.g. `'phq_sleep'`). `COMPOSITE_ITEM_COUNT['phq_sleep']` is undefined →
function returns immediately. PHQ-9 composite scoring never fires via cascade answers.
**Fix:** Pass `_cqMap[ans.original_col]?.c` (the source/composite column) as the third argument,
not `ans.original_col` directly.

---

## MEDIUM

### M1 — `snapshot_time` format mismatch between `infer.ts` and `mcp.ts`
**File:** `core/infer.ts` line ~107, `core/mcp.ts` line ~197
**Problem:** `infer.ts` writes `snapshot_time` as `HH:MM:SS` (via `toLocaleTimeString('sv')`).
`mcp.ts` writes `YYYY-MM-DDTHH:MM:SS` (via `toLocaleString('sv').replace(' ','T')`).
Same `snapshot_time` column, part of composite PRIMARY KEY. `ORDER BY date DESC, snapshot_time DESC`
lexicographically ranks ISO-datetime strings above plain time strings → wrong temporal ordering
when rows from both paths exist.
**Fix:** Standardize both to same format. Use `toLocaleString('sv').replace(' ', 'T').slice(0,19)`
everywhere → always `YYYY-MM-DDTHH:MM:SS`.

### M2 — `anxiety_level` still in few-shot examples — invalid node name
**File:** `core/nlu.ts` `CLASSIFY_INTENT_SHOTS` ~line 861
**Problem:** Few-shot example emits `"queryNodes":["anxiety_level"]`. Not a valid DBN node
(valid: `stress_ema`, `stress_helplessness`). Silently filtered at runtime but trains model
to output this invalid name.
**Fix:** Replace `anxiety_level` → `stress_ema` in the few-shot example.

### M3 — `value_map` defined in config but `resolveNodeValue` ignores it
**File:** `core/nlu.ts` `resolveNodeValue()`, `configs/feature_node_config.json`
**Problem:** 9 source column bins define `value_map` (numeric option code → intermediate label,
e.g. `1 → 'married'`). `resolveNodeValue` only reads `state_map`, ignores `value_map`.
For `marital_status`, `general_health` etc., user answers with numeric `raw_value` (e.g. `1`).
`rawStr = '1'` not in `state_map` → always returns null → never reaches inference.
**Fix:** In `resolveNodeValue`, before applying `state_map`, check if `value_map` exists.
Apply `value_map` first to convert numeric raw_value string to intermediate label, then apply
`state_map` to get final node state.

### M4 — Composite guard incorrectly blocks independent binary questions
**File:** `core/nlu.ts` `resolveNodeValue()` ~line 589
**Problem:** `pain_level` (2 items: CDQ001, CDQ010), `diabetes_status` (3 items),
`chronic_condition` (8 items) trigger the composite early-return
(`COMPOSITE_ITEM_COUNT[col] > 1 → null`). These are independent yes/no questions, not summed
scales. Their `state_map` and `value_map` are correctly defined but never execute.
**Fix:** Add an `is_composite` boolean flag to `source_column_bins` in `feature_node_config.json`
for true summed instruments only (PHQ-9, GAD-7, etc.). Change composite guard to check
`binConf.is_composite === true` instead of `COMPOSITE_ITEM_COUNT[col] > 1`.
Mark only true summed instruments as `is_composite: true`.

### M5 — `resetSession()` never called — stale beliefs bleed across sessions
**File:** `core/mcp.ts` `resetSession()`
**Problem:** `_session` state (`latestBeliefs`, `turnStartBeliefs`, `directEvidenceNodes`)
persists for JS process lifetime. On new session (user navigates away and back), stale
`latestBeliefs` from previous session becomes `turnStartBeliefs` baseline for new session's
first `get_changed_nodes` → phantom changed nodes reported on fresh session.
**Fix:** Call `resetSession()` in `agent.ts` `startTurn()` when `sessionId` differs from
last known session (store `_lastSessionId` in module scope, compare on each `startTurn` call).

---

## LOW

### L1 — `store_indirect_evidence` validates node_name but not node_value
**File:** `core/mcp.ts` `handleStoreIndirectEvidence()`
**Problem:** `node_name` validated against `MODEL_PARENTS` but `node_value` accepts any string.
Gemma can store invalid state (e.g. `stress_ema = 'very_high'`). In `runLBP`,
`slist.indexOf(evidence[node])` returns -1 → defaults to state[0] → wrong hard evidence.
**Fix:** After node_name validation, check `nodeValue` against `model.states[nodeName]`.
Return `{stored: false, reason: 'invalid_node_value', valid_values: model.states[nodeName]}`
if not found.

### L2 — Fix 2 in evidenceLayer dead code (subsumed by Fix 3)
**File:** `core/evidenceLayer.ts`
**Problem:** Fix 2 (sensor nodes with no recent window → marginal prior) adds no nodes that
Fix 3 (universal fallback) would not already add. The guard `if (nodeName in STALENESS_DAYS) continue`
means those nodes were already handled in the sensorless loop. Fix 2 is unreachable dead code.
**Fix:** Remove Fix 2 entirely. Fix 3 universally covers all absent nodes.

### L3 — `App.tsx` model paths are empty strings — init silently skipped
**File:** `Codebase/mobile/App.tsx`
**Problem:** `NLU_MODEL_PATH` and `AGENT_MODEL_PATH` are empty strings. Init guard
`if (NLU_MODEL_PATH && AGENT_MODEL_PATH)` skips all model initialization silently.
App runs with `modelsReady=false`, no error surfaced to user.
**Fix:** Addressed by C1-REG fix. Ensure paths are populated and missing paths produce
a visible error state, not silent skip.
