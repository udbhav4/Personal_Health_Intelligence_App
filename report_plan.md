# Doctor Report PDF — Full Implementation Plan

## Overview

User types symptom in Chat → Report tab → app collects 180 days of data → hidden pattern detector runs → Gemma generates narrative focused on **what the patient forgot or didn't realize is connected** → HTML rendered → PDF saved on-device → browsable in Profile tab → shareable via system share sheet.

### Core Philosophy

**This is NOT a symptom summary.** The doctor already knows the symptom — the patient will tell them verbally. The report's sole value is surfacing:
- Things the patient mentioned months ago but stopped tracking
- Passive data patterns the patient never consciously noticed (screen time spikes, step count crashes, sleep degradation)
- Cross-domain correlations (e.g., high screen time → poor sleep → low steps → stress escalation, all co-occurring 3 weeks before complaint peaks)
- DBN nodes that are elevated but were never mentioned in conversation
- Anomaly weeks where 3+ metrics deviated simultaneously

The doctor must read this and think: *"The patient would never have told me this."* If the report only describes the stated symptom, it has failed.

---

## Option Decision: Hybrid

- **Generation entry point:** Chat screen, Report feature tab (already exists as stub at ChatScreen.tsx:502)
- **History/browse:** Profile tab, new "Reports" section
- **Delivery:** System share sheet (expo-sharing) → user sends to doctor

Rationale: Chat-only = no persistence. Profile-only = buried form. Hybrid = conversational entry + persistent history + re-share anytime.

---

## Mode Lock: Report → Ultra Only

**Already implemented.** ChatScreen.tsx line 443–445:
```ts
const allowedModes: Mode[] =
  feature === 'Journal' ? ['Glance'] :
  feature === 'Talk'    ? ['Glance', 'Reflect'] : ['Ultra'];
```
- Report maps to `['Ultra']` only
- Non-allowed modes already `disabled` in mode menu modal (line 734)
- Switching to Report auto-sets Ultra via useEffect (line 447–448)
- **No code change needed for this requirement**

---

## Library Choice

**`expo-print` + `expo-sharing`**

| Library | Verdict |
|---|---|
| `expo-print` | Android WebView → PDF via native print subsystem. Expo SDK 54 compatible. No native rebuild. **Use this.** |
| `expo-sharing` | System share sheet. Works with any file URI. **Use alongside expo-print.** |
| `react-native-html-to-pdf` | Android write-permission bugs on API 33+. Poorly maintained. Avoid. |
| `@react-pdf/renderer` | Web/Node only. Incompatible with RN runtime. Avoid. |
| `react-native-pdf-lib` | Abandoned. Targets RN < 0.71. Avoid. |

Install command (run from `codebase/mobile/`):
```
npx expo install expo-print expo-sharing
```

---

## Data Pipeline

```
Symptom input
    ↓
gatherReportData() — sync SQLite reads, 180-day window
    ↓
detectHiddenPatterns(rawData) — pure TS, no Gemma, 6 detection algorithms
    ↓
classifyPatternTiers(patterns, symptom) — pure TS, deterministic keyword map
    ↓  tier1: symptom-linked patterns   tier2: notable but unrelated
ReportDataObject (typed struct, tier1Patterns + tier2Patterns separate)
    ↓
generateReportNarrative(data, onProgress) — Gemma Plan-and-Execute (6 calls)
    Planner receives pre-labeled tiers — does NOT reclassify, only narrates
    ↓
buildReportHtml(narrative, data) — pure fn, inline CSS
    Tier 1 → highlighted boxes (sections 3+4)
    Tier 2 → greyed "Additional Health Context" section (bottom, before limitations)
    ↓
Print.printToFileAsync({ html }) — expo-print → temp PDF URI
    ↓
FileSystem.copyAsync → documentDirectory/reports/<timestamp>.pdf
    ↓
writeDoctorReport(db, symptom, uri, generatedAt) — SQLite insert
    ↓
Sharing.shareAsync(uri) — system share sheet
```

---

## Data Collection — 180-Day Window

All queries run via `db.executeSync()`. No async DB layer.

### Query A — Sensorless nodes (180 days)
```sql
SELECT node_name, node_value, confidence, data_source, raw_text, report_date
FROM user_data_sensorless
WHERE is_active = 1
  AND report_date >= date('now', '-180 days')
ORDER BY node_name, report_date DESC;
```
Post-process: group by `node_name`, keep highest-confidence row per node, filter `confidence > 0.3`.

### Query B — Sensor windows (180 days, aggregated weekly)
```sql
SELECT strftime('%Y-%W', date) AS week,
       node_name,
       source_column,
       AVG(CAST(raw_value AS REAL)) AS weekly_avg,
       COUNT(*) AS day_count
FROM sensor_windows
WHERE date >= date('now', '-180 days')
  AND source_column IN ('hourly_steps', 'active_ratio', 'screen_time')
GROUP BY week, node_name, source_column
ORDER BY week ASC, node_name ASC;
```
Result: ~26 weekly rows per metric. Safe for Gemma context. Trend = first 4 weeks avg vs last 4 weeks avg.

### Query C — Chat messages (180 days, last 30 user messages)
```sql
SELECT content, topic, created_at
FROM chat_messages
WHERE role = 'user'
  AND is_active = 1
  AND evicted = 0
  AND created_at >= datetime('now', '-180 days')
ORDER BY created_at DESC
LIMIT 30;
```
Post-process: filter `length(content) > 20` (skip one-word messages), keep 5 most relevant excerpts.

### Query F — Forgotten complaints (chat messages older than 60 days mentioning health)
```sql
SELECT content, topic, created_at
FROM chat_messages
WHERE role = 'user'
  AND is_active = 1
  AND evicted = 0
  AND created_at < datetime('now', '-60 days')
  AND created_at >= datetime('now', '-180 days')
  AND (
    content LIKE '%pain%' OR content LIKE '%tired%' OR content LIKE '%stress%'
    OR content LIKE '%sleep%' OR content LIKE '%anxious%' OR content LIKE '%depress%'
    OR content LIKE '%hurt%' OR content LIKE '%sick%' OR content LIKE '%fatigue%'
    OR content LIKE '%ache%' OR content LIKE '%numb%' OR content LIKE '%dizzy%'
    OR content LIKE '%headache%' OR content LIKE '%back%' OR content LIKE '%eye%'
  )
ORDER BY created_at DESC
LIMIT 15;
```
These are complaints the patient mentioned months ago and likely forgot — highest-value content for the doctor. Stored in `forgottenComplaints[]`.

### Query D — Memory summaries (180 days, all)
```sql
SELECT summary_text, created_at
FROM memory_summaries
WHERE created_at >= datetime('now', '-180 days')
ORDER BY created_at DESC
LIMIT 50;
```
180d window, up to 50 summaries. Each summary = evicted session condensed. More summaries = richer forgotten pattern coverage for Gemma.

### Query E — DBN belief trajectory (180 days, not just latest)
```sql
SELECT dbn_beliefs, node_confidences, node_data_sources, summary_line, created_at
FROM inference_snapshots
WHERE created_at >= datetime('now', '-180 days')
ORDER BY created_at ASC;
```
Full 180d belief history — not just latest snapshot. Used by `detectHiddenPatterns()` to find DBN-level sustained trends (e.g., depression belief drifting from `low` → `moderate` → `high` over 12 weeks) that the latest snapshot alone cannot show. Also injected into Gemma executor calls as weekly belief series per node.

---

## ReportDataObject Type

```typescript
interface DbnSnapshotEntry {
  created_at:        string;
  dbn_beliefs:       string;  // JSON string — parse to Record<string, number[]>
  node_confidences:  string;  // JSON string
  summary_line:      string | null;
}

interface SensorlessNodeSummary {
  node_name:   string;
  node_value:  string;
  confidence:  number;
  data_source: string;
  raw_text:    string | null;
  report_date: string;
}

interface SensorTrendEntry {
  week:        string;  // YYYY-WW
  node_name:   string;
  source_col:  string;
  weekly_avg:  number;
  day_count:   number;
}

type PatternType =
  | 'temporal_correlation'   // 2+ metrics move together across weeks
  | 'anomaly_week'           // week where 3+ metrics simultaneously deviated from baseline
  | 'sustained_trend'        // metric worsening steadily >4 weeks without mention
  | 'forgotten_complaint'    // health-related chat message > 60 days ago
  | 'silent_node'            // DBN node elevated but never mentioned in chat
  | 'contradictory_state';   // patient said one thing; passive data shows another

interface HiddenPattern {
  type:        PatternType;
  description: string;     // pre-computed plain English — ready to paste into HTML
  nodes:       string[];   // involved DBN node names
  weeks?:      string[];   // YYYY-WW weeks involved (if temporal)
  severity:    'notable' | 'significant' | 'critical';
}

interface ReportDataObject {
  symptom:              string;
  profile:              UserProfile;
  generatedAt:          string;               // ISO timestamp
  beliefs:              BeliefResult | null;  // from AppContext
  sensorlessNodes:      SensorlessNodeSummary[];
  sensorTrends:         SensorTrendEntry[];   // 26 weeks × metrics
  chatExcerpts:         string[];             // up to 5 recent excerpts
  forgottenComplaints:  string[];             // chat > 60 days ago, health-related (raw strings)
  memorySummaries:      string[];             // up to 50, 180d window
  dbnTrajectory:        DbnSnapshotEntry[];   // 180d belief history, ordered oldest→newest
  snapshotLine:         string | null;        // latest snapshot summary_line
  trendDirection:       Record<string, 'up' | 'down' | 'flat'>;
  tier1Patterns:        HiddenPattern[];      // symptom-linked — main report sections 3+4
  tier2Patterns:        HiddenPattern[];      // notable but unrelated — greyed supplementary section
  anomalyWeeks:         string[];             // YYYY-WW weeks with 3+ simultaneous deviations
  silentNodes:          string[];             // DBN nodes elevated but never mentioned
  keywordMatchedNodes:  Set<string>;          // which nodes the symptom keywords resolved to
}
```

---

## Tier Classification — `classifyPatternTiers()`

### Why pure TypeScript, not Gemma

Gemma 4-2B-IT is too small to reliably classify relevance. Asking it to output a JSON classification layer on top of an already-complex planner call risks JSON parse failures and misclassification. This must be deterministic — a failed tier classification means the doctor sees irrelevant data in section 3 (high visibility) or misses relevant data entirely.

**Rule:** `classifyPatternTiers()` is pure TypeScript, zero model calls, runs synchronously in <5ms.

### Implementation

Location: `src/core/reportDataCollector.ts`, exported alongside `gatherReportData` and `detectHiddenPatterns`.

```typescript
export function classifyPatternTiers(
  patterns:  HiddenPattern[],
  symptom:   string,
): { tier1: HiddenPattern[]; tier2: HiddenPattern[] }
```

### How classification works

**Step 1 — Extract symptom keywords**
Lowercase the symptom string, tokenize on spaces/punctuation:
```typescript
const tokens = symptom.toLowerCase().split(/[\s,\.;!?]+/);
```

**Step 2 — Build relevant node set via `SYMPTOM_DOMAIN_MAP`**
Hardcoded map from symptom keywords → DBN nodes likely relevant:

```typescript
const SYMPTOM_DOMAIN_MAP: Record<string, string[]> = {
  // Pain / physical
  pain:       ['pain_level', 'physical_stress', 'physical_health', 'exercise', 'bmi'],
  back:       ['pain_level', 'physical_stress', 'physical_health', 'exercise'],
  knee:       ['pain_level', 'physical_stress', 'exercise', 'bmi'],
  neck:       ['pain_level', 'physical_stress', 'stress_ema'],
  headache:   ['pain_level', 'mental_stress', 'sleep_quality', 'physical_stress'],
  eye:        ['pain_level', 'screen_time', 'sleep_quality', 'physical_stress'],
  chest:      ['pain_level', 'physical_stress', 'mental_stress', 'stress_ema'],

  // Mental / emotional
  stress:     ['mental_stress', 'stress_ema', 'stress_helplessness', 'stress_self_efficacy', 'mood', 'sleep_quality', 'productivity'],
  anxious:    ['mental_stress', 'stress_ema', 'negative_affect', 'sleep_quality', 'mood'],
  anxiety:    ['mental_stress', 'stress_ema', 'negative_affect', 'sleep_quality', 'mood'],
  depress:    ['depression', 'mood', 'negative_affect', 'loneliness', 'social_events_negative', 'positive_affect'],
  sad:        ['depression', 'mood', 'negative_affect', 'loneliness'],
  lonely:     ['loneliness', 'social_events_negative', 'mood', 'depression'],
  frustrated: ['mental_stress', 'stress_ema', 'mood', 'negative_affect', 'productivity'],
  angry:      ['mental_stress', 'negative_affect', 'mood'],
  overwhelm:  ['mental_stress', 'stress_helplessness', 'productivity', 'stress_ema'],

  // Sleep / fatigue
  sleep:      ['sleep_quality', 'sleep_disturbances', 'mental_stress', 'screen_time', 'physical_stress'],
  tired:      ['sleep_quality', 'physical_stress', 'mental_stress', 'bmi', 'physical_health'],
  fatigue:    ['sleep_quality', 'physical_stress', 'bmi', 'physical_health', 'depression'],
  exhausted:  ['sleep_quality', 'physical_stress', 'mental_stress', 'depression'],
  insomnia:   ['sleep_quality', 'sleep_disturbances', 'mental_stress', 'screen_time'],

  // Activity / physical health
  exercise:   ['exercise', 'physical_health', 'physical_stress', 'bmi'],
  active:     ['exercise', 'physical_health', 'bmi'],
  weight:     ['bmi', 'exercise', 'physical_health'],

  // Cognitive
  focus:      ['mental_stress', 'sleep_quality', 'productivity', 'stress_ema'],
  memory:     ['mental_stress', 'sleep_quality', 'depression'],
  productive: ['productivity', 'mental_stress', 'sleep_quality', 'mood'],
  work:       ['productivity', 'stress_ema', 'mental_stress', 'stress_helplessness'],

  // Social
  social:     ['loneliness', 'social_events_positive', 'social_events_negative', 'mood'],
  lonely:     ['loneliness', 'social_events_negative', 'depression'],
  isolated:   ['loneliness', 'social_events_negative', 'depression', 'mood'],
};
```

**Step 3 — Compute relevant node set**
```typescript
const relevantNodes = new Set<string>();
for (const token of tokens) {
  for (const [keyword, nodes] of Object.entries(SYMPTOM_DOMAIN_MAP)) {
    if (token.includes(keyword) || keyword.includes(token)) {
      nodes.forEach(n => relevantNodes.add(n));
    }
  }
}
// Passive sensor nodes always relevant (universal context for any complaint)
relevantNodes.add('screen_time');
relevantNodes.add('activity');
```

**Step 4 — Classify each pattern**
```typescript
tier1 if ANY of:
  a) pattern.nodes.some(n => relevantNodes.has(n))   // symptom-linked node
  b) pattern.type === 'forgotten_complaint'           // always T1 — patient forgot it
  c) pattern.type === 'anomaly_week'                  // multi-metric crash always relevant
  d) pattern.severity === 'critical'                  // critical always shown prominently

tier2 otherwise (notable health context, shown separately at bottom)
```

**Step 5 — Fallback: no keywords matched**
If `relevantNodes.size === 0` after step 3 (symptom text too vague or unrecognized):
- All `forgotten_complaint` and `anomaly_week` patterns → T1
- All others → T2
- Gemma receives fallback instruction: "Symptom keywords not recognized. Focus narrative on forgotten complaints and anomaly weeks."

### Edge cases handled

| Case | Behaviour |
|---|---|
| Symptom = "I feel bad" (no keywords) | Fallback — forgotten complaints + anomaly weeks as T1 |
| All patterns → T1 (broad symptom like "stress") | Accepted — all shown in main sections, nothing in T2 |
| No patterns detected at all | T1 = empty, T2 = empty, report generates from DBN beliefs + sensor tables only |
| `forgotten_complaint` with nodes=[] | Always T1 regardless (no node overlap needed) |

---

## Hidden Pattern Detector — `detectHiddenPatterns()`

Runs **before** Gemma. Pure TypeScript, synchronous, no model needed. Produces `HiddenPattern[]` that both the HTML builder and Gemma planner consume.

Location: `src/core/reportDataCollector.ts` (exported alongside `gatherReportData`).

### Detection algorithms

**1. Temporal correlation**
For each pair of metrics in `sensorTrends` (steps, screen_time, weekly sensorless values):
- Compute week-over-week delta per metric
- If two metrics have same-direction delta (both worsen) in ≥3 of the same weeks → `temporal_correlation`
- Example output: `"Screen time and stress both escalated in the same 4 weeks (weeks 12–15). Steps dropped in the same window."`

**2. Anomaly weeks**
Per week: count how many metrics are in their worst tertile (bottom 33% of their 26-week range for steps/mood/sleep; top 33% for stress/screen_time/depression).
- ≥3 metrics in worst tertile simultaneously → `anomaly_week`, severity `significant`
- ≥5 metrics → severity `critical`
- Store YYYY-WW labels in `anomalyWeeks[]`

**3. Sustained trend (silent degradation)**
For each metric: compute linear trend slope over 26 weeks.
- Slope worsening for >8 consecutive weeks + metric never appeared in recent chat (last 14 days) → `sustained_trend`
- Example: "Sleep quality has been declining steadily for 10 weeks. It has not come up in recent conversation."

**4. Forgotten complaints**
Each row from Query F → `forgotten_complaint` pattern with severity `significant`.
- Description: `"[X weeks ago] Patient mentioned: '[excerpt]'. Not raised since."`
- Truncate excerpt to 80 chars.

**5. Silent nodes**
For each DBN node where `beliefs` has dominant state = high-risk value (e.g., depression=moderate, loneliness=high, pain_level=high) AND `confidence > 0.6` AND node_name does not appear in any `chatExcerpts` content → `silent_node`
- Example: `"DBN estimates loneliness as 'high' (confidence 0.72) but it has never been raised in conversation."`

**6. Contradictory state**
If a sensorless self-report says one thing (e.g., exercise=high, confidence 0.9) but sensor_windows weekly avg steps < 3000 for the same weeks → `contradictory_state`
- Example: `"Patient self-reported regular exercise, but step counts averaged under 3,000/day for 5 of those weeks."`

---

## Context Window Budget

Gemma n_ctx = 4096 tokens (~1 token ≈ 4 chars → ~16k chars total per call).
Each call must fit: system prompt + user prompt + n_predict within 4096 tokens.

**Rule: 180d data is collected in full for TS detection. Gemma never sees raw rows. Only compressed summaries derived from that data enter Gemma's context.**

### Token budget per call

| Call | System | User prompt | n_predict | Total |
|---|---|---|---|---|
| Planner | ~150 tok | ~500 tok | 350 tok | ~1000 tok ✓ |
| Executor ×5 | ~100 tok | ~400 tok | 200 tok | ~700 tok ✓ |

All well within 4096. Safety headroom: ~3000 tokens unused — never inject raw rows.

### What gets compressed before Gemma sees it

| Raw data (180d, TS only) | Compressed form injected into Gemma |
|---|---|
| 180d sensorless rows (hundreds of rows) | `detectHiddenPatterns()` pre-computed descriptions — plain strings, ≤60 chars each |
| 180d memory summaries (up to 50 rows) | Cosine sim via sqlite-vec → top 10 semantically similar to symptom, each ≤120 chars |
| 180d DBN trajectory (daily snapshots) | Per-node weekly dominant state string: `"high,high,mod,low,high,..."` (26 chars/node, ~30 nodes = ~200 tokens total) |
| 26-week sensor aggregates | 3 numbers per metric: first-4-week avg, last-4-week avg, trend direction. `"steps: 4200→2800(↓)"` |
| Forgotten complaints (Query F) | Up to 8 entries, each: `"[N weeks ago] <80-char excerpt>"` |
| T1 patterns | Up to 10 descriptions, ≤60 chars each. Truncate beyond 10. |
| T2 patterns | Up to 6 descriptions, ≤50 chars each. Truncate beyond 6. |

**Compression functions** — in `reportDataCollector.ts`:
```typescript
// ASYNC — uses embedText() + sqlite-vec cosine sim
compressMemories(db, symptom): Promise<string[]>
  // Production path: embedText(symptom) → vec_distance_cosine → top 10, ≤120 chars each
  // Fallback (embed fails or no embeddings): keyword-match → top 10, ≤120 chars each

// SYNC — pure TS
compressDbnTrajectory(snapshots): Record<string, string> // node → "high,mod,low,..." weekly
compressSensorTrends(trends): string                     // "steps: 4200→2800(↓) screen: 9.1→10.3(↑)"
compressPatterns(patterns, limit, maxLen): string[]      // truncate array + strings
```

**`compressMemories()` implementation detail:**
```typescript
async function compressMemories(db: DB, symptom: string): Promise<string[]> {
  const queryVec = await embedText(symptom).catch(() => null);
  if (queryVec) {
    const rows = db.executeSync(`
      SELECT summary_text,
             vec_distance_cosine(embedding, ?) AS dist
      FROM   memory_summaries
      WHERE  embedding IS NOT NULL
        AND  created_at >= datetime('now', '-180 days')
      ORDER  BY dist ASC
      LIMIT  10
    `, [queryVec]).rows as { summary_text: string }[];
    if (rows.length > 0)
      return rows.map(r => r.summary_text.slice(0, 120));
  }
  // fallback: keyword match (dev/seed = no embeddings; embed model unavailable)
  const tokens = symptom.toLowerCase().split(/[\s,\.;!?]+/);
  const rows = db.executeSync(`
    SELECT summary_text FROM memory_summaries
    WHERE  created_at >= datetime('now', '-180 days')
    ORDER  BY created_at DESC LIMIT 20
  `).rows as { summary_text: string }[];
  return rows
    .filter(r => tokens.some(t => r.summary_text.toLowerCase().includes(t)))
    .slice(0, 10)
    .map(r => r.summary_text.slice(0, 120));
}
```

`gatherReportData()` is **async** (needed for `compressMemories`). Called from `ReportGeneratorScreen` which is already async.

---

## Gemma — Plan-and-Execute (6 calls)

Uses `runAgentCompletion()` exported from `agent.ts`. Same Gemma ctx as Talk mode. Total ~30s.

**Gemma never sees raw DB rows. Only compressed summaries from TS compression functions above.**

### Call 1 — Planner (1×, n_predict=350, temperature=0.0)

Gemma's job: write 5 section prose instructions. NOT reclassify tiers — that's already done by TS.

```
System (~150 tok):
  You are writing a medical report for a doctor. Surface what the patient forgot or
  didn't notice. Act like a detective. Never restate the chief complaint.

User (~500 tok):
  Complaint: "{symptom}"

  TIER 1 patterns (symptom-linked) — up to 10, ≤60 chars each:
  {tier1Descriptions}

  TIER 2 patterns (additional context) — up to 6, ≤50 chars each:
  {tier2Descriptions}

  Passive sensor summary: {compressSensorTrends output}
  DBN trajectory (key nodes): {compressDbnTrajectory output, relevant nodes only}
  Anomaly weeks: {anomalyWeeks joined by ", "}

  Write JSON array of exactly 5 section plans:
  [{ "section_title": string, "tier": "1"|"2"|"both", "instruction": string }]
  Mandatory: "What the Patient Forgot"(1), "Passive Data Patterns"(1),
             "How This Connects"(1), "Questions for Doctor"(both), "Data Limitations"(both)
  Output JSON only. No markdown fences.
```

JSON extraction: `text.match(/\[[\s\S]+\]/)` → try/catch → static fallback plan if fails:
```typescript
const FALLBACK_PLAN = [
  { section_title: 'What the Patient Forgot',     tier: '1',    instruction: 'Surface forgotten complaints and silent elevated DBN nodes from Tier 1.' },
  { section_title: 'Passive Data Patterns',       tier: '1',    instruction: 'Describe temporal correlations and anomaly weeks. Cite specific values.' },
  { section_title: 'How This Connects',           tier: '1',    instruction: 'Link Tier 1 patterns non-obviously to the stated complaint.' },
  { section_title: 'Questions for Your Doctor',   tier: 'both', instruction: 'List 3–5 specific questions grounded in detected patterns.' },
  { section_title: 'Data Limitations',            tier: 'both', instruction: 'State what this passive data cannot confirm or diagnose.' },
];
```

### Calls 2–6 — Executors (5×, n_predict=200 each)

Each call is small and focused. Receives only what that section needs.

```
System (~100 tok):
  Write one section of a medical report. Rules:
  - Cite specific values (dates, averages, node states). No vague statements.
  - Max 120 words. Never restate the chief complaint.

User (~400 tok, section-specific):
  Section: "{section_title}"
  Instruction: "{instruction from planner}"

  [if tier=1]: Tier 1 data:
    Patterns: {tier1Descriptions — same compressed list as planner}
    Relevant memories (top 3 keyword-matched, ≤120 chars each): {compressMemories top 3}
    DBN trajectory (relevant nodes only): {compressDbnTrajectory filtered to keywordMatchedNodes}
    Sensor: {compressSensorTrends}

  [if tier=both]: also include:
    Tier 2 patterns: {tier2Descriptions}

  [if tier=2]: Tier 2 data only:
    Patterns: {tier2Descriptions}
```

n_predict=200 per executor (120 words ≈ 160 tokens + headroom). 5 executors × 200 = 1000 tokens generated total.

### Mandatory validator

After all 5 section outputs concatenated, check for:
- `tier1Patterns.length > 0` → output must contain at least one specific data value (number, week, percentage)
- Output contains "forgot" OR "not mentioned" OR "silent" OR "not raised" (section 1 guard)
- Output contains "data limitations" or "Data Limitations" (section 5 guard)

Missing → append hardcoded fallback text for that section only. Report always generates.

---

## HTML Report Structure

`buildReportHtml(narrative, data)` — pure function, no side effects.

HTML constraints for `expo-print` Android compatibility:
- All CSS inline (no `<link>`, no external fonts)
- Font: `font-family: -apple-system, Helvetica, Arial, sans-serif`
- White background, black text (PDF always light mode)
- Tables: `border: 1px solid #ccc; border-collapse: collapse` (explicit — WebView can drop borderless)
- `@media print { body { -webkit-print-color-adjust: exact; } }` in `<style>` block
- No flexbox in HTML (Android print CSS support ~2016 level)
- Max width 720px, centered, 40px padding
- No external resources (images, fonts, CDN)

Report section order (most clinically valuable content first):

```
┌─────────────────────────────────────────────────────────────┐
│  1. HEADER + DISCLAIMER                                      │
├─────────────────────────────────────────────────────────────┤
│  2. CHIEF COMPLAINT  (2–3 lines, patient's words verbatim)  │
├─────────────────────────────────────────────────────────────┤
│  ██████  TIER 1 — SYMPTOM-LINKED  ██████                    │
│                                                             │
│  3. WHAT THE PATIENT FORGOT       [amber box]               │
│  4. PASSIVE DATA PATTERNS         [blue box]                │
│  5. GEMMA NARRATIVE (sections 1–4 of plan, tier 1 only)     │
├─────────────────────────────────────────────────────────────┤
│  6. SENSOR TREND TABLE (steps + screen, 26 weeks)           │
│     anomaly weeks highlighted amber                         │
│  7. SELF-REPORTED NODES TABLE                               │
│     silent nodes marked ★                                   │
│  8. DBN BELIEF SUMMARY                                      │
├─────────────────────────────────────────────────────────────┤
│  ░░░░░░  TIER 2 — ADDITIONAL HEALTH CONTEXT  ░░░░░░         │
│  (greyed section, visually separated, labelled:             │
│  "Notable health patterns not directly related              │
│   to your stated complaint")                                │
│  9. TIER 2 PATTERN LIST (plain bullets, no narrative)       │
├─────────────────────────────────────────────────────────────┤
│  10. DATA LIMITATIONS             [always last, hardcoded]  │
└─────────────────────────────────────────────────────────────┘
```

**Section 1 — Header**
- Patient name, date generated
- Bold disclaimer: *"This report surfaces passive health patterns the patient may not have mentioned. It is not a clinical assessment."*

**Section 2 — Chief Complaint** *(brief — context only)*
- Patient's exact words in italics, inside a light grey box
- Label above: "Patient's stated complaint"
- No elaboration. Doctor hears details verbally.

**Section 3 — What the Patient Forgot** *(amber background box, `tier1Patterns` only)*
- Only patterns where `type === 'forgotten_complaint'` OR `type === 'silent_node'` AND tier === 1
- Bulleted list. Each bullet = one dated fact.
- Label: *"Detected from passive data and past conversations. The patient did not raise these."*
- If empty (no T1 forgotten/silent patterns): section shows "No forgotten complaints detected in the 180-day window." — section still renders, not hidden.

**Section 4 — Passive Data Patterns** *(blue background box, `tier1Patterns` only)*
- Only patterns where `type === 'temporal_correlation'` OR `'anomaly_week'` OR `'sustained_trend'` OR `'contradictory_state'` AND tier === 1
- Passive sensor patterns always included here (steps + screen time trends) regardless of tier — they are universal context
- Each pattern: one sentence with specific numbers
- Anomaly weeks explicitly listed: "Week of [date]: N indicators simultaneously worsened."
- If empty: shows step/screen summary only.

**Section 5 — Gemma Narrative** *(plain white, 4 sub-sections from plan-and-execute)*
- Subsection 1: What the Patient Forgot (expanded prose, tier 1)
- Subsection 2: Passive Data Patterns (expanded prose, tier 1)
- Subsection 3: How This Connects to Your Complaint (links tier 1 patterns to symptom)
- Subsection 4: Questions for Your Doctor (grounded in all detected patterns)

**Sections 6–8 — Supporting Evidence Tables**
- Sensor Trend Table: weekly avg steps + screen time, 26 weeks, trend arrows; anomaly week rows = amber `background-color: #fff8e1`
- Self-Reported Nodes Table: node name (plain English), value, confidence, date; silent nodes get ★ prefix
- DBN Belief Summary: dominant state + confidence; only `confidence > 0.5`; node labels from `NODE_LABELS` map

**Section 9 — Tier 2: Additional Health Context** *(greyed — `opacity: 0.75`, `border-left: 3px solid #ccc`)*
- Section label: *"Notable health patterns not directly related to your stated complaint. Included for completeness."*
- Plain bulleted list — `tier2Patterns[].description` only, no Gemma prose
- If `tier2Patterns` is empty: section omitted entirely (only section that can be hidden)

**Section 10 — Data Limitations** *(hardcoded — never from Gemma)*
- Always rendered. Always last. Verbatim paragraph.

**HTML constraints (all sections)**
- All CSS inline, no external resources
- Font: `-apple-system, Helvetica, Arial, sans-serif`
- White background, black text
- Tables: `border: 1px solid #ccc; border-collapse: collapse` explicit
- No flexbox (Android print CSS ~2016)
- Max width 720px, centered, 40px padding
- `@media print { body { -webkit-print-color-adjust: exact; } }`

DBN node → plain English lookup (partial, expand as needed):
```typescript
const NODE_LABELS: Record<string, string> = {
  mental_stress:      'Mental Stress Estimate',
  physical_stress:    'Physical Stress Estimate',
  pain_level:         'Pain Level',
  mood:               'Mood',
  sleep_quality:      'Sleep Quality',
  sleep_disturbances: 'Sleep Disturbances',
  depression:         'Depression Indicator',
  stress_ema:         'Stress (Self-Reported)',
  productivity:       'Productivity',
  exercise:           'Exercise Level',
  loneliness:         'Loneliness',
  negative_affect:    'Negative Affect',
  positive_affect:    'Positive Affect',
  bmi:                'BMI',
  age:                'Age Group',
  sex:                'Sex',
};
```

---

## New SQLite Table — `doctor_reports`

Migration M003 in `db.ts` → `runMigrations()`:

```sql
CREATE TABLE IF NOT EXISTS doctor_reports (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  symptom      TEXT NOT NULL,
  file_uri     TEXT NOT NULL,
  generated_at TEXT NOT NULL,
  created_at   TEXT DEFAULT (datetime('now'))
);
```

Helper functions to add to `db.ts`:
```typescript
export interface DoctorReport {
  id:           number;
  symptom:      string;
  file_uri:     string;
  generated_at: string;
  created_at:   string;
}

export function writeDoctorReport(
  db: DB, symptom: string, fileUri: string, generatedAt: string,
): void

export function getDoctorReports(db: DB): DoctorReport[]
```

All use `db.executeSync()`.

---

## New Files

| File | Purpose |
|---|---|
| `src/core/reportDataCollector.ts` | Sync SQL reads → `ReportDataObject`; exports `detectHiddenPatterns()`, `classifyPatternTiers()`, and all 4 compression functions |
| `src/core/reportHtmlBuilder.ts` | Pure fn: `ReportDataObject` + narrative → HTML string |
| `src/core/reportAgent.ts` | Plan-and-Execute Gemma calls → narrative string |
| `src/screens/ReportGeneratorScreen.tsx` | Orchestrator UI (Modal), progress, share |

---

## Modified Files

| File | Change |
|---|---|
| `src/core/db.ts` | Add `doctor_reports` table (M003), `DoctorReport` type, `writeDoctorReport`, `getDoctorReports` |
| `src/core/agent.ts` | Add exported `runAgentCompletion()` and `isAgentBusy()` |
| `src/screens/ChatScreen.tsx` | Replace TODO stub (line 502) with Modal mount of `ReportGeneratorScreen` |
| `src/screens/ProfileScreen.tsx` | Add Reports section below existing fields |

---

## ReportGeneratorScreen State Machine

```
idle → generating → done
              ↓
            error (retry button)
```

State:
```typescript
symptom:      string        // TextInput value
status:       'idle' | 'generating' | 'done' | 'error'
progressText: string        // forwarded from reportAgent onProgress
savedUri:     string | null // file URI of generated PDF
```

Progress messages sequence:
1. "Collecting your data..."
2. "Writing report plan..."
3. "Writing [section_title]... (1 of 5)"
4. "Writing [section_title]... (2 of 5)"
5. "Writing [section_title]... (3 of 5)"
6. "Writing [section_title]... (4 of 5)"
7. "Writing [section_title]... (5 of 5)"
8. "Generating PDF..."

Generation sequence:
1. `gatherReportData(db, symptom, profile, beliefs)` — sync
2. `generateReportNarrative(data, onProgress)` — async, ~30s
3. `buildReportHtml(narrative, data)` — sync
4. `Print.printToFileAsync({ html })` — async → temp URI
5. `FileSystem.makeDirectoryAsync(documentDirectory + 'reports/', { intermediates: true })` — idempotent
6. `FileSystem.copyAsync({ from: tempUri, to: savedPath })` — to permanent location
7. `writeDoctorReport(db, symptom, savedPath, generatedAt)` — sync
8. Set `status = 'done'`, `savedUri = savedPath`

Props: `{ onClose: () => void }` — gets `db`, `profile`, `beliefs` from `useAppContext()` internally.

---

## ChatScreen.tsx Changes

Add state:
```typescript
const [showReport, setShowReport] = useState(false);
```

Replace TODO stub (line 501–506):
```typescript
} else {
  setShowReport(true);
}
```

Add Modal at bottom of return JSX (before closing View):
```tsx
<Modal
  visible={showReport}
  animationType="slide"
  onRequestClose={() => setShowReport(false)}
>
  <ReportGeneratorScreen onClose={() => setShowReport(false)} />
</Modal>
```

Import additions:
```typescript
import { Modal } from 'react-native';
import ReportGeneratorScreen from './ReportGeneratorScreen';
```

---

## ProfileScreen.tsx Changes

Import additions:
```typescript
import { getDoctorReports, type DoctorReport } from '../core/db';
import * as Sharing from 'expo-sharing';
```

Change `useAppContext` import from type-only to value import (needed to get `db`).

Add state:
```typescript
const [reports, setReports] = useState<DoctorReport[]>([]);
```

In `useEffect` triggered by `isActive`:
```typescript
if (isActive && db) setReports(getDoctorReports(db));
```

Add "Reports" section below `</View>` that closes the fields section:
- Section header: "YOUR REPORTS" (styled like existing `fieldLabel`)
- Per report card: symptom snippet (truncated to 60 chars), date, Share icon button
- Share button: `Sharing.shareAsync(report.file_uri, { mimeType: 'application/pdf', dialogTitle: 'Share with your doctor' })`
- Empty state text: "No reports yet. Tap Chat → Report to generate one."

---

## agent.ts Additions

### `runAgentCompletion` (used by reportAgent.ts)
```typescript
export async function runAgentCompletion(
  systemPrompt: string,
  userPrompt:   string,
  nPredict:     number = 500,
): Promise<string> {
  await ensureAgent();
  if (!_ctx) throw new Error('agent: context not available');
  const result = await _ctx.completion({
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user',   content: userPrompt   },
    ],
    n_predict:   nPredict,
    temperature: 0.1,
    stop:        ['<|im_end|>', '<end_of_turn>'],
  });
  return cleanGemmaOutput(result.text);
}
```

### `isAgentBusy` (used by ReportGeneratorScreen to disable button)
```typescript
let _agentBusy = false;

export function isAgentBusy(): boolean { return _agentBusy; }
```
Set `_agentBusy = true` at start of `runAgentCompletion`, `false` in finally block.

---

## Implementation Order

Execute tasks in this exact order (each depends on previous):

| # | Task | File(s) |
|---|---|---|
| 1 | `npx expo install expo-print expo-sharing` | package.json |
| 2 | Add `doctor_reports` table migration M003 + helpers | db.ts |
| 3 | Add `runAgentCompletion` + `isAgentBusy` exports | agent.ts |
| 4 | Create `reportDataCollector.ts` | new file |
| 5 | Create `reportHtmlBuilder.ts` | new file |
| 6 | Create `reportAgent.ts` | new file |
| 7 | Create `ReportGeneratorScreen.tsx` | new file |
| 8 | Wire Modal into ChatScreen (replace TODO stub) | ChatScreen.tsx |
| 9 | Add Reports section to ProfileScreen | ProfileScreen.tsx |
| 10 | Build + smoke test end-to-end | — |

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| 30s generation time feels blank | `onProgress` callback shows step-by-step status per section |
| Android HTML rendering quirks in expo-print | No flexbox, no external fonts, explicit table borders, inline CSS only — test HTML with static fixture before wiring Gemma |
| File permissions on Android | Always write to `documentDirectory` (app-scoped, no permissions needed). Share sheet handles user export to Downloads |
| Gemma context busy during report generation | `isAgentBusy()` disables "Generate Report" button; UI shows "Finish current conversation first" |
| Planner JSON parse failure | Regex extraction + try/catch + static fallback plan — report always generates |
| Missing `reports/` subdirectory | `makeDirectoryAsync(..., { intermediates: true })` before copy — idempotent, no throw if exists |
| Gemma context evicted mid-report (OS memory pressure) | `ensureAgent()` called at each executor call start, not once at top |
| 180-day sensor data = thousands of raw rows | Aggregate to weekly averages in SQL (GROUP BY week) → ~26 rows per metric → safe for Gemma context |

---

## Smoke Test Checklist

After build (`npx expo run:android`):

- [ ] Tap Chat → Report tab → mode auto-switches to Ultra
- [ ] Glance and Reflect appear disabled (greyed) in mode menu
- [ ] Type symptom → tap Generate → progress messages appear sequentially
- [ ] PDF generated (check `documentDirectory/reports/` via FileSystem log)
- [ ] Share button opens system share sheet with PDF
- [ ] Navigate to Profile tab → Reports section shows the new report
- [ ] Tap Share on saved report → re-sharing works
- [ ] Generate 2nd report → both appear in Profile Reports list
- [ ] Trigger with Gemma busy (mid Talk turn) → button disabled, correct message shown

### Tier classification checks (critical)

- [ ] Type symptom "I have back pain" → `classifyPatternTiers` maps to `pain_level`, `physical_stress`, `exercise`, `bmi` nodes → any pattern touching those nodes = T1
- [ ] Type symptom "I feel bad" (no keywords match) → fallback fires → forgotten_complaint + anomaly_week patterns = T1, rest = T2
- [ ] T1 patterns appear in sections 3+4 (amber/blue boxes) — verified in PDF
- [ ] T2 patterns appear ONLY in section 9 (greyed, below supporting tables) — not mixed into main sections
- [ ] Section 9 absent entirely when `tier2Patterns` is empty
- [ ] Passive sensor data (steps + screen trends) appears in section 4 regardless of tier (always relevant)
- [ ] `keywordMatchedNodes` logged in dev console — verify correct node set for symptom entered

### Content quality checks

- [ ] Section 2 (Chief Complaint) is 2–3 lines max — patient's words only, no elaboration
- [ ] Section 3 contains at least 1 forgotten complaint or silent node — not a restatement of symptom
- [ ] Section 4 contains specific numbers (week dates, averages, step counts) — not vague
- [ ] Gemma narrative does NOT repeat the chief complaint as main content
- [ ] "Questions for Your Doctor" grounded in detected patterns — not generic
- [ ] Anomaly weeks highlighted amber in sensor table
- [ ] Silent nodes marked ★ in health nodes table
- [ ] Data Limitations always last, always present
