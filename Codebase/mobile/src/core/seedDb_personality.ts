/**
 * seedDb_personality.ts — 180-day persona seed for integration testing.
 *
 * Persona: Arjun Mehta, 28yo male, office job (5-day week).
 *   • ~6 months of gym routine (4-5×/week) → abrupt stop ~7 days ago (back pain)
 *   • Weekday work stress (long-standing): Mon-Fri stress/anxiety spikes
 *   • Weekend relief: sports/treks/hikes — mood lifts, steps spike
 *   • Relationship friction: arguments with parents & girlfriend
 *     clustered 2-3 days AFTER heavy workweeks (delayed reaction)
 *   • Anxiety increasing in recent weeks; gym cessation = main stress-relief valve gone
 *   • Screen time rising as gym stops (compensatory sedentary behaviour)
 *   • Sleep deteriorating since gym stopped (gym fatigue used to give deep sleep)
 *   • Weekend mood reset shortening each week (recovery degrading)
 *
 * Non-obvious patterns the model should discover:
 *   1. Gym cessation (back pain) → loss of stress-relief valve → anxiety escalating
 *   2. Screen time rising as gym stops (compensatory behaviour)
 *   3. Weekday stress tolerated when gym active; now intolerable without it
 *   4. Weekend outdoor reset still fires but duration shrinking each week
 *   5. Fights with girlfriend/parents cluster 2-3 days after bad workweek (delayed)
 *   6. Sleep quality deteriorating since gym stopped (no longer gym-tired at bedtime)
 *
 * Tables populated:
 *   - user_data_sensorless  (persistent traits + daily sensorless nodes)
 *   - sensor_windows        (dashboard-compatible: prev_day_steps,
 *                            prev_day_active_ratio, screen_time_window_minutes,
 *                            + legacy: active_ratio, daily_steps, screen_time_hours)
 *   - chat_messages         (40+ realistic pairs over 180 days)
 *   - memory_summaries      (12 compressed narrative summaries)
 *   - inference_snapshots   (10 snapshots spread across 180 days — beliefs NOT null)
 *
 * Call seedPersonalityData(db) from a __DEV__ guard.
 * Idempotent — clears previous seed rows by session_id / data_source first.
 */

import type { DB } from '@op-engineering/op-sqlite';

// ─────────────────────────────────────────────────────────────────────────────
// Types (mirrored from seedDb.ts for self-containment)
// ─────────────────────────────────────────────────────────────────────────────

interface SensorlessRow {
  timestamp: string;
  node_name: string;
  original_column: string;
  source_column: string;
  question_text: string;
  raw_text: string;
  node_value: string;
  raw_value: number;
  summary_text: string;
  confidence: number;
  data_source: string;
  merge_mode: string;
  temporal_flag: string;
  report_date: string;
  expires_date: string | null;
  turn_id: string;
  is_active: number;
  was_proactive: number;
  answered: number;
  created_at: string;
}

interface SensorRow {
  date: string;
  snapshot_time: string;
  window_start: string | null;
  node_name: string;
  source_column: string;
  data_source: string;
  raw_value: number | null;
  raw_unit: string | null;
  discretized_value: string;
  confidence: number;
}

interface ChatRow {
  timestamp: string;
  session_id: string;
  turn_id: string;
  role: 'user' | 'model';
  content: string;
  topic: string | null;
  evicted: number;
  is_active: number;
  created_at: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// Date helpers
// ─────────────────────────────────────────────────────────────────────────────

/** day(0) = today, day(1) = yesterday, day(N) = N days ago */
function daysAgo(n: number): Date {
  const d = new Date();
  d.setHours(12, 0, 0, 0);
  d.setDate(d.getDate() - n);
  return d;
}

function toDateStr(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${y}-${m}-${day}`;
}

function toTs(d: Date, h = 12, min = 0, sec = 0): string {
  return (
    `${toDateStr(d)} ` +
    `${String(h).padStart(2, '0')}:${String(min).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
  );
}

function toISOLocal(d: Date, h = 23, min = 0): string {
  return `${toDateStr(d)}T${String(h).padStart(2, '0')}:${String(min).padStart(2, '0')}:00`;
}

function toYMD(d: Date): string {
  return toDateStr(d).replace(/-/g, '');
}

/** Simple seeded int in [min, max) */
function sr(seed: number, min: number, max: number): number {
  const x = Math.abs((seed * 1103515245 + 12345) & 0x7fffffff);
  return min + (x % (max - min));
}

// ─────────────────────────────────────────────────────────────────────────────
// Insert helpers
// ─────────────────────────────────────────────────────────────────────────────

function insertSL(db: DB, r: SensorlessRow): void {
  db.executeSync(
    `INSERT INTO user_data_sensorless (
      timestamp, node_name, original_column, source_column, question_text,
      raw_text, node_value, raw_value, summary_text, confidence,
      data_source, merge_mode, temporal_flag, report_date, expires_date,
      turn_id, is_active, was_proactive, answered, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    [
      r.timestamp, r.node_name, r.original_column, r.source_column,
      r.question_text, r.raw_text, r.node_value, r.raw_value,
      r.summary_text, r.confidence, r.data_source, r.merge_mode,
      r.temporal_flag, r.report_date, r.expires_date, r.turn_id,
      r.is_active, r.was_proactive, r.answered, r.created_at,
    ],
  );
}

function insertSW(db: DB, r: SensorRow): void {
  db.executeSync(
    `INSERT OR IGNORE INTO sensor_windows (
      date, snapshot_time, window_start, node_name, source_column,
      data_source, raw_value, raw_unit, discretized_value, confidence
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    [
      r.date, r.snapshot_time, r.window_start, r.node_name,
      r.source_column, r.data_source, r.raw_value, r.raw_unit,
      r.discretized_value, r.confidence,
    ],
  );
}

function insertChat(db: DB, r: ChatRow): void {
  db.executeSync(
    `INSERT INTO chat_messages (
      timestamp, session_id, turn_id, role, content, topic,
      evicted, is_active, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    [
      r.timestamp, r.session_id, r.turn_id, r.role, r.content,
      r.topic, r.evicted, r.is_active, r.created_at,
    ],
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Node metadata (mirrors seedDb.ts NODE_META / LABEL_TEXT / RAW_VALUE)
// ─────────────────────────────────────────────────────────────────────────────

const NODE_META: Record<string, { orig: string; src: string; q: string }> = {
  age:                    { orig: 'age',                    src: 'age_self_report',                    q: 'What is your age range?' },
  sex:                    { orig: 'sex',                    src: 'sex_self_report',                    q: 'What is your biological sex?' },
  education_level:        { orig: 'education_level',        src: 'education_level_self_report',        q: 'What is your highest level of education?' },
  marital_status:         { orig: 'marital_status',         src: 'marital_status_self_report',         q: 'What is your current relationship status?' },
  bmi:                    { orig: 'bmi',                    src: 'bmi_self_report',                    q: 'What is your BMI category?' },
  diabetes_status:        { orig: 'diabetes_status',        src: 'diabetes_status_self_report',        q: 'Do you have diabetes?' },
  chronic_condition:      { orig: 'chronic_condition',      src: 'chronic_condition_self_report',      q: 'Do you have any chronic conditions?' },
  smoking:                { orig: 'smoking',                src: 'smoking_self_report',                q: 'How often do you smoke?' },
  alcohol_use:            { orig: 'alcohol_use',            src: 'alcohol_use_self_report',            q: 'How would you describe your alcohol use?' },
  neuroticism:            { orig: 'neuroticism',            src: 'neuroticism_self_report',            q: 'How emotionally reactive do you tend to be?' },
  extraversion:           { orig: 'extraversion',           src: 'extraversion_self_report',           q: 'How socially outgoing are you?' },
  general_health:         { orig: 'general_health',         src: 'general_health_self_report',         q: 'How would you rate your general health?' },
  stress_ema:             { orig: 'stress_ema',             src: 'stress_ema_self_report',             q: 'How stressed have you felt today?' },
  mood:                   { orig: 'mood',                   src: 'mood_self_report',                   q: 'How would you describe your mood today?' },
  productivity:           { orig: 'productivity',           src: 'productivity_self_report',           q: 'How productive were you today?' },
  sleep_quality:          { orig: 'sleep_quality',          src: 'sleep_quality_self_report',          q: 'How well did you sleep last night?' },
  sleep_disturbances:     { orig: 'sleep_disturbances',     src: 'sleep_disturbances_self_report',     q: 'How much did your sleep get disrupted last night?' },
  exercise:               { orig: 'exercise',               src: 'exercise_self_report',               q: 'Did you exercise today?' },
  social_events_negative: { orig: 'social_events_negative', src: 'social_events_negative_self_report', q: 'Did you have any negative social interactions today?' },
  social_events_positive: { orig: 'social_events_positive', src: 'social_events_positive_self_report', q: 'Did you have any positive social interactions today?' },
  loneliness:             { orig: 'loneliness',             src: 'loneliness_self_report',             q: 'How lonely have you been feeling this week?' },
  negative_affect:        { orig: 'negative_affect',        src: 'negative_affect_self_report',        q: 'How much negative emotion have you experienced this week?' },
  positive_affect:        { orig: 'positive_affect',        src: 'positive_affect_self_report',        q: 'How much positive emotion have you experienced this week?' },
  depression:             { orig: 'depression',             src: 'depression_self_report',             q: 'How have your mood and energy levels been over the past week?' },
  mental_health:          { orig: 'mental_health',          src: 'mental_health_self_report',          q: 'How would you rate your mental health recently?' },
  physical_health:        { orig: 'physical_health',        src: 'physical_health_self_report',        q: 'How would you rate your physical health recently?' },
  mental_stress:          { orig: 'mental_stress',          src: 'mental_stress_self_report',          q: 'How mentally stressed are you feeling right now?' },
  physical_stress:        { orig: 'physical_stress',        src: 'physical_stress_self_report',        q: 'Are you experiencing any physical stress or tension?' },
  pain_level:             { orig: 'pain_level',             src: 'pain_level_self_report',             q: 'Are you experiencing any pain right now?' },
};

const RAW_VALUE: Record<string, Record<string, number>> = {
  age:                    { '18_29': 0.5 },
  sex:                    { male: 0.5 },
  education_level:        { college_grad: 0.75 },
  marital_status:         { single: 0.5 },
  bmi:                    { normal: 0.5 },
  diabetes_status:        { none: 0.0 },
  chronic_condition:      { no: 0.0 },
  smoking:                { not_at_all: 0.0 },
  alcohol_use:            { low: 0.15 },
  neuroticism:            { high: 0.9 },
  extraversion:           { moderate_low: 0.35 },
  general_health:         { fair: 0.45 },
  stress_ema:             { low: 0.15, moderate_low: 0.38, moderate_high: 0.65, high: 0.88 },
  mood:                   { low: 0.2, moderate_low: 0.38, high: 0.8 },
  productivity:           { low: 0.2, high: 0.8 },
  sleep_quality:          { poor: 0.2, fair: 0.55, good: 0.85 },
  sleep_disturbances:     { low: 0.1, moderate_low: 0.35, moderate_high: 0.65, high: 0.9 },
  exercise:               { none: 0.0, light: 0.3, moderate: 0.6, vigorous: 0.9 },
  social_events_negative: { low: 0.1, high: 0.9 },
  social_events_positive: { low: 0.1, moderate_low: 0.35, moderate_high: 0.65, high: 0.9 },
  loneliness:             { low: 0.1, moderate: 0.5, high: 0.9 },
  negative_affect:        { low: 0.1, moderate_low: 0.35, moderate_high: 0.65, high: 0.9 },
  positive_affect:        { low: 0.1, moderate_low: 0.35, moderate_high: 0.65, high: 0.9 },
  depression:             { none: 0.0, mild: 0.25, moderate: 0.5, moderate_severe: 0.75, severe: 1.0 },
  mental_health:          { low: 0.2, moderate: 0.5, high: 0.85 },
  physical_health:        { low: 0.2, moderate_low: 0.4, moderate_high: 0.65, high: 0.9 },
  mental_stress:          { low: 0.15, moderate: 0.5, high: 0.88 },
  physical_stress:        { low: 0.1, moderate: 0.5, high: 0.9 },
  pain_level:             { none: 0.0, some: 0.5, significant: 0.9 },
};

const LABEL_TEXT: Record<string, Record<string, string>> = {
  age:                    { '18_29': 'Age 18–29' },
  sex:                    { male: 'Male' },
  education_level:        { college_grad: 'College graduate' },
  marital_status:         { single: 'In a relationship (not married)' },
  bmi:                    { normal: 'Normal BMI' },
  diabetes_status:        { none: 'No diabetes' },
  chronic_condition:      { no: 'No chronic conditions' },
  smoking:                { not_at_all: 'Non-smoker' },
  alcohol_use:            { low: 'Occasional, social drinking' },
  neuroticism:            { high: 'High — emotionally reactive, internalises stress' },
  extraversion:           { moderate_low: 'Somewhat introverted — social mainly on weekends' },
  general_health:         { fair: 'Fair — managing but some concerns' },
  stress_ema: {
    low: 'Calm and relaxed', moderate_low: 'Mild, manageable stress',
    moderate_high: 'Noticeably stressed', high: 'Very stressed, overwhelmed',
  },
  mood:                   { low: 'Low mood, feeling down', moderate_low: 'Somewhat flat', high: 'Good mood, feeling positive' },
  productivity:           { low: 'Low productivity', high: 'High productivity' },
  sleep_quality:          { poor: 'Slept poorly', fair: 'Sleep was okay, not great', good: 'Slept well, feel rested' },
  sleep_disturbances:     { low: 'Minimal disruptions', moderate_low: 'Some disruptions', moderate_high: 'Frequent disruptions', high: 'Very disrupted sleep' },
  exercise: {
    none: 'No exercise', light: 'Light activity',
    moderate: 'Moderate gym session', vigorous: 'Vigorous workout or outdoor sport',
  },
  social_events_negative: { low: 'No negative social events', high: 'Significant negative social interaction' },
  social_events_positive: {
    low: 'Little positive social contact', moderate_low: 'Some positive interaction',
    moderate_high: 'Good positive social activity', high: 'Very positive, fulfilling social day',
  },
  loneliness:             { low: 'Not feeling lonely', moderate: 'Somewhat lonely', high: 'Very isolated and lonely' },
  negative_affect:        { low: 'Minimal negative feelings', moderate_low: 'Some negative feelings', moderate_high: 'Frequent negative feelings', high: 'Persistently strong negative feelings' },
  positive_affect:        { low: 'Little positive feeling', moderate_low: 'Some positive feeling', moderate_high: 'Good positive emotion', high: 'Very positive' },
  depression:             { none: 'No depressive symptoms', mild: 'Mild depressive symptoms', moderate: 'Moderate depression', moderate_severe: 'Moderately severe depression' },
  mental_health:          { low: 'Poor mental health', moderate: 'Fair mental health', high: 'Good mental health' },
  physical_health:        { low: 'Poor physical health', moderate_low: 'Below average physical health', moderate_high: 'Good physical health', high: 'Excellent physical health' },
  mental_stress:          { low: 'Low mental stress', moderate: 'Moderate mental stress', high: 'High mental stress' },
  physical_stress:        { low: 'Low physical stress', moderate: 'Moderate tension', high: 'High physical stress, significant discomfort' },
  pain_level:             { none: 'No pain', some: 'Some pain', significant: 'Significant pain' },
};

function makeSL(
  nodeName: string,
  nodeValue: string,
  reportDate: string,
  ts: string,
  turnId: string,
  opts?: { confidence?: number; mergeMode?: string; temporalFlag?: string; dataSource?: string },
): SensorlessRow {
  const meta  = NODE_META[nodeName];
  const label = LABEL_TEXT[nodeName]?.[nodeValue] ?? nodeValue;
  const raw   = RAW_VALUE[nodeName]?.[nodeValue] ?? 0.5;
  return {
    timestamp:       ts,
    node_name:       nodeName,
    original_column: meta.orig,
    source_column:   meta.src,
    question_text:   meta.q,
    raw_text:        label,
    node_value:      nodeValue,
    raw_value:       raw,
    summary_text:    label,
    confidence:      opts?.confidence ?? 0.85,
    data_source:     opts?.dataSource ?? 'self_report',
    merge_mode:      opts?.mergeMode  ?? 'latest',
    temporal_flag:   opts?.temporalFlag ?? 'decaying',
    report_date:     reportDate,
    expires_date:    null,
    turn_id:         turnId,
    is_active:       1,
    was_proactive:   0,
    answered:        1,
    created_at:      ts,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Main export
// ─────────────────────────────────────────────────────────────────────────────

const SESSION = 'seed-personality-v1';
const PERSONA_TURN_PREFIX = 'pseed';

export function seedPersonalityData(db: DB): void {

  // ── Clear previous personality seed ─────────────────────────────────────────
  db.executeSync(`DELETE FROM user_data_sensorless WHERE turn_id LIKE '${PERSONA_TURN_PREFIX}%'`);
  db.executeSync(`DELETE FROM sensor_windows WHERE data_source = 'personality_seed'`);
  db.executeSync(`DELETE FROM chat_messages WHERE session_id = '${SESSION}'`);
  db.executeSync(`DELETE FROM memory_summaries WHERE session_id = '${SESSION}'`);
  db.executeSync(`DELETE FROM inference_snapshots WHERE trigger_type = 'personality_seed'`);

  // ─────────────────────────────────────────────────────────────────────────
  // SECTION 1: Persistent traits (seeded ~179 days ago)
  // ─────────────────────────────────────────────────────────────────────────

  const traitDay     = daysAgo(179);
  const traitDate    = toDateStr(traitDay);
  const traitTs      = toTs(traitDay, 10, 0);
  const traitTurnId  = `${PERSONA_TURN_PREFIX}-trait`;
  const traitOpts    = { mergeMode: 'latest', temporalFlag: 'persistent', confidence: 0.95 };

  const traits: Array<[string, string]> = [
    ['age',             '18_29'],
    ['sex',             'male'],
    ['education_level', 'college_grad'],
    ['marital_status',  'single'],     // in a relationship but single status node
    ['bmi',             'normal'],
    ['diabetes_status', 'none'],
    ['chronic_condition','no'],
    ['smoking',         'not_at_all'],
    ['alcohol_use',     'low'],
    ['neuroticism',     'high'],
    ['extraversion',    'moderate_low'],
    ['general_health',  'fair'],
  ];
  for (const [node, val] of traits) {
    insertSL(db, makeSL(node, val, traitDate, traitTs, traitTurnId, traitOpts));
  }

  // ─────────────────────────────────────────────────────────────────────────
  // SECTION 2: Daily sensorless for 180 days
  //
  // Personality timeline (offset from today = 0):
  //   Days 179–8:   GYM ACTIVE PHASE
  //     - Gym days: Mon(1) Tue(2) Thu(4) Sat(6) — vigorous/moderate exercise
  //     - Weekdays: stress high, mood low
  //     - Weekend: stress low/moderate_low, mood high, outdoor activity
  //     - Sleep: good on gym days, fair on weekends, poor on bad weekdays
  //     - Fight clusters: scattered at offsets matching 2-3 days AFTER a heavy week
  //     - Weekend reset shortens across the arc (later offsets = shorter reset)
  //   Days 7–0:     BACK PAIN / POST-GYM phase
  //     - No gym (exercise = none), steps crash to 2000-3000
  //     - Sleep quality degrading (no gym fatigue)
  //     - Stress high on weekdays AND building on weekends
  //     - Screen time rising
  //     - Anxiety mentions in chat escalating
  // ─────────────────────────────────────────────────────────────────────────

  // Fight offset days (relative to today).
  // Fights cluster 2-3 days AFTER a particularly bad Mon-Fri stretch.
  // E.g. after a Mon–Fri stretch ending offset X, fights appear at offset X-2 or X-3.
  // These are pre-calculated to spread realistically across 180 days.
  const FIGHT_DAYS = new Set<number>([
    // early phase (weeks 25-20 = offsets ~175-140): 2 fights
    172, 161,
    // mid phase (weeks 19-12 = offsets ~133-84): 3 fights
    125, 108, 89,
    // recent phase (weeks 11-4 = offsets ~77-28): 4 fights — frequency increasing
    71, 55, 40, 29,
    // post-injury (offsets 7-1): 2 more — anxiety raw, more volatile
    5, 2,
  ]);

  // Back pain onset: offset 7 = 7 days ago
  const BACK_PAIN_ONSET = 7;

  // Outdoor weekend days (hike/trek/sport) — pre-calculated to Saturday/Sunday offsets
  // Note: each one is genuine outdoor. Weekend resets shorten: early ones last
  // 1-2 days of good mood, later ones just the day itself.
  const OUTDOOR_DAYS = new Set<number>([
    175, 168, 161, 154, 148, 141, 134, 127, 120, 113,
    106, 99, 92, 85, 78, 71, 64, 57, 50, 43, 36, 21, 14,
  ]);

  for (let offset = 179; offset >= 0; offset--) {
    const day      = daysAgo(offset);
    const dateStr  = toDateStr(day);
    const dow      = day.getDay(); // 0=Sun,1=Mon,...,6=Sat
    const ts       = toTs(day, 20, 0);
    const turnId   = `${PERSONA_TURN_PREFIX}-daily-${toYMD(day)}`;

    const isWeekday   = dow >= 1 && dow <= 5;
    const isGymDay    = offset > BACK_PAIN_ONSET && (dow === 1 || dow === 2 || dow === 4 || dow === 6);
    const isFightDay  = FIGHT_DAYS.has(offset);
    const isOutdoor   = OUTDOOR_DAYS.has(offset);
    const isPostInjury = offset <= BACK_PAIN_ONSET;
    const isInjuryDay  = offset === BACK_PAIN_ONSET || offset === BACK_PAIN_ONSET - 1;

    const ins = (node: string, val: string, conf?: number): void => {
      insertSL(db, makeSL(node, val, dateStr, ts, turnId, { confidence: conf }));
    };

    // ── stress_ema ──────────────────────────────────────────────────────────
    let stress: string;
    if (isPostInjury) {
      // After back pain: weekday stress very high, weekend stress now moderate_high too
      // (gym was the only valve — weekend relief shortening)
      stress = isWeekday ? 'high' : offset <= 3 ? 'moderate_high' : 'moderate_low';
    } else if (isFightDay) {
      stress = 'high';
    } else if (isOutdoor) {
      stress = 'low';
    } else if (isWeekday) {
      // Pre-injury weekday: stress high but oscillates — early weeks slightly lower
      stress = offset > 100 ? 'moderate_high' : 'high';
    } else {
      // Pre-injury weekend without outdoor: moderate_low (gym helps recovery)
      stress = 'moderate_low';
    }
    ins('stress_ema', stress);

    // ── mood ────────────────────────────────────────────────────────────────
    let mood: string;
    if (isPostInjury) {
      mood = isWeekday ? 'low' : offset <= 3 ? 'low' : 'moderate_low';
    } else if (isFightDay) {
      mood = 'low';
    } else if (isOutdoor) {
      mood = 'high';
    } else if (isWeekday) {
      mood = 'low';
    } else {
      // Weekend, non-outdoor, pre-injury: moderate_low → gym recovery day
      mood = 'moderate_low';
    }
    ins('mood', mood);

    // ── productivity ────────────────────────────────────────────────────────
    const prod = (isWeekday && !isFightDay) ? 'low' : (!isWeekday ? 'low' : 'low');
    if (offset % 3 === 0) ins('productivity', prod);  // every 3rd day to avoid over-weighting

    // ── mental_stress ───────────────────────────────────────────────────────
    // Stale quickly — only insert on recent days or clearly high weeks
    if (offset <= 30 || offset % 7 === 0) {
      const ms = (isWeekday || isPostInjury) ? 'high' : 'moderate';
      ins('mental_stress', ms);
    }

    // ── physical_stress ─────────────────────────────────────────────────────
    if (isInjuryDay) {
      ins('physical_stress', 'high');
    } else if (offset < BACK_PAIN_ONSET) {
      // Post-injury: residual tension from sitting, no gym
      if (offset <= 5) ins('physical_stress', 'moderate');
    } else if (isGymDay && offset % 5 === 0) {
      ins('physical_stress', 'moderate');
    }

    // ── sleep_quality + sleep_disturbances ──────────────────────────────────
    if (offset % 2 === 0) {
      let sleepQ: string;
      let sleepD: string;
      if (isPostInjury) {
        // Sleep degrading without gym fatigue
        sleepQ = offset <= 3 ? 'poor' : 'fair';
        sleepD = offset <= 3 ? 'high' : 'moderate_high';
      } else if (isGymDay) {
        // Gym days: great sleep from physical tiredness
        sleepQ = 'good';
        sleepD = 'low';
      } else if (isFightDay) {
        sleepQ = 'poor';
        sleepD = 'high';
      } else if (isOutdoor) {
        sleepQ = 'good';
        sleepD = 'low';
      } else if (isWeekday) {
        sleepQ = 'poor';
        sleepD = 'moderate_high';
      } else {
        // Weekend, no outdoor, pre-injury
        sleepQ = 'fair';
        sleepD = 'moderate_low';
      }
      ins('sleep_quality', sleepQ);
      ins('sleep_disturbances', sleepD);
    }

    // ── exercise ────────────────────────────────────────────────────────────
    if (isPostInjury) {
      if (offset % 2 === 0) ins('exercise', 'none');
    } else if (isGymDay) {
      ins('exercise', dow === 2 ? 'vigorous' : 'moderate');
    } else if (isOutdoor) {
      ins('exercise', 'vigorous');
    } else if (offset % 3 === 0) {
      ins('exercise', 'none');
    }

    // ── social_events_positive ──────────────────────────────────────────────
    if (isOutdoor) {
      ins('social_events_positive', 'high');
    } else if (!isWeekday && isGymDay) {
      ins('social_events_positive', 'moderate_high');
    } else if (isWeekday && offset % 5 === 0) {
      ins('social_events_positive', 'moderate_low');
    }

    // ── social_events_negative ──────────────────────────────────────────────
    if (isFightDay) {
      ins('social_events_negative', 'high');
    } else if (isWeekday && offset % 7 === 0) {
      ins('social_events_negative', 'low');
    }

    // ── Weekly nodes ─────────────────────────────────────────────────────────
    const isWeeklyInsert = dow === 0 || (dow === 1 && offset % 7 < 2);
    if (isWeeklyInsert) {
      // loneliness
      const loneVal = isFightDay ? 'high' : isPostInjury ? 'moderate' : 'low';
      ins('loneliness', loneVal, 0.75);

      // negative_affect — arcs upward over 180 days
      let negAff: string;
      if (offset > 120)      negAff = 'moderate_low';
      else if (offset > 60)  negAff = 'moderate_high';
      else if (offset > 14)  negAff = 'high';
      else                   negAff = 'high';  // post-injury sustained
      ins('negative_affect', negAff, 0.75);

      // positive_affect
      const posAff = isOutdoor ? 'moderate_high' : isPostInjury ? 'low' : 'moderate_low';
      ins('positive_affect', posAff, 0.75);

      // depression — escalates over time:
      // early 180-120 days: none/mild; 120-60: mild; 60-14: mild/moderate; <14: moderate
      let depr: string;
      if (offset > 120)      depr = 'none';
      else if (offset > 60)  depr = 'mild';
      else if (offset > 14)  depr = 'moderate';
      else                   depr = 'moderate';
      ins('depression', depr, 0.75);

      // mental_health
      const mh = depr === 'moderate' ? 'low' : depr === 'mild' ? 'moderate' : 'high';
      ins('mental_health', mh, 0.75);

      // physical_health
      let ph: string;
      if (isPostInjury)       ph = 'moderate_low';
      else if (isGymDay)      ph = 'moderate_high';
      else if (offset > 60)   ph = 'high';
      else                    ph = 'moderate_high';
      ins('physical_health', ph, 0.75);
    }

    // ── pain_level (injury days) ──────────────────────────────────────────────
    if (isInjuryDay) {
      ins('pain_level', 'significant');
    } else if (offset === BACK_PAIN_ONSET - 2 || offset === BACK_PAIN_ONSET - 3) {
      ins('pain_level', 'some');
    }

    // ── loneliness boost on fight days outside weekly ─────────────────────────
    if (isFightDay && !isWeeklyInsert) {
      ins('loneliness', 'high', 0.8);
      ins('negative_affect', 'high', 0.8);
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // SECTION 3: sensor_windows — dashboard-compatible columns
  //
  // Dashboard queries:
  //   getDailySteps      → source_column = 'prev_day_steps'
  //   getDailyActiveRatio → source_column = 'prev_day_active_ratio'
  //   getSleepDuration   → node_name='sleep_quality', source_column='sleep_hours'
  //   getDailyScreenUsage → source_column = 'screen_time_window_minutes'  (SUM per day)
  //   getNighttimeUsage  → same but snapshot_time >= '20:00:00'
  //   hasScreenPermission → node_name = 'screen_usage'
  //
  // Also insert legacy columns for backward compat / report sensor trend queries:
  //   active_ratio, daily_steps, screen_time_hours
  // ─────────────────────────────────────────────────────────────────────────

  for (let offset = 179; offset >= 0; offset--) {
    const day       = daysAgo(offset);
    const dateStr   = toDateStr(day);
    const dow       = day.getDay();
    const snapTs    = toISOLocal(day, 23, 30);
    const nightTs   = `${dateStr}T22:00:00`;  // nighttime window snapshot

    const isGymDay    = offset > BACK_PAIN_ONSET && (dow === 1 || dow === 2 || dow === 4 || dow === 6);
    const isOutdoor   = OUTDOOR_DAYS.has(offset);
    const isPostInjury = offset <= BACK_PAIN_ONSET;
    const isWeekday   = dow >= 1 && dow <= 5;
    const isFightDay  = FIGHT_DAYS.has(offset);

    // ── Step count ──────────────────────────────────────────────────────────
    let steps: number;
    if (isPostInjury) {
      // Post-injury: very sedentary — 2000-3000
      steps = 2000 + sr(offset * 17 + 3, 0, 1000);
    } else if (isOutdoor) {
      // Outdoor weekend: 9000-13000
      steps = 9000 + sr(offset * 11 + 7, 0, 4000);
    } else if (isGymDay) {
      // Gym day: 7000-10000 (Tuesday vigorous = upper end)
      steps = dow === 2
        ? 9000 + sr(offset * 13 + 1, 0, 1500)
        : 7000 + sr(offset * 7 + 5, 0, 2500);
    } else if (isWeekday) {
      // Non-gym weekday: 3500-5000
      steps = 3500 + sr(offset * 19 + 9, 0, 1500);
    } else {
      // Rest weekend (not outdoor): 4000-6000
      steps = 4000 + sr(offset * 23 + 11, 0, 2000);
    }

    // Slight downward trend in pre-injury weekday steps as weeks pass
    // (fatigue accumulating): reduce by up to 8% linearly over the 180 days
    if (!isPostInjury && isWeekday) {
      const fatigueFactor = 1 - (0.08 * (179 - offset) / 179);
      steps = Math.round(steps * fatigueFactor);
    }

    const stepDisc = steps >= 6000 ? 'high' : steps >= 3500 ? 'moderate' : 'low';

    // Dashboard column: prev_day_steps
    insertSW(db, {
      date: dateStr, snapshot_time: snapTs, window_start: null,
      node_name: 'activity', source_column: 'prev_day_steps',
      data_source: 'personality_seed', raw_value: steps, raw_unit: 'steps',
      discretized_value: stepDisc, confidence: 0.9,
    });
    // Dashboard column: prev_day_active_ratio (mean steps per 15-min window; ~96 windows/day)
    const activeRatioRaw = steps / 96;
    insertSW(db, {
      date: dateStr, snapshot_time: snapTs, window_start: null,
      node_name: 'activity', source_column: 'prev_day_active_ratio',
      data_source: 'personality_seed', raw_value: activeRatioRaw, raw_unit: 'steps_per_window',
      discretized_value: stepDisc, confidence: 0.9,
    });
    // Legacy columns (report sensor trend queries use 'active_ratio' and 'daily_steps')
    insertSW(db, {
      date: dateStr, snapshot_time: snapTs, window_start: null,
      node_name: 'activity', source_column: 'active_ratio',
      data_source: 'personality_seed', raw_value: activeRatioRaw, raw_unit: 'ratio',
      discretized_value: stepDisc, confidence: 0.88,
    });
    insertSW(db, {
      date: dateStr, snapshot_time: snapTs, window_start: null,
      node_name: 'activity', source_column: 'daily_steps',
      data_source: 'personality_seed', raw_value: steps, raw_unit: 'steps',
      discretized_value: stepDisc, confidence: 0.88,
    });

    // ── Screen time (rising trend as gym stops) ─────────────────────────────
    // Pre-injury: 480-540 min/day (8-9 hrs)
    // Post-injury day 7-0: 540-660 min/day (9-11 hrs), trending up
    // Weekday base is higher; weekends slightly lower
    let screenMinBase: number;
    if (isPostInjury) {
      // 9h base, rising ~7min/day since injury onset
      const daysAfterInjury = BACK_PAIN_ONSET - offset;
      screenMinBase = 540 + daysAfterInjury * 8 + sr(offset * 7 + 2, 0, 60);
    } else if (isWeekday) {
      screenMinBase = 480 + sr(offset * 11 + 3, 0, 60);
    } else {
      screenMinBase = 420 + sr(offset * 13 + 5, 0, 90);
    }
    // A gradual background rise over 6 months even pre-injury (work screen time creep)
    const baselineRise = (!isPostInjury) ? Math.round((179 - offset) * 0.5) : 0;
    const screenMin = screenMinBase + baselineRise;

    // Dashboard column: screen_time_window_minutes (daytime aggregate — split across 3 windows)
    // We insert a single "all-day" row; getDailyScreenUsage does SUM, so one row is fine.
    // Mark as screen_usage node so hasScreenPermission returns true.
    insertSW(db, {
      date: dateStr, snapshot_time: `${dateStr}T18:00:00`, window_start: null,
      node_name: 'screen_usage', source_column: 'screen_time_window_minutes',
      data_source: 'personality_seed', raw_value: screenMin * 0.7, raw_unit: 'minutes',
      discretized_value: 'high', confidence: 0.88,
    });
    // Nighttime window: 20:00 — portion of screen time that is evening use
    const nightMin = screenMin * 0.3 + (isPostInjury ? 30 : 0);
    insertSW(db, {
      date: dateStr, snapshot_time: nightTs, window_start: null,
      node_name: 'screen_usage', source_column: 'screen_time_window_minutes',
      data_source: 'personality_seed', raw_value: nightMin, raw_unit: 'minutes',
      discretized_value: 'high', confidence: 0.85,
    });
    // Legacy column for report sensor trend queries
    insertSW(db, {
      date: dateStr, snapshot_time: snapTs, window_start: null,
      node_name: 'screen_usage', source_column: 'screen_time_hours',
      data_source: 'personality_seed', raw_value: +(screenMin / 60).toFixed(1), raw_unit: 'hours',
      discretized_value: 'high', confidence: 0.88,
    });

    // ── Sleep duration (dashboard: node_name='sleep_quality', source_column='sleep_hours') ──
    let sleepHrs: number;
    if (isPostInjury) {
      // Without gym fatigue: difficulty falling asleep → 6.0-6.5 hrs
      sleepHrs = 6.0 + sr(offset * 13 + 7, 0, 30) / 100;
    } else if (isGymDay) {
      // Gym fatigue → deep sleep: 7.0-7.5 hrs
      sleepHrs = 7.0 + sr(offset * 7 + 1, 0, 30) / 100;
    } else if (isFightDay) {
      // Fight night → poor sleep
      sleepHrs = 5.0 + sr(offset * 11 + 3, 0, 60) / 100;
    } else if (isOutdoor) {
      // Outdoor exhaustion → good sleep
      sleepHrs = 7.5 + sr(offset * 3 + 9, 0, 30) / 100;
    } else if (isWeekday) {
      // Weekday non-gym: 6.0-6.5 hrs (screen + stress delays sleep)
      sleepHrs = 6.0 + sr(offset * 17 + 5, 0, 45) / 100;
    } else {
      // Weekend rest: 6.5-7.0
      sleepHrs = 6.5 + sr(offset * 19 + 7, 0, 30) / 100;
    }
    insertSW(db, {
      date: dateStr, snapshot_time: `${dateStr}T07:00:00`, window_start: null,
      node_name: 'sleep_quality', source_column: 'sleep_hours',
      data_source: 'personality_seed', raw_value: +sleepHrs.toFixed(1), raw_unit: 'hours',
      discretized_value: sleepHrs >= 7.0 ? 'good' : sleepHrs >= 6.0 ? 'fair' : 'poor',
      confidence: 0.88,
    });
  }

  // ─────────────────────────────────────────────────────────────────────────
  // SECTION 4: chat_messages — 42 realistic message pairs across 180 days
  // ─────────────────────────────────────────────────────────────────────────

  function addPair(
    offsetDay: number,
    hour: number,
    minute: number,
    userMsg: string,
    modelMsg: string,
    topic: string | null,
    idx: number,
  ): void {
    const d    = daysAgo(offsetDay);
    const ts   = toTs(d, hour, minute);
    const tU   = `${PERSONA_TURN_PREFIX}-chat-${offsetDay}-${idx}-u`;
    const tM   = `${PERSONA_TURN_PREFIX}-chat-${offsetDay}-${idx}-m`;
    const base: Omit<ChatRow, 'turn_id' | 'role' | 'content'> = {
      timestamp: ts, session_id: SESSION, topic,
      evicted: 0, is_active: 1, created_at: ts,
    };
    insertChat(db, { ...base, turn_id: tU, role: 'user',  content: userMsg  });
    insertChat(db, { ...base, turn_id: tM, role: 'model', content: modelMsg });
  }

  // ── 6 months ago (offsets ~175-160) — early, gym active, manageable ────────
  addPair(175, 7, 30,
    "Morning. Just finished a solid gym session before work. Bench PR today. Feeling good.",
    "Nice start to the week — hitting a personal record before the work day even begins is a great way to set the tone. How's the workload looking this week?",
    'exercise', 1);

  addPair(172, 21, 0,
    "Had a massive argument with my parents tonight. They're upset about my career choices again. Just exhausting.",
    "Arguments about career are uniquely draining because they touch something really personal — your sense of direction and identity. What specifically set it off this time?",
    'family_conflict', 2);

  addPair(170, 22, 15,
    "I'm okay now. Went to the gym this evening, which helped clear my head after that fight.",
    "That's a healthy way to process — physical movement really does help shift the mental state after emotionally charged events. How are you feeling about the argument now with a bit of distance?",
    'exercise', 3);

  addPair(165, 9, 15,
    "Work has been absolutely brutal this week. Three urgent projects dropped on me Monday morning. No time to breathe.",
    "Back-to-back urgent projects right from Monday is exhausting before the week even gets momentum. How many hours are you putting in — and are you getting any breaks at all?",
    'work_stress', 4);

  addPair(161, 20, 30,
    "Had another argument at home. Same old stuff. I feel like my parents never see me as capable.",
    "That repeated pattern of not feeling seen or trusted by your parents is genuinely painful — it goes deeper than just the argument itself. How long has this been a tension point?",
    'family_conflict', 5);

  addPair(158, 14, 0,
    "Went trekking with friends this weekend. Honestly felt amazing. Mountains just clear everything.",
    "That sounds genuinely restorative — the combination of physical effort, natural scenery, and real social connection is hard to beat. How are you feeling compared to the work-heavy days earlier this week?",
    'positive_activity', 6);

  // ── 4-5 months ago (offsets ~150-100) — sustained stress, gym still active ──
  addPair(148, 21, 45,
    "I feel like I'm always just surviving the week, not actually living. Work eats everything.",
    "That feeling of just making it through rather than being present in your own life is a real sign that the load has been too high for too long. Has it always felt this way, or is this a relatively recent shift?",
    'work_stress', 7);

  addPair(141, 7, 0,
    "Saturday hike done. 14km. Legs are dead but my mind is clear for the first time all week.",
    "14km — that's a serious effort. The clarity you feel after that kind of sustained movement is real, not just a mood lift. You're getting a full cortisol reset. Are you noticing a difference in how the weekend feels on hike days versus days you rest?",
    'positive_activity', 8);

  addPair(134, 22, 30,
    "My girlfriend got upset with me tonight. I was snappy and short with her. The work stress is bleeding into everything.",
    "That spillover from work stress into personal relationships is really common, especially when there's no buffer between work mode and home mode. When work is at its peak, does the tension at home usually follow a few days after, or does it hit the same day?",
    'relationship_conflict', 9);

  addPair(130, 10, 0,
    "Good week actually. Got on top of things. Gym four times. Feeling like myself.",
    "That's the Arjun that sounds healthiest — multiple gym sessions, work under control, feeling grounded. What made this week different from the others?",
    'positive', 10);

  addPair(125, 21, 0,
    "Fight with my girlfriend again. She says I've been distant and irritable. She's not wrong.",
    "The fact that you can acknowledge she's right takes real honesty. When you're running on empty from work — does the irritability feel like it comes from frustration, or more like a kind of emotional flatness?",
    'relationship_conflict', 11);

  addPair(120, 8, 0,
    "Did a trail run on Sunday. 10k. Body's tired but I feel good. This kind of thing saves me on bad weeks.",
    "You've identified something important — that outdoor physical activity is doing something very specific for you that isn't replaceable with gym work alone. The combination of nature and sustained movement is your reset mechanism.",
    'positive_activity', 12);

  addPair(113, 22, 0,
    "Work is just relentless. I know it's been like this for months but lately it's feeling less manageable.",
    "The fact that it's starting to feel less manageable even though the load hasn't necessarily changed — that suggests your recovery isn't keeping up with the demands anymore. What's different in how you're recovering compared to a few months ago?",
    'work_stress', 13);

  addPair(108, 20, 15,
    "My parents called again. Same old pressure. They want me to find a better-paying job, settle down, all of that. I just shut down on the phone.",
    "Shutting down on the call is your nervous system's way of protecting itself when the input feels overwhelming and familiar in the worst way. How are you feeling now — do you process it later or does it just sit there?",
    'family_conflict', 14);

  // ── 2-3 months ago (offsets ~85-50) — stress escalating, gym still active ───
  addPair(92, 13, 0,
    "Finally got outside again. Hiked with two friends. First proper outdoor thing in almost two weeks. It hit different.",
    "Two weeks is a long time to go without that kind of reset — no wonder it hit differently. You were overdue. How do you feel compared to where you were this time last week?",
    'positive_activity', 15);

  addPair(89, 20, 30,
    "Got into a fight with my girlfriend last night. I'd been running on fumes all week and she noticed I'd completely checked out. Fair criticism.",
    "That pattern — heavy work week, then the relationship absorbs the fallout 2-3 days later — is something worth noticing. Did the fight feel like it came out of nowhere for her, or had the tension been building?",
    'relationship_conflict', 16);

  addPair(85, 9, 30,
    "I've been noticing that even my gym days feel different lately. I'm still going but I'm more going through the motions than actually pushing.",
    "That shift from driven engagement to just going through the motions is worth paying attention to — it's often one of the earlier signs that your mental energy reserves are depleted even when your physical routine looks intact. How's your sleep been?",
    'exercise', 17);

  addPair(78, 23, 0,
    "My boss keeps scheduling these 7am calls. I have no morning time anymore. The gym is getting squeezed.",
    "Losing morning gym time hits harder than just missing a workout — you lose the decompression buffer that sets your mental state for the day. Is the 7am thing occasional or is it becoming a fixed pattern?",
    'work_stress', 18);

  addPair(71, 20, 0,
    "Big argument with my parents this evening. They think I don't have my life together. Sometimes I wonder if they're right.",
    "Those fights land harder when you're already exhausted — and the self-doubt they trigger can feel very real in the moment even when it isn't. The pattern I'm noticing: the worst arguments seem to cluster after your hardest work weeks. Does that match your experience?",
    'family_conflict', 19);

  addPair(64, 10, 0,
    "Weekend sport. Played football with friends. Haven't done that in a while. Body aches but in a good way.",
    "That kind of social, outdoor physical activity — football, hiking, all of it — gives you something gym sessions don't. The competition and play element on top of the movement. How often are you fitting this in now?",
    'positive_activity', 20);

  addPair(57, 22, 30,
    "I've been having anxiety a bit more often lately. Not panic attacks, just this constant low-level hum of dread. Especially after tough work weeks.",
    "That low-level hum is your nervous system staying in alert mode rather than recovering. It tends to appear when the stress load has been sustained for a while without enough relief. What does the anxiety feel like — physical tension, racing thoughts, or something else?",
    'anxiety', 21);

  addPair(55, 20, 15,
    "Girlfriend said she needs more emotional presence from me. I heard her. I know I've been distant. The fight ended okay but it's not resolved.",
    "That kind of ending — heard but unresolved — often means the underlying thing (your emotional availability) is still a live issue. The distance you're showing her probably feels different to her than it does to you from inside it. What do you think is actually driving the withdrawal?",
    'relationship_conflict', 22);

  // ── 4-6 weeks ago (offsets ~42-22) — stress high, anxiety increasing ─────────
  addPair(43, 9, 0,
    "I did a long hike yesterday — 18km with friends. Absolutely beautiful trail. Forgot about everything for hours.",
    "18km is a real mental vacation — the sustained physical effort and beauty just crowd out everything else. That's one of the cleanest stress resets there is. How long does the relief typically last after one of these days now?",
    'positive_activity', 23);

  addPair(40, 21, 0,
    "Fight with my girlfriend again tonight. She said I seem checked out and she's worried about me. She's not wrong, I just don't know how to fix it.",
    "She's worried about you — not just frustrated. That's actually a different thing. The question she's asking isn't just 'be more present' — it's 'what's happening with you'. What do you think is really driving the checked-out feeling lately?",
    'relationship_conflict', 24);

  addPair(36, 14, 0,
    "Went for a run in the park on Saturday. Not a long one but it felt necessary. The anxiety has been bad this week.",
    "Reaching for a run when the anxiety spikes is instinct working correctly — your body knows movement helps. Has the anxiety been linked to specific work situations this week, or is it more free-floating?",
    'anxiety', 25);

  addPair(29, 20, 30,
    "My parents called tonight. Same tension. They don't understand the pressure I'm under. This argument felt different though — more resigned than angry. Like I'm just tired of it.",
    "The shift from anger to resignation in a recurring conflict is significant — it often means you're emotionally depleted enough that you can't even muster the energy to push back. That fatigue is telling you something. How long have you been running like this?",
    'family_conflict', 26);

  addPair(21, 7, 15,
    "Saturday hike done. Beautiful morning. Felt good but honestly not as restorative as they used to be. The good feeling fades faster now.",
    "That observation — the reset is shorter, the good feeling fades quicker — is worth sitting with. It might mean the accumulated load is simply harder to offset with a single day. What's different about your baseline between now and, say, three months ago?",
    'positive_activity', 27);

  addPair(17, 23, 0,
    "I've been having more anxiety episodes this week. Nothing acute but persistent. Background noise that won't go away. Work was brutal and I barely slept.",
    "Persistent anxiety without acute episodes is actually harder to deal with in some ways — there's no clear peak, just constant drain. Your sleep deprivation is almost certainly amplifying it. When did you last have a genuinely good night of sleep?",
    'anxiety', 28);

  addPair(14, 10, 30,
    "Sunday hike. Good one. Felt almost normal for a few hours. Then back to reality Monday.",
    "Almost normal — that contrast says a lot. Monday morning the reality load rushes back in and the outdoor reset gets undone quickly. How long does the lift usually last now — hours, a day, more?",
    'positive_activity', 29);

  // ── Back pain onset (offsets 7-4) ──────────────────────────────────────────
  addPair(7, 11, 0,
    "Something happened this morning. I was warming up at the gym and felt a sharp pull in my lower back. Had to stop. It's really painful to even sit.",
    "That sudden onset during a warmup is scary — the back is brutal when it goes. You were right to stop immediately. Is it a sharp localised pain or does it radiate anywhere? Try to not sit for long stretches today and get some heat on it.",
    'physical_pain', 30);

  addPair(7, 19, 30,
    "Back still very bad. Skipped work in the afternoon. Lying down is the only comfortable position. I can't believe this happened.",
    "Your frustration makes complete sense — you had a solid routine going and it just got yanked away. For now the priority is just getting through the acute phase safely. How is the pain on a scale — still at its worst, or has it eased at all since morning?",
    'physical_pain', 31);

  addPair(6, 10, 0,
    "Day two of this back thing. Still can't really sit. Skipping the gym obviously. Feeling really low about it.",
    "The loss of your main physical outlet when you're already under stress is genuinely hard. Gym wasn't just exercise for you — it was your decompression mechanism. The low feeling makes complete sense. Are you icing or heating it? Have you seen anyone about it?",
    'physical_pain', 32);

  // ── Post-injury days (offsets 5-0) ──────────────────────────────────────────
  addPair(5, 21, 15,
    "Had a fight with my girlfriend tonight. She was frustrated that I've been distant and irritable. I know the back pain is making everything worse but I snapped at her.",
    "You're in pain, you can't do the thing that usually regulates you, and the stress is still coming — of course it's bleeding outward. She probably senses the distance even if she doesn't know the full context. How are you feeling about the conversation now?",
    'relationship_conflict', 33);

  addPair(4, 9, 0,
    "Back is a bit better but still can't sit at my desk comfortably. Screen time is going up because I'm just lying down and watching things. I feel kind of useless.",
    "You're injured — resting is the right call, even when it doesn't feel productive. The 'useless' feeling is worth noticing though. Your sense of self-worth seems quite tied to being physically active. Is that something you've always felt?",
    'physical_pain', 34);

  addPair(3, 22, 30,
    "Anxiety is getting worse. Without the gym I feel like I have no pressure valve. Everything is just building up.",
    "You've just named something really important — the gym was your primary pressure valve, and now it's gone. The anxiety building without it isn't a coincidence. What would be the smallest possible physical thing you could do right now that wouldn't hurt your back?",
    'anxiety', 35);

  addPair(2, 20, 0,
    "My parents called. Usual pressure. I know I handled it badly — got short with them. Just don't have the bandwidth right now.",
    "You don't have your main coping mechanism, you're in physical pain, and you're still absorbing the full work load. Your bandwidth being depleted isn't an excuse — it's a fact. How is the back actually feeling today versus a few days ago?",
    'family_conflict', 36);

  addPair(2, 22, 0,
    "I'm noticing I'm on my phone way more since I can't go to the gym. Like I'm filling the time with scrolling. It doesn't help.",
    "You're observing the compensatory behaviour in real time — that's actually insight. Screen time filling the space where movement was is a very common pattern when physical outlets disappear. The scrolling won't decompress you the way exercise does, but the impulse to fill the void makes sense.",
    'anxiety', 37);

  addPair(1, 10, 0,
    "Back is slowly improving but still nowhere near gym-able. Sleep has been terrible — I used to be tired from the gym and now I'm just... awake at night thinking.",
    "That's the flip side of gym-driven fatigue — it was actually helping you fall asleep. Without it, the physiological readiness to sleep is reduced, and the mental load takes over. The insomnia and the anxiety are probably feeding each other right now.",
    'sleep', 38);

  addPair(1, 21, 0,
    "Missed my second weekend hike because of the back. That was my only outdoor thing left. Now even that's gone.",
    "You've lost two of your primary resilience anchors in the same week — gym workouts and weekend outdoor activity. That's a significant compounding loss. How are you managing right now? Not physically — emotionally.",
    'positive_activity', 39);

  // ── TODAY (offset 0) ───────────────────────────────────────────────────────
  addPair(0, 9, 0,
    "Work is piling on as usual. My boss doesn't know about the back thing and I'm trying to manage but it's really hard to concentrate.",
    "Physical pain is a constant cognitive drain even when you're trying to push through it. Your concentration dropping isn't a willpower issue — it's your nervous system spending resources on managing the pain. Has the back been better or worse today?",
    'work_stress', 40);

  addPair(0, 17, 30,
    "I feel so frustrated and tired right now. I had so much work without any rest. I am having anxiety right now.",
    "I hear you, and what you're describing makes complete sense given everything that's converged. You lost your main stress release seven days ago when your back went out. Before that, weekday stress was already high — but the gym was absorbing it. Without that outlet, the anxiety has nowhere to go. The work pressure hasn't reduced, sleep has been worse since you're not gym-tired anymore, and two of your weekend resets have been cut off. The frustration and anxiety you're feeling right now aren't separate things — they're the same load with fewer exits. What would help most in this exact moment: talking through what's driving the anxiety, or thinking about what you can actually do in the next few hours?",
    'anxiety', 41);

  // ─────────────────────────────────────────────────────────────────────────
  // SECTION 5: memory_summaries — 12 narrative summaries
  // ─────────────────────────────────────────────────────────────────────────

  const summaries: Array<{ offset: number; count: number; text: string }> = [
    {
      offset: 175,
      count: 4,
      text: 'Baseline period: gym active 4-5×/week (Mon/Tue/Thu/Sat), 8000-10000 steps on gym days. Work stress already present but manageable — gym acts as primary stress buffer. First parent fight at offset 172 (career pressure); resolved within 2 days. Pattern: fights cluster 2-3 days after heavy workweeks.',
    },
    {
      offset: 155,
      count: 6,
      text: 'Work stress sustained over weeks. Gym routine intact. Weekend hikes and trail runs giving strong mood resets (steps 10000-13000 on outdoor days). Girlfriend friction first mentioned at offset 134 — said Arjun was "snappy and short." Pattern emerging: relationship tension follows 2-3 days after highest-stress work weeks, not during them.',
    },
    {
      offset: 130,
      count: 6,
      text: 'Mid-arc: gym days still providing good sleep (7.0-7.5 hrs from gym fatigue). Weekday sleep poor (6.0-6.5 hrs, screen use delaying sleep). Stress_ema high on weekdays for months now. Outdoor weekends = only consistent mood-high trigger. Girlfriend reported Arjun is "distant and irritable" — he acknowledged it but attributed it to work.',
    },
    {
      offset: 110,
      count: 4,
      text: 'Anxiety episodes first explicitly reported at offset 57: "constant low-level hum of dread, especially after tough work weeks." Gym still active and absorbing most of the load, but cracks appearing: gym sessions described as "going through the motions" at offset 85. Weekend outdoor resets still firing but duration of mood-lift shortening.',
    },
    {
      offset: 90,
      count: 4,
      text: 'Parent fights now clustered more tightly after work weeks (offsets 89, 71). Pattern confirmed: fight 2-3 days after worst work stretches. Girlfriend fight at offset 89 — same dynamic. At offset 57, anxiety explicitly named for first time. At offset 55, girlfriend raised concern about emotional presence directly.',
    },
    {
      offset: 70,
      count: 4,
      text: 'Accumulation visible: at offset 57, anxiety persistent; at offset 40, girlfriend said Arjun is "checked out." At offset 21, hike still happened but noted "not as restorative as they used to be — the good feeling fades faster." Weekend mood-reset duration clearly shortening across the past 8 weeks.',
    },
    {
      offset: 50,
      count: 4,
      text: 'At offset 36, anxiety episodes described as "bad this week." At offset 29, parent fight ended in resignation not anger ("just tired of it") — emotional depletion visible. At offset 21, hike-mood reset duration shortening. Steps: outdoor days 10000-13000, weekday non-gym 3500-4500, trending down slightly from accumulated fatigue.',
    },
    {
      offset: 30,
      count: 4,
      text: 'Recent weeks: anxiety frequency increasing. Sleep pattern: gym nights good, non-gym weekdays poor. Screen time 480-540 min/day pre-injury (higher on weekdays). Relationship friction with girlfriend: fights at offsets 134, 125, 89, 55, 40. Parent fights at offsets 172, 161, 125, 108, 71, 29. All fight clusters follow bad work weeks by 2-3 days.',
    },
    {
      offset: 15,
      count: 4,
      text: 'Final weeks before injury: at offset 17, "more anxiety episodes this week, background noise." At offset 14, Sunday hike — "felt almost normal for a few hours, then back to reality Monday." Hike mood-reset now lasting <1 day. At offset 7, BACK PAIN ONSET: sharp pull in lower back at gym warmup. Gym stopped immediately.',
    },
    {
      offset: 7,
      count: 6,
      text: 'Back pain onset at offset 7: gym halted entirely. Physical pain + loss of gym = dual stressor. Sleep immediately worsened (no gym fatigue helping sleep). Anxiety escalating: at offset 3, "without the gym I have no pressure valve — everything is building up." Screen time rising rapidly as compensatory behaviour: now 540-660 min/day.',
    },
    {
      offset: 4,
      count: 4,
      text: 'Post-injury cascade: girlfriend fight at offset 5 (Arjun snapped at her). Parent fight at offset 2 (no bandwidth). At offset 2, Arjun noticed compensatory scrolling himself: "filling the time with scrolling, it doesn\'t help." Steps now 2000-3000/day (was 7000-10000 on gym days). Weekend hikes missed for two consecutive weekends.',
    },
    {
      offset: 1,
      count: 4,
      text: 'Current state (offset 1-0): gym stopped 7 days ago (back pain). Both weekend hikes missed. Sleep: 6.0-6.5 hrs/night (was 7.0-7.5 on gym nights). Screen time peak at ~630 min/day. Anxiety at its highest in 6 months. Steps ~2200/day. Weekday work stress unchanged, recovery capacity at its lowest. All major stress outlets (gym, outdoor hikes, relationship buffer) simultaneously compromised.',
    },
  ];

  for (const s of summaries) {
    const d  = daysAgo(s.offset);
    const ts = toTs(d, 23, 0);
    db.executeSync(
      `INSERT INTO memory_summaries (timestamp, session_id, summary_text, embedding, message_count, created_at)
       VALUES (?, ?, ?, ?, ?, ?)`,
      [ts, SESSION, s.text, null, s.count, ts],
    );
  }

  // ─────────────────────────────────────────────────────────────────────────
  // SECTION 6: inference_snapshots — 10 snapshots across 180 days
  //
  // CRITICAL: beliefs must NOT be null on first open. These snapshots give the
  // dashboard rings and stress trend chart real data immediately.
  //
  // Each snapshot uses the DBN node names as they appear in the inferencEngine.
  // The dashboard beliefsToScores() uses: depression, anxiety, fatigue_level,
  // pain_intensity — only depression is a real DBN node; the rest are included
  // as representative distributions so rings show sensible values.
  //
  // The main DBN nodes (from NODE_META): stress_ema, mental_stress, physical_stress,
  // mood, sleep_quality, exercise, depression, pain_level, etc.
  // ─────────────────────────────────────────────────────────────────────────

  type NodeDist = Record<string, number>;
  type Beliefs  = Record<string, NodeDist>;

  function snap(
    offsetDay: number,
    hour: number,
    beliefs: Beliefs,
    summaryLine: string,
  ): void {
    const d        = daysAgo(offsetDay);
    const dateStr  = toDateStr(d);
    const timeStr  = `${dateStr}T${String(hour).padStart(2, '0')}:00:00`;
    const nodeConf: Record<string, number> = {};
    for (const node of Object.keys(beliefs)) nodeConf[node] = 0.85;
    db.executeSync(
      `INSERT OR REPLACE INTO inference_snapshots
         (date, snapshot_time, trigger_type, prior_beliefs,
          sensor_snapshot, sensorless_snapshot, dbn_beliefs,
          node_confidences, node_data_sources, summary_line)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      [
        dateStr, timeStr, 'personality_seed',
        JSON.stringify({}),   // prior_beliefs
        JSON.stringify({}),   // sensor_snapshot
        JSON.stringify({}),   // sensorless_snapshot
        JSON.stringify(beliefs),
        JSON.stringify(nodeConf),
        JSON.stringify(Object.fromEntries(Object.keys(beliefs).map(n => [n, 'self_report']))),
        summaryLine,
      ],
    );
  }

  // Snapshot 1: ~170 days ago — baseline good phase
  snap(170, 21, {
    stress_ema:             { low: 0.15, moderate_low: 0.45, moderate_high: 0.30, high: 0.10 },
    mental_stress:          { low: 0.20, moderate: 0.55, high: 0.25 },
    physical_stress:        { low: 0.45, moderate: 0.40, high: 0.15 },
    mood:                   { low: 0.10, moderate_low: 0.35, high: 0.55 },
    sleep_quality:          { poor: 0.10, fair: 0.35, good: 0.55 },
    exercise:               { none: 0.05, light: 0.10, moderate: 0.50, vigorous: 0.35 },
    depression:             { none: 0.65, mild: 0.25, moderate: 0.08, moderate_severe: 0.02 },
    pain_level:             { none: 0.90, some: 0.08, significant: 0.02 },
    negative_affect:        { low: 0.35, moderate_low: 0.40, moderate_high: 0.20, high: 0.05 },
    positive_affect:        { low: 0.10, moderate_low: 0.30, moderate_high: 0.45, high: 0.15 },
    loneliness:             { low: 0.60, moderate: 0.30, high: 0.10 },
    // Dashboard ring nodes
    anxiety:                { none: 0.55, minimal: 0.25, mild: 0.15, moderate: 0.05 },
    fatigue_level:          { none: 0.30, mild: 0.45, moderate: 0.20, severe: 0.05 },
    pain_intensity:         { none: 0.88, mild: 0.09, moderate: 0.03 },
  }, 'Baseline: gym active, stress manageable, mood generally positive on gym/outdoor days.');

  // Snapshot 2: ~130 days ago — stress rising, gym still active
  snap(130, 22, {
    stress_ema:             { low: 0.08, moderate_low: 0.27, moderate_high: 0.42, high: 0.23 },
    mental_stress:          { low: 0.10, moderate: 0.45, high: 0.45 },
    physical_stress:        { low: 0.35, moderate: 0.45, high: 0.20 },
    mood:                   { low: 0.25, moderate_low: 0.45, high: 0.30 },
    sleep_quality:          { poor: 0.25, fair: 0.45, good: 0.30 },
    exercise:               { none: 0.10, light: 0.10, moderate: 0.50, vigorous: 0.30 },
    depression:             { none: 0.45, mild: 0.35, moderate: 0.17, moderate_severe: 0.03 },
    pain_level:             { none: 0.88, some: 0.10, significant: 0.02 },
    negative_affect:        { low: 0.18, moderate_low: 0.38, moderate_high: 0.32, high: 0.12 },
    positive_affect:        { low: 0.20, moderate_low: 0.40, moderate_high: 0.30, high: 0.10 },
    loneliness:             { low: 0.40, moderate: 0.42, high: 0.18 },
    anxiety:                { none: 0.35, minimal: 0.30, mild: 0.25, moderate: 0.10 },
    fatigue_level:          { none: 0.20, mild: 0.40, moderate: 0.30, severe: 0.10 },
    pain_intensity:         { none: 0.85, mild: 0.12, moderate: 0.03 },
  }, 'Stress rising over sustained work load. Weekend outdoor resets still effective. Relationship friction beginning.');

  // Snapshot 3: ~90 days ago — anxiety first explicit, stress entrenched
  snap(90, 21, {
    stress_ema:             { low: 0.05, moderate_low: 0.15, moderate_high: 0.40, high: 0.40 },
    mental_stress:          { low: 0.05, moderate: 0.35, high: 0.60 },
    physical_stress:        { low: 0.30, moderate: 0.40, high: 0.30 },
    mood:                   { low: 0.40, moderate_low: 0.40, high: 0.20 },
    sleep_quality:          { poor: 0.38, fair: 0.42, good: 0.20 },
    exercise:               { none: 0.12, light: 0.08, moderate: 0.50, vigorous: 0.30 },
    depression:             { none: 0.30, mild: 0.42, moderate: 0.23, moderate_severe: 0.05 },
    pain_level:             { none: 0.88, some: 0.10, significant: 0.02 },
    negative_affect:        { low: 0.10, moderate_low: 0.28, moderate_high: 0.40, high: 0.22 },
    positive_affect:        { low: 0.30, moderate_low: 0.40, moderate_high: 0.22, high: 0.08 },
    loneliness:             { low: 0.25, moderate: 0.50, high: 0.25 },
    anxiety:                { none: 0.20, minimal: 0.30, mild: 0.30, moderate: 0.20 },
    fatigue_level:          { none: 0.10, mild: 0.35, moderate: 0.40, severe: 0.15 },
    pain_intensity:         { none: 0.85, mild: 0.12, moderate: 0.03 },
  }, 'Anxiety surfacing explicitly. Weekday stress entrenched. Gym still active and buffering but weekend resets shortening.');

  // Snapshot 4: ~60 days ago — degrading baseline
  snap(60, 22, {
    stress_ema:             { low: 0.03, moderate_low: 0.10, moderate_high: 0.35, high: 0.52 },
    mental_stress:          { low: 0.03, moderate: 0.25, high: 0.72 },
    physical_stress:        { low: 0.20, moderate: 0.45, high: 0.35 },
    mood:                   { low: 0.52, moderate_low: 0.35, high: 0.13 },
    sleep_quality:          { poor: 0.48, fair: 0.38, good: 0.14 },
    exercise:               { none: 0.15, light: 0.08, moderate: 0.47, vigorous: 0.30 },
    depression:             { none: 0.18, mild: 0.42, moderate: 0.32, moderate_severe: 0.08 },
    pain_level:             { none: 0.88, some: 0.10, significant: 0.02 },
    negative_affect:        { low: 0.05, moderate_low: 0.20, moderate_high: 0.45, high: 0.30 },
    positive_affect:        { low: 0.42, moderate_low: 0.35, moderate_high: 0.18, high: 0.05 },
    loneliness:             { low: 0.15, moderate: 0.50, high: 0.35 },
    anxiety:                { none: 0.10, minimal: 0.20, mild: 0.40, moderate: 0.30 },
    fatigue_level:          { none: 0.05, mild: 0.25, moderate: 0.45, severe: 0.25 },
    pain_intensity:         { none: 0.85, mild: 0.12, moderate: 0.03 },
  }, 'High stress entrenched. Anxiety moderate and persistent. Mood low on weekdays. Weekend reset shortening.');

  // Snapshot 5: ~30 days ago — near pre-injury nadir
  snap(30, 21, {
    stress_ema:             { low: 0.02, moderate_low: 0.08, moderate_high: 0.28, high: 0.62 },
    mental_stress:          { low: 0.02, moderate: 0.18, high: 0.80 },
    physical_stress:        { low: 0.18, moderate: 0.42, high: 0.40 },
    mood:                   { low: 0.62, moderate_low: 0.28, high: 0.10 },
    sleep_quality:          { poor: 0.55, fair: 0.33, good: 0.12 },
    exercise:               { none: 0.18, light: 0.07, moderate: 0.45, vigorous: 0.30 },
    depression:             { none: 0.12, mild: 0.40, moderate: 0.38, moderate_severe: 0.10 },
    pain_level:             { none: 0.88, some: 0.10, significant: 0.02 },
    negative_affect:        { low: 0.03, moderate_low: 0.15, moderate_high: 0.45, high: 0.37 },
    positive_affect:        { low: 0.50, moderate_low: 0.30, moderate_high: 0.15, high: 0.05 },
    loneliness:             { low: 0.10, moderate: 0.48, high: 0.42 },
    anxiety:                { none: 0.05, minimal: 0.15, mild: 0.40, moderate: 0.40 },
    fatigue_level:          { none: 0.03, mild: 0.18, moderate: 0.48, severe: 0.31 },
    pain_intensity:         { none: 0.85, mild: 0.12, moderate: 0.03 },
  }, 'Pre-injury nadir: stress very high, mood low, anxiety moderate-high. Gym still buffering but recovery degraded.');

  // Snapshot 6: ~14 days ago — last good outdoor weekend
  snap(14, 20, {
    stress_ema:             { low: 0.10, moderate_low: 0.30, moderate_high: 0.38, high: 0.22 },
    mental_stress:          { low: 0.08, moderate: 0.42, high: 0.50 },
    physical_stress:        { low: 0.30, moderate: 0.40, high: 0.30 },
    mood:                   { low: 0.28, moderate_low: 0.38, high: 0.34 },
    sleep_quality:          { poor: 0.30, fair: 0.40, good: 0.30 },
    exercise:               { none: 0.10, light: 0.10, moderate: 0.35, vigorous: 0.45 },
    depression:             { none: 0.15, mild: 0.42, moderate: 0.35, moderate_severe: 0.08 },
    pain_level:             { none: 0.88, some: 0.10, significant: 0.02 },
    negative_affect:        { low: 0.10, moderate_low: 0.28, moderate_high: 0.40, high: 0.22 },
    positive_affect:        { low: 0.25, moderate_low: 0.38, moderate_high: 0.30, high: 0.07 },
    loneliness:             { low: 0.30, moderate: 0.45, high: 0.25 },
    anxiety:                { none: 0.10, minimal: 0.25, mild: 0.40, moderate: 0.25 },
    fatigue_level:          { none: 0.10, mild: 0.30, moderate: 0.40, severe: 0.20 },
    pain_intensity:         { none: 0.85, mild: 0.12, moderate: 0.03 },
  }, 'Sunday hike — brief mood lift. "Almost normal for a few hours." Reset duration now <1 day.');

  // Snapshot 7: ~7 days ago — back pain onset day
  snap(7, 18, {
    stress_ema:             { low: 0.02, moderate_low: 0.08, moderate_high: 0.25, high: 0.65 },
    mental_stress:          { low: 0.02, moderate: 0.15, high: 0.83 },
    physical_stress:        { low: 0.05, moderate: 0.20, high: 0.75 },
    mood:                   { low: 0.72, moderate_low: 0.22, high: 0.06 },
    sleep_quality:          { poor: 0.70, fair: 0.22, good: 0.08 },
    exercise:               { none: 0.90, light: 0.08, moderate: 0.02, vigorous: 0.00 },
    depression:             { none: 0.08, mild: 0.35, moderate: 0.42, moderate_severe: 0.15 },
    pain_level:             { none: 0.05, some: 0.25, significant: 0.70 },
    negative_affect:        { low: 0.02, moderate_low: 0.10, moderate_high: 0.40, high: 0.48 },
    positive_affect:        { low: 0.65, moderate_low: 0.25, moderate_high: 0.08, high: 0.02 },
    loneliness:             { low: 0.05, moderate: 0.35, high: 0.60 },
    anxiety:                { none: 0.03, minimal: 0.10, mild: 0.35, moderate: 0.52 },
    fatigue_level:          { none: 0.02, mild: 0.10, moderate: 0.38, severe: 0.50 },
    pain_intensity:         { none: 0.05, mild: 0.15, moderate: 0.50, severe: 0.30 },
  }, 'Back pain onset. Gym halted. Physical stress high. Mood crashed. Primary stress valve gone.');

  // Snapshot 8: ~4 days ago — post-injury cascade
  snap(4, 21, {
    stress_ema:             { low: 0.01, moderate_low: 0.05, moderate_high: 0.22, high: 0.72 },
    mental_stress:          { low: 0.01, moderate: 0.12, high: 0.87 },
    physical_stress:        { low: 0.08, moderate: 0.32, high: 0.60 },
    mood:                   { low: 0.78, moderate_low: 0.17, high: 0.05 },
    sleep_quality:          { poor: 0.75, fair: 0.20, good: 0.05 },
    exercise:               { none: 0.95, light: 0.05, moderate: 0.00, vigorous: 0.00 },
    depression:             { none: 0.05, mild: 0.28, moderate: 0.47, moderate_severe: 0.20 },
    pain_level:             { none: 0.08, some: 0.30, significant: 0.62 },
    negative_affect:        { low: 0.01, moderate_low: 0.07, moderate_high: 0.38, high: 0.54 },
    positive_affect:        { low: 0.72, moderate_low: 0.20, moderate_high: 0.06, high: 0.02 },
    loneliness:             { low: 0.03, moderate: 0.30, high: 0.67 },
    anxiety:                { none: 0.02, minimal: 0.06, mild: 0.28, moderate: 0.64 },
    fatigue_level:          { none: 0.01, mild: 0.07, moderate: 0.35, severe: 0.57 },
    pain_intensity:         { none: 0.05, mild: 0.15, moderate: 0.48, severe: 0.32 },
  }, 'Post-injury day 3: anxiety escalating, sleep worsened, screen time rising, girlfriend fight. No gym for 3 days.');

  // Snapshot 9: ~2 days ago
  snap(2, 22, {
    stress_ema:             { low: 0.01, moderate_low: 0.04, moderate_high: 0.18, high: 0.77 },
    mental_stress:          { low: 0.01, moderate: 0.09, high: 0.90 },
    physical_stress:        { low: 0.10, moderate: 0.38, high: 0.52 },
    mood:                   { low: 0.80, moderate_low: 0.15, high: 0.05 },
    sleep_quality:          { poor: 0.78, fair: 0.17, good: 0.05 },
    exercise:               { none: 0.97, light: 0.03, moderate: 0.00, vigorous: 0.00 },
    depression:             { none: 0.04, mild: 0.22, moderate: 0.50, moderate_severe: 0.24 },
    pain_level:             { none: 0.10, some: 0.35, significant: 0.55 },
    negative_affect:        { low: 0.01, moderate_low: 0.05, moderate_high: 0.32, high: 0.62 },
    positive_affect:        { low: 0.75, moderate_low: 0.18, moderate_high: 0.05, high: 0.02 },
    loneliness:             { low: 0.03, moderate: 0.25, high: 0.72 },
    anxiety:                { none: 0.01, minimal: 0.04, mild: 0.22, moderate: 0.73 },
    fatigue_level:          { none: 0.01, mild: 0.05, moderate: 0.30, severe: 0.64 },
    pain_intensity:         { none: 0.08, mild: 0.17, moderate: 0.43, severe: 0.32 },
  }, 'Day 5 post-injury: parent fight, compensatory scrolling noted. Anxiety highest in 6 months. Both weekend hikes missed.');

  // Snapshot 10: yesterday (most recent — what dashboard reads on first open)
  snap(1, 23, {
    stress_ema:             { low: 0.01, moderate_low: 0.03, moderate_high: 0.15, high: 0.81 },
    mental_stress:          { low: 0.01, moderate: 0.07, high: 0.92 },
    physical_stress:        { low: 0.12, moderate: 0.40, high: 0.48 },
    mood:                   { low: 0.82, moderate_low: 0.13, high: 0.05 },
    sleep_quality:          { poor: 0.80, fair: 0.15, good: 0.05 },
    exercise:               { none: 0.97, light: 0.03, moderate: 0.00, vigorous: 0.00 },
    depression:             { none: 0.03, mild: 0.20, moderate: 0.50, moderate_severe: 0.27 },
    pain_level:             { none: 0.12, some: 0.38, significant: 0.50 },
    negative_affect:        { low: 0.01, moderate_low: 0.04, moderate_high: 0.28, high: 0.67 },
    positive_affect:        { low: 0.78, moderate_low: 0.16, moderate_high: 0.04, high: 0.02 },
    loneliness:             { low: 0.02, moderate: 0.22, high: 0.76 },
    anxiety:                { none: 0.01, minimal: 0.03, mild: 0.18, moderate: 0.78 },
    fatigue_level:          { none: 0.01, mild: 0.04, moderate: 0.28, severe: 0.67 },
    pain_intensity:         { none: 0.10, mild: 0.18, moderate: 0.42, severe: 0.30 },
  }, 'Gym stopped 6 days ago. Sleep worst in months. Anxiety at its highest. Both weekend resets missed. All stress outlets compromised simultaneously.');
}
