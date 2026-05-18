/**
 * seedDb_middleAged.ts — 180-day persona seed for integration testing.
 *
 * Persona: Vikram Nair, 40yo male, married, sedentary office job.
 *   • No exercise routine — plays badminton once every 3-5 weeks (not conditioned)
 *   • 8-10 hrs/day at desk; chronic work stress, demanding boss, deadlines
 *   • BMI overweight; sleep poor (late screen, stress, occasional insomnia)
 *   • Oblique neck/back stiffness mentions scattered across chats (desk-related, casual)
 *   • Physical_stress rises steadily over 6 months from desk posture + muscle deconditioning
 *   • Back injury onset at offset 2: played badminton, acute onset during play
 *
 * Non-obvious patterns the model should discover (NOT stated anywhere in chats):
 *   1. Sedentary months → core deconditioning + poor posture → spine load elevated
 *   2. Chronic stress → persistent muscle tension (back/neck) → injury risk amplified
 *   3. BMI overweight + low steps = additional lumbar load
 *   4. Sudden unaccustomed badminton on deconditioned back = almost inevitable injury
 *   5. Poor sleep + high stress = slow tissue recovery → won't heal quickly
 *
 * Tables populated:
 *   - user_data_sensorless  (persistent traits + daily sensorless nodes)
 *   - sensor_windows        (steps, screen_time, sleep_hours — dashboard-compatible)
 *   - chat_messages         (37 realistic pairs over 180 days)
 *   - memory_summaries      (12 compressed narrative summaries)
 *   - inference_snapshots   (10 snapshots spread across 180 days)
 *
 * Call seedMiddleAgedData(db) from a __DEV__ guard.
 * Idempotent — clears previous seed rows by session_id / data_source first.
 *
 * ISOLATION (all differ from seedDb_personality.ts):
 *   SESSION              = 'seed-middleaged-v1'
 *   PERSONA_TURN_PREFIX  = 'maseed'
 *   data_source          = 'middleaged_seed'
 *   trigger_type         = 'middleaged_seed'
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
  const y   = d.getFullYear();
  const m   = String(d.getMonth() + 1).padStart(2, '0');
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
// Node metadata — mirrors seedDb.ts NODE_META / LABEL_TEXT / RAW_VALUE
// New values added: marital_status.married, bmi.overweight, neuroticism.moderate_high
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
  age:                    { '40_49': 0.65 },
  sex:                    { male: 0.5 },
  education_level:        { college_grad: 0.75 },
  marital_status:         { married: 0.7 },                 // new
  bmi:                    { overweight: 0.7 },              // new
  diabetes_status:        { none: 0.0 },
  chronic_condition:      { no: 0.0 },
  smoking:                { not_at_all: 0.0 },
  alcohol_use:            { low: 0.15 },
  neuroticism:            { moderate_high: 0.7 },           // new
  extraversion:           { moderate_low: 0.35 },
  general_health:         { fair: 0.45 },
  stress_ema:             { low: 0.15, moderate_low: 0.38, moderate_high: 0.65, high: 0.88 },
  mood:                   { low: 0.2, moderate_low: 0.38, moderate_high: 0.62, high: 0.8 },
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
  age:                    { '40_49': 'Age 40–49' },
  sex:                    { male: 'Male' },
  education_level:        { college_grad: 'College graduate' },
  marital_status:         { married: 'Married' },                               // new
  bmi:                    { overweight: 'Overweight' },                          // new
  diabetes_status:        { none: 'No diabetes' },
  chronic_condition:      { no: 'No chronic conditions' },
  smoking:                { not_at_all: 'Non-smoker' },
  alcohol_use:            { low: 'Occasional, social drinking' },
  neuroticism:            { moderate_high: 'Moderately high emotional reactivity' }, // new
  extraversion:           { moderate_low: 'Somewhat introverted — social mainly on weekends' },
  general_health:         { fair: 'Fair — managing but some concerns' },
  stress_ema: {
    low: 'Calm and relaxed', moderate_low: 'Mild, manageable stress',
    moderate_high: 'Noticeably stressed', high: 'Very stressed, overwhelmed',
  },
  mood: {
    low: 'Low mood, feeling down', moderate_low: 'Somewhat flat',
    moderate_high: 'Decent mood, reasonably positive', high: 'Good mood, feeling positive',
  },
  productivity:           { low: 'Low productivity', high: 'High productivity' },
  sleep_quality:          { poor: 'Slept poorly', fair: 'Sleep was okay, not great', good: 'Slept well, feel rested' },
  sleep_disturbances:     { low: 'Minimal disruptions', moderate_low: 'Some disruptions', moderate_high: 'Frequent disruptions', high: 'Very disrupted sleep' },
  exercise: {
    none: 'No exercise', light: 'Light activity',
    moderate: 'Moderate exercise session', vigorous: 'Vigorous workout or outdoor sport',
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
// Isolation constants — all differ from seedDb_personality.ts
// ─────────────────────────────────────────────────────────────────────────────

const SESSION             = 'seed-middleaged-v1';
const PERSONA_TURN_PREFIX = 'maseed';

// ─────────────────────────────────────────────────────────────────────────────
// Main export
// ─────────────────────────────────────────────────────────────────────────────

export function seedMiddleAgedData(db: DB): void {

  // ── Clear previous middle-aged seed ─────────────────────────────────────────
  db.executeSync(`DELETE FROM user_data_sensorless WHERE turn_id LIKE '${PERSONA_TURN_PREFIX}%'`);
  db.executeSync(`DELETE FROM sensor_windows WHERE data_source = 'middleaged_seed'`);
  db.executeSync(`DELETE FROM chat_messages WHERE session_id = '${SESSION}'`);
  db.executeSync(`DELETE FROM memory_summaries WHERE session_id = '${SESSION}'`);
  db.executeSync(`DELETE FROM inference_snapshots WHERE trigger_type = 'middleaged_seed'`);

  // ─────────────────────────────────────────────────────────────────────────
  // SECTION 1: Persistent traits (seeded ~179 days ago)
  // ─────────────────────────────────────────────────────────────────────────

  const traitDay    = daysAgo(179);
  const traitDate   = toDateStr(traitDay);
  const traitTs     = toTs(traitDay, 10, 0);
  const traitTurnId = `${PERSONA_TURN_PREFIX}-trait`;
  const traitOpts   = { mergeMode: 'latest', temporalFlag: 'persistent', confidence: 0.95 };

  const traits: Array<[string, string]> = [
    ['age',              '40_49'],
    ['sex',              'male'],
    ['education_level',  'college_grad'],
    ['marital_status',   'married'],
    ['bmi',              'overweight'],
    ['diabetes_status',  'none'],
    ['chronic_condition','no'],
    ['smoking',          'not_at_all'],
    ['alcohol_use',      'low'],
    ['neuroticism',      'moderate_high'],
    ['extraversion',     'moderate_low'],
    ['general_health',   'fair'],
  ];
  for (const [node, val] of traits) {
    insertSL(db, makeSL(node, val, traitDate, traitTs, traitTurnId, traitOpts));
  }

  // ─────────────────────────────────────────────────────────────────────────
  // SECTION 2: Daily sensorless for 180 days
  //
  // Key signals:
  //   SPORT_DAYS (offsets): 165, 130, 95, 60, 30, 15, 2 — vague "went for a game"
  //   SPORT_ONSET = 2       back injury occurred during play at offset 2
  //   STIFFNESS_DAYS:       120, 80, 45, 20 — oblique casual desk-related stiffness
  //   WORK_FIGHT_DAYS:      155, 110, 75, 42, 18 — boss/workload conflict
  //
  // Physical_stress arc (KEY deconditioning signal):
  //   offsets 179–90:  low  (every 3-4 weekdays)
  //   offsets 89–30:   moderate (consistent, chronic desk load building)
  //   offsets 29–3:    moderate  (high side — accumulation phase)
  //   offsets 2–0:     high (injury phase)
  //
  // Exercise: vigorous on sport days, none on all others (~96.1% none)
  // ─────────────────────────────────────────────────────────────────────────

  // Injury onset = 2 days ago (badminton → acute back injury)
  const SPORT_ONSET = 2;

  // Badminton/sport days (once every 3-5 weeks, sparse)
  const SPORT_DAYS = new Set<number>([165, 130, 95, 60, 30, 15, 2]);

  // Oblique stiffness mention days — data-layer physical_stress bump
  // These correspond to the casual chat mentions: "neck is killing me", "back stiff"
  const STIFFNESS_DAYS = new Set<number>([120, 80, 45, 20]);

  // Work conflict / boss tension days — social_events_negative + high stress
  const WORK_FIGHT_DAYS = new Set<number>([155, 110, 75, 42, 18]);

  for (let offset = 179; offset >= 0; offset--) {
    const day     = daysAgo(offset);
    const dateStr = toDateStr(day);
    const dow     = day.getDay();    // 0=Sun,1=Mon,...,6=Sat
    const ts      = toTs(day, 20, 0);
    const turnId  = `${PERSONA_TURN_PREFIX}-daily-${toYMD(day)}`;

    const isWeekday     = dow >= 1 && dow <= 5;
    const isSportDay    = SPORT_DAYS.has(offset);
    const isStiffDay    = STIFFNESS_DAYS.has(offset);
    const isWorkFight   = WORK_FIGHT_DAYS.has(offset);
    const isInjuryPhase = offset <= SPORT_ONSET;
    const isInjuryDay   = offset === SPORT_ONSET;
    const isPostInjury  = offset < SPORT_ONSET;   // days 1, 0

    const ins = (node: string, val: string, conf?: number): void => {
      insertSL(db, makeSL(node, val, dateStr, ts, turnId, { confidence: conf }));
    };

    // ── stress_ema ──────────────────────────────────────────────────────────
    // Pattern: weekday stress high for months; slightly lower early arc.
    // Weekend: moderate_low (family) except recent weeks where even weekends stressed.
    // Sport days: brief relief (moderate_low).
    // Work-fight days: high.
    let stress: string;
    if (isWorkFight) {
      stress = 'high';
    } else if (isSportDay && !isInjuryDay) {
      stress = 'moderate_low';
    } else if (isInjuryPhase) {
      // injury day + post: stress elevated (pain + immobility + work piling)
      stress = 'high';
    } else if (isWeekday) {
      // Early (179-90): moderate_high; Late (89-0): high — escalating
      stress = offset > 90 ? 'moderate_high' : 'high';
    } else {
      // Weekend: moderate_low early, moderate_high recent (less recovery)
      stress = offset > 60 ? 'moderate_low' : 'moderate_high';
    }
    ins('stress_ema', stress);

    // ── mood ────────────────────────────────────────────────────────────────
    let mood: string;
    if (isInjuryPhase) {
      mood = 'low';
    } else if (isWorkFight) {
      mood = 'low';
    } else if (isSportDay) {
      // Sport day: brief positive lift
      mood = 'moderate_low';
    } else if (isWeekday) {
      mood = 'low';
    } else {
      // Weekend: family time = moderate positive
      mood = offset > 60 ? 'moderate_high' : 'moderate_low';
    }
    ins('mood', mood);

    // ── productivity ────────────────────────────────────────────────────────
    // Weekdays every 3rd day: low (stress drains output)
    if (isWeekday && offset % 3 === 0) {
      ins('productivity', 'low');
    }

    // ── mental_stress ───────────────────────────────────────────────────────
    // Weekdays + fight days: high. Weekends: moderate. Recent: high even on weekends.
    if (offset <= 30 || offset % 7 === 0) {
      const ms = (isWeekday || isWorkFight || isInjuryPhase) ? 'high' : 'moderate';
      ins('mental_stress', ms);
    }

    // ── physical_stress (KEY SIGNAL — rising arc over 180 days) ────────────
    // This is the primary deconditioning + desk tension signal.
    // Insert every 3-4 weekdays consistently; stiffness days get a bump.
    // Arc: low (179-90) → moderate (89-30) → moderate→high (29-3) → high (2-0)
    if (isInjuryPhase) {
      // Injury day and post: high physical stress (acute + residual)
      ins('physical_stress', 'high');
    } else if (isStiffDay) {
      // Stiffness-complaint days: moderate (desk tension visible in data)
      ins('physical_stress', 'moderate');
    } else if (isWeekday && offset % 3 === 1) {
      // Regular weekday insertions — arc based on timeframe
      if (offset > 90) {
        ins('physical_stress', 'low');
      } else if (offset > 29) {
        ins('physical_stress', 'moderate');
      } else {
        // Recent 29-3: moderate trending toward high — body under cumulative load
        const val = offset > 10 ? 'moderate' : 'high';
        ins('physical_stress', val);
      }
    } else if (isSportDay && !isInjuryDay) {
      // Day after sport: brief muscle soreness / exertion spike
      ins('physical_stress', 'moderate');
    }

    // ── sleep_quality + sleep_disturbances (every 2 days) ──────────────────
    if (offset % 2 === 0) {
      let sleepQ: string;
      let sleepD: string;
      if (isInjuryPhase) {
        // Pain disrupts sleep
        sleepQ = 'poor';
        sleepD = 'high';
      } else if (isWorkFight) {
        // Work tension nights: poor sleep
        sleepQ = 'poor';
        sleepD = 'high';
      } else if (isSportDay) {
        // Physical exertion → slightly better sleep (but not great — deconditioned body aches)
        sleepQ = 'fair';
        sleepD = 'moderate_low';
      } else if (isWeekday) {
        // Weekday: late screen use + stress → poor to fair
        // Early arc fair, late arc mostly poor
        sleepQ = offset > 90 ? 'fair' : 'poor';
        sleepD = offset > 90 ? 'moderate_high' : 'high';
      } else {
        // Weekend: slightly better but still affected by screen + stress
        sleepQ = offset > 60 ? 'fair' : 'poor';
        sleepD = offset > 60 ? 'moderate_low' : 'moderate_high';
      }
      ins('sleep_quality', sleepQ);
      ins('sleep_disturbances', sleepD);
    }

    // ── exercise (vigorous on sport days, none otherwise) ───────────────────
    if (isSportDay && !isInjuryDay) {
      // Played sport — vigorous (but injury day = had to stop, insert none)
      ins('exercise', 'vigorous');
    } else if (offset % 2 === 0) {
      // Every other day to keep signal dense without over-weighting
      ins('exercise', 'none');
    }
    // On injury day: exercise = none (had to stop midway)
    if (isInjuryDay) {
      ins('exercise', 'none');
    }

    // ── social_events_positive (weekends — family) ──────────────────────────
    if (!isWeekday && !isInjuryPhase && offset % 4 === 0) {
      ins('social_events_positive', 'moderate_high');
    } else if (isSportDay && !isInjuryDay) {
      // Sport = social outing with friends/colleagues
      ins('social_events_positive', 'moderate_low');
    }

    // ── social_events_negative (work fight days) ────────────────────────────
    if (isWorkFight) {
      ins('social_events_negative', 'high');
    } else if (isWeekday && offset % 9 === 0) {
      ins('social_events_negative', 'low');
    }

    // ── Weekly nodes (Sunday or first-of-week) ──────────────────────────────
    const isWeeklyInsert = dow === 0 || (dow === 1 && offset % 7 < 2);
    if (isWeeklyInsert) {
      // loneliness — moderate on weekdays (isolated at desk), better on weekends
      const loneVal = isInjuryPhase ? 'moderate' : isWorkFight ? 'moderate' : 'low';
      ins('loneliness', loneVal, 0.75);

      // negative_affect — arcs upward across 180 days
      let negAff: string;
      if (offset > 120)     negAff = 'moderate_low';
      else if (offset > 60) negAff = 'moderate_high';
      else                  negAff = 'high';
      ins('negative_affect', negAff, 0.75);

      // positive_affect — weekends family positive, weekdays low
      const posAff = isSportDay ? 'moderate_low'
        : isInjuryPhase ? 'low'
        : !isWeekday ? 'moderate_high'
        : 'low';
      ins('positive_affect', posAff, 0.75);

      // depression — escalates but stays mild (not a depressed person — stressed)
      let depr: string;
      if (offset > 120)     depr = 'none';
      else if (offset > 60) depr = 'mild';
      else if (offset > 10) depr = 'mild';
      else                  depr = 'moderate';
      ins('depression', depr, 0.75);

      // mental_health
      const mh = depr === 'moderate' ? 'low' : depr === 'mild' ? 'moderate' : 'high';
      ins('mental_health', mh, 0.75);

      // physical_health — steady decline: high early → moderate_high mid → moderate_low recent → low post-injury
      let ph: string;
      if (isInjuryPhase)      ph = 'low';
      else if (offset > 120)  ph = 'high';
      else if (offset > 60)   ph = 'moderate_high';
      else                    ph = 'moderate_low';
      ins('physical_health', ph, 0.75);
    }

    // ── pain_level ───────────────────────────────────────────────────────────
    // None on all pre-injury days (not a chronic pain person).
    // Significant from injury day onwards (acute phase).
    if (isInjuryPhase) {
      ins('pain_level', 'significant');
    } else if (isWeeklyInsert) {
      // Explicit none on weekly snapshots — makes the arc from none→significant unmistakable
      ins('pain_level', 'none');
    }

    // ── Negative affect boost on work-fight days ─────────────────────────────
    if (isWorkFight && !isWeeklyInsert) {
      ins('negative_affect', 'high', 0.8);
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // SECTION 3: sensor_windows — dashboard-compatible
  //
  // Columns inserted:
  //   prev_day_steps, prev_day_active_ratio (dashboard getDailySteps)
  //   active_ratio, daily_steps (legacy report sensor trend)
  //   screen_time_window_minutes (daytime + nighttime windows)
  //   screen_time_hours (legacy)
  //   sleep_hours (dashboard getSleepDuration)
  // ─────────────────────────────────────────────────────────────────────────

  for (let offset = 179; offset >= 0; offset--) {
    const day     = daysAgo(offset);
    const dateStr = toDateStr(day);
    const dow     = day.getDay();
    const snapTs  = toISOLocal(day, 23, 30);
    const nightTs = `${dateStr}T22:00:00`;

    const isWeekday      = dow >= 1 && dow <= 5;
    const isSportDay     = SPORT_DAYS.has(offset);
    const isInjuryDay    = offset === SPORT_ONSET;
    const isPostInjury   = offset < SPORT_ONSET;

    // ── Step count (reflects sedentary office life) ──────────────────────────
    // Weekday non-sport: 3000-4500 (desk-bound)
    // Sport day:         6000-9000 (badminton + walking)
    // Weekend non-sport: 3500-5500 (slightly more movement)
    // Post-injury (1-0): 1500-2500 (pain limits movement)
    let steps: number;
    if (isPostInjury) {
      steps = 1500 + sr(offset * 17 + 3, 0, 1000);
    } else if (isSportDay && !isInjuryDay) {
      steps = 6000 + sr(offset * 11 + 7, 0, 3000);
    } else if (isInjuryDay) {
      // Injury day: started sport but stopped → moderate steps
      steps = 3500 + sr(offset * 13 + 1, 0, 1500);
    } else if (isWeekday) {
      steps = 3000 + sr(offset * 19 + 9, 0, 1500);
    } else {
      // Weekend rest: 3500-5500
      steps = 3500 + sr(offset * 23 + 11, 0, 2000);
    }

    const stepDisc = steps >= 6000 ? 'high' : steps >= 3500 ? 'moderate' : 'low';

    // prev_day_steps
    insertSW(db, {
      date: dateStr, snapshot_time: snapTs, window_start: null,
      node_name: 'activity', source_column: 'prev_day_steps',
      data_source: 'middleaged_seed', raw_value: steps, raw_unit: 'steps',
      discretized_value: stepDisc, confidence: 0.9,
    });
    // prev_day_active_ratio
    const activeRatioRaw = steps / 96;
    insertSW(db, {
      date: dateStr, snapshot_time: snapTs, window_start: null,
      node_name: 'activity', source_column: 'prev_day_active_ratio',
      data_source: 'middleaged_seed', raw_value: activeRatioRaw, raw_unit: 'steps_per_window',
      discretized_value: stepDisc, confidence: 0.9,
    });
    // Legacy: active_ratio
    insertSW(db, {
      date: dateStr, snapshot_time: snapTs, window_start: null,
      node_name: 'activity', source_column: 'active_ratio',
      data_source: 'middleaged_seed', raw_value: activeRatioRaw, raw_unit: 'ratio',
      discretized_value: stepDisc, confidence: 0.88,
    });
    // Legacy: daily_steps
    insertSW(db, {
      date: dateStr, snapshot_time: snapTs, window_start: null,
      node_name: 'activity', source_column: 'daily_steps',
      data_source: 'middleaged_seed', raw_value: steps, raw_unit: 'steps',
      discretized_value: stepDisc, confidence: 0.88,
    });

    // ── Screen time (work screen + evening scrolling) ──────────────────────
    // Weekday: 600-720 min/day (work laptop + evening scrolling)
    // Weekend: 480-600 min (less work, more leisure scrolling)
    // Post-injury: higher (lying down scrolling)
    let screenMin: number;
    if (isPostInjury) {
      // Post-injury: lying down, scrolling more — 660-780
      const daysAfterInjury = SPORT_ONSET - offset;
      screenMin = 660 + daysAfterInjury * 15 + sr(offset * 7 + 2, 0, 60);
    } else if (isWeekday) {
      screenMin = 600 + sr(offset * 11 + 3, 0, 120);
    } else {
      screenMin = 480 + sr(offset * 13 + 5, 0, 120);
    }
    // Background creep over 6 months (work screen time increasing)
    const baselineRise = (!isPostInjury) ? Math.round((179 - offset) * 0.3) : 0;
    screenMin = screenMin + baselineRise;

    // Daytime screen: screen_time_window_minutes (70% of daily total)
    insertSW(db, {
      date: dateStr, snapshot_time: `${dateStr}T18:00:00`, window_start: null,
      node_name: 'screen_usage', source_column: 'screen_time_window_minutes',
      data_source: 'middleaged_seed', raw_value: screenMin * 0.7, raw_unit: 'minutes',
      discretized_value: 'high', confidence: 0.88,
    });
    // Nighttime screen: 20:00+ (heavy evening scrolling)
    const nightMin = screenMin * 0.3 + (isPostInjury ? 40 : 0);
    insertSW(db, {
      date: dateStr, snapshot_time: nightTs, window_start: null,
      node_name: 'screen_usage', source_column: 'screen_time_window_minutes',
      data_source: 'middleaged_seed', raw_value: nightMin, raw_unit: 'minutes',
      discretized_value: 'high', confidence: 0.85,
    });
    // Legacy: screen_time_hours
    insertSW(db, {
      date: dateStr, snapshot_time: snapTs, window_start: null,
      node_name: 'screen_usage', source_column: 'screen_time_hours',
      data_source: 'middleaged_seed', raw_value: +(screenMin / 60).toFixed(1), raw_unit: 'hours',
      discretized_value: 'high', confidence: 0.88,
    });

    // ── Sleep duration ───────────────────────────────────────────────────────
    // Weekday: 5.5-6.5 hrs (late screen use, stress, occasional insomnia)
    // Weekend: 6.5-7.5 hrs
    // Post-injury: 5.0-6.0 hrs (pain disrupts)
    let sleepHrs: number;
    if (isPostInjury) {
      sleepHrs = 5.0 + sr(offset * 13 + 7, 0, 80) / 100;
    } else if (isWeekday) {
      sleepHrs = 5.5 + sr(offset * 17 + 5, 0, 100) / 100;
    } else {
      // Weekend: more rest but still screen + stress limited
      sleepHrs = 6.5 + sr(offset * 19 + 7, 0, 100) / 100;
    }
    insertSW(db, {
      date: dateStr, snapshot_time: `${dateStr}T07:00:00`, window_start: null,
      node_name: 'sleep_quality', source_column: 'sleep_hours',
      data_source: 'middleaged_seed', raw_value: +sleepHrs.toFixed(1), raw_unit: 'hours',
      discretized_value: sleepHrs >= 7.0 ? 'good' : sleepHrs >= 6.0 ? 'fair' : 'poor',
      confidence: 0.88,
    });
  }

  // ─────────────────────────────────────────────────────────────────────────
  // SECTION 4: chat_messages — 37 realistic pairs over 180 days
  //
  // CRITICAL CONSTRAINTS:
  //   - "badminton" NOT mentioned before injury day (offset 2)
  //   - Sport references before offset 2: "went for a game", "played with the guys" (vague)
  //   - No mention of psychosomatic link anywhere
  //   - Stiffness complaints before injury: desk-related, casual, NOT serious
  //   - Stress complaints: "tensed", "stressed" — NOT linked to back in any message
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

  // ── EARLY PHASE (offsets 175-120): manageable but stressed ─────────────────

  addPair(175, 9, 0,
    "Good morning. Just starting the week. Work already feels heavy before it has even begun.",
    "That sense of weight before Monday even gets going is worth paying attention to — it suggests you're not arriving rested, but already bracing. What's the biggest thing on your plate this week?",
    'work_stress', 1);

  addPair(170, 21, 30,
    "Had a long day. Meetings back to back, barely moved from my desk. My neck is killing me from staring at the screen all day.",
    "Eight or nine hours of screen time without moving is genuinely taxing on the neck and shoulders — the muscles stay contracted for long periods. Is the neck stiffness happening most days or only when it's been a particularly desk-heavy day?",
    'physical_discomfort', 2);

  addPair(165, 20, 0,
    "Went for a game with some friends from the area this evening. Good fun, bit out of breath, good to get out.",
    "That kind of unplanned social activity is good for you — not just the movement, but the shift in context from work mode. How are you feeling tonight versus a typical desk-heavy weekday evening?",
    'social_positive', 3);

  addPair(160, 22, 0,
    "I've been so tensed lately. Work is not giving me any breathing room. The boss is very demanding.",
    "That sustained tension from a demanding environment takes a real toll over time — especially when there's no clear end to it. How long has the pressure from your boss been at this level?",
    'work_stress', 4);

  addPair(155, 21, 15,
    "Had a rough day at work. My manager pulled me aside for a review that felt more like an attack. Very demoralising.",
    "A performance conversation that felt like an attack rather than a discussion is genuinely destabilising — it puts you on the defensive and makes it hard to even process the content. How are you feeling now that you've had a few hours away from it?",
    'work_conflict', 5);

  addPair(148, 19, 30,
    "Good weekend. Went out with the family on Sunday. Kids were happy. These are the moments that make everything worthwhile.",
    "Family days like that do something nothing else can — they shift the entire mental register. It sounds like you genuinely disconnected for a while. How different does Monday feel after a weekend like that versus a quiet one at home?",
    'family_positive', 6);

  addPair(140, 22, 0,
    "Work has been relentless for weeks now. I feel like I am always running behind, never catching up.",
    "That constant catching-up feeling is exhausting — it means you're never operating from a position of control, always reactive. Has the pace been this sustained for months, or did something shift recently to push it up?",
    'work_stress', 7);

  addPair(130, 20, 45,
    "Went for a game with the guys this weekend. Haven't done that in a while. Was a bit stiff the next morning but felt good otherwise.",
    "Good to hear you got out. That after-effect stiffness is pretty normal after a gap — the body isn't used to that kind of burst activity when it's been mostly sitting. Did the stiffness pass by midday or did it linger?",
    'social_positive', 8);

  addPair(125, 21, 0,
    "Very stressed again. Deadlines are piling up and my boss keeps adding more. I don't know when this will ease.",
    "Piling deadlines without relief is a real endurance problem — the end keeps moving. When you're under this kind of sustained load, what does your evening routine look like? Are you actually getting any mental downtime?",
    'work_stress', 9);

  addPair(120, 22, 30,
    "Neck and upper back are stiff today. Too many hours hunched over the laptop. Just one of those desk days.",
    "That stiffness is the body's way of flagging that the sustained posture has built up tension. A lot of desk workers get this by mid-afternoon and just push through. Are you getting up and moving at all during the day or is it mostly continuous sitting?",
    'physical_discomfort', 10);

  // ── MID PHASE (offsets 119-45): stress building, physical creep ─────────────

  addPair(115, 20, 0,
    "Work is really getting to me. I feel drained even before I reach home in the evening.",
    "Arriving home already depleted is a real sign that the load has been too high for too long — you're spending everything before you even get to the people at home. How long has the commute-home feeling been this way?",
    'work_stress', 11);

  addPair(110, 21, 30,
    "Difficult day. Had a disagreement with my manager over a project scope. He doesn't listen to anyone. Very frustrating.",
    "Disagreements with a manager who shuts down input are uniquely draining — it combines the frustration of the conflict itself with the helplessness of knowing it won't land. How is the project affected by this, practically speaking?",
    'work_conflict', 12);

  addPair(103, 22, 0,
    "I've been so tensed I can't sleep properly. Last two nights I kept waking up thinking about work.",
    "Work anxiety entering your sleep is a clear signal that the stress has crossed a threshold — it's no longer contained to work hours. The fact that it's happening across multiple nights suggests the load isn't resolving between days. What are you typically thinking about when you wake?",
    'sleep', 13);

  addPair(96, 20, 30,
    "Good Sunday. We took the kids to the park. My wife and I had some time together in the evening. Felt human again.",
    "That combination — active family time and quiet time with your wife — is genuinely restorative. It sounds like the weekend did its job for once. Are you feeling it already lifting heading into Monday, or does the anticipation start to creep in?",
    'family_positive', 14);

  addPair(90, 21, 15,
    "Back feels a bit stiff lately. Been sitting for too long at work. Nothing serious, just notice it more.",
    "That kind of low-grade back stiffness from extended sitting is very common but worth paying attention to — it means the postural load is accumulating. Are you getting any stretching in, or is it just sitting through it?",
    'physical_discomfort', 15);

  addPair(82, 22, 0,
    "Neck pain from the screen — too many hours on the laptop again. Comes and goes but today it was bad.",
    "Screen neck is incredibly common in long-desk days — the head-forward position puts a lot of load on the cervical spine. When it gets bad like today, does it stay localised to the neck or does it radiate down to the shoulders or between the shoulder blades?",
    'physical_discomfort', 16);

  addPair(80, 20, 45,
    "My back was stiff yesterday after sitting through two back-to-back reviews. Took some time to ease after I got up and moved around.",
    "The fact that moving around eased it is actually a good sign — it means it's postural load rather than something structural. But the pattern of it building through sedentary stretches is worth tracking. Are these review marathons regular or unusually frequent this month?",
    'physical_discomfort', 17);

  addPair(75, 21, 0,
    "My manager escalated a project issue to senior leadership without telling me first. Made me look completely incompetent in front of everyone. I am fuming.",
    "Being blindsided by your manager in front of leadership is a deeply destabilising experience — it undermines both your credibility and your trust in him. How are you processing this tonight? Is it mostly the anger or is there a layer of anxiety underneath it too?",
    'work_conflict', 18);

  addPair(68, 21, 45,
    "Work has been somewhat manageable this week. Had a rare evening without any pending tasks. Watched a film with my wife.",
    "A proper evening off sounds like it was genuinely needed. There's something about actually stopping — not just slowing down — that resets the system differently. How often do you get evenings like that?",
    'positive', 19);

  addPair(60, 20, 30,
    "Played a game with some colleagues this afternoon. First time in a while. Good fun. A bit sore in the legs afterward but nothing major.",
    "Getting out with colleagues and mixing a bit of activity into what's otherwise a static workday is good — a different kind of social than the usual office dynamic. How long had it been since the last time you played?",
    'social_positive', 20);

  addPair(52, 22, 15,
    "I feel physically tired even though I barely moved all day. How does sitting at a desk for 9 hours exhaust you?",
    "It's a real phenomenon — sustained mental load and postural tension without physical release is genuinely draining in a different way than physical exertion. The body is holding constant low-grade muscle tension to maintain posture while the mind is running at full capacity. Do you notice the fatigue more in your body or your mind at the end of these days?",
    'physical_discomfort', 21);

  addPair(45, 21, 0,
    "My back has been stiff on and off this week. Probably from the desk. Gets better if I move around a bit but comes back by afternoon.",
    "The on-again off-again pattern across the week suggests the postural load from your desk setup is creating a chronic low-level tension. Your body is telling you it needs movement breaks. Do you have any flexibility to stand up and move even briefly between meetings?",
    'physical_discomfort', 22);

  addPair(42, 22, 30,
    "Terrible week at work. My manager keeps moving the goalposts. I am exhausted and just want a break.",
    "Moving goalposts is one of the more depleting forms of work stress — it makes your effort feel futile because nothing you do is ever complete. That combination of exhaustion and futility is heavy. How long have you been dealing with this manager?",
    'work_conflict', 23);

  // ── RECENT PHASE (offsets 44-3): accumulation visible ─────────────────────

  addPair(35, 20, 0,
    "Work stress is really getting to me this month. I cannot switch off even in the evenings.",
    "The inability to switch off is a sign that the stress has exceeded what normal rest can contain — it's leaking into recovery time. What does your evening look like right now — are you checking work messages after hours?",
    'work_stress', 24);

  addPair(28, 21, 30,
    "Nice family weekend. Took the kids to their school event on Saturday. Wife was happy. These days are important.",
    "It sounds like Saturday gave you a real anchor — something that wasn't about work or pressure, just the family. Those moments matter especially when the weeks are heavy. Did you feel the benefit carry into Sunday or did the work anxiety come back?",
    'family_positive', 25);

  addPair(22, 22, 0,
    "My shoulders and neck are always tight these days. I think it's the stress as much as the desk. Everything feels tense.",
    "Persistent tightness in the neck and shoulders is your body holding the load of the day — both the physical posture and the mental tension tend to manifest in the same muscles. Are you getting any regular movement or is it mostly sitting through the day?",
    'physical_discomfort', 26);

  addPair(20, 21, 0,
    "My upper back and neck were really stiff today. Worst it's been in a while. I just got up and walked around after dinner and it helped a little.",
    "The fact that even a short walk after dinner gave some relief confirms this is postural and muscular — the muscles just need interruption of the sustained load. When the stiffness is this noticeable, it means you've been at the desk for very long unbroken stretches. Are there days where you barely get up at all?",
    'physical_discomfort', 27);

  addPair(18, 20, 30,
    "Boss gave me a very hard time in the all-hands meeting today. Singled me out in front of the whole team. Humiliating.",
    "Being singled out publicly by your manager is one of the more demoralising things that can happen at work — the professional embarrassment compounds with the injustice of it. What was his stated reason, and how are your colleagues reacting?",
    'work_conflict', 28);

  addPair(15, 20, 0,
    "Went for a game with Ravi today after work. Felt good to get out. Back felt a bit stiff afterward but okay by evening.",
    "Getting out and moving is exactly the kind of break you've needed. That post-game stiffness is pretty typical when you've been desk-bound for weeks — the body needs time to adjust to sudden activity. Did it ease fully by the time you slept, or was it still there in the morning?",
    'social_positive', 29);

  addPair(10, 21, 15,
    "Work stress is at its worst. I feel like I am running on empty. Even the weekends feel short.",
    "Running on empty while the weekends no longer feel long enough to recover is a sign the deficit is building — you're drawing down on reserves rather than filling them. How long has it been since you had a stretch of days that actually felt like rest?",
    'work_stress', 30);

  addPair(7, 22, 0,
    "Very low energy today. I barely did anything at home, just sat on the couch. Don't know why I feel so drained.",
    "That kind of profound tiredness even on a quieter day — when you haven't done much but feel depleted anyway — often comes from a long accumulated load rather than any single day's effort. How has the sleep been this week?",
    'fatigue', 31);

  addPair(5, 21, 0,
    "Work is piling up. My manager sent three emails after 10pm last night. I could not sleep well.",
    "Late-night messages from your manager are crossing a clear boundary even if they feel like the norm — they are forcing your nervous system to stay in work mode well into what should be recovery time. How long has this after-hours contact been a pattern?",
    'work_stress', 32);

  // ── INJURY DAY (offset 2) ──────────────────────────────────────────────────

  addPair(2, 11, 30,
    "I played badminton this morning with Ravi and my back just gave out mid-game. Severe pain. I stopped immediately. Can't even stand straight.",
    "That sounds very alarming — the kind of sudden onset during movement that puts you out immediately is the back signalling something sharp. You were right to stop straight away. Is the pain localised to one side of the lower back, or is it spreading? Please rest completely, apply ice for the first 20 minutes then heat, and avoid any bending or twisting. See a doctor today if the pain is severe or radiating down the leg.",
    'physical_pain', 33);

  addPair(2, 20, 0,
    "Still in a lot of pain. Lying flat is the only comfortable position. Doctor said muscle spasm and to rest for a week. This is awful timing with work.",
    "Muscle spasm diagnosis is reassuring in the sense that it's mechanical, not structural — but it's brutally painful and the recovery timeline is frustrating, especially with work pressure on top. Follow the doctor's rest recommendation strictly. What's the plan with your manager for this week?",
    'physical_pain', 34);

  // ── POST-INJURY (offsets 1-0) ──────────────────────────────────────────────

  addPair(1, 21, 0,
    "Back still very painful. Lying down mostly. Work is piling up and I can't sit at the desk. Feeling anxious about everything.",
    "You're in acute pain and completely removed from your normal routine — that anxiety is a very natural response to the loss of control. Your body needs this recovery time even though the timing is terrible. Is your wife able to help manage things at home this week?",
    'physical_pain', 35);

  addPair(0, 10, 0,
    "Third day of this back pain. It is not getting better. Sleep is terrible, can't get comfortable. Work emails are mounting up.",
    "Three days of significant pain with poor sleep is exhausting — the sleep deprivation and pain create a feedback loop that makes everything feel heavier. Muscle injuries like this often feel the same or worse for the first 3-5 days before they turn. Are you managing to rest without the phone and laptop, or is the work anxiety pulling you back to the screen?",
    'physical_pain', 36);

  addPair(0, 19, 0,
    "I cannot sit at all without pain. Work is completely paused. I feel so useless just lying here.",
    "Resting when your body is telling you it must is not being useless — it's the only thing that actually works for muscle recovery. The 'useless' feeling is the stress response reframing necessary rest as failure. You've been under sustained pressure for months; your body has finally asked for a stop. Let it stop.",
    'physical_pain', 37);

  // ─────────────────────────────────────────────────────────────────────────
  // SECTION 5: memory_summaries — 12 narrative summaries
  //
  // These tell the full 6-month story ending at the injury.
  // No psychosomatic link stated. Stress arc + physical arc both present.
  // Stiffness complaints in data AND summaries (oblique, desk-attributed).
  // ─────────────────────────────────────────────────────────────────────────

  const summaries: Array<{ offset: number; count: number; text: string }> = [
    {
      offset: 175,
      count: 4,
      text: 'Baseline established at offset 175. Vikram is a 40yo married male, sedentary office job, 8-10 hrs/day at desk. BMI overweight. No exercise routine. Steps ~3000-4500 on weekdays. Screen time 600-720 min/day on weekdays (work + evening). Sleep weekdays 5.5-6.5 hrs. Work stress moderate-high from day one. Played a game with friends at offset 165 — first sport activity noted; described as "a bit out of breath." Explicitly not a conditioned exerciser.',
    },
    {
      offset: 155,
      count: 4,
      text: 'Stress pattern emerging: sustained work pressure, boss pressure, deadlines. At offset 160, user said he had been so tensed lately. At offset 155, work fight: manager pulled user aside in a demoralising review. social_events_negative=high. Neck stiffness first mentioned at offset 170 (casually: neck is killing me from staring at the screen). Pattern: desk stiffness appears periodically, always attributed to screen/posture, not taken seriously.',
    },
    {
      offset: 135,
      count: 4,
      text: 'Work stress sustained. Second sport outing at offset 130: "went for a game with the guys. Was a bit stiff the next morning but okay." Stiffness passed quickly. At offset 125, user stressed again: "deadlines piling up, boss keeps adding more." At offset 120, neck/upper back stiffness mentioned casually: "too many hours hunched over the laptop." User not alarmed — frames it as normal desk consequence. physical_stress rising from low to moderate in data.',
    },
    {
      offset: 110,
      count: 4,
      text: 'Work conflict at offset 110: manager escalated project without informing user, embarrassed him. At offset 103, sleep disrupted by work anxiety: "waking up thinking about work." At offset 96, good family weekend: children, time with wife, recovery feels partial. At offset 90, back stiffness explicitly mentioned: "back feels a bit stiff lately, been sitting too long." User downplays it — "nothing serious." physical_stress data shows moderate now on weekdays.',
    },
    {
      offset: 85,
      count: 4,
      text: 'Oblique physical complaints accumulating: at offset 82, "neck pain from the screen — today it was bad." At offset 80, "back was stiff yesterday after sitting through two back-to-back reviews." Stiffness eased with movement — user sees it as postural only. At offset 75, work fight: manager bypassed user in front of senior leadership — demoralising. stress_ema=high. sleep_quality degrading on weekdays. physical_stress=moderate consistently on weekday data.',
    },
    {
      offset: 65,
      count: 4,
      text: 'Third sport outing at offset 60: "played a game with some colleagues." Sore legs afterward. At offset 52, user said: "I feel physically tired even though I barely moved all day" — does not understand cause. physical_health declining: high (early) → moderate_high (mid) in weekly data. Steps remain 3000-4500 weekdays. sleep_quality poor on weekdays, fair on weekends. Weekday stress_ema: high. Tension in neck and shoulders appearing in chats consistently.',
    },
    {
      offset: 45,
      count: 4,
      text: 'At offset 45, back stiffness again: "on and off this week, comes back by afternoon." At offset 42, work fight: manager moving goalposts, user exhausted. At offset 35, "work stress is really getting to me, cannot switch off in evenings." stress_ema=high consistently weekdays. mental_stress=high. physical_stress=moderate-high on weekday data. Weekly physical_health now moderate_low. sleep_quality poor most weekdays. User has been sedentary for months — no sustained exercise at any point.',
    },
    {
      offset: 30,
      count: 4,
      text: 'Fourth sport outing at offset 30: brief game, no significant complaints noted afterward. At offset 28, good family weekend. At offset 22, user noted: "shoulders and neck are always tight these days — everything feels tense." At offset 20, "upper back and neck really stiff today — worst it has been in a while." User walked after dinner and it helped. Stiffness framed entirely as desk/stress — user has no physical fitness routine to counteract it. physical_stress data: moderate-high recent 30 days.',
    },
    {
      offset: 20,
      count: 4,
      text: 'At offset 18, work conflict: manager singled user out publicly in all-hands meeting. Humiliating. At offset 15, fifth sport outing: "went for a game with Ravi. Back felt a bit stiff afterward but okay by evening." This is the last pre-injury sport day. Important: back stiffness after sport at offset 15 — resolved by evening, user did not escalate concern. Steps on sport days 6000-9000. Weekday steps still ~3000-4500. Exercise on non-sport days: zero for the full 6 months.',
    },
    {
      offset: 10,
      count: 4,
      text: 'Final pre-injury week: at offset 10, "work stress at its worst, running on empty." At offset 7, very low energy at home. At offset 5, manager sending late-night emails, sleep disrupted. stress_ema=high. physical_stress=high (data). sleep_quality=poor. Steps weekdays 3000-4500. The user is at his most deconditioned, most stressed, and most sleep-deprived point in the 6-month window. No exercise, poor sleep, high stress, overweight BMI. physical_health=low in data.',
    },
    {
      offset: 2,
      count: 4,
      text: 'INJURY DAY (offset 2): Vikram played badminton with Ravi in the morning. Acute back pain mid-game. Stopped immediately. Pain severe — "cannot stand straight." Doctor assessed: muscle spasm, prescribed rest for one week. pain_level=significant. physical_stress=high. exercise=none (stopped). mood=low. stress_ema=high. This is an acute-on-chronic event: sudden movement after months of deconditioning + ongoing desk tension. User has not made this connection.',
    },
    {
      offset: 1,
      count: 4,
      text: 'Post-injury state (offsets 1-0): completely immobile, lying flat. Sleep very poor (pain + position). Steps 1500-2500/day. Work paused, emails mounting, anxiety about work piling on top of pain. At offset 1, user anxious about everything. At offset 0, "third day, not getting better, sleep terrible." physical_stress=high. stress_ema=high. pain_level=significant. Screen time elevated (lying-down scrolling). No recovery mechanism in place — no exercise baseline to return to, poor sleep, high stress, no movement.',
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
  // Arc: stress rising, physical_stress creeping, pain none→significant
  // Dashboard: beliefs, rings, trend charts populated from first open
  // ─────────────────────────────────────────────────────────────────────────

  type NodeDist = Record<string, number>;
  type Beliefs  = Record<string, NodeDist>;

  function snap(
    offsetDay: number,
    hour: number,
    beliefs: Beliefs,
    summaryLine: string,
  ): void {
    const d       = daysAgo(offsetDay);
    const dateStr = toDateStr(d);
    const timeStr = `${dateStr}T${String(hour).padStart(2, '0')}:00:00`;
    const nodeConf: Record<string, number> = {};
    for (const node of Object.keys(beliefs)) nodeConf[node] = 0.85;
    db.executeSync(
      `INSERT OR REPLACE INTO inference_snapshots
         (date, snapshot_time, trigger_type, prior_beliefs,
          sensor_snapshot, sensorless_snapshot, dbn_beliefs,
          node_confidences, node_data_sources, summary_line)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      [
        dateStr, timeStr, 'middleaged_seed',
        JSON.stringify({}),
        JSON.stringify({}),
        JSON.stringify({}),
        JSON.stringify(beliefs),
        JSON.stringify(nodeConf),
        JSON.stringify(Object.fromEntries(Object.keys(beliefs).map(n => [n, 'self_report']))),
        summaryLine,
      ],
    );
  }

  // Snapshot 1: ~170 days ago — baseline, early phase
  snap(170, 21, {
    stress_ema:      { low: 0.10, moderate_low: 0.35, moderate_high: 0.40, high: 0.15 },
    mental_stress:   { low: 0.15, moderate: 0.50, high: 0.35 },
    physical_stress: { low: 0.55, moderate: 0.35, high: 0.10 },
    mood:            { low: 0.25, moderate_low: 0.45, moderate_high: 0.25, high: 0.05 },
    sleep_quality:   { poor: 0.25, fair: 0.55, good: 0.20 },
    exercise:        { none: 0.85, light: 0.08, moderate: 0.05, vigorous: 0.02 },
    depression:      { none: 0.60, mild: 0.30, moderate: 0.08, moderate_severe: 0.02 },
    pain_level:      { none: 0.95, some: 0.04, significant: 0.01 },
    negative_affect: { low: 0.30, moderate_low: 0.45, moderate_high: 0.20, high: 0.05 },
    positive_affect: { low: 0.15, moderate_low: 0.40, moderate_high: 0.35, high: 0.10 },
    loneliness:      { low: 0.55, moderate: 0.35, high: 0.10 },
    anxiety:         { none: 0.50, minimal: 0.30, mild: 0.15, moderate: 0.05 },
    fatigue_level:   { none: 0.25, mild: 0.45, moderate: 0.25, severe: 0.05 },
    pain_intensity:  { none: 0.93, mild: 0.05, moderate: 0.02 },
  }, 'Baseline: sedentary office worker, desk stiffness beginning, stress moderate-high, no exercise baseline.');

  // Snapshot 2: ~130 days ago — stress rising, physical_stress beginning to build
  snap(130, 21, {
    stress_ema:      { low: 0.05, moderate_low: 0.20, moderate_high: 0.45, high: 0.30 },
    mental_stress:   { low: 0.08, moderate: 0.42, high: 0.50 },
    physical_stress: { low: 0.38, moderate: 0.45, high: 0.17 },
    mood:            { low: 0.35, moderate_low: 0.45, moderate_high: 0.15, high: 0.05 },
    sleep_quality:   { poor: 0.38, fair: 0.45, good: 0.17 },
    exercise:        { none: 0.88, light: 0.06, moderate: 0.04, vigorous: 0.02 },
    depression:      { none: 0.45, mild: 0.38, moderate: 0.14, moderate_severe: 0.03 },
    pain_level:      { none: 0.94, some: 0.05, significant: 0.01 },
    negative_affect: { low: 0.18, moderate_low: 0.40, moderate_high: 0.32, high: 0.10 },
    positive_affect: { low: 0.25, moderate_low: 0.42, moderate_high: 0.25, high: 0.08 },
    loneliness:      { low: 0.40, moderate: 0.42, high: 0.18 },
    anxiety:         { none: 0.35, minimal: 0.30, mild: 0.25, moderate: 0.10 },
    fatigue_level:   { none: 0.15, mild: 0.40, moderate: 0.35, severe: 0.10 },
    pain_intensity:  { none: 0.91, mild: 0.07, moderate: 0.02 },
  }, 'Stress rising. Back/neck stiffness from desk increasingly noted. Exercise still near-zero. Sleep degrading weekdays.');

  // Snapshot 3: ~95 days ago — chronic stress, physical_stress moderate
  snap(95, 21, {
    stress_ema:      { low: 0.03, moderate_low: 0.12, moderate_high: 0.42, high: 0.43 },
    mental_stress:   { low: 0.05, moderate: 0.30, high: 0.65 },
    physical_stress: { low: 0.20, moderate: 0.55, high: 0.25 },
    mood:            { low: 0.48, moderate_low: 0.38, moderate_high: 0.10, high: 0.04 },
    sleep_quality:   { poor: 0.50, fair: 0.38, good: 0.12 },
    exercise:        { none: 0.90, light: 0.05, moderate: 0.03, vigorous: 0.02 },
    depression:      { none: 0.30, mild: 0.42, moderate: 0.23, moderate_severe: 0.05 },
    pain_level:      { none: 0.93, some: 0.06, significant: 0.01 },
    negative_affect: { low: 0.10, moderate_low: 0.30, moderate_high: 0.40, high: 0.20 },
    positive_affect: { low: 0.38, moderate_low: 0.40, moderate_high: 0.17, high: 0.05 },
    loneliness:      { low: 0.25, moderate: 0.50, high: 0.25 },
    anxiety:         { none: 0.20, minimal: 0.28, mild: 0.35, moderate: 0.17 },
    fatigue_level:   { none: 0.08, mild: 0.32, moderate: 0.42, severe: 0.18 },
    pain_intensity:  { none: 0.90, mild: 0.08, moderate: 0.02 },
  }, 'Stress entrenched. Physical_stress moderate on weekdays consistently. Neck and back stiffness pattern established. No exercise.');

  // Snapshot 4: ~60 days ago — accumulation phase
  snap(60, 21, {
    stress_ema:      { low: 0.02, moderate_low: 0.08, moderate_high: 0.35, high: 0.55 },
    mental_stress:   { low: 0.03, moderate: 0.22, high: 0.75 },
    physical_stress: { low: 0.12, moderate: 0.55, high: 0.33 },
    mood:            { low: 0.58, moderate_low: 0.32, moderate_high: 0.08, high: 0.02 },
    sleep_quality:   { poor: 0.60, fair: 0.30, good: 0.10 },
    exercise:        { none: 0.92, light: 0.05, moderate: 0.02, vigorous: 0.01 },
    depression:      { none: 0.20, mild: 0.42, moderate: 0.30, moderate_severe: 0.08 },
    pain_level:      { none: 0.93, some: 0.06, significant: 0.01 },
    negative_affect: { low: 0.05, moderate_low: 0.22, moderate_high: 0.45, high: 0.28 },
    positive_affect: { low: 0.48, moderate_low: 0.35, moderate_high: 0.13, high: 0.04 },
    loneliness:      { low: 0.15, moderate: 0.50, high: 0.35 },
    anxiety:         { none: 0.10, minimal: 0.20, mild: 0.42, moderate: 0.28 },
    fatigue_level:   { none: 0.05, mild: 0.22, moderate: 0.48, severe: 0.25 },
    pain_intensity:  { none: 0.91, mild: 0.07, moderate: 0.02 },
  }, 'Chronic desk load: physical_stress moderate consistently. User feels physically tired despite inactivity. Sleep poor weekdays.');

  // Snapshot 5: ~30 days ago — stress high, physical_stress high-moderate
  snap(30, 21, {
    stress_ema:      { low: 0.01, moderate_low: 0.05, moderate_high: 0.28, high: 0.66 },
    mental_stress:   { low: 0.02, moderate: 0.15, high: 0.83 },
    physical_stress: { low: 0.08, moderate: 0.45, high: 0.47 },
    mood:            { low: 0.65, moderate_low: 0.25, moderate_high: 0.07, high: 0.03 },
    sleep_quality:   { poor: 0.68, fair: 0.25, good: 0.07 },
    exercise:        { none: 0.93, light: 0.04, moderate: 0.02, vigorous: 0.01 },
    depression:      { none: 0.15, mild: 0.40, moderate: 0.35, moderate_severe: 0.10 },
    pain_level:      { none: 0.92, some: 0.07, significant: 0.01 },
    negative_affect: { low: 0.03, moderate_low: 0.15, moderate_high: 0.45, high: 0.37 },
    positive_affect: { low: 0.55, moderate_low: 0.30, moderate_high: 0.12, high: 0.03 },
    loneliness:      { low: 0.10, moderate: 0.48, high: 0.42 },
    anxiety:         { none: 0.05, minimal: 0.15, mild: 0.40, moderate: 0.40 },
    fatigue_level:   { none: 0.03, mild: 0.17, moderate: 0.48, severe: 0.32 },
    pain_intensity:  { none: 0.90, mild: 0.08, moderate: 0.02 },
  }, 'Shoulders and neck chronically tight. stress_ema=high. physical_stress=moderate-high consistently. Exercise zero for months.');

  // Snapshot 6: ~15 days ago — last sport day (pre-injury stiffness noted)
  snap(15, 20, {
    stress_ema:      { low: 0.05, moderate_low: 0.15, moderate_high: 0.38, high: 0.42 },
    mental_stress:   { low: 0.05, moderate: 0.22, high: 0.73 },
    physical_stress: { low: 0.10, moderate: 0.42, high: 0.48 },
    mood:            { low: 0.42, moderate_low: 0.38, moderate_high: 0.15, high: 0.05 },
    sleep_quality:   { poor: 0.55, fair: 0.35, good: 0.10 },
    exercise:        { none: 0.70, light: 0.08, moderate: 0.05, vigorous: 0.17 },
    depression:      { none: 0.12, mild: 0.40, moderate: 0.38, moderate_severe: 0.10 },
    pain_level:      { none: 0.90, some: 0.08, significant: 0.02 },
    negative_affect: { low: 0.05, moderate_low: 0.20, moderate_high: 0.45, high: 0.30 },
    positive_affect: { low: 0.45, moderate_low: 0.38, moderate_high: 0.13, high: 0.04 },
    loneliness:      { low: 0.20, moderate: 0.48, high: 0.32 },
    anxiety:         { none: 0.08, minimal: 0.18, mild: 0.40, moderate: 0.34 },
    fatigue_level:   { none: 0.03, mild: 0.18, moderate: 0.47, severe: 0.32 },
    pain_intensity:  { none: 0.88, mild: 0.09, moderate: 0.03 },
  }, 'Sport day at offset 15. Back stiff after game — resolved by evening. physical_stress high. Work stress highest in months.');

  // Snapshot 7: ~7 days ago — pre-injury week nadir
  snap(7, 21, {
    stress_ema:      { low: 0.01, moderate_low: 0.04, moderate_high: 0.20, high: 0.75 },
    mental_stress:   { low: 0.01, moderate: 0.10, high: 0.89 },
    physical_stress: { low: 0.05, moderate: 0.30, high: 0.65 },
    mood:            { low: 0.72, moderate_low: 0.20, moderate_high: 0.06, high: 0.02 },
    sleep_quality:   { poor: 0.75, fair: 0.20, good: 0.05 },
    exercise:        { none: 0.95, light: 0.04, moderate: 0.01, vigorous: 0.00 },
    depression:      { none: 0.10, mild: 0.38, moderate: 0.38, moderate_severe: 0.14 },
    pain_level:      { none: 0.93, some: 0.06, significant: 0.01 },
    negative_affect: { low: 0.02, moderate_low: 0.12, moderate_high: 0.42, high: 0.44 },
    positive_affect: { low: 0.62, moderate_low: 0.26, moderate_high: 0.09, high: 0.03 },
    loneliness:      { low: 0.10, moderate: 0.45, high: 0.45 },
    anxiety:         { none: 0.03, minimal: 0.12, mild: 0.38, moderate: 0.47 },
    fatigue_level:   { none: 0.02, mild: 0.12, moderate: 0.45, severe: 0.41 },
    pain_intensity:  { none: 0.91, mild: 0.07, moderate: 0.02 },
  }, 'Pre-injury state: stress_ema=high, physical_stress=high, sleep=poor, exercise=zero. Body at most deconditioned point in 6 months.');

  // Snapshot 8: ~5 days ago — 3 days before injury
  snap(5, 21, {
    stress_ema:      { low: 0.01, moderate_low: 0.04, moderate_high: 0.18, high: 0.77 },
    mental_stress:   { low: 0.01, moderate: 0.08, high: 0.91 },
    physical_stress: { low: 0.04, moderate: 0.28, high: 0.68 },
    mood:            { low: 0.75, moderate_low: 0.18, moderate_high: 0.05, high: 0.02 },
    sleep_quality:   { poor: 0.78, fair: 0.17, good: 0.05 },
    exercise:        { none: 0.96, light: 0.03, moderate: 0.01, vigorous: 0.00 },
    depression:      { none: 0.08, mild: 0.35, moderate: 0.42, moderate_severe: 0.15 },
    pain_level:      { none: 0.92, some: 0.07, significant: 0.01 },
    negative_affect: { low: 0.01, moderate_low: 0.09, moderate_high: 0.40, high: 0.50 },
    positive_affect: { low: 0.68, moderate_low: 0.22, moderate_high: 0.07, high: 0.03 },
    loneliness:      { low: 0.08, moderate: 0.40, high: 0.52 },
    anxiety:         { none: 0.02, minimal: 0.08, mild: 0.35, moderate: 0.55 },
    fatigue_level:   { none: 0.01, mild: 0.09, moderate: 0.43, severe: 0.47 },
    pain_intensity:  { none: 0.90, mild: 0.08, moderate: 0.02 },
  }, 'Manager sending late-night emails. Sleep disrupted. physical_stress=high. Stress at peak. No movement for weeks.');

  // Snapshot 9: injury day (offset 2)
  snap(2, 18, {
    stress_ema:      { low: 0.01, moderate_low: 0.03, moderate_high: 0.15, high: 0.81 },
    mental_stress:   { low: 0.01, moderate: 0.07, high: 0.92 },
    physical_stress: { low: 0.02, moderate: 0.15, high: 0.83 },
    mood:            { low: 0.82, moderate_low: 0.13, moderate_high: 0.04, high: 0.01 },
    sleep_quality:   { poor: 0.82, fair: 0.13, good: 0.05 },
    exercise:        { none: 0.95, light: 0.04, moderate: 0.01, vigorous: 0.00 },
    depression:      { none: 0.07, mild: 0.30, moderate: 0.45, moderate_severe: 0.18 },
    pain_level:      { none: 0.03, some: 0.17, significant: 0.80 },
    negative_affect: { low: 0.01, moderate_low: 0.06, moderate_high: 0.32, high: 0.61 },
    positive_affect: { low: 0.75, moderate_low: 0.18, moderate_high: 0.05, high: 0.02 },
    loneliness:      { low: 0.05, moderate: 0.32, high: 0.63 },
    anxiety:         { none: 0.01, minimal: 0.05, mild: 0.28, moderate: 0.66 },
    fatigue_level:   { none: 0.01, mild: 0.06, moderate: 0.38, severe: 0.55 },
    pain_intensity:  { none: 0.03, mild: 0.12, moderate: 0.45, severe: 0.40 },
  }, 'INJURY DAY: acute back pain during badminton. pain_level=significant. Physical_stress=high. Mood crashed. Doctor: muscle spasm, rest 1 week.');

  // Snapshot 10: yesterday — post-injury, 3rd day (what dashboard reads on first open)
  snap(1, 23, {
    stress_ema:      { low: 0.01, moderate_low: 0.02, moderate_high: 0.12, high: 0.85 },
    mental_stress:   { low: 0.01, moderate: 0.05, high: 0.94 },
    physical_stress: { low: 0.02, moderate: 0.18, high: 0.80 },
    mood:            { low: 0.85, moderate_low: 0.11, moderate_high: 0.03, high: 0.01 },
    sleep_quality:   { poor: 0.85, fair: 0.11, good: 0.04 },
    exercise:        { none: 0.98, light: 0.02, moderate: 0.00, vigorous: 0.00 },
    depression:      { none: 0.05, mild: 0.25, moderate: 0.48, moderate_severe: 0.22 },
    pain_level:      { none: 0.02, some: 0.15, significant: 0.83 },
    negative_affect: { low: 0.01, moderate_low: 0.04, moderate_high: 0.28, high: 0.67 },
    positive_affect: { low: 0.80, moderate_low: 0.14, moderate_high: 0.04, high: 0.02 },
    loneliness:      { low: 0.03, moderate: 0.28, high: 0.69 },
    anxiety:         { none: 0.01, minimal: 0.03, mild: 0.22, moderate: 0.74 },
    fatigue_level:   { none: 0.01, mild: 0.04, moderate: 0.32, severe: 0.63 },
    pain_intensity:  { none: 0.02, mild: 0.10, moderate: 0.42, severe: 0.46 },
  }, 'Day 2 post-injury: pain significant, sleep terrible, steps ~1800, completely immobile. Work piling up. No recovery mechanism visible.');
}
