/**
 * core/reportAgent.ts — Plan-and-Execute Gemma pipeline for the Doctor Report.
 *
 * Architecture (calls: amber + hypothesis loop + planner + 5 executors):
 *   1. Amber bullets — rewrites key patterns into plain English
 *   2. Hypothesis loop — ReAct loop probing 180-day causal chain (up to 6 steps)
 *   3. Planner — outputs a JSON array of 5 section plans
 *   4-8. Executors — each writes one section as plain prose (max ~120 words)
 *
 * Robustness rules:
 *   - Planner output is parsed defensively (regex array extract + JSON.parse).
 *     Any failure or missing-required-section falls back to FALLBACK_PLAN.
 *   - After all executors complete, a validator scans the merged output and
 *     patches in: data-citation note, forgot-keyword note, or limitations text
 *     if any are missing.
 *
 * All prompts are written for Gemma 4-2B-IT Q4_K_M
 * Format instructions appear at the END of each prompt where the model is most
 * likely to follow them. JSON output is fenced by example so the model copies
 * the shape rather than narrating.
 */

import type { DB } from '@op-engineering/op-sqlite';

import { runAgentCompletion, runReportHypothesisLoop } from './agent';
import type { HiddenPattern, ReportDataObject } from './reportDataCollector';
import {
  compressMemories,
  compressDbnTrajectory,
  compressSensorTrends,
  compressPatterns,
} from './reportDataCollector';

// ── Types ─────────────────────────────────────────────────────────────────────

export interface SectionNarrative {
  section_title: string;
  content:       string;
}

interface SectionPlan {
  section_title: string;
  tier:          '1' | '2' | 'both';
  instruction:   string;
}

// ── Fallback plan (used on any planner failure) ───────────────────────────────

const FALLBACK_PLAN: SectionPlan[] = [
  { section_title: 'Related but Possibly Missed',  tier: '1',    instruction: 'Surface complaints the patient may not have mentioned and silent elevated DBN nodes from Tier 1. Use phrases like "not mentioned", "may have forgotten", or "not raised".' },
  { section_title: 'Passive Data Patterns',    tier: '1',    instruction: 'Describe temporal correlations and anomaly weeks. Cite specific values (dates, averages, step counts).' },
  { section_title: 'How This Connects',        tier: '1',    instruction: 'Link Tier 1 patterns non-obviously to the stated complaint.' },
  { section_title: 'Questions for Your Doctor', tier: 'both', instruction: 'List 3–5 specific questions grounded in detected patterns, not generic health questions.' },
  { section_title: 'Data Limitations',         tier: 'both', instruction: 'State what this passive data cannot confirm or diagnose.' },
];

const REQUIRED_TITLES = FALLBACK_PLAN.map(p => p.section_title);

// ── Prompts: PLANNER ─────────────────────────────────────────────────────────

const PLANNER_SYSTEM =
`You plan a 5-section doctor report.
The doctor already knows the complaint.
The report surfaces what the patient may not have mentioned or may have forgotten.
You output one JSON array. Nothing else.`;

function plannerUser(
  symptom:             string,
  tier1Descriptions:   string[],
  tier2Descriptions:   string[],
  sensorSummary:       string,
  dbnTrajectory:       Record<string, string>,
  anomalyWeeks:        string,
  hypothesisText:      string,
): string {
  const t1 = tier1Descriptions.length > 0
    ? tier1Descriptions.map((d, i) => `${i + 1}. ${d}`).join('\n')
    : '(none)';
  const t2 = tier2Descriptions.length > 0
    ? tier2Descriptions.map((d, i) => `${i + 1}. ${d}`).join('\n')
    : '(none)';
  const dbn = Object.keys(dbnTrajectory).length > 0
    ? Object.entries(dbnTrajectory).slice(0, 8).map(([k, v]) => `${k}: ${v}`).join('\n')
    : '(no trajectory)';

  const hypothesisBlock = hypothesisText
    ? `\nCAUSAL HYPOTHESIS (from tool-verified data):\n${hypothesisText}\n`
    : '';

  return `COMPLAINT: ${symptom}

TIER 1 (linked to complaint):
${t1}

TIER 2 (other patterns):
${t2}

SENSORS: ${sensorSummary || '(none)'}

DBN TRAJECTORY:
${dbn}

ANOMALY WEEKS: ${anomalyWeeks || '(none)'}
${hypothesisBlock}
Write 5 plans. One per section. Each instruction is one short sentence and names a real item from the data above.${hypothesisText ? ' For the "How This Connects" section, the instruction MUST reference the CAUSAL HYPOTHESIS above.' : ''}

Copy this shape exactly. Replace each "..." with a real instruction. Keep all 5 titles and tiers as shown.

Output JSON only. No other text. No markdown.

[
  {"section_title": "Related but Possibly Missed", "tier": "1", "instruction": "Point out Tier 1 items the patient did not raise, like '..."},
  {"section_title": "Passive Data Patterns", "tier": "1", "instruction": "Describe the sensor or DBN change, like '..."},
  {"section_title": "How This Connects", "tier": "1", "instruction": "${hypothesisText ? "Use the causal hypothesis: trace how <first_node> drove <mid_node> and led to the complaint." : "Link Tier 1 item '...' to the complaint."}"},
  {"section_title": "Questions for Your Doctor", "tier": "both", "instruction": "Write 3 questions about '...' and '..."},
  {"section_title": "Data Limitations", "tier": "both", "instruction": "State what the passive data cannot confirm."}
]`;
}

// ── Prompts: EXECUTOR ────────────────────────────────────────────────────────

const EXECUTOR_SYSTEM =
`You write one short section of a doctor report.
Write plain prose. No headings. No bullets. No JSON.
Use at least one number, week, or node name from the data.
Do not repeat the complaint as the topic.
Do not give diagnoses.`;

function executorUser(
  sectionTitle:        string,
  instruction:         string,
  tier:                string,
  tier1Descriptions:   string[],
  tier2Descriptions:   string[],
  memoryExcerpts:      string[],
  sensorSummary:       string,
  dbnTrajectory:       Record<string, string>,
  hypothesisText:      string,
): string {
  const t1Block = tier1Descriptions.length > 0
    ? tier1Descriptions.map((d, i) => `${i + 1}. ${d}`).join('\n')
    : '(none)';
  const t2Block = tier2Descriptions.length > 0
    ? tier2Descriptions.map((d, i) => `${i + 1}. ${d}`).join('\n')
    : '(none)';
  const mem = memoryExcerpts.length > 0
    ? memoryExcerpts.slice(0, 3).map((m, i) => `${i + 1}. ${m}`).join('\n')
    : '(no relevant memories)';
  const dbn = Object.keys(dbnTrajectory).length > 0
    ? Object.entries(dbnTrajectory).slice(0, 6).map(([k, v]) => `${k}: ${v}`).join('\n')
    : '(no trajectory)';

  // Tier filtering: only show the data the executor is allowed to use.
  const patternBlock = tier === '1'
    ? `PATTERNS:\n${t1Block}`
    : tier === '2'
      ? `PATTERNS:\n${t2Block}`
      : `TIER 1 PATTERNS:\n${t1Block}\n\nTIER 2 PATTERNS:\n${t2Block}`;

  // Per-section anchor: a concrete one-line example + a forced keyword cue.
  let titleHint = '';
  if (sectionTitle === 'Related but Possibly Missed') {
    titleHint =
`This section must use phrases like "not mentioned", "may have forgotten", "not raised", or "silent".
Example: "The patient may not have mentioned rising screen time, yet evening use grew from 9.1h to 10.3h. Stress was silent in recent chats but the model shows it stayed high for 3 weeks."`;
  } else if (sectionTitle === 'Passive Data Patterns') {
    titleHint =
`Cite at least one number or week from the data above.
Example: "Steps dropped from 4200 to 2800 over weeks 2025-43 to 2025-44. The DBN shows depression moving from mild to moderate in the same window."`;
  } else if (sectionTitle === 'How This Connects') {
    if (hypothesisText) {
      titleHint =
`The CAUSAL HYPOTHESIS above was verified by tool data. Your job is to translate it into plain language for the patient.
Start by naming the first node in the chain, state what it did over the tracked period, then trace how that led to the complaint.
Example: "Sleep quality declined steadily over 12 weeks. During the same period, the model shows stress_ema rising and recovery_score falling. This chain — poor sleep compounding stress and reducing recovery — is the most likely path to the fatigue the patient reported."`;
    } else {
      titleHint =
`Link one pattern above to the complaint in a non-obvious way.
Example: "Reduced steps and rising screen time together often precede the kind of fatigue the patient reports, even when sleep itself looks normal."`;
    }
  } else if (sectionTitle === 'Questions for Your Doctor') {
    titleHint =
`Write 3 to 5 short questions in one paragraph. Number them inline as 1) 2) 3). Each question names a real item from the data.
Example: "1) Could the drop in steps from 4200 to 2800 be tied to the fatigue? 2) Should the moderate stress for 3 weeks be screened further? 3) Is the rise in screen time worth tracking?"`;
  } else if (sectionTitle === 'Data Limitations') {
    titleHint =
`This section must use the word "limitations".
Example: "Limitations: this report uses passive sensor data and a probabilistic model. It cannot confirm a diagnosis. Sleep tracking was missing for some weeks."`;
  }

  const hypothesisBlock = hypothesisText
    ? `\nCAUSAL HYPOTHESIS:\n${hypothesisText}\n`
    : '';

  return `SECTION: ${sectionTitle}
TASK: ${instruction}

${patternBlock}
${hypothesisBlock}
MEMORY:
${mem}

SENSORS: ${sensorSummary || '(none)'}

DBN:
${dbn}

${titleHint}

Now write the section. Plain prose. Max 120 words. Do not write the section title. Do not use quotes around the text. Start the prose now.`;
}

// ── Plan parsing / validation ────────────────────────────────────────────────

function extractPlanJson(text: string): SectionPlan[] | null {
  // Find the first balanced JSON array in the text — tolerant of leading/trailing prose.
  // Try non-greedy first (stops at the first closing ']') so trailing prose containing
  // another ']' doesn't get swallowed. If that parse fails, fall back to greedy
  // (which extends to the LAST ']'), since the planner may legitimately emit a
  // nested array whose outer ']' is the correct terminator.
  const candidates: string[] = [];
  const nonGreedy = text.match(/\[[\s\S]+?\]/);
  if (nonGreedy) candidates.push(nonGreedy[0]);
  const greedy = text.match(/\[[\s\S]+\]/);
  if (greedy && greedy[0] !== candidates[0]) candidates.push(greedy[0]);
  if (candidates.length === 0) return null;

  for (const candidate of candidates) {
    try {
      const parsed = JSON.parse(candidate) as unknown;
      if (!Array.isArray(parsed)) continue;
      const result: SectionPlan[] = [];
      for (const item of parsed) {
        if (!item || typeof item !== 'object') continue;
        const obj = item as Record<string, unknown>;
        const title = typeof obj.section_title === 'string' ? obj.section_title : null;
        const tier  = obj.tier === '1' || obj.tier === '2' || obj.tier === 'both' ? obj.tier : null;
        const instr = typeof obj.instruction === 'string' ? obj.instruction : null;
        if (!title || !tier || !instr) continue;
        result.push({ section_title: title, tier, instruction: instr });
      }
      if (result.length > 0) return result;
    } catch {
      // Try next candidate.
    }
  }
  return null;
}

/**
 * Make sure every required section title appears in the plan, in canonical
 * order. Missing entries are patched in from FALLBACK_PLAN.
 */
function ensureRequiredSections(plan: SectionPlan[]): SectionPlan[] {
  const byTitle = new Map<string, SectionPlan>();
  for (const p of plan) byTitle.set(p.section_title, p);
  return REQUIRED_TITLES.map(title => {
    const found = byTitle.get(title);
    if (found) return found;
    return FALLBACK_PLAN.find(f => f.section_title === title)!;
  });
}

// ── Mandatory output validator ───────────────────────────────────────────────

const FORGOT_KEYWORDS = ['may have forgotten', 'not mentioned', 'not raised', 'silent'];

function contentHasNumber(text: string): boolean {
  return /\d/.test(text);
}

function contentMentionsForgot(text: string): boolean {
  const lower = text.toLowerCase();
  return FORGOT_KEYWORDS.some(k => lower.includes(k));
}

function contentMentionsLimitation(text: string): boolean {
  return /limitation/i.test(text);
}

const FALLBACK_LIMITATIONS_TEXT =
  'Note on limitations: this report draws on passive sensor data, self-reported ' +
  'information, and a probabilistic model. It cannot confirm or diagnose any ' +
  'medical condition and should be reviewed by a clinician.';

function appendToContent(content: string, suffix: string): string {
  const sep = content.trim().endsWith('.') ? ' ' : '. ';
  return content.trim() + sep + suffix;
}

function applyValidators(
  sections: SectionNarrative[],
  data:     ReportDataObject,
): SectionNarrative[] {
  if (sections.length === 0) return sections;
  const merged = sections.map(s => s.content).join('\n');

  // 1) If Tier 1 patterns exist but no section content has a number,
  //    append a citation note to section 1.
  if (data.tier1Patterns.length > 0 && !contentHasNumber(merged)) {
    sections[0] = {
      section_title: sections[0].section_title,
      content:       appendToContent(
        sections[0].content,
        '[Note: specific metrics are available in the supporting tables below.]',
      ),
    };
  }

  // 2) If no section uses a "forgot"-class keyword, append a pointer to section 1.
  const mergedAfter = sections.map(s => s.content).join('\n');
  if (!contentMentionsForgot(mergedAfter)) {
    sections[0] = {
      section_title: sections[0].section_title,
      content:       appendToContent(
        sections[0].content,
        "Note: See 'Related but Possibly Missed' section above for health signals not raised in recent conversation.",
      ),
    };
  }

  // 3) If no section mentions limitations, append a fallback limitations
  //    statement to the LAST section.
  const mergedFinal = sections.map(s => s.content).join('\n');
  if (!contentMentionsLimitation(mergedFinal)) {
    const last = sections.length - 1;
    sections[last] = {
      section_title: sections[last].section_title,
      content:       appendToContent(sections[last].content, FALLBACK_LIMITATIONS_TEXT),
    };
  }

  return sections;
}

// ── Public ───────────────────────────────────────────────────────────────────

// ── Amber bullets: Gemma rewrites pattern descriptions into plain English ─────

const AMBER_SYSTEM =
`You rewrite health observations into plain English sentences for a doctor.
One sentence per item. Simple words. No technical terms or metric names.
Start each line with its number.`;

function amberUser(patterns: HiddenPattern[]): string {
  const items = patterns
    .slice(0, 6)
    .map((p, i) => `${i + 1}. ${p.description}`)
    .join('\n');
  return `Rewrite each observation below as one clear English sentence:\n${items}\n\nOutput only the numbered sentences. Nothing else.`;
}

async function generateAmberBullets(patterns: HiddenPattern[]): Promise<string[]> {
  if (patterns.length === 0) return [];
  try {
    const text = await runAgentCompletion(AMBER_SYSTEM, amberUser(patterns), 220);
    const lines = text
      .split('\n')
      .map(l => l.trim())
      .filter(l => /^\d+[\.\)]/.test(l));
    const bullets = lines
      .map(l => l.replace(/^\d+[\.\)]\s*/, '').trim())
      .filter(s => s.length > 10);
    if (bullets.length > 0) return bullets;
  } catch { /* fall through to fallback */ }
  return patterns.map(p => p.description);
}

// ── Public ───────────────────────────────────────────────────────────────────

export interface ReportNarrativeResult {
  narrative:    SectionNarrative[];
  amberBullets: string[];
}

/**
 * Run the Gemma pipeline: amber bullets → hypothesis loop → plan → 5 section executors.
 * Returns both the five narrative sections and the plain-English amber box bullets.
 */
export async function generateReportNarrative(
  db:         DB,
  data:       ReportDataObject,
  onProgress: (msg: string) => void,
): Promise<ReportNarrativeResult> {

  // ── Compress all inputs ───────────────────────────────────────────────────
  const [memoryExcerpts] = await Promise.all([
    compressMemories(db, data.symptom),
  ]);

  // ── Amber bullets (call 1 of 7) ───────────────────────────────────────────
  onProgress('Summarising key findings...');
  const amberPatterns = [
    ...data.tier1Patterns.filter(p => p.type === 'silent_node' || p.type === 'forgotten_complaint' || p.type === 'sustained_trend'),
    ...data.tier1Patterns.filter(p => p.type === 'anomaly_week'),
  ].slice(0, 6);
  const amberBullets = await generateAmberBullets(amberPatterns);

  const tier1Descriptions = compressPatterns(data.tier1Patterns, 10, 60);
  const tier2Descriptions = compressPatterns(data.tier2Patterns, 6,  50);
  const sensorSummary     = compressSensorTrends(data.sensorTrends);
  const dbnTrajectory     = compressDbnTrajectory(data.dbnTrajectory);
  const anomalyWeeksStr   = data.anomalyWeeks.join(', ');

  // ── Hypothesis phase — Gemma probes causal chain ─────────────────────────
  onProgress('Discovering causal patterns...');
  const allPatterns = [...data.tier1Patterns, ...data.tier2Patterns];
  const patternList = allPatterns.slice(0, 8)
    .map(p => `${p.nodes.join(', ')}: ${p.description}`)
    .join('\n');
  let hypothesisText = '';
  try {
    hypothesisText = await runReportHypothesisLoop(
      db, data.symptom, patternList, sensorSummary,
    );
  } catch { /* graceful: empty hypothesis is fine */ }

  // ── Planner call ──────────────────────────────────────────────────────────
  onProgress('Writing report plan...');
  let plan: SectionPlan[] = [];
  try {
    const plannerText = await runAgentCompletion(
      PLANNER_SYSTEM,
      plannerUser(data.symptom, tier1Descriptions, tier2Descriptions,
                  sensorSummary, dbnTrajectory, anomalyWeeksStr, hypothesisText),
      350,
    );
    const extracted = extractPlanJson(plannerText);
    if (extracted) plan = extracted;
  } catch {
    plan = [];
  }
  if (plan.length === 0) plan = [...FALLBACK_PLAN];
  plan = ensureRequiredSections(plan);

  // ── Executor calls (one per section) ──────────────────────────────────────
  const sections: SectionNarrative[] = [];
  for (let i = 0; i < plan.length; i++) {
    const p = plan[i];
    onProgress(`Writing ${p.section_title}... (${i + 1} of ${plan.length})`);
    let content = '';
    try {
      content = await runAgentCompletion(
        EXECUTOR_SYSTEM,
        executorUser(p.section_title, p.instruction, p.tier,
                     tier1Descriptions, tier2Descriptions, memoryExcerpts,
                     sensorSummary, dbnTrajectory, hypothesisText),
        200,
      );
    } catch (e) {
      content = `Section could not be generated: ${e instanceof Error ? e.message : String(e)}`;
    }
    sections.push({
      section_title: p.section_title,
      content:       content.trim() || '(section was empty)',
    });
  }

  // ── Mandatory validators ──────────────────────────────────────────────────
  const narrative = applyValidators(sections, data);
  return { narrative, amberBullets };
}
