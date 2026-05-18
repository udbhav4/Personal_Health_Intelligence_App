/**
 * core/reportHtmlBuilder.ts — Static HTML generator for the Doctor Report PDF.
 *
 * Renders a self-contained HTML document (all CSS inline, no flexbox, no
 * external resources) suitable for `expo-print` on Android.  Android WebView's
 * print CSS support is roughly equivalent to 2016 desktop browsers — tables
 * with explicit borders are used everywhere instead of flex/grid layouts.
 *
 * Section order (most actionable first):
 *   1. Header + disclaimer
 *   2. Chief complaint (verbatim)
 *   3. What the Patient Forgot
 *   4. Passive Data Patterns
 *   5. Gemma narrative sections
 *   6. Sensor trend table
 *   7. Self-reported nodes table
 *   8. DBN belief summary table
 *   9. Tier 2 additional context
 *  10. Data limitations (always last)
 */

import type {
  ReportDataObject, SensorlessNodeSummary,
} from './reportDataCollector';
import { compressSensorTrends } from './reportDataCollector';

// ── Types ─────────────────────────────────────────────────────────────────────

export interface SectionNarrative {
  section_title: string;
  content:       string;
}

// ── Human-readable node labels ────────────────────────────────────────────────

const NODE_LABELS: Record<string, string> = {
  mental_stress: 'Mental Stress Estimate', physical_stress: 'Physical Stress Estimate',
  pain_level: 'Pain Level', mood: 'Mood', sleep_quality: 'Sleep Quality',
  sleep_disturbances: 'Sleep Disturbances', depression: 'Depression Indicator',
  stress_ema: 'Stress (Self-Reported)', productivity: 'Productivity',
  exercise: 'Exercise Level', loneliness: 'Loneliness',
  negative_affect: 'Negative Affect', positive_affect: 'Positive Affect',
  bmi: 'BMI', age: 'Age Group', sex: 'Sex', neuroticism: 'Neuroticism',
  extraversion: 'Extraversion', general_health: 'General Health',
  alcohol_use: 'Alcohol Use', smoking: 'Smoking',
  social_events_positive: 'Positive Social Events', social_events_negative: 'Negative Social Events',
  stress_helplessness: 'Stress Helplessness', stress_self_efficacy: 'Stress Self-Efficacy',
  chronic_condition: 'Chronic Condition', diabetes_status: 'Diabetes Status',
  education_level: 'Education Level', marital_status: 'Marital Status',
};

function labelFor(node: string): string { return NODE_LABELS[node] ?? node.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()); }

function formatState(state: string): string {
  return state.replace(/_/g, '-').replace(/\b\w/g, c => c.toUpperCase());
}

// ── HTML escaping ─────────────────────────────────────────────────────────────

function esc(s: unknown): string {
  if (s == null) return '';
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function formatDate(iso: string | null | undefined): string {
  if (!iso) return '';
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  return d.toISOString().slice(0, 10);
}

// ── Style constants ───────────────────────────────────────────────────────────

const FONT_STACK = "-apple-system, Helvetica, Arial, sans-serif";

const STYLE_BLOCK = `
<style>
  @media print { body { -webkit-print-color-adjust: exact; print-color-adjust: exact; } }
  body { font-family: ${FONT_STACK}; background: #ffffff; color: #000000;
         margin: 0; padding: 0; }
  .wrap { max-width: 720px; margin: 0 auto; padding: 40px; }
  h1 { font-size: 22px; margin: 0 0 4px 0; color: #000000; }
  h2 { font-size: 16px; margin: 24px 0 8px 0; color: #000000;
       border-bottom: 1px solid #cccccc; padding-bottom: 4px; }
  h3 { font-size: 14px; margin: 16px 0 6px 0; color: #000000; }
  p  { font-size: 13px; line-height: 1.5; margin: 0 0 10px 0; color: #000000; }
  ul { margin: 6px 0 10px 22px; padding: 0; }
  li { font-size: 13px; line-height: 1.5; margin: 0 0 4px 0; color: #000000; }
  table { border: 1px solid #cccccc; border-collapse: collapse; width: 100%;
          margin: 8px 0 16px 0; font-size: 12px; }
  th, td { border: 1px solid #cccccc; border-collapse: collapse;
           padding: 6px 8px; text-align: left; vertical-align: top; color: #000000; }
  th { background: #f0f0f0; font-weight: 700; }
  .disclaimer { font-size: 12px; font-weight: 700; color: #000000;
                background: #fafafa; border: 1px solid #cccccc;
                padding: 10px 14px; margin: 12px 0 18px 0; }
  .box-grey { background: #f5f5f5; border-left: 4px solid #cccccc;
              padding: 10px 14px; margin: 8px 0 16px 0; }
  .box-amber { background: #fff8e1; border-left: 4px solid #f59e0b;
               padding: 10px 14px; margin: 8px 0 16px 0; }
  .box-blue { background: #eff6ff; border-left: 4px solid #3b82f6;
              padding: 10px 14px; margin: 8px 0 16px 0; }
  .complaint-quote { font-style: italic; font-size: 14px; color: #000000;
                     margin: 0; }
  .complaint-label { font-size: 11px; text-transform: uppercase;
                     letter-spacing: 0.6px; color: #555555; margin: 0 0 6px 0; }
  .note { font-size: 11px; color: #555555; font-style: italic; margin: 6px 0 0 0; }
  .meta { font-size: 11px; color: #555555; margin: 4px 0 0 0; }
  .tier2 { opacity: 0.85; border-left: 3px solid #cccccc; padding-left: 16px;
           margin: 16px 0; }
  .anomaly-row { background-color: #fff8e1; }
  .star { color: #d97706; font-weight: 700; }
</style>
`;

// ── Sections ──────────────────────────────────────────────────────────────────

function sectionHeader(data: ReportDataObject): string {
  const name = esc((data.profile.name || 'Patient').trim());
  const today = formatDate(new Date().toISOString());
  return `
    <h1>Doctor Report</h1>
    <p class="meta">Patient: ${name} &nbsp;&middot;&nbsp; Date: ${esc(today)}</p>
    <div class="disclaimer">
      This report surfaces passive health patterns the patient may not have mentioned.
      It is not a clinical assessment.
    </div>
  `;
}

function sectionChiefComplaint(data: ReportDataObject): string {
  return `
    <h2>Chief Complaint</h2>
    <div class="box-grey">
      <p class="complaint-label">Patient's stated complaint</p>
      <p class="complaint-quote">"${esc(data.symptom.trim())}"</p>
    </div>
  `;
}

function sectionForgotten(data: ReportDataObject, amberBullets?: string[]): string {
  let body: string;
  if (amberBullets && amberBullets.length > 0) {
    const items = amberBullets.map(b => `<li>${esc(b)}</li>`).join('');
    body = `<ul>${items}</ul>`;
  } else {
    const relevant = data.tier1Patterns.filter(
      p => p.type === 'forgotten_complaint' || p.type === 'silent_node',
    );
    if (relevant.length === 0) {
      body = `<p>No additional health signals detected in the 180-day window.</p>`;
    } else {
      const items = relevant.map(p => `<li>${esc(p.description)}</li>`).join('');
      body = `<ul>${items}</ul>`;
    }
  }
  return `
    <h2>Related but Possibly Missed</h2>
    <div class="box-amber">
      <p class="note">These health signals were found in your data but were not mentioned in recent conversation.</p>
      ${body}
    </div>
  `;
}

function sectionPassivePatterns(data: ReportDataObject): string {
  const relevant = data.tier1Patterns.filter(
    p => p.type === 'temporal_correlation' ||
         p.type === 'anomaly_week' ||
         p.type === 'sustained_trend' ||
         p.type === 'contradictory_state',
  );
  const sensorSummary = compressSensorTrends(data.sensorTrends);

  let body = '';
  if (relevant.length > 0) {
    const items = relevant.map(p => `<li>${esc(p.description)}</li>`).join('');
    body += `<ul>${items}</ul>`;
  }
  body += `<p><strong>Passive sensor summary:</strong> ${esc(sensorSummary)}</p>`;

  if (data.anomalyWeeks.length > 0) {
    body += `<p><strong>Anomaly weeks (≥3 worst-tertile metrics):</strong> ${esc(data.anomalyWeeks.join(', '))}</p>`;
  }

  return `
    <h2>Passive Data Patterns</h2>
    <div class="box-blue">
      ${body}
    </div>
  `;
}

function sectionNarrative(narrative: SectionNarrative[]): string {
  if (narrative.length === 0) return '';
  return narrative.map(s => `
    <h2>${esc(s.section_title)}</h2>
    <p>${esc(s.content).replace(/\n+/g, '</p><p>')}</p>
  `).join('');
}

function sectionSensorTrendTable(data: ReportDataObject): string {
  if (data.sensorTrends.length === 0) {
    return `
      <h2>Weekly Sensor Trends</h2>
      <p>No sensor data available.</p>
    `;
  }

  // Pivot per week
  const byWeek: Map<string, Map<string, number>> = new Map();
  for (const t of data.sensorTrends) {
    if (!byWeek.has(t.week)) byWeek.set(t.week, new Map());
    byWeek.get(t.week)!.set(t.source_column, t.weekly_avg);
  }
  const weeks = Array.from(byWeek.keys()).sort();
  const anomalySet = new Set(data.anomalyWeeks);

  const rows = weeks.slice(-26).map(wk => {
    const cells = byWeek.get(wk)!;
    const steps   = cells.get('hourly_steps');
    const screen  = cells.get('screen_time');
    const active  = cells.get('active_ratio');
    const rowCls  = anomalySet.has(wk) ? ' class="anomaly-row"' : '';
    const fmt = (v: number | undefined, digits = 0) =>
      v == null || !isFinite(v) ? '—' : v.toFixed(digits);
    return `<tr${rowCls}>
      <td>${esc(wk)}</td>
      <td>${esc(fmt(steps, 0))}</td>
      <td>${esc(fmt(screen, 1))}</td>
      <td>${esc(fmt(active, 2))}</td>
    </tr>`;
  }).join('');

  return `
    <h2>Weekly Sensor Trends (last ${Math.min(26, weeks.length)} weeks)</h2>
    <table>
      <thead><tr>
        <th>Week</th><th>Avg Steps</th><th>Avg Screen (min)</th><th>Active Ratio</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <p class="note">Highlighted rows indicate anomaly weeks (≥3 health metrics in the worst tertile).</p>
  `;
}

function sectionSelfReported(data: ReportDataObject): string {
  const rows: SensorlessNodeSummary[] = data.sensorlessSummaries
    .filter(s => s.confidence != null && s.confidence > 0.3)
    .sort((a, b) => labelFor(a.node_name).localeCompare(labelFor(b.node_name)));

  if (rows.length === 0) {
    return `
      <h2>Self-Reported Health Areas</h2>
      <p>No self-reported entries with confidence above 0.3 in the 180-day window.</p>
    `;
  }

  const silentSet = new Set(data.silentNodes);
  const body = rows.map(r => {
    const star = silentSet.has(r.node_name) ? '<span class="star">&#9733;</span> ' : '';
    return `<tr>
      <td>${star}${esc(labelFor(r.node_name))}</td>
      <td>${esc(formatState(r.node_value))}</td>
      <td>${esc(formatDate(r.report_date))}</td>
    </tr>`;
  }).join('');

  return `
    <h2>Self-Reported Health Areas</h2>
    <table>
      <thead><tr>
        <th>Health Area</th><th>Status</th><th>Date</th>
      </tr></thead>
      <tbody>${body}</tbody>
    </table>
    <p class="note">&#9733; marks areas not mentioned in recent conversation but detected as elevated.</p>
  `;
}

function sectionDbnBeliefs(data: ReportDataObject): string {
  if (!data.beliefs) {
    return `
      <h2>Model Belief Summary</h2>
      <p>No model belief data available.</p>
    `;
  }

  interface BeliefRow { node: string; state: string; prob: number; }
  const rows: BeliefRow[] = [];
  for (const [node, dist] of Object.entries(data.beliefs)) {
    let bestState = ''; let bestProb = -1;
    for (const [s, p] of Object.entries(dist)) {
      if ((p as number) > bestProb) { bestProb = p as number; bestState = s; }
    }
    if (bestProb > 0.5) rows.push({ node, state: bestState, prob: bestProb });
  }
  if (rows.length === 0) {
    return `
      <h2>Model Belief Summary</h2>
      <p>No belief estimates above 50% confidence at this time.</p>
    `;
  }
  rows.sort((a, b) => labelFor(a.node).localeCompare(labelFor(b.node)));

  const body = rows.map(r => `<tr>
    <td>${esc(labelFor(r.node))}</td>
    <td>${esc(formatState(r.state))}</td>
  </tr>`).join('');

  return `
    <h2>Model Belief Summary</h2>
    <table>
      <thead><tr>
        <th>Health Area</th><th>Estimated Status</th>
      </tr></thead>
      <tbody>${body}</tbody>
    </table>
  `;
}

function sectionTier2(data: ReportDataObject): string {
  if (data.tier2Patterns.length === 0) return '';
  const items = data.tier2Patterns.map(p => `<li>${esc(p.description)}</li>`).join('');
  return `
    <div class="tier2">
      <h2>Additional Health Context</h2>
      <p class="note">Notable health patterns not directly related to your stated complaint. Included for completeness.</p>
      <ul>${items}</ul>
    </div>
  `;
}

const LIMITATIONS_TEXT =
  `This report is generated from passive sensor data, self-reported information, and a ` +
  `probabilistic health model. It cannot diagnose medical conditions. Sensor data accuracy ` +
  `depends on device usage patterns. Self-reported information relies on user recall. DBN ` +
  `belief estimates carry uncertainty — confidence values are displayed alongside each ` +
  `estimate. This report should be reviewed by a qualified healthcare professional alongside ` +
  `a full clinical assessment.`;

function sectionLimitations(): string {
  return `
    <h2>Data Limitations</h2>
    <p>${esc(LIMITATIONS_TEXT)}</p>
  `;
}

// ── Public ────────────────────────────────────────────────────────────────────

/**
 * Build the complete HTML document for the Doctor Report PDF.
 * Combines hard-coded structural sections with the Gemma-generated narrative.
 */
export function buildReportHtml(
  narrative:    SectionNarrative[],
  data:         ReportDataObject,
  amberBullets?: string[],
): string {
  const body =
    sectionHeader(data) +
    sectionChiefComplaint(data) +
    sectionForgotten(data, amberBullets) +
    sectionPassivePatterns(data) +
    sectionNarrative(narrative) +
    sectionSensorTrendTable(data) +
    sectionSelfReported(data) +
    sectionDbnBeliefs(data) +
    sectionTier2(data) +
    sectionLimitations();

  return `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>Doctor Report</title>
${STYLE_BLOCK}
</head>
<body>
<div class="wrap">
${body}
</div>
</body>
</html>`;
}

// Re-export helpers used by other modules / tests
export { NODE_LABELS };
