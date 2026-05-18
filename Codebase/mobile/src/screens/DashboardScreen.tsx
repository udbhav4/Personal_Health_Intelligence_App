import React, { useState, useEffect } from 'react';
import {
  View, Text, ScrollView, StyleSheet, Dimensions,
  Platform, TouchableOpacity, Linking,
} from 'react-native';
import Animated, {
  FadeIn, useSharedValue, withTiming, Easing,
} from 'react-native-reanimated';
import Svg, { Circle, Defs, LinearGradient, Stop, Line, Path, Rect } from 'react-native-svg';
import type { UserProfile } from '../core/AppContext';
import { useAppContext } from '../core/AppContext';
import type { BeliefResult } from '../core/inferenceEngine';
import type { DB } from '@op-engineering/op-sqlite';

const { width: SCREEN_W } = Dimensions.get('window');

// ── Colours ───────────────────────────────────────────────────────────────────

const C = {
  bg:       '#0B0E14',
  card:     '#13171F',
  muted:    '#1A1F27',
  border:   '#22262D',
  fg:       '#FFFFFF',
  mutedFg:  '#7A8494',
  primary:  '#FB923C',
  secondary:'#2D7A7F',
  accent:   '#FDE68A',
};

// ── Belief → 0–100 score mapping ──────────────────────────────────────────────
// Maps dominant DBN states to stress scores.
// mental stress proxy: depression + anxiety nodes
// physical stress proxy: fatigue + pain nodes

const STATE_STRESS_MAP: Record<string, Record<string, number>> = {
  depression:     { none: 5, minimal: 25, mild: 50, moderate: 70, moderately_severe: 85, severe: 98 },
  anxiety:        { none: 5, minimal: 22, mild: 45, moderate: 68, moderately_severe: 83, severe: 96 },
  fatigue_level:  { none: 5, mild: 30, moderate: 58, severe: 82, very_severe: 96 },
  pain_intensity: { none: 5, mild: 28, moderate: 55, severe: 80, very_severe: 96 },
  // Direct DBN stress nodes — used when fatigue/pain nodes absent (real inference output)
  mental_stress:  { low: 8, moderate: 50, high: 88 },
  physical_stress: { low: 8, moderate: 50, high: 88 },
};

function beliefsToScores(beliefsOrNull: BeliefResult | null): { mental: number; physical: number } {
  if (!beliefsOrNull) return { mental: 0, physical: 0 };
  const beliefs = beliefsOrNull;

  function nodeScore(node: string): number | null {
    const dist = beliefs[node];
    if (!dist) return null;
    const map = STATE_STRESS_MAP[node];
    if (!map) return null;
    let score = 0;
    for (const [state, prob] of Object.entries(dist)) {
      score += (map[state] ?? 50) * (prob as number);
    }
    return score;
  }

  const dep  = nodeScore('depression');
  const anx  = nodeScore('anxiety');
  const fat  = nodeScore('fatigue_level');
  const pain = nodeScore('pain_intensity');
  const ms   = nodeScore('mental_stress');
  const ps   = nodeScore('physical_stress');

  const mentalValues   = [dep, anx, ms].filter((v): v is number => v !== null);
  const physicalValues = [fat, pain, ps].filter((v): v is number => v !== null);

  const mental   = mentalValues.length  > 0 ? Math.round(mentalValues.reduce((a, b) => a + b, 0)  / mentalValues.length)  : 0;
  const physical = physicalValues.length > 0 ? Math.round(physicalValues.reduce((a, b) => a + b, 0) / physicalValues.length) : 0;

  return { mental, physical };
}

// ── Data query helpers ────────────────────────────────────────────────────────

interface DayLabel { iso: string; label: string }

function getLast7Days(n: number): DayLabel[] {
  const result: DayLabel[] = [];
  for (let i = n - 1; i >= 0; i--) {
    const d = new Date(Date.now() - i * 86_400_000);
    result.push({ iso: d.toLocaleDateString('sv'), label: d.toLocaleDateString('en', { weekday: 'short' }) });
  }
  return result;
}

interface StressPoint { day: string; mental: number; physical: number }

function getDailyStressScores(db: DB, days: number): StressPoint[] {
  const dates = getLast7Days(days);
  const rows = db.executeSync(
    `SELECT date, dbn_beliefs FROM inference_snapshots WHERE date >= ? ORDER BY date ASC, snapshot_time ASC`,
    [dates[0].iso],
  ).rows as unknown as { date: string; dbn_beliefs: string | null }[];
  const byDate: Record<string, { mental: number; physical: number }[]> = {};
  for (const row of rows) {
    if (!row.dbn_beliefs) continue;
    try {
      const beliefs = JSON.parse(row.dbn_beliefs) as BeliefResult;
      const s = beliefsToScores(beliefs);
      if (!byDate[row.date]) byDate[row.date] = [];
      byDate[row.date].push(s);
    } catch {}
  }
  let lastMental = 0, lastPhysical = 0;
  return dates.map(({ iso, label }) => {
    const arr = byDate[iso];
    if (arr && arr.length > 0) {
      lastMental   = Math.round(arr.reduce((s, x) => s + x.mental,   0) / arr.length);
      lastPhysical = Math.round(arr.reduce((s, x) => s + x.physical, 0) / arr.length);
    }
    return { day: label, mental: lastMental, physical: lastPhysical };
  });
}

function getDailySteps(db: DB, days: number): ChartPoint[] {
  const dates = getLast7Days(days);
  const rows = db.executeSync(
    `SELECT date, raw_value FROM sensor_windows WHERE source_column = 'prev_day_steps' AND date >= ? ORDER BY date ASC`,
    [dates[0].iso],
  ).rows as unknown as { date: string; raw_value: number | null }[];
  const byDate: Record<string, number> = {};
  for (const r of rows) byDate[r.date] = r.raw_value ?? 0;
  return dates.map(({ iso, label }) => ({ day: label, value: byDate[iso] ?? 0 }));
}

function getDailyActiveRatio(db: DB, days: number): ChartPoint[] {
  // raw_value = mean steps per 15-min window; ratio = raw_value / 1500; chart shows %
  const dates = getLast7Days(days);
  const rows = db.executeSync(
    `SELECT date, raw_value FROM sensor_windows WHERE source_column = 'prev_day_active_ratio' AND date >= ? ORDER BY date ASC`,
    [dates[0].iso],
  ).rows as unknown as { date: string; raw_value: number | null }[];
  const byDate: Record<string, number> = {};
  for (const r of rows) byDate[r.date] = r.raw_value != null ? Math.min((r.raw_value / 1500) * 100, 100) : 0;
  return dates.map(({ iso, label }) => ({ day: label, value: +(byDate[iso] ?? 0).toFixed(1) }));
}

function getSleepDuration(db: DB, days: number): ChartPoint[] {
  const dates = getLast7Days(days);
  const rows = db.executeSync(
    `SELECT date, raw_value FROM sensor_windows WHERE node_name = 'sleep_quality' AND source_column = 'sleep_hours' AND date >= ? ORDER BY date ASC`,
    [dates[0].iso],
  ).rows as unknown as { date: string; raw_value: number | null }[];
  const byDate: Record<string, number> = {};
  for (const r of rows) byDate[r.date] = r.raw_value ?? 0;
  return dates.map(({ iso, label }) => ({ day: label, value: +(byDate[iso] ?? 0).toFixed(1) }));
}

function getDailyScreenUsage(db: DB, days: number): ChartPoint[] {
  const dates = getLast7Days(days);
  const rows = db.executeSync(
    `SELECT date, SUM(raw_value) AS total_min FROM sensor_windows WHERE source_column = 'screen_time_window_minutes' AND date >= ? GROUP BY date ORDER BY date ASC`,
    [dates[0].iso],
  ).rows as unknown as { date: string; total_min: number | null }[];
  const byDate: Record<string, number> = {};
  for (const r of rows) byDate[r.date] = +(((r.total_min ?? 0) / 60).toFixed(1));
  return dates.map(({ iso, label }) => ({ day: label, value: byDate[iso] ?? 0 }));
}

function getNighttimeUsage(db: DB, days: number): ChartPoint[] {
  // rows where snapshot_time >= 20:00 OR <= 05:00 (crosses midnight)
  const dates = getLast7Days(days);
  const rows = db.executeSync(
    `SELECT date, SUM(raw_value) AS total_min FROM sensor_windows WHERE source_column = 'screen_time_window_minutes' AND date >= ? AND (TIME(snapshot_time) >= '20:00:00' OR TIME(snapshot_time) <= '05:00:00') GROUP BY date ORDER BY date ASC`,
    [dates[0].iso],
  ).rows as unknown as { date: string; total_min: number | null }[];
  const byDate: Record<string, number> = {};
  for (const r of rows) byDate[r.date] = +(((r.total_min ?? 0) / 60).toFixed(1));
  return dates.map(({ iso, label }) => ({ day: label, value: byDate[iso] ?? 0 }));
}

function hasScreenPermission(db: DB): boolean {
  return (db.executeSync(
    `SELECT 1 FROM sensor_windows WHERE node_name = 'screen_usage' LIMIT 1`,
  ).rows.length) > 0;
}

// ── Circular progress ring ────────────────────────────────────────────────────

function CircularProgress({ value, label }: { value: number; label: string }) {
  const SIZE = 96;
  const SW   = 8;
  const R    = (SIZE - SW) / 2;
  const cx   = SIZE / 2;
  const cy   = SIZE / 2;
  const circ = 2 * Math.PI * R;

  const progress   = useSharedValue(0);
  const [animScore, setAnimScore] = useState(0);

  useEffect(() => {
    progress.value = withTiming(value / 100, { duration: 1400, easing: Easing.out(Easing.cubic) });
    let start: number | null = null;
    const dur = 1400;
    const tick = (ts: number) => {
      if (start === null) start = ts;
      const t = Math.min((ts - start) / dur, 1);
      const eased = 1 - Math.pow(1 - t, 3);
      setAnimScore(Math.round(value * eased));
      if (t < 1) requestAnimationFrame(tick);
    };
    const id = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(id);
  }, [value]);

  const dashOffset = circ - (animScore / 100) * circ;
  const numColor   = value < 40 ? C.secondary : value <= 70 ? C.accent : C.primary;
  const critical   = value > 75;
  const gradId     = `grad-${label.replace(/\s+/g, '')}`;

  return (
    <View style={ringStyles.container}>
      <Animated.View
        style={[
          ringStyles.ringWrapper,
          critical && ringStyles.ringWrapperCritical,
        ]}
      >
        <Svg width={SIZE} height={SIZE} viewBox={`0 0 ${SIZE} ${SIZE}`} style={{ overflow: 'visible' }}>
          <Defs>
            <LinearGradient id={gradId} x1="0%" y1="100%" x2="100%" y2="0%">
              <Stop offset="0%"   stopColor={C.secondary} />
              <Stop offset="40%"  stopColor={value <= 40 ? C.secondary : C.secondary} />
              <Stop offset="60%"  stopColor={value <= 40 ? C.secondary : C.accent} />
              <Stop offset="80%"  stopColor={value <= 70 ? (value <= 40 ? C.secondary : C.accent) : '#F2AB62'} />
              <Stop offset="100%" stopColor={value <= 70 ? (value <= 40 ? C.secondary : C.accent) : C.primary} />
            </LinearGradient>
          </Defs>
          {/* Track */}
          <Circle cx={cx} cy={cy} r={R} fill="none" stroke={C.border} strokeWidth={SW} />
          {/* Arc */}
          <Circle
            cx={cx} cy={cy} r={R}
            fill="none"
            stroke={`url(#${gradId})`}
            strokeWidth={SW}
            strokeLinecap="butt"
            strokeDasharray={circ}
            strokeDashoffset={dashOffset}
            transform={`rotate(-90 ${cx} ${cy})`}
          />
        </Svg>
        <View style={ringStyles.center}>
          <Text style={[ringStyles.scoreText, { color: numColor }]}>{animScore}</Text>
          <Text style={ringStyles.scoreMax}>/ 100</Text>
        </View>
      </Animated.View>
      <Text style={ringStyles.label}>{label}</Text>
    </View>
  );
}

const ringStyles = StyleSheet.create({
  container: {
    alignItems: 'center',
    gap: 8,
  },
  ringWrapper: {
    position: 'relative',
    width: 96,
    height: 96,
  },
  ringWrapperCritical: {
    // pulse indicator — subtle scale animation not added to keep this self-contained
  },
  center: {
    position: 'absolute',
    top: 0, left: 0, right: 0, bottom: 0,
    alignItems: 'center',
    justifyContent: 'center',
  },
  scoreText: {
    fontSize: 22,
    fontWeight: '700',
    fontVariant: ['tabular-nums'],
  },
  scoreMax: {
    fontSize: 10,
    color: C.mutedFg,
  },
  label: {
    fontSize: 11,
    fontWeight: '500',
    color: C.mutedFg,
    textAlign: 'center',
  },
});

// ── Mini area-chart using SVG paths ───────────────────────────────────────────

interface ChartPoint { day: string; value: number; }

const Y_LABEL_W = 30;

function YLabels({ max, min }: { max: number | string; min: number | string }) {
  return (
    <View style={chartStyles.yAxis}>
      <Text style={chartStyles.yLabel}>{max}</Text>
      <Text style={chartStyles.yLabel}>{min}</Text>
    </View>
  );
}

function AreaChartCard({
  title, data, color, gradId,
}: { title: string; data: ChartPoint[]; color: string; gradId: string }) {
  const W = SCREEN_W - 32 - 32 - Y_LABEL_W;
  const H = 90;
  const padV = 8;

  if (!data.length) return null;

  const minVal = Math.min(...data.map(d => d.value));
  const maxVal = Math.max(...data.map(d => d.value));
  const range  = maxVal - minVal || 1;

  const xs = data.map((_, i) => (i / (data.length - 1)) * W);
  const ys = data.map(d => padV + (1 - (d.value - minVal) / range) * (H - 2 * padV));

  const pathD = xs.map((x, i) => (i === 0 ? `M${x},${ys[i]}` : `L${x},${ys[i]}`)).join(' ');
  const fillD = `${pathD} L${xs[xs.length - 1]},${H} L${xs[0]},${H} Z`;

  return (
    <View style={chartStyles.card}>
      <Text style={chartStyles.title}>{title}</Text>
      <View style={chartStyles.chartRow}>
        <YLabels max={maxVal} min={minVal} />
        <Svg width={W} height={H}>
          <Defs>
            <LinearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
              <Stop offset="0%"   stopColor={color} stopOpacity={0.55} />
              <Stop offset="100%" stopColor={color} stopOpacity={0} />
            </LinearGradient>
          </Defs>
          {[0.25, 0.5, 0.75].map((f, i) => (
            <Line key={i} x1={0} y1={padV + f * (H - 2 * padV)} x2={W} y2={padV + f * (H - 2 * padV)}
              stroke="rgba(255,255,255,0.05)" strokeWidth={1} strokeDasharray="2 4" />
          ))}
          <Path d={fillD} fill={`url(#${gradId})`} />
          <Path d={pathD} stroke={color} strokeWidth={1.5} fill="none" />
        </Svg>
      </View>
      <View style={[chartStyles.xAxis, { marginLeft: Y_LABEL_W }]}>
        {data.map(d => (
          <Text key={d.day} style={chartStyles.xLabel}>{d.day}</Text>
        ))}
      </View>
    </View>
  );
}

const chartStyles = StyleSheet.create({
  card: {
    borderRadius: 16,
    padding: 16,
    backgroundColor: 'rgba(255,255,255,0.03)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.10)',
  },
  title: {
    fontSize: 13,
    fontWeight: '500',
    color: C.mutedFg,
    marginBottom: 12,
  },
  chartRow: {
    flexDirection: 'row',
    alignItems: 'stretch',
  },
  yAxis: {
    width: Y_LABEL_W,
    justifyContent: 'space-between',
    alignItems: 'flex-end',
    paddingRight: 4,
  },
  yLabel: {
    fontSize: 9,
    color: C.mutedFg,
  },
  xAxis: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 4,
  },
  xLabel: {
    fontSize: 10,
    color: C.mutedFg,
  },
});

// ── Dual-line chart (Mental vs Physical stress) ───────────────────────────────

interface DualChartPoint { day: string; a: number; b: number }

function DualLineChartCard({ title, data, colorA, colorB, gradIdA, gradIdB }: {
  title: string; data: DualChartPoint[]; colorA: string; colorB: string; gradIdA: string; gradIdB: string;
}) {
  const W = SCREEN_W - 64 - Y_LABEL_W;
  const H = 90;
  const padV = 8;
  if (!data.length) return null;
  const allVals = data.flatMap(d => [d.a, d.b]);
  const minVal  = Math.min(...allVals);
  const maxVal  = Math.max(...allVals);
  const range   = maxVal - minVal || 1;
  const toY     = (v: number) => padV + (1 - (v - minVal) / range) * (H - 2 * padV);
  const xs      = data.map((_, i) => (i / Math.max(data.length - 1, 1)) * W);
  const ysA     = data.map(d => toY(d.a));
  const ysB     = data.map(d => toY(d.b));
  const mkLine  = (ys: number[]) => xs.map((x, i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${ys[i].toFixed(1)}`).join(' ');
  const mkFill  = (ys: number[]) => `${mkLine(ys)} L${xs[xs.length-1].toFixed(1)},${H} L${xs[0].toFixed(1)},${H} Z`;
  return (
    <View style={chartStyles.card}>
      <Text style={chartStyles.title}>{title}</Text>
      <View style={{ flexDirection: 'row', gap: 16, marginBottom: 8 }}>
        <View style={{ flexDirection: 'row', alignItems: 'center', gap: 4 }}>
          <View style={{ width: 10, height: 2, backgroundColor: colorA, borderRadius: 1 }} />
          <Text style={{ fontSize: 10, color: C.mutedFg }}>Mental</Text>
        </View>
        <View style={{ flexDirection: 'row', alignItems: 'center', gap: 4 }}>
          <View style={{ width: 10, height: 2, backgroundColor: colorB, borderRadius: 1 }} />
          <Text style={{ fontSize: 10, color: C.mutedFg }}>Physical</Text>
        </View>
      </View>
      <View style={chartStyles.chartRow}>
        <YLabels max={maxVal} min={minVal} />
        <Svg width={W} height={H}>
          <Defs>
            <LinearGradient id={gradIdA} x1="0" y1="0" x2="0" y2="1">
              <Stop offset="0%"   stopColor={colorA} stopOpacity={0.3} />
              <Stop offset="100%" stopColor={colorA} stopOpacity={0} />
            </LinearGradient>
            <LinearGradient id={gradIdB} x1="0" y1="0" x2="0" y2="1">
              <Stop offset="0%"   stopColor={colorB} stopOpacity={0.3} />
              <Stop offset="100%" stopColor={colorB} stopOpacity={0} />
            </LinearGradient>
          </Defs>
          {[0.25, 0.5, 0.75].map((f, i) => (
            <Line key={i} x1={0} y1={padV + f * (H - 2 * padV)} x2={W} y2={padV + f * (H - 2 * padV)}
              stroke="rgba(255,255,255,0.05)" strokeWidth={1} strokeDasharray="2 4" />
          ))}
          <Path d={mkFill(ysA)} fill={`url(#${gradIdA})`} />
          <Path d={mkLine(ysA)} stroke={colorA} strokeWidth={1.5} fill="none" />
          <Path d={mkFill(ysB)} fill={`url(#${gradIdB})`} />
          <Path d={mkLine(ysB)} stroke={colorB} strokeWidth={1.5} fill="none" />
        </Svg>
      </View>
      <View style={[chartStyles.xAxis, { marginLeft: Y_LABEL_W }]}>
        {data.map(d => <Text key={d.day} style={chartStyles.xLabel}>{d.day}</Text>)}
      </View>
    </View>
  );
}

// ── Bar chart ─────────────────────────────────────────────────────────────────

function BarChartCard({ title, data, color }: { title: string; data: ChartPoint[]; color: string }) {
  const W = SCREEN_W - 64 - Y_LABEL_W;
  const H = 80;
  const N = data.length || 1;
  const maxVal = Math.max(...data.map(d => d.value), 1);
  const slotW  = W / N;
  const barW   = slotW * 0.55;
  return (
    <View style={chartStyles.card}>
      <Text style={chartStyles.title}>{title}</Text>
      <View style={chartStyles.chartRow}>
        <YLabels max={maxVal} min={0} />
        <Svg width={W} height={H}>
          {data.map((d, i) => {
            const bH = Math.max(2, (d.value / maxVal) * H);
            return (
              <Rect key={d.day} x={i * slotW + (slotW - barW) / 2} y={H - bH}
                width={barW} height={bH} fill={color} opacity={d.value > 0 ? 1 : 0.2} rx={3} />
            );
          })}
        </Svg>
      </View>
      <View style={[chartStyles.xAxis, { marginLeft: Y_LABEL_W }]}>
        {data.map(d => <Text key={d.day} style={chartStyles.xLabel}>{d.day}</Text>)}
      </View>
    </View>
  );
}

// ── Screen permission gate ────────────────────────────────────────────────────

function ScreenPermissionGate({ hasPermission, children }: { hasPermission: boolean; children: React.ReactNode }) {
  if (Platform.OS === 'ios') {
    return (
      <View style={gateStyles.iosCard}>
        <Text style={gateStyles.iosText}>Screen usage metrics are not available on iPhone.</Text>
      </View>
    );
  }
  const openSettings = async () => {
    try { await Linking.sendIntent('android.settings.USAGE_ACCESS_SETTINGS'); }
    catch { Linking.openSettings(); }
  };
  return (
    <View>
      <View pointerEvents={hasPermission ? 'auto' : 'none'} style={{ opacity: hasPermission ? 1 : 0.15 }}>
        {children}
      </View>
      {!hasPermission && (
        <View style={[StyleSheet.absoluteFillObject, gateStyles.overlay]}>
          <Text style={gateStyles.overlayTitle}>Usage Access Needed</Text>
          <Text style={gateStyles.overlayBody}>
            Settings → Apps → Special App Access → Usage Access → Gliimr
          </Text>
          <TouchableOpacity style={gateStyles.settingsBtn} onPress={openSettings} activeOpacity={0.85}>
            <Text style={gateStyles.settingsBtnText}>Open Settings</Text>
          </TouchableOpacity>
        </View>
      )}
    </View>
  );
}

const gateStyles = StyleSheet.create({
  iosCard: {
    borderRadius: 16, padding: 20,
    backgroundColor: 'rgba(255,255,255,0.03)',
    borderWidth: 1, borderColor: 'rgba(255,255,255,0.10)',
    alignItems: 'center',
  },
  iosText:  { fontSize: 13, color: '#7A8494', textAlign: 'center' },
  overlay: {
    borderRadius: 16,
    backgroundColor: 'rgba(11,14,20,0.88)',
    alignItems: 'center', justifyContent: 'center',
    padding: 24, gap: 10,
  },
  overlayTitle: { fontSize: 15, fontWeight: '600', color: '#FFFFFF' },
  overlayBody:  { fontSize: 12, color: '#7A8494', textAlign: 'center', lineHeight: 18 },
  settingsBtn:  { marginTop: 4, backgroundColor: '#FB923C', borderRadius: 12, paddingHorizontal: 20, paddingVertical: 10 },
  settingsBtnText: { fontSize: 13, fontWeight: '600', color: '#000' },
});

// ── Main DashboardScreen ──────────────────────────────────────────────────────

export default function DashboardScreen({ profile }: { profile: UserProfile }) {
  const { db, beliefs } = useAppContext();
  const scores = beliefsToScores(beliefs);

  const [stressData,      setStressData]      = useState<StressPoint[]>([]);
  const [stepsData,       setStepsData]       = useState<ChartPoint[]>([]);
  const [activeRatioData, setActiveRatioData] = useState<ChartPoint[]>([]);
  const [sleepData,       setSleepData]       = useState<ChartPoint[]>([]);
  const [screenData,      setScreenData]      = useState<ChartPoint[]>([]);
  const [nighttimeData,   setNighttimeData]   = useState<ChartPoint[]>([]);
  const [screenPerm,      setScreenPerm]      = useState(false);

  useEffect(() => {
    if (!db) return;
    try { setStressData(getDailyStressScores(db, 7)); }      catch {}
    try { setStepsData(getDailySteps(db, 7)); }              catch {}
    try { setActiveRatioData(getDailyActiveRatio(db, 7)); }  catch {}
    try { setSleepData(getSleepDuration(db, 7)); }           catch {}
    try { setScreenData(getDailyScreenUsage(db, 7)); }       catch {}
    try { setNighttimeData(getNighttimeUsage(db, 7)); }      catch {}
    try { setScreenPerm(hasScreenPermission(db)); }          catch {}
  }, [db]);

  const hour = new Date().getHours();
  const greeting = hour < 12 ? 'Good morning' : hour < 17 ? 'Good afternoon' : 'Good evening';

  return (
    <ScrollView style={dashStyles.container} contentContainerStyle={dashStyles.content} showsVerticalScrollIndicator={false}>
      <Animated.View entering={FadeIn.duration(400)} style={dashStyles.header}>
        <Text style={dashStyles.headerSub}>{greeting},</Text>
        <Text style={dashStyles.headerName}>{profile.name || 'there'}</Text>
      </Animated.View>

      <View style={dashStyles.divider} />

      {/* Current stress snapshot — rings */}
      <Animated.View entering={FadeIn.duration(400).delay(100)} style={dashStyles.card}>
        <Text style={dashStyles.sectionLabel}>DAILY STRESS OVERVIEW</Text>
        <View style={dashStyles.ringsRow}>
          <CircularProgress value={scores.mental}   label="Mental Stress" />
          <View style={dashStyles.ringDivider} />
          <CircularProgress value={scores.physical} label="Physical Stress" />
        </View>
      </Animated.View>

      <View style={dashStyles.divider} />

      {/* 7-day charts */}
      <Animated.View entering={FadeIn.duration(400).delay(200)} style={{ gap: 12 }}>
        <DualLineChartCard
          title="Stress Trend (7 days)"
          data={stressData.map(d => ({ day: d.day, a: d.mental, b: d.physical }))}
          colorA={C.secondary}
          colorB={C.primary}
          gradIdA="stressMentalGrad"
          gradIdB="stressPhysicalGrad"
        />
        <BarChartCard title="Daily Steps" data={stepsData} color={C.primary} />
        <AreaChartCard title="Active Ratio (%)" data={activeRatioData} color={C.primary} gradId="activeRatioGrad" />

        <ScreenPermissionGate hasPermission={screenPerm}>
          <View style={{ gap: 12 }}>
            <BarChartCard title="Sleep Duration (hrs)" data={sleepData}    color={C.secondary} />
            <BarChartCard title="Screen Usage (hrs/day)" data={screenData} color={C.primary}   />
            <BarChartCard title="Phone in Dark (hrs)"  data={nighttimeData} color={C.mutedFg}  />
          </View>
        </ScreenPermissionGate>
      </Animated.View>
    </ScrollView>
  );
}

const dashStyles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: C.bg,
  },
  content: {
    paddingHorizontal: 16,
    paddingTop: 24,
    paddingBottom: 16,
    gap: 16,
  },
  header: {},
  headerSub: {
    fontSize: 13,
    color: C.mutedFg,
  },
  headerName: {
    fontSize: 24,
    fontWeight: '700',
    color: C.fg,
  },
  divider: {
    height: 1,
    backgroundColor: 'rgba(251,146,60,0.25)',
  },
  card: {
    borderRadius: 16,
    padding: 20,
    backgroundColor: 'rgba(255,255,255,0.03)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.10)',
  },
  sectionLabel: {
    fontSize: 11,
    color: C.mutedFg,
    letterSpacing: 1.5,
    textTransform: 'uppercase',
    marginBottom: 16,
  },
  ringsRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
  },
  ringDivider: {
    width: 1,
    height: 96,
    backgroundColor: C.border,
  },
});
