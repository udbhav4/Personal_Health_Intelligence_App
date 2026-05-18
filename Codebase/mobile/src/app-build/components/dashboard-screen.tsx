'use client'
import {
  ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid,
} from 'recharts'
import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import type { UserProfile } from '@/hooks/use-store'

interface Props { profile: UserProfile }

const screenData = [
  { day: 'Mon', value: 4.2 }, { day: 'Tue', value: 5.8 }, { day: 'Wed', value: 3.1 },
  { day: 'Thu', value: 6.4 }, { day: 'Fri', value: 7.2 }, { day: 'Sat', value: 5.0 },
  { day: 'Sun', value: 4.6 },
]
const activityData = [
  { day: 'Mon', value: 6200 }, { day: 'Tue', value: 4800 }, { day: 'Wed', value: 8100 },
  { day: 'Thu', value: 7400 }, { day: 'Fri', value: 5600 }, { day: 'Sat', value: 9200 },
  { day: 'Sun', value: 7800 },
]

// ─── Severity-aware ring ──────────────────────────────────────────────────────
// Uses a conic-gradient div with a radial mask so the colour is non-repeating.
// The arc always starts at eucalyptus (top, -90°).
// As score rises the head colour progresses: eucalyptus → warm sand → apricot.
// The transition thresholds:
//   0–40: full eucalyptus
//   40–70: eucalyptus → warm sand blend
//   70–100: eucalyptus → warm sand → apricot blend
// The dark track covers the remainder of the circle after the score arc.
function buildConicGradient(score: number): string {
  const s = Math.max(0, Math.min(100, score))
  const pct = s // score % maps directly to degrees (0–360)

  if (s <= 40) {
    // Eucalyptus only arc, then dark track
    return `conic-gradient(from -90deg, #2D7A7F 0%, #2D7A7F ${pct}%, #22262D ${pct}%, #22262D 100%)`
  }
  if (s <= 70) {
    // Eucalyptus → Warm Sand, then dark track
    // The mid-point of the sand reach is at the score boundary
    return `conic-gradient(from -90deg, #2D7A7F 0%, #FDE68A ${pct}%, #22262D ${pct}%, #22262D 100%)`
  }
  // Eucalyptus → Warm Sand (at ~70%) → Apricot (at score%), then dark track
  return `conic-gradient(from -90deg, #2D7A7F 0%, #FDE68A 70%, #FB923C ${pct}%, #22262D ${pct}%, #22262D 100%)`
}

function CircularProgress({ value, label }: { value: number; label: string }) {
  const SIZE = 96
  const SW   = 8      // stroke width
  const R    = (SIZE - SW) / 2   // radius to centre of stroke
  const cx   = SIZE / 2
  const cy   = SIZE / 2
  const circ = 2 * Math.PI * R
  const critical = value > 75

  // Score number colour follows severity
  const numColor = value < 40 ? '#2D7A7F' : value <= 70 ? '#FDE68A' : '#FB923C'
  // Head colour (tip of arc)
  const headColor = value < 40 ? '#2D7A7F' : value <= 70 ? '#FDE68A' : '#FB923C'

  const [animScore, setAnimScore] = useState(0)
  useEffect(() => {
    const start = performance.now()
    const dur = 1400
    const tick = (now: number) => {
      const t = Math.min((now - start) / dur, 1)
      const eased = 1 - Math.pow(1 - t, 3)
      setAnimScore(Math.round(value * eased))
      if (t < 1) requestAnimationFrame(tick)
    }
    const id = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(id)
  }, [value])

  // Compute head-dot position: arc goes clockwise from top (-90deg)
  const headAngleDeg = -90 + (animScore / 100) * 360
  const headAngleRad = (headAngleDeg * Math.PI) / 180
  const headX = cx + R * Math.cos(headAngleRad)
  const headY = cy + R * Math.sin(headAngleRad)

  const dashOffset = circ - (animScore / 100) * circ
  const glowColor = value < 40 ? 'rgba(45,122,127,0.45)' : value <= 70 ? 'rgba(253,230,138,0.45)' : 'rgba(251,146,60,0.45)'
  const gradId = `rg-${label.replace(/\s+/g, '')}`

  return (
    <div className="flex flex-col items-center gap-2">
      <motion.div
        className="relative"
        style={{ width: SIZE, height: SIZE }}
        animate={critical ? { scale: [1, 1.03, 1] } : { scale: 1 }}
        transition={critical ? { duration: 2.4, repeat: Infinity, ease: 'easeInOut' } : {}}
      >
        <svg
          width={SIZE} height={SIZE}
          viewBox={`0 0 ${SIZE} ${SIZE}`}
          style={{ overflow: 'visible' }}
        >
          <defs>
            {/* 3-stop gradient along the arc direction */}
            <linearGradient id={gradId} x1="0%" y1="0%" x2="100%" y2="100%" gradientUnits="userSpaceOnUse">
              <stop offset="0%"   stopColor="#2D7A7F" />
              <stop offset="60%"  stopColor={value <= 40 ? '#2D7A7F' : '#FDE68A'} />
              <stop offset="100%" stopColor={value <= 70 ? (value <= 40 ? '#2D7A7F' : '#FDE68A') : '#FB923C'} />
            </linearGradient>
          </defs>

          {/* Track */}
          <circle cx={cx} cy={cy} r={R} fill="none" stroke="#22262D" strokeWidth={SW} />

          {/* Coloured arc — starts from top (rotate -90deg) */}
          <motion.circle
            cx={cx} cy={cy} r={R}
            fill="none"
            stroke={`url(#${gradId})`}
            strokeWidth={SW}
            strokeLinecap="butt"
            strokeDasharray={circ}
            initial={{ strokeDashoffset: circ }}
            animate={{ strokeDashoffset: dashOffset }}
            transition={{ duration: 1.4, ease: [0.33, 1, 0.68, 1] }}
            transform={`rotate(-90 ${cx} ${cy})`}
          />

          {/* Round soft cap at head position */}
          <motion.circle
            cx={headX} cy={headY}
            r={SW / 2}
            fill={headColor}
            initial={{ opacity: 0 }}
            animate={{ opacity: animScore > 2 ? 1 : 0, cx: headX, cy: headY }}
            transition={{ duration: 0.1 }}
            style={{ filter: `drop-shadow(0 0 3px ${glowColor})` }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-2xl font-bold tabular-nums" style={{ color: numColor }}>{value}</span>
          <span className="text-[10px] text-muted-foreground">/ 100</span>
        </div>
      </motion.div>
      <span className="text-xs font-medium text-muted-foreground text-center">{label}</span>
    </div>
  )
}

function ChartCard({ title, data, color, gradientId }: {
  title: string; data: typeof screenData; color: string; gradientId: string
}) {
  return (
    <div
      className="rounded-2xl p-4"
      style={{
        background: 'rgba(255,255,255,0.03)',
        backdropFilter: 'blur(12px)',
        WebkitBackdropFilter: 'blur(12px)',
        border: '1px solid rgba(255,255,255,0.10)',
      }}
    >
      <p className="text-sm font-medium text-muted-foreground mb-3">{title}</p>
      <ResponsiveContainer width="100%" height={110}>
        <AreaChart data={data} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
          <defs>
            <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="15%" stopColor={color} stopOpacity={0.55} />
              <stop offset="100%" stopColor={color} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="2 4" stroke="rgba(255,255,255,0.05)" vertical={false} />
          <XAxis dataKey="day" tick={{ fill: '#7A8494', fontSize: 11 }} axisLine={false} tickLine={false} />
          <YAxis tick={{ fill: '#7A8494', fontSize: 11 }} axisLine={false} tickLine={false} />
          <Tooltip
            contentStyle={{ background: '#13171F', border: '1px solid #22262D', borderRadius: 12 }}
            labelStyle={{ color: '#7A8494', fontSize: 12 }}
            itemStyle={{ color: '#fff', fontSize: 12 }}
          />
          <Area
            type="monotone"
            dataKey="value"
            stroke={color}
            strokeWidth={1.5}
            fill={`url(#${gradientId})`}
            dot={false}
            activeDot={{ r: 4, fill: color }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}

export default function DashboardScreen({ profile }: Props) {
  const hour = new Date().getHours()
  const greeting = hour < 12 ? 'Good morning' : hour < 17 ? 'Good afternoon' : 'Good evening'

  return (
    <div
      className="flex flex-col h-full overflow-y-auto px-4 pt-6 pb-4 gap-4"
      style={{
        background: 'radial-gradient(circle at 50% 20%, rgba(45,122,127,0.12) 0%, rgba(11,14,20,1) 80%)',
      }}
    >
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: -12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
        <p className="text-muted-foreground text-sm">{greeting},</p>
        <h2 className="text-2xl font-bold text-foreground text-balance">{profile.name || 'there'}</h2>
      </motion.div>

      <div className="h-px" style={{ background: 'rgba(251,146,60,0.25)' }} />

      {/* Stress rings */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.1 }}
        className="rounded-2xl p-5"
        style={{
          background: 'rgba(255,255,255,0.03)',
          backdropFilter: 'blur(12px)',
          WebkitBackdropFilter: 'blur(12px)',
          border: '1px solid rgba(255,255,255,0.10)',
        }}
      >
        <p className="text-xs text-muted-foreground uppercase tracking-widest mb-4">Daily Stress Overview</p>
        <div className="flex justify-around">
          <CircularProgress value={68} label="Mental Stress" />
          <div className="w-px bg-border" />
          <CircularProgress value={78} label="Physical Stress" />
        </div>
      </motion.div>

      <div className="h-px" style={{ background: 'rgba(251,146,60,0.25)' }} />

      {/* Charts */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.2 }}
        className="flex flex-col gap-3"
      >
        <ChartCard
          title="Screen Usage (hrs/day)"
          data={screenData}
          color="#2D7A7F"
          gradientId="screenGrad"
        />
        <ChartCard
          title="Physical Activity (steps)"
          data={activityData}
          color="#FB923C"
          gradientId="activityGrad"
        />
      </motion.div>
    </div>
  )
}
