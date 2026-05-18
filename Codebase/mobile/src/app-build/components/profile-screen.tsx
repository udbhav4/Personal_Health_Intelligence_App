'use client'
import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Pencil, Check, X, User, Calendar, Scale, Ruler, UserCircle2, ChevronDown } from 'lucide-react'
import type { UserProfile } from '@/hooks/use-store'

interface Props {
  profile: UserProfile
  onSave: (data: UserProfile) => void
  isActive?: boolean
}

const FIELDS: { key: keyof UserProfile; label: string; type: string }[] = [
  { key: 'name',      label: 'Name',      type: 'text' },
  { key: 'age',       label: 'Age',       type: 'number' },
  { key: 'sex',       label: 'Sex',       type: 'select' },
  { key: 'birthdate', label: 'Birthdate', type: 'date-wheel' },
  { key: 'weight',    label: 'Weight',    type: 'text' },
  { key: 'height',    label: 'Height',    type: 'text' },
]

const ICONS: Record<keyof UserProfile, React.ReactNode> = {
  name:      <User        size={14} style={{ color: '#FB923C' }} />,
  age:       <UserCircle2 size={14} style={{ color: '#FB923C' }} />,
  sex:       <UserCircle2 size={14} style={{ color: '#FB923C' }} />,
  birthdate: <Calendar    size={14} style={{ color: '#FB923C' }} />,
  weight:    <Scale       size={14} style={{ color: '#FB923C' }} />,
  height:    <Ruler       size={14} style={{ color: '#FB923C' }} />,
}

// ─── Inline wheel date picker (same as onboarding) ───────────────────────────
const MONTHS_W = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
const DAYS_W   = Array.from({ length: 31 }, (_, i) => String(i + 1).padStart(2, '0'))
const YEARS_W  = Array.from({ length: 100 }, (_, i) => String(new Date().getFullYear() - i))

function WheelCol({ items, value, onChange }: { items: string[]; value: string; onChange:(v:string)=>void }) {
  const ITEM_H = 34
  const ref = useRef<HTMLDivElement>(null)
  const init = useRef(false)
  const idx = items.indexOf(value)

  useEffect(() => {
    if (!init.current) { init.current = true; ref.current?.scrollTo({ top: Math.max(0,idx)*ITEM_H }) }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const onScroll = () => {
    if (!ref.current) return
    const i = Math.round(ref.current.scrollTop / ITEM_H)
    const c = Math.max(0, Math.min(items.length - 1, i))
    if (items[c] !== value) onChange(items[c])
  }

  return (
    <div className="relative flex-1 overflow-hidden" style={{ height: ITEM_H * 3 }}>
      <div className="absolute inset-x-0 top-0 h-6 pointer-events-none z-10" style={{ background: 'linear-gradient(to bottom,#13171F,transparent)' }} />
      <div className="absolute inset-x-0 bottom-0 h-6 pointer-events-none z-10" style={{ background: 'linear-gradient(to top,#13171F,transparent)' }} />
      <div className="absolute inset-x-0 z-10 rounded-md pointer-events-none" style={{ top: ITEM_H, height: ITEM_H, background: 'rgba(251,146,60,0.1)', border: '1px solid rgba(251,146,60,0.3)' }} />
      <div ref={ref} onScroll={onScroll} className="h-full overflow-y-scroll" style={{ scrollSnapType:'y mandatory', scrollbarWidth:'none', WebkitOverflowScrolling:'touch' }}>
        <div style={{ height: ITEM_H }} />
        {items.map(item => (
          <div key={item} style={{ height: ITEM_H, scrollSnapAlign:'center', display:'flex', alignItems:'center', justifyContent:'center', fontSize:13, color: item===value?'#FB923C':'#7A8494', fontWeight: item===value?600:400, cursor:'pointer' }}
            onClick={() => { onChange(item); ref.current?.scrollTo({ top: items.indexOf(item)*ITEM_H }) }}
          >{item}</div>
        ))}
        <div style={{ height: ITEM_H }} />
      </div>
    </div>
  )
}

function DateWheelInline({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const parts = (value || '01-Jan-2000').split('-')
  const [day, setDay]     = useState(parts[0] || '01')
  const [month, setMonth] = useState(parts[1] || 'Jan')
  const [year, setYear]   = useState(parts[2] || '2000')

  const emit = (d: string, m: string, y: string) => {
    setDay(d); setMonth(m); setYear(y); onChange(`${d}-${m}-${y}`)
  }

  return (
    <div className="w-full flex gap-0.5 items-stretch" style={{ height: 34*3, background:'#1A1F27', border:'1.5px solid #FB923C', borderRadius:'0.75rem', overflow:'hidden', padding:'0 8px' }}>
      <WheelCol items={DAYS_W}   value={day}   onChange={d => emit(d, month, year)} />
      <div className="w-px self-stretch my-2" style={{ background:'rgba(255,255,255,0.1)' }} />
      <WheelCol items={MONTHS_W} value={month} onChange={m => emit(day, m, year)} />
      <div className="w-px self-stretch my-2" style={{ background:'rgba(255,255,255,0.1)' }} />
      <WheelCol items={YEARS_W}  value={year}  onChange={y => emit(day, month, y)} />
    </div>
  )
}

// ─── Main ────────────────────────────────────────────────────────────────────
export default function ProfileScreen({ profile, onSave, isActive }: Props) {
  const [editing, setEditing] = useState<keyof UserProfile | null>(null)
  const [draft, setDraft] = useState('')

  const isDraftValid = draft.trim().length > 0

  const startEdit = (key: keyof UserProfile) => {
    // Block opening another field while one is dirty-open
    if (editing !== null) return
    setDraft(profile[key] || '')
    setEditing(key)
  }

  const confirmEdit = () => {
    if (!editing || !isDraftValid) return
    onSave({ ...profile, [editing]: draft })
    setEditing(null)
  }

  const cancelEdit = () => {
    setEditing(null)
    setDraft('')
  }

  // Revert unsaved edit when user swipes away (isActive goes false)
  useEffect(() => {
    if (!isActive && editing !== null) {
      setEditing(null)
      setDraft('')
    }
  }, [isActive]) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="flex flex-col h-full overflow-y-auto px-4 pt-5 pb-4 gap-3">
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
        <p className="text-muted-foreground text-xs">Your</p>
        <h2 className="text-xl font-bold text-foreground">Profile</h2>
      </motion.div>

      <div className="h-px" style={{ background: 'rgba(251,146,60,0.2)' }} />

      {/* Avatar */}
      <div className="flex justify-center py-1">
        <div
          className="w-16 h-16 rounded-full flex items-center justify-center text-2xl font-bold flex-shrink-0"
          style={{ background: 'linear-gradient(135deg, #FB923C, #2D7A7F)', color: '#fff' }}
        >
          {(profile.name || '?')[0].toUpperCase()}
        </div>
      </div>

      {/* Fields */}
      <div className="flex flex-col gap-2">
        {FIELDS.map(({ key, label, type }) => {
          const isEditing = editing === key
          const isBlocked = editing !== null && !isEditing
          const currentVal = profile[key] || ''

          return (
            <motion.div
              key={key}
              layout
              className="rounded-xl px-3 py-2.5"
              style={{ background: '#1A1F27', border: isEditing ? '1px solid rgba(251,146,60,0.5)' : '1px solid transparent', opacity: isBlocked ? 0.45 : 1 }}
            >
              {/* Label row */}
              <div className="flex items-center justify-between min-w-0">
                <div className="flex items-center gap-1.5 min-w-0 mr-2">
                  <span className="flex-shrink-0">{ICONS[key]}</span>
                  <span className="text-[10px] text-muted-foreground uppercase tracking-wide truncate">{label}</span>
                </div>
                {/* Right action buttons — always inline, never overflow */}
                <div className="flex items-center gap-1 flex-shrink-0">
                  {isEditing ? (
                    <>
                      {/* Cancel */}
                      <motion.button whileTap={{ scale: 0.9 }} onClick={cancelEdit}
                        className="w-7 h-7 rounded-lg flex items-center justify-center"
                        style={{ background: '#22262D' }}
                      >
                        <X size={12} className="text-muted-foreground" />
                      </motion.button>
                      {/* Confirm — disabled + dimmed if draft empty */}
                      <motion.button
                        whileTap={isDraftValid ? { scale: 0.9 } : {}}
                        onClick={isDraftValid ? confirmEdit : undefined}
                        className="w-7 h-7 rounded-lg flex items-center justify-center transition-all"
                        style={{ background: isDraftValid ? '#FB923C' : 'rgba(251,146,60,0.25)', cursor: isDraftValid ? 'pointer' : 'not-allowed' }}
                        aria-disabled={!isDraftValid}
                      >
                        <Check size={12} style={{ color: isDraftValid ? '#000' : 'rgba(0,0,0,0.35)' }} strokeWidth={3} />
                      </motion.button>
                    </>
                  ) : (
                    <motion.button
                      whileTap={!isBlocked ? { scale: 0.9 } : {}}
                      onClick={() => !isBlocked && startEdit(key)}
                      className="w-7 h-7 rounded-lg flex items-center justify-center"
                      style={{ background: 'transparent', cursor: isBlocked ? 'not-allowed' : 'pointer' }}
                    >
                      <Pencil size={12} style={{ color: '#FB923C' }} />
                    </motion.button>
                  )}
                </div>
              </div>

              {/* Value / Edit row */}
              <AnimatePresence mode="wait">
                {isEditing ? (
                  <motion.div
                    key="edit"
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.18 }}
                    className="mt-2"
                  >
                    {type === 'select' ? (
                      <div className="relative">
                        <select
                          value={draft}
                          onChange={e => setDraft(e.target.value)}
                          className="w-full bg-input border border-border rounded-lg px-3 py-2 text-foreground text-sm appearance-none focus:outline-none"
                          style={{ borderColor: '#FB923C' }}
                        >
                          <option value="Male">Male</option>
                          <option value="Female">Female</option>
                        </select>
                        <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground pointer-events-none" size={13} />
                      </div>
                    ) : type === 'date-wheel' ? (
                      <DateWheelInline value={draft} onChange={setDraft} />
                    ) : (
                      /* Expandable input: up to 2 lines, then scroll */
                      <textarea
                        value={draft}
                        onChange={e => setDraft(e.target.value)}
                        rows={1}
                        autoFocus
                        className="w-full bg-input border border-border rounded-lg px-3 py-2 text-foreground text-sm focus:outline-none resize-none leading-relaxed"
                        style={{
                          borderColor: '#FB923C',
                          caretColor: '#FB923C',
                          maxHeight: '2.9em', // ~2 lines
                          overflowY: 'auto',
                        }}
                        onInput={e => {
                          const el = e.currentTarget
                          el.style.height = 'auto'
                          el.style.height = Math.min(el.scrollHeight, 46) + 'px'
                        }}
                      />
                    )}
                  </motion.div>
                ) : (
                  <motion.p
                    key="view"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="mt-1 text-sm font-medium leading-snug"
                    style={{ color: currentVal ? '#fff' : '#7A8494' }}
                  >
                    {currentVal || 'Not set'}
                  </motion.p>
                )}
              </AnimatePresence>
            </motion.div>
          )
        })}
      </div>
    </div>
  )
}
