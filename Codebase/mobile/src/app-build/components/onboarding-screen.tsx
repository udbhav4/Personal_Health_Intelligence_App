'use client'
import { useState, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Check, ChevronDown, ChevronRight } from 'lucide-react'
import type { UserProfile } from '@/hooks/use-store'

interface Props {
  onComplete: (data: UserProfile) => void
}

const STEPS = ['name', 'age', 'sex', 'birthdate', 'weight', 'height'] as const
type Step = typeof STEPS[number]

const STEP_CONFIG: Record<Step, { label: string; placeholder: string; type: string }> = {
  name:      { label: 'What should I call you?',   placeholder: 'Your name',   type: 'text' },
  age:       { label: 'How old are you?',           placeholder: 'e.g. 24',    type: 'number' },
  sex:       { label: 'What is your sex?',          placeholder: 'Select...',  type: 'select' },
  birthdate: { label: 'When were you born?',        placeholder: 'YYYY-MM-DD', type: 'date' },
  weight:    { label: 'What is your weight?',       placeholder: 'e.g. 68 kg', type: 'text' },
  height:    { label: 'What is your height?',       placeholder: 'e.g. 175 cm',type: 'text' },
}

function isValid(step: Step, val: string) {
  if (!val.trim()) return false
  if (step === 'age') return Number(val) > 0 && Number(val) < 130
  return true
}

// ─── Fast Wheel Date Picker ───────────────────────────────────────────────────
const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
const DAYS   = Array.from({ length: 31 }, (_, i) => String(i + 1).padStart(2, '0'))
const YEARS  = Array.from({ length: 100 }, (_, i) => String(new Date().getFullYear() - i))

function WheelColumn({ items, value, onChange }: {
  items: string[]; value: string; onChange: (v: string) => void
}) {
  const ITEM_H = 40
  const ref = useRef<HTMLDivElement>(null)
  const idx = items.indexOf(value)
  const scrollTo = (i: number) => {
    ref.current?.scrollTo({ top: i * ITEM_H, behavior: 'auto' })
  }
  const handleScroll = () => {
    if (!ref.current) return
    const i = Math.round(ref.current.scrollTop / ITEM_H)
    const clamped = Math.max(0, Math.min(items.length - 1, i))
    if (items[clamped] !== value) onChange(items[clamped])
  }
  // Scroll to current value on mount / value change
  const initRef = useRef(false)
  if (!initRef.current && typeof window !== 'undefined') {
    initRef.current = true
    setTimeout(() => scrollTo(Math.max(0, idx)), 0)
  }

  return (
    <div className="relative flex-1 overflow-hidden" style={{ height: ITEM_H * 3 }}>
      {/* fade top/bottom */}
      <div className="absolute inset-x-0 top-0 h-8 pointer-events-none z-10"
        style={{ background: 'linear-gradient(to bottom, #13171F, transparent)' }} />
      <div className="absolute inset-x-0 bottom-0 h-8 pointer-events-none z-10"
        style={{ background: 'linear-gradient(to top, #13171F, transparent)' }} />
      {/* selection highlight */}
      <div className="absolute inset-x-0 pointer-events-none z-10 rounded-lg"
        style={{ top: ITEM_H, height: ITEM_H, background: 'rgba(251,146,60,0.12)', border: '1px solid rgba(251,146,60,0.35)' }} />
      <div
        ref={ref}
        onScroll={handleScroll}
        className="h-full overflow-y-scroll"
        style={{ scrollSnapType: 'y mandatory', WebkitOverflowScrolling: 'touch', scrollbarWidth: 'none' }}
      >
        <div style={{ height: ITEM_H }} /> {/* top spacer */}
        {items.map(item => (
          <div
            key={item}
            onClick={() => { onChange(item); scrollTo(items.indexOf(item)) }}
            style={{
              height: ITEM_H,
              scrollSnapAlign: 'center',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 15,
              fontWeight: items.indexOf(item) === items.indexOf(value) ? 600 : 400,
              color: items.indexOf(item) === items.indexOf(value) ? '#FB923C' : '#7A8494',
              cursor: 'pointer',
              transition: 'color 0.15s, font-weight 0.15s',
            }}
          >{item}</div>
        ))}
        <div style={{ height: ITEM_H }} /> {/* bottom spacer */}
      </div>
    </div>
  )
}

function DateWheelPicker({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  // value format: DD-MM-YYYY
  const parts = value ? value.split('-') : ['01', '01', String(new Date().getFullYear() - 25)]
  const [day, setDay]     = useState(parts[0] || '01')
  const [month, setMonth] = useState(parts[1] || '01')
  const [year, setYear]   = useState(parts[2] || String(new Date().getFullYear() - 25))

  const update = (d: string, m: string, y: string) => {
    setDay(d); setMonth(m); setYear(y)
    onChange(`${d}-${m}-${y}`)
  }

  return (
    <div
      className="w-full flex gap-1 items-stretch"
      style={{
        background: '#1A1F27',
        border: '1.5px solid #FB923C',
        borderRadius: '1rem',
        overflow: 'hidden',
        padding: '0 12px',
      }}
    >
      <WheelColumn items={DAYS}   value={day}   onChange={d => update(d, month, year)} />
      <div className="w-px self-stretch my-2" style={{ background: 'rgba(255,255,255,0.12)' }} />
      <WheelColumn items={MONTHS} value={month} onChange={m => update(day, m, year)} />
      <div className="w-px self-stretch my-2" style={{ background: 'rgba(255,255,255,0.12)' }} />
      <WheelColumn items={YEARS}  value={year}  onChange={y => update(day, month, y)} />
    </div>
  )
}

// ─── Disclaimer screens ──────────────────────────────────────────────────────
const DISCLAIMERS = [
  {
    heading: 'Not a Medical Device',
    body: 'Gliimr is a wellness companion, not a licensed medical tool. It does not diagnose, treat, or prevent any health condition.',
  },
  {
    heading: 'Your Data Stays Private',
    body: 'Everything you share stays on your device. We never sell or share your personal health information with third parties.',
  },
  {
    heading: 'AI Has Limits',
    body: 'Our AI is trained to be empathetic and insightful, but it can make mistakes. Always consult a professional for serious health concerns.',
  },
  {
    heading: 'You Are in Control',
    body: 'You can delete your data at any time from the Profile screen. Your journey with Gliimr is entirely on your terms.',
  },
]

function DisclaimerFlow({ name, onFinish }: { name: string; onFinish: () => void }) {
  const [idx, setIdx] = useState(0)
  const [dir, setDir] = useState(1)

  const isLast = idx === DISCLAIMERS.length - 1

  const next = () => {
    if (isLast) { onFinish(); return }
    setDir(1)
    setIdx(i => i + 1)
  }

  const variants = {
    enter:  (d: number) => ({ x: d > 0 ? 60 : -60, opacity: 0 }),
    center: { x: 0, opacity: 1 },
    exit:   (d: number) => ({ x: d > 0 ? -60 : 60, opacity: 0 }),
  }

  return (
    <div className="fixed inset-0 bg-background flex flex-col items-center justify-center px-6 overflow-hidden">
      <motion.div initial={{ opacity: 0, y: -16 }} animate={{ opacity: 1, y: 0 }} className="mb-10 text-center">
        <h1
          className="text-2xl font-bold tracking-tight"
          style={{
            background: 'linear-gradient(90deg, #FB923C 10%, #FDE68A 50%, #2D7A7F 90%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
          }}
        >Before we begin</h1>
        <p className="text-muted-foreground text-sm mt-1">A few things to keep in mind</p>
      </motion.div>

      {/* Dots */}
      <div className="flex gap-2 mb-8">
        {DISCLAIMERS.map((_, i) => (
          <motion.div
            key={i}
            animate={{ backgroundColor: i <= idx ? '#FB923C' : '#22262D', width: i === idx ? 24 : 8 }}
            transition={{ duration: 0.3 }}
            className="h-2 rounded-full"
          />
        ))}
      </div>

      <div className="w-full max-w-sm relative overflow-hidden" style={{ minHeight: 180 }}>
        <AnimatePresence custom={dir} mode="wait">
          <motion.div
            key={idx}
            custom={dir}
            variants={variants}
            initial="enter"
            animate="center"
            exit="exit"
            transition={{ duration: 0.3, ease: 'easeInOut' }}
            className="absolute inset-0 flex flex-col gap-4"
          >
            <p className="text-xs text-muted-foreground">{idx + 1} / {DISCLAIMERS.length}</p>
            <h2 className="text-xl font-semibold text-foreground leading-relaxed">{DISCLAIMERS[idx].heading}</h2>
            <p className="text-muted-foreground text-sm leading-relaxed">{DISCLAIMERS[idx].body}</p>
          </motion.div>
        </AnimatePresence>
      </div>

      <div className="w-full max-w-sm mt-10">
        <motion.button
          whileTap={{ scale: 0.97 }}
          onClick={next}
          className="w-full py-4 rounded-2xl font-semibold flex items-center justify-center gap-2 transition-all"
          style={{ background: '#FB923C', color: '#000' }}
        >
          {isLast ? `Got it, let's go!` : 'Next'}
          <ChevronRight size={18} />
        </motion.button>
      </div>
    </div>
  )
}

// ─── Main Onboarding ─────────────────────────────────────────────────────────
export default function OnboardingScreen({ onComplete }: Props) {
  const [step, setStep] = useState(0)
  const [values, setValues] = useState<Record<Step, string>>({
    name: '', age: '', sex: '', birthdate: '', weight: '', height: '',
  })
  const [direction, setDirection] = useState(1)
  const [showDisclaimer, setShowDisclaimer] = useState(false)

  const currentStep = STEPS[step]
  const config = STEP_CONFIG[currentStep]
  const allFilled = STEPS.every(s => isValid(s, values[s]))
  const currentValid = isValid(currentStep, values[currentStep])

  const goNext = () => {
    if (!currentValid) return
    if (step < STEPS.length - 1) { setDirection(1); setStep(s => s + 1) }
  }
  const goBack = () => {
    if (step > 0) { setDirection(-1); setStep(s => s - 1) }
  }
  const handleComplete = () => {
    if (!allFilled) return
    setShowDisclaimer(true)
  }

  const variants = {
    enter:  (d: number) => ({ x: d > 0 ? 60 : -60, opacity: 0 }),
    center: { x: 0, opacity: 1 },
    exit:   (d: number) => ({ x: d > 0 ? -60 : 60, opacity: 0 }),
  }

  if (showDisclaimer) {
    return (
      <DisclaimerFlow
        name={values.name}
        onFinish={() => onComplete(values as UserProfile)}
      />
    )
  }

  return (
    <div className="fixed inset-0 bg-background flex flex-col items-center justify-center px-6 overflow-hidden">
      {/* Brand */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="mb-10 text-center"
      >
        <h1
          className="text-3xl font-bold tracking-tight"
          style={{
            background: 'linear-gradient(90deg, #FB923C 10%, #FDE68A 50%, #2D7A7F 90%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
          }}
        >Gliimr</h1>
        <p className="text-muted-foreground text-sm mt-1">Your health companion for life</p>
      </motion.div>

      {/* Progress dots */}
      <div className="flex gap-2 mb-8">
        {STEPS.map((s, i) => (
          <motion.div
            key={s}
            animate={{
              backgroundColor: isValid(s, values[s]) ? '#FB923C' : i === step ? '#2D7A7F' : '#22262D',
              width: i === step ? 24 : 8,
            }}
            transition={{ duration: 0.3 }}
            className="h-2 rounded-full"
          />
        ))}
      </div>

      {/* Card */}
      <div className="w-full max-w-sm relative overflow-hidden" style={{ minHeight: currentStep === 'birthdate' ? 200 : 220 }}>
        <AnimatePresence custom={direction} mode="wait">
          <motion.div
            key={currentStep}
            custom={direction}
            variants={variants}
            initial="enter"
            animate="center"
            exit="exit"
            transition={{ duration: 0.3, ease: 'easeInOut' }}
            className="absolute inset-0 flex flex-col gap-4"
          >
            <p className="text-muted-foreground text-sm">{`Step ${step + 1} of ${STEPS.length}`}</p>
            <h2 className="text-xl font-semibold text-foreground leading-relaxed">{config.label}</h2>

            <div className="relative">
              {config.type === 'select' ? (
                <div className="relative">
                  <select
                    value={values.sex}
                    onChange={e => setValues(v => ({ ...v, sex: e.target.value }))}
                    className="w-full bg-input border border-border rounded-2xl px-5 py-4 text-foreground text-base appearance-none focus:outline-none transition-colors"
                    style={{ borderColor: values.sex ? '#FB923C' : undefined }}
                  >
                    <option value="" disabled>Select sex...</option>
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                  </select>
                  <ChevronDown className="absolute right-4 top-1/2 -translate-y-1/2 text-muted-foreground pointer-events-none" size={18} />
                  {/* Check badge outside the select, so it won't overlap the dropdown */}
                  <AnimatePresence>
                    {values.sex && (
                      <motion.div
                        initial={{ scale: 0, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        exit={{ scale: 0, opacity: 0 }}
                        className="absolute right-4 top-1/2 -translate-y-1/2 w-6 h-6 rounded-full flex items-center justify-center"
                        style={{ background: '#FB923C' }}
                      >
                        <Check size={13} className="text-black" strokeWidth={3} />
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              ) : currentStep === 'birthdate' ? (
                /* Date wheel: check indicator shown below the picker, not overlapping */
                <div className="flex flex-col gap-2">
                  <DateWheelPicker
                    value={values.birthdate}
                    onChange={v => setValues(prev => ({ ...prev, birthdate: v }))}
                  />
                  <AnimatePresence>
                    {currentValid && (
                      <motion.div
                        initial={{ opacity: 0, y: -4 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -4 }}
                        className="flex items-center gap-1.5 px-1"
                      >
                        <div className="w-4 h-4 rounded-full flex items-center justify-center flex-shrink-0" style={{ background: '#FB923C' }}>
                          <Check size={9} className="text-black" strokeWidth={3} />
                        </div>
                        <span className="text-xs" style={{ color: '#FB923C' }}>Date selected</span>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              ) : (
                <input
                  type={config.type}
                  placeholder={config.placeholder}
                  value={values[currentStep]}
                  onChange={e => setValues(v => ({ ...v, [currentStep]: e.target.value }))}
                  onKeyDown={e => e.key === 'Enter' && goNext()}
                  className="w-full bg-input border border-border rounded-2xl px-5 py-4 text-foreground text-base placeholder:text-muted-foreground focus:outline-none transition-colors"
                  style={{ borderColor: currentValid ? '#FB923C' : undefined, paddingRight: currentValid ? '3rem' : undefined }}
                  autoFocus
                />
              )}

              {/* Floating check for text/number inputs only */}
              {currentStep !== 'birthdate' && config.type !== 'select' && (
                <AnimatePresence>
                  {currentValid && (
                    <motion.div
                      initial={{ scale: 0, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      exit={{ scale: 0, opacity: 0 }}
                      className="absolute right-4 top-1/2 -translate-y-1/2 w-6 h-6 rounded-full flex items-center justify-center pointer-events-none"
                      style={{ background: '#FB923C' }}
                    >
                      <Check size={13} className="text-black" strokeWidth={3} />
                    </motion.div>
                  )}
                </AnimatePresence>
              )}
            </div>
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Nav buttons */}
      <div className="w-full max-w-sm mt-8 flex gap-3">
        {step > 0 && (
          <motion.button
            whileTap={{ scale: 0.95 }}
            onClick={goBack}
            className="flex-1 py-4 rounded-2xl font-medium transition-colors"
            style={{ background: '#1A1F27', color: '#fff' }}
          >
            Back
          </motion.button>
        )}

        {step < STEPS.length - 1 ? (
          <motion.button
            whileTap={{ scale: 0.95 }}
            onClick={goNext}
            disabled={!currentValid}
            className="flex-1 py-4 rounded-2xl font-medium transition-all"
            style={{
              background: currentValid ? '#FB923C' : '#22262D',
              color: currentValid ? '#000' : '#7A8494',
            }}
          >
            Next
          </motion.button>
        ) : (
          <motion.button
            whileTap={{ scale: 0.95 }}
            onClick={handleComplete}
            disabled={!allFilled}
            animate={allFilled
              ? { boxShadow: '0 0 24px rgba(251,146,60,0.55)' }
              : { boxShadow: '0 0 0px rgba(251,146,60,0)' }
            }
            transition={{ duration: 0.4 }}
            className="flex-1 py-4 rounded-2xl font-semibold transition-all"
            style={{
              background: allFilled ? '#FB923C' : '#22262D',
              color: allFilled ? '#000' : '#7A8494',
            }}
          >
            Complete
          </motion.button>
        )}
      </div>
    </div>
  )
}
