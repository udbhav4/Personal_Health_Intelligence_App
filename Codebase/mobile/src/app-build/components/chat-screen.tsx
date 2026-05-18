'use client'
import { useState, useRef, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronDown } from 'lucide-react'
import type { UserProfile } from '@/hooks/use-store'

type Feature = 'Journal' | 'Talk' | 'Report'
type Mode = 'Glance' | 'Reflect' | 'Ultra'

interface Message {
  id: string
  role: 'user' | 'model'
  text: string
  phase?: 'thinking' | 'response'
}

const PLACEHOLDERS = [
  'Tell me something about your lifestyle',
  'How is it going?',
  "What's happening?",
  'How do you feel?',
  'What did you do today?',
]

const GLANCE_LOADING = ['Understanding what you said', 'Thinking', 'Hold on...']

const MODE_INFO: Record<Mode, { sub: string }> = {
  Glance:  { sub: 'Short and quick!' },
  Reflect: { sub: 'Deep thought' },
  Ultra:   { sub: 'Heavy analysis' },
}

const SIMULATED: Record<Feature, Record<Mode, string>> = {
  Journal: {
    Glance:  "Quick snapshot: you're keeping consistent with your journal. Patterns suggest mild cognitive load today — nothing alarming. Keep tracking!",
    Reflect: '',
    Ultra:   '',
  },
  Talk: {
    Glance:  "Got it. You seem to be in a reflective mood. That's healthy — processing emotions is step one. I'm here whenever you want to go deeper.",
    Reflect: "Let me think through what you've shared... There's a recurring theme of low-energy mornings and high-focus evenings, which aligns with a delayed circadian rhythm. Worth exploring sleep hygiene.",
    Ultra:   '',
  },
  Report: {
    Glance:  '',
    Reflect: '',
    Ultra:   "Deep analysis initiated. Cross-referencing your last 14 days of inputs: sleep averages 6.1h (below optimal), screen time peaks at 10–11pm (disrupting melatonin), and physical activity is front-loaded on weekdays only. Recommend a 3-week experiment: shift screen cutoff to 9pm, add a 15-min wind-down walk.",
  },
}

const AGENTIC_THOUGHTS: Record<Exclude<Mode, 'Glance'>, string> = {
  Reflect: 'Analysing conversational context and cross-referencing behavioural history to surface deeper insights...',
  Ultra:   'Running multi-modal health pattern analysis across all available data streams — this may take a moment...',
}

// Typewriter: 25ms for thinking/Glance, 45ms for final response
function TypingText({ text, speed = 45 }: { text: string; speed?: number }) {
  const [displayed, setDisplayed] = useState('')
  useEffect(() => {
    setDisplayed('')
    let i = 0
    const iv = setInterval(() => {
      i++
      setDisplayed(text.slice(0, i))
      if (i >= text.length) clearInterval(iv)
    }, speed)
    return () => clearInterval(iv)
  }, [text, speed])
  return <span>{displayed}</span>
}

function PlaceholderRotator({ visible }: { visible: boolean }) {
  const [idx, setIdx] = useState(0)
  const [key, setKey] = useState(0)
  useEffect(() => {
    const iv = setInterval(() => {
      setIdx(i => (i + 1) % PLACEHOLDERS.length)
      setKey(k => k + 1)
    }, 4000)
    return () => clearInterval(iv)
  }, [])
  if (!visible) return null
  return (
    <span
      key={key}
      className="absolute inset-0 flex items-center pointer-events-none text-muted-foreground text-sm placeholder-slide"
      style={{ overflow: 'hidden', whiteSpace: 'nowrap', lineHeight: '1.5' }}
    >
      {PLACEHOLDERS[idx]}
    </span>
  )
}

// Send arrow: short horizontal → smooth 90° turn → long vertical → clean arrowhead
// active prop: when button is glowing/pressed, stroke turns black
function CurvedArrow({ active }: { active?: boolean }) {
  const stroke = active ? '#000000' : 'white'
  return (
    <svg width="26" height="26" viewBox="0 0 26 26" fill="none" xmlns="http://www.w3.org/2000/svg">
      {/* Horizontal base (shorter) then smooth corner then long vertical */}
      <path
        d="M5 19 L12 19 Q14.5 19, 14.5 16.5 L14.5 7"
        stroke={stroke}
        strokeWidth="1.4"
        strokeLinecap="round"
        fill="none"
      />
      {/* Large arrowhead — wider V for clear visibility */}
      <path
        d="M10.5 10.5 L14.5 6 L18.5 10.5"
        stroke={stroke}
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
    </svg>
  )
}

export default function ChatScreen({ profile }: { profile: UserProfile }) {
  const [feature, setFeature] = useState<Feature>('Talk')
  const [mode, setMode] = useState<Mode>('Glance')
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [loadingTextIdx, setLoadingTextIdx] = useState(0)
  const [showModeMenu, setShowModeMenu] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const bottomRef   = useRef<HTMLDivElement>(null)
  const modeMenuRef = useRef<HTMLDivElement>(null)

  // Close mode menu on any click outside it
  useEffect(() => {
    if (!showModeMenu) return
    const handler = (e: MouseEvent) => {
      if (modeMenuRef.current && !modeMenuRef.current.contains(e.target as Node)) {
        setShowModeMenu(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [showModeMenu])

  const allowedModes: Mode[] =
    feature === 'Journal' ? ['Glance'] :
    feature === 'Talk'    ? ['Glance', 'Reflect'] :
                            ['Ultra']

  useEffect(() => {
    if (!allowedModes.includes(mode)) setMode(allowedModes[0])
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [feature])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    if (!loading || mode !== 'Glance') return
    setLoadingTextIdx(0)
    const iv = setInterval(() => setLoadingTextIdx(i => (i + 1) % GLANCE_LOADING.length), 3000)
    return () => clearInterval(iv)
  }, [loading, mode])

  // Auto-expand textarea
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 150) + 'px'
  }, [input])

  const send = useCallback(() => {
    if (!input.trim() || loading) return
    const userMsg: Message = { id: Date.now().toString(), role: 'user', text: input.trim() }
    setMessages(m => [...m, userMsg])
    setInput('')
    setLoading(true)

    const delay = mode === 'Ultra' ? 4000 : mode === 'Reflect' ? 3000 : 2500

    setTimeout(() => {
      const response = SIMULATED[feature][mode]
      const modelMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'model',
        text: response,
        phase: 'response',
      }
      setMessages(m => [...m, modelMsg])
      setLoading(false)
    }, delay)
  }, [input, loading, feature, mode])

  const hasMessages = messages.length > 0
  const canSend = !!input.trim() && !loading

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Top heading bar — only shown once a chat has started */}
      <AnimatePresence>
        {hasMessages && (
          <motion.div
            initial={{ opacity: 0, y: -12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.25 }}
            className="flex-shrink-0 px-4 pt-3 pb-2"
          >
            <h2 className="text-base font-semibold text-center leading-snug">
              Talk with{' '}
              <span
                style={{
                  background: 'linear-gradient(90deg, #FB923C 10%, #FDE68A 50%, #2D7A7F 90%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundClip: 'text',
                }}
              >Gliimr</span>
            </h2>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto px-4 pt-4 pb-2 flex flex-col">
        {!hasMessages ? (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="flex-1 flex flex-col items-center justify-center text-center px-4 gap-6"
          >
            <div>
              {/* Personalised greeting */}
              <p
                className="text-2xl font-bold mb-3 leading-relaxed"
                style={{
                  background: 'linear-gradient(90deg, #FB923C 10%, #FDE68A 50%, #2D7A7F 90%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundClip: 'text',
                }}
              >
                Hello {profile.name || 'there'}!
              </p>
              <p className="text-muted-foreground text-sm leading-relaxed text-balance">
                I am{' '}
                <span
                  className="font-semibold"
                  style={{
                    background: 'linear-gradient(90deg, #FB923C 10%, #FDE68A 50%, #2D7A7F 90%)',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                    backgroundClip: 'text',
                  }}
                >Gliimr</span>
                , your (health) companion for life. Keep telling me about yourself and I&apos;ll keep knowing you more. Looking forward to our first conversation!
              </p>
            </div>
          </motion.div>
        ) : (
          <div className="flex flex-col gap-3">
            <AnimatePresence initial={false}>
              {messages.map(msg => (
                <motion.div
                  key={msg.id}
                  initial={{ opacity: 0, y: 8, scale: 0.97 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  transition={{ duration: 0.22 }}
                  className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className="max-w-[82%] px-4 py-3 text-sm leading-relaxed"
                    style={{
                      background: msg.role === 'user' ? '#FB923C' : '#1E232B',
                      color: msg.role === 'user' ? '#000000' : '#FFFFFF',
                      borderRadius: msg.role === 'user'
                        ? '18px 18px 4px 18px'
                        : '18px 18px 18px 4px',
                      boxShadow: msg.role === 'model' ? '0 4px 18px rgba(0,0,0,0.4)' : undefined,
                      overflowWrap: 'anywhere',
                      wordBreak: 'break-all',
                      hyphens: 'auto',
                    }}
                  >
                    {msg.phase === 'response'
                      ? <TypingText text={msg.text} speed={45} />
                      : msg.text
                    }
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>

            {/* Loading — naked typewriter on background for Glance; elevated bubble for agentic */}
            <AnimatePresence>
              {loading && (
                <motion.div
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 8 }}
                  className="flex justify-start items-center gap-2 pl-1"
                >
                  <div className="gradient-ring" />
                  {mode === 'Glance' ? (
                    <AnimatePresence mode="wait">
                      <motion.span
                        key={loadingTextIdx}
                        initial={{ opacity: 0, y: 5 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -5 }}
                        transition={{ duration: 0.25 }}
                        className="text-sm"
                        style={{ color: '#7EC4C8' }}
                      >
                        <TypingText text={GLANCE_LOADING[loadingTextIdx]} speed={25} />
                      </motion.span>
                    </AnimatePresence>
                  ) : (
                    <span className="text-sm" style={{ color: '#7EC4C8', maxWidth: 220 }}>
                      <TypingText text={AGENTIC_THOUGHTS[mode as Exclude<Mode, 'Glance'>]} speed={25} />
                    </span>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* ── Unified Chat Bar ─────────────────────────────────── */}
      <div className="px-3 pb-3 pt-2 flex-shrink-0">
        <div
          className="rounded-[2rem] px-4 py-3 flex flex-col gap-2"
          style={{
            background: '#0B0E14',
            border: '1px solid rgba(255,255,255,0.15)',
          }}
        >
          {/* Top row: Mode dropdown left */}
          <div className="flex items-center">
              <div className="relative" ref={modeMenuRef}>
              <motion.button
                whileTap={{ scale: 0.95 }}
                onClick={(e) => { e.stopPropagation(); setShowModeMenu(m => !m) }}
                className="flex items-center gap-1 px-3 py-1 rounded-full text-[10px] font-semibold transition-all"
                style={{
                  background: 'rgba(251,146,60,0.14)',
                  border: '1px solid rgba(251,146,60,0.55)',
                  color: '#FB923C',
                }}
              >
                {mode}
                <ChevronDown size={10} />
              </motion.button>

              <AnimatePresence>
                {showModeMenu && (
                  <motion.div
                    initial={{ opacity: 0, y: 8, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: 8, scale: 0.95 }}
                    transition={{ duration: 0.15 }}
                    className="absolute bottom-full mb-2 left-0 w-44 rounded-2xl overflow-hidden z-50"
                    style={{ background: '#13171F', border: '1px solid #22262D' }}
                  >
                    {(['Glance', 'Reflect', 'Ultra'] as Mode[]).map(m => {
                      const enabled = allowedModes.includes(m)
                      const selected = mode === m
                      return (
                        <button
                          key={m}
                          disabled={!enabled}
                          onClick={() => { if (enabled) { setMode(m); setShowModeMenu(false) } }}
                          className="w-full text-left px-4 py-3 transition-colors"
                          style={{
                            background: selected ? 'rgba(251,146,60,0.12)' : 'transparent',
                            borderLeft: selected ? '2px solid #FB923C' : '2px solid transparent',
                            opacity: enabled ? 1 : 0.3,
                            cursor: enabled ? 'pointer' : 'not-allowed',
                          }}
                        >
                          <p className="text-sm font-medium" style={{ color: selected ? '#FB923C' : '#fff' }}>{m}</p>
                          <p className="text-xs text-muted-foreground">{MODE_INFO[m].sub}</p>
                        </button>
                      )
                    })}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>

          {/* Middle: Auto-expanding textarea */}
          <div className="relative">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }}
              rows={1}
              className="w-full bg-transparent text-foreground text-sm resize-none focus:outline-none leading-relaxed"
              style={{
                caretColor: '#FB923C',
                maxHeight: 150,
                overflowY: 'auto',
              }}
            />
            <PlaceholderRotator visible={!input} />
          </div>

          {/* Bottom row: Feature pills left + Send button right */}
          <div className="flex items-center justify-between">
            <div className="flex gap-1.5">
              {(['Journal', 'Talk', 'Report'] as Feature[]).map(f => (
                <motion.button
                  key={f}
                  whileTap={{ scale: 0.93 }}
                  onClick={() => setFeature(f)}
                  className="px-3 py-1 rounded-full text-xs font-medium transition-all"
                  style={
                    feature === f
                      ? { background: 'rgba(45,122,127,0.35)', color: '#2D7A7F', border: '1px solid rgba(45,122,127,0.7)' }
                      : { background: 'rgba(45,122,127,0.10)', color: '#7A8494', border: '1px solid transparent' }
                  }
                >
                  {f}
                </motion.button>
              ))}
            </div>

            {/* Send button — solid Apricot circle, curved-tail arrow */}
            <motion.button
              whileTap={{ scale: 0.9 }}
              onClick={send}
              disabled={!canSend}
              className="w-9 h-9 rounded-full flex items-center justify-center flex-shrink-0 transition-all"
              style={{
                background: canSend ? '#FB923C' : 'rgba(251,146,60,0.2)',
                boxShadow: canSend ? '0 0 8px rgba(251,146,60,0.22)' : 'none',
              }}
            >
              <CurvedArrow />
            </motion.button>
          </div>
        </div>
      </div>
    </div>
  )
}
