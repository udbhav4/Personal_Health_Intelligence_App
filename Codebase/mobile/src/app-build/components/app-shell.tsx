'use client'
import { useState, useRef, useCallback } from 'react'
import { motion } from 'framer-motion'
import { LayoutDashboard, MessageCircle, UserCircle } from 'lucide-react'
import DashboardScreen from './dashboard-screen'
import ChatScreen from './chat-screen'
import ProfileScreen from './profile-screen'
import type { UserProfile } from '@/hooks/use-store'

type Tab = 'dashboard' | 'chat' | 'profile'

const TABS: { id: Tab; label: string; icon: React.ComponentType<{ size?: number }> }[] = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'chat',      label: 'Chat',      icon: MessageCircle },
  { id: 'profile',   label: 'Profile',   icon: UserCircle },
]

const TAB_ORDER: Tab[] = ['dashboard', 'chat', 'profile']

interface Props {
  profile: UserProfile
  onProfileSave: (p: UserProfile) => void
}

export default function AppShell({ profile, onProfileSave }: Props) {
  const [tab, setTab] = useState<Tab>('chat')
  // offset: -1=dashboard(0%), 0=chat(33.33%), 1=profile(66.66%)
  const tabIdx = TAB_ORDER.indexOf(tab)

  // Swipe gesture
  const swipeStartX = useRef<number | null>(null)

  const switchTab = useCallback((next: Tab) => {
    setTab(next)
  }, [])

  const onTouchStart = (e: React.TouchEvent) => {
    swipeStartX.current = e.touches[0].clientX
  }
  const onTouchEnd = (e: React.TouchEvent) => {
    if (swipeStartX.current === null) return
    const dx = e.changedTouches[0].clientX - swipeStartX.current
    swipeStartX.current = null
    if (Math.abs(dx) < 50) return
    const cur = TAB_ORDER.indexOf(tab)
    if (dx < 0 && cur < TAB_ORDER.length - 1) switchTab(TAB_ORDER[cur + 1])
    if (dx > 0 && cur > 0) switchTab(TAB_ORDER[cur - 1])
  }

  return (
    <div className="flex flex-col h-[100dvh] overflow-hidden bg-background">
      {/* Screen area — all three screens stay mounted; translate slides between them */}
      <div
        className="flex-1 overflow-hidden relative"
        onTouchStart={onTouchStart}
        onTouchEnd={onTouchEnd}
      >
        {/* Sliding strip: 3 screens side by side, translate by tabIdx.
            initial=false prevents the spring from animating from x:0 on mount,
            which caused the wrong screen to flash on first load. */}
        <motion.div
          className="absolute inset-y-0 flex"
          style={{ width: '300%', left: 0 }}
          initial={{ x: `${-tabIdx * (100 / 3)}%` }}
          animate={{ x: `${-tabIdx * (100 / 3)}%` }}
          transition={{ type: 'spring', stiffness: 320, damping: 36, mass: 0.9 }}
        >
          {TAB_ORDER.map(id => (
            <div key={id} className="overflow-hidden" style={{ width: '33.3333%', flexShrink: 0 }}>
              {id === 'dashboard' && <DashboardScreen profile={profile} />}
              {id === 'chat'      && <ChatScreen profile={profile} />}
              {id === 'profile'   && <ProfileScreen profile={profile} onSave={onProfileSave} isActive={tab === 'profile'} />}
            </div>
          ))}
        </motion.div>
      </div>

      {/* Bottom Nav */}
      <nav className="glass-nav px-6 pt-2 pb-5 flex items-center justify-around flex-shrink-0">
        {TABS.map(({ id, label, icon: Icon }) => {
          const active = tab === id
          return (
            <motion.button
              key={id}
              whileTap={{ scale: 0.9 }}
              onClick={() => switchTab(id)}
              className="flex flex-col items-center gap-0.5 px-4 py-1 rounded-xl transition-all"
            >
              <Icon size={22} style={{ color: active ? '#FB923C' : '#7A8494' }} />
              <span className="text-xs font-medium transition-colors" style={{ color: active ? '#FB923C' : '#7A8494' }}>
                {label}
              </span>
              {active && (
                <motion.div
                  layoutId="nav-dot"
                  className="w-1 h-1 rounded-full"
                  style={{ background: '#FB923C' }}
                />
              )}
            </motion.button>
          )
        })}
      </nav>
    </div>
  )
}
