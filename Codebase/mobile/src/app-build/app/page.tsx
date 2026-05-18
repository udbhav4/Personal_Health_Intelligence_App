'use client'
import { motion, AnimatePresence } from 'framer-motion'
import { useStore } from '@/hooks/use-store'
import OnboardingScreen from '@/components/onboarding-screen'
import AppShell from '@/components/app-shell'

export default function Page() {
  const { hasCompletedOnboarding, profile, saveProfile, updateProfile } = useStore()

  // Hydrating — render nothing to avoid flash
  if (hasCompletedOnboarding === null) {
    return <div className="fixed inset-0 bg-background" />
  }

  return (
    <div className="fixed inset-0 bg-background overflow-hidden">
      <AnimatePresence mode="wait">
        {!hasCompletedOnboarding ? (
          <motion.div
            key="onboarding"
            className="absolute inset-0"
            exit={{ scale: 1.08, opacity: 0 }}
            transition={{ duration: 0.45, ease: 'easeInOut' }}
          >
            <OnboardingScreen onComplete={saveProfile} />
          </motion.div>
        ) : (
          <motion.div
            key="app"
            className="absolute inset-0"
            initial={{ scale: 0.92, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.45, ease: 'easeOut' }}
          >
            <AppShell profile={profile} onProfileSave={updateProfile} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
