'use client'
import { useState, useEffect } from 'react'

export interface UserProfile {
  name: string
  age: string
  sex: string
  birthdate: string
  weight: string
  height: string
}

const STORAGE_KEY = 'gliimr_onboarding'
const COMPLETED_KEY = 'gliimr_completed'

export function useStore() {
  const [hasCompletedOnboarding, setHasCompletedOnboarding] = useState<boolean | null>(null)
  const [profile, setProfile] = useState<UserProfile>({
    name: '', age: '', sex: '', birthdate: '', weight: '', height: '',
  })

  useEffect(() => {
    const completed = localStorage.getItem(COMPLETED_KEY) === 'true'
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved) {
      try { setProfile(JSON.parse(saved)) } catch {}
    }
    setHasCompletedOnboarding(completed)
  }, [])

  const saveProfile = (data: UserProfile) => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data))
    localStorage.setItem(COMPLETED_KEY, 'true')
    setProfile(data)
    setHasCompletedOnboarding(true)
  }

  const updateProfile = (data: UserProfile) => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data))
    setProfile(data)
  }

  return { hasCompletedOnboarding, profile, saveProfile, updateProfile }
}
