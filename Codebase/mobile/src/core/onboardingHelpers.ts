/**
 * Onboarding computation helpers — BMI and Age.
 *
 * These helpers compute and discretize values that are written to user_data_sensorless
 * (NOT sensor_windows).  They are UI-agnostic: call them from whatever onboarding
 * screen collects the user's weight/height/birthdate.
 *
 * Both entries use expires_date = null (permanent until the user updates them).
 */

// ── BMI ────────────────────────────────────────────────────────────────────────

/**
 * Compute Body Mass Index from weight in kg and height in cm.
 * Formula: weight / (height_m)^2
 */
export function computeBmi(weightKg: number, heightCm: number): number {
  return weightKg / ((heightCm / 100) ** 2);
}

/**
 * Discretize BMI into state labels matching the DBN node config.
 * Thresholds: [0, 18.5, 25, 30, 100]
 * State labels: 'underweight' | 'normal' | 'overweight' | 'obese'
 */
export function discretizeBmi(bmi: number): string {
  if (bmi < 18.5) return 'underweight';
  if (bmi < 25)   return 'normal';
  if (bmi < 30)   return 'overweight';
  return 'obese';
}

// ── Age ────────────────────────────────────────────────────────────────────────

/**
 * Compute age in fractional years from an ISO birthdate string (YYYY-MM-DD).
 * Uses milliseconds for precision — no timezone issues for date-only inputs.
 */
export function computeAgeYears(birthdateIso: string): number {
  return (Date.now() - new Date(birthdateIso).getTime()) / (365.25 * 86_400_000);
}

/**
 * Discretize age in years into state labels matching the DBN node config.
 * Thresholds: [18, 30, 45, 60, 100]
 * State labels: '18_29' | '30_44' | '45_59' | '60_plus'
 */
export function discretizeAge(ageYears: number): string {
  if (ageYears < 30) return '18_29';
  if (ageYears < 45) return '30_44';
  if (ageYears < 60) return '45_59';
  return '60_plus';
}
