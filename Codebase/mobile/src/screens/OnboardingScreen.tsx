import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  View, Text, TextInput, TouchableOpacity, ScrollView,
  StyleSheet, Dimensions, Platform,
} from 'react-native';
import Animated, {
  useSharedValue, useAnimatedStyle, withSpring, withTiming,
  FadeIn, FadeOut, SlideInRight, SlideOutLeft, SlideInLeft, SlideOutRight,
  interpolateColor, useDerivedValue,
} from 'react-native-reanimated';
import Svg, { Path } from 'react-native-svg';
import GliimrText from '../components/GliimrText';
import type { UserProfile } from '../core/AppContext';

const { width: SCREEN_W } = Dimensions.get('window');

// ── Colours (from globals.css) ────────────────────────────────────────────────

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

// ── Date-wheel constants ──────────────────────────────────────────────────────

const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
const DAYS   = Array.from({ length: 31 }, (_, i) => String(i + 1).padStart(2, '0'));
const YEARS  = Array.from({ length: 100 }, (_, i) => String(new Date().getFullYear() - i));

const ITEM_H = 40;

function WheelColumn({
  items, value, onChange,
}: { items: string[]; value: string; onChange: (v: string) => void }) {
  const scrollRef = useRef<ScrollView>(null);
  const idx       = items.indexOf(value);

  useEffect(() => {
    const i = Math.max(0, idx);
    setTimeout(() => scrollRef.current?.scrollTo({ y: i * ITEM_H, animated: false }), 50);
  // scroll to initial position once on mount
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const onMomentumEnd = useCallback((e: { nativeEvent: { contentOffset: { y: number } } }) => {
    const i = Math.round(e.nativeEvent.contentOffset.y / ITEM_H);
    const clamped = Math.max(0, Math.min(items.length - 1, i));
    if (items[clamped] !== value) onChange(items[clamped]);
  }, [items, value, onChange]);

  return (
    <View style={styles.wheelCol}>
      <View style={styles.wheelFadeTop} pointerEvents="none" />
      <View style={styles.wheelFadeBot} pointerEvents="none" />
      <View style={styles.wheelHighlight} pointerEvents="none" />
      <ScrollView
        ref={scrollRef}
        showsVerticalScrollIndicator={false}
        snapToInterval={ITEM_H}
        decelerationRate="fast"
        onMomentumScrollEnd={onMomentumEnd}
        contentContainerStyle={{ paddingVertical: ITEM_H }}
        style={{ height: ITEM_H * 3 }}
      >
        {items.map(item => (
          <TouchableOpacity
            key={item}
            onPress={() => {
              onChange(item);
              const i = items.indexOf(item);
              scrollRef.current?.scrollTo({ y: i * ITEM_H, animated: true });
            }}
          >
            <View style={[styles.wheelItem]}>
              <Text style={[
                styles.wheelItemText,
                item === value && styles.wheelItemSelected,
              ]}>
                {item}
              </Text>
            </View>
          </TouchableOpacity>
        ))}
      </ScrollView>
    </View>
  );
}

function DateWheelPicker({
  value, onChange,
}: { value: string; onChange: (v: string) => void }) {
  const parts = value ? value.split('-') : ['01', 'Jan', String(new Date().getFullYear() - 25)];
  const [day,   setDay]   = useState(parts[0] || '01');
  const [month, setMonth] = useState(parts[1] || 'Jan');
  const [year,  setYear]  = useState(parts[2] || String(new Date().getFullYear() - 25));

  const emit = (d: string, m: string, y: string) => {
    setDay(d); setMonth(m); setYear(y);
    onChange(`${d}-${m}-${y}`);
  };

  return (
    <View style={styles.wheelContainer}>
      <WheelColumn items={DAYS}   value={day}   onChange={d => emit(d, month, year)} />
      <View style={styles.wheelDivider} />
      <WheelColumn items={MONTHS} value={month} onChange={m => emit(day, m, year)} />
      <View style={styles.wheelDivider} />
      <WheelColumn items={YEARS}  value={year}  onChange={y => emit(day, month, y)} />
    </View>
  );
}

// ── Check icon ────────────────────────────────────────────────────────────────

function CheckIcon({ size = 13 }: { size?: number }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none">
      <Path
        d="M20 6L9 17L4 12"
        stroke="#000000"
        strokeWidth={3}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </Svg>
  );
}

// ── Disclaimer flow ───────────────────────────────────────────────────────────

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
];

function DisclaimerFlow({ onFinish }: { onFinish: () => void }) {
  const [idx, setIdx] = useState(0);
  const [dir, setDir] = useState<1 | -1>(1);
  const isLast = idx === DISCLAIMERS.length - 1;

  const next = () => {
    if (isLast) { onFinish(); return; }
    setDir(1);
    setIdx(i => i + 1);
  };

  const dotWidths = DISCLAIMERS.map((_, i) =>
    useSharedValue(i === 0 ? 24 : 8)
  );
  const dotColors = DISCLAIMERS.map((_, i) =>
    useSharedValue(i === 0 ? 1 : 0)
  );

  useEffect(() => {
    DISCLAIMERS.forEach((_, i) => {
      dotWidths[i].value = withTiming(i <= idx ? 24 : 8, { duration: 300 });
      dotColors[i].value = withTiming(i <= idx ? 1 : 0, { duration: 300 });
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [idx]);

  return (
    <View style={styles.fullScreen}>
      <Animated.View entering={FadeIn.duration(400)} style={styles.disclaimerHeader}>
        <Text style={styles.gradientTitle}>Before we begin</Text>
        <Text style={styles.subLabel}>A few things to keep in mind</Text>
      </Animated.View>

      <View style={styles.dotsRow}>
        {DISCLAIMERS.map((_, i) => {
          const animStyle = useAnimatedStyle(() => ({
            width: withTiming(i <= idx ? 24 : 8, { duration: 300 }),
            backgroundColor: withTiming(i <= idx ? C.primary : C.border, { duration: 300 }),
          }));
          return (
            <Animated.View key={i} style={[styles.dot, animStyle]} />
          );
        })}
      </View>

      <View style={[styles.cardArea, { minHeight: 180 }]}>
        <Animated.View
          key={idx}
          entering={dir > 0 ? SlideInRight.duration(300) : SlideInLeft.duration(300)}
          exiting={dir > 0 ? SlideOutLeft.duration(300) : SlideOutRight.duration(300)}
          style={styles.stepCard}
        >
          <Text style={styles.stepCount}>{idx + 1} / {DISCLAIMERS.length}</Text>
          <Text style={styles.stepHeading}>{DISCLAIMERS[idx].heading}</Text>
          <Text style={styles.stepBody}>{DISCLAIMERS[idx].body}</Text>
        </Animated.View>
      </View>

      <View style={styles.navRow}>
        <TouchableOpacity onPress={next} style={styles.primaryBtn} activeOpacity={0.85}>
          <Text style={styles.primaryBtnText}>{isLast ? "Got it, let's go!" : 'Next'}</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

// ── Main onboarding ───────────────────────────────────────────────────────────

type Step = 'name' | 'sex' | 'birthdate' | 'weight' | 'height';
const STEPS: Step[] = ['name', 'sex', 'birthdate', 'weight', 'height'];

const STEP_CONFIG: Record<Step, { label: string; placeholder: string; type: string }> = {
  name:      { label: 'What should I call you?',   placeholder: 'Your name',    type: 'text' },
  sex:       { label: 'What is your sex?',          placeholder: 'Select...',   type: 'select' },
  birthdate: { label: 'When were you born?',        placeholder: 'DD-Mon-YYYY', type: 'date' },
  weight:    { label: 'What is your weight?',       placeholder: 'e.g. 68 kg',  type: 'text' },
  height:    { label: 'What is your height?',       placeholder: 'e.g. 175 cm', type: 'text' },
};

function isValid(step: Step, val: string): boolean {
  if (!val.trim()) return false;
  return true;
}

function ProgressDot({ isActive, isFilled }: { isActive: boolean; isFilled: boolean }) {
  const dotStyle = useAnimatedStyle(() => ({
    width: withTiming(isActive ? 24 : 8, { duration: 300 }),
    backgroundColor: withTiming(
      isFilled ? C.primary : isActive ? C.secondary : C.border,
      { duration: 300 },
    ),
  }));
  return <Animated.View style={[styles.dot, dotStyle]} />;
}

interface Props {
  onComplete: (data: UserProfile) => Promise<void>;
}

export default function OnboardingScreen({ onComplete }: Props) {
  const [step,   setStep]   = useState(0);
  const [values, setValues] = useState<Record<Step, string>>({
    name: '', sex: '', birthdate: '', weight: '', height: '',
  });
  const [direction, setDirection] = useState<1 | -1>(1);
  const [showDisclaimer, setShowDisclaimer] = useState(false);

  const currentStep  = STEPS[step];
  const config       = STEP_CONFIG[currentStep];
  const allFilled    = STEPS.every(s => isValid(s, values[s]));
  const currentValid = isValid(currentStep, values[currentStep]);

  const goNext = () => {
    if (!currentValid) return;
    if (step < STEPS.length - 1) { setDirection(1); setStep(s => s + 1); }
  };
  const goBack = () => {
    if (step > 0) { setDirection(-1); setStep(s => s - 1); }
  };
  const handleComplete = () => {
    if (!allFilled) return;
    setShowDisclaimer(true);
  };

  if (showDisclaimer) {
    return (
      <DisclaimerFlow
        onFinish={() => onComplete(values as UserProfile)}
      />
    );
  }

  return (
    <View style={styles.fullScreen}>
      {/* Brand */}
      <Animated.View entering={FadeIn.duration(600)} style={styles.brand}>
        <GliimrText style={styles.gradientTitle} />
        <Text style={styles.subLabel}>Your health companion for life</Text>
      </Animated.View>

      {/* Progress dots */}
      <View style={styles.dotsRow}>
        {STEPS.map((s, i) => (
          <ProgressDot
            key={s}
            isActive={i === step}
            isFilled={isValid(s, values[s])}
          />
        ))}
      </View>

      {/* Card area */}
      <View style={[styles.cardArea, { minHeight: currentStep === 'birthdate' ? 200 : 220 }]}>
        <Animated.View
          key={currentStep}
          entering={direction > 0 ? SlideInRight.duration(300) : SlideInLeft.duration(300)}
          exiting={direction > 0 ? SlideOutLeft.duration(300) : SlideOutRight.duration(300)}
          style={styles.stepCard}
        >
          <Text style={styles.stepCount}>Step {step + 1} of {STEPS.length}</Text>
          <Text style={styles.stepHeading}>{config.label}</Text>

          {config.type === 'select' ? (
            <View>
              <View style={[
                styles.selectContainer,
                values.sex ? styles.selectActive : null,
              ]}>
                {(['Male', 'Female'] as const).map(opt => (
                  <TouchableOpacity
                    key={opt}
                    onPress={() => setValues(v => ({ ...v, sex: opt }))}
                    style={[styles.selectOption, values.sex === opt && styles.selectOptionActive]}
                    activeOpacity={0.8}
                  >
                    <Text style={[styles.selectOptionText, values.sex === opt && styles.selectOptionTextActive]}>
                      {opt}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
              {values.sex ? (
                <Animated.View entering={FadeIn.duration(200)} style={styles.checkBadgeRow}>
                  <View style={styles.checkBadge}><CheckIcon size={11} /></View>
                  <Text style={styles.checkLabel}>{values.sex} selected</Text>
                </Animated.View>
              ) : null}
            </View>
          ) : currentStep === 'birthdate' ? (
            <View>
              <DateWheelPicker
                value={values.birthdate}
                onChange={v => setValues(prev => ({ ...prev, birthdate: v }))}
              />
              {currentValid ? (
                <Animated.View entering={FadeIn.duration(200)} style={styles.checkBadgeRow}>
                  <View style={styles.checkBadge}><CheckIcon size={9} /></View>
                  <Text style={styles.checkLabel}>Date selected</Text>
                </Animated.View>
              ) : null}
            </View>
          ) : (
            <View style={styles.inputWrapper}>
              <TextInput
                style={[
                  styles.textInput,
                  currentValid && styles.textInputActive,
                ]}
                placeholder={config.placeholder}
                placeholderTextColor={C.mutedFg}
                value={values[currentStep]}
                onChangeText={t => setValues(v => ({ ...v, [currentStep]: t }))}
                onSubmitEditing={goNext}
                keyboardType={config.type === 'number' ? 'numeric' : 'default'}
                returnKeyType="next"
                autoFocus
              />
              {currentValid ? (
                <Animated.View entering={FadeIn.duration(200)} exiting={FadeOut.duration(150)} style={styles.inputCheck}>
                  <CheckIcon size={13} />
                </Animated.View>
              ) : null}
            </View>
          )}
        </Animated.View>
      </View>

      {/* Nav buttons */}
      <View style={[styles.navRow, step > 0 && styles.navRowDouble]}>
        {step > 0 && (
          <TouchableOpacity onPress={goBack} style={styles.secondaryBtn} activeOpacity={0.85}>
            <Text style={styles.secondaryBtnText}>Back</Text>
          </TouchableOpacity>
        )}

        {step < STEPS.length - 1 ? (
          <TouchableOpacity
            onPress={goNext}
            disabled={!currentValid}
            style={[styles.primaryBtn, !currentValid && styles.primaryBtnDisabled]}
            activeOpacity={currentValid ? 0.85 : 1}
          >
            <Text style={[styles.primaryBtnText, !currentValid && styles.primaryBtnTextDisabled]}>Next</Text>
          </TouchableOpacity>
        ) : (
          <TouchableOpacity
            onPress={handleComplete}
            disabled={!allFilled}
            style={[styles.primaryBtn, !allFilled && styles.primaryBtnDisabled, allFilled && styles.primaryBtnGlow]}
            activeOpacity={allFilled ? 0.85 : 1}
          >
            <Text style={[styles.primaryBtnText, !allFilled && styles.primaryBtnTextDisabled]}>Complete</Text>
          </TouchableOpacity>
        )}
      </View>
    </View>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  fullScreen: {
    flex: 1,
    backgroundColor: C.bg,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 24,
    overflow: 'hidden',
  },
  brand: {
    marginBottom: 40,
    alignItems: 'center',
  },
  gradientTitle: {
    fontSize: 30,
    fontWeight: '700',
    letterSpacing: -0.5,
    // React Native can't do inline gradient text; use primary colour as visual anchor
    color: C.primary,
    marginBottom: 4,
  },
  subLabel: {
    fontSize: 13,
    color: C.mutedFg,
    marginTop: 4,
  },
  dotsRow: {
    flexDirection: 'row',
    gap: 8,
    marginBottom: 32,
  },
  dot: {
    height: 8,
    borderRadius: 4,
  },
  cardArea: {
    width: '100%',
    maxWidth: 380,
    position: 'relative',
    marginBottom: 8,
    overflow: 'visible',
  },
  stepCard: {
    gap: 16,
  },
  stepCount: {
    fontSize: 12,
    color: C.mutedFg,
  },
  stepHeading: {
    fontSize: 20,
    fontWeight: '600',
    color: C.fg,
    lineHeight: 28,
  },
  stepBody: {
    fontSize: 14,
    color: C.mutedFg,
    lineHeight: 22,
  },
  disclaimerHeader: {
    marginBottom: 40,
    alignItems: 'center',
  },
  inputWrapper: {
    position: 'relative',
  },
  textInput: {
    width: '100%',
    backgroundColor: C.muted,
    borderWidth: 1,
    borderColor: C.border,
    borderRadius: 16,
    paddingHorizontal: 20,
    paddingVertical: 16,
    fontSize: 16,
    color: C.fg,
    paddingRight: 52,
  },
  textInputActive: {
    borderColor: C.primary,
  },
  inputCheck: {
    position: 'absolute',
    right: 16,
    top: '50%',
    marginTop: -12,
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: C.primary,
    alignItems: 'center',
    justifyContent: 'center',
  },
  selectContainer: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 4,
  },
  selectActive: {},
  selectOption: {
    flex: 1,
    paddingVertical: 14,
    borderRadius: 16,
    alignItems: 'center',
    backgroundColor: C.muted,
    borderWidth: 1,
    borderColor: C.border,
  },
  selectOptionActive: {
    backgroundColor: 'rgba(251,146,60,0.14)',
    borderColor: C.primary,
  },
  selectOptionText: {
    fontSize: 15,
    color: C.mutedFg,
    fontWeight: '500',
  },
  selectOptionTextActive: {
    color: C.primary,
    fontWeight: '600',
  },
  checkBadgeRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginTop: 8,
    paddingHorizontal: 4,
  },
  checkBadge: {
    width: 18,
    height: 18,
    borderRadius: 9,
    backgroundColor: C.primary,
    alignItems: 'center',
    justifyContent: 'center',
  },
  checkLabel: {
    fontSize: 12,
    color: C.primary,
  },
  wheelContainer: {
    flexDirection: 'row',
    backgroundColor: '#1A1F27',
    borderWidth: 1.5,
    borderColor: C.primary,
    borderRadius: 16,
    overflow: 'hidden',
    paddingHorizontal: 12,
  },
  wheelCol: {
    flex: 1,
    position: 'relative',
  },
  wheelFadeTop: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: 32,
    zIndex: 10,
    // gradient fades not possible in RN without LinearGradient — use semi-opaque overlay
    backgroundColor: 'rgba(26,31,39,0.75)',
    pointerEvents: 'none',
  },
  wheelFadeBot: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    height: 32,
    zIndex: 10,
    backgroundColor: 'rgba(26,31,39,0.75)',
    pointerEvents: 'none',
  },
  wheelHighlight: {
    position: 'absolute',
    top: ITEM_H,
    left: 0,
    right: 0,
    height: ITEM_H,
    zIndex: 5,
    backgroundColor: 'rgba(251,146,60,0.12)',
    borderWidth: 1,
    borderColor: 'rgba(251,146,60,0.35)',
    borderRadius: 8,
    pointerEvents: 'none',
  },
  wheelDivider: {
    width: 1,
    marginVertical: 8,
    backgroundColor: 'rgba(255,255,255,0.12)',
  },
  wheelItem: {
    height: ITEM_H,
    alignItems: 'center',
    justifyContent: 'center',
  },
  wheelItemText: {
    fontSize: 15,
    color: C.mutedFg,
    fontWeight: '400',
  },
  wheelItemSelected: {
    color: C.primary,
    fontWeight: '600',
  },
  navRow: {
    width: '100%',
    maxWidth: 380,
    marginTop: 32,
    flexDirection: 'row',
  },
  navRowDouble: {
    gap: 12,
  },
  primaryBtn: {
    flex: 1,
    backgroundColor: C.primary,
    borderRadius: 16,
    paddingVertical: 16,
    alignItems: 'center',
  },
  primaryBtnDisabled: {
    backgroundColor: 'rgba(251,146,60,0.07)',
    borderWidth: 1,
    borderColor: 'rgba(251,146,60,0.22)',
  },
  primaryBtnGlow: {
    shadowColor: C.primary,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.55,
    shadowRadius: 12,
    elevation: 8,
  },
  primaryBtnText: {
    fontSize: 15,
    fontWeight: '600',
    color: '#000000',
  },
  primaryBtnTextDisabled: {
    color: 'rgba(255,255,255,0.3)',
  },
  secondaryBtn: {
    flex: 1,
    backgroundColor: '#1A1F27',
    borderRadius: 16,
    paddingVertical: 16,
    alignItems: 'center',
  },
  secondaryBtnText: {
    fontSize: 15,
    fontWeight: '500',
    color: C.fg,
  },
});
