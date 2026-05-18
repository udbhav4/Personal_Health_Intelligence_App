import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  View, Text, TextInput, TouchableOpacity, ScrollView,
  StyleSheet,
} from 'react-native';
import Animated, {
  FadeIn, FadeOut, useAnimatedStyle, withTiming, Layout,
} from 'react-native-reanimated';
import Svg, { Path } from 'react-native-svg';
import * as Sharing from 'expo-sharing';
import type { UserProfile } from '../core/AppContext';
import { useAppContext } from '../core/AppContext';
import { getDoctorReports, type DoctorReport } from '../core/db';

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
};

// ── Icon components ───────────────────────────────────────────────────────────

function PencilIcon() {
  return (
    <Svg width={12} height={12} viewBox="0 0 24 24" fill="none">
      <Path
        d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"
        stroke={C.primary}
        strokeWidth={2}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </Svg>
  );
}
function CheckIcon() {
  return (
    <Svg width={12} height={12} viewBox="0 0 24 24" fill="none">
      <Path d="M20 6L9 17L4 12" stroke="#000" strokeWidth={3} strokeLinecap="round" strokeLinejoin="round" />
    </Svg>
  );
}
function XIcon() {
  return (
    <Svg width={12} height={12} viewBox="0 0 24 24" fill="none">
      <Path d="M18 6L6 18M6 6l12 12" stroke={C.mutedFg} strokeWidth={2} strokeLinecap="round" />
    </Svg>
  );
}
function ShareIcon() {
  return (
    <Svg width={14} height={14} viewBox="0 0 24 24" fill="none">
      <Path
        d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8M16 6l-4-4-4 4M12 2v13"
        stroke={C.primary}
        strokeWidth={2}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </Svg>
  );
}

// ── Wheel picker (reused from onboarding) ─────────────────────────────────────

const MONTHS_W = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
const DAYS_W   = Array.from({ length: 31 }, (_, i) => String(i + 1).padStart(2, '0'));
const YEARS_W  = Array.from({ length: 100 }, (_, i) => String(new Date().getFullYear() - i));

const ITEM_H = 34;

function WheelCol({ items, value, onChange }: { items: string[]; value: string; onChange: (v: string) => void }) {
  const scrollRef = useRef<ScrollView>(null);
  const idx = items.indexOf(value);

  useEffect(() => {
    setTimeout(() => scrollRef.current?.scrollTo({ y: Math.max(0, idx) * ITEM_H, animated: false }), 50);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <View style={wStyles.col}>
      <View style={wStyles.fadeTop} pointerEvents="none" />
      <View style={wStyles.fadeBot} pointerEvents="none" />
      <View style={wStyles.highlight} pointerEvents="none" />
      <ScrollView
        ref={scrollRef}
        showsVerticalScrollIndicator={false}
        snapToInterval={ITEM_H}
        decelerationRate="fast"
        onMomentumScrollEnd={e => {
          const i = Math.round(e.nativeEvent.contentOffset.y / ITEM_H);
          const c = Math.max(0, Math.min(items.length - 1, i));
          if (items[c] !== value) onChange(items[c]);
        }}
        contentContainerStyle={{ paddingVertical: ITEM_H }}
        style={{ height: ITEM_H * 3 }}
      >
        {items.map(item => (
          <TouchableOpacity
            key={item}
            onPress={() => {
              onChange(item);
              scrollRef.current?.scrollTo({ y: items.indexOf(item) * ITEM_H, animated: true });
            }}
          >
            <View style={{ height: ITEM_H, alignItems: 'center', justifyContent: 'center' }}>
              <Text style={[wStyles.itemText, item === value && wStyles.itemSelected]}>{item}</Text>
            </View>
          </TouchableOpacity>
        ))}
      </ScrollView>
    </View>
  );
}

function DateWheelInline({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const parts = (value || '01-Jan-2000').split('-');
  const [day,   setDay]   = useState(parts[0] || '01');
  const [month, setMonth] = useState(parts[1] || 'Jan');
  const [year,  setYear]  = useState(parts[2] || '2000');

  const emit = (d: string, m: string, y: string) => {
    setDay(d); setMonth(m); setYear(y); onChange(`${d}-${m}-${y}`);
  };

  return (
    <View style={wStyles.container}>
      <WheelCol items={DAYS_W}   value={day}   onChange={d => emit(d, month, year)} />
      <View style={wStyles.divider} />
      <WheelCol items={MONTHS_W} value={month} onChange={m => emit(day, m, year)} />
      <View style={wStyles.divider} />
      <WheelCol items={YEARS_W}  value={year}  onChange={y => emit(day, month, y)} />
    </View>
  );
}

const wStyles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    backgroundColor: '#1A1F27',
    borderWidth: 1.5,
    borderColor: C.primary,
    borderRadius: 12,
    overflow: 'hidden',
    paddingHorizontal: 8,
  },
  col: {
    flex: 1,
    position: 'relative',
  },
  fadeTop: {
    position: 'absolute',
    top: 0, left: 0, right: 0,
    height: 24,
    zIndex: 10,
    backgroundColor: 'rgba(26,31,39,0.75)',
  },
  fadeBot: {
    position: 'absolute',
    bottom: 0, left: 0, right: 0,
    height: 24,
    zIndex: 10,
    backgroundColor: 'rgba(26,31,39,0.75)',
  },
  highlight: {
    position: 'absolute',
    top: ITEM_H,
    left: 0, right: 0,
    height: ITEM_H,
    zIndex: 5,
    backgroundColor: 'rgba(251,146,60,0.10)',
    borderWidth: 1,
    borderColor: 'rgba(251,146,60,0.30)',
    borderRadius: 6,
  },
  divider: {
    width: 1,
    marginVertical: 8,
    backgroundColor: 'rgba(255,255,255,0.10)',
  },
  itemText: {
    fontSize: 13,
    color: C.mutedFg,
    fontWeight: '400',
  },
  itemSelected: {
    color: C.primary,
    fontWeight: '600',
  },
});

// ── Field config ──────────────────────────────────────────────────────────────

type FieldKey = keyof UserProfile;

const FIELDS: { key: FieldKey; label: string; type: string }[] = [
  { key: 'name',      label: 'Name',      type: 'text' },
  { key: 'sex',       label: 'Sex',       type: 'select' },
  { key: 'birthdate', label: 'Birthdate', type: 'date-wheel' },
  { key: 'weight',    label: 'Weight',    type: 'text' },
  { key: 'height',    label: 'Height',    type: 'text' },
];

// ── Props ─────────────────────────────────────────────────────────────────────

interface Props {
  profile:  UserProfile;
  onSave:   (data: UserProfile) => Promise<void>;
  isActive?: boolean;
}

// ── Main ──────────────────────────────────────────────────────────────────────

export default function ProfileScreen({ profile, onSave, isActive }: Props) {
  const { db } = useAppContext();
  const [editing, setEditing] = useState<FieldKey | null>(null);
  const [draft,   setDraft]   = useState('');
  const [reports, setReports] = useState<DoctorReport[]>([]);

  const isDraftValid = draft.trim().length > 0;

  const startEdit = (key: FieldKey) => {
    if (editing !== null) return;
    setDraft(profile[key] || '');
    setEditing(key);
  };

  const confirmEdit = useCallback(async () => {
    if (!editing || !isDraftValid) return;
    await onSave({ ...profile, [editing]: draft });
    setEditing(null);
  }, [editing, isDraftValid, draft, profile, onSave]);

  const cancelEdit = () => { setEditing(null); setDraft(''); };

  useEffect(() => {
    if (!isActive && editing !== null) {
      setEditing(null);
      setDraft('');
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isActive]);

  useEffect(() => {
    if (isActive && db) setReports(getDoctorReports(db));
  }, [isActive, db]);

  return (
    <ScrollView
      style={pStyles.container}
      contentContainerStyle={pStyles.content}
      showsVerticalScrollIndicator={false}
    >
      {/* Header */}
      <Animated.View entering={FadeIn.duration(400)}>
        <Text style={pStyles.headerSub}>Your</Text>
        <Text style={pStyles.headerTitle}>Profile</Text>
      </Animated.View>

      <View style={pStyles.divider} />

      {/* Avatar */}
      <View style={pStyles.avatarRow}>
        <View style={pStyles.avatar}>
          <Text style={pStyles.avatarText}>
            {(profile.name || '?')[0].toUpperCase()}
          </Text>
        </View>
      </View>

      {/* Fields */}
      <View style={pStyles.fields}>
        {FIELDS.map(({ key, label, type }) => {
          const isEditing  = editing === key;
          const isBlocked  = editing !== null && !isEditing;
          const currentVal = profile[key] || '';

          return (
            <Animated.View
              key={key}
              layout={Layout.duration(180)}
              style={[
                pStyles.fieldRow,
                isEditing  && pStyles.fieldRowEditing,
                isBlocked  && pStyles.fieldRowBlocked,
              ]}
            >
              {/* Label + action row */}
              <View style={pStyles.fieldHeader}>
                <Text style={pStyles.fieldLabel}>{label.toUpperCase()}</Text>
                <View style={pStyles.fieldActions}>
                  {isEditing ? (
                    <>
                      <TouchableOpacity onPress={cancelEdit} style={pStyles.iconBtn} activeOpacity={0.8}>
                        <XIcon />
                      </TouchableOpacity>
                      <TouchableOpacity
                        onPress={isDraftValid ? confirmEdit : undefined}
                        style={[pStyles.iconBtn, isDraftValid ? pStyles.iconBtnConfirm : pStyles.iconBtnConfirmDisabled]}
                        activeOpacity={isDraftValid ? 0.8 : 1}
                      >
                        <CheckIcon />
                      </TouchableOpacity>
                    </>
                  ) : (
                    <TouchableOpacity
                      onPress={() => !isBlocked && startEdit(key)}
                      style={pStyles.iconBtnTransparent}
                      activeOpacity={isBlocked ? 1 : 0.8}
                    >
                      <PencilIcon />
                    </TouchableOpacity>
                  )}
                </View>
              </View>

              {/* Value / edit */}
              {isEditing ? (
                <Animated.View entering={FadeIn.duration(180)} exiting={FadeOut.duration(180)} style={{ marginTop: 8 }}>
                  {type === 'select' ? (
                    <View style={pStyles.selectRow}>
                      {(['Male', 'Female'] as const).map(opt => (
                        <TouchableOpacity
                          key={opt}
                          onPress={() => setDraft(opt)}
                          style={[pStyles.selectOpt, draft === opt && pStyles.selectOptActive]}
                        >
                          <Text style={[pStyles.selectOptText, draft === opt && pStyles.selectOptTextActive]}>{opt}</Text>
                        </TouchableOpacity>
                      ))}
                    </View>
                  ) : type === 'date-wheel' ? (
                    <DateWheelInline value={draft} onChange={setDraft} />
                  ) : (
                    <TextInput
                      style={pStyles.textInput}
                      value={draft}
                      onChangeText={setDraft}
                      keyboardType={type === 'number' ? 'numeric' : 'default'}
                      autoFocus
                      multiline
                    />
                  )}
                </Animated.View>
              ) : (
                <Animated.Text
                  entering={FadeIn.duration(150)}
                  style={[pStyles.fieldValue, !currentVal && pStyles.fieldValueEmpty]}
                >
                  {currentVal || 'Not set'}
                </Animated.Text>
              )}
            </Animated.View>
          );
        })}
      </View>

      {/* Reports */}
      <View style={pStyles.reportsSection}>
        <Text style={pStyles.reportsSectionTitle}>YOUR REPORTS</Text>
        {reports.length === 0 ? (
          <Text style={pStyles.reportsEmpty}>No reports yet. Tap Chat → Report to generate one.</Text>
        ) : (
          reports.map(report => (
            <View key={report.id} style={pStyles.reportCard}>
              <View style={pStyles.reportCardContent}>
                <Text style={pStyles.reportSymptom} numberOfLines={2}>
                  {report.symptom.length > 60 ? report.symptom.slice(0, 60) + '…' : report.symptom}
                </Text>
                <Text style={pStyles.reportDate}>
                  {new Date(report.created_at).toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })}
                </Text>
              </View>
              <TouchableOpacity
                style={pStyles.shareBtn}
                activeOpacity={0.8}
                onPress={() => Sharing.shareAsync(report.file_uri, { mimeType: 'application/pdf', dialogTitle: 'Share with your doctor' })}
              >
                <ShareIcon />
              </TouchableOpacity>
            </View>
          ))
        )}
      </View>
    </ScrollView>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────

const pStyles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: C.bg,
  },
  content: {
    paddingHorizontal: 16,
    paddingTop: 20,
    paddingBottom: 16,
    gap: 12,
  },
  headerSub: {
    fontSize: 11,
    color: C.mutedFg,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: C.fg,
  },
  divider: {
    height: 1,
    backgroundColor: 'rgba(251,146,60,0.20)',
  },
  avatarRow: {
    alignItems: 'center',
    paddingVertical: 4,
  },
  avatar: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: C.primary,
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatarText: {
    fontSize: 26,
    fontWeight: '700',
    color: C.fg,
  },
  fields: {
    gap: 8,
  },
  fieldRow: {
    borderRadius: 12,
    paddingHorizontal: 12,
    paddingVertical: 10,
    backgroundColor: '#1A1F27',
    borderWidth: 1,
    borderColor: 'transparent',
  },
  fieldRowEditing: {
    borderColor: 'rgba(251,146,60,0.5)',
  },
  fieldRowBlocked: {
    opacity: 0.45,
  },
  fieldHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  fieldLabel: {
    fontSize: 10,
    color: C.mutedFg,
    letterSpacing: 0.8,
    textTransform: 'uppercase',
  },
  fieldActions: {
    flexDirection: 'row',
    gap: 4,
    alignItems: 'center',
  },
  iconBtn: {
    width: 28,
    height: 28,
    borderRadius: 8,
    backgroundColor: C.border,
    alignItems: 'center',
    justifyContent: 'center',
  },
  iconBtnTransparent: {
    width: 28,
    height: 28,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
  },
  iconBtnConfirm: {
    backgroundColor: C.primary,
  },
  iconBtnConfirmDisabled: {
    backgroundColor: 'rgba(251,146,60,0.25)',
  },
  fieldValue: {
    marginTop: 4,
    fontSize: 14,
    fontWeight: '500',
    color: C.fg,
  },
  fieldValueEmpty: {
    color: C.mutedFg,
  },
  textInput: {
    backgroundColor: C.muted,
    borderWidth: 1,
    borderColor: C.primary,
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 8,
    color: C.fg,
    fontSize: 14,
    maxHeight: 46,
  },
  selectRow: {
    flexDirection: 'row',
    gap: 8,
  },
  selectOpt: {
    flex: 1,
    paddingVertical: 8,
    borderRadius: 8,
    alignItems: 'center',
    backgroundColor: C.muted,
    borderWidth: 1,
    borderColor: C.border,
  },
  selectOptActive: {
    backgroundColor: 'rgba(251,146,60,0.14)',
    borderColor: C.primary,
  },
  selectOptText: {
    fontSize: 13,
    color: C.mutedFg,
    fontWeight: '500',
  },
  selectOptTextActive: {
    color: C.primary,
    fontWeight: '600',
  },
  reportsSection: {
    marginTop: 8,
    gap: 8,
  },
  reportsSectionTitle: {
    fontSize: 10,
    color: C.mutedFg,
    letterSpacing: 0.8,
    textTransform: 'uppercase',
    marginBottom: 4,
  },
  reportsEmpty: {
    fontSize: 13,
    color: C.mutedFg,
    fontStyle: 'italic',
    paddingVertical: 8,
  },
  reportCard: {
    flexDirection: 'row',
    alignItems: 'center',
    borderRadius: 12,
    paddingHorizontal: 12,
    paddingVertical: 10,
    backgroundColor: '#1A1F27',
    borderWidth: 1,
    borderColor: '#22262D',
  },
  reportCardContent: {
    flex: 1,
    marginRight: 8,
  },
  reportSymptom: {
    fontSize: 13,
    fontWeight: '500',
    color: C.fg,
  },
  reportDate: {
    fontSize: 11,
    color: C.mutedFg,
    marginTop: 2,
  },
  shareBtn: {
    width: 32,
    height: 32,
    borderRadius: 8,
    backgroundColor: '#1A1F27',
    borderWidth: 1,
    borderColor: '#22262D',
    alignItems: 'center',
    justifyContent: 'center',
  },
});
