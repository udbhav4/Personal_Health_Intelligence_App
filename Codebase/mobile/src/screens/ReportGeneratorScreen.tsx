/**
 * screens/ReportGeneratorScreen.tsx — Doctor Report generation UI.
 *
 * Modal-style full-screen view that:
 *   1. takes a free-text symptom from the user
 *   2. runs gatherReportData → generateReportNarrative → buildReportHtml
 *   3. prints to a temp PDF, copies it into FileSystem.documentDirectory/reports/
 *   4. records the report in the doctor_reports SQLite table
 *   5. offers a "Share PDF" action via expo-sharing
 *
 * Guards against double-generation by checking isAgentBusy() — if the chat
 * agent is mid-turn, the user is asked to wait rather than racing the model.
 */

import React, { useState, useCallback, useEffect, useRef } from 'react';
import {
  View, Text, TextInput, TouchableOpacity, ScrollView, StyleSheet,
  ActivityIndicator, StatusBar, Platform,
} from 'react-native';

import * as Print      from 'expo-print';
import * as FileSystem from 'expo-file-system/legacy';
import * as Sharing    from 'expo-sharing';

import { useAppContext }         from '../core/AppContext';
import { isAgentBusy }           from '../core/agent';
import { writeDoctorReport }     from '../core/db';
import { gatherReportData }      from '../core/reportDataCollector';
import { generateReportNarrative } from '../core/reportAgent';
import { buildReportHtml }       from '../core/reportHtmlBuilder';

// ── Colours (match Chat/Profile palettes) ─────────────────────────────────────

const C = {
  bg:       '#0B0E14',
  card:     '#13171F',
  muted:    '#1A1F27',
  border:   '#22262D',
  fg:       '#FFFFFF',
  mutedFg:  '#7A8494',
  primary:  '#FB923C',
  secondary:'#2D7A7F',
  danger:   '#dc2626',
  success:  '#10b981',
};

type Status = 'idle' | 'generating' | 'done' | 'error';

interface Props {
  onClose:         () => void;
  initialSymptom?: string;
}

// ── Main ─────────────────────────────────────────────────────────────────────

export default function ReportGeneratorScreen({ onClose, initialSymptom }: Props) {
  const { db, profile, beliefs } = useAppContext();

  const [symptom,      setSymptom]      = useState(initialSymptom ?? '');
  const [status,       setStatus]       = useState<Status>('idle');
  const [progressText, setProgressText] = useState('');
  const [savedUri,     setSavedUri]     = useState<string | null>(null);
  const [errorMsg,     setErrorMsg]     = useState('');

  const handleGenerate = useCallback(async () => {
    if (!symptom.trim() || status === 'generating') return;
    if (!db) {
      setErrorMsg('Database is not ready yet. Please try again in a moment.');
      setStatus('error');
      return;
    }
    if (isAgentBusy()) {
      setErrorMsg('Please finish your current conversation first.');
      setStatus('error');
      return;
    }

    setErrorMsg('');
    setStatus('generating');
    setProgressText('Collecting your data...');

    try {
      // 1. Gather data
      const data = await gatherReportData(db, symptom.trim(), profile, beliefs);

      // 2. Generate narrative + amber bullets
      const { narrative, amberBullets } = await generateReportNarrative(db, data, (msg) => setProgressText(msg));

      // 3. Build HTML
      setProgressText('Generating PDF...');
      const html = buildReportHtml(narrative, data, amberBullets);

      // 4. Print to temp file
      const { uri: tempUri } = await Print.printToFileAsync({ html });

      // 5. Permanent copy
      const reportsDir = FileSystem.documentDirectory + 'reports/';
      await FileSystem.makeDirectoryAsync(reportsDir, { intermediates: true });
      const filename  = `report-${Date.now()}.pdf`;
      const savedPath = reportsDir + filename;
      await FileSystem.copyAsync({ from: tempUri, to: savedPath });

      // 6. Write to DB
      writeDoctorReport(db, symptom.trim(), savedPath, new Date().toISOString());

      // 7. Done
      setSavedUri(savedPath);
      setStatus('done');
      setProgressText('');
    } catch (e) {
      setErrorMsg(e instanceof Error ? e.message : String(e));
      setStatus('error');
    }
  }, [symptom, status, db, profile, beliefs]);

  const handleShare = useCallback(async () => {
    if (!savedUri) return;
    try {
      await Sharing.shareAsync(savedUri, {
        mimeType:    'application/pdf',
        dialogTitle: 'Share with your doctor',
      });
    } catch (e) {
      setErrorMsg(e instanceof Error ? e.message : String(e));
      setStatus('error');
    }
  }, [savedUri]);

  const handleReset = useCallback(() => {
    setStatus('idle');
    setErrorMsg('');
    setProgressText('');
    setSavedUri(null);
  }, []);

  const autoStarted = useRef(false);
  useEffect(() => {
    if (initialSymptom?.trim() && !autoStarted.current) {
      autoStarted.current = true;
      handleGenerate();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // run once on mount — symptom state already holds initialSymptom

  const busyGuard       = isAgentBusy() && status === 'idle';
  const generateDisabled = !symptom.trim() || status === 'generating';

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Generate Report</Text>
        <TouchableOpacity
          onPress={onClose}
          style={styles.closeBtn}
          activeOpacity={0.8}
        >
          <Text style={styles.closeBtnText}>✕</Text>
        </TouchableOpacity>
      </View>
      <View style={styles.divider} />

      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
        keyboardShouldPersistTaps="handled"
      >
        {/* Intro */}
        <Text style={styles.intro}>
          Describe your main health concern. The app will surface passive patterns
          you may not have mentioned and prepare a PDF you can share with your doctor.
        </Text>

        {/* Symptom input */}
        <Text style={styles.label}>Your main health concern</Text>
        <TextInput
          style={styles.textInput}
          value={symptom}
          onChangeText={setSymptom}
          placeholder="e.g., I've been having back pain for 3 weeks"
          placeholderTextColor={C.mutedFg}
          multiline
          editable={status === 'idle' || status === 'error'}
        />

        {/* Idle / Error states: Generate button (or busy guard) */}
        {(status === 'idle' || status === 'error') && (
          <>
            {busyGuard ? (
              <View style={styles.warningBox}>
                <Text style={styles.warningText}>
                  Active conversation in progress — please wait
                </Text>
              </View>
            ) : (
              <TouchableOpacity
                onPress={handleGenerate}
                disabled={generateDisabled}
                style={[styles.primaryBtn, generateDisabled && styles.primaryBtnDisabled]}
                activeOpacity={0.85}
              >
                <Text style={styles.primaryBtnText}>Generate Report</Text>
              </TouchableOpacity>
            )}
          </>
        )}

        {/* Error message */}
        {status === 'error' && errorMsg ? (
          <View style={styles.errorBox}>
            <Text style={styles.errorText}>{errorMsg}</Text>
            <TouchableOpacity
              onPress={handleReset}
              style={styles.secondaryBtn}
              activeOpacity={0.85}
            >
              <Text style={styles.secondaryBtnText}>Try Again</Text>
            </TouchableOpacity>
          </View>
        ) : null}

        {/* Generating state */}
        {status === 'generating' && (
          <View style={styles.progressCard}>
            <ActivityIndicator size="small" color={C.primary} />
            <Text style={styles.progressText}>{progressText || 'Working...'}</Text>
            <Text style={styles.progressHint}>
              This may take a minute or two. Please keep the app open.
            </Text>
          </View>
        )}

        {/* Done state */}
        {status === 'done' && (
          <View style={styles.successCard}>
            <Text style={styles.successTitle}>Report ready</Text>
            <Text style={styles.successText}>
              Your report has been saved to your device. Share it with your doctor
              when you're ready.
            </Text>
            <TouchableOpacity
              onPress={handleShare}
              style={styles.primaryBtn}
              activeOpacity={0.85}
            >
              <Text style={styles.primaryBtnText}>Share PDF</Text>
            </TouchableOpacity>
            <TouchableOpacity
              onPress={onClose}
              style={styles.secondaryBtn}
              activeOpacity={0.85}
            >
              <Text style={styles.secondaryBtnText}>Done</Text>
            </TouchableOpacity>
          </View>
        )}
      </ScrollView>
    </View>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: C.bg,
    paddingTop: Platform.OS === 'android' ? (StatusBar.currentHeight ?? 24) + 8 : 16,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingBottom: 12,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: C.fg,
  },
  closeBtn: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: C.muted,
    alignItems: 'center',
    justifyContent: 'center',
  },
  closeBtnText: {
    color: C.fg,
    fontSize: 16,
    fontWeight: '600',
  },
  divider: {
    height: 1,
    backgroundColor: 'rgba(251,146,60,0.20)',
  },
  scroll: {
    flex: 1,
  },
  scrollContent: {
    padding: 16,
    gap: 12,
  },
  intro: {
    fontSize: 13,
    lineHeight: 20,
    color: C.mutedFg,
    marginBottom: 8,
  },
  label: {
    fontSize: 11,
    color: C.mutedFg,
    letterSpacing: 0.6,
    textTransform: 'uppercase',
    marginBottom: 4,
  },
  textInput: {
    backgroundColor: C.muted,
    borderWidth: 1,
    borderColor: C.border,
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 10,
    color: C.fg,
    fontSize: 14,
    minHeight: 96,
    textAlignVertical: 'top',
  },
  primaryBtn: {
    backgroundColor: C.primary,
    borderRadius: 10,
    paddingVertical: 12,
    alignItems: 'center',
    marginTop: 8,
  },
  primaryBtnDisabled: {
    backgroundColor: 'rgba(251,146,60,0.30)',
  },
  primaryBtnText: {
    color: '#000',
    fontWeight: '700',
    fontSize: 14,
  },
  secondaryBtn: {
    backgroundColor: C.muted,
    borderRadius: 10,
    paddingVertical: 12,
    alignItems: 'center',
    marginTop: 8,
    borderWidth: 1,
    borderColor: C.border,
  },
  secondaryBtnText: {
    color: C.fg,
    fontWeight: '600',
    fontSize: 14,
  },
  warningBox: {
    backgroundColor: 'rgba(251,146,60,0.10)',
    borderWidth: 1,
    borderColor: 'rgba(251,146,60,0.35)',
    borderRadius: 10,
    paddingVertical: 14,
    paddingHorizontal: 16,
    alignItems: 'center',
    marginTop: 8,
  },
  warningText: {
    color: C.primary,
    fontSize: 13,
    fontWeight: '600',
    textAlign: 'center',
  },
  errorBox: {
    backgroundColor: 'rgba(220,38,38,0.10)',
    borderWidth: 1,
    borderColor: 'rgba(220,38,38,0.35)',
    borderRadius: 10,
    padding: 14,
    marginTop: 8,
    gap: 8,
  },
  errorText: {
    color: C.danger,
    fontSize: 13,
    lineHeight: 18,
  },
  progressCard: {
    backgroundColor: C.card,
    borderWidth: 1,
    borderColor: C.border,
    borderRadius: 12,
    padding: 20,
    alignItems: 'center',
    gap: 10,
    marginTop: 8,
  },
  progressText: {
    color: C.fg,
    fontSize: 14,
    fontWeight: '600',
    textAlign: 'center',
  },
  progressHint: {
    color: C.mutedFg,
    fontSize: 12,
    textAlign: 'center',
    lineHeight: 18,
  },
  successCard: {
    backgroundColor: C.card,
    borderWidth: 1,
    borderColor: 'rgba(16,185,129,0.35)',
    borderRadius: 12,
    padding: 16,
    gap: 8,
    marginTop: 8,
  },
  successTitle: {
    color: C.success,
    fontSize: 15,
    fontWeight: '700',
  },
  successText: {
    color: C.fg,
    fontSize: 13,
    lineHeight: 19,
    marginBottom: 4,
  },
});
