import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  View, Text, StyleSheet,
  Animated as RNAnimated,
} from 'react-native';
import Animated, { FadeInDown } from 'react-native-reanimated';
import GliimrText from '../components/GliimrText';
import * as FileSystem from 'expo-file-system/legacy';

import type { ModelPaths } from '../core/initModels';

// ── Palette ───────────────────────────────────────────────────────────────────

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

// ── Constants ─────────────────────────────────────────────────────────────────

export const MODELS_DIR = `${FileSystem.documentDirectory}models/`;

// Verify these HuggingFace resolve URLs match the hosted filenames before release.
const MODEL_DEFS = [
  {
    key:      'nlu'   as const,
    name:     'Model 1',
    role:     'Understands what you said',
    sizeMb:   808,
    filename: 'Llama-3.2-1B-Instruct-Q4_K_M.gguf',
    url:      'https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf',
  },
  {
    key:      'embed' as const,
    name:     'Model 2',
    role:     'Remembers it',
    sizeMb:   90,
    filename: 'nomic-embed-text-v1.5.Q4_K_M.gguf',
    url:      'https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf',
  },
  {
    key:      'agent' as const,
    name:     'Model 3',
    role:     'Your health companion',
    sizeMb:   3183,
    filename: 'gemma-4-E2B-it-Q4_K_M.gguf',
    url:      'https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf',
  },
] as const;

type ModelKey    = typeof MODEL_DEFS[number]['key'];
type ModelStatus = 'pending' | 'downloading' | 'done' | 'error';

interface ModelState {
  status:   ModelStatus;
  progress: number;
  error?:   string;
}

// ── Screen ────────────────────────────────────────────────────────────────────

interface Props {
  onComplete: (paths: ModelPaths) => void;
}

export default function ModelDownloadScreen({ onComplete }: Props) {
  const [states, setStates] = useState<Record<ModelKey, ModelState>>({
    nlu:   { status: 'pending', progress: 0 },
    embed: { status: 'pending', progress: 0 },
    agent: { status: 'pending', progress: 0 },
  });
  const [allDone, setAllDone] = useState(false);

  const progressAnims = useRef<Record<ModelKey, RNAnimated.Value>>({
    nlu:   new RNAnimated.Value(0),
    embed: new RNAnimated.Value(0),
    agent: new RNAnimated.Value(0),
  }).current;

  const patchState = useCallback((key: ModelKey, patch: Partial<ModelState>) => {
    setStates(prev => ({ ...prev, [key]: { ...prev[key], ...patch } }));
  }, []);

  useEffect(() => {
    let alive = true;

    async function run() {
      try {
      await FileSystem.makeDirectoryAsync(MODELS_DIR, { intermediates: true }).catch(() => {});

      const paths: Partial<Record<ModelKey, string>> = {};

      for (const m of MODEL_DEFS) {
        if (!alive) return;
        const dest = `${MODELS_DIR}${m.filename}`;

        const info = await FileSystem.getInfoAsync(dest, { size: true });
        const minBytes = m.sizeMb * 1024 * 1024 * 0.95;
        if (info.exists && (info as any).size >= minBytes) {
          if (alive) patchState(m.key, { status: 'done', progress: 1 });
          RNAnimated.timing(progressAnims[m.key], {
            toValue: 1, duration: 300, useNativeDriver: false,
          }).start();
          paths[m.key] = dest;
          continue;
        }
        // File missing or truncated — delete stale copy before re-downloading
        if (info.exists) await FileSystem.deleteAsync(dest, { idempotent: true });

        if (alive) patchState(m.key, { status: 'downloading' });

        try {
          const dl = FileSystem.createDownloadResumable(
            m.url,
            dest,
            {},
            ({ totalBytesWritten, totalBytesExpectedToWrite }) => {
              if (!alive) return;
              const ratio = totalBytesExpectedToWrite > 0
                ? totalBytesWritten / totalBytesExpectedToWrite
                : 0;
              patchState(m.key, { progress: ratio });
              RNAnimated.timing(progressAnims[m.key], {
                toValue: ratio, duration: 80, useNativeDriver: false,
              }).start();
            },
          );

          const result = await dl.downloadAsync();
          if (!alive) return;
          if (!result?.uri) throw new Error('No URI returned');

          // Verify the downloaded file is actually a model (not a 404 HTML page)
          const dlInfo = await FileSystem.getInfoAsync(dest, { size: true } as any);
          const dlBytes = (dlInfo as any).size ?? 0;
          const halfExpected = m.sizeMb * 1024 * 1024 * 0.5;
          if (dlBytes < halfExpected) {
            await FileSystem.deleteAsync(dest, { idempotent: true });
            throw new Error(
              `Downloaded only ${(dlBytes / 1024 / 1024).toFixed(1)} MB — expected ~${m.sizeMb} MB. Check the URL or your connection.`,
            );
          }

          patchState(m.key, { status: 'done', progress: 1 });
          RNAnimated.timing(progressAnims[m.key], {
            toValue: 1, duration: 300, useNativeDriver: false,
          }).start();
          paths[m.key] = dest;
        } catch (e) {
          if (!alive) return;
          patchState(m.key, {
            status: 'error',
            error: e instanceof Error ? e.message : String(e),
          });
          return;
        }
      }

      if (alive && paths.nlu && paths.embed && paths.agent) {
        setAllDone(true);
        onComplete({ nlu: paths.nlu, embed: paths.embed, agent: paths.agent });
      }
      } catch (e) {
        if (alive) patchState('nlu', { status: 'error', error: `Setup error: ${String(e)}` });
      }
    }

    run();
    return () => { alive = false; };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const hasError = MODEL_DEFS.some(m => states[m.key].status === 'error');

  return (
    <View style={styles.root}>
      {/* ── Header ─────────────────────────────────────────────────── */}
      <Animated.View entering={FadeInDown.duration(500)} style={styles.header}>
        <GliimrText style={styles.brand} />
        <Text style={styles.title}>Setting up AI</Text>
        <Text style={styles.subtitle}>
          Three on-device models are downloading.{'\n'}This only happens once.
        </Text>
      </Animated.View>

      {/* ── Model cards ────────────────────────────────────────────── */}
      <View style={styles.cards}>
        {MODEL_DEFS.map((m, i) => (
          <ModelCard
            key={m.key}
            def={m}
            state={states[m.key]}
            progressAnim={progressAnims[m.key]}
            delay={160 + i * 120}
          />
        ))}
      </View>

      {/* ── Footer ─────────────────────────────────────────────────── */}
      <Animated.View entering={FadeInDown.duration(500).delay(600)} style={styles.footer}>
        {hasError ? (
          <Text style={styles.footerError}>
            Download failed. Check your connection and restart the app.
          </Text>
        ) : allDone ? (
          <Text style={styles.footerDone}>All models ready. Starting app…</Text>
        ) : (
          <Text style={styles.footerHint}>Keep the app open · Wi-Fi recommended</Text>
        )}
      </Animated.View>
    </View>
  );
}

// ── ModelCard ─────────────────────────────────────────────────────────────────

interface CardProps {
  def:          typeof MODEL_DEFS[number];
  state:        ModelState;
  progressAnim: RNAnimated.Value;
  delay:        number;
}

function ModelCard({ def, state, progressAnim, delay }: CardProps) {
  const isDone   = state.status === 'done';
  const isActive = state.status === 'downloading';
  const isError  = state.status === 'error';
  const pct      = Math.round(state.progress * 100);
  const sizeLabel = def.sizeMb >= 1000
    ? `${(def.sizeMb / 1000).toFixed(1)} GB`
    : `${def.sizeMb} MB`;

  return (
    <Animated.View
      entering={FadeInDown.duration(450).delay(delay)}
      style={[
        styles.card,
        isActive && styles.cardActive,
        isDone   && styles.cardDone,
        isError  && styles.cardError,
      ]}
    >
      {/* Top row: name + badge */}
      <View style={styles.cardTop}>
        <View style={styles.cardInfo}>
          <Text style={[
            styles.modelName,
            isActive && styles.modelNameActive,
            isDone   && styles.modelNameDone,
          ]}>
            {def.name}
          </Text>
          <Text style={styles.modelRole}>{def.role}</Text>
        </View>

        <View style={styles.badgeWrap}>
          {isDone ? (
            <View style={styles.checkBadge}>
              <Text style={styles.checkText}>✓</Text>
            </View>
          ) : (
            <Text style={[styles.sizeText, isActive && styles.sizeTextActive]}>
              {sizeLabel}
            </Text>
          )}
        </View>
      </View>

      {/* Progress bar */}
      <View style={styles.track}>
        <RNAnimated.View
          style={[
            styles.fill,
            isDone  && styles.fillDone,
            isError && styles.fillError,
            {
              width: progressAnim.interpolate({
                inputRange:  [0, 1],
                outputRange: ['0%', '100%'],
              }),
            },
          ]}
        />
      </View>

      {/* Status label */}
      <Text style={[
        styles.statusText,
        isActive && styles.statusActive,
        isDone   && styles.statusDone,
        isError  && styles.statusError,
      ]}>
        {isDone    ? 'Complete'
        : isError  ? (state.error ?? 'Error')
        : isActive ? `Downloading  ${pct}%`
        : 'Waiting'}
      </Text>
    </Animated.View>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: C.bg,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 24,
    gap: 32,
  },

  // Header
  header: {
    alignItems: 'center',
    gap: 8,
  },
  brand: {
    fontSize: 34,
    fontWeight: '800',
    color: C.primary,
    letterSpacing: -1,
    marginBottom: 4,
  },
  title: {
    fontSize: 22,
    fontWeight: '700',
    color: C.fg,
    letterSpacing: -0.5,
  },
  subtitle: {
    fontSize: 14,
    color: C.mutedFg,
    textAlign: 'center',
    lineHeight: 22,
    marginTop: 4,
  },

  // Cards list
  cards: {
    width: '100%',
    maxWidth: 380,
    gap: 12,
  },

  // Card base
  card: {
    backgroundColor: C.card,
    borderRadius: 20,
    padding: 20,
    borderWidth: 1,
    borderColor: C.border,
    gap: 12,
  },
  cardActive: {
    borderColor: C.primary,
    shadowColor: C.primary,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.3,
    shadowRadius: 14,
    elevation: 8,
  },
  cardDone: {
    borderColor: C.secondary,
  },
  cardError: {
    borderColor: '#FF4D4D',
  },

  // Card top row
  cardTop: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
  },
  cardInfo: {
    flex: 1,
    gap: 3,
  },
  modelName: {
    fontSize: 16,
    fontWeight: '600',
    color: C.mutedFg,
  },
  modelNameActive: {
    color: C.primary,
  },
  modelNameDone: {
    color: C.fg,
  },
  modelRole: {
    fontSize: 13,
    color: C.mutedFg,
  },
  badgeWrap: {
    marginLeft: 12,
    alignItems: 'flex-end',
    justifyContent: 'center',
  },
  checkBadge: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: C.secondary,
    alignItems: 'center',
    justifyContent: 'center',
  },
  checkText: {
    fontSize: 14,
    fontWeight: '700',
    color: '#FFFFFF',
  },
  sizeText: {
    fontSize: 12,
    color: C.mutedFg,
    fontWeight: '500',
  },
  sizeTextActive: {
    color: C.primary,
  },

  // Progress bar
  track: {
    height: 6,
    backgroundColor: C.muted,
    borderRadius: 3,
    overflow: 'hidden',
  },
  fill: {
    height: '100%',
    backgroundColor: C.primary,
    borderRadius: 3,
  },
  fillDone: {
    backgroundColor: C.secondary,
  },
  fillError: {
    backgroundColor: '#FF4D4D',
  },

  // Status text
  statusText: {
    fontSize: 12,
    color: C.mutedFg,
    fontWeight: '500',
  },
  statusActive: {
    color: C.primary,
  },
  statusDone: {
    color: C.secondary,
  },
  statusError: {
    color: '#FF4D4D',
  },

  // Footer
  footer: {
    alignItems: 'center',
    paddingHorizontal: 16,
  },
  footerHint: {
    fontSize: 13,
    color: C.mutedFg,
    textAlign: 'center',
  },
  footerDone: {
    fontSize: 14,
    color: C.secondary,
    fontWeight: '600',
    textAlign: 'center',
  },
  footerError: {
    fontSize: 13,
    color: '#FF4D4D',
    textAlign: 'center',
    lineHeight: 20,
  },
});
