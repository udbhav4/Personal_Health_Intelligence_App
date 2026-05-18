import React, {
  useState, useRef, useEffect, useCallback,
} from 'react';
import {
  View, Text, TextInput, TouchableOpacity, ScrollView,
  StyleSheet, Dimensions, Modal, Pressable,
  KeyboardAvoidingView, Platform, StatusBar,
} from 'react-native';
import Animated, {
  FadeIn, FadeOut, SlideInDown, SlideOutDown,
  useSharedValue, useAnimatedStyle, withSpring, withTiming,
  withRepeat, withSequence,
} from 'react-native-reanimated';
import Svg, { Path, Defs, LinearGradient, Stop } from 'react-native-svg';
import GliimrText from '../components/GliimrText';
import type { UserProfile } from '../core/AppContext';
import { useAppContext } from '../core/AppContext';
import { startTurn, completeTurn, runJournalTurn } from '../core/agent';
import ReportGeneratorScreen from './ReportGeneratorScreen';
import { useBeliefRefresh } from '../hooks/useBeliefRefresh';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import type { DisplayQuestion, QuestionAnswer, AgentMode } from '../core/agent';

// ── Colours ───────────────────────────────────────────────────────────────────

const C = {
  bg:       '#0B0E14',
  card:     '#13171F',
  muted:    '#1A1F27',
  border:   '#22262D',
  elevated: '#1E232B',
  fg:       '#FFFFFF',
  mutedFg:  '#7A8494',
  primary:  '#FB923C',
  secondary:'#2D7A7F',
  accent:   '#FDE68A',
  teal:     '#2D7A7F',
};

type Feature = 'Journal' | 'Talk' | 'Report';
type Mode    = 'Glance'  | 'Reflect' | 'Ultra';

const MODE_INFO: Record<Mode, { sub: string }> = {
  Glance:  { sub: 'Short and quick!' },
  Reflect: { sub: 'Deep thought' },
  Ultra:   { sub: 'Heavy analysis' },
};

const GLANCE_LOADING = ['Reading your text....', 'Thinking about it....', 'Hold on....'];

interface Message {
  id:     string;
  role:   'user' | 'model';
  text:   string;
  phase?: 'thinking' | 'response';
  isTyping?: boolean;
}

// ── Typewriter component ──────────────────────────────────────────────────────

function TypingText({ text, speed = 45 }: { text: string; speed?: number }) {
  const [displayed, setDisplayed] = useState('');
  useEffect(() => {
    setDisplayed('');
    let i = 0;
    const iv = setInterval(() => {
      i++;
      setDisplayed(text.slice(0, i));
      if (i >= text.length) clearInterval(iv);
    }, speed);
    return () => clearInterval(iv);
  }, [text, speed]);
  return <Text>{displayed}</Text>;
}

// ── Gradient loading ring (approximation in RN) ───────────────────────────────

function GradientRing() {
  const rotation = useSharedValue(0);
  useEffect(() => {
    rotation.value = withRepeat(withTiming(360, { duration: 1600 }), -1, false);
  }, [rotation]);

  const animStyle = useAnimatedStyle(() => ({
    transform: [{ rotate: `${rotation.value}deg` }],
  }));

  return (
    <Animated.View style={[styles.ringWrapper, animStyle]}>
      <Svg width={22} height={22} viewBox="0 0 22 22">
        <Defs>
          <LinearGradient id="ringGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <Stop offset="0%"   stopColor={C.secondary} stopOpacity="0"    />
            <Stop offset="25%"  stopColor={C.secondary} stopOpacity="0.5"  />
            <Stop offset="55%"  stopColor={C.accent}    stopOpacity="0.85" />
            <Stop offset="80%"  stopColor={C.primary}   stopOpacity="0.95" />
            <Stop offset="100%" stopColor={C.primary}   stopOpacity="1"    />
          </LinearGradient>
        </Defs>
        <Path
          d="M11 2 A9 9 0 1 1 10.9 2"
          stroke="url(#ringGrad)"
          strokeWidth={3}
          strokeLinecap="round"
          fill="none"
        />
      </Svg>
    </Animated.View>
  );
}

// ── Curved send arrow SVG ─────────────────────────────────────────────────────

function CurvedArrow({ active }: { active?: boolean }) {
  const stroke = active ? '#000000' : 'white';
  return (
    <Svg width={26} height={26} viewBox="0 0 26 26" fill="none">
      <Path
        d="M5 19 L12 19 Q14.5 19, 14.5 16.5 L14.5 7"
        stroke={stroke}
        strokeWidth={1.4}
        strokeLinecap="round"
      />
      <Path
        d="M10.5 10.5 L14.5 6 L18.5 10.5"
        stroke={stroke}
        strokeWidth={1.8}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </Svg>
  );
}

// ── QuestionSheet — slide-up panel for two-phase Talk flow ───────────────────

const SHEET_HEIGHT = Dimensions.get('window').height * 0.60;

interface QuestionSheetProps {
  questions:    DisplayQuestion[];
  currentIndex: number;
  onAnswer:     (ans: QuestionAnswer) => void;
  onSkip:       () => void;
}

function QuestionSheet({ questions, currentIndex, onAnswer, onSkip }: QuestionSheetProps) {
  const [inputVal, setInputVal] = useState('');
  const [selected, setSelected] = useState<string | null>(null);

  const current = questions[currentIndex];

  useEffect(() => {
    setInputVal('');
    setSelected(null);
  }, [currentIndex]);

  if (!current) return null;

  const isFollowupHeader = current.kind === 'followup';

  const submit = (rawText: string, rawValue: number | null) => {
    onAnswer({
      original_col: current.original_col,
      node_name:    current.node_name,
      raw_value:    rawValue,
      raw_text:     rawText,
    });
  };

  return (
    <>
      <View style={sheetStyles.overlay} pointerEvents="none" />
      <Animated.View
        entering={SlideInDown.springify().damping(18)}
        exiting={SlideOutDown.duration(220)}
        style={sheetStyles.sheet}
      >
        <View style={sheetStyles.handle} />

        {/* Header */}
        <View style={sheetStyles.header}>
          {isFollowupHeader ? (
            <>
              <Text style={sheetStyles.labelOrange}>BEFORE WE CONTINUE</Text>
              <Text style={sheetStyles.heading}>Let me re-clarify something you mentioned</Text>
            </>
          ) : (
            <>
              <Text style={sheetStyles.labelTeal}>QUICK CHECK-IN</Text>
              <Text style={sheetStyles.heading}>Would love to know you better</Text>
            </>
          )}
        </View>

        {/* Progress dots */}
        <View style={sheetStyles.dotsRow}>
          {questions.map((_, i) => {
            if (i < currentIndex) {
              return <View key={i} style={sheetStyles.dotAnswered} />;
            }
            if (i === currentIndex) {
              return <View key={i} style={sheetStyles.dotCurrent} />;
            }
            return <View key={i} style={sheetStyles.dotUpcoming} />;
          })}
        </View>

        {/* Question body — keyed so it fades on transition */}
        <Animated.View key={currentIndex} entering={FadeIn.duration(200)} exiting={FadeOut.duration(150)} style={sheetStyles.questionBody}>
          <Text style={sheetStyles.questionText}>{current.question}</Text>

          {current.opts && current.opts.length > 0 ? (
            <View style={sheetStyles.optsContainer}>
              {current.opts.map(opt => (
                <TouchableOpacity
                  key={opt.v}
                  onPress={() => { setSelected(String(opt.v)); submit(opt.l, opt.v); }}
                  style={[sheetStyles.optBtn, selected === String(opt.v) && sheetStyles.optBtnActive]}
                  activeOpacity={0.8}
                >
                  <Text style={[sheetStyles.optText, selected === String(opt.v) && sheetStyles.optTextActive]}>
                    {opt.l}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          ) : current.range ? (
            <View style={sheetStyles.inputRow}>
              <TextInput
                style={sheetStyles.textInput}
                placeholder={`${current.range.min}–${current.range.max} ${current.range.unit}`}
                placeholderTextColor={C.mutedFg}
                keyboardType="numeric"
                value={inputVal}
                onChangeText={setInputVal}
              />
              <TouchableOpacity
                onPress={() => { const n = Number(inputVal); submit(inputVal, isNaN(n) ? null : n); }}
                style={sheetStyles.okBtn}
              >
                <Text style={sheetStyles.okBtnText}>OK</Text>
              </TouchableOpacity>
            </View>
          ) : (
            <View style={sheetStyles.inputRow}>
              <TextInput
                style={[sheetStyles.textInput, { flex: 1 }]}
                placeholder="Your answer..."
                placeholderTextColor={C.mutedFg}
                value={inputVal}
                onChangeText={setInputVal}
              />
              <TouchableOpacity
                onPress={() => submit(inputVal, null)}
                style={sheetStyles.okBtn}
              >
                <Text style={sheetStyles.okBtnText}>OK</Text>
              </TouchableOpacity>
            </View>
          )}
        </Animated.View>

        {/* Skip button */}
        <TouchableOpacity onPress={onSkip} style={sheetStyles.skipBtn} activeOpacity={0.7}>
          <Text style={sheetStyles.skipText}>Skip answering</Text>
        </TouchableOpacity>
      </Animated.View>
    </>
  );
}

const sheetStyles = StyleSheet.create({
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(11,14,20,0.40)',
  },
  sheet: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    height: SHEET_HEIGHT,
    backgroundColor: C.card,
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    borderTopWidth: 1,
    borderTopColor: 'rgba(251,146,60,0.25)',
    paddingHorizontal: 20,
    paddingBottom: 16,
  },
  handle: {
    alignSelf: 'center',
    marginTop: 10,
    marginBottom: 18,
    width: 32,
    height: 4,
    borderRadius: 2,
    backgroundColor: 'rgba(255,255,255,0.15)',
  },
  header: {
    marginBottom: 16,
    gap: 4,
  },
  labelOrange: {
    fontSize: 10,
    fontWeight: '700',
    color: C.primary,
    letterSpacing: 0.8,
  },
  labelTeal: {
    fontSize: 10,
    fontWeight: '700',
    color: C.secondary,
    letterSpacing: 0.8,
  },
  heading: {
    fontSize: 16,
    fontWeight: '700',
    color: C.fg,
  },
  dotsRow: {
    flexDirection: 'row',
    gap: 6,
    alignItems: 'center',
    marginBottom: 20,
  },
  dotAnswered: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: C.primary,
  },
  dotCurrent: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: C.secondary,
  },
  dotUpcoming: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderColor: C.border,
  },
  questionBody: {
    flex: 1,
  },
  questionText: {
    fontSize: 15,
    color: C.fg,
    lineHeight: 22,
    marginBottom: 16,
  },
  optsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  optBtn: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: 'rgba(45,122,127,0.12)',
    borderWidth: 1,
    borderColor: 'rgba(45,122,127,0.3)',
  },
  optBtnActive: {
    backgroundColor: 'rgba(45,122,127,0.35)',
    borderColor: C.secondary,
  },
  optText: {
    fontSize: 13,
    color: C.mutedFg,
  },
  optTextActive: {
    color: C.secondary,
    fontWeight: '600',
  },
  inputRow: {
    flexDirection: 'row',
    gap: 8,
    alignItems: 'center',
  },
  textInput: {
    flex: 1,
    backgroundColor: C.muted,
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 8,
    color: C.fg,
    fontSize: 13,
    borderWidth: 1,
    borderColor: C.border,
  },
  okBtn: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 10,
    backgroundColor: C.secondary,
  },
  okBtnText: {
    color: C.fg,
    fontWeight: '600',
    fontSize: 13,
  },
  skipBtn: {
    alignItems: 'center',
    paddingVertical: 14,
  },
  skipText: {
    fontSize: 13,
    color: C.mutedFg,
  },
});

// ── Main ChatScreen ───────────────────────────────────────────────────────────

export default function ChatScreen({ profile }: { profile: UserProfile }) {
  const { db, sessionId, modelsReady } = useAppContext();
  const refreshBeliefs = useBeliefRefresh();
  const insets = useSafeAreaInsets();

  const [feature,         setFeature]         = useState<Feature>('Talk');
  const [mode,            setMode]            = useState<Mode>('Glance');
  const [messages,        setMessages]        = useState<Message[]>([]);
  const [input,           setInput]           = useState('');
  const [loading,         setLoading]         = useState(false);
  const [loadingTextIdx,  setLoadingTextIdx]  = useState(0);
  const [showModeMenu,    setShowModeMenu]    = useState(false);
  const [pendingQuestions, setPendingQs]      = useState<DisplayQuestion[]>([]);
  const [pendingTurnId,   setPendingTurnId]   = useState<string | null>(null);
  const [answeredCount,   setAnsweredCount]   = useState(0);
  const [currentThought,  setCurrentThought]  = useState<string | null>(null);
  const [showReport,      setShowReport]      = useState(false);
  const [reportInitialText, setReportInitialText] = useState('');

  const scrollRef = useRef<ScrollView>(null);
  const answersBuffer = useRef<QuestionAnswer[]>([]);

  const allowedModes: Mode[] =
    feature === 'Journal' ? ['Glance'] :
    feature === 'Talk'    ? ['Glance', 'Reflect'] : ['Ultra'];

  useEffect(() => {
    if (!allowedModes.includes(mode)) setMode(allowedModes[0]);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [feature]);

  useEffect(() => {
    if (!loading) return;
    setLoadingTextIdx(0);
    const iv = setInterval(() => setLoadingTextIdx(i => (i + 1) % GLANCE_LOADING.length), 3000);
    return () => clearInterval(iv);
  }, [loading]);

  useEffect(() => {
    setTimeout(() => scrollRef.current?.scrollToEnd({ animated: true }), 100);
  }, [messages]);

  const agentMode: AgentMode = mode === 'Reflect' ? 'reflect' : 'glance';

  const send = useCallback(async () => {
    if (!input.trim() || loading || !db) return;
    const text = input.trim();
    setInput('');
    setLoading(true);

    const userMsg: Message = { id: Date.now().toString(), role: 'user', text };
    setMessages(m => [...m, userMsg]);

    try {
      if (!modelsReady) {
        setMessages(m => [...m, {
          id: Date.now().toString(), role: 'model',
          text: "AI models not loaded yet. Add GGUF model paths in AppContext.tsx to enable chat.",
          phase: 'response',
        }]);
        return;
      }
      if (feature === 'Journal') {
        const ack = await runJournalTurn(db, sessionId, text);
        setMessages(m => [...m, { id: Date.now().toString(), role: 'model', text: ack, phase: 'response' }]);
        refreshBeliefs();
      } else if (feature === 'Talk') {
        const result = await startTurn(db, sessionId, text, agentMode, (thought) => {
          setCurrentThought(thought);
        });
        if (result.done) {
          setMessages(m => [...m, { id: Date.now().toString(), role: 'model', text: result.response, phase: 'response' }]);
          refreshBeliefs();
        } else {
          // Two-phase flow: show questions, collect answers, then call completeTurn
          setPendingTurnId(result.turnId);
          setPendingQs(result.questions);
          answersBuffer.current = [];
          setAnsweredCount(0);
        }
      } else {
        // Report feature — pass typed text as initial symptom and open modal
        setReportInitialText(text);
        setShowReport(true);
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      console.error('[ChatScreen] startTurn error:', msg);
      setMessages(m => [...m, {
        id: Date.now().toString(), role: 'model',
        text: `Error: ${msg}`, phase: 'response',
      }]);
    } finally {
      setLoading(false);
      setCurrentThought(null);
    }
  }, [input, loading, db, sessionId, feature, agentMode]);

  const handleAnswer = useCallback(async (ans: QuestionAnswer) => {
    if (!pendingTurnId || !db) return;
    answersBuffer.current.push(ans);
    const newCount = answeredCount + 1;
    setAnsweredCount(newCount);

    if (newCount >= pendingQuestions.length) {
      const turnId = pendingTurnId;
      setPendingQs([]);
      setPendingTurnId(null);
      setLoading(true);
      try {
        const response = await completeTurn(db, turnId, agentMode, answersBuffer.current, (thought) => {
          setCurrentThought(thought);
        });
        setMessages(m => [...m, { id: Date.now().toString(), role: 'model', text: response, phase: 'response' }]);
        refreshBeliefs();
      } catch {
        setMessages(m => [...m, {
          id: Date.now().toString(), role: 'model',
          text: "Something went wrong. Please try again.", phase: 'response',
        }]);
      } finally {
        setLoading(false);
        setCurrentThought(null);
        answersBuffer.current = [];
      }
    }
  }, [pendingTurnId, pendingQuestions.length, answeredCount, db, agentMode]);

  const handleSkip = useCallback(async () => {
    if (!pendingTurnId || !db) return;
    const turnId = pendingTurnId;
    const partial = [...answersBuffer.current];
    setPendingQs([]);
    setPendingTurnId(null);
    setLoading(true);
    try {
      const response = await completeTurn(db, turnId, agentMode, partial, (thought) => {
        setCurrentThought(thought);
      });
      setMessages(m => [...m, { id: Date.now().toString(), role: 'model', text: response, phase: 'response' }]);
      refreshBeliefs();
    } catch {
      setMessages(m => [...m, {
        id: Date.now().toString(), role: 'model',
        text: "Something went wrong. Please try again.", phase: 'response',
      }]);
    } finally {
      setLoading(false);
      setCurrentThought(null);
      answersBuffer.current = [];
    }
  }, [pendingTurnId, db, agentMode]);

  const hasMessages = messages.length > 0;
  const canSend     = !!input.trim() && !loading && !!db;

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      keyboardVerticalOffset={Platform.OS === 'ios' ? 0 : 0}
    >
      {/* Top heading bar — shown once chat has started */}
      {hasMessages && (
        <Animated.View entering={FadeIn.duration(250)} style={styles.topBar}>
          <View style={styles.topBarRow}>
            <Text style={styles.topBarText}>Talk with </Text>
            <GliimrText style={styles.gradientNameSize} />
          </View>
        </Animated.View>
      )}

      {/* Messages area */}
      <ScrollView
        ref={scrollRef}
        style={styles.messageArea}
        contentContainerStyle={styles.messageContent}
        showsVerticalScrollIndicator={false}
      >
        {!hasMessages ? (
          <Animated.View entering={FadeIn.duration(500)} style={styles.emptyState}>
            <Text style={styles.greetingText}>Hello {profile.name || 'there'}!</Text>
            <View style={styles.greetingBodyRow}>
              <Text style={styles.greetingBody}>Hello! I am </Text>
              <GliimrText style={{ fontSize: 14, fontWeight: '700' }} />
              <Text style={styles.greetingBody}>, your (health) companion for life. I will listen to you, understand you, and give you insights of your mental and physical health. The more you talk, the more I know you. Let's begin!</Text>
            </View>
          </Animated.View>
        ) : (
          <>
            {messages.map(msg => (
              <Animated.View
                key={msg.id}
                entering={FadeIn.duration(220)}
                style={[styles.msgRow, msg.role === 'user' ? styles.msgRowUser : styles.msgRowModel]}
              >
                <View style={[styles.bubble, msg.role === 'user' ? styles.bubbleUser : styles.bubbleModel]}>
                  <Text style={[styles.bubbleText, msg.role === 'user' ? styles.bubbleTextUser : styles.bubbleTextModel]}>
                    {msg.phase === 'response'
                      ? <TypingText text={msg.text} speed={45} />
                      : msg.text
                    }
                  </Text>
                </View>
              </Animated.View>
            ))}


            {/* Loading indicator */}
            {loading && (
              <Animated.View entering={FadeIn.duration(200)} exiting={FadeOut.duration(200)} style={styles.loadingRow}>
                <GradientRing />
                <Animated.View key={currentThought ?? loadingTextIdx} entering={FadeIn.duration(250)}>
                  <Text style={styles.loadingText} numberOfLines={1} ellipsizeMode="tail">
                    {currentThought
                      ? currentThought
                      : mode === 'Glance'
                        ? GLANCE_LOADING[loadingTextIdx]
                        : 'Analysing and cross-referencing...'
                    }
                  </Text>
                </Animated.View>
              </Animated.View>
            )}
          </>
        )}
      </ScrollView>

      {/* Unified chat bar */}
      <View style={[styles.chatBarWrapper, { paddingBottom: Math.max(12, insets.bottom + 4) }]}>
        <View style={styles.chatBar}>
          {/* Top row: mode dropdown */}
          <View style={styles.chatBarTop}>
            <TouchableOpacity
              onPress={() => setShowModeMenu(m => !m)}
              style={styles.modeBtn}
              activeOpacity={0.85}
            >
              <Text style={styles.modeBtnText}>{mode}</Text>
              <Text style={styles.modeChevron}>▾</Text>
            </TouchableOpacity>
          </View>

          {/* Middle: text input */}
          <View style={styles.inputArea}>
            {!input && (
              <View style={styles.placeholderOverlay} pointerEvents="none">
                <Text style={styles.placeholderText}>Talk with Gliimr</Text>
              </View>
            )}
            <TextInput
              style={styles.textInput}
              value={input}
              onChangeText={setInput}
              multiline
            />
          </View>

          {/* Bottom row: feature pills + send */}
          <View style={styles.chatBarBottom}>
            <View style={styles.featurePills}>
              {(['Journal', 'Talk', 'Report'] as Feature[]).map(f => (
                <TouchableOpacity
                  key={f}
                  onPress={() => setFeature(f)}
                  style={[styles.pill, feature === f && styles.pillActive]}
                  activeOpacity={0.8}
                >
                  <Text style={[styles.pillText, feature === f && styles.pillTextActive]}>{f}</Text>
                </TouchableOpacity>
              ))}
            </View>

            {/* Send button */}
            <TouchableOpacity
              onPress={send}
              disabled={!canSend}
              style={[styles.sendBtn, canSend ? styles.sendBtnActive : styles.sendBtnInactive]}
              activeOpacity={0.8}
            >
              <CurvedArrow active={canSend} />
            </TouchableOpacity>
          </View>
        </View>
      </View>

      {/* Question sheet */}
      {pendingQuestions.length > 0 && pendingTurnId !== null && (
        <QuestionSheet
          questions={pendingQuestions}
          currentIndex={answeredCount}
          onAnswer={handleAnswer}
          onSkip={handleSkip}
        />
      )}

      {/* Report generator modal */}
      <Modal
        visible={showReport}
        animationType="slide"
        onRequestClose={() => { setShowReport(false); setReportInitialText(''); }}
      >
        <ReportGeneratorScreen
          initialSymptom={reportInitialText}
          onClose={() => { setShowReport(false); setReportInitialText(''); }}
        />
      </Modal>

      {/* Mode menu modal */}
      <Modal
        transparent
        visible={showModeMenu}
        animationType="fade"
        onRequestClose={() => setShowModeMenu(false)}
      >
        <Pressable style={styles.modalOverlay} onPress={() => setShowModeMenu(false)}>
          <Animated.View entering={FadeIn.duration(150)} style={styles.modeMenu}>
            {(['Glance', 'Reflect', 'Ultra'] as Mode[]).map(m => {
              const enabled  = allowedModes.includes(m);
              const selected = mode === m;
              return (
                <TouchableOpacity
                  key={m}
                  disabled={!enabled}
                  onPress={() => { if (enabled) { setMode(m); setShowModeMenu(false); } }}
                  style={[styles.modeMenuItem, selected && styles.modeMenuItemSelected]}
                  activeOpacity={enabled ? 0.8 : 1}
                >
                  <Text style={[styles.modeMenuItemText, selected && styles.modeMenuItemTextSelected, !enabled && styles.modeMenuItemDisabled]}>
                    {m}
                  </Text>
                  <Text style={[styles.modeMenuSub, !enabled && styles.modeMenuItemDisabled]}>
                    {MODE_INFO[m].sub}
                  </Text>
                </TouchableOpacity>
              );
            })}
          </Animated.View>
        </Pressable>
      </Modal>
    </KeyboardAvoidingView>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: C.bg,
  },
  topBar: {
    paddingHorizontal: 16,
    paddingTop: Platform.OS === 'android' ? (StatusBar.currentHeight ?? 24) + 4 : 14,
    paddingBottom: 8,
    alignItems: 'center',
  },
  topBarRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  topBarText: {
    fontSize: 15,
    fontWeight: '600',
    color: C.fg,
  },
  gradientName: {
    color: C.primary,
    fontWeight: '700',
  },
  gradientNameSize: {
    fontSize: 15,
    fontWeight: '700',
  },
  messageArea: {
    flex: 1,
  },
  messageContent: {
    paddingHorizontal: 16,
    paddingTop: 16,
    paddingBottom: 8,
    flexGrow: 1,
  },
  emptyState: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 16,
    gap: 12,
  },
  greetingText: {
    fontSize: 24,
    fontWeight: '700',
    color: C.primary,
    textAlign: 'center',
    marginBottom: 4,
  },
  greetingBodyRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    alignItems: 'center',
    justifyContent: 'center',
  },
  greetingBody: {
    fontSize: 14,
    color: C.mutedFg,
    textAlign: 'center',
    lineHeight: 22,
  },
  msgRow: {
    marginBottom: 10,
  },
  msgRowUser: {
    alignItems: 'flex-end',
  },
  msgRowModel: {
    alignItems: 'flex-start',
  },
  bubble: {
    maxWidth: '82%',
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  bubbleUser: {
    backgroundColor: C.primary,
    borderRadius: 18,
    borderBottomRightRadius: 4,
  },
  bubbleModel: {
    backgroundColor: C.elevated,
    borderRadius: 18,
    borderBottomLeftRadius: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.4,
    shadowRadius: 9,
    elevation: 5,
  },
  bubbleText: {
    fontSize: 14,
    lineHeight: 21,
  },
  bubbleTextUser: {
    color: '#000000',
  },
  bubbleTextModel: {
    color: C.fg,
  },
  loadingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingLeft: 4,
    marginBottom: 8,
  },
  loadingText: {
    fontSize: 13,
    color: C.teal,
    maxWidth: 220,
  },
  ringWrapper: {
    width: 22,
    height: 22,
    shadowColor: '#FB923C',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.7,
    shadowRadius: 6,
    elevation: 6,
  },
  chatBarWrapper: {
    paddingHorizontal: 12,
    paddingBottom: 0,
    paddingTop: 8,
  },
  chatBar: {
    backgroundColor: '#0B0E14',
    borderRadius: 32,
    paddingHorizontal: 16,
    paddingVertical: 14,
    gap: 10,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.15)',
  },
  chatBarTop: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  modeBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: 20,
    backgroundColor: 'rgba(251,146,60,0.14)',
    borderWidth: 1,
    borderColor: 'rgba(251,146,60,0.55)',
    gap: 4,
  },
  modeBtnText: {
    fontSize: 13,
    fontWeight: '600',
    color: C.primary,
  },
  modeChevron: {
    fontSize: 11,
    color: C.primary,
  },
  inputArea: {
    minHeight: 32,
    position: 'relative',
  },
  placeholderOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    pointerEvents: 'none',
  },
  placeholderText: {
    color: '#7A8494',
    fontSize: 14,
    lineHeight: 21,
  },
  textInput: {
    color: C.fg,
    fontSize: 14,
    lineHeight: 21,
    maxHeight: 150,
    backgroundColor: 'transparent',
    padding: 0,
  },
  chatBarBottom: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  featurePills: {
    flexDirection: 'row',
    gap: 6,
  },
  pill: {
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: 20,
    backgroundColor: 'rgba(45,122,127,0.10)',
    borderWidth: 1,
    borderColor: 'transparent',
  },
  pillActive: {
    backgroundColor: 'rgba(45,122,127,0.35)',
    borderColor: 'rgba(45,122,127,0.7)',
  },
  pillText: {
    fontSize: 13,
    fontWeight: '500',
    color: C.mutedFg,
  },
  pillTextActive: {
    color: C.secondary,
  },
  sendBtn: {
    width: 36,
    height: 36,
    borderRadius: 18,
    alignItems: 'center',
    justifyContent: 'center',
  },
  sendBtnActive: {
    backgroundColor: C.primary,
    shadowColor: C.primary,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.22,
    shadowRadius: 4,
    elevation: 4,
  },
  sendBtnInactive: {
    backgroundColor: 'rgba(251,146,60,0.2)',
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'flex-end',
    paddingHorizontal: 12,
    paddingBottom: 120,
  },
  modeMenu: {
    backgroundColor: '#13171F',
    borderRadius: 16,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: '#22262D',
    maxWidth: 176,
  },
  modeMenuItem: {
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderLeftWidth: 2,
    borderLeftColor: 'transparent',
  },
  modeMenuItemSelected: {
    backgroundColor: 'rgba(251,146,60,0.12)',
    borderLeftColor: C.primary,
  },
  modeMenuItemText: {
    fontSize: 14,
    fontWeight: '500',
    color: C.fg,
  },
  modeMenuItemTextSelected: {
    color: C.primary,
  },
  modeMenuItemDisabled: {
    opacity: 0.3,
  },
  modeMenuSub: {
    fontSize: 12,
    color: C.mutedFg,
    marginTop: 2,
  },
});
