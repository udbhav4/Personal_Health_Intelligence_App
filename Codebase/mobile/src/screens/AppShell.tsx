import React, { useState, useRef, useCallback } from 'react';
import {
  View, Text, TouchableOpacity, StyleSheet, Dimensions,
  PanResponder,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import Animated, {
  useSharedValue, useAnimatedStyle, withSpring,
  withTiming, FadeIn,
} from 'react-native-reanimated';
import Svg, { Path, Rect, Circle } from 'react-native-svg';

import DashboardScreen from './DashboardScreen';
import ChatScreen      from './ChatScreen';
import ProfileScreen   from './ProfileScreen';
import type { UserProfile } from '../core/AppContext';

const { width: SCREEN_W } = Dimensions.get('window');

// ── Colours ───────────────────────────────────────────────────────────────────

const C = {
  bg:      '#0B0E14',
  primary: '#FB923C',
  mutedFg: '#7A8494',
};

// ── Tab icons (SVG approximations of lucide icons) ───────────────────────────

function DashboardIcon({ active }: { active: boolean }) {
  const col = active ? C.primary : C.mutedFg;
  return (
    <Svg width={22} height={22} viewBox="0 0 24 24" fill="none">
      <Rect x={3} y={3} width={7} height={7} rx={1} stroke={col} strokeWidth={1.8} />
      <Rect x={14} y={3} width={7} height={7} rx={1} stroke={col} strokeWidth={1.8} />
      <Rect x={3} y={14} width={7} height={7} rx={1} stroke={col} strokeWidth={1.8} />
      <Rect x={14} y={14} width={7} height={7} rx={1} stroke={col} strokeWidth={1.8} />
    </Svg>
  );
}

function ChatIcon({ active }: { active: boolean }) {
  const col = active ? C.primary : C.mutedFg;
  return (
    <Svg width={22} height={22} viewBox="0 0 24 24" fill="none">
      <Path
        d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"
        stroke={col}
        strokeWidth={1.8}
        strokeLinejoin="round"
      />
    </Svg>
  );
}

function ProfileIcon({ active }: { active: boolean }) {
  const col = active ? C.primary : C.mutedFg;
  return (
    <Svg width={22} height={22} viewBox="0 0 24 24" fill="none">
      <Circle cx={12} cy={8} r={4} stroke={col} strokeWidth={1.8} />
      <Path d="M4 20c0-4 3.58-7 8-7s8 3 8 7" stroke={col} strokeWidth={1.8} strokeLinecap="round" />
    </Svg>
  );
}

// ── Tab config ────────────────────────────────────────────────────────────────

type Tab = 'dashboard' | 'chat' | 'profile';
const TAB_ORDER: Tab[] = ['dashboard', 'chat', 'profile'];

const TAB_LABELS: Record<Tab, string> = {
  dashboard: 'Dashboard',
  chat:      'Chat',
  profile:   'Profile',
};

// ── Props ─────────────────────────────────────────────────────────────────────

interface Props {
  profile:       UserProfile;
  onProfileSave: (p: UserProfile) => Promise<void>;
}

// ── Main ──────────────────────────────────────────────────────────────────────

export default function AppShell({ profile, onProfileSave }: Props) {
  const insets = useSafeAreaInsets();
  const [tab, setTab] = useState<Tab>('chat');
  const tabIdx = TAB_ORDER.indexOf(tab);

  const translateX = useSharedValue(-tabIdx * SCREEN_W);

  const switchTab = useCallback((next: Tab) => {
    const idx = TAB_ORDER.indexOf(next);
    translateX.value = withSpring(-idx * SCREEN_W, { stiffness: 320, damping: 36, mass: 0.9 });
    setTab(next);
  }, [translateX]);

  // Start on chat tab
  React.useEffect(() => {
    translateX.value = -TAB_ORDER.indexOf('chat') * SCREEN_W;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const panResponder = useRef(
    PanResponder.create({
      onMoveShouldSetPanResponder: (_, g) => Math.abs(g.dx) > Math.abs(g.dy) && Math.abs(g.dx) > 10,
      onPanResponderRelease: (_, g) => {
        if (Math.abs(g.dx) < 50) return;
        setTab(cur => {
          const cur_idx = TAB_ORDER.indexOf(cur);
          if (g.dx < 0 && cur_idx < TAB_ORDER.length - 1) {
            const next = TAB_ORDER[cur_idx + 1];
            const idx = cur_idx + 1;
            translateX.value = withSpring(-idx * SCREEN_W, { stiffness: 320, damping: 36, mass: 0.9 });
            return next;
          }
          if (g.dx > 0 && cur_idx > 0) {
            const next = TAB_ORDER[cur_idx - 1];
            const idx = cur_idx - 1;
            translateX.value = withSpring(-idx * SCREEN_W, { stiffness: 320, damping: 36, mass: 0.9 });
            return next;
          }
          return cur;
        });
      },
    })
  ).current;

  const stripStyle = useAnimatedStyle(() => ({
    transform: [{ translateX: translateX.value }],
  }));

  return (
    <View style={styles.container}>
      {/* Sliding strip */}
      <View style={styles.screenArea} {...panResponder.panHandlers}>
        <Animated.View style={[styles.strip, stripStyle]}>
          {TAB_ORDER.map(id => (
            <View key={id} style={styles.screen}>
              {id === 'dashboard' && <DashboardScreen profile={profile} />}
              {id === 'chat'      && <ChatScreen profile={profile} />}
              {id === 'profile'   && <ProfileScreen profile={profile} onSave={onProfileSave} isActive={tab === 'profile'} />}
            </View>
          ))}
        </Animated.View>
      </View>

      {/* Bottom nav */}
      <View style={[styles.nav, { paddingBottom: Math.max(20, insets.bottom + 8) }]}>
        {TAB_ORDER.map(id => {
          const active = tab === id;
          return (
            <TouchableOpacity
              key={id}
              onPress={() => switchTab(id)}
              style={styles.navItem}
              activeOpacity={0.8}
            >
              {id === 'dashboard' && <DashboardIcon active={active} />}
              {id === 'chat'      && <ChatIcon      active={active} />}
              {id === 'profile'   && <ProfileIcon   active={active} />}
              <Text style={[styles.navLabel, active && styles.navLabelActive]}>
                {TAB_LABELS[id]}
              </Text>
              {active && (
                <Animated.View entering={FadeIn.duration(200)} style={styles.navDot} />
              )}
            </TouchableOpacity>
          );
        })}
      </View>
    </View>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: C.bg,
  },
  screenArea: {
    flex: 1,
    overflow: 'hidden',
  },
  strip: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    left: 0,
    flexDirection: 'row',
    width: SCREEN_W * 3,
  },
  screen: {
    width: SCREEN_W,
    overflow: 'hidden',
  },
  nav: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-around',
    paddingHorizontal: 24,
    paddingTop: 8,
    paddingBottom: 20,
    backgroundColor: 'rgba(11,14,20,0.82)',
    borderTopWidth: 1,
    borderTopColor: 'rgba(251,146,60,0.15)',
  },
  navItem: {
    alignItems: 'center',
    gap: 2,
    paddingHorizontal: 16,
    paddingVertical: 4,
    borderRadius: 12,
  },
  navLabel: {
    fontSize: 11,
    fontWeight: '500',
    color: C.mutedFg,
  },
  navLabelActive: {
    color: C.primary,
  },
  navDot: {
    width: 4,
    height: 4,
    borderRadius: 2,
    backgroundColor: C.primary,
  },
});
