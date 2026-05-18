import React from 'react';
import {
  View, ActivityIndicator, Text, StyleSheet,
} from 'react-native';
import Animated, {
  FadeIn, FadeOut,
} from 'react-native-reanimated';

import { useAppContext }      from '../core/AppContext';
import OnboardingScreen       from './OnboardingScreen';
import ModelDownloadScreen    from './ModelDownloadScreen';
import AppShell               from './AppShell';

const C = {
  bg:       '#0B0E14',
  primary:  '#FB923C',
  mutedFg:  '#7A8494',
  danger:   '#FF4D4D',
};

export default function RootScreen() {
  const {
    appReady, initError,
    modelsDownloaded, onModelsReady,
    hasCompletedOnboarding, profile,
    saveProfile, updateProfile,
  } = useAppContext();

  if (!appReady) {
    return (
      <View style={styles.loader}>
        <ActivityIndicator size="large" color={C.primary} />
        <Text style={styles.loaderText}>Loading…</Text>
      </View>
    );
  }

  if (initError) {
    return (
      <View style={styles.loader}>
        <Text style={styles.errorTitle}>Startup failed</Text>
        <Text style={styles.errorBody}>{initError}</Text>
      </View>
    );
  }

  // Hydrating — render nothing to avoid flash
  if (hasCompletedOnboarding === null) {
    return <View style={styles.loader} />;
  }

  if (!hasCompletedOnboarding) {
    return (
      <Animated.View entering={FadeIn.duration(450)} exiting={FadeOut.duration(450)} style={StyleSheet.absoluteFill}>
        <OnboardingScreen onComplete={saveProfile} />
      </Animated.View>
    );
  }

  if (!modelsDownloaded) {
    return (
      <Animated.View entering={FadeIn.duration(450)} exiting={FadeOut.duration(300)} style={StyleSheet.absoluteFill}>
        <ModelDownloadScreen onComplete={onModelsReady} />
      </Animated.View>
    );
  }

  return (
    <Animated.View entering={FadeIn.duration(450)} style={StyleSheet.absoluteFill}>
      <AppShell profile={profile} onProfileSave={updateProfile} />
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  loader: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: C.bg,
    gap: 12,
  },
  loaderText: {
    fontSize: 14,
    color: C.mutedFg,
  },
  errorTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: C.danger,
    textAlign: 'center',
  },
  errorBody: {
    fontSize: 13,
    color: C.mutedFg,
    textAlign: 'center',
    marginHorizontal: 32,
  },
});
