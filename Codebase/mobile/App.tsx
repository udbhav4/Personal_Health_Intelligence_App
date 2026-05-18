import React from 'react';
import { View, StyleSheet } from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider } from 'react-native-safe-area-context';

import { AppProvider }  from './src/core/AppContext';
import RootScreen        from './src/screens/RootScreen';

export default function App() {
  return (
    <SafeAreaProvider>
      <AppProvider>
        <View style={styles.root}>
          <StatusBar style="light" />
          <RootScreen />
        </View>
      </AppProvider>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#0B0E14' },
});
