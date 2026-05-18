import React from 'react';
import { Text, type TextStyle } from 'react-native';
import MaskedView from '@react-native-masked-view/masked-view';
import { LinearGradient } from 'expo-linear-gradient';

interface Props {
  style?: TextStyle;
}

export default function GliimrText({ style }: Props) {
  try {
    return (
      <MaskedView
        maskElement={
          <Text style={[{ color: '#FB923C' }, style]}>Gliimr</Text>
        }
      >
        <LinearGradient
          colors={['#FB923C', '#F9B96A', '#FDE68A', '#90C4BF', '#2D7A7F']}
          locations={[0, 0.25, 0.5, 0.75, 1]}
          start={{ x: 0, y: 0.5 }}
          end={{ x: 1, y: 0.5 }}
        >
          <Text style={[style, { opacity: 0 }]}>Gliimr</Text>
        </LinearGradient>
      </MaskedView>
    );
  } catch {
    return <Text style={[style, { color: '#FB923C' }]}>Gliimr</Text>;
  }
}
