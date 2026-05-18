/**
 * Clock collector — derives time_of_day node from device local time.
 *
 * Uses new Date() which respects the device's timezone automatically.
 * No permissions required.
 *
 * Bins:
 *   hour < 4 OR hour >= 20  → 'night'
 *   4  <= hour < 12          → 'morning'
 *   12 <= hour < 17          → 'afternoon'
 *   17 <= hour < 20          → 'evening'
 */

import type { CollectorResult } from './types';

function discretizeHour(hour: number): string {
  if (hour < 4 || hour >= 20) return 'night';
  if (hour < 12)              return 'morning';
  if (hour < 17)              return 'afternoon';
  return 'evening';
}

export function collectClock(): CollectorResult {
  const now  = new Date();
  const hour = now.getHours(); // integer 0–23 in device local time

  return {
    node_name:         'time_of_day',
    source_column:     'time_of_day',
    data_source:       'system_clock',
    raw_value:         hour,
    raw_unit:          'hour',
    discretized_value: discretizeHour(hour),
    confidence:        1.0,
  };
}
