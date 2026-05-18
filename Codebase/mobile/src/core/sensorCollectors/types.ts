/**
 * Shared output type for all sensor collectors.
 * Each collector returns one or more CollectorResult objects — one per
 * (node_name, source_column) pair it produces.
 */

export interface CollectorResult {
  node_name:         string;
  source_column:     string;
  data_source:       string;
  raw_value:         number | null;
  raw_unit:          string;
  discretized_value: string;
  confidence:        number;
}
