"""
Step 5c: Learn P(col_val | node_state) likelihood tables per source column.

For each node and each of its source columns, counts co-occurrences of
(discretized_column_value, node_state) across all datasets that have both.
Normalises row-wise to produce P(col_val | node_state).

Input:
  datasets/data_preprocessed/intermediate_use_files/{studentlife,lifesnaps,nhanes}_discretized_cols.csv
    — per-source-column discretized labels, written by data_discretization.py
  datasets/data_preprocessed/{studentlife,lifesnaps,nhanes}_discretized.csv
    — node_name columns (node states) + structural columns
  configs/feature_node_config.json
    — source_column_bins[col].state_labels  (possible discrete values per col)
    — source_columns[ds]                    (which cols each dataset contributes)
    — state_labels                          (node state label list)

Output:
  Updates configs/feature_node_config.json:
    nodes[node_name].column_likelihoods[col] = {
        node_state: {col_val: probability, ...},
        ...
    }
  i.e.  column_likelihoods[col][node_state][col_val] = P(col_val | node_state)
"""

import os, json
import pandas as pd

_ROOT   = os.path.join(os.path.dirname(__file__), '..', '..', '..')
_CONFIG = os.path.join(_ROOT, 'configs', 'feature_node_config.json')
_PROC   = os.path.join(_ROOT, 'datasets', 'data_preprocessed')
_INTER  = os.path.join(_ROOT, 'datasets', 'data_preprocessed', 'intermediate_use_files')

_STRUCTURAL = {'user_id', 'date', 'hour', 'dataset'}

_DATASETS = ['studentlife', 'lifesnaps', 'nhanes']


def _load_config():
    if not os.path.exists(_CONFIG):
        raise FileNotFoundError(f'Config not found: {_CONFIG}')
    with open(_CONFIG) as f:
        return json.load(f)

def _save_config(cfg):
    with open(_CONFIG, 'w') as f:
        json.dump(cfg, f, indent=2)
    print('Config updated with column_likelihoods.')


def _load_frames():
    disc_cols, disc_nodes = {}, {}
    for ds in _DATASETS:
        cols_path  = os.path.join(_INTER, f'{ds}_discretized_cols.csv')
        nodes_path = os.path.join(_PROC,  f'{ds}_discretized.csv')
        if not os.path.exists(cols_path):
            raise FileNotFoundError(
                f'{cols_path} not found — run Step 5 (discretization) first.'
            )
        if not os.path.exists(nodes_path):
            raise FileNotFoundError(
                f'{nodes_path} not found — run Step 5 (discretization) first.'
            )
        disc_cols[ds]  = pd.read_csv(cols_path,  low_memory=False)
        disc_nodes[ds] = pd.read_csv(nodes_path, low_memory=False)
    return disc_cols, disc_nodes



def build_likelihoods():
    cfg  = _load_config()
    nodes = cfg['nodes']

    disc_cols, disc_nodes = _load_frames()

    for node_name, node_cfg in nodes.items():
        scb          = node_cfg.get('source_column_bins', {})
        node_states  = node_cfg['state_labels']
        src_cols_map = node_cfg.get('source_columns', {})

        if not scb:
            continue

        likelihoods = {}

        for col, col_cfg in scb.items():
            col_state_labels = col_cfg.get('state_labels', node_states)
            parts = []

            for ds in _DATASETS:
                ds_cols = src_cols_map.get(ds, [])
                if col not in ds_cols:
                    continue

                cols_df  = disc_cols[ds]
                nodes_df = disc_nodes[ds]

                if col not in cols_df.columns or node_name not in nodes_df.columns:
                    continue

                paired = pd.DataFrame({
                    'col_val':    cols_df[col].values,
                    'node_state': nodes_df[node_name].values,
                }).dropna()
                if len(paired):
                    parts.append(paired)

            if not parts:
                print(f'  [WARN] {node_name}/{col}: no paired data found')
                likelihoods[col] = {}
                continue

            all_paired = pd.concat(parts, ignore_index=True)
            ct = pd.crosstab(all_paired['node_state'], all_paired['col_val'])

            # Ensure all expected states and values are present
            for ns in node_states:
                if ns not in ct.index:
                    ct.loc[ns] = 0
            for cv in col_state_labels:
                if cv not in ct.columns:
                    ct[cv] = 0
            ct = ct.loc[node_states, col_state_labels]

            row_totals = ct.sum(axis=1)
            prob_table = {}
            for ns in node_states:
                total = row_totals[ns]
                if total == 0:
                    prob_table[ns] = {cv: 0.0 for cv in col_state_labels}
                else:
                    prob_table[ns] = {cv: round(int(ct.loc[ns, cv]) / total, 6)
                                      for cv in col_state_labels}
            likelihoods[col] = prob_table
            print(f'  {node_name}/{col}: likelihoods computed over {len(all_paired)} rows')

        node_cfg['column_likelihoods'] = likelihoods

    _save_config(cfg)
    print('\nLikelihood tables complete.')


if __name__ == '__main__':
    build_likelihoods()
