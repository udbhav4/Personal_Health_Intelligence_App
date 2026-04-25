"""
Step 6: Compute NHANES CPT priors.

For each Cat1/Cat2 node (nodes with NHANES source columns), counts
(nhanes_available_parent_states, node_state) from nhanes_discretized.csv
and normalises row-wise with Laplace smoothing.

nhanes_available_parents = forced_parents ∩ nodes with nhanes source_columns,
sorted alphabetically (for deterministic key ordering).

Prior key format: "parent1=s1|parent2=s2"  (parents in sorted order).
No-parent nodes use key "".

All parent-state combinations are enumerated from each parent's state_labels
so the CPT is complete — unseen combos fall back to a Laplace-smoothed
uniform distribution over the child's state_labels.

Output:
  Updates configs/feature_node_config.json:
    nodes[node].prior = {
        "parent1=s1|parent2=s2": {node_state: probability, ...},
        ...
    }
  Cat3 nodes (no NHANES data): prior = {}
"""

import os, json
from itertools import product

import pandas as pd

_ROOT        = os.path.join(os.path.dirname(__file__), '..', '..')
_CONFIG      = os.path.join(_ROOT, 'configs', 'feature_node_config.json')
_NHANES_DISC = os.path.join(_ROOT, 'datasets', 'data_preprocessed',
                            'nhanes_discretized.csv')

_LAPLACE_ALPHA = 1


def _load_config():
    if not os.path.exists(_CONFIG):
        raise FileNotFoundError(f'Config not found: {_CONFIG}')
    with open(_CONFIG) as f:
        return json.load(f)

def _save_config(cfg):
    with open(_CONFIG, 'w') as f:
        json.dump(cfg, f, indent=2)
    print('Config updated with NHANES priors.')


def _nhanes_node_set(nodes):
    return {n for n, nc in nodes.items()
            if nc.get('source_columns', {}).get('nhanes')}


def _row_prob(counts, state_labels):
    """Laplace-smoothed probability dict for one parent-combo row."""
    total = sum(counts.get(s, 0) for s in state_labels) + _LAPLACE_ALPHA * len(state_labels)
    return {s: round((counts.get(s, 0) + _LAPLACE_ALPHA) / total, 6)
            for s in state_labels}


def build_priors():
    cfg   = _load_config()
    nodes = cfg['nodes']

    if not os.path.exists(_NHANES_DISC):
        raise FileNotFoundError(
            f'{_NHANES_DISC} not found — run Step 5 (discretization) first.'
        )
    df = pd.read_csv(_NHANES_DISC, low_memory=False)

    nhanes_set = _nhanes_node_set(nodes)

    for node_name, node_cfg in nodes.items():
        if node_name not in nhanes_set:
            node_cfg['prior'] = {}
            continue

        state_labels = node_cfg['state_labels']

        if node_name not in df.columns:
            print(f'  [WARN] {node_name}: missing from nhanes_discretized.csv')
            node_cfg['prior'] = {}
            continue

        forced_parents = node_cfg.get('forced_parents', [])
        nhanes_parents = sorted(
            p for p in forced_parents
            if p in nhanes_set and p in df.columns
        )

        work = df[[node_name] + nhanes_parents].dropna(subset=[node_name])
        if nhanes_parents:
            work = work.dropna(subset=nhanes_parents)

        prior_table = {}

        if not nhanes_parents:
            counts = work[node_name].value_counts().to_dict()
            prior_table[''] = _row_prob(counts, state_labels)
        else:
            parent_state_lists = [nodes[p]['state_labels'] for p in nhanes_parents]
            for combo in product(*parent_state_lists):
                key  = '|'.join(f'{p}={s}' for p, s in zip(nhanes_parents, combo))
                mask = pd.Series(True, index=work.index)
                for p, s in zip(nhanes_parents, combo):
                    mask &= (work[p] == s)
                subset = work.loc[mask, node_name]
                counts = subset.value_counts().to_dict()
                prior_table[key] = _row_prob(counts, state_labels)

        node_cfg['prior'] = prior_table
        print(f'  {node_name}: {len(prior_table)} combo(s), {len(work)} rows')

    _save_config(cfg)
    print('\nNHANES priors complete.')


if __name__ == '__main__':
    build_priors()
