"""
Step 5: Multi-column discretization + k-modes node state derivation.

Phase 1 — Per-column discretization:
  For each node, each source column is discretized independently using its own
  bins/method defined in source_column_bins (config). Methods:
    clinical    — pd.cut with fixed thresholds + state_labels
    kmeans      — silhouette-optimal k in [2,4] on pooled values across all
                  datasets; writes k + bin_edges back to config
    passthrough — map numeric codes -> string labels via value_map (if present)
                  or cast to str directly

Phase 2 — Node state derivation via k-modes:
  For each node, collect the matrix of per-row discretized column values.
  Run k-modes with k = len(node.state_labels) to cluster rows.
  Order clusters by average ordinal rank across columns -> assign state_labels
  in order (low -> high severity).

Outputs:
  datasets/data_preprocessed/{studentlife,lifesnaps,nhanes}_discretized.csv
Updates:
  configs/feature_node_config.json  (kmeans source_column_bins: k, bin_edges)
"""

import os, json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kmodes.kmodes import KModes

_ROOT      = os.path.join(os.path.dirname(__file__), '..', '..', '..')
_CONFIG    = os.path.join(_ROOT, 'configs', 'feature_node_config.json')
_PROC      = os.path.join(_ROOT, 'datasets', 'data_preprocessed')
_OUT       = os.path.join(_ROOT, 'datasets', 'data_preprocessed')
_INTER_OUT = os.path.join(_ROOT, 'datasets', 'data_preprocessed', 'intermediate_use_files')

_STRUCTURAL = ['user_id', 'date', 'hour', 'dataset']

_LABELS_BY_K = {
    2: ['low', 'high'],
    3: ['low', 'moderate', 'high'],
    4: ['low', 'moderate_low', 'moderate_high', 'high'],
}


# ── Config I/O ────────────────────────────────────────────────────────────────

def _load_config():
    if not os.path.exists(_CONFIG):
        raise FileNotFoundError(f'Config not found: {_CONFIG}')
    with open(_CONFIG) as f:
        return json.load(f)

def _save_config(cfg):
    with open(_CONFIG, 'w') as f:
        json.dump(cfg, f, indent=2)
    print('Config updated with kmeans bin results.')


# ── Phase 1 helpers ───────────────────────────────────────────────────────────

def _silhouette_k(values_1d):
    X = values_1d.reshape(-1, 1)
    n_distinct = len(np.unique(values_1d))
    best_k, best_score, best_km = 2, -1.0, None
    for k in range(2, 5):
        if len(values_1d) < k * 2:
            continue
        if n_distinct < k:
            continue
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(X, labels,
                                 sample_size=min(len(X), 2000),
                                 random_state=42)
        if score > best_score:
            best_score, best_k, best_km = score, k, km
    if best_km is None:
        if n_distinct < 2:
            return 2, [float(values_1d[0])]
        best_km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X)
        best_k  = 2
    centroids = sorted(best_km.cluster_centers_.flatten())
    return best_k, centroids

def _interior_edges(centroids):
    return [(centroids[i] + centroids[i+1]) / 2.0 for i in range(len(centroids)-1)]


def _discretize_column(series, col_cfg, pooled_vals=None):
    """
    Discretize a single Series according to col_cfg.
    For kmeans, bin_edges must already be set (pre-computed from pooled_vals).
    Returns a string-dtype Series with NaN preserved.
    """
    method = col_cfg['method']

    if method == 'clinical':
        thresholds   = col_cfg['thresholds']
        state_labels = col_cfg['state_labels']
        result = pd.cut(series, bins=thresholds, labels=state_labels,
                        right=False, include_lowest=True)
        return result.astype(object)

    elif method == 'kmeans':
        bin_edges    = col_cfg['bin_edges']
        state_labels = col_cfg['state_labels']
        bins         = [-np.inf] + bin_edges + [np.inf]
        result = pd.cut(series, bins=bins, labels=state_labels,
                        right=True, include_lowest=True)
        return result.astype(object)

    elif method == 'passthrough':
        val_map = col_cfg.get('value_map')
        if val_map:
            # JSON keys are strings; coerce series to str for mapping
            return series.map(lambda v: val_map.get(str(int(v)), np.nan)
                              if pd.notna(v) else np.nan)
        return series.astype(str).where(series.notna(), np.nan)

    else:
        raise ValueError(f'Unknown column discretization method: {method!r}')


def _fit_kmeans_bins(col, node_name, datasets, scb_entry):
    """
    Fit kmeans on pooled non-null values of col across all datasets.
    Writes k, bin_edges, state_labels back into scb_entry in-place.
    Returns updated scb_entry.
    """
    parts = []
    for df in datasets.values():
        if col in df.columns:
            vals = df[col].dropna().to_numpy().astype(float)
            if len(vals):
                parts.append(vals)

    if not parts:
        print(f'  [WARN] {node_name}/{col}: no data for kmeans, defaulting k=2')
        scb_entry.update({'k': 2, 'bin_edges': [], 'state_labels': ['low', 'high']})
        return scb_entry

    pooled = np.concatenate(parts)
    if len(pooled) < 4:
        scb_entry.update({'k': 2, 'bin_edges': [], 'state_labels': ['low', 'high']})
        return scb_entry

    best_k, centroids = _silhouette_k(pooled)
    bin_edges         = _interior_edges(centroids)
    labels            = _LABELS_BY_K.get(best_k, _LABELS_BY_K[2])

    scb_entry.update({
        'k':           best_k,
        'bin_edges':   [round(float(e), 6) for e in bin_edges],
        'state_labels': labels,
    })
    print(f'  kmeans {node_name}/{col}: k={best_k}, '
          f'edges={[round(e,4) for e in bin_edges]}')
    return scb_entry


# ── Phase 2 helpers ───────────────────────────────────────────────────────────

def _ordinal_rank(label, state_labels):
    """Return ordinal index of label in state_labels list, or NaN if not found."""
    try:
        return state_labels.index(label)
    except ValueError:
        return np.nan


def _assign_node_states(col_matrices, node_state_labels, node_name,
                        source_col_state_labels):
    """
    col_matrices: dict {col: Series of discretized string labels (index = df.index)}
    node_state_labels: list of k state label strings for the node
    source_col_state_labels: dict {col: list of that col's state_labels}

    Returns: Series of node state labels (same index as input series).

    Strategy:
      1. Build matrix of ordinal ranks (NaN for missing/unknown values).
      2. Run k-modes with k = len(node_state_labels) on the string label matrix
         (missing values imputed with each column's modal value before k-modes).
      3. Order clusters by mean ordinal rank -> assign node_state_labels in order.
    """
    k = len(node_state_labels)

    # Build combined index from all columns
    all_idx = None
    for s in col_matrices.values():
        all_idx = s.index if all_idx is None else all_idx.union(s.index)

    if all_idx is None or len(all_idx) == 0:
        return pd.Series(dtype=object)

    # Align all column series to the union index
    mat = pd.DataFrame(index=all_idx)
    for col, s in col_matrices.items():
        mat[col] = s.reindex(all_idx)

    # Drop rows where ALL columns are NaN
    valid_mask = mat.notna().any(axis=1)
    mat_valid  = mat[valid_mask].copy()

    if len(mat_valid) == 0:
        print(f'  [SKIP] {node_name}: no valid rows for k-modes')
        return pd.Series(np.nan, index=all_idx, dtype=object)

    if len(mat_valid) < k * 2:
        raise RuntimeError(
            f'{node_name}: only {len(mat_valid)} valid rows across all source columns '
            f'— need at least {k * 2} for k-modes with k={k}.'
        )

    # Impute NaN with modal value per column
    mat_imp = mat_valid.copy()
    for col in mat_imp.columns:
        mode_val = mat_imp[col].mode()
        if len(mode_val):
            mat_imp[col] = mat_imp[col].fillna(mode_val.iloc[0])
        else:
            mat_imp[col] = mat_imp[col].fillna('unknown')
    mat_imp = mat_imp.fillna('unknown')

    # K-modes clustering — fit on sample for speed, predict on full matrix.
    # Sample size: 10% of data, clamped to [k*100, 10_000].
    # Distinct row patterns ≤ (max_states)^n_cols ≤ 256, so k*100 rows see
    # each pattern many times; 10k cap prevents stalling on 500k+ row nodes.
    _fit_n = max(k * 100, min(10_000, len(mat_imp) // 10))
    km = KModes(n_clusters=k, init='Huang', n_init=5, verbose=0)
    if len(mat_imp) > _fit_n:
        fit_sample = mat_imp.sample(n=_fit_n, random_state=42)
        km.fit(fit_sample.values)
        cluster_ids = km.predict(mat_imp.values)
    else:
        cluster_ids = km.fit_predict(mat_imp.values)

    # Compute mean ordinal rank per cluster to order them
    ordinal_mat = np.full(mat_valid.shape, np.nan)
    for j, col in enumerate(mat_valid.columns):
        col_labels = source_col_state_labels.get(col, [])
        ordinal_mat[:, j] = mat_valid[col].map(
            lambda v, cl=col_labels: _ordinal_rank(v, cl)
        ).values

    cluster_mean_rank = {}
    for c in range(k):
        rows = ordinal_mat[cluster_ids == c]
        finite = rows[np.isfinite(rows)]
        cluster_mean_rank[c] = float(np.mean(finite)) if len(finite) else 0.0

    # Sort cluster IDs by mean rank -> assign state labels in order
    sorted_clusters = sorted(cluster_mean_rank, key=lambda c: cluster_mean_rank[c])
    cluster_to_label = {cid: node_state_labels[i]
                        for i, cid in enumerate(sorted_clusters)}

    result = pd.Series(np.nan, index=all_idx, dtype=object)
    result[valid_mask] = pd.Series(cluster_ids, index=mat_valid.index).map(cluster_to_label)
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def run_discretization():
    cfg   = _load_config()
    nodes = cfg['nodes']

    for name in ('studentlife_preprocessed.csv', 'lifesnaps_preprocessed.csv', 'nhanes_preprocessed.csv'):
        path = os.path.join(_PROC, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f'Preprocessed file not found: {path} — run Steps 1-4 first.')
    sl = pd.read_csv(os.path.join(_PROC, 'studentlife_preprocessed.csv'))
    ls = pd.read_csv(os.path.join(_PROC, 'lifesnaps_preprocessed.csv'))
    nh = pd.read_csv(os.path.join(_PROC, 'nhanes_preprocessed.csv'))
    datasets = {'studentlife': sl, 'lifesnaps': ls, 'nhanes': nh}

    # Output frames: structural columns only to start
    out     = {}
    col_out = {}   # per-dataset discretized source columns (saved separately)
    for ds, df in datasets.items():
        struct    = [c for c in _STRUCTURAL if c in df.columns]
        out[ds]   = df[struct].copy()
        col_out[ds] = df[struct].copy()

    config_modified = False

    for node_name, node_cfg in nodes.items():
        scb          = node_cfg.get('source_column_bins', {})
        state_labels = node_cfg['state_labels']

        if not scb:
            print(f'  [SKIP] {node_name}: no source_column_bins defined')
            continue

        print(f'\n[{node_name}] k={len(state_labels)} states')

        # ── Phase 1: fit kmeans bins for any column that needs it ──────────
        for col, col_cfg in scb.items():
            if col_cfg['method'] == 'kmeans' and 'bin_edges' not in col_cfg:
                _fit_kmeans_bins(col, node_name, datasets, col_cfg)
                config_modified = True

        # ── Phase 1: discretize each source column per dataset ─────────────
        col_series = {}   # {col: Series across ALL rows (all datasets combined)}

        for col, col_cfg in scb.items():
            parts = []
            for ds, df in datasets.items():
                src_cols = node_cfg['source_columns'].get(ds, [])
                if col not in src_cols and col not in df.columns:
                    continue
                if col not in df.columns:
                    continue
                disc = _discretize_column(df[col], col_cfg)
                # Tag index to avoid collision between datasets
                disc.index = pd.MultiIndex.from_arrays(
                    [np.full(len(disc), ds), df.index],
                    names=['dataset', 'orig_idx']
                )
                parts.append(disc)

            if parts:
                combined = pd.concat(parts)
                col_series[col] = combined
                print(f'  col {col!r}: discretized '
                      f'({sum(p.notna().sum() for p in parts)} non-null)')
                # Save per-dataset discretized column values for likelihood tables
                for ds in datasets:
                    ds_mask = combined.index.get_level_values('dataset') == ds
                    ds_col  = combined[ds_mask]
                    ds_col.index = ds_col.index.get_level_values('orig_idx')
                    col_out[ds][col] = ds_col.reindex(col_out[ds].index)

        if not col_series:
            print(f'  [SKIP] {node_name}: no columns found in any dataset')
            continue

        # ── Phase 2: k-modes node state derivation ─────────────────────────
        source_col_state_labels = {
            col: scb[col].get('state_labels', state_labels)
            for col in col_series
        }

        node_state_series = _assign_node_states(
            col_series, state_labels, node_name, source_col_state_labels
        )
        print(f'  node states assigned: '
              f'{node_state_series.notna().sum()} / {len(node_state_series)} rows')

        # ── Unpack multi-index back to per-dataset ─────────────────────────
        for ds in datasets:
            ds_mask = node_state_series.index.get_level_values('dataset') == ds
            ds_series = node_state_series[ds_mask]
            ds_series.index = ds_series.index.get_level_values('orig_idx')
            out[ds][node_name] = ds_series.reindex(out[ds].index)

    os.makedirs(_OUT, exist_ok=True)
    for ds, df in out.items():
        path = os.path.join(_OUT, f'{ds}_discretized.csv')
        df.to_csv(path, index=False)
        print(f'Saved {len(df)} rows, {len(df.columns)} cols -> {path}')

    os.makedirs(_INTER_OUT, exist_ok=True)
    for ds, df in col_out.items():
        path = os.path.join(_INTER_OUT, f'{ds}_discretized_cols.csv')
        df.to_csv(path, index=False)
        print(f'Saved {len(df)} rows, {len(df.columns)} cols -> {path}')

    if config_modified:
        _save_config(cfg)


if __name__ == '__main__':
    run_discretization()
