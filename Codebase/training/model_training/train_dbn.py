"""
Train Dynamic Bayesian Network via full Structural EM.

Steps
-----
1. Load training_final.csv; add latent columns (all NaN)
2. Bootstrap latent columns with uniform-random state draws
3. Warm-start: HillClimbSearch on mode-imputed data (limited steps) → initial DAG
4. Structural EM loop
      M-step  : HillClimbSearch on augmented data → new structure; MLE CPT fit
      E-step  : VE MAP per row (deduplicated + parallel) → update latent columns
      Subsample: 20 % of rows in early iterations, 100 % in final three
5. Build DynamicBayesianNetwork (intra-slice learned edges + temporal self-edges)
6. Fit DBN CPTs from consecutive hourly pairs
7. Save model pickle + updated bn_structure.json

Optimisations (priority order, per design doc)
-----------------------------------------------
1  Evidence deduplication — MAP per unique observable pattern, map back
2  Parallelisation        — E-step VE across CPU cores
3  Data subsampling       — 20 % early, 100 % final three iterations
4  Search constraints     — forbidden temporal→static, required forced edges, max_indegree cap
5  Warm start             — limited HillClimbSearch as initial DAG
6  Incremental search     — 30 HC steps early, 500 in final iterations
7  Structural caching     — skipped (low impact, high effort)
"""

import json
import os
import pickle
import warnings
import multiprocessing as mp

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from pgmpy.causal_discovery import ExpertKnowledge
from pgmpy.estimators import BIC
from pgmpy.models import DiscreteBayesianNetwork, DynamicBayesianNetwork
from pgmpy.inference import VariableElimination

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from pgmpy.estimators import HillClimbSearch

# ── Paths ───────────────────────────────────────────────────────────────────────
_ROOT      = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
_CONFIG    = os.path.join(_ROOT, 'configs', 'feature_node_config.json')
_TRAIN_CSV = os.path.join(_ROOT, 'datasets', 'final_dataset', 'training_final.csv')
_OUT_JSON  = os.path.join(_ROOT, 'configs', 'bn_structure.json')
_OUT_PKL   = os.path.join(_ROOT, 'models', 'dbn_model.pkl')

# ── Constants ───────────────────────────────────────────────────────────────────
_STRUCTURAL     = {'user_id', 'date', 'hour', 'dataset'}
_NULL_THRESHOLD = 0.99
_MAX_INDEGREE   = 4
_EM_MAX_ITER    = 20
_FINAL_ITERS    = 4       # last N iterations use 100 % data + full HC
_SUBSAMPLE_FRAC = 0.30
_HC_STEPS_WARMUP = 150
_HC_STEPS_EARLY  = 100
_HC_STEPS_FULL   = 500
_CONVERGENCE_PAT = 3      # consecutive unchanged-edge iterations to stop
_N_WORKERS       = max(1, (os.cpu_count() or 2) - 1)

def _load_config() -> dict:
    with open(_CONFIG, encoding='utf-8') as f:
        return json.load(f)


def _state_names_dict(nodes_cfg: dict, trainable: set) -> dict:
    """Return {node: [state_labels]} for every trainable node."""
    return {n: nodes_cfg[n]['state_labels'] for n in trainable if 'state_labels' in nodes_cfg.get(n, {})}


# ══════════════════════════════════════════════════════════════════════════════
#  Edge helpers
# ══════════════════════════════════════════════════════════════════════════════

def _collect_forced_edges(nodes_cfg: dict, trainable: set) -> list:
    """Collect all forced (parent, child) edges where both endpoints are trainable."""
    edges = set()
    for child, nc in nodes_cfg.items():
        if child not in trainable:
            continue
        for parent in nc.get('forced_parents', []):
            if parent in trainable:
                edges.add((parent, child))
    return sorted(edges)


def _build_expert_knowledge(nodes_cfg: dict, trainable: set, forced_edges: list) -> ExpertKnowledge:
    """
    forbidden: temporal/latent node → static node
    required:  all forced edges
    """
    static_nodes   = {n for n in trainable
                      if not nodes_cfg.get(n, {}).get('temporal', False)
                      and not nodes_cfg.get(n, {}).get('latent', False)}
    dynamic_nodes  = trainable - static_nodes

    forbidden = [(d, s) for d in dynamic_nodes for s in static_nodes]
    required  = list(forced_edges)

    return ExpertKnowledge(forbidden_edges=forbidden, required_edges=required)


# ══════════════════════════════════════════════════════════════════════════════
#  Data helpers
# ══════════════════════════════════════════════════════════════════════════════

def _mode_impute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].isna().any():
            mode = df[col].mode()
            if len(mode):
                df[col] = df[col].fillna(mode.iloc[0])
    return df


def _init_latent_random(df: pd.DataFrame, latent_names: list, nodes_cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    for name in latent_names:
        states = nodes_cfg[name]['state_labels']
        df[name] = rng.choice(states, size=len(df))
    return df


def _subsample(df: pd.DataFrame, frac: float, rng: np.random.Generator) -> pd.DataFrame:
    if frac >= 1.0:
        return df
    return df.sample(frac=frac, random_state=int(rng.integers(1e6))).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
#  E-step (VE MAP) — top-level worker for multiprocessing
# ══════════════════════════════════════════════════════════════════════════════

def _ve_worker(args: tuple) -> list:
    """Infer MAP states for latent nodes given each evidence dict. Returns list of {node: state}."""
    model_bytes, evidence_list, latent_names, fallback = args
    model = pickle.loads(model_bytes)
    ve    = VariableElimination(model)
    results = []
    for ev in evidence_list:
        try:
            q = ve.map_query(variables=latent_names, evidence=ev, show_progress=False)
            results.append({n: q[n] for n in latent_names})
        except Exception:
            results.append(dict(fallback))
    return results


def _estep(
    fitted_model: DiscreteBayesianNetwork,
    df_aug:       pd.DataFrame,
    latent_names: list,
    obs_names:    list,
    n_workers:    int,
    fallback:     dict,
    pool,
) -> pd.DataFrame:
    """
    For each row build an evidence dict from observable nodes.
    Deduplicate patterns → run VE MAP once per unique pattern → map results back.
    Parallelise across n_workers processes via persistent pool.
    Returns df_aug with latent columns updated.
    """
    # Build evidence dict per row — to_dict avoids iterrows overhead on large frames
    model_nodes = set(fitted_model.nodes())
    valid_obs   = [c for c in obs_names if c in model_nodes]
    records     = df_aug[valid_obs].to_dict('records')
    evidence_rows = [{c: v for c, v in rec.items() if pd.notna(v)} for rec in records]

    # Deduplicate
    key_to_ev  = {}
    row_keys   = []
    for ev in evidence_rows:
        key = tuple(sorted(ev.items()))
        key_to_ev[key] = ev
        row_keys.append(key)

    unique_keys = list(key_to_ev.keys())
    unique_evs  = [key_to_ev[k] for k in unique_keys]

    # Exactly n_workers chunks — even load, each worker builds VE once per iteration
    idxs   = np.array_split(np.arange(len(unique_evs)), n_workers)
    chunks = [[unique_evs[i] for i in ix] for ix in idxs if len(ix)]

    model_bytes = pickle.dumps(fitted_model)
    worker_args = [(model_bytes, chunk, latent_names, fallback) for chunk in chunks]

    if pool is not None:
        chunk_results = pool.map(_ve_worker, worker_args)
    else:
        chunk_results = [_ve_worker(a) for a in worker_args]

    flat_results  = [r for chunk in chunk_results for r in chunk]
    key_to_result = {k: r for k, r in zip(unique_keys, flat_results)}

    # Map back to all rows
    df_out = df_aug.copy()
    latent_vals = {n: [] for n in latent_names}
    for key in row_keys:
        res = key_to_result[key]
        for n in latent_names:
            latent_vals[n].append(res[n])

    for n in latent_names:
        df_out[n] = latent_vals[n]

    return df_out


# ══════════════════════════════════════════════════════════════════════════════
#  M-step
# ══════════════════════════════════════════════════════════════════════════════

def _mstep_structure(
    df_aug:          pd.DataFrame,
    forced_edges:    list,
    expert_knowledge: ExpertKnowledge,
    max_iter_hc:     int,
    trainable_sorted: list,
) -> DiscreteBayesianNetwork:
    """Learn intra-slice DAG via HillClimbSearch on mode-imputed augmented data."""
    df_imp = _mode_impute(df_aug[trainable_sorted])

    seed = DiscreteBayesianNetwork()
    seed.add_nodes_from(trainable_sorted)
    for p, c in forced_edges:
        seed.add_edge(p, c)

    hc = HillClimbSearch(df_imp)
    learned = hc.estimate(
        start_dag=seed,
        scoring_method=BIC(df_imp),
        tabu_length=max(len(forced_edges), 10),
        max_indegree=_MAX_INDEGREE,
        expert_knowledge=expert_knowledge,
        max_iter=max_iter_hc,
        show_progress=False,
    )
    return learned


def _mstep_params(
    structure:   DiscreteBayesianNetwork,
    df_aug:      pd.DataFrame,
    state_names: dict,
) -> DiscreteBayesianNetwork:
    """Fit CPTs via MLE on mode-imputed augmented data."""
    df_imp = _mode_impute(df_aug[sorted(structure.nodes())])
    fitted = structure.fit(df_imp, state_names=state_names)
    return fitted


# ══════════════════════════════════════════════════════════════════════════════
#  DBN construction & fitting
# ══════════════════════════════════════════════════════════════════════════════

def _build_dbn(intra_model: DiscreteBayesianNetwork, temporal_nodes: set) -> DynamicBayesianNetwork:
    """
    Build 2-TBN:
      intra-slice edges : ((A, 0), (B, 0)) for each A→B in learned intra model
      inter-slice edges : ((node, 0), (node, 1)) for each temporal/latent node
    """
    dbn = DynamicBayesianNetwork()

    intra_edges = [((p, 0), (c, 0)) for p, c in intra_model.edges()]
    inter_edges = [((n, 0), (n, 1)) for n in sorted(temporal_nodes)]

    dbn.add_edges_from(intra_edges + inter_edges)
    return dbn


def _build_dbn_fit_df(
    df_aug:       pd.DataFrame,
    all_nodes:    list,
    state_names:  dict,
) -> pd.DataFrame:
    """
    Build a DataFrame with tuple columns (node, time_slice) from consecutive
    hourly rows belonging to the same user on the same date.
    Columns: (node, 0) and (node, 1) for t and t+1.
    """
    df_s = df_aug.sort_values(['user_id', 'date', 'hour']).reset_index(drop=True)

    pair_rows = []
    for i in range(len(df_s) - 1):
        r0 = df_s.iloc[i]
        r1 = df_s.iloc[i + 1]
        same_user = r0['user_id'] == r1['user_id']
        same_date = r0['date'] == r1['date']
        consec    = int(r1['hour']) - int(r0['hour']) == 1
        if not (same_user and same_date and consec):
            continue
        pair = {}
        for n in all_nodes:
            pair[(n, 0)] = r0.get(n)
            pair[(n, 1)] = r1.get(n)
        pair_rows.append(pair)

    if not pair_rows:
        print('  [dbn] No consecutive hourly pairs found; duplicating rows as self-pairs.')
        for _, row in df_aug.iterrows():
            pair = {}
            for n in all_nodes:
                pair[(n, 0)] = row.get(n)
                pair[(n, 1)] = row.get(n)
            pair_rows.append(pair)

    pairs_df = pd.DataFrame(pair_rows)

    # Mode-impute per column (columns are tuples)
    for col in pairs_df.columns:
        if pairs_df[col].isna().any():
            node = col[0]
            if node in state_names:
                pairs_df[col] = pairs_df[col].fillna(state_names[node][0])
            else:
                mode = pairs_df[col].mode()
                if len(mode):
                    pairs_df[col] = pairs_df[col].fillna(mode.iloc[0])

    return pairs_df


# ══════════════════════════════════════════════════════════════════════════════
#  Save outputs
# ══════════════════════════════════════════════════════════════════════════════

def _save(
    dbn:             DynamicBayesianNetwork,
    intra_model:     DiscreteBayesianNetwork,
    forced_edges:    list,
    trainable:       set,
    excluded:        set,
    temporal_nodes:  set,
    latent_names:    list,
) -> None:
    all_intra  = sorted(intra_model.edges())
    forced_set = set(map(tuple, forced_edges))
    learned    = sorted(e for e in all_intra if tuple(e) not in forced_set)

    structure_json = {
        'trainable_nodes':  sorted(trainable),
        'excluded_nodes':   sorted(excluded),
        'latent_nodes':     sorted(latent_names),
        'temporal_nodes':   sorted(temporal_nodes),
        'forced_edges':     [[p, c] for p, c in forced_edges],
        'learned_edges':    [[p, c] for p, c in learned],
        'all_edges':        [[p, c] for p, c in all_intra],
        'inter_slice_edges': sorted(temporal_nodes),
    }

    os.makedirs(os.path.dirname(_OUT_PKL), exist_ok=True)

    with open(_OUT_JSON, 'w') as f:
        json.dump(structure_json, f, indent=2)
    print(f'  Saved structure JSON → {_OUT_JSON}')

    with open(_OUT_PKL, 'wb') as f:
        pickle.dump(dbn, f)
    print(f'  Saved DBN pickle    → {_OUT_PKL}')


# ══════════════════════════════════════════════════════════════════════════════
#  Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def train_dbn() -> DynamicBayesianNetwork:
    rng = np.random.default_rng(42)

    # ── 1. Load config ─────────────────────────────────────────────────────────
    print('\n=== Step 1: Loading config ===')
    cfg       = _load_config()
    nodes_cfg = cfg['nodes']

    # ── 2. Load training data ───────────────────────────────────────────────────
    print('\n=== Step 2: Loading training data ===')
    df_raw = pd.read_csv(_TRAIN_CSV, low_memory=False)
    print(f'  Loaded {len(df_raw):,} rows × {len(df_raw.columns)} columns')

    # Keep structural cols separately for DBN pair-building later
    structural_cols = [c for c in _STRUCTURAL if c in df_raw.columns]
    df_struct = df_raw[structural_cols].copy()
    df        = df_raw.drop(columns=structural_cols)

    # ── 3. Identify trainable nodes ─────────────────────────────────────────────
    null_rates = df.isnull().mean()
    trainable  = set(null_rates[null_rates < _NULL_THRESHOLD].index)
    # Latent nodes are 100 % NaN in CSV → force-include
    latent_names = sorted(n for n, nc in nodes_cfg.items() if nc.get('latent'))
    trainable |= set(latent_names)
    # Add latent columns to df (all NaN)
    for name in latent_names:
        if name not in df.columns:
            df[name] = np.nan

    excluded = set(df_raw.drop(columns=structural_cols).columns) - trainable
    trainable_sorted = sorted(trainable)

    print(f'  Trainable nodes ({len(trainable)}): {trainable_sorted}')
    print(f'  Excluded nodes  ({len(excluded)}):  {sorted(excluded)}')
    print(f'  Latent nodes    ({len(latent_names)}): {latent_names}')

    temporal_nodes = {n for n in trainable if nodes_cfg.get(n, {}).get('temporal', False)}
    obs_names      = sorted(trainable - set(latent_names))

    # ── 4. Build forced edges + constraints ────────────────────────────────────
    print('\n=== Step 3: Collecting forced edges & constraints ===')
    forced_edges = _collect_forced_edges(nodes_cfg, trainable)
    print(f'  Forced edges ({len(forced_edges)}):')
    for p, c in forced_edges:
        print(f'    {p} → {c}')

    expert_knowledge = _build_expert_knowledge(nodes_cfg, trainable, forced_edges)
    state_names      = _state_names_dict(nodes_cfg, trainable)

    # ── 5. Bootstrap latent columns ─────────────────────────────────────────────
    print('\n=== Step 4: Bootstrapping latent columns ===')
    df_aug = _init_latent_random(df[trainable_sorted], latent_names, nodes_cfg, rng)

    # ── 6. Warm start ───────────────────────────────────────────────────────────
    print('\n=== Step 5: Warm-start HillClimbSearch ===')
    df_warm = _subsample(df_aug, 0.5, rng)
    current_structure = _mstep_structure(
        df_warm, forced_edges, expert_knowledge, _HC_STEPS_WARMUP, trainable_sorted
    )
    print(f'  Warm-start edges: {len(current_structure.edges())}')

    # ── 7. Structural EM loop ───────────────────────────────────────────────────
    print('\n=== Step 6: Structural EM loop ===')
    prev_edges    = None
    no_change_cnt = 0
    fallback      = {n: nodes_cfg[n]['state_labels'][0] for n in latent_names}

    _mp_pool = mp.get_context('spawn').Pool(_N_WORKERS) if _N_WORKERS > 1 else None

    for it in range(_EM_MAX_ITER):
        is_final = it >= (_EM_MAX_ITER - _FINAL_ITERS)
        frac     = 1.0 if is_final else _SUBSAMPLE_FRAC
        hc_steps = _HC_STEPS_FULL if is_final else _HC_STEPS_EARLY

        print(f'\n  --- EM iteration {it + 1}/{_EM_MAX_ITER} '
              f'(frac={frac:.0%}, hc_steps={hc_steps}) ---')

        df_iter = _subsample(df_aug, frac, rng)

        # M-step: fit CPTs on current structure + augmented data
        print('  M-step: fitting CPTs ...')
        fitted_model = _mstep_params(current_structure, df_iter, state_names)

        # E-step: infer latent states for ALL rows (not just subsample)
        n_unique_obs = len(df_aug[obs_names].drop_duplicates())
        print(f'  E-step: {len(df_aug):,} rows → {n_unique_obs:,} unique patterns, '
              f'{_N_WORKERS} worker(s) ...')
        df_aug = _estep(fitted_model, df_aug, latent_names, obs_names, _N_WORKERS, fallback, _mp_pool)

        # M-step: re-learn structure on freshly imputed data
        df_iter_aug = _subsample(df_aug, frac, rng)
        print('  M-step: HillClimbSearch ...')
        new_structure = _mstep_structure(
            df_iter_aug, forced_edges, expert_knowledge, hc_steps, trainable_sorted
        )
        edge_set = frozenset(new_structure.edges())
        print(f'  New structure: {len(edge_set)} edges')

        # Convergence check
        if edge_set == prev_edges:
            no_change_cnt += 1
            print(f'  Structure unchanged ({no_change_cnt}/{_CONVERGENCE_PAT})')
            if no_change_cnt >= _CONVERGENCE_PAT:
                print('  Converged — stopping EM early.')
                current_structure = new_structure
                break
        else:
            no_change_cnt = 0

        prev_edges        = edge_set
        current_structure = new_structure

    if _mp_pool is not None:
        _mp_pool.terminate()
        _mp_pool.join()

    # ── 8. Final CPT fit on full data ───────────────────────────────────────────
    print('\n=== Step 7: Final CPT fit on full data ===')
    fitted_final = _mstep_params(current_structure, df_aug, state_names)

    # ── 9. Build DynamicBayesianNetwork ─────────────────────────────────────────
    print('\n=== Step 8: Building DynamicBayesianNetwork ===')
    dbn = _build_dbn(fitted_final, temporal_nodes)
    print(f'  DBN intra-slice edges : {len(list(dbn.get_intra_edges()))}')
    print(f'  DBN inter-slice edges : {len(list(dbn.get_inter_edges()))}')

    # ── 10. Fit DBN CPTs from consecutive pairs ──────────────────────────────────
    print('\n=== Step 9: Fitting DBN CPTs from consecutive hourly pairs ===')
    df_aug_full = pd.concat(
        [df_struct.reset_index(drop=True), df_aug.reset_index(drop=True)], axis=1
    )
    pairs_df = _build_dbn_fit_df(df_aug_full, trainable_sorted, state_names)
    print(f'  Consecutive pairs: {len(pairs_df):,}')
    dbn.fit(pairs_df)
    print(f'  DBN CPDs fitted: {len(dbn.cpds)}')

    # ── 11. Save ─────────────────────────────────────────────────────────────────
    print('\n=== Step 10: Saving ===')
    _save(dbn, fitted_final, forced_edges, trainable, excluded, temporal_nodes, latent_names)

    return dbn


if __name__ == '__main__':
    mp.freeze_support()
    train_dbn()
