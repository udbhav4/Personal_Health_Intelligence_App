"""
Train DBN — v2: Chow-Liu warmup + Epsilon-Greedy Structural EM.

Changes vs train_dbn.py
-----------------------
Warmup  : Chow-Liu tree (max spanning tree of MI) replaces HC warmup.
          Better seed than random HC — captures strongest pairwise signals.
EM loop : Epsilon-greedy phased search.
          Phase 1 (explore) — high epsilon, shallow HC, 20% subsample.
          Phase 2 (balance) — decaying epsilon, medium HC, 30% subsample.
          Phase 3 (exploit) — pure greedy, full HC, 100% subsample.
          Epsilon decays by _EPSILON_DECAY each iteration.
E-step  : VE MAP, deduplicated + parallel (unchanged).
"""

import json
import os
import pickle
import random
import warnings
import multiprocessing as mp

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from pgmpy.causal_discovery import ExpertKnowledge
from pgmpy.estimators import BIC, TreeSearch
from pgmpy.models import DiscreteBayesianNetwork, DynamicBayesianNetwork
from pgmpy.inference import VariableElimination

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from pgmpy.estimators import HillClimbSearch

# ── Paths ────────────────────────────────────────────────────────────────────────
_ROOT      = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
_CONFIG    = os.path.join(_ROOT, 'configs', 'feature_node_config.json')
_TRAIN_CSV = os.path.join(_ROOT, 'datasets', 'final_dataset', 'training_final.csv')
_OUT_JSON  = os.path.join(_ROOT, 'configs', 'bn_structure_v2.json')
_OUT_PKL   = os.path.join(_ROOT, 'models', 'dbn_model_v2.pkl')

# ── Constants ────────────────────────────────────────────────────────────────────
_STRUCTURAL      = {'user_id', 'date', 'hour', 'dataset'}
_NULL_THRESHOLD  = 0.99
_MAX_INDEGREE    = 4
_EM_MAX_ITER     = 20
_CONVERGENCE_PAT = 3

# Phase boundaries (fraction of _EM_MAX_ITER)
_PHASE1_END = int(_EM_MAX_ITER * 0.35)   # iters 0–6  : explore
_PHASE2_END = int(_EM_MAX_ITER * 0.75)   # iters 7–14 : balance
# iters 15–19 : exploit (= _FINAL_ITERS equivalent)

# HC steps per phase — shallow early (epsilon handles diversity), deep final
_HC_STEPS_PHASE1 = 60
_HC_STEPS_PHASE2 = 150
_HC_STEPS_PHASE3 = 500

# Subsampling per phase
_FRAC_PHASE1 = 0.20
_FRAC_PHASE2 = 0.30
_FRAC_PHASE3 = 1.00

# Epsilon-greedy decay
_EPSILON_START  = 0.40   # 40% random perturbation probability at iter 0
_EPSILON_DECAY  = 0.80   # multiply per iteration
_PERTURB_N_OPS  = 4      # random edge ops when perturbing structure

_N_WORKERS = max(1, (os.cpu_count() or 2) - 1)


def _load_config() -> dict:
    with open(_CONFIG, encoding='utf-8') as f:
        return json.load(f)


def _state_names_dict(nodes_cfg: dict, trainable: set) -> dict:
    return {n: nodes_cfg[n]['state_labels'] for n in trainable if 'state_labels' in nodes_cfg.get(n, {})}


# ══════════════════════════════════════════════════════════════════════════════
#  Edge helpers
# ══════════════════════════════════════════════════════════════════════════════

def _collect_forced_edges(nodes_cfg: dict, trainable: set) -> list:
    edges = set()
    for child, nc in nodes_cfg.items():
        if child not in trainable:
            continue
        for parent in nc.get('forced_parents', []):
            if parent in trainable:
                edges.add((parent, child))
    return sorted(edges)


def _build_expert_knowledge(nodes_cfg: dict, trainable: set, forced_edges: list) -> ExpertKnowledge:
    static_nodes  = {n for n in trainable
                     if not nodes_cfg.get(n, {}).get('temporal', False)
                     and not nodes_cfg.get(n, {}).get('latent', False)}
    dynamic_nodes = trainable - static_nodes
    forbidden     = [(d, s) for d in dynamic_nodes for s in static_nodes]
    return ExpertKnowledge(forbidden_edges=forbidden, required_edges=list(forced_edges))


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


def _init_latent_random(df, latent_names, nodes_cfg, rng):
    df = df.copy()
    for name in latent_names:
        df[name] = rng.choice(nodes_cfg[name]['state_labels'], size=len(df))
    return df


def _subsample(df, frac, rng):
    if frac >= 1.0:
        return df
    return df.sample(frac=frac, random_state=int(rng.integers(1e6))).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Warmup — Chow-Liu tree
# ══════════════════════════════════════════════════════════════════════════════

def _chow_liu_warmup(
    df_aug:          pd.DataFrame,
    trainable_sorted: list,
    latent_names:    list,
    forced_edges:    list,
    expert_knowledge: ExpertKnowledge,
    rng,
) -> DiscreteBayesianNetwork:
    """
    Build initial DAG via Chow-Liu max spanning tree on observed nodes,
    then add latent nodes + forced edges + run short HC to fix constraints.
    Chow-Liu can't include latents (all-NaN) — HC finalises their connections.
    """
    obs_nodes = [n for n in trainable_sorted if n not in latent_names]
    df_obs    = _mode_impute(df_aug[obs_nodes])

    # Chow-Liu on observed nodes only
    ts   = TreeSearch(_subsample(df_obs, 0.50, rng))
    tree = ts.estimate(estimator_type='chow-liu', show_progress=False)

    # Seed: forced edges FIRST (priority), then tree edges that don't conflict
    seed = DiscreteBayesianNetwork()
    seed.add_nodes_from(trainable_sorted)
    for p, c in forced_edges:
        try:
            seed.add_edge(p, c)
        except Exception:
            pass
    for p, c in tree.edges():
        if p in seed.nodes() and c in seed.nodes():
            try:
                seed.add_edge(p, c)
            except Exception:
                pass

    # Short HC to wire latent nodes into structure
    df_imp = _mode_impute(df_aug[trainable_sorted])
    hc     = HillClimbSearch(df_imp)
    result = hc.estimate(
        start_dag=seed,
        scoring_method=BIC(df_imp),
        tabu_length=max(len(forced_edges), 10),
        max_indegree=_MAX_INDEGREE,
        expert_knowledge=expert_knowledge,
        max_iter=80,
        show_progress=False,
    )
    print(f'  Chow-Liu warmup edges: {len(result.edges())}')
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Epsilon-greedy structure perturbation
# ══════════════════════════════════════════════════════════════════════════════

def _perturb_structure(
    model:           DiscreteBayesianNetwork,
    trainable_sorted: list,
    forced_edges:    list,
    n_ops:           int,
    rng,
) -> DiscreteBayesianNetwork:
    """Randomly add/remove/reverse edges to escape local optima."""
    forced_set = set(map(tuple, forced_edges))
    nodes      = trainable_sorted
    perturbed  = model.copy()
    ops        = ['add', 'remove', 'reverse']

    for _ in range(n_ops):
        op = rng.choice(ops)
        u, v = rng.choice(nodes, size=2, replace=False)
        try:
            if op == 'add' and not perturbed.has_edge(u, v) and (u, v) not in forced_set:
                perturbed.add_edge(u, v)
            elif op == 'remove' and perturbed.has_edge(u, v) and (u, v) not in forced_set:
                perturbed.remove_edge(u, v)
            elif op == 'reverse' and perturbed.has_edge(u, v) and (u, v) not in forced_set:
                perturbed.remove_edge(u, v)
                try:
                    perturbed.add_edge(v, u)
                except Exception:
                    perturbed.add_edge(u, v)  # restore — reverse creates cycle
        except Exception:
            pass   # skip if creates cycle or violates constraints

    return perturbed


# ══════════════════════════════════════════════════════════════════════════════
#  E-step — VE MAP (unchanged from train_dbn.py)
# ══════════════════════════════════════════════════════════════════════════════

def _ve_worker(args: tuple) -> list:
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


def _estep(fitted_model, df_aug, latent_names, obs_names, n_workers, fallback, pool):
    model_nodes   = set(fitted_model.nodes())
    valid_obs     = [c for c in obs_names if c in model_nodes]
    records       = df_aug[valid_obs].to_dict('records')
    evidence_rows = [{c: v for c, v in rec.items() if pd.notna(v)} for rec in records]

    key_to_ev = {}
    row_keys  = []
    for ev in evidence_rows:
        key = tuple(sorted(ev.items()))
        key_to_ev[key] = ev
        row_keys.append(key)

    unique_keys = list(key_to_ev.keys())
    unique_evs  = [key_to_ev[k] for k in unique_keys]

    idxs        = np.array_split(np.arange(len(unique_evs)), n_workers)
    chunks      = [[unique_evs[i] for i in ix] for ix in idxs if len(ix)]
    model_bytes = pickle.dumps(fitted_model)
    worker_args = [(model_bytes, chunk, latent_names, fallback) for chunk in chunks]

    chunk_results = pool.map(_ve_worker, worker_args) if pool else [_ve_worker(a) for a in worker_args]

    flat_results  = [r for chunk in chunk_results for r in chunk]
    key_to_result = {k: r for k, r in zip(unique_keys, flat_results)}

    df_out      = df_aug.copy()
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
    df_aug:           pd.DataFrame,
    current_structure: DiscreteBayesianNetwork,
    forced_edges:     list,
    expert_knowledge: ExpertKnowledge,
    trainable_sorted: list,
    hc_steps:         int,
    epsilon:          float,
    rng,
) -> DiscreteBayesianNetwork:
    """HC from current structure; with prob epsilon, perturb seed first."""
    seed = (
        _perturb_structure(current_structure, trainable_sorted, forced_edges, _PERTURB_N_OPS, rng)
        if rng.random() < epsilon
        else current_structure
    )
    df_imp = _mode_impute(df_aug[trainable_sorted])
    hc     = HillClimbSearch(df_imp)
    return hc.estimate(
        start_dag=seed,
        scoring_method=BIC(df_imp),
        tabu_length=max(len(forced_edges), 10),
        max_indegree=_MAX_INDEGREE,
        expert_knowledge=expert_knowledge,
        max_iter=hc_steps,
        show_progress=False,
    )


def _mstep_params(structure, df_aug, state_names):
    df_imp = _mode_impute(df_aug[sorted(structure.nodes())])
    return structure.fit(df_imp, state_names=state_names)


# ══════════════════════════════════════════════════════════════════════════════
#  DBN construction & fitting
# ══════════════════════════════════════════════════════════════════════════════

def _build_dbn(intra_model, temporal_nodes):
    dbn = DynamicBayesianNetwork()
    dbn.add_edges_from(
        [((p, 0), (c, 0)) for p, c in intra_model.edges()] +
        [((n, 0), (n, 1)) for n in sorted(temporal_nodes)]
    )
    return dbn


def _build_dbn_fit_df(df_aug, all_nodes, state_names):
    df_s      = df_aug.sort_values(['user_id', 'date', 'hour']).reset_index(drop=True)
    pair_rows = []
    for i in range(len(df_s) - 1):
        r0, r1 = df_s.iloc[i], df_s.iloc[i + 1]
        if not (r0['user_id'] == r1['user_id'] and
                r0['date']    == r1['date']    and
                int(r1['hour']) - int(r0['hour']) == 1):
            continue
        pair = {}
        for n in all_nodes:
            pair[(n, 0)] = r0.get(n)
            pair[(n, 1)] = r1.get(n)
        pair_rows.append(pair)

    if not pair_rows:
        print('  [dbn] No consecutive pairs; using self-pairs.')
        for _, row in df_aug.iterrows():
            pair = {(n, 0): row.get(n) for n in all_nodes}
            pair.update({(n, 1): row.get(n) for n in all_nodes})
            pair_rows.append(pair)

    pairs_df = pd.DataFrame(pair_rows)
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
#  Save
# ══════════════════════════════════════════════════════════════════════════════

def _save(dbn, intra_model, forced_edges, trainable, excluded, temporal_nodes, latent_names):
    all_intra  = sorted(intra_model.edges())
    forced_set = set(map(tuple, forced_edges))
    learned    = sorted(e for e in all_intra if tuple(e) not in forced_set)
    structure_json = {
        'trainable_nodes':   sorted(trainable),
        'excluded_nodes':    sorted(excluded),
        'latent_nodes':      sorted(latent_names),
        'temporal_nodes':    sorted(temporal_nodes),
        'forced_edges':      [[p, c] for p, c in forced_edges],
        'learned_edges':     [[p, c] for p, c in learned],
        'all_edges':         [[p, c] for p, c in all_intra],
        'inter_slice_edges': sorted(temporal_nodes),
        'warmup':            'chow-liu',
        'em_search':         'epsilon-greedy',
    }
    os.makedirs(os.path.dirname(_OUT_PKL), exist_ok=True)
    with open(_OUT_JSON, 'w') as f:
        json.dump(structure_json, f, indent=2)
    print(f'  Saved JSON  → {_OUT_JSON}')
    with open(_OUT_PKL, 'wb') as f:
        pickle.dump(dbn, f)
    print(f'  Saved model → {_OUT_PKL}')


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def train_dbn() -> DynamicBayesianNetwork:
    rng = np.random.default_rng(42)

    print('\n=== Step 1: Config ===')
    cfg       = _load_config()
    nodes_cfg = cfg['nodes']

    print('\n=== Step 2: Load data ===')
    df_raw = pd.read_csv(_TRAIN_CSV, low_memory=False)
    print(f'  {len(df_raw):,} rows × {len(df_raw.columns)} cols')

    structural_cols = [c for c in _STRUCTURAL if c in df_raw.columns]
    df_struct = df_raw[structural_cols].copy()
    df        = df_raw.drop(columns=structural_cols)

    print('\n=== Step 3: Trainable nodes ===')
    null_rates       = df.isnull().mean()
    trainable        = set(null_rates[null_rates < _NULL_THRESHOLD].index)
    latent_names     = sorted(n for n, nc in nodes_cfg.items() if nc.get('latent'))
    trainable       |= set(latent_names)
    for name in latent_names:
        if name not in df.columns:
            df[name] = np.nan

    excluded         = set(df_raw.drop(columns=structural_cols).columns) - trainable
    trainable_sorted = sorted(trainable)
    temporal_nodes   = {n for n in trainable if nodes_cfg.get(n, {}).get('temporal', False)}
    obs_names        = sorted(trainable - set(latent_names))

    print(f'  Trainable: {len(trainable)} | Excluded: {len(excluded)} | Latent: {latent_names}')

    print('\n=== Step 4: Forced edges + constraints ===')
    forced_edges     = _collect_forced_edges(nodes_cfg, trainable)
    expert_knowledge = _build_expert_knowledge(nodes_cfg, trainable, forced_edges)
    state_names      = _state_names_dict(nodes_cfg, trainable)
    print(f'  Forced: {len(forced_edges)} edges')

    print('\n=== Step 5: Bootstrap latents ===')
    df_aug = _init_latent_random(df[trainable_sorted], latent_names, nodes_cfg, rng)

    print('\n=== Step 6: Chow-Liu warmup ===')
    current_structure = _chow_liu_warmup(
        df_aug, trainable_sorted, latent_names, forced_edges, expert_knowledge, rng
    )

    print('\n=== Step 7: Epsilon-Greedy Structural EM ===')
    print(f'  Phases — 1: iters 0-{_PHASE1_END-1} | 2: {_PHASE1_END}-{_PHASE2_END-1} | 3: {_PHASE2_END}-{_EM_MAX_ITER-1}')
    prev_edges    = None
    no_change_cnt = 0
    fallback      = {n: nodes_cfg[n]['state_labels'][0] for n in latent_names}
    epsilon       = _EPSILON_START

    _mp_pool = mp.get_context('spawn').Pool(_N_WORKERS) if _N_WORKERS > 1 else None

    for it in range(_EM_MAX_ITER):
        # Phase assignment
        if it < _PHASE1_END:
            phase, frac, hc_steps = 1, _FRAC_PHASE1, _HC_STEPS_PHASE1
        elif it < _PHASE2_END:
            phase, frac, hc_steps = 2, _FRAC_PHASE2, _HC_STEPS_PHASE2
        else:
            phase, frac, hc_steps = 3, _FRAC_PHASE3, _HC_STEPS_PHASE3

        print(f'\n  --- iter {it+1}/{_EM_MAX_ITER} | phase={phase} frac={frac:.0%} hc={hc_steps} ε={epsilon:.3f} ---')

        df_iter      = _subsample(df_aug, frac, rng)
        print('  M-step: fit CPTs ...')
        fitted_model = _mstep_params(current_structure, df_iter, state_names)

        n_uniq = len(df_aug[obs_names].drop_duplicates())
        print(f'  E-step: {len(df_aug):,} rows → {n_uniq:,} unique patterns, {_N_WORKERS} worker(s) ...')
        df_aug = _estep(fitted_model, df_aug, latent_names, obs_names, _N_WORKERS, fallback, _mp_pool)

        df_iter_aug   = _subsample(df_aug, frac, rng)
        print('  M-step: HillClimbSearch ...')
        new_structure = _mstep_structure(
            df_iter_aug, current_structure, forced_edges,
            expert_knowledge, trainable_sorted, hc_steps,
            epsilon if phase < 3 else 0.0, rng
        )
        edge_set = frozenset(new_structure.edges())
        print(f'  Edges: {len(edge_set)}')

        if edge_set == prev_edges:
            no_change_cnt += 1
            print(f'  Unchanged ({no_change_cnt}/{_CONVERGENCE_PAT})')
            if no_change_cnt >= _CONVERGENCE_PAT:
                print('  Converged.')
                current_structure = new_structure
                break
        else:
            no_change_cnt = 0

        prev_edges        = edge_set
        current_structure = new_structure
        epsilon           = max(0.0, epsilon * _EPSILON_DECAY)

    if _mp_pool is not None:
        _mp_pool.terminate()
        _mp_pool.join()

    print('\n=== Step 8: Final CPT fit ===')
    fitted_final = _mstep_params(current_structure, df_aug, state_names)

    print('\n=== Step 9: Build DBN ===')
    dbn = _build_dbn(fitted_final, temporal_nodes)
    print(f'  Intra: {len(list(dbn.get_intra_edges()))} | Inter: {len(list(dbn.get_inter_edges()))}')

    print('\n=== Step 10: Fit DBN CPTs ===')
    df_aug_full = pd.concat([df_struct.reset_index(drop=True), df_aug.reset_index(drop=True)], axis=1)
    pairs_df    = _build_dbn_fit_df(df_aug_full, trainable_sorted, state_names)
    print(f'  Pairs: {len(pairs_df):,}')
    dbn.fit(pairs_df)
    print(f'  CPDs: {len(dbn.cpds)}')

    print('\n=== Step 11: Save ===')
    _save(dbn, fitted_final, forced_edges, trainable, excluded, temporal_nodes, latent_names)

    return dbn


if __name__ == '__main__':
    mp.freeze_support()
    train_dbn()
