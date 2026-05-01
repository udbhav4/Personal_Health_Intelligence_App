"""
Train DBN — v3: Chow-Liu warmup + Phased-Perturbation-HC Structural EM + Loopy BP (sum-product) E-step.

Changes vs train_dbn_v2.py
--------------------------
E-step  : True loopy BP (max-product) on original DAG — no triangulation,
          no junction tree, no treewidth blowup.
          O(iters × nodes × states^(max_indegree+1)) per pattern vs VE's exponential.
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
from pgmpy.estimators import BIC, BayesianEstimator, StructureScore, TreeSearch
from pgmpy.models import DiscreteBayesianNetwork, DynamicBayesianNetwork

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from pgmpy.estimators import HillClimbSearch

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT      = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
_CONFIG    = os.path.join(_ROOT, 'configs', 'feature_node_config.json')
_TRAIN_CSV = os.path.join(_ROOT, 'datasets', 'final_dataset', 'training_final.csv')
_OUT_JSON  = os.path.join(_ROOT, 'configs', 'dbn_structure.json')
_OUT_PKL   = os.path.join(_ROOT, 'models', 'dbn_model.pkl')

# ── Constants ─────────────────────────────────────────────────────────────────
_STRUCTURAL      = {'user_id', 'date', 'hour', 'dataset'}
_NULL_THRESHOLD  = 0.99
_MAX_INDEGREE            = 3
_MAX_INDEGREE_RESTRICTED = 2   # for most restricted nodes
_MAX_INDEGREE_TIGHT      = 1   # for root traits + monthly static nodes (extrav, neuroticism, sleep_disturbances)

# -------------------------------------------------------------------------
# This is a set of nodes that have been proven to overfit over the data due to extreme
# sparsity and high cardinality. Hence, these need to be treated differently by capping
# their in-degree more aggressively and using a stronger BDeu prior during parameter fitting.
# This is a pragmatic solution to prevent the EM search from overfitting to noise in these nodes.
_RESTRICTED_NODES = {
    'positive_affect', 'stress_self_efficacy', 'physical_health',
    'negative_affect', 'mental_health',
}
# root trait + monthly-only nodes: near-zero temporal signal, CPT explodes with 2+ parents
_TIGHT_NODES = {'extraversion', 'neuroticism', 'sleep_disturbances'}
# stress_self_efficacy: stuck at uniform (train -1.39) with ESS=20 — reduce to 10.
_MEDIUM_ESS_NODES = {'stress_self_efficacy'}
_HIGH_ESS_NODES   = (_RESTRICTED_NODES | _TIGHT_NODES) - _MEDIUM_ESS_NODES

_BDEU_ESS        = 5    # default
_MEDIUM_BDEU_ESS = 10   # stress_self_efficacy
_HIGH_BDEU_ESS   = 20   # remaining restricted nodes
# -------------------------------------------------------------------------
_EM_MAX_ITER     = 20
_CONVERGENCE_PAT = 3

_PHASE1_END = int(_EM_MAX_ITER * 0.35)
_PHASE2_END = int(_EM_MAX_ITER * 0.75)

_HC_STEPS_PHASE1 = 200
_HC_STEPS_PHASE2 = 300
_HC_STEPS_PHASE3 = 500
 
_FRAC_PHASE1 = 0.40
_FRAC_PHASE2 = 0.70
_FRAC_PHASE3 = 1.00

_PERTURB_N_OPS  = 12  # edge ops per perturbation kick
_PERTURB_KICK_N = 8   # extra edges stripped on a hard kick when stuck

_LBP_MAX_ITER = 15    # message-passing iterations per evidence pattern
_LBP_TOL      = 1e-3  # early-stop if max belief delta < this

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
    forced_set    = set(map(tuple, forced_edges))
    forbidden     = [(d, s) for d in dynamic_nodes for s in static_nodes
                     if (d, s) not in forced_set]
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
#  Warmup — Chow-Liu tree (identical to v2)
# ══════════════════════════════════════════════════════════════════════════════

def _chow_liu_warmup(
    df_aug:           pd.DataFrame,
    trainable_sorted: list,
    latent_names:     list,
    forced_edges:     list,
    expert_knowledge: ExpertKnowledge,
    rng,
) -> DiscreteBayesianNetwork:
    obs_nodes = [n for n in trainable_sorted if n not in latent_names]
    df_obs    = _mode_impute(df_aug[obs_nodes])

    ts   = TreeSearch(_subsample(df_obs, 0.50, rng))
    tree = ts.estimate(estimator_type='chow-liu', show_progress=False)

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

    df_imp = _mode_impute(df_aug[trainable_sorted])
    hc     = HillClimbSearch(df_imp)
    result = hc.estimate(
        start_dag=seed,
        scoring_method=_PerNodeBIC(df_imp),
        tabu_length=max(len(forced_edges), 10),
        max_indegree=_MAX_INDEGREE,
        expert_knowledge=expert_knowledge,
        max_iter=80,
        show_progress=False,
    )
    print(f'  Chow-Liu warmup edges: {len(result.edges())}')
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Structure sanitization — remove reverse-of-required edges, re-add required
# ══════════════════════════════════════════════════════════════════════════════

def _sanitize_structure(model, forced_edges):
    """
    HC can remove a required edge mid-search for BIC gain then add its reverse.
    Strip all reverse-of-required edges and re-enforce required edges before any
    HC call, so the estimator never opens with a contradicted required edge.
    """
    forced_set = set(map(tuple, forced_edges))
    result     = model.copy()
    for p, c in forced_set:
        if result.has_edge(c, p):
            result.remove_edge(c, p)
    for p, c in forced_set:
        if not result.has_edge(p, c):
            try:
                result.add_edge(p, c)
            except Exception:
                pass
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Structure perturbation (phase 1+2 always, phase 3 never)
# ══════════════════════════════════════════════════════════════════════════════

def _hard_kick(model, forced_edges, n_remove, rng):
    """Strip n_remove random non-forced edges to escape a local optimum plateau."""
    forced_set    = set(map(tuple, forced_edges))
    removable     = [e for e in model.edges() if tuple(e) not in forced_set]
    if not removable:
        return model
    kicked = model.copy()
    chosen = rng.choice(len(removable), size=min(n_remove, len(removable)), replace=False)
    for i in chosen:
        u, v = removable[i]
        try:
            kicked.remove_edge(u, v)
        except Exception:
            pass
    return kicked


def _perturb_structure(model, trainable_sorted, forced_edges, expert_knowledge, n_ops, rng):
    import networkx as nx
    forced_set    = set(map(tuple, forced_edges))
    # Combine: reverses of forced edges + all expert_knowledge forbidden edges
    ek_forbidden  = set(map(tuple, getattr(expert_knowledge, 'forbidden_edges', []) or []))
    all_forbidden = {(c, p) for p, c in forced_set} | ek_forbidden
    perturbed     = model.copy()
    ops           = ['add', 'remove', 'reverse']

    for _ in range(n_ops):
        op   = str(rng.choice(ops))
        u, v = (str(x) for x in rng.choice(trainable_sorted, size=2, replace=False))
        try:
            if op == 'add' and not perturbed.has_edge(u, v) \
                    and (u, v) not in forced_set and (u, v) not in all_forbidden:
                perturbed.add_edge(u, v)
            elif op == 'remove' and perturbed.has_edge(u, v) and (u, v) not in forced_set:
                perturbed.remove_edge(u, v)
            elif op == 'reverse' and perturbed.has_edge(u, v) \
                    and (u, v) not in forced_set and (v, u) not in all_forbidden:
                perturbed.remove_edge(u, v)
                try:
                    perturbed.add_edge(v, u)
                except Exception:
                    perturbed.add_edge(u, v)
        except Exception:
            pass

    if not nx.is_directed_acyclic_graph(perturbed):
        return model
    return perturbed


# ══════════════════════════════════════════════════════════════════════════════
#  E-step — True Loopy BP (max-product)
# ══════════════════════════════════════════════════════════════════════════════

def _loopy_bp_beliefs_estep(cpds_dict, states_dict, parents_dict, children_dict,
                             evidence, max_iter, tol):
    """
    Sum-product loopy BP on original DAG — returns full marginal beliefs.

    Parent message  : contract CPT over parent beliefs (right-to-left matmul).
    Child λ-message : contract child CPT over child belief + other-parent beliefs,
                      then multiply; axis tracking processes dims in descending order
                      so higher axes are removed before lower ones — avoids index shifts.

    Returning beliefs (not argmax) keeps the soft-assignment interface; callers
    take argmax for hard-EM assignment now, and can switch to expected counts
    (soft-EM) later without changing this function.
    """
    nodes = list(cpds_dict.keys())

    beliefs = {}
    for n in nodes:
        slist = states_dict[n]
        k     = len(slist)
        if n in evidence:
            b   = np.zeros(k)
            val = evidence[n]
            b[slist.index(val) if val in slist else 0] = 1.0
        else:
            b = np.ones(k) / k
        beliefs[n] = b

    free_nodes = [n for n in nodes if n not in evidence]

    for _ in range(max_iter):
        max_delta = 0.0

        for node in free_nodes:
            parents  = parents_dict[node]
            children = children_dict[node]
            cpt      = cpds_dict[node].values   # (k_node, k_p1, ..., k_pN)

            # ── Parent message ────────────────────────────────────────────────
            # Contract CPT over parent beliefs from right axis to left.
            msg = cpt.copy()
            for p in reversed(parents):
                msg = msg @ beliefs[p]
            # msg shape: (k_node,)

            # ── Child λ-messages ──────────────────────────────────────────────
            for child in children:
                child_cpd     = cpds_dict[child]
                child_parents = child_cpd.variables[1:]   # list, not including child itself
                node_axis     = child_parents.index(node) # node's position in child's parent list
                child_cpt     = child_cpd.values          # (k_child, k_pa0, ..., k_paN)

                # Contract child dimension (axis 0) with child belief
                lmsg = np.tensordot(beliefs[child], child_cpt, axes=([0], [0]))
                # lmsg shape: (k_pa0, ..., k_paN); node at axis node_axis

                # Contract all other parent dims in descending order.
                # Processing high → low ensures that removing axis j > i
                # never shifts the current index of a lower axis i.
                cur_node_axis = node_axis
                for i in range(len(child_parents) - 1, -1, -1):
                    if i == cur_node_axis:
                        continue
                    # axis i in current lmsg corresponds to child_parents[i]
                    # because only higher axes (already contracted) have been removed
                    lmsg = np.tensordot(lmsg, beliefs[child_parents[i]], axes=([i], [0]))
                    if i < cur_node_axis:
                        cur_node_axis -= 1
                # lmsg shape: (k_node,)

                msg = msg * lmsg

            # ── Normalize + convergence check ─────────────────────────────────
            s     = msg.sum()
            new_b = msg / s if s > 0 else np.ones(len(msg)) / len(msg)
            max_delta = max(max_delta, float(np.abs(new_b - beliefs[node]).max()))
            beliefs[node] = new_b

        if max_delta < tol:
            break

    return beliefs


def _lbp_worker(args: tuple) -> list:
    model_bytes, evidence_list, latent_names, fallback = args
    model = pickle.loads(model_bytes)

    # Pre-extract graph info once per worker (amortised over all evidence patterns)
    cpds_dict     = {cpd.variable: cpd for cpd in model.cpds}
    states_dict   = {}
    for cpd in model.cpds:
        for var, states in cpd.state_names.items():
            if var not in states_dict:
                states_dict[var] = list(states)
    parents_dict  = {n: list(model.get_parents(n))  for n in model.nodes()}
    children_dict = {n: list(model.get_children(n)) for n in model.nodes()}

    results = []
    for ev in evidence_list:
        try:
            beliefs = _loopy_bp_beliefs_estep(
                cpds_dict, states_dict, parents_dict, children_dict,
                ev, _LBP_MAX_ITER, _LBP_TOL,
            )
            # Hard assignment (argmax) for current hard-EM; swap for expected
            # counts here when upgrading to soft-EM in a future step.
            results.append({n: states_dict[n][int(np.argmax(beliefs[n]))]
                            for n in latent_names})
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

    chunk_results = pool.map(_lbp_worker, worker_args) if pool else [_lbp_worker(a) for a in worker_args]

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

class _PerNodeBIC(StructureScore):
    """BIC wrapper that hard-caps in-degree per node tier during HC search."""
    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self._bic = BIC(data, **kwargs)

    def local_score(self, variable, parents):
        if variable in _TIGHT_NODES and len(parents) > _MAX_INDEGREE_TIGHT:
            return -float('inf')
        if variable in _RESTRICTED_NODES and len(parents) > _MAX_INDEGREE_RESTRICTED:
            return -float('inf')
        return self._bic.local_score(variable, parents)


def _mstep_structure(df_aug, current_structure, forced_edges, expert_knowledge,
                      trainable_sorted, hc_steps, perturb, rng):
    seed = (
        _perturb_structure(current_structure, trainable_sorted, forced_edges, expert_knowledge, _PERTURB_N_OPS, rng)
        if perturb
        else current_structure
    )
    seed          = _sanitize_structure(seed, forced_edges)
    clean_current = _sanitize_structure(current_structure, forced_edges)

    df_imp = _mode_impute(df_aug[trainable_sorted])
    hc     = HillClimbSearch(df_imp)
    hc_kwargs = dict(
        scoring_method=_PerNodeBIC(df_imp),
        tabu_length=max(len(forced_edges), 10),
        max_indegree=_MAX_INDEGREE,
        expert_knowledge=expert_knowledge,
        max_iter=hc_steps,
        show_progress=False,
    )
    try:
        return hc.estimate(start_dag=seed, **hc_kwargs)
    except Exception as e:
        print(f'  [mstep] perturbed seed failed: {e}')
    try:
        return hc.estimate(start_dag=clean_current, **hc_kwargs)
    except Exception as e:
        print(f'  [mstep] fallback HC failed: {e}')
    return clean_current


def _mstep_params(structure, df_aug, state_names):
    # No mode imputation: NaN rows are dropped per-CPD by BayesianEstimator,
    # avoiding the bias of counting missing values toward the modal state.
    df_fit    = df_aug[sorted(structure.nodes())]
    estimator = BayesianEstimator(structure, df_fit, state_names=state_names)
    cpds = [
        estimator.estimate_cpd(n, prior_type='BDeu',
                               equivalent_sample_size=(
                                   _HIGH_BDEU_ESS   if n in _HIGH_ESS_NODES   else
                                   _MEDIUM_BDEU_ESS if n in _MEDIUM_ESS_NODES else
                                   _BDEU_ESS))
        for n in structure.nodes()
    ]
    structure.add_cpds(*cpds)
    return structure


# ══════════════════════════════════════════════════════════════════════════════
#  DBN construction & fitting
# ══════════════════════════════════════════════════════════════════════════════

def _fit_dbn_bdeu(dbn, pairs_df, state_names, ess):
    # BayesianEstimator can't handle DynamicNode objects directly.
    # Build a mirror BayesianNetwork with plain-tuple node names that match
    # pairs_df columns, fit BDeu there, then hand the CPDs to the DBN.
    edges  = [(tuple(p), tuple(c)) for p, c in dbn.edges()]
    mirror = DiscreteBayesianNetwork(edges)
    dbn_states = {(n, t): state_names[n] for n, t in mirror.nodes()}
    be   = BayesianEstimator(mirror, pairs_df, state_names=dbn_states)
    cpds = be.get_parameters(prior_type='BDeu', equivalent_sample_size=ess)
    dbn.add_cpds(*cpds)


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

    # NaN values are left as-is; BayesianEstimator drops them per-CPD rather
    # than imputing, so missing observations don't inflate any state's count.
    df = pd.DataFrame(pair_rows)
    # pd.DataFrame promotes tuple keys to MultiIndex; flatten to flat tuple Index
    # so BayesianEstimator can access columns as ('node', 0) / ('node', 1).
    df.columns = pd.Index(df.columns.tolist())
    return df


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
        'inter_slice_edges': [[n, n] for n in sorted(temporal_nodes)],
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

    # Fix #3: validate constraints are self-consistent before training starts
    import networkx as nx
    _forced_g = nx.DiGraph(forced_edges)
    if not nx.is_directed_acyclic_graph(_forced_g):
        raise ValueError(f'forced_edges contain a cycle: {list(nx.simple_cycles(_forced_g))}')
    _ek_forbidden = set(map(tuple, getattr(expert_knowledge, 'forbidden_edges', []) or []))
    _conflicts = {e for e in map(tuple, forced_edges) if e in _ek_forbidden}
    if _conflicts:
        raise ValueError(f'required_edges ∩ forbidden_edges is non-empty: {_conflicts}')

    print('\n=== Step 5: Bootstrap latents ===')
    df_aug = _init_latent_random(df[trainable_sorted], latent_names, nodes_cfg, rng)

    print('\n=== Step 6: Chow-Liu warmup ===')
    current_structure = _chow_liu_warmup(
        df_aug, trainable_sorted, latent_names, forced_edges, expert_knowledge, rng
    )
    current_structure = _sanitize_structure(current_structure, forced_edges)

    print('\n=== Step 7: Phased Structural EM (Loopy BP E-step) ===')
    print(f'  Phases — 1: iters 0-{_PHASE1_END-1} (perturb+HC200) | 2: {_PHASE1_END}-{_PHASE2_END-1} (perturb+HC300) | 3: {_PHASE2_END}-{_EM_MAX_ITER-1} (HC500)')
    print(f'  LBP: max_iter={_LBP_MAX_ITER} tol={_LBP_TOL}')
    prev_edges    = None
    no_change_cnt = 0
    prev_phase    = 0
    fallback      = {n: nodes_cfg[n]['state_labels'][0] for n in latent_names}

    _mp_pool = mp.get_context('spawn').Pool(_N_WORKERS) if _N_WORKERS > 1 else None

    for it in range(_EM_MAX_ITER):
        if it < _PHASE1_END:
            phase, frac, hc_steps = 1, _FRAC_PHASE1, _HC_STEPS_PHASE1
        elif it < _PHASE2_END:
            phase, frac, hc_steps = 2, _FRAC_PHASE2, _HC_STEPS_PHASE2
        else:
            phase, frac, hc_steps = 3, _FRAC_PHASE3, _HC_STEPS_PHASE3

        if phase != prev_phase:
            no_change_cnt = 0
            prev_phase    = phase

        print(f'\n  --- iter {it+1}/{_EM_MAX_ITER} | phase={phase} frac={frac:.0%} hc={hc_steps} perturb={"Y" if phase < 3 else "N"} ---')

        df_iter      = _subsample(df_aug, frac, rng)
        print('  M-step: fit CPTs ...')
        fitted_model = _mstep_params(current_structure, df_iter, state_names)

        n_uniq = len(df_aug[obs_names].drop_duplicates())
        print(f'  E-step (LBP): {len(df_aug):,} rows → {n_uniq:,} unique patterns, {_N_WORKERS} worker(s) ...')
        df_aug = _estep(fitted_model, df_aug, latent_names, obs_names, _N_WORKERS, fallback, _mp_pool)

        df_iter_aug   = _subsample(df_aug, frac, rng)
        print('  M-step: HillClimbSearch ...')
        new_structure = _mstep_structure(
            df_iter_aug, current_structure, forced_edges,
            expert_knowledge, trainable_sorted, hc_steps,
            phase < 3, rng
        )
        edge_set = frozenset(new_structure.edges())
        print(f'  Edges: {len(edge_set)}')

        if edge_set == prev_edges:
            no_change_cnt += 1
            print(f'  Unchanged ({no_change_cnt}/{_CONVERGENCE_PAT})')
            if no_change_cnt >= _CONVERGENCE_PAT and phase == 3:
                print('  Converged.')
                current_structure = new_structure
                break
            if phase < 3:
                print(f'  Hard kick: stripping {_PERTURB_KICK_N} edges to escape plateau ...')
                current_structure = _hard_kick(new_structure, forced_edges, _PERTURB_KICK_N, rng)
                continue
        else:
            no_change_cnt = 0

        prev_edges        = edge_set
        current_structure = new_structure

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
    _fit_dbn_bdeu(dbn, pairs_df, state_names, _BDEU_ESS)
    print(f'  CPDs: {len(dbn.cpds)} (BDeu ESS={_BDEU_ESS})')

    print('\n=== Step 11: Save ===')
    _save(dbn, fitted_final, forced_edges, trainable, excluded, temporal_nodes, latent_names)

    return dbn


if __name__ == '__main__':
    mp.freeze_support()
    train_dbn()
