"""
validate_dbn.py — Two utilities:

  validate_sequences()
    Sequential per-user validation on validation_final.csv.
    For each user's temporal sequence:
      t=0 : b_0 = LBP(all non-NaN nodes at t=0)  — initialise memory
      t>=1: predictive_beliefs = LBP(temporal_prior from b_{t-1}, no t evidence)
            LL  += log predictive_beliefs[n][actual] for all non-NaN n at t
            acc += (argmax == actual)              for all non-NaN n at t
            b_t  = LBP(temporal_prior + all non-NaN at t as hard evidence)
            propagate b_t → t+1
    LL is one-step-ahead predictive: measures how well temporal memory alone
    predicts the next observed slice, before seeing any current evidence.

  impulse_response(impulse_node, impulse_state)
    Memory Horizon analysis via KL-divergence decay.
    Step A — Steady state (T=50, converge tol=1e-4)
    Step B — Clamp impulse_node=impulse_state at t=0
    Step C — Track KL(beliefs_t ‖ steady) for t=1..MAX_HORIZON
    Step D — Memory Horizon = first t where max KL across target nodes < 0.05 nats

Threshold constants:
  KL_THRESHOLD = 0.05 nats
    • Impulse-Response Function (IRF) 5% reversion criterion (Sims, 1980).
    • "Negligible divergence" in information theory (Cover & Thomas, 2006).
    • Matches Elman (1990) effective memory threshold for RNNs.

  STEADY_TOL = 1e-4
    • MCMC stationarity criterion (Gelman & Rubin, 1992).
    • Standard HMM forward-algorithm convergence check.

  MAX_HORIZON = 168 slices
    • 168 hours = 1 week. Covers one full behavioural cycle for health DBNs.

Usage:
  python validate_dbn.py validate
  python validate_dbn.py impulse --node stress_level --state high
  python validate_dbn.py impulse --node stress_level --state high --targets phq_severity sleep_quality
"""

import argparse
import json
import os
import pickle
import warnings
from collections import deque
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

_ROOT      = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
_TRAIN_CSV = os.path.join(_ROOT, 'datasets', 'final_dataset', 'training_final.csv')
_VAL_CSV   = os.path.join(_ROOT, 'datasets', 'final_dataset', 'validation_final.csv')
_MODEL_PKL = os.path.join(_ROOT, 'models', 'dbn_model.pkl')
_OUT_DIR   = os.path.join(_ROOT, 'models')

_LBP_MAX_ITER = 15
_LBP_TOL      = 1e-3
_KL_THRESHOLD = 0.05    # nats — Sims 1980 / Cover & Thomas 2006
_STEADY_T     = 50      # slices for steady-state burn-in
_STEADY_TOL   = 1e-4    # per-node marginal delta for steady-state convergence
_MAX_HORIZON  = 168     # 168 h = 1 week — one full behavioural cycle
_KL_EPS       = 1e-10   # numerical stability floor for KL


# ── Parallel worker ────────────────────────────────────────────────────────────

_worker_model_data = None   # initialised once per worker process by _init_worker


def _init_worker(model_tuple):
    global _worker_model_data
    _worker_model_data = model_tuple


def _validate_user_worker(uid, seq_records, avail_cols, global_step_start):
    """Per-user temporal + inference LL accumulation — runs inside a worker process."""
    (cpds_dict, inter_trans, states_dict, parents_dict, children_dict,
     topo_order, model_nodes, state_idx) = _worker_model_data

    t_total = {n: 0   for n in model_nodes}
    t_ll    = {n: 0.0 for n in model_nodes}
    i_total = {n: 0   for n in model_nodes}
    i_ll    = {n: 0.0 for n in model_nodes}

    if len(seq_records) < 2:
        return uid, t_total, t_ll, i_total, i_ll

    row0   = seq_records[0]
    ev0    = {n: row0[n] for n in avail_cols if pd.notna(row0.get(n))}
    b_prev = _loopy_bp_beliefs(cpds_dict, states_dict, parents_dict,
                               children_dict, ev0, topo_order=topo_order)

    global_step = global_step_start
    for step in range(1, len(seq_records)):
        row            = seq_records[step]
        temporal_prior = _propagate_temporal(inter_trans, b_prev)

        observed = {n: row[n] for n in avail_cols if pd.notna(row.get(n))}
        if not observed:
            b_prev = _loopy_bp_beliefs(
                cpds_dict, states_dict, parents_dict, children_dict,
                evidence={}, prior_factors=temporal_prior, topo_order=topo_order)
            global_step += 1
            continue

        obs_nodes = list(observed.keys())

        for n, actual in observed.items():
            idx = state_idx[n].get(actual, -1)
            if idx < 0:
                continue
            b = temporal_prior.get(n)
            if b is None:
                b = np.ones(len(states_dict[n])) / len(states_dict[n])
            b     = np.asarray(b, dtype=float)
            s     = b.sum()
            b     = b / s if s > 0 else b
            p_act = max(float(b[idx]), _KL_EPS)
            t_total[n] += 1
            t_ll[n]    += np.log(p_act)

        rng      = np.random.default_rng(global_step + _MASK_SEED)
        n_mask   = max(1, int(len(obs_nodes) * _MASK_FRAC))
        masked   = set(rng.choice(obs_nodes, size=n_mask, replace=False).tolist())
        evidence = {n: v for n, v in observed.items() if n not in masked}

        b_infer = _loopy_bp_beliefs(
            cpds_dict, states_dict, parents_dict, children_dict,
            evidence=evidence, prior_factors=temporal_prior, topo_order=topo_order
        )
        for n in masked:
            actual = observed[n]
            idx    = state_idx[n].get(actual, -1)
            if idx < 0:
                continue
            p_act = max(float(b_infer[n][idx]), _KL_EPS)
            i_total[n] += 1
            i_ll[n]    += np.log(p_act)

        b_prev = dict(b_infer)
        for n, actual in observed.items():
            idx = state_idx[n].get(actual, -1)
            if idx < 0:
                continue
            gt      = np.zeros(len(states_dict[n]))
            gt[idx] = 1.0
            b_prev[n] = gt

        global_step += 1

    return uid, t_total, t_ll, i_total, i_ll


# ── DBN extraction ─────────────────────────────────────────────────────────────

class _SliceCPD:
    """Wraps a DBN TabularCPD, stripping time-slice indices from all variable tuples."""
    def __init__(self, cpd):
        self.variable    = cpd.variable[0]
        self.variables   = [v[0] for v in cpd.variables]
        self.values      = cpd.values
        self.state_names = {v[0]: list(states)
                            for v, states in cpd.state_names.items()}


def _load_model():
    with open(_MODEL_PKL, 'rb') as f:
        return pickle.load(f)



def _extract_model(dbn):
    """
    Extract intra-slice BN and inter-slice transition matrices from a fitted DBN.

    Returns
    -------
    cpds_dict    : {node: _SliceCPD}
    inter_trans  : {node: ndarray (k, k)}   P(n_t | n_{t-1}) for temporal nodes
    states_dict  : {node: [state_labels]}
    parents_dict : {node: [parent_names]}
    children_dict: {node: [child_names]}
    """
    cpds_dict   = {}
    inter_trans = {}
    states_dict = {}

    for cpd in dbn.cpds:
        var = cpd.variable
        try:
            node_name, t = var
        except (TypeError, ValueError):
            continue
        if t == 0:
            sc = _SliceCPD(cpd)
            cpds_dict[node_name]   = sc
            states_dict[node_name] = list(cpd.state_names.get(var, []))
        elif t == 1:
            # pgmpy replicates intra-slice structure to t=1, so the t=1 CPD
            # includes both the self-loop (n,0)→(n,1) AND intra-slice (p,1)→(n,1)
            # parents.  Marginalise out the t=1 parents (uniform) to recover the
            # 2-d marginal transition  P(n_t | n_{t-1}).
            vals    = cpd.values.astype(float)
            t1_axes = []
            for axis, pvar in enumerate(cpd.variables[1:], start=1):
                try:
                    _, p_t = pvar
                    if p_t == 1:
                        t1_axes.append(axis)
                except (TypeError, ValueError):
                    pass
            for ax in sorted(t1_axes, reverse=True):
                vals = vals.mean(axis=ax)
            inter_trans[node_name] = vals   # shape (k_n, k_n_prev) after marginalisation

    parents_dict  = {n: cpd.variables[1:] for n, cpd in cpds_dict.items()}
    children_dict = {n: [] for n in cpds_dict}
    for n, parents in parents_dict.items():
        for p in parents:
            if p in children_dict:
                children_dict[p].append(n)

    topo_order = _topological_sort(parents_dict)
    return cpds_dict, inter_trans, states_dict, parents_dict, children_dict, topo_order


# ── Safe tensor contraction ───────────────────────────────────────────────────

def _safe_contract(msg, b):
    """Contract last axis of msg with b. Falls back to uniform if sizes mismatch."""
    b = np.asarray(b, dtype=float).ravel()
    k = msg.shape[-1]
    if len(b) != k:
        b = np.ones(k) / k
    return np.atleast_1d(msg @ b)


def _topological_sort(parents_dict):
    """Kahn's algorithm — parents before children."""
    in_deg   = {n: 0 for n in parents_dict}
    children = {n: [] for n in parents_dict}
    for n, ps in parents_dict.items():
        for p in ps:
            if p in children:
                children[p].append(n)
                in_deg[n] += 1
    q     = deque(n for n, d in in_deg.items() if d == 0)
    order = []
    while q:
        n = q.popleft()
        order.append(n)
        for c in children[n]:
            in_deg[c] -= 1
            if in_deg[c] == 0:
                q.append(c)
    remaining = set(parents_dict) - set(order)
    order.extend(sorted(remaining))
    return order


# ── Sum-product Loopy BP ───────────────────────────────────────────────────────

def _loopy_bp_beliefs(cpds_dict, states_dict, parents_dict, children_dict,
                       evidence, prior_factors=None, topo_order=None,
                       max_iter=_LBP_MAX_ITER, tol=_LBP_TOL):
    """
    Sum-product loopy BP on intra-slice BN.

    evidence      : {node: state_str} — hard-clamped nodes (belief fixed to one-hot)
    prior_factors : {node: ndarray}   — temporal-prior messages (from _propagate_temporal).
                                        Treated as persistent unary factors multiplied into
                                        every belief update, not just initialisation — this
                                        is the correct factor-graph treatment of inter-slice
                                        messages.

    Returns {node: ndarray (k,)} — posterior marginals.
    """
    beliefs = {}
    for n in cpds_dict:
        slist = states_dict[n]
        k     = len(slist)
        if n in evidence:
            b   = np.zeros(k)
            idx = slist.index(evidence[n]) if evidence[n] in slist else 0
            b[idx] = 1.0
        elif prior_factors and n in prior_factors:
            # warm-start from temporal prior so convergence is faster
            raw = np.asarray(prior_factors[n], dtype=float).ravel()
            b   = raw if len(raw) == k else np.ones(k) / k
            s   = b.sum()
            b   = b / s if s > 0 else np.ones(k) / k
        else:
            b = np.ones(k) / k
        beliefs[n] = b

    free_nodes = [n for n in cpds_dict if n not in evidence]
    if topo_order is not None:
        free_set = set(free_nodes)
        fwd      = [n for n in topo_order if n in free_set]
        sweep    = fwd + list(reversed(fwd))
    else:
        sweep = free_nodes

    for _ in range(max_iter):
        max_delta = 0.0
        for node in sweep:
            parents  = parents_dict[node]
            children = children_dict[node]
            cpt      = cpds_dict[node].values

            msg = cpt.copy()
            for p in reversed(parents):
                msg = _safe_contract(msg, beliefs[p])

            for child in children:
                child_cpd     = cpds_dict[child]
                child_parents = child_cpd.variables[1:]
                node_axis     = child_parents.index(node)
                child_cpt     = child_cpd.values

                child_k = child_cpt.shape[0]
                bc      = beliefs[child]
                if len(bc) != child_k:
                    bc = np.ones(child_k) / child_k
                lmsg = np.tensordot(bc, child_cpt, axes=([0], [0]))
                cur_node_axis = node_axis
                for i in range(len(child_parents) - 1, -1, -1):
                    if i == cur_node_axis:
                        continue
                    expected_k = lmsg.shape[i]
                    bp = beliefs[child_parents[i]]
                    if len(bp) != expected_k:
                        bp = np.ones(expected_k) / expected_k
                    lmsg = np.tensordot(lmsg, bp, axes=([i], [0]))
                    if i < cur_node_axis:
                        cur_node_axis -= 1
                msg = msg * lmsg

            # Temporal prior: persistent unary factor (inter-slice message from t-1)
            if prior_factors and node in prior_factors:
                pf = np.asarray(prior_factors[node], dtype=float).ravel()
                if len(pf) == len(msg):
                    msg = msg * pf

            s     = msg.sum()
            new_b = msg / s if s > 0 else np.ones(len(msg)) / len(msg)
            max_delta = max(max_delta, float(np.abs(new_b - beliefs[node]).max()))
            beliefs[node] = new_b

        if max_delta < tol:
            break

    return beliefs


# ── KL divergence ──────────────────────────────────────────────────────────────

def _kl_div(p, q, eps=_KL_EPS):
    """KL(p ‖ q) in nats."""
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


# ── Temporal propagation ───────────────────────────────────────────────────────

def _propagate_temporal(inter_trans, beliefs):
    """P(n_t) = Σ P(n_t | n_{t-1}) * b_{t-1}(n) for temporal nodes with self-loops.
    Nodes whose t=1 CPD had no t=0 parent (1-d trans) return the marginal directly."""
    priors = {}
    for node, trans in inter_trans.items():
        if node not in beliefs:
            continue
        if trans.ndim == 1:
            s = trans.sum()
            priors[node] = trans / s if s > 0 else trans
        else:
            prior = _safe_contract(trans, beliefs[node])
            s     = prior.sum()
            priors[node] = prior / s if s > 0 else prior
    return priors


# ── Sequential validation ──────────────────────────────────────────────────────

_MASK_FRAC = 0.20   # fraction of non-NaN nodes randomly hidden for LL_inference
_MASK_SEED = 0      # per-step rng seed offset (combined with step index)


def validate_sequences(split='val'):
    """
    Sequential per-user validation — single-pass masking routine.

    split : 'val' | 'train' | 'both'
      'val'   — validation_final.csv only  (default)
      'train' — training_final.csv only
      'both'  — runs both and prints a train vs val LL comparison at the end
    """
    print('\n=== validate_sequences ===')
    dbn = _load_model()
    cpds_dict, inter_trans, states_dict, parents_dict, children_dict, topo_order = _extract_model(dbn)
    model_nodes = set(cpds_dict.keys())

    csv_map       = {'train': _TRAIN_CSV, 'val': _VAL_CSV}
    splits_to_run = ['train', 'val'] if split == 'both' else [split]

    def _run_split(df, label):
        print(f'\n── {label.upper()} split ──')
        print(f'  Rows      : {len(df):,}')
        print(f'  Users     : {df["user_id"].nunique()}')
        print(f'  Mask frac : {_MASK_FRAC:.0%}  (LL_inference hidden set)')

        t_total = {n: 0   for n in model_nodes}
        t_ll    = {n: 0.0 for n in model_nodes}

        i_total = {n: 0   for n in model_nodes}
        i_ll    = {n: 0.0 for n in model_nodes}

        cs_total = {n: 0   for n in model_nodes}
        cs_ll    = {n: 0.0 for n in model_nodes}

        avail_cols = [n for n in model_nodes if n in df.columns]
        state_idx  = {n: {s: i for i, s in enumerate(states_dict[n])} for n in model_nodes}

        user_groups = [
            (uid, grp.sort_values(['date', 'hour']).reset_index(drop=True))
            for uid, grp in df.groupby('user_id')
        ]
        n_users = len(user_groups)
        print(f'  Starting temporal eval on {n_users} users ...')

        # Pre-compute per-user global_step offsets to preserve exact masking seeds
        step_starts, cumsum = [], 0
        for uid, seq in user_groups:
            step_starts.append(cumsum)
            if len(seq) >= 2:
                cumsum += len(seq) - 1

        model_tuple = (cpds_dict, inter_trans, states_dict, parents_dict,
                       children_dict, topo_order, model_nodes, state_idx)
        n_workers   = min(multiprocessing.cpu_count(), max(1, n_users))

        with ProcessPoolExecutor(max_workers=n_workers,
                                 initializer=_init_worker,
                                 initargs=(model_tuple,)) as ex:
            futs = {
                ex.submit(_validate_user_worker,
                          uid,
                          seq.to_dict('records'),
                          avail_cols,
                          step_starts[i]): (i, uid)
                for i, (uid, seq) in enumerate(user_groups)
            }
            ordered = [None] * n_users
            for fut in as_completed(futs):
                i, uid = futs[fut]
                ordered[i] = fut.result()
            for i, (uid, seq) in enumerate(user_groups):
                _, pt, pl, it, il = ordered[i]

                for n in model_nodes:
                    t_total[n] += pt[n]
                    t_ll[n]    += pl[n]
                    i_total[n] += it[n]
                    i_ll[n]    += il[n]

        nhanes_rows = df[df['user_id'].isna()].reset_index(drop=True)
        print(f'\n  NHANES cross-sectional rows: {len(nhanes_rows)}')
        for cs_idx, (_, row) in enumerate(nhanes_rows.iterrows()):
            observed = {n: row[n] for n in model_nodes
                        if n in row.index and pd.notna(row[n])}
            if not observed:
                continue
            obs_nodes = list(observed.keys())
            rng      = np.random.default_rng(cs_idx + _MASK_SEED)
            n_mask   = max(1, int(len(obs_nodes) * _MASK_FRAC))
            masked   = set(rng.choice(obs_nodes, size=n_mask, replace=False).tolist())
            evidence = {n: v for n, v in observed.items() if n not in masked}

            b_infer = _loopy_bp_beliefs(
                cpds_dict, states_dict, parents_dict, children_dict,
                evidence=evidence, topo_order=topo_order
            )
            for n in masked:
                actual = observed[n]
                slist  = states_dict.get(n, [])
                if actual not in slist:
                    continue
                b     = b_infer[n]
                p_act = max(float(b[slist.index(actual)]), _KL_EPS)
                cs_total[n] += 1
                cs_ll[n]    += np.log(p_act)

        def _report(title, totals, lls, nodes):
            print(f'\n{title}')
            print(f'{"Node":<30} {"MeanLL":>12} {"N":>8}')
            print('-' * 53)
            out, mlls = {}, []
            for n in sorted(nodes):
                tot = totals[n]
                if tot == 0:
                    continue
                mll = lls[n] / tot
                print(f'{n:<30} {mll:>12.4f} {tot:>8}')
                out[n] = {'mean_log_likelihood': round(mll, 4), 'n': tot}
                mlls.append(mll)
            if mlls:
                joint = float(np.mean(mlls))
                print(f'\n  Joint model mean log-likelihood: {joint:.4f}')
                out['__joint__'] = round(joint, 4)
            return out

        res_t  = _report(f'── LL_temporal  [{label}]',
                         t_total, t_ll, model_nodes)
        res_i  = _report(f'── LL_inference [{label}]',
                         i_total, i_ll, model_nodes)
        res_cs = _report(f'── LL_cross_sectional [{label}]',
                         cs_total, cs_ll, model_nodes)
        return {'temporal': res_t, 'inference': res_i, 'cross_sectional': res_cs}

    all_results = {}
    for sp in splits_to_run:
        df = pd.read_csv(csv_map[sp], low_memory=False)
        all_results[sp] = _run_split(df, sp)

    if split == 'both':
        print('\n── Train vs Val joint LL ──')
        print(f'  {"metric":<22} {"train":>10} {"val":>10} {"gap (val-train)":>16}')
        print('  ' + '-' * 62)
        for metric in ('temporal', 'inference', 'cross_sectional'):
            t_j = all_results['train'][metric].get('__joint__')
            v_j = all_results['val'][metric].get('__joint__')
            if t_j is not None and v_j is not None:
                print(f'  {metric:<22} {t_j:>10.4f} {v_j:>10.4f} {v_j - t_j:>+16.4f}')
            else:
                print(f'  {metric:<22} {"N/A":>10} {"N/A":>10}')

    out_path = os.path.join(_OUT_DIR, 'model_results.json')
    os.makedirs(_OUT_DIR, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nSaved → {out_path}')
    return all_results


# ── Category helpers ───────────────────────────────────────────────────────────

def _load_node_categories():
    """Return {node_name: category} from feature_node_config.json."""
    cfg_path = os.path.join(_ROOT, 'configs', 'feature_node_config.json')
    with open(cfg_path, encoding='utf-8') as f:
        cfg = json.load(f)
    return {n: attrs.get('category', 'Cat1') for n, attrs in cfg['nodes'].items()}



# ── Impulse response ───────────────────────────────────────────────────────────

def _steady_state(cpds_dict, states_dict, parents_dict, children_dict,
                  inter_trans, topo_order=None, max_t=_STEADY_T, tol=_STEADY_TOL):
    """Run DBN forward with no evidence until marginals converge."""
    beliefs = {n: np.ones(len(states_dict[n])) / len(states_dict[n])
               for n in cpds_dict}
    for t in range(max_t):
        temporal_priors = _propagate_temporal(inter_trans, beliefs)
        new_beliefs     = _loopy_bp_beliefs(
            cpds_dict, states_dict, parents_dict, children_dict,
            evidence={}, prior_factors=temporal_priors, topo_order=topo_order
        )
        max_delta = max(float(np.abs(new_beliefs[n] - beliefs[n]).max())
                        for n in cpds_dict)
        beliefs = new_beliefs
        if max_delta < tol:
            print(f'  Steady state converged at t={t+1}  (delta={max_delta:.2e})')
            break
    else:
        print(f'  [WARN] Steady state did not converge in {max_t} slices')
    return beliefs


def _run_impulse(cpds_dict, inter_trans, states_dict, parents_dict, children_dict,
                  steady, impulse_node, impulse_state, topo_order=None, verbose=True):
    """
    Core impulse response for one (node, state) pair.
    Reuses pre-computed model and steady state.
    Returns (horizon, kl_history).
    """
    target_nodes = sorted(cpds_dict.keys())

    b0    = _loopy_bp_beliefs(cpds_dict, states_dict, parents_dict, children_dict,
                               evidence={impulse_node: impulse_state},
                               topo_order=topo_order)
    kl_t0 = {n: _kl_div(b0[n], steady[n]) for n in target_nodes if n in b0 and n in steady}

    if verbose:
        print(f'  KL at t=0: {", ".join(f"{n}={v:.4f}" for n, v in kl_t0.items())}')
        print(f'\n  {"t":>4}  {"max_KL":>10}  {"note"}')
        print('  ' + '-' * 35)

    kl_history   = []
    prev_beliefs = b0
    horizon      = None

    for t in range(1, _MAX_HORIZON + 1):
        temporal_priors = _propagate_temporal(inter_trans, prev_beliefs)
        curr            = _loopy_bp_beliefs(
            cpds_dict, states_dict, parents_dict, children_dict,
            evidence={}, prior_factors=temporal_priors, topo_order=topo_order
        )
        kl_vals = {n: _kl_div(curr[n], steady[n])
                   for n in target_nodes if n in curr and n in steady}
        max_kl  = max(kl_vals.values()) if kl_vals else 0.0

        entry = {'t': t, 'max_kl': round(max_kl, 6)}
        entry.update({f'kl_{n}': round(v, 6) for n, v in kl_vals.items()})
        kl_history.append(entry)

        note = ''
        if horizon is None and max_kl < _KL_THRESHOLD:
            horizon = t
            note    = '<-- HORIZON'
        if verbose:
            print(f'  {t:>4}  {max_kl:>10.5f}  {note}')

        prev_beliefs = curr
        if horizon is not None and t >= horizon + 3:
            break

    if verbose:
        if horizon is None:
            print(f'\n  Memory Horizon NOT reached within {_MAX_HORIZON} slices.')
        else:
            print(f'\n  Memory Horizon = {horizon} hour(s)')

    return horizon, kl_history


def impulse_response(impulse_node, impulse_state):
    """
    Memory Horizon analysis for a single (node, state) pair — verbose output.

    impulse_node  : node clamped at t=0
    impulse_state : state to clamp to
    target_nodes  : unused (kept for CLI compatibility; always tracks all nodes)

    Returns (horizon, kl_history).
    """
    print(f'\n=== impulse_response: {impulse_node} = {impulse_state} ===')
    dbn = _load_model()
    cpds_dict, inter_trans, states_dict, parents_dict, children_dict, topo_order = _extract_model(dbn)

    if impulse_node not in cpds_dict:
        raise ValueError(f'{impulse_node!r} not in model')
    if impulse_state not in states_dict.get(impulse_node, []):
        raise ValueError(f'{impulse_state!r} not a valid state of {impulse_node}; '
                         f'valid: {states_dict.get(impulse_node)}')

    print(f'  KL threshold : {_KL_THRESHOLD} nats')
    print(f'  Max horizon  : {_MAX_HORIZON} hours')
    print('\n  Step A — Steady state ...')
    steady = _steady_state(cpds_dict, states_dict, parents_dict, children_dict,
                           inter_trans, topo_order=topo_order)

    print(f'\n  Step B — Impulse: {impulse_node} = {impulse_state}')
    horizon, kl_history = _run_impulse(
        cpds_dict, inter_trans, states_dict, parents_dict, children_dict,
        steady, impulse_node, impulse_state, topo_order=topo_order, verbose=True
    )

    out_data = {
        'impulse_node':         impulse_node,
        'impulse_state':        impulse_state,
        'kl_threshold':         _KL_THRESHOLD,
        'memory_horizon_hours': horizon,
        'kl_history':           kl_history,
    }
    out_path = os.path.join(_OUT_DIR, f'impulse_{impulse_node}_{impulse_state}.json')
    os.makedirs(_OUT_DIR, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f'  Saved → {out_path}')
    return horizon, kl_history


def impulse_response_all():
    """
    Run impulse response for every node × every state in the model.
    Steady state computed once and shared across all runs.

    Per-node result: horizon per state + per-category target horizon + node-level mean/max.
    Summary: overall + by_impulse_category + by_target_category.

    Saves: models/impulse_response_all.json
    """
    print('\n=== impulse_response_all ===')
    dbn = _load_model()
    cpds_dict, inter_trans, states_dict, parents_dict, children_dict, topo_order = _extract_model(dbn)

    node_categories = _load_node_categories()
    # fill any model nodes absent from config with Cat1
    for n in cpds_dict:
        node_categories.setdefault(n, 'Cat1')
    categories = sorted(set(node_categories[n] for n in cpds_dict))

    print(f'  Nodes        : {len(cpds_dict)}')
    print(f'  Categories   : {categories}')
    print(f'  KL threshold : {_KL_THRESHOLD} nats')
    print(f'  Max horizon  : {_MAX_HORIZON} hours')

    print('\n  Computing steady state (once) ...')
    steady = _steady_state(cpds_dict, states_dict, parents_dict, children_dict,
                           inter_trans, topo_order=topo_order)

    all_horizons = []

    per_node   = {}
    total_runs = sum(len(states_dict[n]) for n in cpds_dict)
    done       = 0

    for node in sorted(cpds_dict.keys()):
        states        = states_dict[node]
        node_cat      = node_categories.get(node, 'Cat1')
        state_results = {}

        for state in states:
            done += 1
            print(f'  [{done}/{total_runs}] {node} = {state} ...', end='', flush=True)

            horizon, _ = _run_impulse(
                cpds_dict, inter_trans, states_dict, parents_dict, children_dict,
                steady, node, state, topo_order=topo_order, verbose=False
            )
            state_results[state] = horizon
            tag = f'{horizon}h' if horizon is not None else f'>{_MAX_HORIZON}h'
            print(f' horizon={tag}')

            if horizon is not None:
                all_horizons.append(horizon)

        finite = [h for h in state_results.values() if h is not None]
        per_node[node] = {
            'impulse_category': node_cat,
            'by_state':         state_results,
            'mean_horizon':     round(float(np.mean(finite)), 2) if finite else None,
            'max_horizon':      int(max(finite))                  if finite else None,
        }

    def _stats(horizons, n_total):
        if not horizons:
            return {'note': f'No run reached horizon within {_MAX_HORIZON} slices'}
        return {
            'mean_horizon_hours':   round(float(np.mean(horizons)),   2),
            'median_horizon_hours': round(float(np.median(horizons)), 2),
            'min_horizon_hours':    int(min(horizons)),
            'max_horizon_hours':    int(max(horizons)),
            'pct_reached':          round(len(horizons) / n_total * 100, 1),
        }

    n_runs_total = sum(len(states_dict[n]) for n in cpds_dict)

    # by_category: mean horizon of nodes *within* each category (node-level, not run-level)
    node_mean_by_cat = {cat: [] for cat in categories}
    for n, r in per_node.items():
        if r['mean_horizon'] is not None:
            node_mean_by_cat[r['impulse_category']].append(r['mean_horizon'])
    n_nodes_by_cat = {cat: sum(1 for n in cpds_dict if node_categories.get(n) == cat)
                      for cat in categories}

    summary = {
        'overall':    _stats(all_horizons, n_runs_total),
        'by_category': {cat: _stats(node_mean_by_cat[cat], n_nodes_by_cat[cat])
                        for cat in categories},
    }

    # ── Print ─────────────────────────────────────────────────────────────────
    print(f'\n── Overall Memory Horizon ──')
    for k, v in summary['overall'].items():
        print(f'  {k}: {v}')

    print(f'\n── By category ──')
    print(f'  {"Cat":<8} {"Mean":>8} {"Median":>8} {"Min":>6} {"Max":>6} {"Reached%":>10}')
    print('  ' + '-' * 52)
    for cat in categories:
        r = summary['by_category'][cat]
        if 'note' in r:
            print(f'  {cat:<8}  {r["note"]}')
        else:
            print(f'  {cat:<8} {r["mean_horizon_hours"]:>8.1f} '
                  f'{r["median_horizon_hours"]:>8.1f} '
                  f'{r["min_horizon_hours"]:>6} {r["max_horizon_hours"]:>6} '
                  f'{r["pct_reached"]:>9.1f}%')

    print(f'\n── Per-node horizons ──')
    print(f'{"Node":<30} {"Cat":<6} {"MeanH":>7} {"MaxH":>7}  By state')
    print('-' * 80)
    for n in sorted(per_node):
        r    = per_node[n]
        mh   = f'{r["mean_horizon"]}h' if r['mean_horizon'] is not None else f'>{_MAX_HORIZON}h'
        xh   = f'{r["max_horizon"]}h'  if r['max_horizon']  is not None else f'>{_MAX_HORIZON}h'
        by_s = '  '.join(f'{s}:{h if h is not None else "∞"}h'
                         for s, h in r['by_state'].items())
        print(f'{n:<30} {r["impulse_category"]:<6} {mh:>7} {xh:>7}  {by_s}')

    out_data = {
        'kl_threshold':    _KL_THRESHOLD,
        'max_horizon_cap': _MAX_HORIZON,
        'summary':         summary,
        'per_node':        per_node,
    }
    out_path = os.path.join(_OUT_DIR, 'impulse_response_all.json')
    os.makedirs(_OUT_DIR, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f'\nSaved → {out_path}')
    return out_data


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DBN validation + impulse response')
    sub    = parser.add_subparsers(dest='cmd')

    p_val = sub.add_parser('validate', help='Sequential per-user accuracy + log-likelihood')
    p_val.add_argument('--split', choices=['val', 'train', 'both'], default='val',
                       help='Which split to evaluate (default: val)')
    sub.add_parser('impulse-all', help='Memory Horizon for all nodes x states')

    p_ir = sub.add_parser('impulse', help='Memory Horizon for a single node+state')
    p_ir.add_argument('--node',  required=True, help='Node to clamp at t=0')
    p_ir.add_argument('--state', required=True, help='State to clamp to')

    args = parser.parse_args()
    if args.cmd == 'impulse':
        impulse_response(args.node, args.state)
    elif args.cmd == 'impulse-all':
        impulse_response_all()
    else:
        validate_sequences(split=getattr(args, 'split', 'val'))
