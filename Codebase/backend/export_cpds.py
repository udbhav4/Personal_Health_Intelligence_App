"""
export_cpds.py — Run once after training to produce configs/cpd_tables.json.

Reads models/dbn_model.pkl, extracts everything the TypeScript LBP engine
needs, and writes it to configs/cpd_tables.json.

What gets exported:
  cpds        {node: {values, variables, state_names}}   intra-slice CPD tables
  inter_trans {node: [[...]]}                            inter-slice transition matrices
  states      {node: [str, ...]}                         state label order per node
  parents     {node: [str, ...]}                         parent names per node (t=0 names only)
  topo_order  [str, ...]                                 topological node order

Usage:
  python -m codebase.backend.export_cpds
"""

import json
import os
import pickle

import numpy as np

_ROOT      = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
_MODEL_PKL = os.path.join(_ROOT, 'models', 'dbn_model.pkl')
_OUT_PATH  = os.path.join(_ROOT, 'configs', 'cpd_tables.json')


def _topological_sort(parents_dict: dict) -> list:
    from collections import deque
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
    order.extend(sorted(set(parents_dict) - set(order)))
    return order


def _extract_and_export(dbn) -> dict:
    cpds_out    = {}
    inter_trans = {}
    states_dict = {}
    parents_dict = {}

    for cpd in dbn.cpds:
        var = cpd.variable
        try:
            node_name, t = var
        except (TypeError, ValueError):
            continue

        if t == 0:
            # Strip tuple keys → plain node name strings
            state_names_clean = {
                v[0]: list(cpd.state_names.get(v, []))
                for v in cpd.variables
            }
            variables_clean = [v[0] for v in cpd.variables]
            states_dict[node_name]  = list(cpd.state_names.get(var, []))
            parents_dict[node_name] = variables_clean[1:]  # exclude self

            cpds_out[node_name] = {
                'values':      cpd.values.astype(float).tolist(),
                'variables':   variables_clean,
                'state_names': state_names_clean,
            }

        elif t == 1:
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
            # Always output a 2D matrix — if 1D (no t=1 parent, static/background node),
            # wrap as identity-like [vals] so applyInterSlice can safely do row.reduce().
            result = vals.tolist()
            if not isinstance(result[0], list):
                # 1D result means no t=1 parent — this is a static/background node.
                # Construct a proper k×k identity matrix: each state transitions to
                # itself with probability 1 (no temporal drift).
                k = len(result)
                result = [[1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]
            inter_trans[node_name] = result

    topo_order = _topological_sort(parents_dict)

    return {
        'cpds':        cpds_out,
        'inter_trans': inter_trans,
        'states':      states_dict,
        'parents':     parents_dict,
        'topo_order':  topo_order,
    }


def main():
    if not os.path.exists(_MODEL_PKL):
        raise FileNotFoundError(f'pkl not found: {_MODEL_PKL}\nTrain the DBN first.')

    print(f'Loading {_MODEL_PKL} ...')
    with open(_MODEL_PKL, 'rb') as f:
        dbn = pickle.load(f)

    print('Extracting CPD tables ...')
    payload = _extract_and_export(dbn)

    node_count  = len(payload['cpds'])
    trans_count = len(payload['inter_trans'])
    print(f'  {node_count} intra-slice nodes, {trans_count} inter-slice transitions')

    with open(_OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    size_kb = os.path.getsize(_OUT_PATH) / 1024
    print(f'Written -> {_OUT_PATH}  ({size_kb:.1f} KB)')


if __name__ == '__main__':
    main()
