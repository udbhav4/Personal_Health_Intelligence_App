/**
 * On-device DBN inference engine.
 * Port of _loopy_bp_beliefs() + temporal prior logic from codebase/backend/server.py.
 * Reads cpd-tables.json (exported by export_cpds.py) — no Python at runtime.
 */

import cpdData from '../assets/cpd-tables.json';

// ── Types ─────────────────────────────────────────────────────────────────────

type Tensor = number | Tensor[];

export type BeliefMap    = Record<string, number[]>;
export type EvidenceMap  = Record<string, string>;
export type PriorFactors = Record<string, number[]>;
export type BeliefResult = Record<string, Record<string, number>>;

interface CpdEntry {
  values:      Tensor;
  variables:   string[];
  state_names: Record<string, string[]>;
}

interface CpdTables {
  cpds:       Record<string, CpdEntry>;
  inter_trans: Record<string, number[][]>;
  states:     Record<string, string[]>;
  parents:    Record<string, string[]>;
  topo_order: string[];
}

const model = cpdData as unknown as CpdTables;

// ── LBP constants ─────────────────────────────────────────────────────────────

const LBP_MAX_ITER = 60;
const LBP_TOL      = 1e-4;

// ── Tensor helpers ────────────────────────────────────────────────────────────

function normalize(arr: number[]): number[] {
  const s = arr.reduce((a, b) => a + b, 0);
  return s > 0 ? arr.map(v => v / s) : arr.map(() => 1 / arr.length);
}

function mulVecs(a: number[], b: number[]): number[] {
  return a.map((v, i) => v * b[i]);
}

function safeVec(arr: number[], k: number): number[] {
  return arr.length === k ? arr : Array(k).fill(1 / k);
}

function scalarMul(t: Tensor, s: number): Tensor {
  if (typeof t === 'number') return t * s;
  return (t as Tensor[]).map(sub => scalarMul(sub, s));
}

function addTensors(a: Tensor, b: Tensor): Tensor {
  if (typeof a === 'number') return (a as number) + (b as number);
  return (a as Tensor[]).map((sub, i) => addTensors(sub, (b as Tensor[])[i]));
}

/**
 * Contract last axis of tensor with 1D vec.
 * Mirrors numpy's (tensor @ vec) / _safe_contract in server.py.
 */
function contractLast(t: Tensor, vec: number[]): Tensor {
  if (typeof (t as Tensor[])[0] === 'number') {
    return (t as number[]).reduce((s, v, i) => s + v * vec[i], 0);
  }
  return (t as Tensor[]).map(sub => contractLast(sub, vec));
}

/**
 * Contract axis `axis` of tensor with 1D vec.
 * Mirrors numpy's tensordot(tensor, vec, axes=([axis],[0])) in server.py.
 */
function contractAxis(t: Tensor, axis: number, vec: number[]): Tensor {
  if (axis === 0) {
    const arr = t as Tensor[];
    let result = scalarMul(arr[0], vec[0]);
    for (let i = 1; i < arr.length; i++) {
      result = addTensors(result, scalarMul(arr[i], vec[i]));
    }
    return result;
  }
  return (t as Tensor[]).map(sub => contractAxis(sub, axis - 1, vec));
}

// ── Children map (derived once) ───────────────────────────────────────────────

function buildChildren(): Record<string, string[]> {
  const { cpds, parents } = model;
  const ch: Record<string, string[]> = {};
  for (const n of Object.keys(cpds)) ch[n] = [];
  for (const [n, ps] of Object.entries(parents)) {
    for (const p of ps) {
      if (p in ch) ch[p].push(n);
    }
  }
  return ch;
}

const _children = buildChildren();

// ── Core LBP ──────────────────────────────────────────────────────────────────

export function runLBP(
  evidence:     EvidenceMap,
  priorFactors?: PriorFactors,
): BeliefMap {
  const { cpds, states, parents, topo_order } = model;

  // Initialise beliefs
  const beliefs: BeliefMap = {};
  for (const node of Object.keys(cpds)) {
    const slist = states[node];
    const k     = slist.length;
    if (node in evidence) {
      const b   = Array(k).fill(0);
      const idx = slist.indexOf(evidence[node]);
      b[idx < 0 ? 0 : idx] = 1.0;
      beliefs[node] = b;
    } else if (priorFactors && node in priorFactors) {
      beliefs[node] = normalize(safeVec(priorFactors[node], k));
    } else {
      beliefs[node] = Array(k).fill(1 / k);
    }
  }

  const freeNodes = topo_order.filter(n => !(n in evidence));
  const sweep     = [...freeNodes, ...freeNodes.slice().reverse()];

  for (let iter = 0; iter < LBP_MAX_ITER; iter++) {
    let maxDelta = 0;

    for (const node of sweep) {
      const nodeParents   = parents[node];
      const nodeChildren  = _children[node];
      const k             = states[node].length;
      let   msg: Tensor   = cpds[node].values;

      // Parent → node: contract CPT with each parent belief (reversed → last axis first)
      for (let pi = nodeParents.length - 1; pi >= 0; pi--) {
        msg = contractLast(msg, beliefs[nodeParents[pi]]);
      }

      // Child → node: accumulate child messages
      for (const child of nodeChildren) {
        const childParents  = parents[child];
        const childK        = states[child].length;
        const bc            = safeVec(beliefs[child], childK);
        const nodeAxisInChild = childParents.indexOf(node);
        if (nodeAxisInChild < 0) continue;

        // tensordot(bc, childCpt, axes=([0],[0])) — contracts child axis 0
        let lmsg: Tensor      = contractAxis(cpds[child].values, 0, bc);
        let curNodeAxis       = nodeAxisInChild;

        // Contract all child-parent axes except current node's axis (high → low)
        for (let i = childParents.length - 1; i >= 0; i--) {
          if (i === curNodeAxis) continue;
          lmsg = contractAxis(lmsg, i, beliefs[childParents[i]]);
          if (i < curNodeAxis) curNodeAxis--;
        }

        // lmsg is now 1D [k_node] — multiply into msg
        const lmsgArr = Array.isArray(lmsg) ? lmsg as number[] : [lmsg as number];
        if (lmsgArr.length === k) {
          msg = mulVecs(msg as number[], lmsgArr);
        }
      }

      // Temporal prior factor (persistent unary factor, applied every update)
      if (priorFactors && node in priorFactors) {
        const pf = safeVec(priorFactors[node], k);
        if ((msg as number[]).length === k) {
          msg = mulVecs(msg as number[], pf);
        }
      }

      const newB  = normalize(msg as number[]);
      let   delta = 0;
      for (let i = 0; i < k; i++) delta = Math.max(delta, Math.abs(newB[i] - beliefs[node][i]));
      maxDelta      = Math.max(maxDelta, delta);
      beliefs[node] = newB;
    }

    if (maxDelta < LBP_TOL) break;
  }

  return beliefs;
}

// ── Temporal prior (inter-slice) ──────────────────────────────────────────────

/**
 * Given t-1 beliefs, apply inter-slice transition matrices → prior vectors for t.
 * Port of _temporal_priors_from_last_snapshot in server.py.
 */
export function applyInterSlice(prevBeliefs: BeliefResult): PriorFactors {
  const { inter_trans, states } = model;
  const priors: PriorFactors   = {};

  for (const [node, trans] of Object.entries(inter_trans)) {
    if (!(node in prevBeliefs) || !(node in states)) continue;
    const slist   = states[node];
    const k       = slist.length;
    if (trans.length !== k || trans[0]?.length !== k) {
      console.warn(`applyInterSlice: inter_trans shape mismatch for ${node} (expected ${k}×${k}, got ${trans.length}×${trans[0]?.length ?? 0}), skipping`);
      continue;
    }
    const prevVec = slist.map(s => prevBeliefs[node]?.[s] ?? 0);
    const norm    = normalize(prevVec);
    // trans shape [k_t, k_t-1]: prior[i] = sum_j trans[i][j] * norm[j]
    const prior   = trans.map(row => row.reduce((acc, v, j) => acc + v * norm[j], 0));
    priors[node]  = normalize(prior);
  }

  return priors;
}

// ── Output formatting ─────────────────────────────────────────────────────────

// ── Topology exports (used by mcp.ts for get_changed_nodes labelling) ─────────

export const MODEL_PARENTS  = model.parents;
export const MODEL_CHILDREN = _children;
export const MODEL_STATES   = model.states;

/** Convert raw belief vectors → {node: {state: prob}} (matches server.py output format). */
export function formatBeliefs(beliefs: BeliefMap): BeliefResult {
  const result: BeliefResult = {};
  for (const [node, probs] of Object.entries(beliefs)) {
    const slist    = model.states[node];
    result[node]   = Object.fromEntries(slist.map((s, i) => [s, +probs[i].toFixed(6)]));
  }
  return result;
}
