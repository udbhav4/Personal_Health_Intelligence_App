"""
Visualize the learned Bayesian Network structure.

Usage:
  python visualize_bn.py                          # reads bn_structure.json
  python visualize_bn.py --struct bn_structure_lbp.json

Reads:  configs/<struct>.json + configs/feature_node_config.json
Writes: configs/<struct>.png

Node colour:
  Cat1 (health outcomes)      — coral red   #e74c3c
  Cat2 (demographics/NHANES)  — steel blue  #2980b9
  Cat3 (behavioural/passive)  — teal        #27ae60
  Cat4 (latent mediators)     — purple      #8e44ad

Edge style:
  forced   (domain knowledge) — solid dark grey
  learned  (data-driven)      — dashed light grey
  temporal (self t→t+1)       — orange arc loop
"""

import argparse
import os
import json
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import FancyArrowPatch
import networkx as nx

_ROOT      = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
_CONFIG    = os.path.join(_ROOT, 'configs', 'feature_node_config.json')
_STRUCT    = os.path.join(_ROOT, 'configs', 'bn_structure_lbp.json')

_CAT_COLOR = {'Cat1': '#e74c3c', 'Cat2': '#2980b9', 'Cat3': '#27ae60', 'Cat4': '#8e44ad'}
_DEFAULT_C = '#95a5a6'

_NODE_SIZE = 3800
_FONT_SIZE = 8


def _load(struct_path):
    with open(struct_path) as f:
        struct = json.load(f)
    with open(_CONFIG) as f:
        cfg = json.load(f)
    return struct, cfg['nodes']


def _build_graph(struct):
    G = nx.DiGraph()
    G.add_nodes_from(struct['trainable_nodes'])

    forced_set = {(p, c) for p, c in struct['forced_edges']}
    for p, c in struct['all_edges']:
        G.add_edge(p, c, forced=(p, c) in forced_set, temporal=False)

    # Temporal self-loops stored as [[n, n], ...] pairs
    temporal_nodes = set()
    for pair in struct.get('inter_slice_edges', []):
        if isinstance(pair, list) and len(pair) == 2:
            temporal_nodes.add(pair[0])
        elif isinstance(pair, str):
            temporal_nodes.add(pair)

    return G, forced_set, temporal_nodes


def _node_label(name, nodes_cfg):
    nc = nodes_cfg.get(name, {})
    k  = len(nc.get('state_labels', []))
    return f"{name}\n({k} states)"


def _draw_temporal_loops(ax, pos, temporal_nodes, color='#e67e22'):
    if not pos:
        return
    xs    = [p[0] for p in pos.values()]
    ys    = [p[1] for p in pos.values()]
    xspan = max(xs) - min(xs) or 1.0
    yspan = max(ys) - min(ys) or 1.0
    r     = min(xspan, yspan) * 0.04   # 4% of layout scale

    for n in temporal_nodes:
        if n not in pos:
            continue
        x, y = pos[n]
        patch = FancyArrowPatch(
            posA=(x - r, y + r * 1.2),
            posB=(x + r, y + r * 1.2),
            arrowstyle='-|>',
            color=color,
            linewidth=1.8,
            connectionstyle='arc3,rad=-0.75',
            mutation_scale=14,
            zorder=5,
        )
        ax.add_patch(patch)


def visualize(struct_path, out_path):
    struct, nodes_cfg        = _load(struct_path)
    G, _, temporal_nodes = _build_graph(struct)

    # ── Layout — circular/brain shape ──────────────────────────────────────
    # sfdp packs nodes into a compact blob; neato with overlap removal gives
    # a tight organic cluster. Both approximate a "brain" silhouette better
    # than hierarchical dot or ring shell_layout.
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='sfdp',
                                           args='-Goverlap=prism -Grepulsiveforce=2.5 '
                                                '-Elen=1.2 -Gmaxiter=2000')
    except Exception:
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='neato',
                                               args='-Goverlap=false -Gsep="+15"')
        except Exception:
            try:
                pos = nx.kamada_kawai_layout(G, scale=5.0)
            except Exception:
                pos = nx.spring_layout(G, k=3.5, seed=42, iterations=500)

    # ── Node attributes ────────────────────────────────────────────────────
    colors = []
    labels = {}
    for n in G.nodes():
        cat = nodes_cfg.get(n, {}).get('category', '')
        colors.append(_CAT_COLOR.get(cat, _DEFAULT_C))
        labels[n] = _node_label(n, nodes_cfg)

    # ── Edge sets ──────────────────────────────────────────────────────────
    forced_edges  = [(p, c) for p, c, d in G.edges(data=True) if d.get('forced')]
    learned_edges = [(p, c) for p, c, d in G.edges(data=True) if not d.get('forced')]

    # ── Figure ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(40, 40))
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')

    # Temporal nodes get a distinct border
    border_colors = [
        '#e67e22' if n in temporal_nodes else 'white'
        for n in G.nodes()
    ]
    border_widths = [
        3.0 if n in temporal_nodes else 1.5
        for n in G.nodes()
    ]

    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=colors,
                           node_size=_NODE_SIZE,
                           alpha=0.92,
                           linewidths=border_widths,
                           edgecolors=border_colors)

    nx.draw_networkx_edges(G, pos, ax=ax,
                           edgelist=forced_edges,
                           edge_color='#ffffff',
                           width=2.2,
                           alpha=0.95,
                           arrows=True,
                           arrowsize=20,
                           arrowstyle='-|>',
                           connectionstyle='arc3,rad=0.08',
                           min_source_margin=28,
                           min_target_margin=28)

    nx.draw_networkx_edges(G, pos, ax=ax,
                           edgelist=learned_edges,
                           edge_color='#a0a8c0',
                           width=1.4,
                           alpha=0.80,
                           arrows=True,
                           arrowsize=16,
                           arrowstyle='-|>',
                           style='dashed',
                           connectionstyle='arc3,rad=0.08',
                           min_source_margin=28,
                           min_target_margin=28)

    _draw_temporal_loops(ax, pos, temporal_nodes)

    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                            font_size=_FONT_SIZE,
                            font_color='#ffffff',
                            font_weight='bold')

    # ── Legend ─────────────────────────────────────────────────────────────
    legend_nodes = [
        mpatches.Patch(color=_CAT_COLOR['Cat1'], label='Cat1 — Health outcomes'),
        mpatches.Patch(color=_CAT_COLOR['Cat2'], label='Cat2 — Demographics / NHANES'),
        mpatches.Patch(color=_CAT_COLOR['Cat3'], label='Cat3 — Behavioural / passive'),
        mpatches.Patch(color=_CAT_COLOR['Cat4'], label='Cat4 — Latent mediators'),
    ]
    legend_edges = [
        mlines.Line2D([], [], color='#2c3e50', linewidth=2,   linestyle='solid',  label=f'Forced edge ({len(forced_edges)})'),
        mlines.Line2D([], [], color='#7f8c8d', linewidth=1.2, linestyle='dashed', label=f'Learned edge ({len(learned_edges)})'),
        mlines.Line2D([], [], color='#e67e22', linewidth=1.8, linestyle='solid',  label=f'Temporal self-loop ({len(temporal_nodes)})'),
    ]
    ax.legend(handles=legend_nodes + legend_edges,
                       loc='upper left', fontsize=11,
                       framealpha=0.6, edgecolor='#444466',
                       facecolor='#1a1a2e', labelcolor='white')

    ax.set_title(
        f'Bayesian Network Structure  —  '
        f'{len(G.nodes())} nodes  |  '
        f'{len(forced_edges)} forced + {len(learned_edges)} learned = {len(G.edges())} intra-edges  |  '
        f'{len(temporal_nodes)} temporal self-loops',
        fontsize=15, fontweight='bold', pad=18, color='white'
    )
    ax.axis('off')
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--struct', default=None,
                        help='Path to bn_structure JSON (default: configs/bn_structure.json)')
    args = parser.parse_args()

    struct_path = args.struct or _STRUCT
    if not os.path.isabs(struct_path):
        struct_path = os.path.join(_ROOT, 'configs', struct_path)
    out_path = os.path.splitext(struct_path)[0] + '.png'

    visualize(struct_path, out_path)
