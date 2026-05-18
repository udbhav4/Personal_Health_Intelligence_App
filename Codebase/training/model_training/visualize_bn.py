"""
Visualize the learned Bayesian Network structure.

Usage:
  python visualize_bn.py                          # reads dbn_structure.json
  python visualize_bn.py --struct dbn_structure.json

Reads:  configs/<struct>.json + configs/feature_node_config.json
Writes: DBN_visualizations/<struct>_img.png

Node colour (blue-cyan theme — cohesive with edges and atmosphere):
  Cat1 (sensor / objective)  — deep royal blue   #1565c0
  Cat2 (self-report / human) — vivid cyan        #00acc1
  Cat3 (latent / inferred)   — very dark navy    #1a2a4a

Edge style:
  forced   (domain knowledge) — bright white glow
  learned  (data-driven)      — cyan glow
  temporal (self t→t+1)       — gold arc loop
"""

import argparse
import os
import json
import math
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import FancyArrowPatch
from matplotlib.patheffects import withStroke
import networkx as nx
import numpy as np

_ROOT   = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
_CONFIG = os.path.join(_ROOT, 'configs', 'feature_node_config.json')
_STRUCT  = os.path.join(_ROOT, 'configs', 'dbn_structure.json')
_OUT_IMG = os.path.join(_ROOT, 'DBN_visualizations', 'dbn_structure_img.png')

_BG = '#020408'
_CAT_COLOR = {
    'Cat1': '#1565c0',   # deep royal blue   — sensor / objective
    'Cat2': '#00acc1',   # vivid cyan        — self-report / subjective
    'Cat3': '#1a2a4a',   # very dark navy    — latent / inferred
}
_DEFAULT_C  = '#57606f'
_TITLE_CLR  = '#a8ecec'
_LEGEND_CLR = '#dfe6e9'   # soft silver

_NODE_SIZE = 1400
_FONT_SIZE = 7


def _load(struct_path):
    with open(struct_path, encoding='utf-8') as f:
        struct = json.load(f)
    with open(_CONFIG, encoding='utf-8') as f:
        cfg = json.load(f)
    return struct, cfg['nodes']


def _build_graph(struct):
    G = nx.DiGraph()
    G.add_nodes_from(struct['trainable_nodes'])
    forced_set = {(p, c) for p, c in struct['forced_edges']}
    for p, c in struct['all_edges']:
        G.add_edge(p, c, forced=(p, c) in forced_set)
    temporal_nodes = set()
    for pair in struct.get('inter_slice_edges', []):
        if isinstance(pair, list) and len(pair) == 2:
            temporal_nodes.add(pair[0])
        elif isinstance(pair, str):
            temporal_nodes.add(pair)
    return G, forced_set, temporal_nodes


def _node_label(name, nodes_cfg):
    k = len(nodes_cfg.get(name, {}).get('state_labels', []))
    return f"{name}\n({k} states)"


def _glow_nodes(ax, G, pos, colors, node_sizes, border_colors, border_widths):
    """Degree-scaled elevation halo + smooth bloom (hubs elevated, leaves flat)."""
    # Elevation halo — only visible on high-degree nodes, fades out for leaves
    degrees  = dict(G.degree())
    max_deg  = max(degrees.values()) or 1
    nodes    = list(G.nodes())
    halo_pos = {n: (pos[n][0] + 0.022, pos[n][1] - 0.022) for n in nodes}
    halo_alphas = [0.22 * (degrees[n] / max_deg) ** 1.5 for n in nodes]
    for n, col, sz, ha in zip(nodes, colors, node_sizes, halo_alphas):
        if ha < 0.03:
            continue   # skip near-zero — leaf nodes get no halo
        nx.draw_networkx_nodes(G, halo_pos, ax=ax,
                               nodelist=[n], node_color=[col],
                               node_size=[sz * 2.2],
                               alpha=ha, linewidths=0, edgecolors='none')
    # Smooth bloom — many small steps, no visible discrete rings
    for size_mult, alpha in [(5.5,0.02),(4.0,0.04),(3.0,0.07),(2.2,0.11),(1.6,0.18),(1.2,0.28),(1.0,0.95)]:
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=colors,
            node_size=[s * size_mult for s in node_sizes],
            alpha=alpha,
            linewidths=border_widths if size_mult == 1.0 else 0,
            edgecolors=border_colors if size_mult == 1.0 else 'none',
        )


def _draw_atmosphere(ax, pos):
    """Smooth gaussian radial haze — no hard edges, no visible ellipse."""
    from matplotlib.colors import LinearSegmentedColormap
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    cx, cy  = np.mean(xs), np.mean(ys)
    xspan   = (max(xs) - min(xs)) * 0.65
    yspan   = (max(ys) - min(ys)) * 0.65
    xmin, xmax = min(xs) - xspan*0.35, max(xs) + xspan*0.35
    ymin, ymax = min(ys) - yspan*0.35, max(ys) + yspan*0.35
    Y, X = np.mgrid[ymin:ymax:300j, xmin:xmax:300j]
    dist  = np.sqrt(((X - cx) / (xspan * 0.75))**2 + ((Y - cy) / (yspan * 0.75))**2)
    glow  = np.exp(-dist**2 * 1.5)
    cmap  = LinearSegmentedColormap.from_list('atm', ['#020408', '#0d3a5c'])
    ax.imshow(glow, extent=[xmin, xmax, ymin, ymax],
              cmap=cmap, alpha=0.60, origin='lower',
              aspect='auto', zorder=0, interpolation='bilinear')


def _glow_edges(ax, G, pos, edgelist, color, width, alpha, arrowsize):
    """4-pass glow: wide diffuse bloom → tightening → sharp bright core."""
    if not edgelist:
        return
    common = dict(
        edgelist=edgelist, arrows=False,
        connectionstyle='arc3,rad=0.07',
        min_source_margin=22, min_target_margin=22,
    )
    arrow_common = dict(
        edgelist=edgelist, arrows=True,
        arrowsize=arrowsize, arrowstyle='-|>',
        connectionstyle='arc3,rad=0.07',
        min_source_margin=22, min_target_margin=22,
    )
    for w_mult, a in [(4, 0.06), (2, 0.12)]:
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=color,
                               width=width * w_mult, alpha=a, **common)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=color,
                           width=width, alpha=alpha, **arrow_common)


def _draw_temporal_loops(ax, pos, temporal_nodes):
    if not pos:
        return
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    r  = (max(xs) - min(xs) or 1.0) * 0.012
    for n in temporal_nodes:
        if n not in pos:
            continue
        x, y = pos[n]
        for lw, alpha in [(4.0, 0.15), (1.8, 0.90)]:
            ax.add_patch(FancyArrowPatch(
                posA=(x - r, y + r * 1.2),
                posB=(x + r, y + r * 1.2),
                arrowstyle='-|>', color='#ff6348',
                linewidth=lw, alpha=alpha,
                connectionstyle='arc3,rad=-0.75',
                mutation_scale=14, zorder=5,
            ))


def visualize(struct_path, out_path):
    struct, nodes_cfg    = _load(struct_path)
    G, _, temporal_nodes = _build_graph(struct)

    # ── Layout — 3 equal-density shells with organic jitter ───────────────
    by_deg  = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)
    n       = len(by_deg)
    n_core  = 2                    # just 2 hub nodes at dead center
    core    = by_deg[:n_core]
    rest    = by_deg[n_core:]
    size    = len(rest) // 3
    shells  = [core, rest[:size], rest[size:2*size], rest[2*size:]]
    shells  = [s for s in shells if s]

    radii  = [0.22, 0.58, 0.88, 1.05]
    rng    = np.random.default_rng(42)
    jitter = 0.18
    pos    = {}
    for si, shell in enumerate(shells):
        r   = radii[min(si, len(radii) - 1)]
        nin = len(shell)
        rot = rng.uniform(0, 2 * math.pi)
        for i, node in enumerate(shell):
            angle     = rot + 2 * math.pi * i / nin
            r_vary    = r + rng.uniform(-0.12, 0.12)
            pos[node] = (
                r_vary * 1.45 * math.cos(angle) + rng.uniform(-jitter, jitter),
                r_vary        * math.sin(angle) + rng.uniform(-jitter, jitter),
            )

    node_list = list(pos.keys())

    # ── Clamp outliers within elliptical boundary ─────────────────────────
    arr       = np.array([pos[n] for n in pos])
    cx, cy    = arr.mean(axis=0)
    max_rx, max_ry = 1.55, 1.05   # ellipse semi-axes matching layout stretch
    for n in list(pos.keys()):
        x, y  = pos[n]
        dx, dy = x - cx, y - cy
        ed    = math.sqrt((dx / max_rx)**2 + (dy / max_ry)**2)
        if ed > 1.0:
            pos[n] = (cx + dx / ed, cy + dy / ed)

    # ── Pull isolated nodes toward nearest neighbour ──────────────────────
    max_gap = 0.52
    for n in node_list:
        x1, y1   = pos[n]
        min_d, nb = float('inf'), None
        for m in node_list:
            if m == n: continue
            d = math.sqrt((pos[m][0]-x1)**2 + (pos[m][1]-y1)**2)
            if d < min_d:
                min_d, nb = d, m
        if nb and min_d > max_gap:
            pull    = (min_d - max_gap) / min_d * 0.5
            mx, my  = pos[nb]
            pos[n]  = (x1 + (mx - x1) * pull, y1 + (my - y1) * pull)

    # ── Separate any nodes that ended up too close ────────────────────────
    min_dist  = 0.22
    for _ in range(60):
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                n1, n2   = node_list[i], node_list[j]
                x1, y1   = pos[n1]
                x2, y2   = pos[n2]
                dx, dy   = x2 - x1, y2 - y1
                dist     = math.sqrt(dx*dx + dy*dy)
                if 1e-6 < dist < min_dist:
                    push     = (min_dist - dist) / 2
                    nx_, ny_ = dx / dist, dy / dist
                    pos[n1]  = (x1 - nx_ * push, y1 - ny_ * push)
                    pos[n2]  = (x2 + nx_ * push, y2 + ny_ * push)

    # ── Node attributes ────────────────────────────────────────────────────
    colors        = [_CAT_COLOR.get(nodes_cfg.get(n, {}).get('category', ''), _DEFAULT_C) for n in G.nodes()]
    labels        = {n: _node_label(n, nodes_cfg) for n in G.nodes()}
    border_colors = ['#ff6348' if n in temporal_nodes else '#a8ecec55' for n in G.nodes()]
    border_widths = [2.5       if n in temporal_nodes else 0.8         for n in G.nodes()]

    # Degree-based node sizing: hubs are large bright hotspots, leaves are small
    degrees   = dict(G.degree())
    max_deg   = max(degrees.values()) or 1
    node_sizes = [
        int(_NODE_SIZE * (0.3 + 2.2 * degrees[n] / max_deg))
        for n in G.nodes()
    ]

    forced_edges  = [(p, c) for p, c, d in G.edges(data=True) if d.get('forced')]
    learned_edges = [(p, c) for p, c, d in G.edges(data=True) if not d.get('forced')]

    # ── Figure ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(60, 36))
    ax.set_facecolor(_BG)
    fig.patch.set_facecolor(_BG)

    # Atmosphere → edges → nodes (back to front)
    _draw_atmosphere(ax, pos)
    _glow_edges(ax, G, pos, learned_edges, '#6ddada', 1.0, 0.45, 14)
    _glow_edges(ax, G, pos, forced_edges,  '#a8ecec', 2.0, 0.95, 18)
    _draw_temporal_loops(ax, pos, temporal_nodes)
    _glow_nodes(ax, G, pos, colors, node_sizes, border_colors, border_widths)

    texts = nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                                    font_size=_FONT_SIZE, font_color='#a8ecec',
                                    font_weight='bold')
    for t in texts.values():
        t.set_path_effects([withStroke(linewidth=2.5, foreground='#00060e')])

    # ── Legend ─────────────────────────────────────────────────────────────
    legend_nodes = [
        mpatches.Patch(color=_CAT_COLOR['Cat1'], label='Cat1 — Sensor-based     (passive / objective)'),
        mpatches.Patch(color=_CAT_COLOR['Cat2'], label='Cat2 — Self-report       (survey / EMA)'),
        mpatches.Patch(color=_CAT_COLOR['Cat3'], label='Cat3 — Latent            (inferred / hidden)'),
    ]
    legend_edges = [
        mlines.Line2D([], [], color='#a8ecec', linewidth=2,   label=f'Forced edge — neural highway ({len(forced_edges)})'),
        mlines.Line2D([], [], color='#6ddada', linewidth=1.2, label=f'Learned edge — synaptic weight ({len(learned_edges)})'),
        mlines.Line2D([], [], color='#ff6348', linewidth=1.8, label=f'Temporal loop — neural firing ({len(temporal_nodes)})'),
    ]
    ax.legend(handles=legend_nodes + legend_edges,
              loc='upper left', fontsize=12,
              framealpha=0.4, edgecolor='#00b4d844',
              facecolor='#020408', labelcolor=_LEGEND_CLR)

    ax.set_title(
        f'Bayesian Network Structure  —  '
        f'{len(G.nodes())} nodes  |  '
        f'{len(forced_edges)} forced + {len(learned_edges)} learned = {len(G.edges())} intra-edges  |  '
        f'{len(temporal_nodes)} temporal self-loops',
        fontsize=16, fontweight='bold', pad=20, color=_TITLE_CLR,
    )
    # Lock view to node bounds so atmosphere ellipses don't squish the diagram
    xs_pos = [p[0] for p in pos.values()]
    ys_pos = [p[1] for p in pos.values()]
    px = (max(xs_pos) - min(xs_pos)) * 0.12
    py = (max(ys_pos) - min(ys_pos)) * 0.18
    ax.set_xlim(min(xs_pos) - px, max(xs_pos) + px)
    ax.set_ylim(min(ys_pos) - py, max(ys_pos) + py)

    ax.axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--struct', default=None)
    args = parser.parse_args()
    struct_path = args.struct or _STRUCT
    if not os.path.isabs(struct_path):
        struct_path = os.path.join(_ROOT, 'configs', struct_path)
    out_img = _OUT_IMG if struct_path == _STRUCT else os.path.splitext(struct_path)[0] + '_img.png'
    visualize(struct_path, out_img)
