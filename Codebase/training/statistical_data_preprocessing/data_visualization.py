import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'configs', 'feature_node_config.json'
)
OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'configs', 'dbn_graph.png'
)

CATEGORY_COLORS = {
    'Cat1': '#2ecc71',
    'Cat2': '#3498db',
    'Cat3': '#e67e22',
}

LAYER = {
    # Layer 0 — root
    'age': 0, 'sex': 0, 'neuroticism': 0, 'extraversion': 0,
    # Layer 1 — demographic
    'education_level': 1, 'marital_status': 1, 'bmi': 1,
    # Layer 2 — chronic
    'smoking': 2, 'alcohol_use': 2,
    'chronic_condition': 2, 'diabetes_status': 2,
    'general_health': 2,
    # Layer 3 — physical
    'pain_level': 3, 'physical_health': 3,
    'activity': 3, 'exercise': 3,
    # Layer 4 — daily
    'sleep_quality': 4, 'sleep_disturbances': 4, 'sleep_physio': 4,
    'heart_rate': 4, 'screen_usage': 4, 'communication': 4,
    'running_tasks': 4, 'social_events_positive': 4, 'social_events_negative': 4,
    # Layer 5 — psychological
    'stress_helplessness': 5, 'stress_self_efficacy': 5,
    'positive_affect': 5, 'negative_affect': 5,
    'loneliness': 5, 'mood': 5, 'productivity': 5, 'stress_ema': 5,
    # Layer 6 — high burden
    'depression': 6, 'mental_health': 6,
}

LAYER_LABELS = {
    0: 'root',
    1: 'demographic',
    2: 'chronic',
    3: 'physical',
    4: 'daily',
    5: 'psychological',
    6: 'high burden',
}


def build_graph(config: dict) -> nx.DiGraph:
    G = nx.DiGraph()
    nodes = config['nodes']
    for name, attrs in nodes.items():
        G.add_node(name, category=attrs['category'])
    for src, dst in config['forced_edges']:
        G.add_edge(src, dst)
    return G


def layered_pos(G: nx.DiGraph, layer_map: dict) -> dict:
    from collections import defaultdict
    layers = defaultdict(list)
    for node in G.nodes:
        layers[layer_map.get(node, 99)].append(node)
    pos = {}
    for layer_idx, nodes in sorted(layers.items()):
        nodes_sorted = sorted(nodes)
        n = len(nodes_sorted)
        for i, node in enumerate(nodes_sorted):
            x = (i - (n - 1) / 2) * 4.0
            y = -layer_idx * 4.5
            pos[node] = (x, y)
    return pos


def main():
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    G = build_graph(config)
    pos = layered_pos(G, LAYER)

    node_colors = [
        CATEGORY_COLORS.get(G.nodes[n].get('category', 'Cat3'), '#cccccc')
        for n in G.nodes
    ]

    fig, ax = plt.subplots(figsize=(48, 36))
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color='#aaaaaa', alpha=0.5,
        arrows=True, arrowsize=12,
        connectionstyle='arc3,rad=0.05',
        width=0.8,
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors, node_size=8000, alpha=0.95,
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=13, font_color='white', font_weight='bold',
    )

    legend_handles = [
        mpatches.Patch(color=c, label=f'{cat} (training+NHANES / NHANES only / training only)'
                       .replace('Cat1', 'Cat1').replace('Cat2', 'Cat2').replace('Cat3', 'Cat3'))
        for cat, c in CATEGORY_COLORS.items()
    ]
    legend_handles = [
        mpatches.Patch(color='#2ecc71', label='Cat1 — training + NHANES prior'),
        mpatches.Patch(color='#3498db', label='Cat2 — NHANES prior only'),
        mpatches.Patch(color='#e67e22', label='Cat3 — training only'),
    ]
    ax.legend(handles=legend_handles, loc='lower right',
              facecolor='#2c2c54', labelcolor='white', fontsize=10)

    # Layer labels on the left margin
    layer_y = {}
    for _, layer_idx in LAYER.items():
        y = -layer_idx * 2.5
        layer_y[layer_idx] = y
    x_min = min(x for x, _ in pos.values()) - 2.8
    for layer_idx, label in LAYER_LABELS.items():
        y = layer_y.get(layer_idx, -layer_idx * 2.5)
        ax.text(x_min, y, label, color='#aaaaaa', fontsize=9,
                va='center', ha='right', style='italic')

    ax.set_title('DBN Node Graph — forced edges', color='white', fontsize=14, pad=12)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved: {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
