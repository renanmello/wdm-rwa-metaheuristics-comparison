import networkx as nx
import matplotlib.pyplot as plt

# ==================== DEFINIÇÃO DAS ARESTAS ====================
nsfnet_edges = [
    (0, 1), (0, 2), (0, 3), (1, 2), (1, 7), (2, 5), (3, 4), (3, 10),
    (4, 6), (4, 5), (5, 8), (5, 12), (6, 7), (7, 9), (8, 9), (9, 11),
    (9, 13), (10, 11), (10, 13), (11, 12)
]

redclara_edges = [
    (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4),
    (3, 5), (4, 5), (4, 6), (5, 6), (5, 7), (6, 7), (6, 8), (7, 8),
    (7, 9), (8, 9), (8, 10), (9, 10), (9, 11), (10, 11), (10, 12),
    (11, 12), (11, 13), (12, 13), (12, 14), (13, 14), (13, 15),
    (14, 15), (14, 16), (15, 16), (15, 17), (16, 17), (16, 18),
    (17, 18), (17, 19), (18, 19), (18, 20), (19, 20), (19, 21),
    (20, 21), (20, 22), (21, 22), (21, 23), (22, 23)
]

janet6_edges = [
    (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4),
    (3, 4), (3, 5), (4, 5), (4, 6), (5, 6), (5, 7), (6, 7), (6, 8),
    (7, 8), (7, 9), (8, 9), (8, 10), (9, 10), (9, 11), (10, 11),
    (10, 12), (11, 12), (11, 13), (12, 13), (12, 14), (13, 14),
    (13, 15), (14, 15), (14, 16), (15, 16), (15, 17), (16, 17),
    (16, 18), (17, 18), (17, 19), (18, 19), (18, 20), (19, 20),
    (19, 21), (20, 21), (20, 22), (21, 22), (21, 23), (22, 23)
]

ipe_edges = [
    (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4),
    (3, 4), (3, 5), (4, 5), (4, 6), (5, 6), (5, 7), (6, 7), (6, 8),
    (7, 8), (7, 9), (8, 9), (8, 10), (9, 10), (9, 11), (10, 11),
    (10, 12), (11, 12), (11, 13), (12, 13), (12, 14), (13, 14),
    (13, 15), (14, 15), (14, 16), (15, 16), (15, 17), (16, 17),
    (16, 18), (17, 18), (17, 19), (18, 19), (18, 20), (19, 20),
    (19, 21), (20, 21), (20, 22), (21, 22), (21, 23), (22, 23),
    (23, 24), (24, 25), (25, 26), (26, 27), (27, 28), (28, 29),
    (29, 30), (30, 31), (31, 32), (32, 33), (33, 34), (34, 35),
    (35, 36), (36, 37), (37, 38), (38, 39), (39, 40), (40, 41),
    (41, 42), (42, 43), (43, 44), (44, 45), (45, 46), (46, 47)
]

# ==================== FUNÇÃO PARA PLOTAR REDE ====================
def plot_network(edges, title, layout='spring', seed=42):
    G = nx.Graph()
    G.add_edges_from(edges)
    
    # Escolha do layout
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=seed, k=0.5)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'spectral':
        pos = nx.spectral_layout(G)
    else:
        pos = nx.circular_layout(G)
    
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue', edgecolors='black')
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    plt.title(title, fontsize=16, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# ==================== PLOTAGEM DAS 4 REDES ====================
plot_network(nsfnet_edges, "NSFNET (14 nós)", layout='spring', seed=42)
plot_network(redclara_edges, "RedCLARA (24 nós)", layout='kamada_kawai')
plot_network(janet6_edges, "JANET6 (24 nós)", layout='kamada_kawai')
plot_network(ipe_edges, "IPE (48 nós)", layout='spectral')