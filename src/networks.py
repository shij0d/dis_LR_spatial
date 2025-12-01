import networkx as nx

def generate_connected_erdos_renyi_graph(n, p):
    # Continue generating graphs until we get a connected one
    seed=2024
    while True:
        G = nx.erdos_renyi_graph(n, p,seed=seed)
        if nx.is_connected(G):
            return G
        seed=seed+1

def generate_connected_erdos_renyi_graph_with_seed(n, p,seed):
    # Continue generating graphs until we get a connected one
    G = nx.erdos_renyi_graph(n, p,seed=seed)
    if nx.is_connected(G):
        is_connected=True
    else:
        is_connected=False
    return G,is_connected
    