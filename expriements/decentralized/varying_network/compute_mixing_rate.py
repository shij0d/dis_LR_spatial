#%%
import sys

# Add the path where your Python packages are located
sys.path.append('/home/shij0d/Documents/Dis_Spatial')

from src.networks import generate_connected_erdos_renyi_graph_with_seed
from src.weights import optimal_weight_matrix
import networkx as nx
import numpy as np


#%%



def mixing_rate(con_pro):
    J=10
    mixing_rate_list=[]
    for seed in range(100):
        er,is_connected= generate_connected_erdos_renyi_graph_with_seed(J, con_pro,seed=seed)
        adj_matrix=nx.adjacency_matrix(er).todense()
        np.fill_diagonal(adj_matrix, 1)
        weights,mixing_rate=optimal_weight_matrix(adj_matrix=adj_matrix)
        mixing_rate_list.append(mixing_rate)
    return np.mean(mixing_rate_list)

print(mixing_rate(0.3))
print(mixing_rate(0.5))
print(mixing_rate(0.8))

#%%
