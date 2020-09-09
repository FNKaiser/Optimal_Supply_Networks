import networkx as nx
import numpy as np

N = 6
sigma = 0.5
mu = 1
gamma = 0.7
K = 1**(1/gamma)

G = Utils.parametrize_tree(6,gamma)
index_source_node = list(G.nodes()).index((0,int(N/2)))

G = Utils.reoptimize_network(G,
                       N, 
                       gamma = gamma,
                       index_source_node = index_source_node,
                       sigma = sigma,
                       K = K,
                       threshold = 1e-20)
potential_edges = []
F = G.copy()
for i in range(len(F.edges())):
    if F[list(F.edges())[i][0]][list(F.edges())[i][1]]['weight'] < 1e-8:
        potential_edges.append(list(F.edges())[i])
F.remove_edges_from(potential_edges)
index_source_node = list(F.nodes()).index((0,int(N/2)))

potential_drop = Utils.calc_potential_drop(F,potential_edges,gamma,index_source_node,mu,sigma)

### To calculate the critical values of gamma, the script for creating Figure4 may be used.
