import networkx as nx
import numpy as np

N = 4
sigma = 0.5
mu = 1
gamma = 0.5
K = 1
threshold = 1e-20

## prepare networks, set up the backbone for the network shown in Figure 6 and optimize it
G = Utils.parametrize_tree(N,gamma)

### randomize capacities
capacities = np.random.random(len(G.edges()))
capacities /= (np.sum(capacities**gamma))**(1/gamma)/(K**gamma)
caps = {list(G.edges())[i]:capacities[i] for i in range(len(G.edges()))}
nx.set_edge_attributes(G,caps,'weight')

source_node = (0,int(N/2))
index_source_node = list(G.nodes()).index(source_node)

G = Utils.reoptimize_network(G,
                       N, 
                       gamma = gamma,
                       index_source_node = index_source_node,
                       sigma = sigma,
                       K = K,
                       threshold = 1e-20)
removed_edges = []
F = G.copy()
for i in range(len(F.edges())):
    if F[list(F.edges())[i][0]][list(F.edges())[i][1]]['weight'] < 1e-8:
        removed_edges.append(list(F.edges())[i])
F.remove_edges_from(removed_edges)

index_source_node = list(F.nodes()).index(source_node)

ns = list(F.nodes())
ns.remove(source_node)
betweenness_F = nx.algorithms.centrality.edge_betweenness_centrality_subset(F,sources=[source_node],targets = ns,normalized = False)
## NOTE: There is a factor 2 missing to match the definitions of betweenness
betweenness_list = np.array([betweenness_F[(u,v)]*2 for u,v in F.edges()])

dissipation = Utils.calc_average_dissipation(F,sigma,mu,index_source_node)
print(dissipation,np.linalg.norm(betweenness_list*sigma**2+betweenness_list**2*mu**2,ord=gamma/(gamma+1)))
