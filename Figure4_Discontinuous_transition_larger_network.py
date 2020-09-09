import Utils
import networkx as nx
import numpy as np

N = 6
gamma = 0.7

# Create backbone tree network
G = Utils.parametrize_tree(N,gamma)

# remove all non tree edges 
# these will be called 'potential edges' since they may potentially add loops to the network
potential_edges = []
F = G.copy()
for i in range(len(F.edges())):
    if F[list(F.edges())[i][0]][list(F.edges())[i][1]]['weight'] < 1e-8:
        potential_edges.append(list(F.edges())[i])
F.remove_edges_from(potential_edges)


gamma_old = gamma
sigma = 0.5
mu = 1
K = 1

# index of the node that serves as source
index_source_node = list(G.nodes()).index((0,int(N/2)))

dissipation_ratios = []
optimized_capacities = []
gammas = np.arange(0.7,0.99,0.001)

count = 0
for gamma in gammas:
    print(count)
    #### adjust the weights of the optimal tree for the new gamma constraint
    tree_for_new_gamma = F.copy()
    tree_for_new_gamma = reoptimize_network(tree_for_new_gamma,
                                      N, 
                                      gamma = gamma,
                                      index_source_node = index_source_node,
                                      sigma = sigma,
                                      K = K,
                                      threshold = 1e-20)
    #### now add the possibility to have loops to the network by adding a small capacity to all potential edges
    F2 = F.copy()
    ### get minimum capacity that was realized
    capacities = np.array([F2[u][v]['weight'] for u,v in F2.edges()])
    minimum_capacity = np.min(capacities[np.logical_not(np.isclose(capacities,0))])
    
    ### perturb all non-existent edges by 1 % of the minimum capacity
    perturbation_strength = minimum_capacity*0.01
    
    ###################
    for e in potential_edges:
        F2.add_weighted_edges_from([(e[0],e[1],perturbation_strength)])
    weights = nx.get_edge_attributes(F2,'weight')
    
    #### renormalize
    normalization_factor = np.sum([weights[(u,v)]**gamma for u,v in F2.edges()])
    
    ### NOTE: this was written assuming K = 1 and needs to be adjusted otherwise
    normalization_factor = normalization_factor**(1/gamma)
    #renormalize
    for key in weights.copy().keys():
        weights[key] /= normalization_factor
    nx.set_edge_attributes(F2,weights,'weight')
    capacities = np.array([F2[u][v]['weight'] for u,v in F2.edges()])
    
    F2 = reoptimize_network(F2,
                                  N, 
                                  gamma = gamma,
                                  index_source_node = index_source_node,
                                  sigma = sigma,
                                  K = K,
                                  threshold = 1e-20)
    # store the optimzed capacities for this value of gamma 
    optimized_capacities.append([])
    for e in potential_edges:
        optimized_capacities[count].append(F2[e[0]][e[1]]['weight'])
        
    count += 1
# optimized capacities in dependence of cost parameter gamma
optimized_capacities = np.array(optimized_capacities)
