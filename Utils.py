import networkx as nx
import numpy as np

def parametrize_tree(N,gamma):
    """ Parametrize a triangular graph of size N with normalization by cost parameter gamma, 
    such that all edges that do not belong to the symmetric tree shown in Fig.2c 
    start with zero capacity and all other capacities are normalized such that
    \sum_{e}k_e^\gamma = 1"""
    
    G = nx.generators.triangular_lattice_graph(N,2*N)
    for i in range(0,int(N/2)-1):
        G.remove_nodes_from([(i,N/2-2*i-1),(i,N/2-2*i-2)])
        G.remove_nodes_from([(i,N/2+2*i+1),(i,N/2+2*i+2)])
    for j in range(0,int(N/2)-2):
        G.remove_nodes_from([(N-j,N/2-2*j-2),(N-j,N/2-2*j-3)])
        G.remove_nodes_from([(N-j,N/2+2*j+2),(N-j,N/2+2*j+3)])
    main_comp = np.max([len(comp) for comp in list(nx.connected_components(G))])
    for comp in list(nx.connected_components(G)):
        if len(comp)!=main_comp:
            G.remove_nodes_from(comp)
            
    es = [((i,int(N/2)),(i+1,int(N/2))) for i in range(N)]
    
    for i in range(len(es)):
        es2_path = [(i+int((j+1)/2),int(N/2)+j) for j in range(int(N/2)+1)]
        es2 = list(zip(es2_path,es2_path[1:]))
        es3_path = [(i+int((j+1)/2),int(N/2)-j) for j in range(int(N/2)+1)]
        es3 = list(zip(es3_path,es3_path[1:]))
        es += es2
        es += es3
        
    caps = {}
    for u,v in G.edges():
        if (u,v) in es or (u,v)[::-1] in es:
            caps[(u,v)] = 1.0/len(es)**(1/gamma)
        else:
            caps[(u,v)] = 0.0
    nx.set_edge_attributes(G,caps,'weight')
    return G


def reoptimize_network(G,N,gamma,index_source_node,sigma,K=1,threshold = 1e-6):
    """Take the given graph G which is assumed to be a minimum dissipation network for a set of parameters,
    now using different parameters to again optimize the underlying network for the new parameters.
    The method used to find capacities that locally minimize the network dissipation was adapted from
    
    F.Corson, "Fluctuations and Redundancy in Optimal Transport Networks", Physical Review Letters (4), 048703, 2010
    """
    sigma = sigma
    K = K
    nof_nodes = len(list(G.nodes()))
    
    correlation_matrix_sources = np.ones((nof_nodes,nof_nodes))
    np.fill_diagonal(correlation_matrix_sources,1+sigma**2)
    correlation_matrix_sources[index_source_node,:] = - (nof_nodes-1)-sigma**2
    correlation_matrix_sources[:,index_source_node] = - (nof_nodes-1)-sigma**2
    correlation_matrix_sources[index_source_node,index_source_node] = (nof_nodes-1)**2+(nof_nodes-1)*sigma**2
    capacities = np.array([G[list(G.edges())[i][0]][list(G.edges())[i][1]]['weight'] for i in range(len(G.edges()))])
    capacities /= (np.sum(capacities**gamma)/K**gamma)

    last_change = 1e5
    iterations = 0
    while last_change>threshold:
        #if iterations % 10==0:
        #    print('Iterations ' + str(iterations))
        #    print('Last change ' +str(last_change))
        line_capacities = np.array([G[u][v]['weight'] for u,v in G.edges()])
        L = nx.laplacian_matrix(G).A
        B = np.linalg.pinv(L)
        I = nx.incidence_matrix(G,oriented=True).A
        flow_correlations = np.linalg.multi_dot([np.diag(line_capacities),I.T,B,correlation_matrix_sources,B,I,np.diag(line_capacities)])
        new_line_capacities = np.zeros(len(G.edges()))
        for i in range(len(flow_correlations)):
            new_line_capacities[i] = flow_correlations[i,i]**(1/(1+gamma))/(np.sum(np.diag(flow_correlations)**(gamma/(1+gamma))))**(1/gamma)*K
        new_cap_dict = {list(G.edges())[i]:new_line_capacities[i] for i in range(len(G.edges()))}
        nx.set_edge_attributes(G,new_cap_dict,'weight')
        last_change = np.sum((new_line_capacities-line_capacities)**2)
        iterations += 1
    return G

def calc_potential_drop(F,potential_edges,gamma,index_source_node,mu,sigma):
    """ Calculate the average squared pressure drop (see Figure 5) for all edges indicated as
    'potential_edges' and the graph F
    """
    N = len(F.nodes())
    #### following is correlation matrix for sources assuming Gaussian sources with unit mean, a single sink and variance sigma
    correlation_matrix_sources = np.ones((N,N))
    np.fill_diagonal(correlation_matrix_sources,mu**2+sigma**2)
    correlation_matrix_sources[index_source_node,:] = - (N-1)*mu**2-sigma**2
    correlation_matrix_sources[:,index_source_node] = - (N-1)*mu**2-sigma**2
    correlation_matrix_sources[index_source_node,index_source_node] = (N-1)**2*mu**2+(N-1)*sigma**2
    #### calculate correlation matrix of angles
    L = nx.laplacian_matrix(F).A
    R = np.linalg.pinv(L)
    ###calculate correlation matrices
    correlation_matrix_thetas = np.linalg.multi_dot([R,correlation_matrix_sources,R.T])
    ### removed edges are potential new edges
    node1 = 0
    node2 = 0
    potential_drops = {}
    for u,v in potential_edges:
        index_u = list(F.nodes()).index(u)
        index_v = list(F.nodes()).index(v)
        new_potential_drop = correlation_matrix_thetas[index_u,index_u]+correlation_matrix_thetas[index_v,index_v] -2*correlation_matrix_thetas[index_u,index_v]
        potential_drops[(u,v)] = new_potential_drop
    return potential_drops

def calc_average_dissipation(G,sigma,mu,index_source_node):
    """Calculate average dissipation for the network encoded by G"""
    
    line_capacities = np.array([G[u][v]['weight'] for u,v in G.edges()])
    
    N = len(G.nodes())
    
    correlation_matrix_sources = np.ones((N,N))
    np.fill_diagonal(correlation_matrix_sources,mu**2+sigma**2)
    correlation_matrix_sources[index_source_node,:] = - (N-1)*mu**2-sigma**2
    correlation_matrix_sources[:,index_source_node] = - (N-1)*mu**2-sigma**2
    correlation_matrix_sources[index_source_node,index_source_node] = (N-1)**2*mu**2+(N-1)*sigma**2
    
    L = nx.laplacian_matrix(G).A
    B = np.linalg.pinv(L)
    I = nx.incidence_matrix(G,oriented=True).A
    flow_correlations = np.linalg.multi_dot([np.diag(line_capacities),I.T,B,correlation_matrix_sources,B,I,np.diag(line_capacities)])
    dissipation = np.sum(np.diag(flow_correlations)/line_capacities)
    return dissipation
