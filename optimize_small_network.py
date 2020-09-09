#simple script to find the optimal capacities for the network shown in Figure 2,a.
import numpy as np
from scipy.optimize import root

def dissipation_lagrangian_derivatives_complete(caps,mu,sigma,gamma,K):
    """ Set up the derivatives resulting from the KKT conditions for the five node network,
    allowing for a loop with capacity kappa
    The resulting capacities are potentially optimal"""
    k1 = caps[0]
    k2 = caps[1]
    kappa = caps[2]
    F1_corr = 4*mu**2+2*sigma**2
    F2_corr = mu**2+sigma**2
    lagrange_multiplier = F2_corr/(gamma*k2**(gamma+1))    
    derivative_k1 = lagrange_multiplier*gamma*k1**(gamma-1)-(F1_corr/k1**2-4*sigma**2*kappa*(k1+kappa)/(k1**2*(2*kappa+k1)**2))
    derivative_kappa = kappa*(lagrange_multiplier*gamma*kappa**(gamma-1)-4*sigma**2/(k1+2*kappa)**2)
    derivative_cost = K**gamma - 2*k1**gamma-2*k2**gamma-kappa**gamma
    return [derivative_k1,derivative_kappa,derivative_cost]

def calculate_optimal_capacities_tree(mu,sigma,gamma,K):
    """ Calculate the optimal capacities for the five node tree network without allowing
    to close the loop. These capacities can be calculated explicitly using the method of Lagrange multipliers (see Eq. 9)"""
    flow_correlations = np.array([4*mu**2+2*sigma**2,mu**2+sigma**2,
                                 4*mu**2+2*sigma**2,mu**2+sigma**2])
    denominator = np.sum(flow_correlations**(gamma/(1+gamma)))**(1/gamma)
    capacities = np.zeros(len(flow_correlations))
    for i in range(len(flow_correlations)):
        capacities[i] = K*flow_correlations[i]**(1/(1+gamma))/denominator
    return capacities
    
def dissipation_function(caps,mu,sigma,gamma,K):
    """ Calculate the average dissipation for the five node network"""
    k1 = caps[0]
    k2 = caps[1]
    kappa = caps[2]
    F1_corr = 4*mu**2+2*sigma**2
    F2_corr = mu**2+sigma**2   
    return 2*F1_corr/k1+2*F2_corr/k2- kappa*4*sigma**2/(k1**2+2*kappa*k1)
    
    
# Calculate the optimal capacities using the formula above
cost = 1
mu = -1
sigma = 3
gammas = np.arange(0.8,1.00001,0.005)

nof_repetitions = 400
count = 0

k_results = []

for gamma in gammas:
    k_results.append([])

    args = (mu,sigma,gamma,cost)   
    for i in range(nof_repetitions):
        # start from different random values of the capacities to find all solution branches
        k0 = np.random.random(3)
        # find roots of derivative function in terms of the capacities
        res = root(dissipation_lagrangian_derivatives_complete,x0 = k0,args = args)
        if res.success:
            k_results[count].append(res.x)
        else:
            k_results[count].append(np.array([-1000,-1000,-1000]))
    count += 1
k_results = np.array(k_results)

# Now remove double entries where the same solution was found multiple times
temp_array = []
unique_capacities = []
for i in range(len(gammas)):
    temp_array.append(np.unique(np.array(k_results[i]).round(decimals=6),axis=0))
    # if solution was not found for this value of gamma, set the respective capacities to nan
    if np.all(temp_array[i]==np.array([-1000,-1000,-1000])):
         temp_array[i] = np.array([np.nan,np.nan,np.nan])
    unique_capacities.append(temp_array[i][np.where(np.logical_not(temp_array[i]==np.array([-1000,-1000,-1000])))])

# Collect gammas for which a solution was found and capacities with higher cost branch (saddle) and lower cost branch (optimum)
realised_gammas = []
lower_cost_capacities = []
higher_cost_capacities = []
for i in range(len(gammas)):
      if not np.all(np.isnan(unique_capacities[i])):
        cost_values = []
        for j in range(int(len(unique_capacities[i])/3)):
            dissipation = dissipation_function(unique_capacities[i][3*j:3*(j+1)],mu,sigma,gammas[i],cost)
            cost_values.append(dissipation)
        indices = np.argsort(cost_values)
        realised_gammas.append(gammas[i])
        lower_cost_capacities.append( unique_capacities[i][3*indices[0]:3*(indices[0]+1)])
        higher_cost_capacities.append(unique_capacities[i][3*indices[-1]:3*(indices[-1]+1)])
lower_cost_capacities = np.array(lower_cost_capacities)
higher_cost_capacities = np.array(higher_cost_capacities)

k_results_tree = np.zeros((len(gammas),2))
count = 0
for gamma in gammas:
    k_results_tree[count] = calculate_optimal_capacities_tree(mu,sigma,gamma,cost)[:2]
    count +=1 
    
# Finally, determine globally optimal capacities
optimal_capacities = np.zeros((len(realised_gammas),3))
for i in range(len(gammas)):
    if gammas[i] in realised_gammas:
        index = realised_gammas.index(gammas[i])
        dissipation_tree = dissipation_function([k_results_tree[i,0],k_results_tree[i,1],0.0],mu,sigma,gammas[i],cost)
        dissipation_loopy = dissipation_function([lower_cost_capacities[index,0],lower_cost_capacities[index,1],lower_cost_capacities[index,2]],mu,sigma,gammas[i],cost)
        if dissipation_tree < dissipation_loopy:
            optimal_capacities[index] = np.array([k_results_tree[i,0],k_results_tree[i,1],0.0])
        else:
            optimal_capacities[index] = np.array([lower_cost_capacities[index,0],lower_cost_capacities[index,1],lower_cost_capacities[index,2]]) 
