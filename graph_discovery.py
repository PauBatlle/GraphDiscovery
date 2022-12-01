#File with the main functions of graph discovery
import jax
import numpy as onp
import jax.numpy as np
from jax import jit, vmap
from jax.config import config
from jax.scipy.linalg import cho_factor, cho_solve
from functools import reduce
import networkx as nx
config.update("jax_enable_x64", True)


def build_graph(X, ks, nugget, possible_edges = None, γ1 = 0.5, γ2 = 0.5, names = None, verbose = False, plot = False):
    """ The main build_graph function, built on top of a networkx directed graph (DiGraph) G
    
    Inputs: 

    X: Dataset, samples of (X1_i, ..., Xn_i), i=1:t. If n variables = nodes in the graph and t timesteps, X is (n x t) numpy array
    ks: Array of kernels, one for each node. Each kernel is a jax vectorized function, so that K(X,X) already builds a matrix with K(X,X)_ij = K(X_i, X_j). See examples
    nugget: the noise σ² of the system, that will be used to detect ancestors
    possible_edges (Default = None): A list of directed edges that could exist (used to discard some prior edges). If None, all edges are possible
    γ1 (Default = 0.5): Threshold to detect ancestors vs noise. Ancestors iff  E_signal > γ1*(E_signal+E_noise)
    γ2 (Default = 0.5): Threshold to detect particular ancestors. Edge to j -> i is removed iff E_without_j > γ2*(E_without_j+E_with_j):
    names (Default = None): The names of the nodes in the graph, for printing and visualization purposes
    verbose (Default = False): Whether to print intermediate results
    plot (Default = False): Whether to plot at the end

    Outputs:

    G: Networkx directed graph
    (Not yet implemented to return the weights of each arrow)
    """
    n = X.shape[0]

    if names is None:
        names = list(range(n))
    
    if possible_edges is None:
        #G is the complete directed graph
        G = nx.complete_graph(n).to_directed()
    else:
        G = nx.DiGraph(possible_edges)


    nodes_with_ancestors = [i for i in G.nodes if G.in_degree[i] > 0]
    for i in nodes_with_ancestors:
        possible_ancestors = [i[0] for i in G.in_edges([i])]
        Ys = X[i]
        Ms = {}
        for j in possible_ancestors:
            Ms[j] = 1 + ks[j](X[j], X[j]) #ks has to be indexed for possible_ancestors: 
        
        Kxx = reduce(np.multiply,list(Ms.values()))
        nuggeted_matrix = Kxx+nugget*np.eye(len(Kxx)) #Kxx
        v = np.linalg.solve(nuggeted_matrix, Ys) #v = K(X,X)^-1 Y
        E_1 = v.T @ Kxx @ v #E_i = Y^T K(X,X)^-1 K_i(X,X) K(X,X)^-1 Y = v^T K_i(X,X) v 
        E_2 = v.T @ np.eye(len(Kxx))*nugget @ v
        if verbose: print('Node {0}, E_Signal = {1}, E_Noise = {2}'.format(names[i], E_1, E_2))
        if E_1 < γ1*(E_2+E_1):
            if verbose: print('Node {0} does not have any ancestors'.format(names[i]))
            for j in possible_ancestors:
                G.remove_edge(j, i)
        if E_1 > γ1*(E_2+E_1):
            if verbose: print('Node {0} has ancestors'.format(names[i]))
            
            for j in possible_ancestors:
                Maux = np.divide(Kxx, Ms[j])
                E1j = v.T @ Maux @ v
                E2j = v.T @ np.multiply(Maux,(Ms[j]-1))@v
                #Actually E2j is just E_1 - E_1j, but this is here for debugging/understanding purposes
                if verbose: print('Decomposing E_signal = {0} in without {1} = {2} and with {1} = {3}'.format(E_1, names[j], E1j, E2j))
                if verbose: print('Noise/Signal Energy Ratio: {0}'.format(E1j/(E2j+E1j)))
                if E1j > γ2*(E2j+E1j):
                    G.remove_edge(j, i)
                    
            if verbose: print('Node {0} has the ancestors {1}'.format(names[i], [names[k[0]] for k in G.in_edges([i])])) 
            #Final_ancestors[i] = ancestors
    
    if plot:
        G = nx.relabel_nodes(G, dict(zip(range(n),names)))
        nx.draw(G, with_labels = True, pos = nx.kamada_kawai_layout(G, G.nodes()), node_size =  600, font_size = 8, alpha = 0.6)
    return G