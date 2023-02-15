import matplotlib.pyplot as plt
import numpy as np

from GraphDiscovery import *

from jax.config import config
config.update("jax_enable_x64", True)

def matern_five_halfs(v1, v2, a=1, sigma=1):
    d = np.sqrt(np.sum((v1 - v2) ** 2))
    return a*(1+np.sqrt(5)*d/sigma +5*d**2/(3*sigma**2))*np.exp(-np.sqrt(5)*d/sigma)



ks = [matern_five_halfs]*3
kpca = GraphDiscovery()

d = 3
preG = nx.complete_graph(d).to_directed()
G = nx.create_empty_copy(preG)
names = onp.array(['x1', 'x2', 'x3'])
G = nx.relabel_nodes(G, dict(zip(range(d), names)))

examing_nodes = range(3)

x1s = np.linspace(-2, 2,100)
x3s = 0.5 * x1s ** 2 #+ onp.random.normal(size = 100)*1e-1
x2s = (-1 - x1s - 3 * x3s)/2 #+ onp.random.normal(size = 100)*1e-1

fig = plt.figure
X = np.array([x1s, x2s, x3s])
gamma = 1e-13
G = kpca.discovery_in_graph(X, ks, gamma, G, names, examing_nodes, beta1=1, beta2=1e-1, beta3=0, epsilon=1e-3, verbose=True)
kpca.plot_graph(G)
plt.show()

no_rels_nodes = [i for i in range(len(G.nodes)) if G.in_degree(names[i]) == 0]
no_rels_nodes_names = names[no_rels_nodes]

if len(no_rels_nodes) >= 1:
    def expand_aux(xi, X):
        return xi * X

    def expand_name_aux(xi_name, X_names):
        return onp.array([xi_name + x for x in X_names])

    new_X = X
    names_expanded = names.copy()
    examined_nodes = []
    for node_ind in no_rels_nodes:
        nodes_to_examine = onp.setdiff1d(range(d), examined_nodes)

        new_data = vmap(expand_aux)(X[node_ind], X[nodes_to_examine].T)
        new_X = np.concatenate((new_X, new_data.T), axis=0)

        new_names = expand_name_aux(names[node_ind], names[nodes_to_examine])
        names_expanded = onp.append(names_expanded, new_names)

        G.add_nodes_from(new_names)
        examined_nodes.append(node_ind)

    X = new_X
    names = names_expanded
    examing_nodes = np.concatenate((np.array(no_rels_nodes), np.arange(d, len(names))))

    G = kpca.discovery_in_graph(X, ks, gamma, G, names, examing_nodes, beta1=1, beta2=1e-1, beta3=0, epsilon=1e-3, verbose=True)
    kpca.plot_graph(G)
    plt.show()

# ks = [matern_five_halfs]*4
# kpca = GraphDiscovery()
# x1s = np.linspace(-2, 2,100)
# x3s = -4/19 - 17/19 * x1s
# x2s = (x1s + x3s)/2
# x4s = (-x2s - x3s)/4
#
# fig = plt.figure
# X = np.array([x1s, x2s, x3s, x4s])
# kpca.build_graph(X, ks, gamma=1e-13, beta1=1, beta2=0, beta3=0, epsilon=1e-3, names=['x1', 'x2', 'x3', 'x4'], verbose=True, plot=True)
# plt.show()


# ks = [matern_five_halfs]*3
# kpca = GraphDiscovery()
#
# x1s = np.linspace(-2, 2,100)
# x3s = x1s/2 #+ onp.random.normal(size = 100)*1e-1
# x2s = (-1 - 2 * x1s - 4 * x3s)/3 #+ onp.random.normal(size = 100)*1e-1
#
# fig = plt.figure
# X = np.array([x1s, x2s, x3s])
# kpca.build_graph(X, ks, gamma=1e-10, beta1=1, beta2=1e-2, beta3=0, epsilon=1e-3, names=['x1', 'x2', 'x3'], verbose=True, plot=True)
# plt.show()


# ks = [matern_five_halfs]*3
# kpca = GraphDiscovery()
#
#
# x1s = np.linspace(-2, 2,100)
# x3s = x1s #+ onp.random.normal(size = 100)*1e-1
# x2s = (-1 - x1s - 3 * x3s)/2 #+ onp.random.normal(size = 100)*1e-1
#
# fig = plt.figure
# X = np.array([x1s, x2s, x3s])
# kpca.build_graph(X, ks, gamma=1e-13, beta1=1, beta2=0, beta3=0, epsilon=1e-3, names=['x1', 'x2', 'x3'], verbose=True, plot=True)
# plt.show()


# ks = [matern_five_halfs]*4
# kpca = GraphDiscovery()
#
#
# x1s = np.linspace(-2, 2,100)
# x3s = -4/19 - 17/19 * x1s
# x2s = (x1s + x3s)/2
# x4s = (-x2s - x3s)/4
#
# # plt.plot(x1s, x1s)
# # plt.plot(x1s, x2s)
# # plt.plot(x1s, x3s)
# # plt.show()
#
# fig = plt.figure
# X = np.array([x1s, x2s, x3s, x4s])
# kpca.build_graph(X, ks, gamma=1e-13, beta1=1, beta2=0, beta3=0, epsilon=1e-3, names=['x1', 'x2', 'x3', 'x4'], verbose=True, plot=True)
# plt.show()




