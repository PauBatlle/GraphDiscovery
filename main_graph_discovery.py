import matplotlib.pyplot as plt
import numpy as np

from GraphDiscovery import *

from jax.config import config
config.update("jax_enable_x64", True)

def matern_five_halfs(v1, v2, a=1, sigma=1):
    d = np.sqrt(np.sum((v1 - v2) ** 2))
    return a*(1+np.sqrt(5)*d/sigma +5*d**2/(3*sigma**2))*np.exp(-np.sqrt(5)*d/sigma)

ks = onp.array([matern_five_halfs]*3)
kpca = GraphDiscovery()

d = 3
preG = nx.complete_graph(d).to_directed()
G = nx.create_empty_copy(preG)
names = onp.array(['x1', 'x2', 'x3'])
G = nx.relabel_nodes(G, dict(zip(range(d), names)))

examing_nodes = np.array([0, 1, 2])

x3s = np.linspace(-2, 2,100)
x1s = x3s ** 2 #+ onp.random.normal(size = 100)*1e-1
x2s = (-1 - x1s - 3 * x3s)/2 #+ onp.random.normal(size = 100)*1e-1

# x1s = np.linspace(-2, 2,100)
# x3s = 0.5 * x1s ** 2 #+ onp.random.normal(size = 100)*1e-1
# x2s = (-1 - x1s - 3 * x3s)/2 #+ onp.random.normal(size = 100)*1e-1

fig = plt.figure
X = np.array([x1s, x2s, x3s])
gamma = 1e-13
G = kpca.discovery_in_graph(X, ks, gamma, G, names, examing_nodes, beta1=1, beta2=1e-2, beta3=1e-10, epsilon=1e-3, verbose=True)
kpca.plot_graph(G)
plt.show()