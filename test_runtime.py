import numpy as np
import pandas as pd
from newGraphDiscovery import GraphDiscoveryNew
from Modes import ModeContainer
import numpy as onp
import matplotlib.pyplot as plt
import networkx as nx
import pickle
from glob import glob

# load data from file
data = np.load('SachsData.npy')

# sub-select data
Ntrain = 2000
l = data[np.random.randint(low=0,high=data.shape[0],size=(Ntrain)),:]
# normalize data
X=(l-np.mean(l,axis=0))/np.std(l,axis=0)

node_names=[
    '$Raf$',
    '$Mek$',
    '$Plcg$',
    '$PIP2$',
    '$PIP3$',
    '$Erk$',
    '$Akt$',
    '$PKA$',
    '$PKC$',
    '$P38$',
    '$Jnk$'
]
modes=ModeContainer.make_container(
    X.T,
    onp.array(node_names),
    {'name':'linear','beta':0.1,'type':'individual','interpolatory':False,'default':True},
    {'name':'quadratic','beta':0.1,'type':'pairwise','interpolatory':False,'default':True},
)

graph_discovery = GraphDiscoveryNew(X.T,onp.array(node_names),modes)

for node in [graph_discovery.names[0]]:
    print(f'inspecting {node}')
    graph_discovery.find_ancestors(node,gamma='auto')
    print('\n')
