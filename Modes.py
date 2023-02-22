import matplotlib.pyplot as plt
import numpy as np
from jax import jit, vmap, grad, hessian
import jax.scipy as jsp
from scipy.optimize import minimize
import networkx as nx
from kernels import *
from scipy.linalg import svd
from jax.config import config
config.update("jax_enable_x64", True)

class Mode(object):
    def __init__(self, nodes, names, beta):
        self.nodes = nodes
        self.beta = beta
        self.names = names

    def coeff(self):
        return np.sqrt(self.beta)

class ConstantMode(Mode):
    """
    Represent the constant kernel, K(x, y) = beta
    """
    def __init__(self, nodes=np.array([0]), names=np.array([]), beta=1):
        super().__init__(nodes, names, beta)

    def kappa(self, x, y):
        return self.beta


    def psi(self, x):
        return np.sqrt(self.beta)

    def to_string(self):
        return ""


class LinearMode(Mode):
    """
    Represent the linear kernel, K(x, y)=beta * x[i] * y[i]
    """
    def __init__(self, nodes, names, beta):
        super().__init__(nodes, names, beta)

    def kappa(self, x, y):
        return self.beta * x[self.nodes[0]] * y[self.nodes[0]]

    def psi(self, x):
        return np.sqrt(self.beta) * x[self.nodes[0]]

    def to_string(self):
        return self.names[0]


class QuadraticMode(Mode):
    """
    Represent the quadratic kernel, K(x, y) = beta * x[i] * x[j] * y[i] * y[j]
    """
    def __init__(self, nodes, names, beta):
        super().__init__(nodes, names, beta)

    def kappa(self, x, y):
        return self.beta * x[self.nodes[0]] * x[self.nodes[1]] * y[self.nodes[0]] * y[self.nodes[1]]

    def psi(self, x):
        return np.sqrt(self.beta) * x[self.nodes[0]] * x[self.nodes[1]]

    def to_string(self):
        return self.names[0] + self.names[1]



