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
    def __init__(self, nodes, names, beta, type):
        self.nodes = nodes
        self.beta = beta
        self.names = names
        self.type = type

    def coeff(self):
        return np.sqrt(self.beta)

class ConstantMode(Mode):
    """
    Represent the constant kernel, K(x, y) = beta
    """
    def __init__(self, nodes=np.array([0]), names=np.array([]), beta=1, type="Constant"):
        super().__init__(nodes, names, beta, type)

    def kappa(self, x, y):
        return self.beta


    def feature(self, x):
        return np.sqrt(self.beta)

    def to_string(self):
        return ""


class LinearMode(Mode):
    """
    Represent the linear kernel, K(x, y)=beta * x[i] * y[i]
    """
    def __init__(self, nodes, names, beta, type="Linear"):
        super().__init__(nodes, names, beta, type)

    def kappa(self, x, y):
        return self.beta * x[self.nodes[0]] * y[self.nodes[0]]

    def feature(self, x):
        return np.sqrt(self.beta) * x[self.nodes[0]]

    def to_string(self):
        return self.names[0]


class QuadraticMode(Mode):
    """
    Represent the quadratic kernel, K(x, y) = beta * x[i] * x[j] * y[i] * y[j]
    """
    def __init__(self, nodes, names, beta, type="Quadratic"):
        super().__init__(nodes, names, beta, type)

    def kappa(self, x, y):
        return self.beta * x[self.nodes[0]] * x[self.nodes[1]] * y[self.nodes[0]] * y[self.nodes[1]]

    def feature(self, x):
        return np.sqrt(self.beta) * x[self.nodes[0]] * x[self.nodes[1]]

    def to_string(self):
        return self.names[0] + self.names[1]



class KernelMode(Mode):
    """
    Represent the product of kernel, K(x, y) = beta * \prod{K_i(x_i,y_i)}
    """
    def __init__(self, nodes, names, beta, ks, type="Kernel"):
        super().__init__(nodes, names, beta, type)
        self.ks = ks

    def kappa(self, x, y):
        val = [self.ks[i](x[self.nodes[i]], y[self.nodes[i]]) for i in range(len(self.ks))]
        return self.beta * reduce(np.multiply, val)

    def feature(self, x):
        return None

    def to_string(self, point):
        val = ["{},".format(self.names[i]) for i in range(len(self.names))]
        prefix = "K("
        postfix = point + ")"
        for n in val: prefix = prefix + n
        return prefix + postfix



