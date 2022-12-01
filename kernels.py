import jax
import numpy as onp
import jax.numpy as np
from jax import jit, vmap
from jax.config import config
from jax.scipy.linalg import cho_factor, cho_solve
from functools import reduce
config.update("jax_enable_x64", True)

##Some kernels that can be used for graph discovery

#Auxiliary functions
def sqeuclidean_distances(x: np.ndarray, y: np.ndarray) -> float:
    return np.sqrt(np.sum( (x - y) ** 2))
dists = jit(vmap(vmap(sqeuclidean_distances, in_axes=(None, 0)), in_axes=(0, None))) #Dark magic line

def dotprod(x: np.ndarray, y: np.ndarray) -> float:
    return np.dot(x, y)
dotprods = jit(vmap(vmap(dotprod, in_axes=(None, 0)), in_axes=(0, None))) 


#The actual kernels

@jit
def matern_five_halfs(v1, v2, a=1, sigma=1):
    d = dists(v1, v2)
    return a*(1+np.sqrt(5)*d/sigma +5*d**2/(3*sigma**2))*np.exp(-np.sqrt(5)*d/sigma)


@jit
def gaussian(v1, v2, a=1, sigma=1):
    d = dists(v1, v2)
    return a*np.exp(-d**2/(2*sigma**2))


@jit
def kpoly(v1, v2, a=1,c=1, deg = 2):
    return a*(dotprods(v1, v2) + c)**deg