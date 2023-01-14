import jax.numpy as np
import numpy as onp
from jax import hessian, vmap, grad, jacrev
import scipy
from jax.config import config
config.update("jax_enable_x64", True)

# x = np.array([-2.        ,  4.89132319,  0.02981385])
# B = np.zeros((3, 3))
# lbds = np.array([1.43588479e+06, 5.27642883e+03, 5.00972663e+02])
#
# f = lambda params: np.dot( (lbds ** (3/2)) * (x.dot(np.reshape(params, (3, 3)).dot(x))), (lbds ** (3/2)) * (x.dot(np.reshape(params, (3, 3)).dot(x))))
# H = hessian(f)(B.flatten())

x = np.array([-2.        ,  4.89132319])
B = np.zeros((2, 2))
lbds = np.array([1.43588479e+06, 5.27642883e+03])

#   Given x, define a function f(B) = |x^TBx|^2 * \sum_{i=1}^2\lambda^2_i.
#   We use jax to compute the Hessian of f w.r.t. B.
#   In the implementation, we view B as a vector.
#   We try different implementation of f and test the accuracy of jax.hessian

def f0(params):
    s = (x[0] ** 2 * params[0] + x[0] * x[1] * (params[1] + params[2]) + x[1] ** 2 * params[3]) ** 2
    return np.dot(lbds ** (3 / 2), lbds ** (3 / 2)) * s
def f1(params):
    r =  (lbds ** (3/2)) * (x.dot(np.reshape(params, (2, 2)).dot(x)))
    return np.dot(r, r)

def f2(params):
    B = np.reshape(params, (2, 2))
    return np.dot(lbds ** (3 / 2), lbds ** (3 / 2)) * np.dot(x, np.dot(B, x)) ** 2

def mtx_mul_vec(mtx, vec):
    l = len(vec)
    vec_ext = np.kron(np.eye(l), np.reshape(x, (-1, 1)))
    return np.dot(mtx, vec_ext)

def vec_mtx_vec(mtx, vec):
    l = len(vec)
    vec_ext = np.kron(np.eye(l), np.reshape(x, (-1, 1)))
    return np.dot(np.dot(vec, vec_ext.T), mtx)

def f3(params):
    return np.dot(lbds ** (3/2), lbds ** (3/2)) * (np.dot(x, mtx_mul_vec(params, x))) ** 2

def f4(params):
    return np.dot(lbds ** (3/2), lbds ** (3/2)) * vec_mtx_vec(params, x) ** 2

def Hessian():
    rho = np.dot(lbds ** (3/2), lbds ** (3/2))
    l = len(x)
    vec = np.reshape(x, (-1, 1))
    vec_ext = np.kron(np.eye(l), np.reshape(x, (-1, 1)))
    return 2 * rho * vec_ext @ vec @ vec.T @ vec_ext.T


Bvec = np.array(onp.random.randn(4))

print("Examine the accuracy of the functions")
print(f0(Bvec) - f1(Bvec))
print(f1(Bvec) - f2(Bvec))
print(f3(Bvec) - f4(Bvec))
print(f4(Bvec) - f0(Bvec))

H  = Hessian()
H0 = hessian(f0)(Bvec)
H1 = hessian(f1)(Bvec)
H2 = hessian(f2)(Bvec)
H3 = hessian(f3)(Bvec)
H4 = hessian(f4)(Bvec)

print("")
print("Examine the symmetry of the Hessian")
print(np.linalg.norm(H0 - H0.T))
print(np.linalg.norm(H1 - H1.T))
print(np.linalg.norm(H2 - H2.T))
print(np.linalg.norm(H3 - H3.T))
print(np.linalg.norm(H4 - H4.T))

print("")
print("Examine the accuracy of the Hessian")
print(np.linalg.norm(H0 - H))
print(np.linalg.norm(H1 - H))
print(np.linalg.norm(H2 - H))
print(np.linalg.norm(H3 - H))
print(np.linalg.norm(H4 - H))

print("")
print("Check the eigenvalues of H")
print(scipy.linalg.eigh(H))
print(scipy.linalg.eigh(H4))