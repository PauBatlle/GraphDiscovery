import matplotlib.pyplot as plt
from KPCA import *
from kernels import matern_five_halfs as k

from jax.config import config
config.update("jax_enable_x64", True)


# x1s = np.linspace(-2, 2, 500)
# x2s = x1s**2 + 1 + onp.random.random(size = len(x1s))*1e-1

# ks = [k]*2
# kpca = KPCA()
#
# X = np.array([x1s, x2s])
# kpca.build_graph(X, ks, gamma=1e-8, gamma2=1, gamma3=1, epsilon=1e-8, tau=0.5, nugget=1e-10, names=['x1', 'x2'], verbose=True, plot=True)
# plt.show()

ks = [k]*3
kpca = KPCA()

#What we expect to recover is X1-> X2 -> X3
# x1s = np.linspace(-2, 2, 100)
# x2s = x1s**2 + 1 + onp.random.random(size = len(x1s))*1e-1
# x3s = x2s**2 - 1 + onp.random.random(size = len(x1s))*1e-1
# X = np.array([x1s, x2s, x3s])
#
# kpca.build_graph(X, ks, gamma=1e-8, gamma2=1, gamma3=1, epsilon=1e-8, tau1=0.5, tau2=0.5, noise_scale = 0.1, nugget=1e-10, names=['x1', 'x2', 'x3'], verbose=True, plot=True)
# plt.show()

# x0s = np.linspace(-2, 2, 100)
# onp.random.seed(0)
# x1s = onp.random.random(100)
# x2s = (x0s + x1s)**2 + onp.random.normal(size = len(x1s))*1e-1
# X = np.array([x0s, x1s, x2s])
#
# kpca.build_graph(X, ks, gamma=1e-8, gamma2=1, gamma3=1, epsilon=1e-8, tau1=0.5, tau2=0.5, names=['x1', 'x2', 'x3'], noise_scale=1e-1, verbose=True, plot=True)


x1s = np.linspace(-2,2,100)
x2s = x1s**2 + 1 + onp.random.normal(size = 100)*1e-1
x3s = (x1s + 2)**3 + onp.random.normal(size = 100)*1e-1
X = np.array([x1s, x2s, x3s])
kpca.build_graph(X, ks, gamma=1e-10, gamma2=1e-1, gamma3=1e-4, epsilon=1e-3, tau1=0.5, tau2=1e-2, names=['x1', 'x2', 'x3'], noise_scale=1e-1, verbose=True, plot=True)
plt.show()
