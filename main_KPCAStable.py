import matplotlib.pyplot as plt
from KPCAStable import *

from jax.config import config
config.update("jax_enable_x64", True)

def matern_five_halfs(v1, v2, a=1, sigma=1):
    d = np.sqrt(np.sum((v1 - v2) ** 2))
    return a*(1+np.sqrt(5)*d/sigma +5*d**2/(3*sigma**2))*np.exp(-np.sqrt(5)*d/sigma)


ks = [matern_five_halfs]*3
kpca = KPCAStable()


x1s = np.linspace(-2, 2,100)
x3s = x1s #+ onp.random.normal(size = 100)*1e-1
x2s = (-1 - x1s - 3 * x3s)/2 #+ onp.random.normal(size = 100)*1e-1

# plt.plot(x1s, x1s)
# plt.plot(x1s, x2s)
# plt.plot(x1s, x3s)
# plt.show()

fig = plt.figure
X = np.array([x1s, x2s, x3s])
kpca.build_graph(X, ks, gamma=1e-13, beta1=1, beta2=0, beta3=0, epsilon=1e-3, tau1=0.5, tau2=0.5, noise_scale=1e-1, nugget=1e-12, names=['x1', 'x2', 'x3'], verbose=True, plot=True)
plt.show()
