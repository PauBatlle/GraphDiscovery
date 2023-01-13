#File with the main functions of graph discovery
import jax
import numpy as onp
import jax.numpy as np
from jax import jit, vmap, grad, hessian
from jax.config import config
from jax.scipy.linalg import cho_factor, cho_solve
import jax.scipy as jsp
from functools import reduce
from scipy.optimize import minimize
import networkx as nx
from itertools import count
import matplotlib.pyplot as plt
from functools import partial
from jax import jit
import jaxopt as jopt
from kernels import *
from scipy.linalg import svd
config.update("jax_enable_x64", True)

def solve_svd(A, b):
    # compute svd of A
    U,s,Vh = svd(A)

    # U diag(s) Vh x = b <=> diag(s) Vh x = U.T b = c
    c = np.dot(U.T,b)
    # diag(s) Vh x = c <=> Vh x = diag(1/s) c = w (trivial inversion of a diagonal matrix)
    w = np.dot(np.diag(1/s),c)
    # Vh x = w <=> x = Vh.H w (where .H stands for hermitian = conjugate transpose)
    x = np.dot(Vh.conj().T,w)
    return x

class KPCA(object):
    def __init__(self):
        pass

    def polyK(self, x, y, gamma):
        x_mtx = np.reshape(x, (-1, 1))
        y_mtx = np.reshape(y, (-1, 1))
        return 1 + np.dot(x, y) + gamma * np.sum((x_mtx @ x_mtx.T) * (y_mtx @ y_mtx.T))

    def Kn(self, kernel, funcs):
        return self.Kmn(kernel, funcs, funcs)

    def Kmn(self, kernel, funcs_l, funcs_r):
        N_l, m_l = funcs_l.shape
        N_r, m_r = funcs_r.shape

        assert m_l == m_r

        BBv = np.reshape(np.tile(funcs_l, (1, N_r)), (N_l * N_r, m_l))
        BBh = np.reshape(np.tile(funcs_r, (N_l, 1)), (N_l * N_r, m_r))

        val = vmap(lambda x, y: kernel(x, y))(BBv, BBh)
        K = np.reshape(val, (N_l, N_r))
        return K

    #@partial(jit, static_argnums=(0,))
    def loss(self, params, gamma, gamma2, gamma3, XT, alphas, Lambdas, L):
        """
        loss function when we use the kernel \kappa(x, x')=1+x^Tx'+\gamma_2\sum_{ij}^dx_ix_jx'_ix'_j+\gamma_3\prod_{j}^d(1+\kappa_j(x_j, x_j'))

        Input:
        params:     list
                the coefficients of the function
        gamma:      real
                penalization parameter for the constraints
        gamma2:     real
                weights for the kernel related to the second-order interactions
        gamma3:     real
                weights for the kernel related to the nonlinear interactions represented by Gaussian regressions
        XT:         matrix
                Dataset. XT is a N x d matrix. Each row represents the data at a given sample slot for all the nodes
        alphas:     matrix
                Eigenvectors. alphas is a r x N matrix. Each row is an eigenvector
        Lambdas:    list
                Eigenvalues. Lambdas contains r values.
        L:      matrix
                The Cholesky decomposition of the kernel matrix associated with the kernel Gamma3 after eliminating contraints.

        Return:
        real
                the loss associated with a given choice of implicit relations between nodes
        """
        d, N = self.d, self.N

        z = params[:N]
        beta0 = params[N]
        beta = params[N+1:N+d+1]
        B = params[N+d+1:]

        Bmtx = np.reshape(B, (d, d))

        f0x = beta0 * np.ones(N)
        f1x = vmap(lambda x: np.dot(beta, x))(XT)
        f2x = vmap(lambda x: np.dot(x, np.dot(Bmtx, x)))(XT)
        f3x = z

        lz = jsp.linalg.solve_triangular(L, z, lower=True)
        #lz = np.linalg.solve(Theta, z)
        loss1 = gamma * beta0 ** 2 + gamma * np.dot(beta, beta) + gamma / gamma2 * np.dot(B, B)  + gamma / gamma3 * np.dot(lz, lz)
        #cts = np.diag(np.sqrt(Lambdas)) @ (alphas @ (f0x + f1x + f2x + f3x))
        cts = np.diag(Lambdas ** (3/2))  @ (alphas @ (f0x + f1x + f2x + f3x))
        loss2 = np.dot(cts, cts) #/ gamma
        return (loss1 + loss2) / 2

    def constrained_loss(self, params, unpack_func, gamma, gamma2, gamma3, XT, alphas, Lambdas, L):
        unpacked_params = unpack_func(params)
        return self.loss(unpacked_params, gamma, gamma2, gamma3, XT, alphas, Lambdas, L)

    def build_graph(self, X, ks, gamma, gamma2=1, gamma3=1, epsilon=1e-3, tau1=0.5, tau2=0.5, noise_scale = 1e-1, nugget=1e-10, names=None, verbose=False, plot=False):
        """ The main build_graph function, built on top of a networkx directed graph (DiGraph) G

        Inputs:

        X: Dataset, samples of (X1_i, ..., Xd_i), i=1:t. If d variables = nodes in the graph and t timesteps, X is (d x N) numpy array
        ks: Array of kernels, one for each node. Each kernel is a jax vectorized function, so that K(X,X) already builds a matrix with K(X,X)_ij = K(X_i, X_j). See examples
        possible_edges (Default = None): A list of directed edges that could exist (used to discard some prior edges). If None, all edges are possible
        tau1 (Default = 0.5): Threshold to detect ancestors vs noise. Ancestors iff  E_signal > γ1*(E_signal+E_noise)
        tau2 (Default = 0.5): Threshold to detect particular ancestors. Edge to j -> i is removed iff E_without_j > γ2*(E_without_j+E_with_j):
        names (Default = None): The names of the nodes in the graph, for printing and visualization purposes
        verbose (Default = False): Whether to print intermediate results
        plot (Default = False): Whether to plot the graph at the end

        Outputs:

        G: Networkx directed graph
        (Not yet implemented to return the weights of each arrow)
        """

        d, N = X.shape
        self.d, self.N = d, N

        Gamma12 = self.Kn(lambda x, y: self.polyK(x, y, gamma2), X.T)

        Ms = {}
        for j in range(d):
            Ms[j] = 1 + ks[j](X[j], X[j])

        Gamma3 = reduce(np.multiply, list(Ms.values()))

        Kxx = Gamma12 + gamma3 * Gamma3
        # compute sorted eigenvalues and eigenfunctions of Kxx
        eigenValues, eigenVectors = np.linalg.eig(Kxx)
        idx = np.argsort(-eigenValues.real)
        lambdas = eigenValues[idx].real
        alphas = eigenVectors[:, idx].real

        # find r such that r = min{n| lambda_{n} / lambda_{1} \leq \epsilon}
        radios = lambdas / lambdas[0]
        r = np.argwhere(radios <= epsilon).flatten()[0]

        # Normalize the eigenvalues, alphas, of Kxx such that \|alphas[i]\|^2=1/(N * lambda_i)
        alphas = vmap(lambda v, lbda: v / (np.sqrt(N * lbda) * np.linalg.norm(v)))(alphas.T[:r], lambdas[:r])
        # Now, alphas is a r x N matrix, where each row is a normalized eigenvector

        # Next, we determine ancestors of Nodes.
        # We first build a complete directed graph
        # Build a complete directed graph
        if names is None:
            names = list(range(d))

        # G is the complete directed graph
        G = nx.complete_graph(d).to_directed()

        for i in G.nodes:
            if verbose: print('Examining Node {0}'.format(names[i]))
            possible_ancestors = [i[0] for i in G.in_edges([i])]
            Ys = X[i]
            Msi = {}
            for j in possible_ancestors:
                Msi[j] = 1 + ks[j](X[j], X[j]) # ks has to be indexed for possible_ancestors:

            Kixx = reduce(np.multiply, list(Msi.values()))
            nuggeted_matrix = Kixx + noise_scale * np.eye(len(Kixx))
            v = np.linalg.solve(nuggeted_matrix, Ys)  # v = K(X,X)^-1 Y
            E_1 = v.T @ Kixx @ v  # E_i = Y^T K(X,X)^-1 K_i(X,X) K(X,X)^-1 Y = v^T K_i(X,X) v
            E_2 = v.T @ np.eye(len(Kixx)) * noise_scale @ v
            if verbose: print('\t Node {0}, E_Signal = {1:.4f}, E_Noise = {2:.4f}'.format(names[i], E_1, E_2))
            if E_1 < tau1 * (E_2 + E_1):
                if verbose: print('\t Node {0} does not have any ancestors'.format(names[i]))
                for j in possible_ancestors:
                    G.remove_edge(j, i)
            if E_1 > tau1 * (E_2 + E_1):
                if verbose: print('\t Node {0} have ancestors'.format(names[i]))
                # Add constraints into the minimization problem such that
                # we can express the ith node as the function of other nodes

                other_nodes = np.setdiff1d(np.array(range(d)), np.array([i]))
                other_Ms = [Ms[k] for k in other_nodes.tolist()]
                Gamma3mi = reduce(np.multiply, other_Ms) #Use multiplication instead of division below seems to be more stable
                #Gamma3mi = np.divide(Gamma3, Ms[i]) #The gram matrix associated with the kernel without the ith kernel

                #nugget_Gamma3mi = Gamma3mi + nugget * np.eye(len(Gamma3mi))
                L_Gamma3mi = np.linalg.cholesky(Gamma3mi + nugget * np.eye(len(Gamma3mi)))
                def unpack_params_i(packed_params):
                    #hard encode the constraints
                    z = packed_params[:N]
                    beta0 = packed_params[N]
                    betami = packed_params[N + 1:N + d]
                    Bm1 = np.reshape(packed_params[N + d:], (d-1, d-1))

                    beta = np.ones(d)
                    other_nodes = np.setdiff1d(np.array(range(d)), np.array([i]))
                    beta = beta.at[other_nodes].set(betami)

                    XX, YY = np.meshgrid(other_nodes, other_nodes, indexing='ij')
                    B = np.zeros((d, d))
                    B = B.at[XX, YY].set(Bm1)

                    return np.concatenate((z, np.array([beta0]), beta, B.flatten()))

                def _unpack_params_i(packed_params):
                    #for debug purpose
                    z = packed_params[:N]
                    beta0 = packed_params[N]
                    betami = packed_params[N + 1:N + d]
                    Bm1 = np.reshape(packed_params[N + d:], (d-1, d-1))

                    beta = np.ones(d)
                    other_nodes = np.setdiff1d(np.array(range(d)), np.array([i]))
                    beta = beta.at[other_nodes].set(betami)

                    XX, YY = np.meshgrid(other_nodes, other_nodes, indexing='ij')
                    B = np.zeros((d, d))
                    B = B.at[XX, YY].set(Bm1)

                    return beta0, beta, B

                params_size = N + 1 + (d - 1) + d ** 2 - d - (d - 1)
                params0 = np.zeros(params_size)
                H_i = hessian(self.constrained_loss)(params0, unpack_params_i, gamma, gamma2, gamma3, X.T, alphas, lambdas[:r], L_Gamma3mi)
                b_i = grad(self.constrained_loss)(params0, unpack_params_i, gamma, gamma2, gamma3, X.T, alphas, lambdas[:r], L_Gamma3mi)
                #LH = jsp.linalg.cholesky(H + 1e-8 * np.eye(len(H)))
                #_params = jsp.linalg.solve_triangular(LH.T, jsp.linalg.solve_triangular(LH, -b, lower=True), lower=False)
                params_i = solve_svd(H_i, -b_i)
                #Since H is ill-conditioned, params is not accurate. We need a robust linear solver
                #Even solve_svd is not accurate enough
                loss_i = self.constrained_loss(params_i, unpack_params_i, gamma, gamma2, gamma3, X.T, alphas, lambdas[:r], L_Gamma3mi)

                if verbose: print("Energy of Node {} is {}".format(names[i], loss_i))
                # Then, we start to identify which nodes can be viewed as an ancestor of the ith node
                possible_ancestors = [j[0] for j in G.in_edges([i])]
                for j in possible_ancestors:
                    # We add extra constraints to remove the dependence of the ith node on the jth node
                    other_nodes = np.setdiff1d(np.array(range(d)), np.array([i, j]))
                    other_Ms = [Ms[k] for k in other_nodes.tolist()]
                    Gamma3mimj = reduce(np.multiply, other_Ms) #Use multiplication instead of division below seems to be more stable
                    #Gamma3mimj = np.divide(Gamma3mi, Ms[j])  # The gram matrix associated with the kernel without the ith kernel
                    L_Gamma3mimj = np.linalg.cholesky(Gamma3mimj + nugget * np.eye(len(Gamma3mimj)))
                    #nugget_Gamma3mimi = Gamma3mimj + nugget * np.eye(len(Gamma3mimj))
                    def unpack_params_ij(packed_params):
                        #hard encode the constraints
                        z = packed_params[:N]
                        beta0 = packed_params[N]
                        betami = packed_params[N+1:N+d-1]
                        Bm1 = np.reshape(packed_params[N+d-1:], (d-2, d-2))

                        beta = np.ones(d)
                        other_nodes = np.setdiff1d(np.array(range(d)), np.array([i, j]))
                        beta = beta.at[other_nodes].set(betami)
                        beta = beta.at[j].set(0)

                        XX, YY = np.meshgrid(other_nodes, other_nodes, indexing='ij')
                        B = np.zeros((d, d))
                        B = B.at[XX, YY].set(Bm1)

                        return np.concatenate((z, np.array([beta0]), beta, B.flatten()))

                    def _unpack_params_ij(packed_params):
                        #for debug purpose
                        z = packed_params[:N]
                        beta0 = packed_params[N]
                        betami = packed_params[N + 1:N + d - 1]
                        Bm1 = np.reshape(packed_params[N + d - 1:], (d - 2, d - 2))

                        beta = np.ones(d)
                        other_nodes = np.setdiff1d(np.array(range(d)), np.array([i, j]))
                        beta = beta.at[other_nodes].set(betami)
                        beta = beta.at[j].set(0)

                        XX, YY = np.meshgrid(other_nodes, other_nodes, indexing='ij')
                        B = np.zeros((d, d))
                        B = B.at[XX, YY].set(Bm1)

                        return beta0, beta, B

                    params_size = N + 1 + (d - 2) + d ** 2 - d - (d - 1) - (d - 1) - (d - 2)
                    params0 = np.zeros(params_size)
                    H_ij = hessian(self.constrained_loss)(params0, unpack_params_ij, gamma, gamma2, gamma3, X.T, alphas, lambdas[:r], L_Gamma3mimj)
                    b_ij = grad(self.constrained_loss)(params0, unpack_params_ij, gamma, gamma2, gamma3, X.T, alphas, lambdas[:r], L_Gamma3mimj)
                    #params = np.linalg.solve(H, -b)
                    params_ij = solve_svd(H_ij, -b_ij)
                    loss_ij = self.constrained_loss(params_ij, unpack_params_ij, gamma, gamma2, gamma3, X.T, alphas, lambdas[:r], L_Gamma3mimj)

                    if verbose: print("Energy of Node {} after eliminating Node {} is {}".format(names[i], names[j], loss_ij))
                    if verbose: print("The ratio is {}".format((loss_ij - loss_i)/loss_i))

                    assert(loss_ij >= loss_i) # loss_ij should be larger than loss_i

                    if (loss_ij - loss_i) < tau2 * loss_i:
                        #Increase in the loss is small. The jth node is not necessary an ancestor of the ith node
                        G.remove_edge(j, i)

            if verbose: print(" ")

        if plot:
            G = nx.relabel_nodes(G, dict(zip(range(d), names)))
            nx.draw(G, with_labels=True, pos=nx.kamada_kawai_layout(G, G.nodes()), node_size=600, font_size=8, alpha=0.6)
        return G


    #The following function may should be deleted in the future
    #This function based on scipy built-in optimization function to optimize the objective function
    #The optimization procedure is not succesful.
    def build_graph_by_optimization(self, X, ks, gamma, gamma2=1, gamma3=1, epsilon=0.01, tau1=0.5, tau2=0.5, nugget=1e-10, names=None, verbose=False, plot=False):
        """ The main build_graph function, built on top of a networkx directed graph (DiGraph) G

        Inputs:

        X: Dataset, samples of (X1_i, ..., Xd_i), i=1:t. If d variables = nodes in the graph and t timesteps, X is (d x N) numpy array
        ks: Array of kernels, one for each node. Each kernel is a jax vectorized function, so that K(X,X) already builds a matrix with K(X,X)_ij = K(X_i, X_j). See examples
        possible_edges (Default = None): A list of directed edges that could exist (used to discard some prior edges). If None, all edges are possible
        tau1 (Default = 0.5): Threshold to detect ancestors vs noise. Ancestors iff  E_signal > γ1*(E_signal+E_noise)
        tau2 (Default = 0.5): Threshold to detect particular ancestors. Edge to j -> i is removed iff E_without_j > γ2*(E_without_j+E_with_j):
        names (Default = None): The names of the nodes in the graph, for printing and visualization purposes
        verbose (Default = False): Whether to print intermediate results
        plot (Default = False): Whether to plot the graph at the end

        Outputs:

        G: Networkx directed graph
        (Not yet implemented to return the weights of each arrow)
        """

        d, N = X.shape
        self.d, self.N = d, N

        Gamma12 = self.Kn(lambda x, y: self.polyK(x, y, gamma2), X.T)

        Ms = {}
        for j in range(d):
            Ms[j] = 1 + ks[j](X[j], X[j])

        Gamma3 = reduce(np.multiply, list(Ms.values()))

        Kxx = Gamma12 + gamma3 * Gamma3
        # compute sorted eigenvalues and eigenfunctions of Kxx
        eigenValues, eigenVectors = np.linalg.eig(Kxx)
        idx = np.argsort(-eigenValues.real)
        lambdas = eigenValues[idx].real
        alphas = eigenVectors[:, idx].real

        # find r such that r = min{n| lambda_{n} / lambda_{1} \leq \epsilon}
        radios = lambdas / lambdas[0]
        r = np.argwhere(radios <= epsilon).flatten()[0]

        # Normalize the eigenvalues, alphas, of Kxx such that \|alphas[i]\|^2=1/(N * lambda_i)
        alphas = vmap(lambda v, lbda: v / (np.sqrt(N * lbda) * np.linalg.norm(v)))(alphas.T[:r], lambdas[:r])
        # Now, alphas is a r x N matrix, where each row is a normalized eigenvector

        # Solving eq (1) in the script without any constraint
        # Compute a loss which is associated a function f in the RKHS associated with k
        # f represents an implicit function determine the relationship between all the nodes
        L = np.linalg.cholesky(Gamma3)

        params_size = d+1+d**2+N
        params0 = np.ones(params_size)
        # ls = jopt.GaussNewton(self.loss, maxiter=int(1e10), verbose=True)
        # res = ls.run(params0, gamma, gamma2, gamma3, X.T, alphas, lambdas[:r], L)
        res = minimize(self.loss, params0, args=(gamma, gamma2, gamma3, X.T, alphas, lambdas[:r], L))
        if not res.success:
            print('Failed to minimize the global loss! Numerical Errors may occur!')
            print(res.message)

        params = res
        global_loss = res.fun

        # Next, we determine ancestors of Nodes.
        # We first build a complete directed graph
        # Build a complete directed graph
        if names is None:
            names = list(range(d))

        # G is the complete directed graph
        G = nx.complete_graph(d).to_directed()

        for i in G.nodes:
            # Add constraints into the minimization problem such that
            # we can express the ith node as the function of other nodes

            c1 = np.zeros(params_size)
            c1 = c1.at[i + 1].set(1.0)
            c1_cts = onp.array([{'type': 'eq', 'fun': lambda x: np.dot(c1, x) - 1}])

            def build_quadratic_cts(i, index):
                c2 = np.zeros((d, d))
                c2 = c2.at[i, index].set(1.0)
                c2 = onp.concatenate((np.zeros(1 + d), c2.flatten(), np.zeros(N)))

                c2t = np.zeros((d, d))
                c2t = c2t.at[index, i].set(1.0)
                c2t = onp.concatenate((np.zeros(1 + d), c2t.flatten(), np.zeros(N)))

                return onp.array([{'type': 'eq', 'fun': lambda x: np.dot(c2, x)}, {'type': 'eq', 'fun': lambda x: np.dot(c2t, x)}])

            c2_cts = onp.array([])
            for index in range(d):
                c2_cts = onp.append(c2_cts, build_quadratic_cts(i, index))

            cts = onp.concatenate((c1_cts, c2_cts.flatten()))

            other_nodes = np.setdiff1d(np.array(range(d)), np.array([i]))
            other_Ms = [Ms[k] for k in other_nodes.tolist()]
            Gamma3mi = reduce(np.multiply, other_Ms)
            #Gamma3mi = np.divide(Gamma3, Ms[i]) #The gram matrix associated with the kernel without the ith kernel

            L_Gamma3mi = np.linalg.cholesky(Gamma3mi + nugget * np.eye(len(Gamma3mi)))

            params0 = params0
            res = minimize(self.loss, params0, args=(d, N, gamma, gamma2, gamma3, X.T, alphas, lambdas[:r], L_Gamma3mi), constraints=cts.tolist(), method='trust-constr')
            if not res.success:
                print('Failed to minimize the loss! Numerical Errors may occur!')
            params = res
            loss_i = res.fun

            assert (loss_i >= global_loss) #loss_i should be larger than global_loss

            if (loss_i - global_loss) <= tau1 * global_loss:
                # Node i can be viewed as a function of other nodes
                # Then, we start to identify which nodes can be viewed as an ancestor of the ith node
                possible_ancestors = [j[0] for j in G.in_edges([i])]
                for j in possible_ancestors:
                    # We add extra constraints to remove the dependence of the ith node on the jth node

                    extra_c1 = np.zeros(params_size)
                    extra_c1 = extra_c1.at[j + 1].set(1.0)
                    extra_c1_cts = onp.array([{'type': 'eq', 'fun': lambda x: np.dot(extra_c1, x)}])

                    extra_c2_cts = onp.array([])
                    for index in range(d):
                        extra_c2_cts = onp.append(extra_c2_cts, build_quadratic_cts(j, index))

                    cts = onp.concatenate((c1_cts, c2_cts, extra_c1_cts, extra_c2_cts))

                    Gamma3mimj = np.divide(Gamma3mi, Ms[j])  # The gram matrix associated with the kernel without the ith kernel

                    L_Gamma3mimj = np.linalg.cholesky(Gamma3mimj + nugget * np.eye(len(Gamma3mimj)))

                    params0 = params.x
                    res = minimize(self.loss, params0, args=(d, N, gamma, gamma2, gamma3, X.T, alphas, lambdas[:r], L_Gamma3mimj), constraints=cts.tolist(), method='trust-constr')
                    if not res.success:
                        print('Failed to minimize the loss! Numerical Errors may occur!')

                    loss_ij = res.fun

                    assert(loss_ij >= loss_i) # loss_ij should be larger than loss_i

                    if (loss_ij - loss_i) < tau2 * loss_i:
                        #Increase in the loss is small. The jth node is not necessary an ancestor of the ith node
                        G.remove_edge(j, i)
            else:
                # Node i cannot be viewed as a function of other nodes
                # Delete the relations in the graph
                possible_ancestors = [j[0] for j in G.in_edges([i])]
                for j in possible_ancestors:
                    G.remove_edge(j, i)

        if plot:
            G = nx.relabel_nodes(G, dict(zip(range(d), names)))
            nx.draw(G, with_labels=True, pos=nx.kamada_kawai_layout(G, G.nodes()), node_size=600, font_size=8, alpha=0.6)
        return G