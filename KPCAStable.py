#File with the main functions of graph discovery
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

class KPCAStable(object):
    def __init__(self):
        pass

    def kernel(self, x, y, beta1, beta2, beta3, ks):
        gaussian_terms = [1 + k(x, y) for k in ks]
        gaussian_term = reduce(np.multiply, gaussian_terms)

        v = x * y
        vv = np.reshape(v, (-1, 1))
        mtx = np.dot(vv, vv.T)
        quadratic_term = np.sum(np.tril(mtx))
        return 1 + beta1 * np.dot(x, y) + beta2 * quadratic_term + beta3 * gaussian_term

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

    def build_graph(self, X, ks, gamma, beta1=1e-1, beta2=1e-2, beta3=1e-3, epsilon=1e-3, tau1=0.5, tau2=0.5, noise_scale = 1e-1, nugget = 1e-10, names=None, verbose=False, plot=False):
        """ The main build_graph function, built on top of a networkx directed graph (DiGraph) G

        Inputs:

        X:          d x N matrix
            Dataset, samples of (X1_i, ..., Xd_i), i=1:t. If d variables = nodes in the graph and t timesteps
        ks:         list
            Array of kernels, one for each node;
        gamma:      positive real
            The regularization constant for the constraints
        beta1:      positive real
            The penalization on the linear kernel
        beta2:      positive real
            The penalization on the quadratic kernel
        beta3:      positive real
            The penalization on the fully nonlinear kernel
        epsilon:    positive real
            The threshold to determine the number of eigenvalues selected
        tau1:       positive real
            Threshold to detect ancestors vs noise. Ancestors iff  E_signal > tau1*(E_signal+E_noise)
        tau2:       positive real
            Threshold to detect particular ancestors. Edge to j -> i is removed iff E_without_j > tau2*(E_without_j+E_with_j)
        noise_scale:    positive real
            The penalization on the noise
        nugget:     positive real
            A nugget added to the covariance matrix for numerical stability
        names (Default = None): list
            The names of the nodes in the graph, for printing and visualization purposes
        verbose (Default = False): Bool
            Whether to print intermediate results
        plot (Default = False): Bool
            Whether to plot the graph at the end

        Outputs:

        G: Networkx directed graph
        (Not yet implemented to return the weights of each arrow)
        """
        d, N = X.shape
        self.d, self.N = d, N

        Kxx = self.Kn(lambda x, y: self.kernel(x, y, beta1, beta2, beta3, ks), X.T)

        # compute sorted eigenvalues and eigenfunctions of Kxx
        eigenValues, eigenVectors = np.linalg.eigh(Kxx)
        idx = np.argsort(-eigenValues.real)
        lambdas = eigenValues[idx].real
        alphas = eigenVectors[:, idx].real

        # find r such that r = min{n| lambda_{n} / lambda_{1} \leq \epsilon}
        radios = lambdas / lambdas[0]
        r = np.argwhere(radios <= epsilon).flatten()[0]

        lambdas = lambdas[:r]
        # Normalize the eigenvalues, alphas, of Kxx such that \|alphas[i]\|^2=1
        alphas = vmap(lambda v, lbda: v / np.linalg.norm(v))(alphas.T[:r], lambdas)
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

            other_nodes = np.setdiff1d(np.array(range(d)), np.array([i]))
            Xmi = X[other_nodes]
            Ksmi = [ks[i] for i in other_nodes]

            Kb = self.Kn(lambda x, y: self.kernel(x, y, beta1, beta2, beta3, Ksmi), Xmi.T) #contains the kernel without the data of the ith node

            nuggeted_matrix = Kb + noise_scale * np.eye(len(Kb))
            v = np.linalg.solve(nuggeted_matrix, Ys)  # v = K(X,X)^-1 Y
            E_1 = v.T @ Kb @ v  # E_i = Y^T K(X,X)^-1 K_i(X,X) K(X,X)^-1 Y = v^T K_i(X,X) v
            E_2 = v.T @ np.eye(len(Kb)) * noise_scale @ v
            """ Debug purpose, plot the fitted solution """
            '''-----------------------------------------------------------'''
            # ri = Kb @ v
            # rni = np.eye(len(Kb)) * noise_scale @ v
            # plt.plot(X[i], 'r')
            # plt.plot(ri, 'g')
            # plt.plot(rni, 'b')
            # plt.show()
            '''-----------------------------------------------------------'''
            if verbose: print('\t Node {0}, E_Signal = {1:.4f}, E_Noise = {2:.4f}'.format(names[i], E_1, E_2))
            if E_1 < tau1 * (E_2 + E_1):
                if verbose: print('\t Node {0} does not have any ancestors'.format(names[i]))
                for j in possible_ancestors:
                    G.remove_edge(j, i)
            if E_1 > tau1 * (E_2 + E_1):
                if verbose: print('\t Node {0} have ancestors'.format(names[i]))
                # Add constraints into the minimization problem such that
                # we can express the ith node as the function of other nodes

                # Add constraint such that f_i = x_i
                Ka = Kxx - Kb
                Li = np.linalg.cholesky(Ka + nugget * np.eye(len(Ka)))
                tmp = jsp.linalg.solve_triangular(Li, X[i], lower=True)
                Ea = np.dot(tmp, tmp)

                # build the vector [varphi, f_i]
                fi = X[i]
                fi = np.reshape(fi, (1, -1))
                tmp = np.dot(fi, alphas.T).flatten()
                varphi_fi = lambdas ** (-1/2) * tmp

                # build the matrix Kb(varphi, varphi)
                func = (lambda lbdai, ai, lbdaj, aj: (lbdai ** (-1/2)) * (lbdaj ** (-1/2)) * np.dot(ai, np.dot(Kb, aj)))
                func = (vmap(func, in_axes=(None, None, 0, 0)))
                func = jit(vmap(func, in_axes=(0, 0, None, None)))
                Kb_varphi_varphi = func(lambdas, alphas, lambdas, alphas)

                nuggeted_Kb_varphi_varphi = Kb_varphi_varphi + gamma * np.eye(len(Kb_varphi_varphi))
                Ltmp = np.linalg.cholesky(nuggeted_Kb_varphi_varphi)
                tmp = jsp.linalg.solve_triangular(Ltmp, varphi_fi, lower=True)
                Eb = np.dot(tmp, tmp)

                """ Debug purpose, print out the linear equation """
                '''-----------------------------------------------------------'''
                # get the weights of the feature maps it works when K(x, y) = \phi(x)^T\phi(y)
                z = jsp.linalg.solve_triangular(Ltmp.T, tmp, lower=False)

                M1 = Xmi * np.sqrt(beta1)
                M1 = np.concatenate((np.ones((1, N)), M1), axis=0)

                M2 = vmap(lambda ai, lbdai: ai * lbdai ** (-1/2))(alphas, lambdas)
                M2 = M2.T

                M = np.dot(M1, M2)

                weights_i = -np.dot(M, z)
                #print the equation representing x_i as the function of other variables
                #this print only support linear equations by now
                print(" ")
                print("Node {} as a function of other nodes".format(names[i]))
                eq = "{}".format(names[i])
                eq = eq + ' + ({})'.format(weights_i[0])
                count = 1
                for ind in other_nodes:
                    eq = eq + ' + '
                    eq = eq + "({} {})".format(weights_i[count], names[ind])
                    count = count + 1
                eq = eq + ' = 0'
                print(eq)
                '''-----------------------------------------------------------'''

                loss_i = Ea + Eb

                if verbose: print("Energy of Node {} is {}".format(names[i], loss_i))
                # Then, we start to identify which nodes can be viewed as an ancestor of the ith node
                possible_ancestors = [j[0] for j in G.in_edges([i])]
                for j in possible_ancestors:
                    # We add extra constraints to remove the dependence of the ith node on the jth node
                    other_nodes = np.setdiff1d(np.array(range(d)), np.array([i, j]))
                    Xmij = X[other_nodes]
                    Ksmij = [ks[i] for i in other_nodes]

                    Kbmj = self.Kn(lambda x, y: self.kernel(x, y, beta1, beta2, beta3, Ksmij), Xmij.T)  # contains the kernel without the data of the ith node

                    # build the matrix Kbmj(varphi, varphi)
                    func = (lambda lbdai, ai, lbdaj, aj: (lbdai ** (-1 / 2)) * (lbdaj ** (-1 / 2)) * np.dot(ai, np.dot(Kbmj, aj)))
                    func = (vmap(func, in_axes=(None, None, 0, 0)))
                    func = jit(vmap(func, in_axes=(0, 0, None, None)))
                    Kbmj_varphi_varphi = func(lambdas, alphas, lambdas, alphas)

                    nuggeted_Kbmj_varphi_varphi = Kbmj_varphi_varphi + gamma * np.eye(len(Kbmj_varphi_varphi))
                    Ltmp = np.linalg.cholesky(nuggeted_Kbmj_varphi_varphi)
                    tmp = jsp.linalg.solve_triangular(Ltmp, varphi_fi, lower=True)
                    Ebmj = np.dot(tmp, tmp)

                    loss_ij = Ea + Ebmj

                    """ Debug purpose, print out the linear equation """
                    '''-----------------------------------------------------------'''
                    # get the weights of the feature maps it works when K(x, y) = \phi(x)^T\phi(y)
                    z = jsp.linalg.solve_triangular(Ltmp.T, tmp, lower=False)
                    M1 = Xmij * np.sqrt(beta1)
                    M1 = np.concatenate((np.ones((1, N)), M1), axis=0)

                    M2 = vmap(lambda ai, lbdai: ai * lbdai ** (-1 / 2))(alphas, lambdas)
                    M2 = M2.T

                    M = np.dot(M1, M2)

                    weights_ij = -np.dot(M, z)
                    # print the equation representing x_i as the function of other variables except the jth node
                    # this print only support linear equations by now
                    print(" ")
                    print("Node {} as a function of other nodes except Node {}".format(names[i], names[j]))
                    eq = "{}".format(names[i])
                    eq = eq + ' + ({})'.format(weights_ij[0])
                    count = 1
                    for ind in other_nodes:
                        eq = eq + ' + '
                        eq = eq + "({} {})".format(weights_ij[count], names[ind])
                        count = count + 1

                    eq = eq + ' = 0'
                    print(eq)
                    '''-----------------------------------------------------------'''


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