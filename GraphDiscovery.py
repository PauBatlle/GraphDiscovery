#File with the main functions of graph discovery
import matplotlib.pyplot as plt
import numpy as onp
from jax import jit, vmap, grad, hessian
import jax.scipy as jsp
from scipy.optimize import minimize
import networkx as nx
from kernels import *
from scipy.linalg import svd
from Modes import *
from jax.config import config
config.update("jax_enable_x64", True)

class GraphDiscovery(object):
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

    def energy_estimate(self, gamma, ga_varphi, Kbs_varphi_varphi):
        """
        Solve the variational problem with constraints and compute the energy ratio of each mode.

        Inputs:

        gamma:              positive real
                        the noise scale
        ga_varphi:          list
                        g_a represents the constraint function. ga_varphi represents the vector [varphi, g_a]
        Kbs_varphi_varphi:  (m x r x r)-tensor
                        a list of matrix, where each matrix is K_{b, i}(varphi, varphi), where K_{b, i} is the ith mode

        Outputs:

        status:             Bool
                        False implies that the data is mostly noise given the constraint. True means there is a signal
        activations:        list or None
                        If status == False, activations = None;
                        Otherwise, activations contains the energy ratio of each mode
        yb:                 list or None
                        If status == False, yb = None;
                        Otherwise, yb = -(Kb(varphi, varphi) + gamma I)^{-1} ga_varphi
        """
        Kb_varphi_varphi = reduce(np.add, Kbs_varphi_varphi)
        noise_Kb_varphi_varphi = Kb_varphi_varphi + gamma * np.eye(len(Kb_varphi_varphi))
        Ltmp = np.linalg.cholesky(noise_Kb_varphi_varphi)
        tmp = jsp.linalg.solve_triangular(Ltmp, ga_varphi, lower=True)
        yb = -jsp.linalg.solve_triangular(Ltmp.T, tmp, lower=False)

        Eb = -np.dot(ga_varphi, yb)

        activation_noise = gamma * np.dot(yb, yb) / Eb

        status = False
        activations = np.array([])
        if activation_noise < 1/2:
            status = True
            # The constraint represented by g contains a signal
            for Kbi_varphi_varphi in Kbs_varphi_varphi:
                Ebi = np.dot(yb, np.dot(Kbi_varphi_varphi, yb))
                activations = np.append(activations, Ebi / Eb)

        return status, activations, yb


    def discovery_in_graph(self, X, ks, gamma, G, names, examing_nodes, beta1=1e-1, beta2=1e-2, beta3=1e-3, epsilon=1e-3, verbose=False):
        """
        The main build_graph function, built on top of a networkx directed graph (DiGraph) G

        Inputs:

        X:          d x N matrix
            Dataset, samples of (X1_i; ...; Xd_i), i=1:N. If d variables = nodes in the graph and N timesteps
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

        # We first build a directed graph
        # preG is the complete directed graph, which encodes potential relations between nodes
        # G is an empty directed graph, represents the final output graph
        #preG = nx.complete_graph(d).to_directed()
        # G = nx.create_empty_copy(preGraph)

        # Create modes
        modes = onp.array([ConstantMode()])
        # Add linear modes
        linear_modes = [LinearMode(onp.array([i]), onp.array([names[i]]), beta1) for i in range(d)]
        modes = onp.append(modes, linear_modes)
        # Add quadratic modes
        quadratic_modes = [QuadraticMode(onp.array([i, j]), onp.array([names[i], names[j]]), beta2) for i in range(d) for j in range(i+1)]
        modes = onp.append(modes, quadratic_modes)
        # The number of modes
        modes_num = len(modes)

        # The correspondence between modes and nodes.
        # Entries at (i, j) equal to 1 implies that there is a relation between the ith mode and the jth node.
        # Otherwise, entries at (i, j) equal to 0 implies there is no relation between the ith mode and the jth node.
        # The matrix siz is (modes_num, d), which means we ignore the constant node.
        correspondence_mtx = onp.zeros((modes_num, d))
        for i in range(len(modes)):
            nodes = modes[i].nodes
            correspondence_mtx[i, nodes] = 1

        ''' For debug purpose, print out the correspondence matrix '''
        # if verbose:
        #     print("The correspondence matrix is ")
        #     print(correspondence_mtx)
        #     print("")

        Kxx = onp.zeros((N, N))
        for mode in modes:
            Kxx = Kxx + self.Kn(lambda x, y: mode.kappa(x, y), X.T)

        ''' For debug purpose, check whether Kxx is formulated correctly '''
        # Kxx = self.Kn(lambda x, y: self.kernel(x, y, beta1, beta2, beta3, ks), X.T)
        # if verbose: print("Kxx error {}".format(np.linalg.norm(Kxx - Theta)))

        # Compute sorted eigenvalues and eigenfunctions of Kxx
        eigenValues, eigenVectors = np.linalg.eigh(Kxx)
        idx = np.argsort(-eigenValues.real)
        lambdas = eigenValues[idx].real
        alphas = eigenVectors[:, idx].real

        # Find r such that r = min{n| lambda_{n} / lambda_{1} \leq \epsilon}
        radios = lambdas / lambdas[0]
        r = np.argwhere(radios <= epsilon).flatten()[0]

        lambdas = lambdas[:r]
        # Normalize the eigenvectors (alphas) of Kxx such that \|alphas[i]\|^2=1
        alphas = vmap(lambda v, lbda: v / np.linalg.norm(v))(alphas.T[:r], lambdas)
        # Now, alphas is a r x N matrix, where each row is a normalized eigenvector

        # Next, we determine ancestors of Nodes.
        Kbs_varphi_varphi = onp.zeros((len(modes), r, r))  # Record the matrix Kb(varphi, varphi) for each mode

        for t in range(len(modes)):
            mode = modes[t]
            Kbt = self.Kn(lambda x, y: mode.kappa(x, y), X.T)
            func = (lambda ar, ac: np.dot(ar, np.dot(Kbt, ac)))
            func = (vmap(func, in_axes=(None, 0)))
            func = jit(vmap(func, in_axes=(0, None)))
            Kbs_varphi_varphi[t, :, :] = func(alphas, alphas)

        for i in examing_nodes:
            if verbose: print('Examining Node {0}'.format(names[i]))
            modes_i = onp.where(correspondence_mtx[:, i] == 1)[0] # The indices of modes related to the ith node

            # The operations below put the mode beta1 * xi * xi' into Class a, all other modes related to the ith
            # node to Class c, and the rest modes to Class b
            activation_codes = onp.ones(modes_num) # Record the activation of each mode, 0 for Class a, 1 for Class b, -1 for Class c
            activation_codes[modes_i] = -1  # Put all the modes related to the ith node to Class c
            activation_codes[i+1] = 0 # Set the (i+1)th mode (beta1 * xi * xi') to be in Class a
            ##################################################################################
            # Deal with the constraint that g_a = x_i

            # build the vector [varphi, ga]
            ga = X[i]
            ga = np.reshape(ga, (1, -1))
            varphi_ga = np.dot(ga, alphas.T).flatten()

            modes_b_indices = onp.where(activation_codes == 1)[0] # Choose those modes in Class b
            Kbis_varphi_varphi = Kbs_varphi_varphi[modes_b_indices, :, :] # Record the matrix Kbi(varphi, varphi) for each activated mode Kbi in Class b


            # Solve the constraint variational problem
            # status == False implies that the signal is mostly noisy, then activations = None and yb = None
            # status == True means that there is a signal given the constraint. Then, activations contains the
            # activation (the ratio of the energy of each mode and the total energy) of each mode
            status, activations, yb = self.energy_estimate(gamma, varphi_ga, Kbis_varphi_varphi)
            activated_modes_indices = modes_b_indices # Record the indices of activated modes of Class b. The indices index the array of modes
            if not status: activated_modes_indices = np.array([])
            # if status == True, we start to iterate and prune out one mode each time until (before) the noise dominates
            while status and (len(activations) > 1):
                sorted_activation_indices = np.argsort(activations).tolist()
                pre_Kbis_varphi_varphi = Kbis_varphi_varphi[sorted_activation_indices[1:]]
                pre_indices = activated_modes_indices[sorted_activation_indices[1:]]

                status, activations, pre_yb = self.energy_estimate(gamma, varphi_ga, pre_Kbis_varphi_varphi)

                if status:
                    Kbis_varphi_varphi = pre_Kbis_varphi_varphi
                    activated_modes_indices = pre_indices
                    yb = pre_yb

            disactivated_modes_indices = np.setdiff1d(modes_b_indices, activated_modes_indices).tolist()
            activation_codes[disactivated_modes_indices] = -1 #Put all the disactivated modes to Class c.


            """ Add all the edges associated with the ith node  """
            modes_b_indices = onp.where(activation_codes == 1)[0]
            modes_b = modes[modes_b_indices]
            for mode in modes_b:
                nodes = mode.nodes
                for node in nodes:
                    if node != i and (not G.has_edge(names[node], names[i])):
                        G.add_edge(names[node], names[i])

            """ Print out the equations """
            """
            The equation found is g = g_a + g_b, where g_a = x_i
            g_b = K_b(., varphi)y_b. We need to compute K_b(., varphi). 
            When, K_b(x, y) = psi(x)^T\psi(y). We have (K_b(.,varphi))_j = psi(x)^T(Psi(X)alpha_j),
            where alpha_j is the jth normalized eigenvector of Kxx, Psi(X)=[psi(X_1);...;psi(X_N)]. Hence, 
            K_b(., varphi) = psi(x)^T Psi(X) alphas. The jth column of Psi(X) contains the value of
            the feature map evaluated at X_j.
            """

            # Get the weights of the feature maps.
            # For instance, if psi(x) = beta * phi(x), we return beta
            coeffs = onp.array([mode.coeff() for mode in modes_b])

            M1 = onp.zeros((len(modes_b), N))
            for t in range(len(modes_b)):
                mode = modes_b[t]
                M1[t, :] = vmap(mode.psi)(X.T)

            M2 = alphas.T
            M = np.dot(M1, M2)

            weights_i = np.dot(M, yb)
            weights_i = weights_i * coeffs
            # print the equation representing x_i as the function of other variables
            if verbose: print("Node {} as a function of other nodes".format(names[i]))
            eq = "{}".format(names[i])
            count = 0
            for mode in modes_b:
                eq = eq + ' + '
                eq = eq + "({} {})".format(weights_i[count], mode.to_string())
                count = count + 1
            eq = eq + ' = 0'
            if verbose: print(eq)
            if verbose: print("")

        return G

    def plot_graph(self, G):
            nx.draw(G, with_labels=True, pos=nx.kamada_kawai_layout(G, G.nodes()), node_size=600, font_size=8, alpha=0.6)

