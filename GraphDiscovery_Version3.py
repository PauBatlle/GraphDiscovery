# File with the main functions of graph discovery
import matplotlib.pyplot as plt
import numpy as np
import numpy as onp
from jax import jit, vmap, grad, hessian
import jax.scipy as jsp
from scipy.optimize import minimize
import networkx as nx
from kernels import *
from scipy.linalg import svd
from Modes import *
from itertools import chain, combinations
from jax.config import config

config.update("jax_enable_x64", True)


# if the signal-noise ratio is greater than 1, there is a signal, otherwise it is a noise.

def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in np.arange(1, len(s) + 1))


class GraphDiscovery(object):
    """
    In this class, we consider linear, quadratic, and product kernels
    """

    def __init__(self):
        pass

    def kernel(self, x, y, beta1, beta2, beta3, ks):
        gaussian_terms = [1 + k(x, y) for k in ks]
        gaussian_term = reduce(np.multiply, gaussian_terms)

        v = x * y
        vv = np.reshape(v, (-1, 1))
        mtx = np.dot(vv, vv.T)
        quadratic_term = np.sum(np.tril(mtx))
        return 1 + beta1 * np.dot(x, y) + beta2 * quadratic_term + beta3 * gaussian_term - beta3

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
        if activation_noise < 1 / 2:
            status = True
            # The constraint represented by g contains a signal
            for Kbi_varphi_varphi in Kbs_varphi_varphi:
                Ebi = np.dot(yb, np.dot(Kbi_varphi_varphi, yb))
                activations = np.append(activations, Ebi / Eb)

        return status, activations, yb

    def discovery_in_graph(self, X, ks, gamma, G, names, examing_nodes, beta1=1e-1, beta2=1e-2, beta3=1e-3,
                           epsilon=1e-3, verbose=False):
        """
        Discover relations in a graph for given nodes, built on top of a networkx directed graph (DiGraph) G

        Inputs:

        X:          d x N matrix
            Dataset, samples of (X1_i; ...; Xd_i), i=1:N. If d variables = nodes in the graph and N timesteps
        ks:         list
            Array of kernels, one for each node;
        gamma:      positive real
            The regularization constant for the constraints
        G:          graph
            The pre-built graph, which already contains some relations
        names:      list
            List of names of nodes in G
        examing_nodes   list
            List of nodes in G to be examined
        beta1:      positive real
            The penalization on the linear kernel
        beta2:      positive real
            The penalization on the quadratic kernel
        beta3:      positive real
            The penalization on the fully nonlinear kernel
        epsilon:    positive real
            The threshold to determine the number of eigenvalues selected
        verbose (Default = False): Bool
            Whether to print intermediate results

        Outputs:

        G: Networkx directed graph
        (Not yet implemented to return the weights of each arrow)
        """
        d, N = X.shape
        self.d, self.N = d, N

        # We always add an extra constant node as a hidden node in the graph to the zero slot
        # The constant node will not appear in the graph but is used in the calculation
        # Thus, the ith node in the graph.nodes is the (i+1)th node in the underneath computations

        # Create modes
        modes = onp.array([ConstantMode(onp.array([-1]))])
        # Add linear modes
        linear_modes = [LinearMode(onp.array([i]), onp.array([names[i]]), beta1) for i in range(d)]
        modes = onp.append(modes, linear_modes)
        # Add quadratic modes
        quadratic_modes = [QuadraticMode(onp.array([i, j]), onp.array([names[i], names[j]]), beta2) for i in range(d)
                           for j in range(i + 1)]
        modes = onp.append(modes, quadratic_modes)

        # Add kernel modes
        kernel_modes = [KernelMode(onp.array(list(s)), names[onp.array(list(s))], beta3, ks[onp.array(list(s))]) for s
                        in powerset(range(d))]
        modes = onp.append(modes, kernel_modes)

        # The codes below are equivalent to the above two lines. Just for debuging purpose
        # for s in powerset(range(d)):
        #     _nodes =
        #     _names = names[_nodes]
        #     _ks = ks[_nodes]
        #     mode = KernelMode(_nodes, _names, beta3, _ks)
        #     modes = onp.append(modes, onp.array([mode]))

        # The number of modes
        modes_num = len(modes)

        # The correspondence between modes and nodes.
        # Entries at (i, j) equal to 1 implies that there is a relation between the ith mode and the jth node.
        # Otherwise, entries at (i, j) equal to 0 implies there is no relation between the ith mode and the jth node.
        # The matrix size is (modes_num, d + 1), which means we have the constant node.
        correspondence_mtx = onp.zeros((modes_num, d + 1))
        for i in range(len(modes)):
            nodes = modes[i].nodes + 1
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

        Kbs_varphi_varphi = onp.zeros((len(modes), r, r))  # Record the matrix Kb(varphi, varphi) for each mode

        for t in range(len(modes)):
            mode = modes[t]
            Kbt = self.Kn(lambda x, y: mode.kappa(x, y), X.T)
            func = (lambda ar, ac: np.dot(ar, np.dot(Kbt, ac)))
            func = (vmap(func, in_axes=(None, 0)))
            func = jit(vmap(func, in_axes=(0, None)))
            Kbs_varphi_varphi[t, :, :] = func(alphas, alphas)

        # Next, we determine ancestors of Nodes.
        examing_nodes_indices = examing_nodes + 1  # We add it by 1 since we added a hidden constant node

        for i in examing_nodes_indices:
            if verbose: print('Examining Node {0}'.format(names[i - 1]))
            modes_i = onp.where(correspondence_mtx[:, i] == 1)[0]  # The indices of modes related to the ith node

            # The operations below put the mode beta1 * xi * xi' into Class a, all other modes related to the ith
            # node to Class c, and the rest modes to Class b
            activation_codes = onp.ones(
                modes_num)  # Record the activation of each mode, 0 for Class a, 1 for Class b, -1 for Class c
            activation_codes[modes_i] = -1  # Put all the modes related to the ith node to Class c
            activation_codes[i] = 0  # Set the ith mode (beta1 * xi * xi') to be in Class a.
            ##################################################################################
            # Deal with the constraint that g_a = x_i

            # build the vector [varphi, ga]
            ga = X[i - 1]  # Get the data of the ith node. i-1 is the index of the data of the ith node in the graph
            ga = np.reshape(ga, (1, -1))
            varphi_ga = np.dot(ga, alphas.T).flatten()

            modes_b_indices = onp.where(activation_codes == 1)[0]  # Choose those modes in Class b
            Kbis_varphi_varphi = Kbs_varphi_varphi[modes_b_indices, :,
                                 :]  # Record the matrix Kbi(varphi, varphi) for each activated mode Kbi in Class b

            # Solve the constraint variational problem
            # status == False implies that the signal is mostly noisy, then activations = None and yb = None
            # status == True means that there is a signal given the constraint. Then, activations contains the
            # activation (the ratio of the energy of each mode and the total energy) of each mode
            status, activations, yb = self.energy_estimate(gamma, varphi_ga, Kbis_varphi_varphi)
            # If status == True, node i has ancestors. We need to determine the ancestors of node i.
            nodes_ac = np.setdiff1d(np.array(range(d + 1)), np.array([i]))  # the ancestor candidates (acronym ac)
            modes_ac = modes_b_indices.copy()
            if not status: nodes_ac = np.array([])
            _activations = activations.copy()

            # We start to determine ancestors of nodes by iteratively pruning out nodes with the least energies
            while status and (len(nodes_ac) > 1):
                energies = onp.zeros(len(nodes_ac))
                for iter in range(len(nodes_ac)):
                    j = nodes_ac[iter]
                    # The indices of activated modes associated with node j. The indices are numbered according to the array of modes
                    modes_j = onp.where(correspondence_mtx[:, j] == 1)[0]
                    activated_modes_j = set(modes_j).intersection(set(modes_b_indices))
                    # The indices of activations associated with mode j. The indices are numbered according to the aray _activations
                    activation_indices_j = onp.where(onp.isin(modes_ac, list(activated_modes_j)))[0]
                    # Compute the sum of activations associated with mode j
                    energies[iter] = np.sum(_activations[activation_indices_j])

                # We sort the energies in an ascending order
                sorted_energies_indices = onp.argsort(energies)
                # Preclude one ancestor candidate with the least energy from the list
                pre_nodes_ac = nodes_ac[sorted_energies_indices]
                pre_nodes_ac = pre_nodes_ac[1:]

                # Next, we get all the activated modes associated with the ancestor candidates
                pre_modes_ac = onp.array([], onp.int64)
                for s in pre_nodes_ac:
                    # The indices of activated modes associated with the potential ancestor s.
                    # The indices are numbered according to the array of modes
                    modes_s = onp.where(correspondence_mtx[:, s] == 1)[0]
                    activated_modes_s = set(modes_s).intersection(set(modes_b_indices))
                    pre_modes_ac = onp.concatenate((pre_modes_ac, onp.array(list(activated_modes_s))))
                pre_modes_ac = onp.array(list(set(pre_modes_ac))).astype(onp.int64)
                # We filter out the modes that related to other nodes except nodes in pre_nodes_ac
                valid_modes_ac = []
                for k in range(len(pre_modes_ac)):
                    mode_ac = pre_modes_ac[k]
                    node_ac = modes[mode_ac].nodes + 1
                    validation_list = np.setdiff1d(node_ac, pre_nodes_ac)
                    if len(validation_list) == 0:
                        valid_modes_ac.append(k)
                pre_modes_ac = pre_modes_ac[onp.array(valid_modes_ac).astype(onp.int64)]
                # Now, pre_modes_ac contains modes related only to nodes in pre_nodes_ac

                # Record the matrix Kbi(varphi, varphi) for each activated mode Kbi in pre_modes_ac
                Kbis_varphi_varphi_candidates = Kbs_varphi_varphi[pre_modes_ac, :, :]

                # Solve the constraint variational problem with only ancestor candidates
                status, _activations, pre_yb = self.energy_estimate(gamma, varphi_ga, Kbis_varphi_varphi_candidates)

                # if status == True, Node i can be expressed as a function of nodes in pre_nodes_ac
                if status:
                    nodes_ac = pre_nodes_ac
                    modes_ac = pre_modes_ac
                    yb = pre_yb
                    activations = _activations

            # Now, nodes_ac contains all the ancestor candidates after pruning out redundant variables.
            # modes_ac contains all the modes related only to nodes in nodes_ac
            # activations contains activations of modes_ac

            # Next, we start to prune out redundant modes associated with nodes in nodes_ac
            if len(nodes_ac) > 0:
                Kbis_varphi_varphi_ancestors = Kbs_varphi_varphi[modes_ac, :, :]
                modes_a = modes_ac  # the modes of ancestors

                status = True
                # Once we have the ancestors, we can prune out the modes associated with the ancestors
                # We prune out the mode of the lowest activation each time
                while status and (len(activations) > 1):
                    sorted_activation_indices = np.argsort(activations).tolist()
                    pre_Kbis_varphi_varphi = Kbis_varphi_varphi_ancestors[sorted_activation_indices[1:], :, :]
                    pre_indices = modes_a[sorted_activation_indices[1:]]

                    status, activations, pre_yb = self.energy_estimate(gamma, varphi_ga, pre_Kbis_varphi_varphi)

                    if status:
                        Kbis_varphi_varphi_ancestors = pre_Kbis_varphi_varphi
                        modes_a = pre_indices
                        yb = pre_yb

                disactivated_modes_indices = np.setdiff1d(modes_b_indices, modes_a).tolist()
                activation_codes[disactivated_modes_indices] = -1  # Put all the disactivated modes to Class c.

                """ Add all the edges associated with the ith node  """
                modes_b_indices = onp.where(activation_codes == 1)[0]
                modes_b = modes[modes_b_indices]
                for mode in modes_b:
                    nodes = mode.nodes + 1
                    for node in nodes:
                        if node != i and (not G.has_edge(names[node - 1], names[i - 1])) and (node != 0):
                            G.add_edge(names[node - 1], names[i - 1])

                if verbose: self.print_equations(i, X, N, names, modes_b, alphas, yb)

        return G

    def print_equations(self, i, X, N, names, modes, alphas, yb):
        """ Print out the equations """

        # We first classify the modes into two categories:
        # modes with features (k(x, y) = psi(x)psi(y)) and modes with kernels (kernels cannot be splitted)

        modes_features = onp.array([])
        modes_kernels = onp.array([])
        for mode in modes:
            if "Kernel" == mode.type:
                modes_kernels = onp.append(modes_kernels, mode)
            else:
                modes_features = onp.append(modes_features, mode)

        """
        The equation found is g = g_a + g_b, where g_a = x_i
        g_b = K_b(., varphi)y_b.

        For simplicity we write g_b = K(., varphi)y
        We have 
            K(., varphi)y = \sum_i^r y_i \sum_s^N K(., x_s) alpha_{s, i}
                          = \sum_s^N K(., x_s) \sum_i^r y_i alpha_{s, i}
        So, we first compute \sum_i^r y_i alpha_{s, i} for each s, which is fixed
        """
        w = np.dot(alphas.T, yb)

        """
        Now , we can write K(., varphi)y = \sum_s^N K(., x_s) w_s

        We first print out equations  related to feature modes

        For K(x, y) = <psi(x), psi(y)>, we have
            K(., varphi)y = <psi(x), \sum_s^N psi(x_s)w_s>.

        Thus, we compute \sum_s^N psi(x_s)w_s
        """
        # print the equation representing x_i as the function of other variables
        print("Node {} as a function of other nodes".format(names[i - 1]))
        eq = "{}".format(names[i - 1])

        if len(modes_features) > 0:
            # Psi = [psi(x_1);...;psi(x_N)]
            Psi = onp.zeros((len(modes_features), N))
            for t in range(len(modes_features)):
                mode = modes_features[t]
                Psi[t, :] = vmap(mode.feature)(X.T)

            # Get the weights of the feature maps.
            # For instance, if psi(x) = beta * phi(x), we return beta
            coeffs = onp.array([mode.coeff() for mode in modes])

            weights_i = np.dot(Psi, w)
            weights_i = weights_i * coeffs

            count = 0
            for mode in modes_features:
                eq = eq + ' + '
                eq = eq + "({} {})".format(weights_i[count], mode.to_string())
                count = count + 1
        if len(modes_kernels) > 0:
            """
            Next, we compute the equations related to non-separable kernels

            Note that K(., varphi)y = \sum_s^N K(., x_s) w_s
            """
            for mode in modes_kernels:
                eq = eq + ' + '
                part = "{} \\ sum_{{s=1}}^{} ".format(mode.coeff() ** 2, N)
                part = part + mode.to_string("x[s]") + "w[s]"
                eq = eq + "(" + part + ")"

        eq = eq + ' = 0'
        print(eq)
        if len(modes_kernels) > 0:
            print("w = ")
            print(w)
        print("")

    def plot_graph(self, G):
        nx.draw(G, with_labels=True, pos=nx.kamada_kawai_layout(G, G.nodes()), node_size=600, font_size=8, alpha=0.6)

