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
    def __init__(self,X,names,beta,G=None):
        self.d, self.N = tuple(X.shape)
        self.X=X
        self.names=names
        
        self.modes=self.setup_modes(beta)
        #self.correspondence_mtx=self.setup_correspondance_matrix()
        self.alphas=self.perform_PCA(epsilon=1e-3)
        #self.Kbs_varphi_varphi=self.setup_Kbs_varphi_varphi()
        if G is None:
            self.G=nx.DiGraph()
            for name in names:
                self.G.add_node(name)
        else:
            self.G=G

    def kernel(self, x, y, beta1, beta2, beta3, ks):
        gaussian_terms = [1 + k(x, y) for k in ks]
        gaussian_term = reduce(np.multiply, gaussian_terms)

        v = x * y
        vv = np.reshape(v, (-1, 1))
        mtx = np.dot(vv, vv.T)
        quadratic_term = np.sum(np.tril(mtx))
        return 1 + beta1 * np.dot(x, y) + beta2 * quadratic_term + beta3 * gaussian_term

    def Kn(kernel, funcs):
        return GraphDiscovery.Kmn(kernel, funcs, funcs)

    def Kmn(kernel, funcs_l, funcs_r):
        N_l, m_l = funcs_l.shape
        N_r, m_r = funcs_r.shape

        assert m_l == m_r

        BBv = np.reshape(np.tile(funcs_l, (1, N_r)), (N_l * N_r, m_l))
        BBh = np.reshape(np.tile(funcs_r, (N_l, 1)), (N_l * N_r, m_r))

        val = vmap(lambda x, y: kernel(x, y))(BBv, BBh)
        K = np.reshape(val, (N_l, N_r))
        return K
    


    def setup_modes(self,beta,l=1):
        constant_mode=ConstantMode()

        l_modes=[LinearMode(onp.array([i]), onp.array([self.names[i]]), 1) for i in range(self.d)]
        linear_modes=SingleModeContainer.compute_linear_mode_container(l_modes,self.X,constant_mode=constant_mode,beta1=beta[0])

        q_modes = [QuadraticMode(onp.array([i, j]), onp.array([self.names[i], self.names[j]]), 1) for i in range(self.d) for j in range(i+1)]
        quadratic_modes=PairwiseModeContainer.compute_quadratic_mode_container(q_modes,self.X,constant_mode=constant_mode)

        nl_modes = [NonLinearMode(onp.array([i]), onp.array([self.names[i]]), 1,lambda x,y:np.exp(-((x-y)/l)**2)) for i in range(self.d)]
        non_linear_modes=CombinatorialModeContainer.compute_gaussian_mode_container(nl_modes,l,self.X,constant_mode=constant_mode)


        modes=ModeContainer([linear_modes,quadratic_modes,non_linear_modes],beta,level=beta.shape[0])

        return modes
    
    #def setup_correspondance_matrix(self):
    #    correspondence_mtx = onp.zeros((len(self.modes), self.d))
    #    for i in range(len(self.modes)):
    #        nodes = self.modes[i].nodes
    #        correspondence_mtx[i, nodes] = 1
    #    return correspondence_mtx
    

    def perform_PCA(self,epsilon):
        Kxx = self.modes.K_mat

        ''' For debug purpose, check whether Kxx is formulated correctly '''
        # Kxx = self.Kn(lambda x, y: self.kernel(x, y, beta1, beta2, beta3, ks), X.T)
        # if verbose: print("Kxx error {}".format(np.linalg.norm(Kxx - Theta)))

        # Compute sorted eigenvalues and eigenfunctions of Kxx
        eigenValues, eigenVectors = np.linalg.eigh(Kxx)
        
        idx = np.argsort(-eigenValues.real)
        
        lambdas = eigenValues[idx].real
        alphas = eigenVectors[:, idx].real

        # Find r such that r = min{n| lambda_{n} / lambda_{1} \leq \epsilon}
        ratios = lambdas / lambdas[0]
        r = np.argwhere(ratios <= epsilon).flatten()[0]

        lambdas = lambdas[:r]
        # Normalize the eigenvectors (alphas) of Kxx such that \|alphas[i]\|^2=1
        alphas = vmap(lambda v, lbda: v / np.linalg.norm(v))(alphas.T[:r], lambdas)
        # Now, alphas is a r x N matrix, where each row is a normalized eigenvector
        self.modes.change_of_basis(alphas)
        return alphas
    
    
    def solve_variationnal(gamma,ga_varphi,Kb_varphi_varphi):
        noise_Kb_varphi_varphi = Kb_varphi_varphi + gamma * np.eye(Kb_varphi_varphi.shape[0])
        Ltmp = np.linalg.cholesky(noise_Kb_varphi_varphi)
        tmp = jsp.linalg.solve_triangular(Ltmp, ga_varphi, lower=True)
        yb = -jsp.linalg.solve_triangular(Ltmp.T, tmp, lower=False)

        return yb

    def activation_noise(gamma, ga_varphi, Kb_varphi_varphi):
        print('matrix',Kb_varphi_varphi)
        yb=GraphDiscovery.solve_variationnal(gamma,ga_varphi,Kb_varphi_varphi)
        print('yb',yb)
        
        Eb = -np.dot(ga_varphi, yb)
        print('Eb',Eb)
        return gamma * np.dot(yb, yb) / Eb

    def activations(gamma, ga_varphi, modes,yb):
        Eb = -np.dot(ga_varphi, yb)
        activation_noise = gamma * np.dot(yb, yb) / Eb
        matrices=onp.array(list(map(lambda i:modes.get_K_except_index(i),range(modes.modes_number))))
        activations=1-activation_noise-np.dot(np.matmul(matrices,yb),yb)/Eb
        

        return activation_noise,activations
    
    def prune_ancestors(modes,gamma, varphi_ga):
        yb=GraphDiscovery.solve_variationnal(gamma,varphi_ga,modes.K_mat)

        activation_noise,activations=GraphDiscovery.activations(gamma, varphi_ga, modes,yb)


        if activation_noise>0.5:
            return np.array([]),None
        while activation_noise<0.5 and (modes.modes_number > 1):
            print('activations',activations)
            print('noise',activation_noise)
            
            sorted_activation_indices = onp.argsort(activations)
            candidate_modes=modes.remove_least_activated(sorted_activation_indices)

            pre_yb=GraphDiscovery.solve_variationnal(gamma, varphi_ga, candidate_modes.K_mat)
            activation_noise,activations = GraphDiscovery.activations(gamma, varphi_ga, candidate_modes,yb)
            if activation_noise<0.5:
                modes=candidate_modes
                yb = pre_yb
        return modes,yb
    
    def find_level(modes,gamma,varphi_ga):
        for level in range(1,modes.beta.shape[0]+1):
            modes.level=level
            activation_noise=GraphDiscovery.activation_noise(gamma,varphi_ga,modes.K_mat)
            print(f'level: {level}',f'activation noise:{activation_noise:.2f}')
            if activation_noise<0.5:
                modes.remove_above_level()
                return True
        return False
        

    

    def prune_ancestors_recursive(modes,gamma,varphi_ga,yb=None):
        new_yb=GraphDiscovery.solve_variationnal(gamma,varphi_ga,modes.K_mat)

        activation_noise,activations=GraphDiscovery.activations(gamma, varphi_ga, modes,yb)

        if activation_noise>0.5:
            if yb is None:
                return ModeContainer([],None,0),None
            else:
                return modes,yb
        else:
            sorted_activation_indices = onp.argsort(activations)
            candidate_modes=modes.remove_least_activated(sorted_activation_indices)
            return GraphDiscovery.prune_ancestors_recursive(candidate_modes,gamma,varphi_ga,new_yb)
    

    
    def add_edges_to_graph(self,modes,index):
        for mode in modes.mode_containers[0].modes:
            nodes = mode.nodes
            for node in nodes:
                self.G.add_edge(self.names[node], self.names[index])
    
    def compute_coefficients_of_equation(self,i,yb,active_modes):
        modes_b_indices = onp.where(activation_codes == 1)[0]
        modes_b = self.modes[modes_b_indices]
        coeffs = onp.array([1]+[np.sqrt(active_modes.beta)])

        M1 = onp.zeros((len(modes_b), self.N))
        for t in range(len(modes_b)):
            mode = modes_b[t]
            M1[t, :] = vmap(mode.psi)(self.X.T)

        M = np.dot(M1, self.alphas.T)

        weights_i = np.dot(M, yb)
        weights_i = weights_i * coeffs
        # print the equation representing x_i as the function of other variables
        eq = f"{self.names[i]} = "
        for count,mode in enumerate(modes_b):
            eq = eq + f"{-round(weights_i[count], 2)} * {mode.to_string()} + "
        eq=eq[:-2]
        return eq

    
    def examine_node(self,i,gamma,verbose=False):
        if verbose: print('Examining Node {0}'.format(self.names[i]))

        # build the vector [varphi, ga]
        ga = self.X[i]
        varphi_ga = np.matmul(self.alphas,ga)
        print('varphi_ga',varphi_ga)
        #print([[str(mode) for mode in mode_container.modes]for mode_container in self.modes.mode_containers])
        active_modes=self.modes.remove_index(i)
        print(active_modes)
        if not GraphDiscovery.find_level(active_modes,gamma,varphi_ga):
            return 'no level found'
        active_modes,yb=GraphDiscovery.prune_ancestors(active_modes,gamma, varphi_ga)
        print(active_modes)

        self.add_edges_to_graph(active_modes,i)
        self.plot_graph()
        #return self.compute_coefficients_of_equation(i,yb,active_modes)




    def plot_graph(self):
            nx.draw(self.G, with_labels=True, pos=nx.kamada_kawai_layout(self.G, self.G.nodes()), node_size=600, font_size=8, alpha=0.6)

