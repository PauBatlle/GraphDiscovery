import numpy as onp
import networkx as nx
from tqdm import tqdm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from functools import partial

class GraphDiscoveryNew():

    def __init__(self,X,beta,names,l=1,verbose=True) -> None:
        self.X=X
        self.print_func = print if verbose else lambda *a, **k: None

        self.print_func('Computing kernel matrix')
        constant_mat=onp.ones((X.shape[1],X.shape[1]))
        linear_mat=onp.expand_dims(X,-1)*onp.expand_dims(X,1)
        quadratic_mat=onp.expand_dims(linear_mat,0)*onp.expand_dims(linear_mat,1)
        diff_X=onp.tile(onp.expand_dims(X,-1),(1,1,X.shape[1]))-onp.tile(onp.expand_dims(X,1),(1,X.shape[1],1))
        gaussian_mat=onp.exp(-(diff_X/l)**2/2)

        self.beta=beta
        level=onp.ones_like(beta)
        self.names=names
        self.name_to_index={name:index for index,name in enumerate(names)}

        self.modes=ModeContainer(constant_mat,linear_mat,quadratic_mat,gaussian_mat,onp.eye(X.shape[1]),names,beta,level)
        

        self.G=nx.DiGraph()
        self.G.add_nodes_from(names)
    

    @property
    def alphas(self):
        try:
            return self._alphas
        except:
            mat=self.modes.get_K()
            self.print_func(f'Performing eigenvalue decomposition on matrix of shape {mat.shape}')
            self._eigenvalues,self._alphas=onp.linalg.eigh(mat)
            return self._alphas
    
    @property
    def eigenvalues(self):
        try:
            return self._eigenvalues
        except:
            mat=self.modes.get_K()
            self.print_func(f'Performing eigenvalue decomposition on matrix of shape {mat.shape}')
            self._eigenvalues,self._alphas=onp.linalg.eigh(mat)
            return self._eigenvalues

    
    def solve_variationnal(ga,K=None,gamma=None,regularize_noise=False,K_inv=None):
        if regularize_noise:
            min_eigenval=onp.linalg.eigvalsh(K)[0]
            noise_level=gamma+abs(min_eigenval)-min_eigenval
        else:
            noise_level=gamma
        if K_inv is None:
            yb=-onp.linalg.solve(K+noise_level*onp.eye(K.shape[0]),ga)
            noise=-noise_level*onp.dot(yb,yb)/onp.dot(ga,yb)
        else:
            yb=-K_inv@ga
            noise=-noise_level*onp.dot(yb,yb)/onp.dot(ga,yb)
        return yb,noise
    
    def Z_test(beta,K_inv):
        K2=K_inv@K_inv
        esp_A=-beta*onp.trace(K_inv)+onp.trace(K2)
        K3=K_inv@K2
        K4=K_inv@K3
        Var_A=beta**2*onp.trace(K2)-2*beta*onp.trace(K3)+onp.trace(K4)
        return -esp_A/onp.sqrt(2*Var_A)
    
    def alphas_from_kPCA_method(alphas,eigenvalues,kPCA='noise regularization',gamma=None,tolerance=None,number_of_eigenvectors=None):
        if kPCA=='noise regularization':
            assert gamma is not None and gamma!='auto'
            if number_of_eigenvectors is not None:
                alphas_to_return=alphas[-number_of_eigenvectors:][eigenvalues>gamma]
            else:
                alphas_to_return=alphas[eigenvalues>gamma]
        elif kPCA=='eigenvalue ratio':
            assert tolerance is not None
            alphas_to_return=alphas[eigenvalues/eigenvalues[-1]>tolerance]
        elif kPCA=='fixed number':
            alphas_to_return=alphas[-number_of_eigenvectors:]
        else:
            raise Exception('kPCA must be in {"noise regularization","eigenvalue ratio","fixed number"}')
        return alphas_to_return
    

    def get_alphas(self,kPCA='noise regularization',gamma=None,tolerance=None,number_of_eigenvectors=None):
        if kPCA=='no':
            return onp.eye(self.modes.get_K().shape[0])
        return GraphDiscoveryNew.alphas_from_kPCA_method(self.alphas,self.eigenvalues,kPCA,gamma,tolerance,number_of_eigenvectors)
        
    
    def find_ancestors(self,name,gamma='auto',kPCA='no',acceptation_logic='default',PCAtolerance=None,number_of_eigenvectors=None,gamma_min=1e-9):
        
        index=self.name_to_index[name]
        alphas=self.get_alphas(kPCA=kPCA,gamma=gamma,tolerance=PCAtolerance,number_of_eigenvectors=number_of_eigenvectors)
        ga=alphas@self.X[index]
        active_modes=self.modes.delete_node(index)
        active_modes=active_modes.change_of_basis(alphas)
        if kPCA!='no':
            self.print_func(f'PCA reduced dimension to {active_modes.constant_mat.shape[0]}')
        
        if acceptation_logic=='default':
            acceptation_logic=GraphDiscoveryNew.acceptation_logic(cutoff=0.4,use_Z=True)
        else:
            assert callable(acceptation_logic)

        for which in ['linear','quadratic','gaussian']:
            K=active_modes.get_K(which)
            if gamma=='auto':
                gamma_used=GraphDiscoveryNew.find_gamma(K)
                gamma_used=max(gamma_used,gamma_min)
            else:
                gamma_used=gamma
                
            K+=gamma_used*onp.eye(K.shape[0])
            K_inv=onp.linalg.inv(K)
            yb,noise=GraphDiscoveryNew.solve_variationnal(ga,gamma=gamma_used,K_inv=K_inv)
            Z=GraphDiscoveryNew.Z_test(noise,gamma_used*K_inv)

            accept=acceptation_logic(noise,Z,which)
            self.print_func(f'{which} kernel (using gamma={gamma_used:.2e})\n n/(n+s)={noise:.2f}, Z={Z:.2f}\n decision : {"refused"*int(not(accept))+"accepted"*int(accept)}')
            if accept:
                break
        
        if not accept:
            self.print_func(f'{name} has no ancestors (n/(s+n)={noise:.2f})\n')
            return
        self.print_func(f'{name} has ancestors with {which} kernel (n/(s+n)={noise:.2f})')
        active_modes.set_level(which)
        _,ancestor_modes=GraphDiscoveryNew.recursive_ancestor_finder(ga,active_modes,yb,gamma_used,acceptation_logic=partial(acceptation_logic,which=which))
        self.print_func('ancestors after pruning: ',ancestor_modes,'\n')
        for ancestor_name in ancestor_modes.names:
            self.G.add_edge(ancestor_name,name,type=which)
    
    def acceptation_logic(cutoff,use_Z):
        def func(noise,Z,which):
            if noise<cutoff:
                return True
            if use_Z and which=='gaussian':
                return abs(Z)>1.96
            return False
        return func


    
    def recursive_ancestor_finder(ga,active_modes,yb,gamma,acceptation_logic):
        energy=-onp.dot(ga,yb)
        activations=[onp.dot(yb,active_modes.get_K_of_index(i)@yb)/energy for i in range(active_modes.node_number)]
        new_modes=active_modes.delete_node(onp.argmin(activations))
        K_inv=new_modes.get_K()
        K_inv+=gamma*onp.eye(K_inv.shape[0])
        K_inv=onp.linalg.inv(K_inv)
        new_yb,new_noise=GraphDiscoveryNew.solve_variationnal(
            ga,
            new_modes.get_K(),
            gamma=gamma,
            K_inv=K_inv)
        new_Z=GraphDiscoveryNew.Z_test(new_noise,gamma*K_inv) ##beta=noise
        accept=acceptation_logic(noise=new_noise,Z=new_Z)
        
        if accept:
            if new_modes.node_number==1:
                return new_yb,new_modes
            else:
                return GraphDiscoveryNew.recursive_ancestor_finder(ga,new_modes,new_yb,gamma,acceptation_logic)
        else:
            return yb,active_modes
        

    def find_gamma(K,option='optimal_ratio',tol=1e-10):
        assert option in {'variance','optimal_ratio'}
        eigenvalues=onp.linalg.eigvalsh(K)
        eigenvalues=eigenvalues[eigenvalues>tol]
        if option=='variance':
            var=lambda gamma_log:-onp.var(1/(1+eigenvalues*onp.exp(-gamma_log)))
            res = minimize(var, onp.array([0]), method='nelder-mead',
                           options={'xatol': 1e-8, 'disp': False})
        if option=='optimal_ratio':
            def to_optimize(gamma_log):
                vals=1/(1+eigenvalues*onp.exp(-gamma_log))
                return abs(0.5-(onp.mean(vals**2)+vals[-1]**2)/(onp.mean(vals)-vals[-1]))
            res = minimize(to_optimize, onp.array([0]), method='nelder-mead',
                           options={'xatol': 1e-8, 'disp': False})
        return onp.exp(res.x[0])

 

  
    def plot_graph(self):
            nx.draw(self.G, with_labels=True, pos=nx.kamada_kawai_layout(self.G, self.G.nodes()), node_size=600, font_size=8, alpha=0.6)



    

class ModeContainer():

    def __init__(self,constant_mat,linear_mat,quadratic_mat,gaussian_mat,alphas,names,beta,level) -> None:
        self.constant_mat=constant_mat
        self.linear_mat=linear_mat
        self.quadratic_mat=quadratic_mat
        self.gaussian_mat=gaussian_mat
        self.alphas=alphas
        self.names=names
        self.beta=beta
        self.level=level
    
    def change_of_basis(self,alphas):
        new_constant_mat=alphas@self.constant_mat@alphas.T
        new_linear_mat=alphas@self.linear_mat@alphas.T
        new_quadratic_mat=alphas@self.quadratic_mat@alphas.T
        return ModeContainer(new_constant_mat,new_linear_mat,new_quadratic_mat,self.gaussian_mat,alphas,self.names,self.beta,self.level)

    @property
    def node_number(self):
        return self.names.shape[0]
    
    def delete_node(self,index):
        new_linear_mat = onp.delete(self.linear_mat,index,axis=0)
        new_quadratic_mat = onp.delete(self.quadratic_mat,index,axis=0)
        new_quadratic_mat = onp.delete(new_quadratic_mat,index,axis=1)
        new_gaussian_mat = onp.delete(self.gaussian_mat,index,axis=0)
        new_names=onp.delete(self.names,index)
        return ModeContainer(self.constant_mat,new_linear_mat,new_quadratic_mat,new_gaussian_mat,self.alphas,new_names,self.beta,self.level)
    
    def get_level(self,chosen_level):
        if chosen_level is None:
            return self.level
        if chosen_level=='linear':
            return onp.array([1]+[0]*(self.beta.shape[0]-1))
        if chosen_level=='quadratic':
            return onp.array([1,1]+[0]*(self.beta.shape[0]-2))
        if chosen_level=='gaussian':
            return onp.ones_like(self.beta)
        
        return onp.array([int(i<=int(chosen_level)) for i in range(self.beta.shape[0])])

    def set_level(self,chosen_level):
        assert chosen_level is not None
        self.level=self.get_level(chosen_level)

    def get_K(self,chosen_level=None):
        coeff=self.beta*self.get_level(chosen_level)
        K=self.constant_mat+coeff[0]*onp.sum(self.linear_mat,axis=0)+coeff[1]*onp.sum(self.quadratic_mat,axis=(0,1))+coeff[2]*self.alphas@onp.prod(self.gaussian_mat+onp.ones_like(self.gaussian_mat),axis=0)@self.alphas.T
        return K
    
    def get_K_of_index(self,index):
        coeff=self.beta*self.level
        res=onp.zeros_like(self.linear_mat[0])
        res+=coeff[0]*self.linear_mat[index]
        res+=coeff[1]*(2*onp.sum(self.quadratic_mat[index],axis=0)-self.quadratic_mat[index,index])
        new_non_linear_mat=onp.delete(self.gaussian_mat,index,axis=0)
        res+=coeff[2]*self.alphas@(self.gaussian_mat[index]*onp.prod(new_non_linear_mat+onp.ones_like(new_non_linear_mat),axis=0))@self.alphas.T
        return res#+self.constant_mat# no constant
    
    def get_K_without_index(self,index):
        coeff=self.beta*self.level
        new_linear_mat = onp.delete(self.linear_mat,index,axis=0)
        new_quadratic_mat = onp.delete(self.quadratic_mat,index,axis=0)
        new_quadratic_mat = onp.delete(new_quadratic_mat,index,axis=1)
        new_gaussian_mat = onp.delete(self.gaussian_mat,index,axis=0)
        res=onp.zeros_like(self.linear_mat[0])
        res+=coeff[0]*onp.sum(new_linear_mat,axis=0)
        res+=coeff[1]*onp.sum(new_quadratic_mat,axis=(0,1))
        res+=coeff[2]*self.alphas@onp.prod(new_gaussian_mat+onp.ones_like(new_gaussian_mat),axis=0)@self.alphas.T
        return res+self.constant_mat 
    
    def __repr__(self) -> str:
        return list(self.names).__repr__()
    
    
    
        