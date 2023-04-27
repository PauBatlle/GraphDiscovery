import numpy as onp
import networkx as nx
from tqdm import tqdm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

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
        mat=self.modes.get_K()
        self.print_func(f'Performing eigenvalue decomposition on matrix of shape {mat.shape}')
        self.eigenvalues,self.alphas=onp.linalg.eigh(mat)

        self.G=nx.DiGraph()
        self.G.add_nodes_from(names)

    
    def solve_variationnal(ga,K,gamma,regularize_noise=False,K_inv=None):
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
        print("beta",beta)
        print('K1',onp.trace(K_inv))
        print('K2',onp.trace(K2))
        print('K3',onp.trace(K3))
        print('K4',onp.trace(K4))
        Var_A=beta**2*onp.trace(K2)-2*beta*onp.trace(K3)+onp.trace(K4)
        #Z=onp.random.normal(size=(20000,K_inv.shape[0]))
        #A=-onp.einsum('ij,ij->i',Z,Z@K2)+alpha*onp.einsum('ij,ij->i',Z,Z@K_inv)
        #print(A.shape)
        #print('mean',onp.mean(A),esp_A)
        #print('var',onp.var(A),2*Var_A)
        print("esp",esp_A)
        print("std",onp.sqrt(2*Var_A))
        return -esp_A/onp.sqrt(2*Var_A)
    
    def alphas_from_kPCA_method(alphas,eigenvalues,kPCA='noise regularization',gamma=None,tolerance=None,number_of_eigenvectors=None):
        if kPCA=='noise regularization':
            assert gamma is not None
            if number_of_eigenvectors is not None:
                alphas=alphas[-number_of_eigenvectors:][eigenvalues>gamma]
            else:
                alphas=alphas[eigenvalues>gamma]
        elif kPCA=='eigenvalue ratio':
            assert tolerance is not None
            alphas=alphas[eigenvalues/eigenvalues[-1]>tolerance]
        elif kPCA=='fixed number':
            alphas=alphas[-number_of_eigenvectors:]
        else:
            raise Exception('kPCA must be in {"noise regularization","eigenvalue ratio","fixed number"}')
        return alphas
    
    def get_alphas(self,**kwargs):
        return GraphDiscoveryNew.alphas_from_kPCA_method(self.alphas,self.eigenvalues,**kwargs)
        
    
    def find_ancestors(self,name,gamma_0='auto',kPCA='noise regularization',tolerance=None,number_of_eigenvectors=None,gamma_min=1e-9):
        
        index=self.name_to_index[name]
        alphas=self.get_alphas(kPCA=kPCA,gamma=gamma_0,tolerance=tolerance,number_of_eigenvectors=number_of_eigenvectors)
        ga=alphas@self.X[index]
        active_modes=self.modes.delete_node(index)
        active_modes=active_modes.change_of_basis(alphas)
        self.print_func(f'PCA reduced dimension to {active_modes.constant_mat.shape[0]}')
        #if gamma_0=='auto':
        #        gamma=GraphDiscoveryNew.find_gamma(active_modes.get_K())
        #        gamma=max(gamma,gamma_min)
        #        print(f'found gamma={gamma:.2e}')
        #else:
        #    gamma=gamma_0

        for which in ['linear','quadratic','gaussian']:
            K_inv=active_modes.get_K(which)
            if gamma_0=='auto':
                gamma=GraphDiscoveryNew.find_gamma(active_modes.get_K(which))
                gamma=max(gamma,gamma_min)
                print(f'found gamma={gamma:.2e}')
            else:
                gamma=gamma_0
                
            K_inv+=gamma*onp.eye(K_inv.shape[0])
            K_inv=onp.linalg.inv(K_inv)
            yb,noise=GraphDiscoveryNew.solve_variationnal(ga,active_modes.get_K(which),gamma,K_inv=K_inv)
            Z=GraphDiscoveryNew.Z_test(noise,gamma*K_inv) ##beta=noise
            #Z=GraphDiscoveryNew.Z_test(noise,K_inv)
            print(which,"noise",noise,"Z",Z)
            if abs(Z)>1.96:
                continue
            print("break instead")
        
        if abs(Z)<1.96:
            self.print_func(f'{name} has no ancestors (Z test={Z:.2f})\n')
            return
        self.print_func(f'{name} has ancestors with {which} kernel (Z test={Z:.2f})')
        active_modes.set_level(which)
        _,ancestor_modes=GraphDiscoveryNew.recursive_ancestor_finder(ga,active_modes,yb,gamma)
        self.print_func(ancestor_modes,'\n')
        for ancestor_name in ancestor_modes.names:
            self.G.add_edge(ancestor_name,name,type=which)

    
    def recursive_ancestor_finder(ga,active_modes,yb,gamma):
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
        
        if abs(new_Z)>1.96:
            if new_modes.node_number==1:
                return new_yb,new_modes
            else:
                return GraphDiscoveryNew.recursive_ancestor_finder(ga,new_modes,new_yb,gamma)
        else:
            return yb,active_modes
        

    def find_gamma(K,option='variance',tol=1e-10):
        assert option in {'variance','optimal_ratio'}
        eigenvalues=onp.linalg.eigvalsh(K)
        print(f"max {eigenvalues.max():.2e}, min {eigenvalues.min():.2e}")
        plt.figure()
        plt.hist(eigenvalues[:-1],bins=100)
        plt.show()
        eigenvalues=eigenvalues[eigenvalues>tol]
        if option=='variance':
            
            var=lambda gamma_log:-onp.var(1/(1+eigenvalues*onp.exp(-gamma_log)))
            res = minimize(var, onp.array([0]), method='nelder-mead',
                           options={'xatol': 1e-8, 'disp': True})
            
            
        if option=='optimal_ratio':
            def to_optimize(gamma_log):
                vals=1/(1+eigenvalues*onp.exp(-gamma_log))
                return abs(0.5-(onp.mean(vals**2)+vals[-1]**2)/(onp.mean(vals)-vals[-1]))

            res = minimize(to_optimize, onp.array([0]), method='nelder-mead',
                           options={'xatol': 1e-8, 'disp': True})
        gammas=onp.logspace(-6,2,20)
        
        #plt.figure()
        #plt.plot(gammas,[to_optimize(gamma) for gamma in gammas])
        #plt.xscale("log")
        #plt.yscale("log")
        #plt.show()
        
        plt.figure()
        plt.plot(range(len(eigenvalues)),eigenvalues)
        plt.yscale('log')
        plt.show()
        plt.figure()
        plt.hist(1/(1+eigenvalues*onp.exp(-res.x[0])),bins=100)
        plt.show()
        return onp.exp(res.x[0])
        

    
    def find_threshold_gamma(self,name,type_of_kernel='gaussian',tol_binary_search=1e-2,kPCA='fixed number',tolerance=None,number_of_eigenvectors=None):
        index=self.name_to_index[name]
        alphas=self.get_alphas(kPCA=kPCA,gamma=None,tolerance=tolerance,number_of_eigenvectors=number_of_eigenvectors)
        ga=alphas@self.X[index]
        active_modes=self.modes.delete_node(index)
        active_modes=active_modes.change_of_basis(alphas)
        return GraphDiscoveryNew.binary_search(ga,active_modes,type_of_kernel,tol_binary_search)


    def binary_search(ga,active_modes,type_of_kernel,tol):
        left=-15
        right=0
        _,noise_left=GraphDiscoveryNew.solve_variationnal(ga,active_modes.get_K(type_of_kernel),10**left)
        while not 0<=noise_left<=1:
            left+=0.5
            _,noise_left=GraphDiscoveryNew.solve_variationnal(ga,active_modes.get_K(type_of_kernel),10**left)
        _,noise_right=GraphDiscoveryNew.solve_variationnal(ga,active_modes.get_K(type_of_kernel),10**right)
        while not 0.5<=noise_right<=1:
            right+=1
            _,noise_right=GraphDiscoveryNew.solve_variationnal(ga,active_modes.get_K(type_of_kernel),10**right)
        noise_middle=1
        while abs(noise_middle-0.5)>tol:
            middle=(left+right)/2
            _,noise_middle=GraphDiscoveryNew.solve_variationnal(ga,active_modes.get_K(type_of_kernel),10**middle)
            #print(f'Gamma :1e{middle:.4f}, Computed noise : {noise_middle}')
            if noise_middle<0.5:
                left=middle
            else:
                right=middle
        return 10**middle
    
    def activations_gamma(self,name,gamma_range,type_of_kernel='gaussian'):
        index=self.name_to_index[name]
        ga=self.alphas@self.X[index]
        active_modes=self.modes.delete_node(index)
        activations=[]
        for gamma in tqdm(gamma_range):
            yb,_=GraphDiscoveryNew.solve_variationnal(ga,active_modes.get_K(type_of_kernel),gamma)

            energy=-onp.dot(ga,yb)
            noise=-gamma*onp.dot(yb,yb)/onp.dot(ga,yb)
            activations.append([noise]+[onp.dot(yb,active_modes.get_K_of_index(i)@yb)/energy for i in range(active_modes.node_number)])
        return activations

    def noise_gamma(self,name,gamma,type_of_kernel='gaussian'):
        index=self.name_to_index[name]
        ga=self.alphas@self.X[index]
        active_modes=self.modes.delete_node(index)
        
        yb,_=GraphDiscoveryNew.solve_variationnal(ga,active_modes.get_K(type_of_kernel),gamma)

        energy=-onp.dot(ga,yb)
        noise=-gamma*onp.dot(yb,yb)/onp.dot(ga,yb)
        return noise

    
    def find_gamma_behavior(self,name,gamma_list,kPCA='noise regularization',tolerance=None,number_of_eigenvectors=None):
        result={'which':[],'activations':[],'noise':[],'ancestors':[]}
        index=self.name_to_index[name]

        alphas=self.get_alphas(kPCA=kPCA,gamma=gamma,tolerance=tolerance,number_of_eigenvectors=number_of_eigenvectors)
        ga=alphas@self.X[index]
        active_modes=self.modes.delete_node(index)
        active_modes=active_modes.change_of_basis(alphas)

        for gamma in gamma_list:
            
            for which in ['linear','quadratic','gaussian']:
                yb,noise=GraphDiscoveryNew.solve_variationnal(ga,active_modes.get_K(which),gamma)
                if 0<noise<=0.5:
                    result['which'].append(which)
                    break
            
            if not 0<noise<=0.5:
                result['noise'].append(noise)
                result['which'].append('noise')
                result['activations'].append([None]*(len(self.names)-1))
                result['ancestors'].append([])
            else:
                active_modes.set_level(which)
                yb,ancestor_modes=GraphDiscoveryNew.recursive_ancestor_finder(ga,active_modes,yb,gamma)
                energy=-onp.dot(ga,yb)
                acti={n:None for n in active_modes.names}
                for i,n in enumerate(ancestor_modes.names):
                    acti[n]=onp.dot(yb,ancestor_modes.get_K_of_index(i)@yb)/energy
                result['activations'].append(list(acti.values()))
                new_noise=-gamma*onp.dot(yb,yb)/onp.dot(ga,yb)
                assert new_noise<0.5
                result['noise'].append(new_noise)
                result['ancestors'].append(ancestor_modes.names)
        return result
    
    def noise_ratio_test(self,name,gamma,type_of_kernel='gaussian',N_samples=1000):
        index=self.name_to_index[name]
        Y=self.X[index]
        active_modes=self.modes.delete_node(index)

        K1=active_modes.get_K(type_of_kernel)
        K=K1+gamma*onp.eye(K1.shape[0])

        Y_sol=onp.linalg.solve(K,Y)

        s_n=onp.dot(Y_sol,(K1-gamma*onp.eye(K1.shape[0]))@Y_sol)

        Z=onp.random.standard_normal((K1.shape[0],N_samples))
        K_inv=onp.linalg.inv(K)
        mat_H1=onp.eye(K1.shape[0])-2*gamma*K_inv
        mat_H0=gamma*K_inv@mat_H1
        sn_H0=onp.einsum('ij,ij->j',Z,mat_H0@Z)
        sn_H1=onp.einsum('ij,ij->j',Z,mat_H1@Z)
        return sn_H0,sn_H1,s_n
    
    def find_gamma_H0(self,name,type_of_kernel='gaussian'):
        index=self.name_to_index[name]
        Y=self.X[index]
        active_modes=self.modes.delete_node(index)

        K1=active_modes.get_K(type_of_kernel)
        tr=onp.trace(K1)
        print(tr)
        mean_val=lambda gamma:tr/gamma-K1.shape[0]
        eigenvalues,eigenvectors=onp.linalg.eigh(K1)
        s_n=lambda gamma:onp.dot(eigenvectors.T@Y,onp.diag((eigenvalues-gamma)/((eigenvalues+gamma)**2))@eigenvectors.T@Y)
        gammas=onp.logspace(3,10,100)
        return gammas,[s_n(gamma)-mean_val(gamma) for gamma in tqdm(gammas)]

    
    def find_gamma_H1(self,name,type_of_kernel='gaussian'):
        index=self.name_to_index[name]
        Y=self.X[index]
        active_modes=self.modes.delete_node(index)

        K1=active_modes.get_K(type_of_kernel)
        eigenvalues,eigenvectors=onp.linalg.eigh(K1)
        mean_val=lambda gamma:K1.shape[0]-2*onp.sum(gamma/(gamma+eigenvalues))
        s_n=lambda gamma:onp.dot(eigenvectors.T@Y,onp.diag((eigenvalues-gamma)/((eigenvalues+gamma)**2))@eigenvectors.T@Y)
        gammas=onp.logspace(-6,2,100)
        return gammas,[s_n(gamma)-mean_val(gamma) for gamma in tqdm(gammas)]

        

        
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
    
    
    
        