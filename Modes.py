import matplotlib.pyplot as plt
import numpy as onp
from jax import jit, vmap, grad, hessian
import jax.scipy as jsp
from scipy.optimize import minimize
import networkx as nx
from kernels import *
from scipy.linalg import svd
from jax.config import config
config.update("jax_enable_x64", True)

class Mode(object):
    def __init__(self, nodes, names, beta):
        self.nodes = nodes
        self.beta = beta
        self.names = names

    def coeff(self):
        return onp.sqrt(self.beta)

class ConstantMode(Mode):
    """
    Represent the constant kernel, K(x, y) = beta
    """
    def __init__(self, nodes=onp.array([]), names=onp.array([]), beta=1):
        super().__init__(nodes, names, beta)

    def kappa(self, x, y):
        return self.beta


    def psi(self, x):
        return onp.sqrt(self.beta)

    def __repr__(self):
        return 'Cst()'
    
    def __str__(self):
        return 'ConstantMode'


class LinearMode(Mode):
    """
    Represent the linear kernel, K(x, y)=beta * x[i] * y[i]
    """
    def __init__(self, nodes, names, beta):
        super().__init__(nodes, names, beta)

    def kappa(self, x, y):
        return self.beta * x[self.nodes[0]] * y[self.nodes[0]]

    def psi(self, x):
        return onp.sqrt(self.beta) * x[self.nodes[0]]

    def __repr__(self):
        return f'LM({self.names[0]})'
    
    def __str__(self):
        return f'LinearMode({self.names[0]})'


class QuadraticMode(Mode):
    """
    Represent the quadratic kernel, K(x, y) = beta * x[i] * x[j] * y[i] * y[j]
    """
    def __init__(self, nodes, names, beta):
        super().__init__(nodes, names, beta)

    def kappa(self, x, y):
        return self.beta * x[self.nodes[0]] * x[self.nodes[1]] * y[self.nodes[0]] * y[self.nodes[1]]

    def psi(self, x):
        return onp.sqrt(self.beta) * x[self.nodes[0]] * x[self.nodes[1]]

    
    def __repr__(self):
        return f'QM({self.names[0]},{self.names[1]})'
    
    def __str__(self):
        return f'QuadraticMode({self.names[0]},{self.names[1]})'

class NonLinearMode(Mode):
    """
    Represent the nonlinear kernel, K(x, y) = beta * k(x[i] ; y[i])
    """
    def __init__(self, nodes, names, beta,k):
        super().__init__(nodes, names, beta)
        self.k=k

    def kappa(self, x, y):
        return self.beta * self.k(x[self.nodes[0]] , y[self.nodes[0]])

    #def psi(self, x):
    #    return onp.sqrt(self.beta) * x[self.nodes[0]] * x[self.nodes[1]]

    
    def __repr__(self):
        return f'NLM({self.names[0]})'
    
    def __str__(self):
        return f'NonLinearMode({self.names[0]})'

class ModeContainer():
    def __init__(self,mode_containers,beta,level):
        self.mode_containers=mode_containers
        self.beta=beta
        self.level=level
        self.modes_number=self.mode_containers[0].modes.shape[0]
    
    @property
    def K_mat(self):
        try:
            return self._kmat
        except:
            print('computing new K_mat')
            matrices=onp.array(list(map(lambda container:container.K_mat,self.mode_containers[:self.level])))
            self._kmat=onp.sum(self.beta[:self.level,onp.newaxis,onp.newaxis]*matrices,axis=0)
            return self._kmat
    
    @K_mat.deleter
    def K_mat(self):
        try:
            del self._kmat
        except:
            pass
    
    @property
    def level(self):
        return self._level
    
    @level.setter
    def level(self,new_level):
        self._level=new_level
        del self.K_mat
    

    def change_of_basis(self,alphas):

        del self.K_mat
        for modeContainer in self.mode_containers:
            modeContainer.change_of_basis(alphas)
        print('is it done now ? ')
    
    
    
    def remove_least_activated(self,sorted_activations):
        new_containers=list(map(lambda container:container.remove_least_activated(sorted_activations),self.mode_containers))
        return ModeContainer(
            mode_containers=new_containers,
            beta=self.beta,
            level=self.level
            )
    
    def remove_index(self,i):
        index_for_removal=onp.argsort([int(k!=i) for k in range(self.modes_number)])
        return self.remove_least_activated(index_for_removal)
    
    def get_K_except_index(self,index):
        matrices=onp.array(list(map(lambda container:container.get_K_except_index(index),self.mode_containers[:self.level])))
        return onp.sum(self.beta[:self.level,onp.newaxis,onp.newaxis]*matrices,axis=0)
    
    def remove_above_level(self):
        self.mode_containers=self.mode_containers[:self.level]
        self.beta=self.beta[:self.level]
    
    def __repr__(self) -> str:
        return '\n'.join([f'level {level+1}: \n'+container.__repr__() for level,container in enumerate(self.mode_containers[:self.level])])



class SingleModeContainer():
    def __init__(self,modes,Kbi_varphi_varphi):
        self.modes=modes
        self.Kbi_varphi_varphi=Kbi_varphi_varphi
    
    def compute_linear_mode_container(modes,X,constant_mode=None,beta1=None):
        Kbi_varphi_varphi=onp.expand_dims(X,-1)*onp.expand_dims(X,1)
        if constant_mode is None:
            return SingleModeContainer(modes,onp.array(Kbi_varphi_varphi))
        else:
            new_modes=onp.array(modes+[constant_mode])
            new_K=onp.concatenate([Kbi_varphi_varphi,onp.ones((1,X.shape[1],X.shape[1]))/beta1])

            return SingleModeContainer(new_modes,new_K)
    
    def compute_mode_container(modes,Kn,X,constant_mode=None,beta1=None):
        func_iterator=lambda mode:Kn(lambda x, y: mode.kappa(x, y), X.T)
        Kbi_varphi_varphi=list(map(func_iterator,modes))
        if constant_mode is None:
            return SingleModeContainer(modes,onp.array(Kbi_varphi_varphi))
        else:
            new_modes=onp.array(modes+[constant_mode])
            new_K=onp.array(Kbi_varphi_varphi+[onp.ones_like(Kbi_varphi_varphi[0])/beta1])
            return SingleModeContainer(new_modes,new_K)

    @property
    def K_mat(self):
        try:
            return self._kmat
        except:

            self._kmat=onp.sum(self.Kbi_varphi_varphi,axis=0)
            return self._kmat
    
    @K_mat.deleter
    def K_mat(self):
        try:
            del self._kmat
        except:
            pass
    
    def change_of_basis(self,alphas):
        print('changing basis')
        del self.K_mat
        self.Kbi_varphi_varphi=onp.matmul(alphas,onp.matmul(self.Kbi_varphi_varphi,alphas.T))

    
    
    def remove_least_activated(self,argsorted_activations):
        new_modes=self.modes[argsorted_activations][1:]
        new_K=self.Kbi_varphi_varphi[argsorted_activations][1:]
        return SingleModeContainer(new_modes,new_K)
    
    def get_K_except_index(self,index):
        masked=onp.ma.array(self.Kbi_varphi_varphi, mask=False)
        masked.mask[index]=True
        return onp.sum(masked,axis=0)
    

    def __repr__(self) -> str:
        return self.modes.__repr__()
    
class PairwiseModeContainer():
    def __init__(self,modes,Kbi_varphi_varphi):
        self.modes=modes #2D array
        self.Kbi_varphi_varphi=Kbi_varphi_varphi
    
    def compute_mode_container(modes,Kn,X,constant_mode=None):
        d,N=tuple(X.shape)
        func_iterator=lambda mode:Kn(lambda x, y: mode.kappa(x, y), X.T)
        Kbi_varphi_varphi = onp.zeros((d,d,N,N))
        indices = onp.tril_indices(d)
        Kbi_varphi_varphi[indices] = onp.array(list(map(func_iterator,modes)))
        Kbi_varphi_varphi=(Kbi_varphi_varphi+onp.transpose(Kbi_varphi_varphi,axes=(1,0,2,3)))/2
        modes_mat=onp.empty((d,d),dtype=object)
        modes_mat[indices]=modes
        for k in range(d):
            for j in range(k,d):
                modes_mat[k,j]=modes_mat[j,k]
        if constant_mode is None:
            return PairwiseModeContainer(modes_mat,Kbi_varphi_varphi)
        else:
            Kbi_varphi_varphi=onp.concatenate([Kbi_varphi_varphi,onp.zeros((1,d,N,N))],axis=0)
            Kbi_varphi_varphi=onp.concatenate([Kbi_varphi_varphi,onp.zeros((d+1,1,N,N))],axis=1)
            modes_mat=onp.concatenate([modes_mat,onp.array([[constant_mode]*d])],axis=0)
            modes_mat=onp.concatenate([modes_mat,onp.array([[constant_mode]]*(d+1))],axis=1)
            return PairwiseModeContainer(modes_mat,Kbi_varphi_varphi)
    
    def compute_quadratic_mode_container(modes,X,constant_mode=None):
        d,N=tuple(X.shape)
        Xnn=onp.expand_dims(X,-1)*onp.expand_dims(X,1)
        Kbi_varphi_varphi=onp.expand_dims(Xnn,0)*onp.expand_dims(Xnn,1)
        indices = onp.tril_indices(d)
        modes_mat=onp.empty((d,d),dtype=object)
        modes_mat[indices]=modes
        for k in range(d):
            for j in range(k,d):
                modes_mat[k,j]=modes_mat[j,k]
        if constant_mode is None:
            return PairwiseModeContainer(modes_mat,Kbi_varphi_varphi)
        else:
            Kbi_varphi_varphi=onp.concatenate([Kbi_varphi_varphi,onp.zeros((1,d,N,N))],axis=0)
            Kbi_varphi_varphi=onp.concatenate([Kbi_varphi_varphi,onp.zeros((d+1,1,N,N))],axis=1)
            modes_mat=onp.concatenate([modes_mat,onp.array([[constant_mode]*d])],axis=0)
            modes_mat=onp.concatenate([modes_mat,onp.array([[constant_mode]]*(d+1))],axis=1)
            return PairwiseModeContainer(modes_mat,Kbi_varphi_varphi)

    
    @property
    def K_mat(self):
        try:
            return self._kmat
        except:
            self._kmat=onp.sum(self.Kbi_varphi_varphi,axis=(0,1))
            return self._kmat
    
    @K_mat.deleter
    def K_mat(self):
        try:
            del self._kmat
        except:
            pass
    
    def change_of_basis(self,alphas):
        print('changing basis')
        del self.K_mat
        self.Kbi_varphi_varphi=onp.matmul(alphas,onp.matmul(self.Kbi_varphi_varphi,alphas.T))

    
    def remove_least_activated(self,argsorted_activations):
        row_col_indexes=onp.ix_(argsorted_activations, argsorted_activations)
        new_modes=self.modes[row_col_indexes][1:,1:]
        new_K=self.Kbi_varphi_varphi[row_col_indexes][1:,1:]
        return PairwiseModeContainer(new_modes,new_K)
    
    def get_K_except_index(self,index):
        masked=onp.ma.array(self.Kbi_varphi_varphi, mask=False)
        masked.mask[index]=True
        masked.mask[:,index]=True
        return onp.sum(masked,axis=(0,1))
    
    def __repr__(self) -> str:
        return self.modes.__repr__()

class CombinatorialModeContainer():
    def __init__(self,modes,Kbi_varphi_varphi,alphas=None):
        self.modes=modes
        self.Kbi_varphi_varphi=Kbi_varphi_varphi
        if alphas is None:
            self.alphas=np.eye(Kbi_varphi_varphi.shape[-1])
        else:
            self.alphas =alphas
    
    def compute_mode_container(modes,Kn,X,constant_mode=None):
        func_iterator=lambda mode:Kn(lambda x, y: mode.kappa(x, y), X.T)
        Kbi_varphi_varphi=list(map(func_iterator,modes))
        if constant_mode is None:
            return CombinatorialModeContainer(modes,Kbi_varphi_varphi)
        else:
            return CombinatorialModeContainer(onp.concatenate([modes,onp.array([constant_mode])]),onp.array(Kbi_varphi_varphi+[onp.zeros_like(Kbi_varphi_varphi[0])]))
    
    def compute_gaussian_mode_container(modes,l,X,constant_mode=None):
        diff_X=onp.tile(onp.expand_dims(X,-1),(1,1,X.shape[1]))-onp.tile(onp.expand_dims(X,1),(1,X.shape[1],1))
        
        Kbi_varphi_varphi=onp.exp(-(diff_X/l)**2)
        if constant_mode is None:
            return CombinatorialModeContainer(modes,Kbi_varphi_varphi)
        else:
            return CombinatorialModeContainer(onp.concatenate([modes,onp.array([constant_mode])]),onp.concatenate([Kbi_varphi_varphi,onp.zeros((1,X.shape[1],X.shape[1]))]))
    
    def change_of_basis(self,alphas):
        del self.K_mat
        self.alphas=alphas
        print('storing alpha of shape '+str(alphas.shape))



    @property
    def K_mat(self):
        try:
            return self._kmat
        except:
            self._kmat=onp.prod(self.Kbi_varphi_varphi+onp.ones_like(self.Kbi_varphi_varphi),axis=(0))
            self._kmat=onp.matmul(self.alphas,onp.matmul(self._kmat,self.alphas.T))
            print(self._kmat.shape)
            print(self.alphas.shape)
            return self._kmat
    
    @K_mat.deleter
    def K_mat(self):
        try:
            del self._kmat
        except:
            pass
    
    def remove_least_activated(self,argsorted_activations):
        new_modes=self.modes[argsorted_activations][1:]
        new_K=self.Kbi_varphi_varphi[argsorted_activations][1:]
        return CombinatorialModeContainer(new_modes,new_K,self.alphas)
    
    def get_K_except_index(self,index):
        masked=onp.ma.array(self.Kbi_varphi_varphi+onp.ones_like(self.Kbi_varphi_varphi), mask=False)
        masked.mask[index]=True
        mat= onp.array(onp.prod(masked,axis=(0)))
        return onp.matmul(self.alphas,onp.matmul(mat,self.alphas.T))
    
    def __repr__(self) -> str:
        return self.modes.__repr__()
        



