<<<<<<< Updated upstream
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

class Mode(object):
    def __init__(self, nodes, names, beta, type):
        self.nodes = nodes
        self.beta = beta
        self.names = names
        self.type = type

    def coeff(self):
        return np.sqrt(self.beta)

class ConstantMode(Mode):
    """
    Represent the constant kernel, K(x, y) = beta
    """
    def __init__(self, nodes=np.array([0]), names=np.array([]), beta=1, type="Constant"):
        super().__init__(nodes, names, beta, type)

    def kappa(self, x, y):
        return self.beta


    def feature(self, x):
        return np.sqrt(self.beta)

    def to_string(self):
        return ""


class LinearMode(Mode):
    """
    Represent the linear kernel, K(x, y)=beta * x[i] * y[i]
    """
    def __init__(self, nodes, names, beta, type="Linear"):
        super().__init__(nodes, names, beta, type)

    def kappa(self, x, y):
        return self.beta * x[self.nodes[0]] * y[self.nodes[0]]

    def feature(self, x):
        return np.sqrt(self.beta) * x[self.nodes[0]]

    def to_string(self):
        return self.names[0]


class QuadraticMode(Mode):
    """
    Represent the quadratic kernel, K(x, y) = beta * x[i] * x[j] * y[i] * y[j]
    """
    def __init__(self, nodes, names, beta, type="Quadratic"):
        super().__init__(nodes, names, beta, type)

    def kappa(self, x, y):
        return self.beta * x[self.nodes[0]] * x[self.nodes[1]] * y[self.nodes[0]] * y[self.nodes[1]]

    def feature(self, x):
        return np.sqrt(self.beta) * x[self.nodes[0]] * x[self.nodes[1]]

    def to_string(self):
        return self.names[0] + self.names[1]



class KernelMode(Mode):
    """
    Represent the product of kernel, K(x, y) = beta * \prod{K_i(x_i,y_i)}
    """
    def __init__(self, nodes, names, beta, ks, type="Kernel"):
        super().__init__(nodes, names, beta, type)
        self.ks = ks

    def kappa(self, x, y):
        val = [self.ks[i](x[self.nodes[i]], y[self.nodes[i]]) for i in range(len(self.ks))]
        return self.beta * reduce(np.multiply, val)

    def feature(self, x):
        return None

    def to_string(self, point):
        val = ["{},".format(self.names[i]) for i in range(len(self.names))]
        prefix = "K("
        postfix = point + ")"
        for n in val: prefix = prefix + n
        return prefix + postfix

=======
import numpy as onp




class ModeContainer:
    '''
        This class is used to store the modes of Graph Discovery
    '''
    def __init__(
        self,
        matrices,
        matrices_types,
        matrices_names,
        variables_names,
        beta,
        level=None,
        used=None
    ) -> None:
        self.constant_mat = onp.ones(matrices[0].shape[-2:])
        self.matrices=matrices
        self.matrices_types=matrices_types
        self.names = variables_names
        self.beta = beta
        self.level = level
        self.matrices_names = matrices_names
        if level is None:
            self.level = onp.ones_like(matrices_names)
        else:
            self.level = level
        if used is not None:
            self.used = used
        else:
            self.used = onp.array([True]*self.names.shape[0])

    @property
    def node_number(self):
        return onp.sum(self.used)
    
    @property
    def active_names(self):
        return self.names[self.used]
    
    def get_index_of_name(self, target_name):
        for i, name in enumerate(self.names):
            if name == target_name:
                if self.used[i]:
                    return i
                else:
                    break
        raise f"{target_name} is not in the modes' list of active namesnames"

    def delete_node_by_name(self, target_name):
        return self.delete_node(self.get_index_of_name(target_name))

    def delete_node(self, index):
        new_used=self.used.copy()
        assert new_used[index]
        new_used[index]=False
        return ModeContainer(
            self.matrices,
            self.matrices_types,
            self.matrices_names,
            self.names,
            self.beta,
            self.level,
            used=new_used
        )

    def get_level(self, chosen_level):
        if chosen_level is None:
            return self.level
        level=[]
        found=False
        for level_name in self.matrices_names:
            if not found:
                level.append(1)
            else:
                level.append(0)
            if level_name==chosen_level:
                found=True
        if not found:
            raise Exception(f"Level {chosen_level} is not in the list of levels {self.matrices_names}")
        return onp.array(level)


    def set_level(self, chosen_level):
        assert chosen_level is not None
        self.level = self.get_level(chosen_level)

    def sum_a_matrix(matrix,matrix_type,used):
        if matrix_type=='individual':
            return onp.sum(matrix,axis=0,where=used[:,None,None])
        if matrix_type=='pairwise':
            return onp.sum(onp.sum(matrix,axis=0,where=used[:,None,None,None]),axis=0,where=used[:,None,None])
        if matrix_type=='combinatorial':
            return onp.prod(matrix+onp.ones_like(matrix),axis=0,where=used[:,None,None])
        raise f"Unknown matrix type {matrix_type}"
    
    def sum_a_matrix_of_index(matrix,matrix_type,used,index):
        if matrix_type=='individual':
            return matrix[index]
        if matrix_type=='pairwise':
            return 2 * onp.sum(matrix[index], axis=0,where=used[:,None,None])- matrix[index, index]
        if matrix_type=='combinatorial':
            used_for_prod = used.copy()
            used_for_prod[index]=False
            return matrix[index]*onp.prod(matrix+onp.ones_like(matrix),axis=0,where=used_for_prod[:,None,None])
        raise f"Unknown matrix type {matrix_type}"

    def get_K(self, chosen_level=None):
        coeff = self.beta * self.get_level(chosen_level)
        K=onp.zeros_like(self.constant_mat)
        K+=self.constant_mat
        for i, matrix in enumerate(self.matrices):
            if coeff[i]!=0:
                K+=coeff[i]*ModeContainer.sum_a_matrix(matrix,self.matrices_types[i],self.used)
        return K
    
    def get_K_of_name(self, name):
        return self.get_K_of_index(self.get_index_of_name(name))

    def get_K_of_index(self, index):
        assert self.used[index]
        coeff = self.beta * self.level
        res = onp.zeros_like(self.constant_mat)
        for i, matrix in enumerate(self.matrices):
            res+=coeff[i]*ModeContainer.sum_a_matrix_of_index(matrix,self.matrices_types[i],self.used,index)
        return res

    def get_K_without_index(self, index):
        assert self.used[index]
        return self.delete_node(index).get_K()
    
    def get_K_without_name(self, name):
        return self.get_K_without_index(self.get_index_of_name(name))

    def __repr__(self) -> str:
        return list(self.active_names).__repr__()
    
    def make_container(X,variable_names,*args):
        assert X.shape[0]==len(variable_names)
        matrices=[]
        matrices_types=[]
        matrices_names=[]
        beta=[]
        for arg in args:
            assert arg['type'] in ['individual','combinatorial','pairwise']
            built=arg.get('built',False)
            if built:
                if arg['type'] in ['individual','combinatorial']:
                    assert arg['matrix'].shape==(len(variable_names),X.shape[1],X.shape[1])
                if arg['type']=='pairwise':
                    assert arg['matrix'].shape==(len(variable_names),len(variable_names),X.shape[1],X.shape[1])
                matrices.append(arg['matrix'])
                matrices_types.append(arg['type'])
                matrices_names.append(arg['name'])
                beta.append(arg['beta'])
            else:
                matrix=ModeContainer.build_matrix(X,**arg)
                matrices.append(matrix)
                matrices_types.append(arg['type'])
                matrices_names.append(arg['name'])
                beta.append(arg['beta'])
            if arg['type']=='pairwise':
                #in the inner workings of the code, off-diagonal matrices are counted twice
                matrices[-1]=matrices[-1] * ((1 + onp.eye(matrices[-1].shape[0]))[:, :, None, None]/2)
        return ModeContainer(
            matrices,
            matrices_types,
            matrices_names,
            variable_names,
            onp.array(beta)
        )
    

    def build_matrix(X,**kwargs):
        default=kwargs.get('default',False)
        if default:
            which=kwargs['name']
            assert which in ['linear','quadratic','gaussian'], f"Unknown default matrix {which}"
            if which=='linear':
                assert kwargs['type']=='individual', "Linear kernel is only available for individual matrices"
                return onp.expand_dims(X, -1) * onp.expand_dims(X, 1)
            if which=='quadratic':
                assert kwargs['type']=='pairwise', "Quadratic kernel is only available for pairwise matrices"
                linear_mat = onp.expand_dims(X, -1) * onp.expand_dims(X, 1)
                return onp.expand_dims(linear_mat, 0) * onp.expand_dims(linear_mat, 1)
            if which=='gaussian':
                assert kwargs['type']=='combinatorial', "Gaussian kernel is only available for combinatorial matrices"
                try:
                    l=kwargs['l']
                except:
                    raise Exception("You must specify the lengthscale of the gaussian kernel")
                assert len(X.shape)==2, "Gaussian kernel is only available for 1D data"
                diff_X = onp.tile(onp.expand_dims(X, -1), (1, 1, X.shape[1])) - onp.tile(
                    onp.expand_dims(X, 1), (1, X.shape[1], 1)
                )
                return onp.exp(-((diff_X / l) ** 2) / 2)
            return onp.ones((X.shape[0],X.shape[0]))
        scipy_kernel=kwargs.get('scipy_kernel',None)
        if scipy_kernel is not None:
            '''must behave like scikit-learn kernels'''
            res=[]
            if kwargs['type'] in ['individual','combinatorial']:
                for col in X:
                    if len(col.shape)>1:
                        res.append(scipy_kernel(col))
                    res.append(scipy_kernel(col.expand_dims(1)))
                return onp.stack(res,axis=0)
            if kwargs['type']=='pairwise':
                res=onp.zeros((X.shape[0],X.shape[0],X.shape[1],X.shape[1]))
                for i,col1 in enumerate(X):
                    for j,col2 in enumerate(X):
                        data=onp.stack([col1,col2],axis=1)
                        res[i,j,:,:]=scipy_kernel(data)
                return res
        
        raise Exception("You must either provide a default kernel or a scipy kernel")
            



>>>>>>> Stashed changes


