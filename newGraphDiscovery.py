import numpy as onp
import networkx as nx
from tqdm import tqdm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from functools import partial
import scipy.linalg

from Modes import ModeContainer
from decision import KernelChooser, ModeChooser,EarlyStopping


class GraphDiscoveryNew:
    def __init__(self, X, names, mode_container,possible_edges=None, verbose=True) -> None:
        self.X = X
        self.print_func = print if verbose else lambda *a, **k: None
        self.names = names
        self.name_to_index = {name: index for index, name in enumerate(names)}
        self.modes = mode_container
        self.possible_edges = possible_edges
        self.G = nx.DiGraph()
        self.G.add_nodes_from(names)



    def solve_variationnal(ga, gamma, cho_factor):
        yb = -scipy.linalg.cho_solve(cho_factor,ga)
        noise = -gamma * onp.dot(yb, yb) / onp.dot(ga, yb)
        return yb, noise

    def Z_test(gamma,cho_factor):
        """computes Z-test using 1000 samples"""
        N=1000
        samples=gamma*onp.random.normal(size=(N,cho_factor[0].shape[0]))
        B_samples=onp.array([GraphDiscoveryNew.solve_variationnal(sample,gamma, cho_factor)[1] for sample in samples])
        return onp.sort(B_samples)[int(0.05*N)],onp.sort(B_samples)[int(0.95*N)]


    def find_ancestors(
        self,
        name,
        gamma="auto",
        gamma_min=1e-9,
        kernel_chooser={}, #dict of parameters for kernel chooser
        mode_chooser={}, #dict of parameters for mode chooser
        early_stopping={}, #dict of parameters for early stopping
        **kwargs
    ):
        index = self.name_to_index[name]
        ga = self.X[index]
        active_modes = self.modes.delete_node_by_name(name)
        if self.possible_edges is not None:
            for possible_name in active_modes.names:
                if possible_name not in self.possible_edges.predecessors(name) and possible_name!=name:
                    active_modes = active_modes.delete_node_by_name(possible_name)


        choose_kernel=KernelChooser(**kernel_chooser)
        choose_mode=ModeChooser(**mode_chooser)
        early_stopping=EarlyStopping(**early_stopping)

        kernel_performance={}
        for which in active_modes.matrices_names:
            K = active_modes.get_K(which)
            if gamma == "auto":
                gamma_used = GraphDiscoveryNew.find_gamma(K=K, interpolatory=active_modes.is_interpolatory(which) , Y=ga,tol=1e-10)
                if gamma_used < gamma_min:
                    self.print_func(
                        f"""gamma too small for set tolerance({gamma_used:.2e}), using {gamma_min:.2e} instead\nThis can either mean that the noise is very low or there is an issue in the automatic determination of gamma. To change the tolerance, change parameter gamma_min"""
                    )
                    gamma_used = gamma_min
            else:
                gamma_used = gamma
            K += gamma_used * onp.eye(K.shape[0])
            c, low = scipy.linalg.cho_factor(K)
            yb, noise = GraphDiscoveryNew.solve_variationnal(
                ga, gamma=gamma_used, cho_factor=(c, low)
            )
            Z_low,Z_high = GraphDiscoveryNew.Z_test(gamma_used,cho_factor=(c, low))
            self.print_func(
                f"{which} kernel (using gamma={gamma_used:.2e})\n n/(n+s)={noise:.2f}, Z={Z_low:.2f}"
            )
            kernel_performance[which]={
                'noise':noise,
                'Z':(Z_low,Z_high),
                'yb':yb,
                'gamma':gamma_used,
            }
        
        which=choose_kernel(kernel_performance)

        if which is None:
            self.print_func(f"{name} has no ancestors\n")
            return
        self.print_func(
            f"{name} has ancestors with {which} kernel (n/(s+n)={kernel_performance[which]['noise']:.2f})"
        )
        active_modes.set_level(which)

        list_of_modes, noises, Zs=GraphDiscoveryNew.iterative_ancestor_finder(
            ga,
            active_modes,
            printer=self.print_func,
            early_stopping=early_stopping,
            auto_gamma=gamma,
            gamma_min=gamma_min,
            **kernel_performance[which]
        )
        ancestor_modes=choose_mode(list_of_modes,noises,Zs)
        #plot evolution of noise and Z, and in second plot on the side evolution of Z_{k+1}-Z_k
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
        axes[0].plot(list(range(1,1+len(noises)))[::-1],noises,label='noise')
        axes[0].plot(list(range(1,1+len(noises)))[::-1],[z[0] for z in Zs],label='5% quantile of random noise')
        axes[0].plot(list(range(1,1+len(noises)))[::-1],[z[1] for z in Zs],label='95% quantile of random noise')
        #color in between the two lines above
        axes[0].fill_between(list(range(1,1+len(noises)))[::-1],[z[0] for z in Zs],[z[1] for z in Zs],alpha=0.2)

        axes[0].axvline(x=ancestor_modes.node_number,linestyle='--',color='k',label=f'chosen number of ancestors={ancestor_modes.node_number}')
        axes[0].set_xlabel('number of ancestors')
        axes[0].set_ylabel('noise')
        axes[0].invert_xaxis()
        axes[0].set_xticks(onp.linspace(len(noises),1,6,dtype=onp.int32,endpoint=True))
        axes[0].legend()
        axes[1].plot(list(range(1,1+len(noises)))[::-1],[noises[i+1]-noises[i] for i in range(len(noises)-1)]+[1-noises[-1]],label='noise increment')
        axes[1].axvline(x=ancestor_modes.node_number,linestyle='--',color='k',label=f'chosen number of ancestors={ancestor_modes.node_number}')
        axes[1].legend()
        axes[1].set_xlabel('number of ancestors')
        axes[1].set_ylabel('noise increment')
        axes[1].invert_xaxis()
        axes[1].set_xticks(onp.linspace(len(noises),1,6,dtype=onp.int32,endpoint=True))
        fig.tight_layout()
        plt.show()

        signal=1-noises[-ancestor_modes.node_number]

        self.print_func("ancestors after pruning: ", ancestor_modes, "\n")
        for used,ancestor_name in zip(ancestor_modes.used,ancestor_modes.names):
            if used:
                self.G.add_edge(ancestor_name, name, type=which,signal=signal)
        


    
    def iterative_ancestor_finder(
        ga, modes, gamma,yb, noise,Z, printer,early_stopping,auto_gamma,gamma_min
    ):
        noises=[noise]
        Zs=[Z]
        list_of_modes=[modes]
        active_modes=modes
        active_yb=yb
        while active_modes.node_number>1 and not early_stopping(list_of_modes,noises,Zs):
            energy = -onp.dot(ga, active_yb)
            activations = {
                name: onp.dot(active_yb, active_modes.get_K_of_name(name) @ active_yb) / energy
                for name in active_modes.active_names
            }
            minimum_activation_name=min(activations, key=activations.get)
            active_modes = active_modes.delete_node_by_name(minimum_activation_name)
            list_of_modes.append(active_modes)
            K = active_modes.get_K()
            if auto_gamma and active_modes.is_interpolatory():
                gamma = GraphDiscoveryNew.find_gamma(K=K, interpolatory=active_modes.is_interpolatory(), Y=ga,tol=1e-10)
                gamma=max(gamma,gamma_min)
            K += gamma * onp.eye(K.shape[0])
            c, low = scipy.linalg.cho_factor(K)
            active_yb, noise = GraphDiscoveryNew.solve_variationnal(
                ga, gamma=gamma, cho_factor=(c, low)
            )
            Z_low,Z_high = GraphDiscoveryNew.Z_test(gamma=gamma,cho_factor=(c, low))
            noises.append(noise)
            Zs.append((Z_low,Z_high))
            printer(f"ancestors : {active_modes}\n n/(n+s)={noise:.2f}, Z={Z_low:.2f}")
        return list_of_modes, noises, Zs


    def find_gamma(K, interpolatory, Y ,tol=1e-10):
        
        eigenvalues,eigenvectors=onp.linalg.eigh(K)
        #plt.figure()
        #plt.plot(eigenvalues,[k/eigenvalues.shape[0] for k in range(eigenvalues.shape[0])])
        #plt.xscale('log')
        #plt.show()
        if not interpolatory:
            
            selected_eigenvalues = eigenvalues < tol
            
            residuals=(eigenvectors[:,selected_eigenvalues]@(eigenvectors[:,selected_eigenvalues].T))@Y
            gamma=onp.linalg.norm(residuals)
            #print(f'gamma through residuals: {gamma}')
            #print(f'mean of eigenvalues: {onp.mean(eigenvalues)}')
            #print(f'geo-mean of eigenvalues: {onp.exp(onp.mean(onp.log(onp.maximum(1e-15,eigenvalues))))}')
            return gamma
        print(f'what about median ? {onp.median(eigenvalues)}')
        #eigenvalues = eigenvalues[eigenvalues > tol]

        def var(gamma_log):
            return -onp.var(1 / (1 + eigenvalues * onp.exp(-gamma_log)))
        """test_gammas=onp.logspace(gamma_log_range[0],gamma_log_range[1],5000)
        vars=[var(onp.log(gamma)) for gamma in test_gammas]
        gamma=test_gammas[onp.argmin(vars)]
        return gamma"""
        res = minimize(
            var,
            onp.array([onp.log(onp.mean(eigenvalues))]),
            method="nelder-mead",
            options={"xatol": 1e-8, "disp": False},
        )
        '''plt.figure()
        plt.plot([k for k in range(len(eigenvalues))],eigenvalues,'o')
        plt.yscale('log')
        plt.show()
        plt.figure()
        plt.plot(onp.logspace(-9,3,100),[var(onp.log(gamma)) for gamma in onp.logspace(-9,3,100)])
        plt.xscale('log')
        plt.show()
        print('gamma through variance: ',onp.exp(res.x[0]))'''
        return onp.exp(res.x[0])

    def plot_graph(self, type_label=True):
        pos = nx.kamada_kawai_layout(self.G, self.G.nodes())
        nx.draw(
            self.G, with_labels=True, pos=pos, node_size=600, font_size=8, alpha=0.6
        )
        if type_label:
            nx.draw_networkx_edge_labels(
                self.G, pos, edge_labels=nx.get_edge_attributes(self.G, "type")
            )

