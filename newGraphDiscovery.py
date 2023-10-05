import numpy as onp
import networkx as nx
from tqdm import tqdm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from functools import partial
import scipy.linalg


class GraphDiscoveryNew:
    def __init__(self, X, beta, names, possible_edges=None, l=1, verbose=True) -> None:
        self.X = X
        self.print_func = print if verbose else lambda *a, **k: None

        self.print_func("Computing kernel matrix")
        constant_mat = onp.ones((X.shape[1], X.shape[1]))
        linear_mat = onp.expand_dims(X, -1) * onp.expand_dims(X, 1)
        quadratic_mat = onp.expand_dims(linear_mat, 0) * onp.expand_dims(linear_mat, 1)
        # take into account off diagonal elements are counted twice
        quadratic_mat = (
            quadratic_mat / 2 * (1 + onp.eye(quadratic_mat.shape[0]))[:, :, None, None]
        )
        diff_X = onp.tile(onp.expand_dims(X, -1), (1, 1, X.shape[1])) - onp.tile(
            onp.expand_dims(X, 1), (1, X.shape[1], 1)
        )
        gaussian_mat = onp.exp(-((diff_X / l) ** 2) / 2)

        self.beta = beta
        level = onp.ones_like(beta)
        self.names = names
        self.name_to_index = {name: index for index, name in enumerate(names)}

        self.modes = ModeContainer(
            constant_mat,
            linear_mat,
            quadratic_mat,
            gaussian_mat,
            names,
            beta,
            level,
        )

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
        return onp.sort(B_samples)[int(0.05*N)]


    def find_ancestors(
        self,
        name,
        gamma="auto",
        acceptation_logic="default",
        gamma_min=1e-9,
    ):
        index = self.name_to_index[name]
        ga = self.X[index]
        active_modes = self.modes.delete_node_by_name(name)
        if self.possible_edges is not None:
            for possible_name in active_modes.names:
                if possible_name not in self.possible_edges.predecessors(name):
                    active_modes = active_modes.delete_node_by_name(possible_name)


        if acceptation_logic == "default":
            acceptation_logic = GraphDiscoveryNew.acceptation_logic(
                cutoff=0.9, use_Z=True
            )
        elif acceptation_logic == "manual":
            acceptation_logic = GraphDiscoveryNew.manual_acceptation()
        else:
            assert callable(acceptation_logic)

        for i, which in enumerate(["linear", "quadratic", "gaussian"]):
            K = active_modes.get_K(which)
            if gamma == "auto":
                gamma_used = GraphDiscoveryNew.find_gamma(K=K, which=which, Y=ga,tol=1e-10)
                gamma_used = max(gamma_used, gamma_min)
            else:
                gamma_used = gamma

            K += gamma_used * onp.eye(K.shape[0])
            c, low = scipy.linalg.cho_factor(K)
            yb, noise = GraphDiscoveryNew.solve_variationnal(
                ga, gamma=gamma_used, cho_factor=(c, low)
            )
            Z = GraphDiscoveryNew.Z_test(gamma_used,cho_factor=(c, low))
            self.print_func(
                f"{which} kernel (using gamma={gamma_used:.2e})\n n/(n+s)={noise:.2f}, Z={Z:.2f}"
            )
            accept = acceptation_logic(noise, Z, which)
            self.print_func(
                f'decision : {"refused"*int(not(accept))+"accepted"*int(accept)}'
            )
            if accept:
                pass
                #break

        if not accept:
            self.print_func(f"{name} has no ancestors (n/(s+n)={noise:.2f})\n")
            return
        self.print_func(
            f"{name} has ancestors with {which} kernel (n/(s+n)={noise:.2f})"
        )
        active_modes.set_level(which)
        _, ancestor_modes = GraphDiscoveryNew.recursive_ancestor_finder(
            ga,
            active_modes,
            yb,
            gamma_used,
            acceptation_logic=partial(acceptation_logic, which=which),
            printer=self.print_func,
        )
        self.print_func("ancestors after pruning: ", ancestor_modes, "\n")
        for ancestor_name in ancestor_modes.names:
            self.G.add_edge(ancestor_name, name, type=which)

    def acceptation_logic(cutoff, use_Z):
        def func(noise, Z, which):
            if noise < cutoff:
                return True
            if use_Z and which == "gaussian":
                return abs(Z) > 1.96 and noise < cutoff
            return False

        return func

    def manual_acceptation():
        def func(noise, Z, which):
            decision = None
            print(f"{which} kernel\n n/(n+s)={noise:.2f}, Z={Z:.2f}")
            while decision is None:
                val = input("Decision ? Y:signal , N:Noise, STOP:exit algorithm")
                if val == "Y":
                    return True
                elif val == "N":
                    return False
                elif val == "STOP":
                    raise Exception("Algorithm stopped")
                else:
                    continue

        return func

    def recursive_ancestor_finder(
        ga, active_modes, yb, gamma, acceptation_logic, printer
    ):
        energy = -onp.dot(ga, yb)
        activations = {
            name: onp.dot(yb, active_modes.get_K_of_name(name) @ yb) / energy
            for name in active_modes.active_names
        }
        minimum_activation_name=min(activations, key=activations.get)

        new_modes = active_modes.delete_node_by_name(minimum_activation_name)
        K = new_modes.get_K()
        K += gamma * onp.eye(K.shape[0])
        c, low = scipy.linalg.cho_factor(K)
        new_yb, new_noise = GraphDiscoveryNew.solve_variationnal(
            ga, gamma=gamma, cho_factor=(c, low)
        )
        new_Z = GraphDiscoveryNew.Z_test(gamma=gamma,cho_factor=(c, low))
        accept = acceptation_logic(noise=new_noise, Z=new_Z)
        printer(f"ancestors : {new_modes}\n n/(n+s)={new_noise:.2f}, Z={new_Z:.2f}")
        printer(f'decision : {"refused"*int(not(accept))+"accepted"*int(accept)}')

        if accept:
            if new_modes.node_number == 1:
                return new_yb, new_modes
            else:
                return GraphDiscoveryNew.recursive_ancestor_finder(
                    ga, new_modes, new_yb, gamma, acceptation_logic, printer
                )
        else:
            return yb, active_modes

    def find_gamma(K, which, Y, tol=1e-10):
        
        eigenvalues,eigenvectors=onp.linalg.eigh(K)
        if which != "gaussian":
            selected_eigenvalues = eigenvalues < tol
            residuals=(eigenvectors[:,selected_eigenvalues]@(eigenvectors[:,selected_eigenvalues].T))@Y
            gamma=onp.linalg.norm(residuals)
            print(f'gamma through residuals: {gamma}')
            print(f'mean of eigenvalues: {onp.mean(eigenvalues)}')
            print(f'geo-mean of eigenvalues: {onp.exp(onp.mean(onp.log(onp.maximum(1e-15,eigenvalues))))}')
            return gamma

        #eigenvalues = eigenvalues[eigenvalues > tol]

        def var(gamma_log):
            return -onp.var(1 / (1 + eigenvalues * onp.exp(-gamma_log)))

        res = minimize(
            var,
            onp.array([-onp.log(onp.mean(eigenvalues))]),
            method="nelder-mead",
            options={"xatol": 1e-8, "disp": False},
        )
        plt.figure()
        plt.plot([k for k in range(len(eigenvalues))],eigenvalues,'o')
        plt.yscale('log')
        plt.show()
        plt.figure()
        plt.plot(onp.logspace(-9,3,100),[var(onp.log(gamma)) for gamma in onp.logspace(-9,3,100)])
        plt.xscale('log')
        plt.show()
        print('gamma through variance: ',onp.exp(res.x[0]))
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


class ModeContainer:
    def __init__(
        self,
        constant_mat,
        linear_mat,
        quadratic_mat,
        gaussian_mat,
        names,
        beta,
        level,
        used=None
    ) -> None:
        self.constant_mat = constant_mat
        self.linear_mat = linear_mat
        self.quadratic_mat = quadratic_mat
        self.gaussian_mat = gaussian_mat
        self.names = names
        self.beta = beta
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
            self.constant_mat,
            self.linear_mat,
            self.quadratic_mat,
            self.gaussian_mat,
            self.names,
            self.beta,
            self.level,
            used=new_used
        )

    def get_level(self, chosen_level):
        if chosen_level is None:
            return self.level
        if chosen_level == "linear":
            return onp.array([1] + [0] * (self.beta.shape[0] - 1))
        if chosen_level == "quadratic":
            return onp.array([1, 1] + [0] * (self.beta.shape[0] - 2))
        if chosen_level == "gaussian":
            return onp.ones_like(self.beta)

        return onp.array(
            [int(i <= int(chosen_level)) for i in range(self.beta.shape[0])]
        )

    def set_level(self, chosen_level):
        assert chosen_level is not None
        self.level = self.get_level(chosen_level)

    def get_K(self, chosen_level=None):
        coeff = self.beta * self.get_level(chosen_level)
        K = (
            self.constant_mat
            + coeff[0] * onp.sum(self.linear_mat, axis=0,where=self.used[:,None,None])
            + coeff[1] * onp.sum(onp.sum(self.quadratic_mat, axis=0,where=self.used[:,None,None,None]), axis=0,where=self.used[:,None,None])
            + coeff[2]
            * onp.prod(self.gaussian_mat + onp.ones_like(self.gaussian_mat), axis=0,where=self.used[:,None,None])
        )
        return K
    
    def get_K_of_name(self, name):
        return self.get_K_of_index(self.get_index_of_name(name))

    def get_K_of_index(self, index):
        assert self.used[index]
        coeff = self.beta * self.level
        res = onp.zeros_like(self.linear_mat[0])
        res += coeff[0] * self.linear_mat[index]
        res += coeff[1] * (
            2 * onp.sum(self.quadratic_mat[index], axis=0,where=self.used[:,None,None])
            - self.quadratic_mat[index, index]
        )
        used_for_prod = self.used.copy()
        used_for_prod[index]=False
        res += (
            coeff[2]
            * (
                self.gaussian_mat[index]
                * onp.prod(
                    self.gaussian_mat + onp.ones_like(self.gaussian_mat), axis=0,where=used_for_prod[:,None,None]
                )
            )
        )
        return res  # +self.constant_mat# no constant

    def get_K_without_index(self, index):
        assert self.used[index]
        return self.delete_node(index).get_K()
    
    def get_K_without_name(self, name):
        return self.get_K_without_index(self.get_index_of_name(name))

    def __repr__(self) -> str:
        return list(self.active_names).__repr__()
