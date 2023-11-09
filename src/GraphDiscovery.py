import numpy as onp
import networkx as nx
from tqdm import tqdm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from functools import partial
import scipy.linalg

from Modes import ModeContainer
from decision import KernelChooser, ModeChooser, EarlyStopping
from plotting_help import partition_layout, plot_noise_evolution


class GraphDiscoveryNew:
    def __init__(
        self, X, names, mode_container, possible_edges=None, verbose=True
    ) -> None:
        self.X = X
        self.print_func = print if verbose else lambda *a, **k: None
        self.names = names
        self.name_to_index = {name: index for index, name in enumerate(names)}
        self.modes = mode_container
        self.possible_edges = possible_edges
        self.G = nx.DiGraph()
        self.G.add_nodes_from(names)

    def prepare_new_graph_with_clusters(self, clusters):
        new_graph = GraphDiscoveryNew(
            self.X, self.names, self.modes, self.possible_edges, verbose=False
        )
        new_graph.print_func = self.print_func
        new_graph.modes.assign_clusters(clusters)
        new_graph.G = self.G.copy()
        edges_to_remove = []
        flattened_clusters = [
            (i, item) for i, sublist in enumerate(clusters) for item in sublist
        ]
        for i, (cluster_index, node) in enumerate(flattened_clusters):
            other_nodes = list(
                set([node for j, node in flattened_clusters if j > i])
                - set(clusters[cluster_index])
            )
            for other_node in other_nodes:
                edges_to_remove.append((node, other_node))
        new_graph.G.remove_edges_from(edges_to_remove)
        return new_graph

    def solve_variationnal(ga, gamma, cho_factor):
        yb = -scipy.linalg.cho_solve(cho_factor, ga)
        noise = -gamma * onp.dot(yb, yb) / onp.dot(ga, yb)
        return yb, noise

    def Z_test(gamma, cho_factor):
        """computes Z-test using 1000 samples"""
        N = 1000
        samples = gamma * onp.random.normal(size=(N, cho_factor[0].shape[0]))
        B_samples = onp.array(
            [
                GraphDiscoveryNew.solve_variationnal(sample, gamma, cho_factor)[1]
                for sample in samples
            ]
        )
        return onp.sort(B_samples)[int(0.05 * N)], onp.sort(B_samples)[int(0.95 * N)]

    def find_ancestors(
        self,
        name,
        gamma="auto",
        gamma_min=1e-9,
        kernel_chooser={},  # dict of parameters for kernel chooser
        mode_chooser={},  # dict of parameters for mode chooser
        early_stopping={},  # dict of parameters for early stopping
        **kwargs,
    ):
        index = self.name_to_index[name]
        ga = self.X[index]
        active_modes = self.modes.delete_node_by_name(name)
        if self.possible_edges is not None:
            for possible_name in active_modes.names:
                if (
                    possible_name not in self.possible_edges.predecessors(name)
                    and possible_name != name
                ):
                    active_modes = active_modes.delete_node_by_name(possible_name)

        choose_kernel = KernelChooser(**kernel_chooser)
        choose_mode = ModeChooser(**mode_chooser)
        early_stopping = EarlyStopping(**early_stopping)

        kernel_performance = {}
        for which in active_modes.matrices_names:
            K = active_modes.get_K(which)
            if gamma == "auto":
                gamma_used = GraphDiscoveryNew.find_gamma(
                    K=K,
                    interpolatory=active_modes.is_interpolatory(which),
                    Y=ga,
                    tol=1e-10,
                    printer=self.print_func,
                    gamma_min=gamma_min,
                )
            else:
                gamma_used = gamma
            K += gamma_used * onp.eye(K.shape[0])
            c, low = scipy.linalg.cho_factor(K)
            yb, noise = GraphDiscoveryNew.solve_variationnal(
                ga, gamma=gamma_used, cho_factor=(c, low)
            )
            Z_low, Z_high = GraphDiscoveryNew.Z_test(gamma_used, cho_factor=(c, low))
            self.print_func(
                f"{which} kernel (using gamma={gamma_used:.2e})\n n/(n+s)={noise:.2f}, Z={Z_low:.2f}"
            )
            kernel_performance[which] = {
                "noise": noise,
                "Z": (Z_low, Z_high),
                "yb": yb,
                "gamma": gamma_used,
            }

        which = choose_kernel(kernel_performance)

        if which is None:
            self.print_func(f"{name} has no ancestors\n")
            return
        self.print_func(
            f"{name} has ancestors with {which} kernel (n/(s+n)={kernel_performance[which]['noise']:.2f})"
        )
        active_modes.set_level(which)

        list_of_modes, noises, Zs = GraphDiscoveryNew.iterative_ancestor_finder(
            ga,
            active_modes,
            printer=self.print_func,
            early_stopping=early_stopping,
            auto_gamma=gamma,
            gamma_min=gamma_min,
            **kernel_performance[which],
        )
        ancestor_modes = choose_mode(list_of_modes, noises, Zs)
        # plot evolution of noise and Z, and in second plot on the side evolution of Z_{k+1}-Z_k
        ancestor_number = [mode.node_number for mode in list_of_modes]
        fig, axes = plot_noise_evolution(
            ancestor_number,
            noises,
            Zs,
            ancestor_modes,
        )
        plt.show()

        signal = 1 - noises[-ancestor_modes.node_number]

        self.print_func("ancestors after pruning: ", ancestor_modes, "\n")
        for ancestor_name, used in ancestor_modes.used.items():
            if used:
                self.G.add_edge(ancestor_name, name, type=which, signal=signal)

    def iterative_ancestor_finder(
        ga, modes, gamma, yb, noise, Z, printer, early_stopping, auto_gamma, gamma_min
    ):
        noises = [noise]
        Zs = [Z]
        list_of_modes = [modes]
        active_modes = modes
        active_yb = yb
        while active_modes.node_number > 1 and not early_stopping(
            list_of_modes, noises, Zs
        ):
            energy = -onp.dot(ga, active_yb)
            activations = [
                (
                    name,
                    onp.dot(active_yb, active_modes.get_K_of_cluster(name) @ active_yb)
                    / energy,
                )
                for name in active_modes.active_clusters
            ]
            minimum_activation_cluster = min(activations, key=lambda x: x[1])[0]
            active_modes = active_modes.delete_cluster(minimum_activation_cluster)
            list_of_modes.append(active_modes)
            K = active_modes.get_K()
            if auto_gamma and active_modes.is_interpolatory():
                gamma = GraphDiscoveryNew.find_gamma(
                    K=K,
                    interpolatory=active_modes.is_interpolatory(),
                    Y=ga,
                    tol=1e-10,
                    printer=printer,
                    gamma_min=gamma_min,
                )
            K += gamma * onp.eye(K.shape[0])
            c, low = scipy.linalg.cho_factor(K)
            active_yb, noise = GraphDiscoveryNew.solve_variationnal(
                ga, gamma=gamma, cho_factor=(c, low)
            )
            Z_low, Z_high = GraphDiscoveryNew.Z_test(gamma=gamma, cho_factor=(c, low))
            noises.append(noise)
            Zs.append((Z_low, Z_high))
            printer(f"ancestors : {active_modes}\n n/(n+s)={noise:.2f}, Z={Z_low:.2f}")
        return list_of_modes, noises, Zs

    def find_gamma(K, interpolatory, Y, gamma_min, printer, tol=1e-10):
        eigenvalues, eigenvectors = onp.linalg.eigh(K)
        if not interpolatory:
            selected_eigenvalues = eigenvalues < tol
            residuals = (
                eigenvectors[:, selected_eigenvalues]
                @ (eigenvectors[:, selected_eigenvalues].T)
            ) @ Y
            gamma = onp.linalg.norm(residuals) ** 2
        else:

            def var(gamma_log):
                return -onp.var(1 / (1 + eigenvalues * onp.exp(-gamma_log)))

            res = minimize(
                var,
                onp.array([onp.log(onp.mean(eigenvalues))]),
                method="nelder-mead",
                options={"xatol": 1e-8, "disp": False},
            )
            gamma = onp.exp(res.x[0])
            gamma_med = onp.median(eigenvalues)
            if not (res.success) or not (gamma_med / 100 < gamma < 100 * gamma_med):
                printer(
                    f"auto gamma through variance maximisation seem to have failed (found gamma={gamma:.2e}). Relying on eigenvalue median instead (gamma={gamma_med:.2e})"
                )
                gamma = gamma_med

        if gamma < gamma_min:
            printer(
                f"""gamma too small for set gamma_min ({gamma:.2e}) needed for numerical stability, using {gamma_min:.2e} instead\nThis can either mean that the noise is very low or there is an issue in the automatic determination of gamma. To change the tolerance, change parameter gamma_min"""
            )
            gamma = gamma_min

        return gamma

    def plot_graph(
        self, type_label=True, node_size=400, cluster_size=10000, font_size=8
    ):
        if len(self.modes.clusters) == len(self.names):
            pos = nx.kamada_kawai_layout(self.G, self.G.nodes())
            nx.draw_networkx(
                self.G,
                with_labels=True,
                pos=pos,
                node_size=node_size,
                font_size=8,
                alpha=0.6,
            )

        cluster_partition = {
            item: i for i, sublist in enumerate(self.modes.clusters) for item in sublist
        }
        pos, hypergraph = partition_layout(self.G, cluster_partition)
        connection_style = "arc3,rad=0"
        node_sizes = [
            node_size if (node in self.G) else cluster_size
            for node in hypergraph.nodes()
        ]
        nx.draw_networkx_nodes(
            hypergraph, pos=pos, nodelist=self.G.nodes(), node_size=node_size, alpha=0.6
        )
        nx.draw_networkx_edges(
            hypergraph,
            pos,
            node_size=node_sizes,
            alpha=0.6,
            edgelist=[
                e for e in hypergraph.edges if not hypergraph.edges[e]["intra_cluster"]
            ],
            style="dashed",
            connectionstyle=connection_style,
        )
        nx.draw_networkx_edges(
            hypergraph,
            pos,
            node_size=node_size,
            alpha=0.6,
            edgelist=[
                e for e in hypergraph.edges if hypergraph.edges[e]["intra_cluster"]
            ],
            connectionstyle=connection_style,
        )
        nx.draw_networkx_nodes(
            hypergraph,
            pos,
            nodelist=set(cluster_partition.values()),
            node_color="None",
            node_size=cluster_size,
            edgecolors="red",
        )

        nx.draw_networkx_labels(
            hypergraph,
            pos,
            labels={node: node for node in self.G.nodes()},
            font_size=font_size,
        )

        if type_label:
            nx.draw_networkx_edge_labels(
                self.G, pos, edge_labels=nx.get_edge_attributes(self.G, "type")
            )
