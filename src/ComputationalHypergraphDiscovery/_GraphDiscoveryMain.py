from functools import partial
import time
import jax.numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import jax.scipy.linalg as jax_linalg
from jax import random
import jax
from jax.scipy.optimize import minimize
from tqdm import tqdm
from . import interpolatory, non_interpolatory


from .Modes import ModeContainer, LinearMode, QuadraticMode, GaussianMode
from .decision import *
from .util import partition_layout, plot_noise_evolution
from .helper_functions import (
    make_find_ancestor_function,
)


class GraphDiscovery:
    """GraphDiscovery is the main class of CHD. It is used to discover the hypergraph structure of a dataset. It contains a Networkx object G that
    stores the results of graph discovery. It also contains a ModeContainer object that stores the kernel matrices of the modes of the dataset.
    To instantiate a graph discovery, you need:
        - X: the dataset, as a numpy array of shape (n_features, n_samples) of real numbers.
        - names: the names of the features, as a list of strings
        - mode_kernels: the kernels used to compute the modes of the dataset (must be a mode kernel object). If None, the default kernels are used (linear, quadratic and gaussian)
        - mode_container: Alternatively, if the kernel matrices are already computed (for instance when reusing computations from a previous graph), you can provide a ModeContainer object,
        which contains the kernel matrices of the modes of the dataset. You cannot provide both mode_kernels and mode_container.
        - clusters: if you want to use clusters of node, you can provide a list of lists of strings, where each sublist is a cluster of nodes. If None, no clustering is used.
        - possible_edges: if you want to restrict the possible edges of the graph, you can provide a networkx.DiGraph object, where each edge is a possible edge of the graph.

    Available class methods:
        - from_dataframe: alternative constructor, that takes a pandas dataframe as input. You can also provide normalize=True to normalize the data before graph discovery, as well as any keyword argument of the constructor.

    Available object methods:
        - fit: performs graph discovery. You can provide a list of targets (nodes of the graph) to discover ancestors of these targets. If None, the ancestors of all nodes are discovered. See the help of the method for more details.
        - plot_graph: plots the graph. You can provide several kwargs to customize the plot. Handles clustered nodes
        - prepare_new_graph_with_clusters: if you want to use clusters of nodes, you can use this method to create a new GraphDiscovery object with the same parameters as the current one, but with different clusters.
        It will also inherit the graph of the current GraphDiscovery object, but with the edges between clusters removed. See the help of the method for more details.

    Other methods:
        Other methods with _ prefix are used internally.
    """

    def __init__(
        self,
        X,
        names,
        kernels=None,
        normalize=True,
        clusters=None,
        possible_edges=None,
        verbose=True,
        gamma_min=1e-9,
    ) -> None:
        """
        Builds GraphDiscovery object. In particular, if mode_container is None, it computes the kernel matrices of the modes of the dataset, which
        can be computationally expensive. If you want to reuse computations from a previous graph, you can provide a ModeContainer object instead.

        Args:
        - X (np.ndarray):the dataset, as an array of shape (n_features, n_samples). Data is treated as real numbers.
        - names (list of strings): the names of the features,
        - mode_kernels (ModeKernelList or ModeKernel, default None): the kernels used to compute the modes of the dataset. If None, the default kernels are used (linear, quadratic and gaussian)
        - mode_container (ModeContainer object, default None): alternatively, if the kernel matrices are already computed (for instance when reusing computations from a previous graph), you can provide a ModeContainer object,
            which contains the kernel matrices of the modes of the dataset. You cannot provide both mode_kernels and mode_container.
        - normalize (boolean, default True): Whether to normalize the data before graph discovery (normalization means centering and scaling to unit variance).
        - clusters (list of lists of strings, default None): if you want to use clusters of node, you can provide a list of lists of strings, where each sublist is a cluster of nodes. If None, no clustering is used.
        - possible_edges (nx.DiGraph object, default None): if you want to restrict the possible edges of the graph, you can provide the possible_edges, where each edge is a possible edge of the graph.
        - verbose (boolean, default True): whether to print information during graph discovery

        Returns:
        - GraphDiscovery object
        """

        assert X.shape[1] == len(
            names
        ), "X must have as many columns as there are names"
        assert len(X.shape) == 2, "X must be a 2D array"
        assert not np.isnan(np.sum(X)), "X must not contain NaN values"
        assert not np.isinf(np.sum(X)), "X must not contain infinite values"
        standard_devs = X.std(axis=0, keepdims=True)
        if np.any(standard_devs == 0):
            raise ValueError(
                "Some features have a standard deviation of 0. This can lead to numerical instability. Please remove these features."
            )

        if normalize:
            self.X = (X - X.mean(axis=0, keepdims=True)) / standard_devs
        else:
            self.X = X
        self.print_func = print if verbose else lambda *a, **k: None
        self.names = names

        self.name_to_index = {name: index for index, name in enumerate(names)}
        self.possible_edges = possible_edges
        self.G = nx.DiGraph()
        self.G.add_nodes_from(names)

        self.modes = ModeContainer(self.names, clusters)
        if kernels is None:
            kernels = [LinearMode(), QuadraticMode(), GaussianMode(l=1)]
        else:
            assert len(kernels) == len(
                set([str(k) for k in kernels])
            ), "Kernels must have different names"
        self._kernels = kernels
        self.gamma_min = gamma_min
        self.prepare_functions()

    @property
    def kernels(self):
        return self._kernels

    @kernels.setter
    def kernels(self, *args, **kwargs):
        raise ValueError(
            "Kernels cannot be changed after initialization as JAX does not support changing the function signature of a jit-compiled function. Please create a new GraphDiscovery object with the desired kernels."
        )

    def from_dataframe(df, **kwargs):
        """
        Alternative constructor, that takes a pandas dataframe as input. You can also provide normalize=True to normalize the data before graph discovery, as well as any keyword argument of the constructor.
        See the help of the constructor for details on kwargs

        Args:
        - df (pandas.DataFrame): the dataframe containing the data (consisting of real numbers).

        - **kwargs: Any keyword argument of the constructor

        Returns:
        - GraphDiscovery object
        """
        X = df.values
        return GraphDiscovery(X=X, names=df.columns, **kwargs)

    def prepare_new_graph_with_clusters(self, clusters):
        """
        If you want to use clusters of nodes, you can use this method to create a new GraphDiscovery object with the same parameters as the current one, but with different clusters.
        It will also inherit the graph of the current GraphDiscovery object, but with the edges between clusters removed.
        The cluster must be a partition of the nodes of the graph (i.e. each node must be in exactly one cluster).

        Args:
        - clusters (list of list of strings) cluster of nodes, where each sublist is a cluster of nodes, and each string is the name of a node

        Returns:
        GraphDiscovery object
        """
        print("to update")

        new_graph = GraphDiscovery(
            X=self.X,
            names=self.names,
            mode_container=self.modes,
            possible_edges=self.possible_edges,
            verbose=True,
        )
        new_graph.print_func = self.print_func
        new_graph.modes.assign_clusters(
            clusters
        )  # assigns clusters to modeContainer object to handle clusters (this returns a new object)
        new_graph.G = self.G.copy()
        edges_to_remove = []
        flattened_clusters = [
            (i, item) for i, sublist in enumerate(clusters) for item in sublist
        ]
        # removing edges between clusters
        for i, (cluster_index, node) in enumerate(flattened_clusters):
            other_nodes = list(
                set([node for j, node in flattened_clusters if j > i])
                - set(clusters[cluster_index])
            )
            for other_node in other_nodes:
                edges_to_remove.append((node, other_node))
        new_graph.G.remove_edges_from(edges_to_remove)
        return new_graph

    def prepare_functions(self):
        self.interpolary_regression_find_gamma = jax.jit(
            jax.vmap(
                interpolatory.perform_regression_and_find_gamma,
                in_axes=(0, 0, None, 0),
            )
        )
        self.non_interpolatory_regression_find_gamma = jax.jit(
            jax.vmap(
                non_interpolatory.perform_regression_and_find_gamma,
                in_axes=(0, 0, None, 0),
            )
        )
        apply_kernel = lambda k, X, Y, which_dim: k(X, Y, which_dim)
        self.vmaped_kernel = {
            k: jax.jit(jax.vmap(partial(apply_kernel, k), in_axes=(None, None, 0)))
            for k in self.kernels
        }
        self.ancestor_finding_step_funcs = {
            kernel: jax.jit(
                make_find_ancestor_function(
                    kernel,
                    gamma_min=self.gamma_min,
                    memory_efficient=kernel.memory_efficient_required,
                )
            )
            for kernel in self.kernels
        }

    def fit(
        self,
        targets=None,
        key=random.PRNGKey(0),
        kernel_chooser=None,
        mode_chooser=None,
        early_stopping=None,
        jit_all=True,
        vmap=True,
    ):
        """
        Performs graph discovery. You can provide a list of targets (nodes of the graph) to discover ancestors of these targets. If None, the ancestors of all nodes are discovered.
        You can also provide several parameters to customize the graph discovery process.

        Args:
        - targets(list of strings, default None): nodes for which we discover the ancestors.
            Each string is the name of a node of the graph. If None, the ancestors of all nodes are discovered.
        - gamma(float or "auto", default "auto"): the gamma parameter used for the regression.
            If "auto", the gamma parameter is automatically determined.
            If a float, the gamma parameter is fixed to this value.
            It is advised to use "auto" for most applications, as a good choice of gamma is crucial for the performance of the algorithm, and unintuitive to find.
        - gamma_min (float, default 1e-9): the minimum value of gamma allowed. If gamma is "auto", the gamma parameter is automatically determined, but must be greater than gamma_min.
            A gamma_min that is too small may lead to numerical instability.
        - kernel_chooser (KernelChooser, default None): a KernelChooser object that chooses the kernel to use for each node. If None, a MinNoiseKernelChooser is used.
        - mode_chooser (ModeChooser, default None): a ModeChooser object that chooses the mode to use for each node. If None, a MaxIncrementModeChooser is used.
        - early_stopping (EarlyStopping object, default None): an object of a class that implements the EarlyStopping interface. If None, a NoEarlyStopping is used.

        See ComputationalHyperGraph.decision module for details on KernelChooser, ModeChooser and EarlyStopping objects.

        Returns:
        - None
        """

        # checking arguments or instantiating default objects
        if kernel_chooser is None:
            kernel_chooser = MinNoiseKernelChooser()
        else:
            kernel_chooser.is_a_kernel_chooser()
        if mode_chooser is None:
            mode_chooser = MaxIncrementModeChooser()
        else:
            mode_chooser.is_a_mode_chooser()
        mode_chooser = jax.jit(jax.vmap(mode_chooser, in_axes=(0, 0, (0, 0))))
        """if early_stopping is None:
            early_stopping = NoEarlyStopping()
        else:
            early_stopping.is_an_early_stopping()"""

        if targets is None:
            targets = self.names
        gas, active_modess = self._prepare_modes_for_ancestor_finding(targets)
        # running ancestor finding for each target
        key, *subkeys = random.split(key, len(targets) + 1)

        gas = np.array(gas)
        active_modess = np.array(active_modess)
        subkeys = np.array(subkeys)

        kernel_performance = self.get_kernel_performance(active_modess, gas, subkeys)

        ybs, gammas, subkeys, noisess, Z_lows, Z_highs, chosen_kernels = (
            GraphDiscovery.choose_kernel(kernel_performance, kernel_chooser)
        )

        for i, kernel in enumerate(self.kernels):
            mask_kernel = chosen_kernels == i

            (
                active_modess_kernel,
                noisess_kernel,
                Z_lows_kernel,
                Z_highs_kernel,
                activationss_kernel,
            ) = GraphDiscovery.prune_ancestors(
                kernel,
                self.ancestor_finding_step_funcs,
                X=self.X,
                active_modess_kernel=active_modess[mask_kernel],
                gas_kernel=gas[mask_kernel],
                ybs_kernel=ybs[mask_kernel],
                gammas_kernel=gammas[mask_kernel],
                subkeys_kernel=subkeys[mask_kernel],
                noise_kernel=noisess[mask_kernel],
                Z_low_kernel=Z_lows[mask_kernel],
                Z_high_kernel=Z_highs[mask_kernel],
                loop_number=len(self.modes.clusters) - 2,
            )
            anomaly_noise = np.logical_or(noisess_kernel > 1.05, noisess_kernel < -0.05)
            anomaly_Z_low = np.logical_or(Z_lows_kernel > 1.05, Z_lows_kernel < -0.05)
            anomaly_Z_high = np.logical_or(
                Z_highs_kernel > 1.05, Z_highs_kernel < -0.05
            )

            if np.any(anomaly_noise) or np.any(anomaly_Z_low) or np.any(anomaly_Z_high):
                print(
                    "-" * 50
                    + "\n** Anomaly in noise or Z values, consider increasing gamma_min **\n"
                    + "-" * 50
                )
                first_occurence_noise = np.argmax(anomaly_noise, axis=1, keepdims=True)
                first_occurence_Z_low = np.argmax(anomaly_Z_low, axis=1, keepdims=True)
                first_occurence_Z_high = np.argmax(
                    anomaly_Z_high, axis=1, keepdims=True
                )
                print(anomaly_noise)
                print(noisess_kernel)
                noisess_kernel = np.where(
                    np.logical_and(
                        np.arange(noisess_kernel.shape[1])[None, :]
                        >= first_occurence_noise,
                        np.any(anomaly_noise, axis=1, keepdims=True),
                    ),
                    1.0,
                    noisess_kernel,
                )
                Z_lows_kernel = np.where(
                    np.logical_and(
                        np.arange(Z_lows_kernel.shape[1])[None, :]
                        >= first_occurence_Z_low,
                        np.any(anomaly_Z_low, axis=1, keepdims=True),
                    ),
                    1.0,
                    Z_lows_kernel,
                )
                Z_highs_kernel = np.where(
                    np.logical_and(
                        np.arange(Z_highs_kernel.shape[1])[None, :]
                        >= first_occurence_Z_high,
                        np.any(anomaly_Z_high, axis=1, keepdims=True),
                    ),
                    1.0,
                    Z_highs_kernel,
                )

            chosen_modes = mode_chooser(
                active_modess_kernel,
                noisess_kernel,
                (Z_lows_kernel, Z_highs_kernel),
            )
            self.process_results(
                targets,
                i,
                mask_kernel,
                chosen_modes,
                active_modess_kernel,
                noisess_kernel,
                Z_lows_kernel,
                Z_highs_kernel,
                activationss_kernel,
                kernel_performance,
            )
        self.process_results(
            targets,
            -1,
            chosen_kernels == -1,
            *([None] * 6),
            kernel_performance,
        )

    def _prepare_modes_for_ancestor_finding(self, names):
        """
        Prepares the modes for finding the ancestors of 'name' by deleting the node with the given name from the modes and
        deleting all nodes that are not possible ancestors of the given node. Note that this deletes also the nodes that are in the same cluster as the given node.

        Args:
        - name (str): the name of the node to be deleted from the modes

        Returns:
        - ga (numpy.ndarray): the adjacency matrix of the node with the given name
        - active_modes (Modes): the modes with the node with the given name deleted and all nodes that are not
                                possible ancestors of the given node deleted
        """
        gas = []
        active_modes = []
        for name in names:
            ga = self.X[:, self.name_to_index[name]]
            active_indexes = self.modes.delete_node_by_name(
                name, self.modes.index_matrix
            )
            if self.possible_edges is not None:
                for possible_name in self.modes.names:
                    if (
                        possible_name not in self.possible_edges.predecessors(name)
                        and possible_name != name
                    ):
                        active_indexes = self.modes.delete_node_by_name(
                            possible_name, active_indexes
                        )
            active_modes.append(active_indexes)
            gas.append(ga)
        return gas, active_modes

    def get_kernel_performance(self, active_modess, gas, subkeys):
        kernel_performance = []
        for kernel in self.kernels:
            K_mat = self.vmaped_kernel[kernel](
                self.X, self.X, np.sum(active_modess, axis=1)
            )
            if kernel.is_interpolatory:
                res = self.interpolary_regression_find_gamma(
                    K_mat, gas, self.gamma_min, subkeys
                )
            else:
                res = self.non_interpolatory_regression_find_gamma(
                    K_mat, gas, self.gamma_min, subkeys
                )
            kernel_performance.append(res)
        kernel_performance = tuple(
            np.stack([k_perf[i] for k_perf in kernel_performance], axis=1)
            for i in range(6)
        )
        return kernel_performance

    def choose_kernel(kernel_performances, kernel_chooser):
        ybs = []
        gammas = []
        subkeys = []
        noisess = []
        Z_lows = []
        Z_highs = []
        chosen_kernels = []
        for i in range(kernel_performances[0].shape[0]):
            which_kernel, yb, noise, Z_low, Z_high, gamma, skey = kernel_chooser(
                [kernel_performances[j][i] for j in range(6)]
            )
            chosen_kernels.append(which_kernel)
            ybs.append(yb)
            noisess.append(noise)
            Z_lows.append(Z_low)
            Z_highs.append(Z_high)
            gammas.append(gamma)
            subkeys.append(skey)
        ybs = np.array(ybs)
        gammas = np.array(gammas)
        subkeys = np.array(subkeys)
        noisess = np.array(noisess)
        Z_lows = np.array(Z_lows)
        Z_highs = np.array(Z_highs)
        chosen_kernels = np.array(chosen_kernels)
        subkeys = np.array(subkeys)
        return ybs, gammas, subkeys, noisess, Z_lows, Z_highs, chosen_kernels

    def prune_ancestors(
        kernel,
        ancestor_finding_step_funcs,
        X,
        active_modess_kernel,
        gas_kernel,
        ybs_kernel,
        gammas_kernel,
        subkeys_kernel,
        noise_kernel,
        Z_low_kernel,
        Z_high_kernel,
        loop_number,
    ):

        ancestor_modess = [np.sum(active_modess_kernel, axis=1)]
        noisess_kernel = [noise_kernel]
        Z_lows_kernel = [Z_low_kernel]
        Z_highs_kernel = [Z_high_kernel]
        activationss_kernel = []

        pbar = tqdm(
            total=loop_number,
            desc=f"Finding ancestors with kernel [{kernel}]",
            position=0,
            leave=True,
        )
        ancestor_finding_step = ancestor_finding_step_funcs[kernel]
        for step in range(loop_number):
            (
                active_modess_kernel,
                ybs_kernel,
                subkeys_kernel,
                activations,
                noise,
                Z_low,
                Z_high,
            ) = ancestor_finding_step(
                X,
                active_modess_kernel,
                gas_kernel,
                gammas_kernel,
                ybs_kernel,
                subkeys_kernel,
            )
            ancestor_modess.append(np.sum(active_modess_kernel, axis=1))
            noisess_kernel.append(noise)
            Z_lows_kernel.append(Z_low)
            Z_highs_kernel.append(Z_high)
            activationss_kernel.append(activations)

            pbar.update(1)
        pbar.close()

        return (
            np.stack(ancestor_modess, axis=1),
            np.stack(noisess_kernel, axis=1),
            np.stack(Z_lows_kernel, axis=1),
            np.stack(Z_highs_kernel, axis=1),
            np.stack(activationss_kernel, axis=1),
        )

    def process_results(
        self,
        targets,
        chosen_kernel,
        mask_kernel,
        chosen_modes,
        ancestor_modess,
        noisess,
        Z_lows,
        Z_highs,
        activationss,
        kernel_performances,
    ):
        """
        Finds ancestors of a given node in the graph. This method is called by the fit method.
        This corresponds algorthm 1 in the paper.
        Has two main components:
            - finding the kernel to use for each node using _compute_kernel_preformances and the kernel_chooser
            - pruning the ancestors of the node using the chosen kernel using _iterative_ancestor_pruner

        Finally, the mode_chooser is used to choose the ancestors of the nodes. Data is stored in the graph an the noise evolutionis plotted.

        Args:
        - name (str): The name of the node to find ancestors for.
        - gamma (float or str): The gamma value to use for kernel computation. If set to "auto", the gamma value will be
            automatically determined based on the interpolatory property of the active modes.
        - gamma_min (float): The minimum gamma value to use for kernel computation.
        - kernel_chooser (callable): A function that takes in a dictionary of kernel performances and returns the key of
            the chosen kernel.
        - mode_chooser (callable): A function that takes in a list of modes, a list of noises, and a list of Zs, and
            returns the chosen mode.
        - early_stopping (bool): Whether to use early stopping during ancestor pruning.

        Returns:
        - None
        """

        index = 0

        for (
            name,
            mask,
            *kernel_performance,
        ) in zip(
            targets,
            mask_kernel,
            *kernel_performances,
        ):
            if not mask:
                continue
            self.print_func(f"\nResults for {name}")
            _, noises_kernel, Z_lows_kernels, Z_highs_kernels, gammas_kernel, _ = (
                kernel_performance
            )
            for i, kernel in enumerate(self.kernels):
                self.print_func(
                    f"Kernel [{kernel}] has n/(n+s)={noises_kernel[i]}, Z=({Z_lows_kernels[i]:.2f}, {Z_highs_kernels[i]:.2f}), gamma={gammas_kernel[i]:.2e}"
                )

            if chosen_kernel == -1:  # case no ancestors, step 7 in the paper
                self.print_func(f"{name} has no ancestors\n")
                index += 1
                continue
            self.print_func(
                f"{name} has ancestors with the kernel [{self.kernels[chosen_kernel]}] | (n/(s+n)={noises_kernel[0]:.2f})"
            )
            chosen_mode = chosen_modes[index]
            ancestor_modes = ancestor_modess[index]
            noises = noisess[index]
            Z_low = Z_lows[index]
            Z_high = Z_highs[index]
            activations = activationss[index]

            # plot evolution of noise and Z, and in second plot on the side evolution of Z_{k+1}-Z_k
            ancestor_number = [np.sum(mode) for mode in ancestor_modes]
            fig, axes = plot_noise_evolution(
                ancestor_number,
                noises,
                [(Zl, Zh) for Zl, Zh in zip(Z_low, Z_high)],
                node_name=name,
                ancestor_modes_number=ancestor_number[chosen_mode],
            )

            # adding ancestors to graph and storing activations (step 19)
            acivation_per_variable = np.sum(
                self.modes.index_matrix * activations[chosen_mode][:, None], axis=0
            )
            active_variables = ancestor_modes[chosen_mode]
            ancestor_names = []
            for i, (activation, active) in enumerate(
                zip(acivation_per_variable, active_variables)
            ):
                if active == 1:
                    ancestor_names.append(self.names[i])
                    self.G.add_edge(
                        self.names[i],
                        name,
                        type=str(self.kernels[chosen_kernel]),
                        signal=activation,
                    )
                elif active == 0:
                    continue
                else:
                    raise ValueError("Inconsistent activation")

            self.print_func(f"Ancestors of {name}: {ancestor_names}\n")
            plt.show()
            index += 1

    def plot_graph(self, type_label=True, **kwargs):
        """
        Plot the hypergraph.

        Args:
        - type_label (bool): Whether to show the type of each edge.
        - node_size (int): Size of the nodes.
        - cluster_size (int): Size of the cluster.
        - font_size (int): Size of the font.
        - with_labels (bool): Whether to show the labels.
        - alpha (float): Transparency of the nodes.
        - connection_style (str): Style of the connection.
        - between_cluster_edge_style (str): Style of the edge between clusters.
        - cluster_edge_color (str): Color of the edge of the cluster.

        Returns:
        - None
        """
        node_size = kwargs.get("node_size", 400)
        cluster_size = kwargs.get("cluster_size", 10000)
        font_size = kwargs.get("font_size", 8)
        with_labels = kwargs.get("with_labels", True)
        alpha = kwargs.get("alpha", 0.6)
        connection_style = kwargs.get("connection_style", "arc3,rad=0")
        between_cluster_edge_style = kwargs.get("between_cluster_edge_style", "dashed")
        cluster_edge_color = kwargs.get("cluster_edge_color", "red")

        if len([node for cluster in self.modes.clusters for node in cluster]) == len(
            self.names
        ):  # test if there are any clusters
            plt.figure(figsize=(10, 4))
            pos = nx.kamada_kawai_layout(self.G, self.G.nodes())
            nx.draw_networkx(
                self.G,
                with_labels=with_labels,
                pos=pos,
                node_size=node_size,
                font_size=font_size,
                alpha=alpha,
            )
            if type_label:
                nx.draw_networkx_edge_labels(
                    self.G, pos, edge_labels=nx.get_edge_attributes(self.G, "type")
                )
            x_values, y_values = zip(*pos.values())
            x_max = max(x_values)
            x_min = min(x_values)
            x_margin = (x_max - x_min) * 0.25
            plt.xlim(x_min - x_margin, x_max + x_margin)
            return

        # plotting clusters
        cluster_partition = {
            item: i for i, sublist in enumerate(self.modes.clusters) for item in sublist
        }
        pos, hypergraph = partition_layout(self.G, cluster_partition)
        x_values, y_values = zip(*pos.values())
        x_max = max(x_values)
        x_min = min(x_values)
        x_margin = (x_max - x_min) * 0.5

        node_sizes = [
            node_size if (node in self.G) else cluster_size
            for node in hypergraph.nodes()
        ]
        nx.draw_networkx_nodes(
            hypergraph,
            pos=pos,
            nodelist=self.G.nodes(),
            node_size=node_size,
            alpha=alpha,
        )
        nx.draw_networkx_edges(
            hypergraph,
            pos,
            node_size=node_sizes,
            alpha=alpha,
            edgelist=[
                e for e in hypergraph.edges if not hypergraph.edges[e]["intra_cluster"]
            ],
            style=between_cluster_edge_style,
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
            edgecolors=cluster_edge_color,
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
        plt.xlim(x_min - x_margin, x_max + x_margin)
