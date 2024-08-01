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
from .helper_functions import make_find_ancestor_function, make_preprocessing_functions


class GraphDiscovery:
    """
    GraphDiscovery is the main class of CHD. It is used to discover the hypergraph structure of a dataset. It contains a Networkx object G that
    stores the results of graph discovery. It also contains a ModeContainer object that stores the kernel matrices of the modes of the dataset.

    To instantiate a graph discovery, you need:
        - X: the dataset, as a numpy array of shape (n_samples, n_features) of real numbers.
        - names: the names of the features, as a list of strings
        - kernels: the kernels used to compute the modes of the dataset (must be a list of ModeKernel objects). If None, the default kernels are used (linear, quadratic and gaussian)
        - normalize: Whether to normalize the data before graph discovery (normalization means centering and scaling to unit variance).
        - clusters: if you want to use clusters of node, you can provide a list of lists of strings, where each sublist is a cluster of nodes. If None, no clustering is used.
        - possible_edges: if you want to restrict the possible edges of the graph, you can provide a networkx.DiGraph object, where each edge is a possible edge of the graph.
        - verbose: whether to print information during graph discovery
        - gamma_min: the minimum value of gamma used in the graph discovery process

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
        Builds GraphDiscovery object.

        Args:
        - X (np.ndarray): the dataset, as an array of shape (n_samples, n_features). Data is treated as real numbers.
        - names (list of strings): the names of the features.
        - kernels (list of ModeKernel, default None): the kernels used to compute the modes of the dataset. If None, the default kernels are used (linear, quadratic, and Gaussian).
        - normalize (boolean, default True): Whether to normalize the data before graph discovery (normalization means centering and scaling to unit variance).
        - clusters (list of lists of strings, default None): if you want to use clusters of nodes, you can provide a list of lists of strings, where each sublist is a cluster of nodes. If None, no clustering is used.
        - possible_edges (nx.DiGraph object, default None): if you want to restrict the possible edges of the graph, you can provide the possible_edges, where each edge is a possible edge of the graph.
        - verbose (boolean, default True): whether to print information during graph discovery.
        - gamma_min (float, default 1e-9): the minimum value for the regularization parameter gamma.

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
            self.mean_x = X.mean(axis=0, keepdims=True)
            self.std_x = standard_devs
            self.X = (X - self.mean_x) / self.std_x
        else:
            self.mean_x = np.zeros((1, X.shape[1]))
            self.std_x = np.ones((1, X.shape[1]))
            self.X = X
        self.print_func = print if verbose else lambda *a, **k: None
        self.names = names

        self.name_to_index = {name: index for index, name in enumerate(names)}
        if possible_edges is not None:
            possible_edges.remove_edges_from(nx.selfloop_edges(possible_edges))
            self.print_func("Converting possible edges to dense adjacency matrix")
            self.possible_edges_adjacency = np.array(
                nx.adjacency_matrix(possible_edges, nodelist=self.names).todense()
            )
        self.possible_edges = possible_edges
        self.G = nx.DiGraph()
        self.G.add_nodes_from(names)

        self.has_clusters = clusters is not None
        self.modes = ModeContainer(self.names, clusters)
        if kernels is None:
            kernels = [LinearMode(), QuadraticMode(), GaussianMode(l=1)]
        else:
            assert len(kernels) == len(
                set([str(k) for k in kernels])
            ), "Kernels must have different names"

        self._kernels = kernels

        self.gamma_min = gamma_min
        self.prepare_functions(is_interpolatory=None)

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

        new_graph = GraphDiscovery(
            X=self.X,
            names=self.names,
            mode_container=self.modes,
            possible_edges=self.possible_edges,
            verbose=True,
        )
        new_graph.print_func = self.print_func
        new_graph.modes = ModeContainer(self.names, clusters)
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
        new_graph.has_clusters = True
        return new_graph

    def prepare_functions(self, is_interpolatory=None):
        """
        Prepares various functions used in the graph discovery process.

        Args:
            is_interpolatory (bool, optional): Flag indicating whether to force a the interpolatory behavior. Useful when a high number of features make non-interpolatory kernels (like quadratic) actually behave like an interpolatory kernel. Defaults to None.

        Returns:
            None
        """
        scales = {k.name: k.scale for k in self.kernels}
        for kernel in self.kernels:
            kernel.setup(self.X, scales=scales)

        self.interpolary_regression_find_gamma = jax.jit(
            jax.vmap(
                interpolatory.perform_regression_and_find_gamma,
                in_axes=(0, 0, 0, 0),
            )
        )
        self.non_interpolatory_regression_find_gamma = jax.jit(
            jax.vmap(
                non_interpolatory.perform_regression_and_find_gamma,
                in_axes=(0, 0, 0, 0),
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
                    scales,
                    has_clusters=self.has_clusters,
                    memory_efficient=kernel.memory_efficient_required,
                    is_interpolatory=is_interpolatory,
                )
            )
            for kernel in self.kernels
        }
        self.remove_ancestors, self.remove_ancestors_no_adj = (
            make_preprocessing_functions()
        )

    def fit(
        self,
        targets=None,
        key=random.PRNGKey(0),
        kernel_chooser=None,
        mode_chooser=None,
        message="",
    ):
        """
        Performs graph discovery. You can provide a list of targets (nodes of the graph) to discover ancestors of these targets. If None, the ancestors of all nodes are discovered.
        You can also provide several parameters to customize the graph discovery process.

        Args:
        - targets(list of strings, default None): nodes for which we discover the ancestors.
            Each string is the name of a node of the graph. If None, the ancestors of all nodes are discovered.
        - kernel_chooser (KernelChooser, default None): a KernelChooser object that chooses the kernel to use for each node. If None, a MinNoiseKernelChooser is used.
        - mode_chooser (ModeChooser, default None): a ModeChooser object that chooses the mode to use for each node. If None, a MaxIncrementModeChooser is used.

        See ComputationalHyperGraph.decision module for details on KernelChooser and ModeChooser objects.

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

        if targets is None:
            targets = self.names
        gas, active_modess = self._prepare_modes_for_ancestor_finding(targets)
        # running ancestor finding for each target
        key, *subkeys = random.split(key, len(targets) + 1)

        subkeys = np.array(subkeys)

        kernel_performance = self.get_kernel_performance(active_modess, gas, subkeys)

        ybs, gammas, subkeys, noisess, Z_lows, Z_highs, chosen_kernels = (
            GraphDiscovery.choose_kernel(kernel_performance, kernel_chooser)
        )

        for i, kernel in enumerate(self.kernels):
            mask_kernel = chosen_kernels == i
            if not np.any(mask_kernel):
                continue

            (
                active_modess_kernel,
                noisess_kernel,
                Z_lows_kernel,
                Z_highs_kernel,
                activationss_kernel,
                gammas_kernel,
                ybs_kernel,
            ) = self.prune_ancestors(
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
                message=message,
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
                gammas_kernel,
                ybs_kernel,
            )
        self.process_results(
            targets,
            -1,
            chosen_kernels == -1,
            *([None] * 6),
            kernel_performance,
            None,
            None,
        )

    def _prepare_modes_for_ancestor_finding(self, names):
        """
        Prepares the modes for finding the ancestors of 'name' by deleting the node with the given name from the modes and
        deleting all nodes that are not possible ancestors of the given node. Note that this deletes also the nodes that are in the same cluster as the given node.

        Args:
        - names (str): the name of the node to be deleted from the modes

        Returns:
        - ga (numpy.ndarray): the adjacency matrix of the node with the given name
        - active_modes (Modes): the modes with the node with the given name deleted and all nodes that are not
                                possible ancestors of the given node deleted
        """
        indexes = np.array([self.modes.name_to_index[name] for name in names])
        gas = self.X[:, indexes].T
        if self.possible_edges is None:
            active_modes = self.remove_ancestors_no_adj(
                self.modes.index_matrix, indexes
            )
        else:
            active_modes = self.remove_ancestors(
                self.possible_edges_adjacency, self.modes.index_matrix, indexes
            )
        return gas, active_modes

    def get_kernel_performance(
        self, active_modess, gas, subkeys, use_interpolatory=None
    ):
        """
        Calculates the performance of each kernel in terms of signal-to-noise ratio.

        Args:
            active_modess (jax.numpy.ndarray): Array of active modes.
            gas (jax.numpy.ndarray): array of target vectors
            subkeys (list): List of subkeys.
            use_interpolatory (bool, optional): Flag indicating whether to use interpolatory regression.
                Defaults to None.

        Returns:
            tuple: A tuple containing the performance metrics for each kernel. The tuple contains 6 arrays:
                - Array 1: The coefficients of the kernel regression.
                - Array 2: The signal-to-noise ratios.
                - Array 3: Z_low for Z-test
                - Array 4: Z_high for Z-test
                - Array 5: gamma found
                - Array 6: key (used for random number generation by JAX)
        """
        kernel_performance = []

        for kernel in self.kernels:
            K_mat = self.vmaped_kernel[kernel](
                self.X, self.X, np.sum(active_modess, axis=1)
            )
            min_eigenvalue = np.min(np.linalg.eigvalsh(K_mat), axis=1)
            gamma_mins = np.where(
                min_eigenvalue + self.gamma_min < 0, -2 * min_eigenvalue, self.gamma_min
            )
            interpolatory_bool = (
                kernel.is_interpolatory
                if use_interpolatory is None
                else use_interpolatory
            )
            if interpolatory_bool:
                res = self.interpolary_regression_find_gamma(
                    K_mat, gas, gamma_mins, subkeys
                )
            else:
                res = self.non_interpolatory_regression_find_gamma(
                    K_mat, gas, gamma_mins, subkeys
                )
            if np.isnan(res[4]).any():
                print(
                    "Somme gammas were found to be NaNs. This error may not be corrected, so corresponding signal-to-noise ratio are set to 1"
                )
                new_res_1 = np.where(np.isnan(res[4]), 1, res[1])
                res = (res[0], new_res_1, *res[2:])
            if np.isnan(res[1]).any():
                error_message = "The regression has returned NaNs, this is likely due to the kernel matrix not being positive definite\n"
                error_message += "See the spectrum below for confirmation. Consider increasing gamma_min\n"
                error_message += (
                    f"{np.linalg.eigvalsh(K_mat[np.argmax(np.isnan(res[1]))])}"
                )
                raise ValueError(error_message)
            kernel_performance.append(res)
        kernel_performance = tuple(
            np.stack([k_perf[i] for k_perf in kernel_performance], axis=1)
            for i in range(6)
        )
        return kernel_performance

    def choose_kernel(kernel_performances, kernel_chooser):
        """
        Choose the best kernel based on the given kernel performances.

        Args:
            kernel_performances (list): A list of kernel performances for each kernel.
            kernel_chooser (function): A function that chooses the best kernel based on the performances.

        Returns:
            tuple: A tuple containing the following arrays:
                - ybs (numpy.ndarray): Array of yb values.
                - gammas (numpy.ndarray): Array of gamma values.
                - subkeys (numpy.ndarray): Array of subkey values.
                - noisess (numpy.ndarray): Array of noise values.
                - Z_lows (numpy.ndarray): Array of Z_low values.
                - Z_highs (numpy.ndarray): Array of Z_high values.
                - chosen_kernels (numpy.ndarray): Array of chosen kernel values.
        """
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
        self,
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
        message,
    ):
        """
        Prunes ancestors based on the given parameters.

        Args:
            kernel: The kernel used for ancestor finding.
            ancestor_finding_step_funcs: function used for ancestor finding.
            X: The input data.
            active_modess_kernel: The active modes.
            gas_kernel: The target vectors
            ybs_kernel: The regression coefficients
            gammas_kernel: The gammas
            subkeys_kernel: The subkeys
            noise_kernel: The noise to signal ratios ratios
            Z_low_kernel: The Z_lows
            Z_high_kernel: The Z_highs
            loop_number: The number of iterations for ancestor finding.
            message: Additional message to display during ancestor finding.

        Returns:
            A tuple containing the following arrays, each aggregating all the values encountered during the ancestor finding process:
            - ancestor_modess: An array of ancestor modes.
            - noisess_kernel: An array of noise to signal ratios.
            - Z_lows_kernel: An array of Z lows
            - Z_highs_kernel: An array of Z highs
            - activationss_kernel: An array of activations
            - gammas: An array of gammas.
            - ybs: An array of regression coefficients.
        """
        ancestor_modess = [np.sum(active_modess_kernel, axis=1)]
        noisess_kernel = [noise_kernel]
        Z_lows_kernel = [Z_low_kernel]
        Z_highs_kernel = [Z_high_kernel]
        gammas = [gammas_kernel]
        ybs = [ybs_kernel]
        activationss_kernel = []

        pbar = tqdm(
            total=loop_number,
            desc=f"Finding ancestors with kernel [{kernel}]",
            position=0,
            leave=True,
        )
        pbar.set_postfix_str(message)
        ancestor_finding_step = ancestor_finding_step_funcs[kernel]
        for step in range(loop_number):
            # because of the very high number of ancestor, gamma_min needs to be recomputed often, to avoid numerical instability
            # and keep it as small as possible
            p = X.shape[1] - step - 1
            if step % 50 == 0 or (p * (p + 1)) / 2 == X.shape[1]:
                K_mat = self.vmaped_kernel[kernel](
                    self.X, self.X, np.sum(active_modess_kernel, axis=1)
                )

                min_eigenvalue = np.linalg.eigvalsh(K_mat)[:, 0]
                gamma_min_kernel = np.where(
                    min_eigenvalue + 1e-9 < 0, -2 * min_eigenvalue, 1e-9
                )

            (
                active_modess_kernel,
                ybs_kernel,
                gammas_kernel,
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
                gamma_min_kernel,
            )
            ancestor_modess.append(np.sum(active_modess_kernel, axis=1))
            noisess_kernel.append(noise)
            Z_lows_kernel.append(Z_low)
            Z_highs_kernel.append(Z_high)
            activationss_kernel.append(activations)
            gammas.append(gammas_kernel)
            ybs.append(ybs_kernel)
            pbar.update(1)
            if np.all(active_modess_kernel == 0):
                break
        pbar.close()
        return (
            np.stack(ancestor_modess, axis=1),
            np.stack(noisess_kernel, axis=1),
            np.stack(Z_lows_kernel, axis=1),
            np.stack(Z_highs_kernel, axis=1),
            np.stack(activationss_kernel, axis=1),
            np.stack(gammas, axis=1),
            np.stack(ybs, axis=1),
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
        gammas,
        ybs_kernel,
    ):
        """
        Process the results of the hypergraph discovery algorithm.

        Args:
            targets (list): List of target names.
            chosen_kernel (int): Index of the chosen kernel.
            mask_kernel (list): List of boolean values indicating whether each target has a mask.
            chosen_modes (list): List of chosen modes for each target.
            ancestor_modess (list): List of ancestor modes for each target.
            noisess (list): List of noise values for each target and mode.
            Z_lows (list): List of lower bounds of Z values for each target and mode.
            Z_highs (list): List of upper bounds of Z values for each target and mode.
            activationss (list): List of activation values for each target and mode.
            kernel_performances (list): List of kernel performances for each target and mode.
            gammas (ndarray): Array of gamma values for each target and mode.
            ybs_kernel (ndarray): Array of yb values for each target and mode.

        Returns:
            None
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
                    f"Kernel [{kernel}] has n/(n+s)={noises_kernel[i]}, Z=({Z_lows_kernels[i]:.2f}, {Z_highs_kernels[i]:.2f}), gamma={max(self.gamma_min,gammas_kernel[i]):.2e}"
                )

            if chosen_kernel == -1:  # case no ancestors, step 7 in the paper
                self.print_func(f"{name} has no ancestors\n")
                index += 1
                continue

            chosen_mode = chosen_modes[index]
            ancestor_modes = ancestor_modess[index]
            noises = noisess[index]
            Z_low = Z_lows[index]
            Z_high = Z_highs[index]
            activations = activationss[index]
            gamma = gammas[index, chosen_mode]
            yb = ybs_kernel[index, chosen_mode]
            self.print_func(
                f"{name} has ancestors with the kernel [{self.kernels[chosen_kernel]}] | (n/(s+n)={float(noisess[index][chosen_mode]):.2f} after pruning)"
            )

            # plot evolution of noise and Z, and in second plot on the side evolution of Z_{k+1}-Z_k
            ancestor_number = [np.sum(mode) for mode in ancestor_modes]
            fig, axes = plot_noise_evolution(
                ancestor_number,
                noises,
                [(Zl, Zh) for Zl, Zh in zip(Z_low, Z_high)],
                node_name=name,
                ancestor_modes_number=ancestor_number[chosen_mode],
            )
            plt.show()
            plt.close(fig)

            # adding ancestors to graph and storing activations (step 19)
            acivation_per_variable = np.sum(
                self.modes.index_matrix * activations[chosen_mode][:, None], axis=0
            )
            active_variables = ancestor_modes[chosen_mode]
            self.G.nodes[name].update(
                {
                    "active_modes": active_variables,
                    "kernel_index": chosen_kernel,
                    "type": str(self.kernels[chosen_kernel]),
                    "gamma": float(gamma),
                    "noise": float(noises[chosen_mode]),
                    "coeff": yb,
                }
            )
            ancestor_names = []
            for i, (activation, active) in enumerate(
                zip(acivation_per_variable, active_variables)
            ):
                if active == 1:
                    ancestor_names.append(self.names[i])
                    self.G.add_edge(self.names[i], name, signal=activation)
                elif active == 0:
                    continue
                else:
                    raise ValueError("Inconsistent activation")

            self.print_func(f"Ancestors of {name}: {ancestor_names}\n")

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

    def predict(self, names, X_pred):
        assert X_pred.shape[1] == self.X.shape[1]
        res = np.zeros((X_pred.shape[0], len(names)))

        # apply mean and std transform
        X_pred_used = (X_pred - self.mean_x) / self.std_x

        k_dic = nx.get_node_attributes(self.G, "kernel_index")
        kernels = np.array([k_dic.get(name, -1) for name in names])
        active_mode_dic = nx.get_node_attributes(self.G, "active_modes")
        active_modess = np.stack(
            [
                active_mode_dic.get(
                    name,
                    np.zeros(
                        self.X.shape[1],
                    ),
                )
                for name in names
            ],
            axis=0,
        )
        yb_dic = nx.get_node_attributes(self.G, "coeff")
        ybs = np.stack(
            [yb_dic.get(name, np.zeros(self.X.shape[0])) for name in names], axis=0
        )

        for i, kernel in enumerate(self.kernels):
            mask = kernels == i
            if not np.any(mask):
                continue
            K_mat = self.vmaped_kernel[kernel](X_pred_used, self.X, active_modess[mask])
            pred = np.einsum("nij,nj->in", K_mat, -ybs[mask])
            res = res.at[:, mask].set(pred)
        # transform the result back
        indexes = np.array([self.name_to_index[name] for name in names])
        res = res * self.std_x[:, indexes] + self.mean_x[:, indexes]

        return res
