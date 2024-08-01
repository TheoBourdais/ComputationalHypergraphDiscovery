import jax
import jax.numpy as np
from functools import partial
from . import interpolatory
from . import non_interpolatory
from .Modes import kernels as kClass


def make_preprocessing_functions():
    """
    Creates and returns two preprocessing functions: remove_non_ancestors and remove_non_ancestors_no_adj.
    These function setup the pruning process by taking the information on the nodesand removing nodes that should be ignored.
    One of them takes into account an adjacency matrix, to handle if possible_edges was provided.

    Returns:
        remove_non_ancestors (function): A function that removes non-ancestors from the modes array based on the given index.
        remove_non_ancestors_no_adj (function): A function that removes non-ancestors from the modes array without using the adjacency matrix.
    """

    def remove_non_ancestors_no_adj(modes, index):
        non_ancestors_bool = np.arange(modes.shape[1]) == index
        cluster_mask = (modes == 1) * non_ancestors_bool[None, :]
        return np.where(cluster_mask, 0, modes)

    def remove_non_ancestors(adj, modes, index):
        non_ancestors_bool = adj[:, index] == 0
        non_ancestors_bool = non_ancestors_bool.at[index].set(True)
        cluster_mask = (modes == 1) * non_ancestors_bool[None, :]
        return np.where(cluster_mask, 0, modes)

    remove_non_ancestors = jax.vmap(
        jax.jit(remove_non_ancestors), in_axes=(None, None, 0)
    )
    remove_non_ancestors_no_adj = jax.vmap(
        jax.jit(remove_non_ancestors_no_adj), in_axes=(None, 0)
    )
    return remove_non_ancestors, remove_non_ancestors_no_adj


def make_activation_function(kernel, scales, has_clusters, memory_efficient):
    """
    Create an function that computes the activations based on the given kernel, scales, and other parameters.
    Significant speedups can be achieved when no cluster is present and the kernel is linear or quadratic.
    for the gaussian kernel, it is possible to use a memory-efficient computation, if vmap uses too much memory.


    Args:
        kernel: The kernel object used for computing activations.
        scales: A dictionary of scales for different modes.
        has_clusters: A boolean indicating whether the kernel has clusters.
        memory_efficient: A boolean indicating whether to use memory-efficient computation.

    Returns:
        The activation function.

    Raises:
        None.
    """
    if isinstance(kernel, kClass.LinearMode) and not has_clusters:

        def get_vecs(X, which_dim_only, yb):
            X_col = X[:, np.argmax(which_dim_only)]
            vec = X_col * yb
            vecsquared = X_col**2 * yb
            return vec, vecsquared

        get_vecs_vmap = jax.vmap(get_vecs, in_axes=(None, 0, None))

        def get_activations(X, yb, ga, active_modes):
            energy = -np.dot(ga, yb)
            vecs, _ = get_vecs_vmap(X, active_modes, yb)
            activations = kernel.scale * np.sum(vecs, axis=1) ** 2 / energy
            """activations = np.clip(activations, 0, None) / np.maximum(
                np.max(activations), 1
            )"""
            activations = np.where(np.all(active_modes == 0, axis=1), 2.0, activations)
            return activations

        return get_activations

    if isinstance(kernel, kClass.QuadraticMode) and not has_clusters:
        alpha = 0.5 * scales["linear"] / kernel.scale

        def get_vecs(X, which_dim_only, yb):
            X_col = X[:, np.argmax(which_dim_only)]
            vec = X_col * yb
            vecsquared = X_col**2 * yb
            return vec, vecsquared

        get_vecs_vmap = jax.vmap(get_vecs, in_axes=(None, 0, None))

        def get_activations(X, yb, ga, active_modes):
            energy = -np.dot(ga, yb)
            which_dim = np.sum(active_modes, axis=0)
            K_mat_all = 2 * (alpha + np.dot(X * which_dim[None, :], X.T))
            vecs, vecsquared = get_vecs_vmap(X, active_modes, yb)
            activations = (
                kernel.scale
                * (
                    np.einsum("ij,ni,nj->n", K_mat_all, vecs, vecs)
                    - np.sum(vecsquared, axis=1) ** 2
                )
                / energy
            )
            """activations = np.clip(activations, 0, None) / np.maximum(
                np.max(activations), 1
            )"""
            activations = np.where(np.all(active_modes == 0, axis=1), 2.0, activations)
            return activations

        return get_activations

    def activation(X, which_dim, which_dim_only, yb):
        mat = kernel.individual_influence(X, X, which_dim, which_dim_only)
        return np.dot(yb, mat @ yb)

    if not memory_efficient:
        activation_vmap = jax.vmap(activation, in_axes=(None, None, 0, None))
    else:

        def activation_vmap(X, which_dim, which_dim_only, yb):
            return jax.lax.map(
                lambda w: activation(X, which_dim, w, yb), which_dim_only
            )

    def get_activations(X, yb, ga, active_modes):
        energy = -np.dot(ga, yb)
        which_dim = np.sum(active_modes, axis=0)
        activations = activation_vmap(X, which_dim, active_modes, yb) / energy
        """activations = np.clip(activations, 0, None) / np.maximum(np.max(activations), 1)"""
        activations = np.where(np.all(active_modes == 0, axis=1), 2.0, activations)
        return activations

    return get_activations


def make_find_ancestor_function(
    kernel, scales, has_clusters, is_interpolatory=None, memory_efficient=False
):
    """
    Creates a function that finds the ancestor of a given input based on certain parameters.

    Args:
        kernel: The kernel function used for computing activations.
        scales: The scales between the different kernels.
        has_clusters: A boolean indicating whether the graph has clusters.
        is_interpolatory: A boolean indicating whether the kernel is interpolatory. If None, the value is determined based on the kernel.
        memory_efficient: A boolean indicating whether to use memory-efficient computations.

    Returns:
        A function that finds the ancestor of a given input.

    """
    get_activations_func = make_activation_function(
        kernel, scales, has_clusters, memory_efficient
    )
    interpolatory_bool = (
        kernel.is_interpolatory if is_interpolatory is None else is_interpolatory
    )

    if interpolatory_bool:
        perform_regression = interpolatory.perform_regression
    else:
        perform_regression = non_interpolatory.perform_regression

    def step(X, active_modes, ga, gamma, yb, key, gamma_min):
        activations = get_activations_func(X, yb, ga, active_modes)
        min_activation = np.argmin(activations)
        active_modes = active_modes.at[min_activation, :].set(0)
        mat = kernel(X, X, np.sum(active_modes, axis=0))
        yb, noise, Z_low, Z_high, gamma, key = perform_regression(
            K=mat, ga=ga, gamma=gamma, key=key, gamma_min=gamma_min
        )
        return (active_modes, yb, gamma, key, activations, noise, Z_low, Z_high)

    return jax.vmap(step, in_axes=(None, 0, 0, 0, 0, 0, 0))
