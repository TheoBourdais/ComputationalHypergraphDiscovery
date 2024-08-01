import jax
import jax.numpy as np
from functools import partial
from . import interpolatory
from . import non_interpolatory
from .Modes import kernels as kClass


def make_preprocessing_functions():
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
