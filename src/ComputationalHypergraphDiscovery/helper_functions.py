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


def make_kernel_performance_function(kernels, gamma_min):
    is_interpolatory = np.array([kernel.is_interpolatory for kernel in kernels])

    # use jax.lax.cond to switch between _perform_regression_autogamma and _perform_regression_fixed_gamma
    def compute_kernel_mats(X, which_dim):
        res = []
        for kernel in kernels:
            res.append(kernel(X, X, which_dim=which_dim))
        return np.stack(res, axis=0)

    f_interpolatory = partial(
        interpolatory.perform_regression_and_find_gamma, gamma_min=gamma_min
    )
    f_non_interpolatory = partial(
        non_interpolatory.perform_regression_and_find_gamma, gamma_min=gamma_min
    )

    def choose_interpolatory(is_interpolatory, kernel_mat, ga, key):
        """Apply either f_interpolatory or f_non_interpolatory based on a condition."""
        return jax.lax.cond(
            is_interpolatory,
            lambda x: f_interpolatory(x, ga=ga, key=key),
            lambda x: f_non_interpolatory(x, ga=ga, key=key),
            kernel_mat,
        )

    mapped_choice = jax.vmap(
        jax.jit(choose_interpolatory, donate_argnums=(1,)), in_axes=(0, 0, None, None)
    )

    def kernel_performance_function(X, which_dim, ga, key):
        kernel_mats = compute_kernel_mats(X, which_dim)

        return mapped_choice(is_interpolatory, kernel_mats, ga, key)

    return kernel_performance_function


def make_activation_function(kernel, memory_efficient):

    def non_zero_activation(X, which_dim, which_dim_only, yb):
        mat = kernel.individual_influence(X, X, which_dim, which_dim_only)
        return np.dot(yb, mat @ yb)

    def zero_activation(X, which_dim, which_dim_only, yb):
        return 0.0

    def activation(X, which_dim, which_dim_only, yb):
        return jax.lax.cond(
            np.any(which_dim_only),
            non_zero_activation,
            zero_activation,
            X,
            which_dim,
            which_dim_only,
            yb,
        )

    if not memory_efficient:
        activation_vmap = jax.vmap(activation, in_axes=(None, None, 0, None))
    else:

        def activation_vmap(X, which_dim, which_dim_only, yb):
            return jax.lax.map(
                lambda w: activation(X, which_dim, w, yb), which_dim_only
            )

    """activation_vmap = jax.vmap(activation, in_axes=(None, 0, 0, 0))
    activation_vmap = jax.vmap(activation_vmap, in_axes=(None, None, 1, None))
"""
    """def get_activations(X, ybs, gas, active_modess):
        energies = -np.vecdot(gas, ybs)
        which_dim = np.sum(active_modess, axis=1)
        activations = activation_vmap(X, which_dim, active_modess, ybs) / energies
        return np.where(
            np.all(active_modess == 0, axis=2, keepdims=True), activations, 2.0
        )"""

    def get_activations(X, yb, ga, active_modes):
        energy = -np.dot(ga, yb)
        which_dim = np.sum(active_modes, axis=0)
        activations = activation_vmap(X, which_dim, active_modes, yb) / energy
        activations = np.clip(activations, 0, None) / np.maximum(np.max(activations), 1)
        return np.where(np.all(active_modes == 0, axis=1), 2.0, activations)

    return get_activations


def make_find_ancestor_function(kernel, gamma_min, memory_efficient=False):

    get_activations_func = make_activation_function(kernel, memory_efficient)
    if kernel.is_interpolatory:
        perform_regression = interpolatory.perform_regression
    else:
        perform_regression = non_interpolatory.perform_regression

    def step(X, active_modes, ga, gamma, yb, key):
        activations = get_activations_func(X, yb, ga, active_modes)
        min_activation = np.argmin(activations)
        active_modes = active_modes.at[min_activation, :].set(0)
        mat = kernel(X, X, np.sum(active_modes, axis=0))
        yb, noise, Z_low, Z_high, gamma, key = perform_regression(
            K=mat, ga=ga, gamma=gamma, key=key, gamma_min=gamma_min
        )
        return (active_modes, yb, key, activations, noise, Z_low, Z_high)

    return jax.vmap(step, in_axes=(None, 0, 0, 0, 0, 0))


def make_for_loop_function(get_activations_func, chosen_kernel, perform_regression):
    def f_for_loop(carry, x):
        active_modes, ga, gamma, yb, key, X, i = carry
        activations = get_activations_func(
            i=i, X=X, yb=yb, ga=ga, active_modes=active_modes
        )
        min_activation = np.argmin(activations)
        active_modes = active_modes.at[min_activation, :].set(0)
        mat = chosen_kernel(i, X, X, np.sum(active_modes, axis=0))
        yb, noise, Z_low, Z_high, gamma, key = perform_regression(
            i=i, K=mat, ga=ga, gamma=gamma, key=key
        )
        return (active_modes, ga, gamma, yb, key, X, i), (
            np.sum(active_modes, axis=0),
            activations,
            noise,
            Z_low,
            Z_high,
        )

    return f_for_loop


def make_regression_func(kernels, gamma_min):
    f_interpolatory = partial(interpolatory.perform_regression, gamma_min=gamma_min)
    f_non_interpolatory = partial(
        non_interpolatory.perform_regression, gamma_min=gamma_min
    )

    branch_functions = [lambda x, k=k: k.is_interpolatory for k in kernels]
    dummy_variable = -1
    interpolatory_func = lambda i: jax.lax.switch(i, branch_functions, dummy_variable)

    def perform_regression(i, K, ga, gamma, key):
        """Apply either f_interpolatory or f_non_interpolatory based on a condition."""
        return jax.lax.cond(
            interpolatory_func(i),
            lambda K, ga, gamma, key: f_interpolatory(K=K, ga=ga, gamma=gamma, key=key),
            lambda K, ga, gamma, keys: f_non_interpolatory(
                K=K, ga=ga, gamma=gamma, key=key
            ),
            K,
            ga,
            gamma,
            key,
        )

    return jax.jit(perform_regression, donate_argnums=(1,))
