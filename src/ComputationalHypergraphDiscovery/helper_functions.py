import jax
import jax.numpy as np
from functools import partial
from . import interpolatory
from . import non_interpolatory
from .Modes import kernels as kClass


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

    mapped_choice = jax.vmap(choose_interpolatory, in_axes=(0, 0, None, None))

    def kernel_performance_function(X, which_dim, ga, key):
        kernel_mats = compute_kernel_mats(X, which_dim)

        return mapped_choice(is_interpolatory, kernel_mats, ga, key)

    return kernel_performance_function


def make_prune_ancestors(kernels, loop_number, gamma_min):
    chosen_kernel = lambda i, *args: jax.lax.switch(i, kernels, *args)
    get_activations = make_activation_functions(kernels)
    perform_regression = make_regression_func(kernels, gamma_min)
    f_for_loop = make_for_loop_function(
        get_activations, chosen_kernel, perform_regression
    )

    def prune_ancestors(i, X, active_modes, noise, Z, ga, gamma, yb, key):
        init = (active_modes, ga, gamma, yb, key, X, i)
        carry, (ancestor_modes, activations, noises, Z_low, Z_high) = jax.lax.scan(
            f=f_for_loop, init=init, length=loop_number
        )
        # key = carry[4]
        active_modes_end = carry[0]
        ancestor_modes = np.concatenate(
            [np.sum(active_modes, axis=0, keepdims=True), ancestor_modes], axis=0
        )
        Z_low = np.concatenate([np.array([Z[0]]), Z_low], axis=0)
        Z_high = np.concatenate([np.array([Z[1]]), Z_high], axis=0)
        noises = np.concatenate([np.array([noise]), noises], axis=0)
        activations = np.concatenate(
            [
                activations,
                (1 - noises[-1]) * (np.sum(active_modes_end, axis=1) > 0)[None, :],
            ],
            axis=0,
        )
        return ancestor_modes, noises, Z_low, Z_high, activations

    return prune_ancestors


def make_activation_functions(kernels):
    def apply_influence(*args, kernel):
        return kernel.individual_influence(*args)

    kernels_only_var = [partial(apply_influence, kernel=kernel) for kernel in kernels]
    chosen_kernel_only_var = lambda i, *args: jax.lax.switch(i, kernels_only_var, *args)
    all_chosen_kernels = jax.vmap(
        chosen_kernel_only_var, in_axes=(0, None, None, None, None)
    )

    def activation_no_zero(i, X, which_dim, which_dim_only, yb, energy):

        mat = chosen_kernel_only_var(i, X, X, which_dim, which_dim_only)
        mats = all_chosen_kernels(np.array([0, 1, 2]), X, X, which_dim, which_dim_only)
        return np.dot(yb, mat @ yb) / energy

    def activation_zero(i, X, which_dim, which_dim_only, yb, energy):
        return 2.0

    def activation(i, X, which_dim, which_dim_only, yb, energy):
        is_zero = np.all(which_dim_only == 0)
        return jax.lax.cond(
            is_zero,
            activation_zero,
            activation_no_zero,
            i,
            X,
            which_dim,
            which_dim_only,
            yb,
            energy,
        )

    activation_vmap = jax.vmap(activation, in_axes=(None, None, None, 0, None, None))

    def get_activations(i, X, yb, ga, active_modes):
        energy = -np.dot(ga, yb)
        which_dim = np.sum(active_modes, axis=0)
        activations = activation_vmap(
            i,
            X,
            which_dim,
            active_modes,
            yb,
            energy,
        )
        return activations

    return get_activations


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

    return perform_regression


def make_find_ancestor_function(
    kernel_performance_function,
    prune_ancestors,
    kernel_chooser,
    mode_chooser,
    loop_number,
):

    def prune_and_choose_ancestor(
        chosen_kernel, X, active_modes, noise, Z_low, Z_high, ga, gamma, yb, key
    ):
        ancestor_modes, noises, Z_low, Z_high, activations = prune_ancestors(
            i=chosen_kernel,
            X=X,
            active_modes=active_modes,
            noise=noise,
            Z=(Z_low, Z_high),
            ga=ga,
            gamma=gamma,
            yb=yb,
            key=key,
        )
        chosen_mode = mode_chooser(ancestor_modes, noises, (Z_low, Z_high))
        return (
            chosen_kernel,
            chosen_mode,
            ancestor_modes,
            noises,
            Z_low,
            Z_high,
            activations,
        )

    def exit_func(i, X, active_modes, noise, Z_low, Z_high, ga, gamma, yb, key):
        placeholder_ancestors = np.zeros((loop_number + 1, active_modes.shape[1]))
        place_holder_choice = -1
        placeholder_noises = np.zeros(loop_number + 1)
        placeholder_Z_low = np.zeros(loop_number + 1)
        placeholder_Z_high = np.zeros(loop_number + 1)
        placeholder_activations = np.zeros((loop_number + 1, active_modes.shape[0]))
        return (
            i,
            place_holder_choice,
            placeholder_ancestors,
            placeholder_noises,
            placeholder_Z_low,
            placeholder_Z_high,
            placeholder_activations,
        )

    def find_ancestor(X, active_modes, ga, key):
        kernel_performance = kernel_performance_function(
            X=X, which_dim=np.sum(active_modes, axis=0), ga=ga, key=key
        )

        which_kernel, yb, noise, Z_low, Z_high, gamma, key = kernel_chooser(
            kernel_performance
        )
        return (
            *jax.lax.cond(
                which_kernel == -1,
                exit_func,
                prune_and_choose_ancestor,
                which_kernel,
                X,
                active_modes,
                noise,
                Z_low,
                Z_high,
                ga,
                gamma,
                yb,
                key,
            ),
            kernel_performance[1:-1],
        )

    return find_ancestor
