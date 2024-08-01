import jax.numpy as np
import jax.scipy.linalg as jax_linalg
from jax import random
import jax


def perform_regression_and_find_gamma(K, ga, gamma_min, key):
    """
    Performs regression and finds the optimal gamma value.

    Args:
        K (array-like): The input data.
        ga (array-like): The target values.
        gamma_min (float): The minimum value of gamma to consider.
        key (str): The key for the regression.

    Returns:
        The result of the regression with the optimal gamma value.
    """
    gamma = find_gamma(K=K, Y=ga)
    return perform_regression(K=K, ga=ga, gamma=gamma, gamma_min=gamma_min, key=key)


def perform_regression(K, ga, gamma, gamma_min, key):
    """
    Perform a Kernel Ridge Regression on the given data using the kernel matrix K and the target values ga.

    If gamma is not provided, it is computed using the _find_gamma method with the given parameters.
    If gamma is provided, gamma_min, printer and interpolatory are ignored.

    The regression is performed by solving a variationnal problem using the cholesky factorization of K.

    Args:
    - K (np.ndarray): Kernel matrix of the chosen kernel.
    - gamma (float or str): The regularization parameter for the regression. If "auto", it will be automatically determined.
    - gamma_min (float): The minimum value of gamma to consider.
    - ga (np.ndarry): The data of the node for which the ancestors are being pruned.
    - key (jax.random.PRNGKey): The random key to use for the Z-test.

    Returns:
    - yb (np.ndarray): The solution of the regression.
    - noise (float): The noise level of the regression.
    - (Z_low, Z_high) (tuple): The Z values of the regression.
    - gamma (float): The gamma value used for the regression.

    """
    gamma_used = jax.lax.max(gamma, gamma_min)
    cho_factor = jax_linalg.cho_factor(K + gamma_used * np.eye(K.shape[0]))

    yb, noise = solve_variationnal(ga=ga, gamma=gamma_used, cho_factor=cho_factor)
    Z_low, Z_high, key = Z_test(gamma=gamma_used, cho_factor=cho_factor, key=key)
    return yb, noise, Z_low, Z_high, gamma, key


def solve_variationnal(ga, gamma, cho_factor):
    """
    Solves a yb=-K^-1@ga  problem using the Cholesky factorization of K, and gives noise

    Args:
    - ga(np.ndarray): A numpy array representing the input matrix.
    - gamma(float): A float representing the gamma value.
    - cho_factor(tuple): output of scipy.linalg.cho_factor(K)

    Returns:
    - yb(np.ndarray): A numpy array representing the solution to the variationnal problem.
    - noise(float): A float representing the noise value.
    """
    yb = -jax_linalg.cho_solve(cho_factor, ga)

    noise = -gamma * np.dot(yb, yb) / np.dot(ga, yb)
    return yb, noise


def Z_test(gamma, cho_factor, key):
    """
    Computes the Z-test for the given gamma and cho_factor.

    Args:
    - gamma (float): The gamma value.
    - cho_factor (tuple): The cho_factor tuple.
    - key (jax.random.PRNGKey): The random key to use for

    Returns:
    - tuple: A tuple containing the 5th percentile and 95th percentile of the B_samples.
    """
    N = 100
    key, subkey = random.split(key)
    samples = random.normal(subkey, shape=(cho_factor[0].shape[0], N))
    yb_samples = -jax_linalg.cho_solve(cho_factor, samples)
    norms = np.linalg.norm(yb_samples, axis=0) ** 2
    inner_products = np.einsum("ij,ij->j", samples, yb_samples)
    B_samples = np.sort(-gamma * norms / inner_products)
    return B_samples[int(0.05 * N)], B_samples[int(0.95 * N)], key


def find_gamma(K, Y):
    """
    Finds the gamma value for regression problem as the residuals of the regression problem

    Args:
    - K (numpy.ndarray): The kernel matrix.
    - Y (numpy.ndarray): The target vector, also called ga.

    Returns:
    - float: The gamma value.
    """
    K_reg = (
        K + np.eye(K.shape[0]) * np.finfo("float64").eps
    )  # in rare edge cases, SVD fails without this
    residuals = np.linalg.lstsq(K_reg, Y, rcond=None)[1]
    gamma = np.sum(residuals)

    return gamma
