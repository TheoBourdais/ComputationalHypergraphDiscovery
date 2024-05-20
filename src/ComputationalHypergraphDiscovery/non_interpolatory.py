import jax.numpy as np
import jax.scipy.linalg as jax_linalg
from jax import random
import jax


def perform_regression_and_find_gamma(K, ga, gamma_min, key):
    gamma = find_gamma(K=K, Y=ga, gamma_min=gamma_min)
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
    - printer (callable): A function to print the results.
    - interpolatory (bool): Whether the kernel is interpolatory.

    Returns:
    - yb (np.ndarray): The solution of the regression.
    - noise (float): The noise level of the regression.
    - (Z_low, Z_high) (tuple): The Z values of the regression.
    - gamma (float): The gamma value used for the regression.

    """
    cho_factor = jax_linalg.cho_factor(K + gamma * np.eye(K.shape[0]))

    yb, noise = solve_variationnal(ga=ga, gamma=gamma, cho_factor=cho_factor)
    Z_low, Z_high, key = Z_test(gamma=gamma, cho_factor=cho_factor, key=key)
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
    # solve yb = -K^-1@ga using the eigendeomposition of K
    yb = -jax_linalg.cho_solve(cho_factor, ga)

    noise = -gamma * np.dot(yb, yb) / np.dot(ga, yb)
    return yb, noise


def Z_test(gamma, cho_factor, key):
    """
    Computes the Z-test for the given gamma and cho_factor.

    Args:
    - gamma (float): The gamma value.
    - cho_factor (tuple): The cho_factor tuple.

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


def find_gamma(K, Y, gamma_min):
    """
    Finds the gamma value for regression problem as the residuals of the regression problem

    Args:
    - K (numpy.ndarray): The kernel matrix.
    - interpolatory (bool): Whether the kernel is interpolatory or not.
    - Y (numpy.ndarray): The target vector, also called ga.
    - gamma_min (float): The minimum value of gamma.
    - printer (function): The function to print messages.
    - tol (float): The tolerance value for the optimisation algorithm.

    Returns:
    - float: The gamma value.
    """
    residuals = np.linalg.lstsq(K, Y, rcond=None)[1]
    gamma = np.sum(residuals)
    gamma = jax.lax.max(gamma, gamma_min)

    return gamma
