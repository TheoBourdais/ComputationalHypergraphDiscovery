import jax.numpy as np
import jax
from jax import random
from jax.scipy.optimize import minimize


def perform_regression(K, ga, gamma, gamma_min, key):
    """
    Perform regression using the given parameters.

    Args:
        K (int): The value of K.
        ga (float): The value of ga.
        gamma (float): The value of gamma.
        gamma_min (float): The minimum value of gamma.
        key (str): The key value.

    Returns:
        The result of performing regression and finding gamma.
    """
    return perform_non_interpolatory_regression_and_find_gamma(K=K, ga=ga, gamma_min=gamma_min, key=key)


def perform_non_interpolatory_regression_and_find_gamma(K, ga, gamma_min, key):
    """
    Perform a Kernel Ridge Regression on the given data using the kernel matrix K and the target values ga.

    If gamma is not provided, it is computed using the _find_gamma method with the given parameters.
    If gamma is provided, gamma_min, printer and interpolatory are ignored.

    The regression is performed by solving a variationnal problem using the cholesky factorization of K.

    Args:
    - K (np.ndarray): Kernel matrix of the chosen kernel.
    - gamma_min (float): The minimum value of gamma to consider.
    - ga (np.ndarry): The data of the node for which the ancestors are being pruned.
    - key (jax.random.PRNGKey): The random key to use for the Z-test.

    Returns:
    - yb (np.ndarray): The solution of the regression.
    - noise (float): The noise level of the regression.
    - (Z_low, Z_high) (tuple): The Z values of the regression.
    - gamma (float): The gamma value used for the regression.
    - key (jax.random.PRNGKey): The random key after the Z-test.

    """
    key, subkey = random.split(key)
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    gamma = find_gamma(
        eigenvalues=eigenvalues, gamma_min=gamma_min, Pga=np.dot(eigenvectors.T, ga)
    )
    gamma_used = jax.lax.max(gamma, gamma_min)

    yb, noise = solve_variationnal(
        ga, gamma=gamma_used, eigenvalues=eigenvalues, eigenvectors=eigenvectors
    )
    Z_low, Z_high = Z_test(gamma_used, eigenvalues, eigenvectors, subkey)
    return yb, noise, Z_low, Z_high, gamma, key


def solve_variationnal(ga, gamma, eigenvalues, eigenvectors):
    """
    Solve the variational problem yb = -K^-1@ga using the eigendecomposition of K.

    Parameters:
    ga (numpy.ndarray): The input vector ga.
    gamma (float): The value of gamma.
    eigenvalues (numpy.ndarray): The eigenvalues of matrix K.
    eigenvectors (numpy.ndarray): The eigenvectors of matrix K.

    Returns:
    yb (numpy.ndarray): The solution vector yb.
    noise (float): The noise term.
    """
    # solve yb = -K^-1@ga using the eigendecomposition of K
    Pga = np.dot(eigenvectors.T, ga)
    coeffs = gamma / (eigenvalues + gamma)
    Pgacoeff = Pga * coeffs
    noise = np.dot(Pgacoeff, Pgacoeff) / np.dot(Pgacoeff, Pga)
    yb = -np.dot(eigenvectors, Pga / (eigenvalues + gamma))

    return yb, noise


def Z_test(gamma, eigenvalues, eigenvectors, key):
    """
    Computes the Z-test for the given gamma and cho_factor.

    Args:
    - gamma (float): The gamma value.
    - eigenvalues (np.ndarray): The eigenvalues of the kernel matrix.
    - eigenvectors (np.ndarray): The eigenvectors of the kernel matrix.
    - key (jax.random.PRNGKey): The random key to use for the Z-test.

    Returns:
    - tuple: A tuple containing the 5th percentile and 95th percentile of the B_samples.
    """
    N = 100
    samples = random.normal(key, shape=(eigenvectors.shape[0], N))
    Pgas = np.dot(eigenvectors.T, samples)
    coeffs = gamma / (eigenvalues + gamma)
    Pgas_coeffs = Pgas * coeffs[:, None]
    noises = np.vecdot(Pgas_coeffs, Pgas_coeffs, axis=0) / np.vecdot(
        Pgas_coeffs, Pgas, axis=0
    )
    B_samples = np.sort(noises)
    return B_samples[int(0.05 * N)], B_samples[int(0.95 * N)]


def find_gamma(eigenvalues, Pga, gamma_min):
    """
    Finds the optimal value of gamma for a given set of eigenvalues.

    Parameters:
    - eigenvalues (array-like): The eigenvalues for which to find the optimal gamma.

    Returns:
    - gamma (float): The optimal value of gamma.

    """

    mean = np.nanmedian(np.where(eigenvalues < gamma_min, np.nan, eigenvalues))

    def loss(gamma):
        ratio = gamma[0] / (eigenvalues + gamma[0])
        return np.sum((Pga * ratio) ** 2) / np.sum(ratio * Pga**2)
        return np.sum((Pga * ratio) ** 2) / np.sum(ratio) ** 2

    test_vals = np.logspace(-6, 6, 10000)
    test = jax.vmap(loss)(test_vals[:, None])
    best_test = np.argmin(test)
    best_guess = np.where(best_test == 0, mean, test_vals[best_test])
    res = minimize(loss, best_guess[None], method="BFGS")
    best_guess = np.where(res.success, res.x[0], mean)
    return best_guess
