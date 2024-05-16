import jax.numpy as jnp
import jax
from copy import copy


class ModeKernel:
    """
    An interface representing a kernel for a mode in a hypergraph.
    This is an abstract class that ressembles Sklearn's kernel interface, but also implements the methods necessary for kernel mode decomposition.

    Attributes:
    -----------
    - beta_scale (float): The scaling factor for the kernel. See the help of ModeKernel.beta_scale for more details.
    - is_interpolatory (bool): Whether the kernel is interpolatory or not. See the help of ModeKernel.is_interpolatory for more details.
    - mode_type (str): The type of the mode kernel. Can be "individual", "pairwise" or "combinatorial". See the help of ModeKernel.mode_type for more details.
    """

    def __init__(self) -> None:
        pass

    @property
    def is_interpolatory(self):
        """Returns a boolean indicating whether the kernel is interpolatory.
        This is used to know how to compute the noise parameter gamma in GraphDiscovery.
        A kernel will be interpolatory if it's associated feature map has more dimensions than the number of data points.
        For example, the Gaussian kernel is interpolatory (its feature map is infinite dimensional),
        while the linear kernel will probably not be (if the input is of dimension d, the number of data points is n, if n>d the kernel is not interpolatory).
        """
        res = self._is_interpolatory
        assert isinstance(res, bool)
        return res

    def __repr__(self) -> str:
        return self.name

    def individual_influence(self, X, Y, which_dim, which_dim_only):
        whole_k = self(X, Y, which_dim)
        rest = which_dim * (1 - which_dim_only)
        rest_k = self(X, Y, rest)
        return whole_k - rest_k

    def __call__(self, X, Y, which_dim):
        return self.kernel(X, Y, which_dim)


class LinearMode(ModeKernel):
    """
    Linear mode kernel.
    This kernel is not interpolatory, and is applied to each column of the data matrix independently (individual type).

    Args:
    - name (str, optional): Name of the kernel. Defaults to None.
    """

    def __init__(self, name=None) -> None:
        self.hyperparameters = {}
        self._is_interpolatory = False
        self.name = name if name is not None else "linear"
        """def kernel(x,y,which_dim):
            return 1+np.dot(x*which_dim,y)
        #vmap kernel along x and y
        self.kernel = jax.vmap(jax.vmap(kernel, in_axes=(None, 0, None)), in_axes=(0, None, None))"""

        def vectorized_kernel(X, Y, which_dim):
            # factor of 2 is added for consistency with the quadratic kernel
            return 1 + 2 * jnp.dot(X * which_dim[None, :], Y.T)

        self.kernel = vectorized_kernel
        self.kernel_only_var = (
            lambda X, Y, which_dim, which_dim_only: self.kernel(X, Y, which_dim_only)
            - 1
        )

    def individual_influence(self, X, Y, which_dim, which_dim_only):
        """
        Compute the influence of each individual data point on the prediction.
        This is done by computing the kernel matrix with the data matrix, and then taking the dot product with the target vector.

        Args:
        - X (np.array): The data matrix.
        - Y (np.array): The target vector.
        - which_dim (np.array): The dimension of the data matrix.
        - which_dim_only (int): The index of the data point to compute the influence for.

        Returns:
        - np.array: The influence of each data point on the prediction.
        """
        return self.kernel_only_var(X, Y, which_dim, which_dim_only)


class QuadraticMode(ModeKernel):
    """
    Quadratic mode kernel.
    This kernel is not interpolatory, and is applied to each pairs of column of the data matrix (pairwise type).

    Args:
    - name (str, optional): Name of the kernel. Defaults to None.
    """

    def __init__(self, name=None) -> None:
        self._is_interpolatory = False
        self.name = name if name is not None else "quadratic"
        """def kernel(x,y,which_dim):
            return (1+np.dot(x*which_dim,y))**2
        def kernel_only_var(x,y,which_dim,which_dim_only):
            return (1+np.dot(x*which_dim,y))*(1+np.dot(x*which_dim_only,y))

        #vmap kernel along x and y
        self.kernel = jax.vmap(jax.vmap(kernel, in_axes=(None, 0, None)), in_axes=(0, None, None))
        self.kernel_only_var=jax.vmap(jax.vmap(kernel_only_var, in_axes=(None, 0, None, None)), in_axes=(0, None, None, None))
"""

        def vectorized_kernel(X, Y, which_dim):
            return (1 + jnp.dot(X * which_dim[None, :], Y.T)) ** 2

        def vectorized_kernel_only_var(X, Y, which_dim, which_dim_only):
            linear_only = jnp.dot(X * which_dim_only[None, :], Y.T)
            rest = which_dim * (1 - which_dim_only)
            linear_rest = jnp.dot(X * rest[None, :], Y.T)
            return (1 + linear_only) ** 2 + 2 * linear_only * linear_rest - 1

        self.kernel = vectorized_kernel
        self.kernel_only_var = vectorized_kernel_only_var

    def individual_influence(self, X, Y, which_dim, which_dim_only):
        """
        Compute the influence of each individual data point on the prediction.
        This is done by computing the kernel matrix with the data matrix, and then taking the dot product with the target vector.

        Args:
        - X (np.array): The data matrix.
        - Y (np.array): The target vector.
        - which_dim (np.array): The dimension of the data matrix.
        - which_dim_only (int): The index of the data point to compute the influence for.

        Returns:
        - np.array: The influence of each data point on the prediction.
        """
        return self.kernel_only_var(X, Y, which_dim, which_dim_only)


class GaussianMode(ModeKernel):
    """
    Gaussian mode kernel.
    This kernel is interpolatory, and of the combinatorial type, i.e. is applied to each columns of the data matrix,
    after that for each subset of the columns, we take the product of the resulting kernel matrix.
    This is then summed over all subsets, allowing to capture every possible combination of interactions between the columns.

    Parameters:
    - l (float): Length scale parameter.
    - name (str, optional): Name of the kernel.

    """

    def __init__(self, l, name=None) -> None:
        self._l = l
        self._is_interpolatory = True
        self.name = name if name is not None else "gaussian"

        def k(x, y, which_dim):
            exps = 1 + which_dim * jnp.exp(-((x - y) ** 2) / (2 * l**2))
            return (1 + jnp.dot(x * which_dim, y)) ** 2 + jnp.prod(exps)

        self.kernel = jax.vmap(
            jax.vmap(k, in_axes=(None, 0, None)), in_axes=(0, None, None)
        )

        def k_only_var(x, y, which_dim, which_dim_only):
            exps = jnp.exp(-((x - y) ** 2) / (2 * l**2))
            rest = which_dim * (1 - which_dim_only)
            exps_only = jnp.where(which_dim_only == 1, exps, 1)
            exps_rest = 1 + rest * exps

            linear_only = jnp.dot(x * which_dim_only, y)
            linear_rest = jnp.dot(x * rest, y)
            quadratic = (1 + linear_only) ** 2 + 2 * linear_only * linear_rest - 1
            return quadratic + jnp.prod(exps_only) * jnp.prod(exps_rest)

        self.kernel_only_var = jax.vmap(
            jax.vmap(k_only_var, in_axes=(None, 0, None, None)),
            in_axes=(0, None, None, None),
        )

    def individual_influence(self, X, Y, which_dim, which_dim_only):
        """
        Compute the influence of each individual data point on the prediction.
        This is done by computing the kernel matrix with the data matrix, and then taking the dot product with the target vector.

        Args:
        - X (np.array): The data matrix.
        - Y (np.array): The target vector.
        - which_dim (np.array): The dimension of the data matrix.
        - which_dim_only (int): The index of the data point to compute the influence for.

        Returns:
        - np.array: The influence of each data point on the prediction.
        """
        return self.kernel_only_var(X, Y, which_dim, which_dim_only)

    @property
    def l(self):
        return self._l

    @l.setter
    def l(self, value):
        raise AttributeError(
            "Length scale cannot be changed due to JAX JIT compilation"
        )
