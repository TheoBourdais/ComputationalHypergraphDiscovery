import functools
import jax.numpy as jnp
import jax
from copy import copy


class ModeKernel:
    """
    Represents a mode kernel used in computational hypergraph discovery.
    """

    def __init__(self) -> None:
        self._scale = 1.0
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

    @property
    def memory_efficient_required(self):
        return self._memory_efficient_required

    @property
    def scale(self):
        return self._scale

    def __repr__(self) -> str:
        return self.name

    def individual_influence(self, X, Y, which_dim, which_dim_only):
        """
        Calculates the individual influence of a given dimension on the kernel value.
        This is the default way, allows to compute activations for any kernel but is slower in general.

        Parameters:
        - X: The input data.
        - Y: The target data.
        - which_dim: The dimension to calculate the influence for.
        - which_dim_only: A binary value indicating whether to consider only the specified dimension.

        Returns:
        The individual influence of the specified dimension on the kernel value.
        """
        whole_k = self(X, Y, which_dim)
        rest = which_dim * (1 - which_dim_only)
        rest_k = self(X, Y, rest)
        return whole_k - rest_k

    def __call__(self, X, Y, which_dim):
        return self.kernel(X, Y, which_dim)

    def __mul__(self, other):
        if isinstance(other, float):
            assert other > 0
            self._scale = other
            return self

    def __rmul__(self, other):
        return self.__mul__(other)


class LinearMode(ModeKernel):
    """
    Linear mode kernel implementation.

    Args:
    - memory_efficient_required (bool): Flag indicating whether memory-efficient mode is required. Default is False.

    Attributes:
    - hyperparameters (dict): Dictionary to store hyperparameters.
    - _is_interpolatory (bool): Flag indicating whether the mode is interpolatory.
    - _memory_efficient_required (bool): Flag indicating whether memory-efficient mode is required.
    - name (str): Name of the mode.
    - _scale (float): Scaling factor for the kernel.

    Methods:
    - setup(X, scales): Set up the kernel with the given data and scales.
    - individual_influence(X, Y, which_dim, which_dim_only): Compute the influence of each individual data point on the prediction.

    """

    def __init__(self, memory_efficient_required=False) -> None:
        super().__init__()
        self.hyperparameters = {}
        self._is_interpolatory = False
        self._memory_efficient_required = memory_efficient_required
        self.name = "linear"
        self._scale = 2.0

    @staticmethod
    def vectorized_kernel(scale, X, Y, which_dim):
        # factor of 2 is added for consistency with the quadratic kernel
        return 1 + scale * jnp.dot(X * which_dim[None, :], Y.T)

    @staticmethod
    def kernel_only_var_fn(mode: "LinearMode", X, Y, which_dim, which_dim_only):
        return mode.kernel(X, Y, which_dim_only) - 1

    def setup(self, X, scales):
        """
        Set up the kernel with the given data and scales.

        Args:
        - X (np.array): The data matrix.
        - scales (dict): Dictionary of scales for different modes.

        """
        assert self.scale == scales[self.name]

        self.kernel = functools.partial(LinearMode.vectorized_kernel, self.scale)

        self.kernel_only_var = functools.partial(LinearMode.kernel_only_var_fn, self)

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
    QuadraticMode is a class that represents a quadratic mode kernel.

    Attributes:
    - _is_interpolatory (bool): Indicates whether the mode kernel is interpolatory.
    - name (str): The name of the mode kernel.
    - _memory_efficient_required (bool): Indicates whether memory efficiency is required.

    Methods:
    - __init__(self, memory_efficient_required=False): Initializes a new instance of the QuadraticMode class.
    - setup(self, X, scales): Sets up the mode kernel with the given data and scales.
    - individual_influence(self, X, Y, which_dim, which_dim_only): Computes the influence of each individual data point on the prediction.

    """

    def __init__(self, memory_efficient_required=False) -> None:
        """
        Initializes a new instance of the QuadraticMode class.

        Args:
        - memory_efficient_required (bool): Indicates whether memory efficiency is required.

        """
        super().__init__()
        self._is_interpolatory = False
        self.name = "quadratic"
        self._memory_efficient_required = memory_efficient_required

    @staticmethod
    def vectorized_kernel(scale, alpha, X, Y, which_dim):
        return scale * (alpha + jnp.dot(X * which_dim[None, :], Y.T)) ** 2 + (
            1 - alpha**2 * scale
        )

    @staticmethod
    def vectorized_kernel_only_var(scale, alpha, X, Y, which_dim, which_dim_only):
        linear_only = jnp.dot(X * which_dim_only[None, :], Y.T)
        rest = which_dim * (1 - which_dim_only)
        linear_rest = jnp.dot(X * rest[None, :], Y.T)

        return (
            scale * (alpha + linear_only) ** 2
            + 2 * scale * linear_only * linear_rest
            - scale * alpha**2
        )

    def setup(self, X, scales):
        """
        Sets up the mode kernel with the given data and scales.

        Args:
        - X (np.array): The data matrix.
        - scales (dict): The scales dictionary.

        """
        assert self.scale == scales[self.name]
        try:
            scales["linear"]
        except KeyError:
            scales["linear"] = 2.0
        alpha = 0.5 * scales["linear"] / self.scale

        self.kernel = functools.partial(
            QuadraticMode.vectorized_kernel, self.scale, alpha
        )

        self.kernel_only_var = functools.partial(
            QuadraticMode.vectorized_kernel_only_var, self.scale, alpha
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


class GaussianMode(ModeKernel):
    """
    Gaussian mode kernel implementation.

    Args:
    - l (float): Length scale parameter.
    - memory_efficient_required (bool): Flag indicating whether memory efficiency is required. Default is True.
    """

    def __init__(self, l, memory_efficient_required=True) -> None:
        super().__init__()
        self._l = l
        self._is_interpolatory = True
        self._memory_efficient_required = memory_efficient_required
        self.name = "gaussian"

        self.quadratic_part = QuadraticMode()

    @staticmethod
    def gaussian_exp(self: "GaussianMode", x, y):
        return jnp.exp(-((x - y) ** 2) / (2 * self.l**2))

    @staticmethod
    def k(mode: "GaussianMode", X, Y, which_dim):
        return (
            mode.scale * jnp.prod(1 + which_dim[None, None, :] * mode.exps, axis=2)
            + mode.quadratic_part(X, Y, which_dim)
            - 1
        )

    @staticmethod
    def k_recompute(mode: "GaussianMode", X, Y, which_dim):
        exps = jax.vmap(
            jax.vmap(mode.gaussian_exp, in_axes=(0, None)), in_axes=(None, 0)
        )(X, Y)
        return (
            mode.scale * jnp.prod(1 + which_dim[None, None, :] * exps, axis=2)
            + mode.quadratic_part(X, Y, which_dim)
            - 1
        )

    @staticmethod
    def k_only_var(mode: "GaussianMode", X, Y, which_dim, which_dim_only):
        rest = which_dim * (1 - which_dim_only)
        only_part = jnp.prod(1 + which_dim_only[None, None, :] * mode.exps, axis=2) - 1
        rest_part = jnp.prod(1 + rest[None, None, :] * mode.exps, axis=2)
        return (
            mode.quadratic_part.individual_influence(X, Y, which_dim, which_dim_only)
            + mode.scale * only_part * rest_part
        )

    @staticmethod
    def k_only_var_recompute(mode: "GaussianMode", X, Y, which_dim, which_dim_only):
        exps = jax.vmap(
            jax.vmap(mode.gaussian_exp, in_axes=(0, None)), in_axes=(None, 0)
        )(X, Y)
        rest = which_dim * (1 - which_dim_only)
        only_part = jnp.prod(1 + which_dim_only[None, None, :] * exps, axis=2) - 1
        rest_part = jnp.prod(1 + rest[None, None, :] * exps, axis=2)
        return (
            mode.quadratic_part.individual_influence(X, Y, which_dim, which_dim_only)
            + mode.scale * only_part * rest_part
        )

    def setup(self, X, scales):
        """
        Setup the Gaussian mode kernel.

        Args:
        - X (np.array): The data matrix.
        - scales (dict): Dictionary of scales for different kernel components.
        """
        assert self.scale == scales[self.name]
        self.quadratic_part._scale = scales["quadratic"]
        self.quadratic_part.setup(X, scales)

        self.gaussian_exp = functools.partial(GaussianMode.gaussian_exp, self)

        self.exps = jax.vmap(
            jax.vmap(self.gaussian_exp, in_axes=(0, None)), in_axes=(None, 0)
        )(X, X)

        self.kernel = functools.partial(GaussianMode.k, self)
        self.kernel_recompute = functools.partial(GaussianMode.k_recompute, self)

        self.kernel_only_var = functools.partial(GaussianMode.k_only_var, self)
        self.kernel_only_var_recompute = functools.partial(
            GaussianMode.k_only_var_recompute, self
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
