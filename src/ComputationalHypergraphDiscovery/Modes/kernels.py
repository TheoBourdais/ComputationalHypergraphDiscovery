import numpy as np
from copy import copy


class ModeKernelList:
    """
    A list of ModeKernels. Allows to handle kernels defined as a sum of kernels.

    Attributes:
    - modes (list): A list of ModeKernels or ModeKernelLists.
    - names (list): A list of the names of the modes.
    - name_to_index (dict): A dictionary mapping mode names to their indices in the list.
    """

    def __init__(self, *args) -> None:
        """
        Initializes a ModeKernelList object.

        Args:
        - *args: Variable length argument list of ModeKernel or ModeKernelList objects.

        Raises:
        - AssertionError: If the input arguments are not ModeKernel or ModeKernelList objects.
        - AssertionError: If the ModeKernelList contains two modes with the same name.
        """
        self.modes = []
        for arg in args:
            assigned = False
            # check if arg is a ModeKernel or a ModeKernelList, else raise an error
            try:
                arg.is_a_ModeKernel()
                self.modes.append(arg)
                assigned = True
            except AttributeError:
                pass
            try:
                arg.is_a_ModeKernelList()
                self.modes.extend(arg.modes)
                assigned = True
            except AttributeError:
                pass
            assert (
                assigned
            ), "ModeKernelList can only be built from ModeKernels or ModeKernelLists"
        self.names = [mode.name for mode in self.modes]
        assert len(self.names) == len(
            set(self.names)
        ), "ModeKernelList cannot contain two modes with the same name"
        self.name_to_index = {name: i for i, name in enumerate(self.names)}

    def __getitem__(self, key):
        """
        Get the mode kernel with the given key.

        Args:
        - key (str or int): The key to use for indexing the mode kernel.

        Returns:
        - The mode kernel with the given key.

        Raises:
        - Exception: If the key is not a string or an integer.
        """
        if isinstance(key, str):
            return self.modes[self.name_to_index[key]]
        if isinstance(key, int):
            return self.modes[key]
        raise Exception("ModeKernelList can only be indexed with strings or integers")

    def __repr__(self) -> str:
        """
        Returns the names of the kernel in the list.
        """
        return self.names.__repr__()

    def __add__(self, other):
        """
        Returns a new ModeKernelList object that contains both self and other ModeKernel objects.
        This addition is not commutative, and ressembles list concatenation.
        """
        return ModeKernelList(self, other)

    def __mul__(self, other):
        """
        Multiply a ModeKernelList object with a float. Each mode in the list is multiplied by the float.

        Parameters:
        - other (float): The object to multiply with.

        Returns:
        ModeKernelList: A new ModeKernelList object resulting from the multiplication.
        """
        return ModeKernelList(*[mode * other for mode in self.modes])

    def __rmul__(self, other):
        """
        Right multiplication of a kernel by a scalar. Same as left multiplication.
        """
        return self.__mul__(other)

    def __call__(self, X: np.array) -> np.array:
        """
        Compute the kernel matrices for the given data matrix X.

        Args
        - X (np.array): The data matrix.

        Returns:
        np.array: The kernel matrix.
        """
        return [mode(X) for mode in self.modes]

    def is_a_ModeKernelList(self):
        return True


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
    def beta_scale(self):
        """
        Returns the beta scale hyperparameter value.
        If our kernel was to return a matrix K when beta_scale=1, then it returns beta_scale*K when beta_scale is different from 1.
        """
        return self.hyperparameters.get("beta_scale", 1)

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
    def mode_type(self):
        """Returns the type of mode for the kernel.

        Our kernels are formed of 1D kernels applied to sets of columns of the data matrix.
        Three modes are available:
        - individual: The kernel is applied to each column of the data matrix independently.
        - pairwise: The kernel is applied to each pair of columns of the data matrix.
        - combinatorial: The kernel is applied to each  columns of the data matrix,
            and for each subset of the columns, we take the product of the resulting kernel matrix.
            This is then summed over all subsets, allowing to capture every possible combination of interactions between the columns.

        While we could imagine applying the kernels to triplets of modes, this is not implemented.

        Returns:
            str: The type of mode for the kernel. Can be "individual", "pairwise", or "combinatorial".
        """
        res = self._mode_type
        assert res in ["individual", "pairwise", "combinatorial"]
        return res

    def __mul__(self, other):
        """
        Multiply the kernel by a scalar. This is used to scale the beta_scale hyperparameter.

        Args:
        - other (float or int): The scalar to multiply the kernel by.

        Returns:
        A new kernel object with the beta_scale hyperparameter scaled by the scalar.
        """
        assert isinstance(other, float) or isinstance(
            other, int
        ), "multiplication only defined with scalars"
        assert other >= 0, "multiplication only defined with positive scalars"
        new_kernel = copy(self)
        new_kernel.hyperparameters = new_kernel.hyperparameters.copy()
        new_kernel.hyperparameters["beta_scale"] = self.beta_scale * other
        return new_kernel

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        """
        Returns a new ModeKernelList object that contains both self and other.
        """
        return ModeKernelList(self, other)

    def __repr__(self) -> str:
        return self.name

    def is_a_ModeKernel(self):
        return True


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
        self._mode_type = "individual"

    def __call__(self, X: np.array) -> np.array:
        assert (
            len(X.shape) == 2
        ), "Linear kernel is only available for X as a stack of vectors"
        return np.expand_dims(X, -1) * np.expand_dims(X, 1)


class QuadraticMode(ModeKernel):
    """
    Quadratic mode kernel.
    This kernel is not interpolatory, and is applied to each pairs of column of the data matrix (pairwise type).

    Args:
    - name (str, optional): Name of the kernel. Defaults to None.
    """

    def __init__(self, name=None) -> None:
        self.hyperparameters = {}
        self._is_interpolatory = False
        self.name = name if name is not None else "quadratic"
        self._mode_type = "pairwise"

    def __call__(self, X: np.array) -> np.array:
        assert (
            len(X.shape) == 2
        ), "Quadratic kernel is only available for X as a stack of vectors"
        linear_mat = np.expand_dims(X, -1) * np.expand_dims(X, 1)
        quadratic_mat = np.expand_dims(linear_mat, 0) * np.expand_dims(linear_mat, 1)
        return quadratic_mat


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
        self.hyperparameters = {"l": l}
        self._is_interpolatory = True
        self.name = name if name is not None else "gaussian"
        self._mode_type = "combinatorial"

    def __call__(self, X: np.array) -> np.array:
        assert (
            len(X.shape) == 2
        ), "Gaussian kernel is only available for X as a stack of vectors"
        diff_X = np.tile(np.expand_dims(X, -1), (1, 1, X.shape[1])) - np.tile(
            np.expand_dims(X, 1), (1, X.shape[1], 1)
        )
        return np.exp(-((diff_X / self.hyperparameters["l"]) ** 2) / 2)


class SklearnMode(ModeKernel):
    """
    A mode kernel that applies a Scikit-learn kernel to the columns of a matrix.
    The user must specify the type of mode kernel (individual, pairwise, or combinatorial) and whether it is interpolatory or not.

    Parameters
    - kernel (callable): A Scikit-learn kernel function that takes a 2D array as input.
    - mode_type (str): The type of mode kernel to compute. Must be one of "individual", "combinatorial", or "pairwise".
    - is_interpolatory (bool): Whether the mode kernel is interpolatory or not. See the help of ModeKernel.is_interpolatory for more details.
    - name (str, optional): The name of the mode kernel. If not provided, defaults to "sklearn_kernel".

    """

    def __init__(self, kernel, mode_type, is_interpolatory, name=None) -> None:
        self.hyperparameters = {"scipy_kernel": kernel}
        self._is_interpolatory = is_interpolatory
        self.name = name if name is not None else "sklearn_kernel"
        self._mode_type = mode_type

    def __call__(self, X: np.array) -> np.array:
        if self.mode_type in ["individual", "combinatorial"]:
            res = []
            for col in X:
                if len(col.shape) > 1:
                    res.append(self.hyperparameters["scipy_kernel"](col))
                res.append(self.hyperparameters["scipy_kernel"](col.expand_dims(1)))
            matrix = np.stack(res, axis=0)
            assert matrix.shape == (X.shape[0], X.shape[1], X.shape[1])
            return matrix
        if self.mode_type == "pairwise":
            res = np.zeros((X.shape[0], X.shape[0], X.shape[1], X.shape[1]))
            for i, col1 in enumerate(X):
                for j, col2 in enumerate(X[: i + 1]):
                    data = np.stack([col1, col2], axis=1)
                    res[i, j, :, :] = self.hyperparameters["scipy_kernel"](data)
            return res


class PreComputedMode(ModeKernel):
    """
    A mode kernel that uses a precomputed kernel matrix.

    Parameters:
    - matrix (np.ndarray): The precomputed kernel matrix.
    - mode_type (str): The type of mode kernel. Can be "individual", "combinatorial", or "pairwise".
    - is_interpolatory (bool): Whether the mode kernel is interpolatory or not.
    name (str, optional): The name of the mode kernel. If not provided, defaults to "precomputed_kernel".
    """

    def __init__(self, matrix, mode_type, is_interpolatory, name=None) -> None:
        self.hyperparameters = {"matrix": matrix}
        self._is_interpolatory = is_interpolatory
        self.name = name if name is not None else "precomputed_kernel"
        self._mode_type = mode_type

    def __call__(self, X: np.array) -> np.array:
        if self.mode_type in ["individual", "combinatorial"]:
            assert self.hyperparameters["matrix"].shape == (
                X.shape[0],
                X.shape[1],
                X.shape[1],
            )
        if self.mode_type == "pairwise":
            assert self.hyperparameters["matrix"].shape == (
                X.shape[0],
                X.shape[0],
                X.shape[1],
                X.shape[1],
            )
        return self.hyperparameters["matrix"]
