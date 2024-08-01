from typing import Any
import matplotlib.pyplot as plt
from .util import plot_noise_evolution
import jax.numpy as np
import jax


class KernelChooser:
    """
    An interface that chooses a kernel for a given dataset.
    Kernel choosers implement the practionner's logic for chosing the kernel.
    Their __call__ method takes as input a dictionary of kernels and their associated noise and Z values, and returns the chosen kernel.

    There are different choices of kernel choosers:
    - MinNoiseKernelChooser: chooses the kernel with least noise and such that noise is lower than random noise Z
    - ThresholdKernelChooser: chooses the first kernel with noise lower than a given threshold
    - CustomKernelChooser: chooses the kernel according to a custom function
    - ManualKernelChooser: asks the user to choose the kernel, by displaying the noise and Z values of each kernel


    Methods:
    --------
    is_a_kernelChooser():
        Returns True if the instance is a KernelChooser object.
    """

    def __init__(self):
        pass

    def is_a_kernel_chooser(self):
        return True


class MinNoiseKernelChooser(KernelChooser):
    """
    A class that chooses the kernel with the least noise and such that noise is lower than random noise Z.
    """

    def __init__(self):
        pass

    def __call__(self, kernel_perfs):
        """
        Chooses the kernel with least noise and such that noise is lower than random noise Z
        """
        ybs, noises, Z_lows, Z_highs, gammas, keys = kernel_perfs

        # Create a mask for conditions where noise is less than Z_low
        valid_indices = noises < Z_lows
        is_valid = np.any(valid_indices)

        # Use jnp.where to find the first valid index, default to -1 if no valid index
        selected_index = np.argmin(np.where(valid_indices, noises, 2))

        return (
            selected_index * is_valid - 1 * (1 - is_valid),
            ybs[selected_index],
            noises[selected_index],
            Z_lows[selected_index],
            Z_highs[selected_index],
            gammas[selected_index],
            keys[selected_index],
        )


class ThresholdKernelChooser(KernelChooser):
    """
    A kernel chooser that selects the kernel with the lowest noise level below a given threshold.

    Args:
        threshold (float): The maximum noise level allowed for a kernel to be selected.
    """

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, kernel_perfs):

        ybs, noises, Z_lows, Z_highs, gammas, keys = kernel_perfs

        # Create a mask for noises less than the threshold
        valid_indices = noises < self.threshold

        # Indices where valid, else set to an out-of-range index
        indices = np.where(valid_indices, np.arange(noises.shape[0]), noises.shape[0])

        # Find the minimum index that is valid
        selected_index = np.min(indices)

        is_valid = selected_index < noises.shape[0]

        return (
            selected_index * is_valid - 1 * (1 - is_valid),
            ybs[selected_index],
            noises[selected_index],
            Z_lows[selected_index],
            Z_highs[selected_index],
            gammas[selected_index],
            keys[selected_index],
        )


class ModeChooser:
    """
    A class that represents a mode chooser. Mode choosers implement the practionner's logic for chosing the mode.
    Their __call__ method takes as input a list of modes and their associated noise and Z values, and returns the chosen mode.

    There are different choices of mode choosers:
    - MaxIncrementModeChooser: chooses the mode with the highest increment in noise
    - ThresholdModeChooser: chooses the first mode with noise lower than a given threshold
    - CustomModeChooser: chooses the mode according to a custom function
    - ManualModeChooser: asks the user to choose the mode, by displaying the noise and Z values of each mode


    Methods:
    --------
    is_a_modeChooser():
        Returns True.
    """

    def __init__(self):
        pass

    def is_a_mode_chooser(self):
        return True

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        Given a list of modes, a list of noises, and a list of Zs, returns the mode selected by the choice function.

        Args:
        - list_of_modes (list): A list of modes.
        - list_of_noises (list): A list of noises.
        - list_of_Zs (list): A list of Zs.

        Returns:
        - The mode selected by the choice function.
        """
        Exception("Not implemented")


class MaxIncrementModeChooser(ModeChooser):
    """
    A mode chooser that selects the mode with the maximum increment in noise level.
    """

    def __init__(self):
        pass

    def __call__(self, list_of_modes, list_of_noises, list_of_Zs):
        increments = list_of_noises[1:] - list_of_noises[:-1]

        # Append the increment from the last noise to 1 (assuming 1 is a boundary or a limit)
        increments = np.append(increments, 1 - list_of_noises[-1])
        return np.argmax(increments)


class ThresholdModeChooser(ModeChooser):
    """
    A mode chooser that selects the first mode whose noise is below a given threshold.

    Args:
    threshold (float): The maximum noise level allowed for a mode to be selected.
    """

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, list_of_modes, list_of_noises, list_of_Zs):
        """
        Given a list of modes, a list of corresponding noise levels, and a list of corresponding Z values,
        returns the mode with the lowest noise level that is below the threshold. If no such mode is found,
        raises an exception, as if it is the case, the program should not have been run.

        Args:
        - list_of_modes (list): A list of modes.
        - list_of_noises (list): A list of noise levels corresponding to the modes.
        - list_of_Zs (list): A list of Z values corresponding to the modes.

        Returns:
        - The mode with the lowest noise level that is below the threshold.

        Raises:
        - Exception: If no mode is found with a noise level below the threshold.
        """
        is_under_threshold = list_of_noises < self.threshold
        valid_indices = np.where(
            is_under_threshold, np.arange(list_of_noises.shape[0]), -1
        )
        return jax.lax.select(
            np.any(is_under_threshold), np.max(valid_indices), np.argmin(list_of_noises)
        )


class EarlyStopping:
    """
    A class used to stop the training of a model early to avoid useless computations.
    """

    def __init__(self):
        pass

    def is_an_earlyStopping(self):
        return True


class NoEarlyStopping(EarlyStopping):
    """Class that implements no early stopping during training.

    This class'__call__ method always return False, indicating that no early stopping should be performed.

    """

    def __init__(self):
        pass

    def __call__(self, list_of_modes, list_of_noises, list_of_Zs):
        return False


class CustomEarlyStopping(EarlyStopping):
    """
    Custom implementation of EarlyStopping that allows for a custom choice function.

    Args:
    - custom_function (callable): A function that takes in three arguments: a list of modes, a list of noises, and a list of Zs.
        The function should return a boolean value indicating whether to stop early or not.
    """

    def __init__(self, custom_function):
        self.choice_function = custom_function

    def __call__(self, list_of_modes, list_of_noises, list_of_Zs):
        res = self.choice_function(list_of_modes, list_of_noises, list_of_Zs)
        assert res in [True, False], f"The result must be a boolean, got : {res}"
        return res
