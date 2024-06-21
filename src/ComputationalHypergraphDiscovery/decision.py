from typing import Any
import matplotlib.pyplot as plt
from .util import plot_noise_evolution


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

    def __call__(self, kernel_choice_dict):
        """
        Chooses the kernel with least noise and such that noise is lower than random noise Z
        """
        current = None
        noise = 2
        for kernel, vals in kernel_choice_dict.items():
            if vals["noise"] < noise and vals["noise"] < vals["Z"][0]:
                current = kernel
                noise = vals["noise"]
        return current


class ThresholdKernelChooser(KernelChooser):
    """
    A kernel chooser that selects the kernel with the lowest noise level below a given threshold.

    Args:
        threshold (float): The maximum noise level allowed for a kernel to be selected.
    """

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, kernel_choice_dict):
        for kernel in kernel_choice_dict.keys():
            if kernel_choice_dict[kernel]["noise"] < self.threshold:
                return kernel
        return None


class CustomKernelChooser(KernelChooser):
    """
    A custom kernel chooser that allows for a user-defined function to choose a kernel.

    Args:
    - chooser_function (callable): A function that takes a dictionary of kernel choices and returns
        the chosen kernel. The function must return None or a key of the input dictionary.


    Once the custom kernel chooser is initialized, it can be called with a dictionary of kernel choices.
    The custom function is then called with the input dictionary as argument, and the chosen kernel is returned.

    """

    def __init__(self, chooser_function):
        self.choice_function = chooser_function

    def __call__(self, kernel_choice_dict):
        """
        Given a dictionary of kernel choices, returns the kernel chosen by the custom function.

        Args:
        - kernel_choice_dict (dict): A dictionary where keys are kernel names and values are scores.

        Returns:
        - str: The name of the chosen kernel.

        Raises:
        - AssertionError: If the chosen kernel is not in the dictionary.
        """
        res = self.choice_function(kernel_choice_dict)
        assert (
            res is None or res in kernel_choice_dict.keys()
        ), f"invalid choice of kernel from custom function: {res}"
        return res


class ManualKernelChooser(KernelChooser):
    """
    A class used to choose a kernel manually. Input "STOP" to raise an error and exit loop

    ...

    Methods
    -------
    __call__(kernel_choice_dict):
        Allows the user to choose a kernel from a dictionary of available kernels.

    """

    def __init__(self):
        pass

    def __call__(self, kernel_choice_dict):
        """Prompt the user to choose a kernel from a dictionary of kernel choices.

        Args:
        - kernel_choice_dict (dict): A dictionary of kernel choices, where each key is a kernel name and each value is a dictionary containing the kernel's noise and Z values.

        Returns:
        - str: The name of the chosen kernel.

        Raises:
        - Exception: If the user enters "STOP".
        - AssertionError: If the user enters an invalid kernel choice.
        """

    def __call__(self, kernel_choice_dict):
        while True:
            input_string = ""
            for kernel in kernel_choice_dict.keys():
                input_string += f"{kernel} kernel: noise:{kernel_choice_dict[kernel]['noise']:.2f},Z:{kernel_choice_dict[kernel]['Z'][0]:.2f}\n"
            choice = input(input_string)
            try:
                if choice == "None":
                    return None
                assert (
                    choice in kernel_choice_dict.keys()
                ), f"invalid choice of kernel: {choice}. Your choice must be either 'None' or in {kernel_choice_dict.keys()} \nWrite 'STOP' to stop the program"
                return choice
            except AssertionError as e:
                if choice == "STOP":
                    raise Exception("User stopped the program")
                print(e)


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

    def is_a_modeChooser(self):
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
        increments = [
            list_of_noises[i + 1] - list_of_noises[i]
            for i in range(len(list_of_noises) - 1)
        ] + [1 - list_of_noises[-1]]
        if len(increments) == 0:
            return list_of_modes[0]
        argmax = max(range(len(increments)), key=lambda i: increments[i])
        return list_of_modes[argmax]


class CustomModeChooser(ModeChooser):
    """
    A custom mode chooser that selects a mode from a list of modes based on a given function.

    Args:
        chooser_function (callable): A function that takes in a list of modes, a list of noises, and a list of Zs, and returns a selected mode.

    """

    def __init__(self, chooser_function):
        self.choice_function = chooser_function

    def __call__(self, list_of_modes, list_of_noises, list_of_Zs):
        """
        Given a list of modes, a list of noises, and a list of Zs, returns the mode selected by the choice function.

        Args:
        - list_of_modes (list): A list of modes.
        - list_of_noises (list): A list of noises.
        - list_of_Zs (list): A list of Zs.

        Returns:
        - The mode selected by the choice function.

        Raises:
        - AssertionError: If the result is not one of the modes in list_of_modes.
        """
        res = self.choice_function(list_of_modes, list_of_noises, list_of_Zs)
        assert (
            res in list_of_modes
        ), f"The result must be one of the modes in list_of_modes, got : {res}"
        return res


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
        for i, mode in enumerate(list_of_modes):
            if list_of_noises[i] < self.threshold:
                return mode
        raise Exception("Unexpectedly found no mode")


class ManualModeChooser(ModeChooser):
    """
    A class that allows the user to manually choose the number of ancestors for a given mode.
    """

    def __init__(self):
        pass

    def __call__(self, list_of_modes, list_of_noises, list_of_Zs):
        """
        This method takes a list of modes, a list of corresponding noise levels, and a list of corresponding Z values,
        and returns a chosen mode based on user input.

        Args:
        - list_of_modes (list): A list of modes.
        - list_of_noises (list): A list of noises.
        - list_of_Zs (list): A list of Zs.

        Returns:
        - list_of_modes[chosen_index] (object): The chosen mode.
        """
        suggested_mode = MaxIncrementModeChooser()(
            list_of_modes, list_of_noises, list_of_Zs
        )
        while True:
            ancestor_number = [mode.node_number for mode in list_of_modes]
            fig, axes = plot_noise_evolution(
                ancestor_number, list_of_noises, list_of_Zs, suggested_mode
            )
            plt.show(block=False)

            choice = input(
                f"Choose number of ancestors. Suggested ={suggested_mode.node_number} "
            )
            try:
                chosen_index = ancestor_number.index(int(choice))
                fig, axes = plot_noise_evolution(
                    ancestor_number,
                    list_of_noises,
                    list_of_Zs,
                    list_of_modes[chosen_index],
                )
                plt.show(block=False)
                assert "Y" == input(
                    "Confirm choice by pressing Y, otherwise press any other key"
                )
                return list_of_modes[chosen_index]
            except ValueError as e:
                print(
                    f"invalid choice of ancestor number: {choice}. Your choice must be in {ancestor_number} \nWrite 'STOP' to stop the program"
                )
            except AssertionError as e:
                print("user did not confirm choice, try again")
            except Exception as e:
                if choice == "STOP":
                    raise Exception("User stopped the program")
                print(e)


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
