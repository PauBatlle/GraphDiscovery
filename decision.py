import matplotlib.pyplot as plt
import numpy as onp
from plotting_help import plot_noise_evolution


class KernelChooser:
    def __init__(self, **kwargs):
        if "custom" in kwargs:
            self.choice_function = kwargs["custom"]
            return
        if "threshold" in kwargs:
            self.threshold = kwargs["threshold"]

            def choice(kernel_choice_dict):
                for kernel in kernel_choice_dict.keys():
                    if kernel_choice_dict[kernel]["noise"] < self.threshold:
                        return kernel
                return None

            self.choice_function = choice
            return
        manual = kwargs.get("manual", False)
        if manual:

            def choice(kernel_choice_dict):
                print("Choose kernel:")
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

            self.choice_function = choice
            return

        def choice(kernel_choice_dict):
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

        self.choice_function = choice

    def __call__(self, kernel_choice_dict):
        res = self.choice_function(kernel_choice_dict)
        assert (
            res is None or res in kernel_choice_dict.keys()
        ), f"invalid choice of kernel: {res}"
        return res


class ModeChooser:
    def __init__(self, **kwargs):
        def default_choice(list_of_modes, list_of_noises, list_of_Zs):
            increments = [
                list_of_noises[i + 1] - list_of_noises[i]
                for i in range(len(list_of_noises) - 1)
            ] + [1 - list_of_noises[-1]]
            if len(increments) == 0:
                return list_of_modes[0]
            argmax = max(range(len(increments)), key=lambda i: increments[i])
            return list_of_modes[argmax]

        if "custom" in kwargs:
            self.choice_function = kwargs["custom"]
            return
        if "threshold" in kwargs:
            self.threshold = kwargs["threshold"]

            def choice(list_of_modes, list_of_noises, list_of_Zs):
                for i, mode in enumerate(list_of_modes):
                    if list_of_noises[i] < self.threshold:
                        return mode
                raise Exception("Unexpectedly found no mode")

            self.choice_function = choice
            return

        manual = kwargs.get("manual", False)
        if manual:

            def choice(list_of_modes, list_of_noises, list_of_Zs):
                suggested_mode = default_choice(
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

            self.choice_function = choice
            return

        self.choice_function = default_choice

    def __call__(self, list_of_modes, list_of_noises, list_of_Zs):
        res = self.choice_function(list_of_modes, list_of_noises, list_of_Zs)
        assert (
            res in list_of_modes
        ), f"The result must be one of the modes in list_of_modes, got : {res}"
        return res


class EarlyStopping:
    def __init__(self, **kwargs):
        if "custom" in kwargs:
            self.choice_function = kwargs["custom"]
        self.choice_function = lambda list_of_modes, list_of_noises, list_of_Zs: False

    def __call__(self, list_of_modes, list_of_noises, list_of_Zs):
        res = self.choice_function(list_of_modes, list_of_noises, list_of_Zs)
        assert res in [True, False], f"The result must be a boolean, got : {res}"
        return res
