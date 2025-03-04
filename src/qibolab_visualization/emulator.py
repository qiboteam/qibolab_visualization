from collections import OrderedDict
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from IPython.display import Latex, display

from qibolab.instruments.emulator.engines.generic import op_from_instruction

def print_hamiltonian(model_config, op_qid_list: list = None):
    """Prints Hamiltonian the model configuration.

    Args:
        model_config (dict): Model configuration dictionary.
        op_qid_list (list, optional): List of qubit/coupler IDs present in the model. If None, extracts this information from model_config. Defaults to None.
    """

    def format_instruction(op_instruction, is_dissipation=False):
        """Reformats op_instructions."""
        if is_dissipation:
            ghz_units = "\\sqrt{{ \\text{GHz} }}"
            return [
                f"${op_instruction[1]}$",
                f"${op_instruction[0]}~{ghz_units}$",
            ]
        else:
            ghz_units = "\\text{GHz}"
            return [
                f"${op_instruction[1]}$",
                f"${op_instruction[0]/2/np.pi}~{ghz_units}$",
            ]

    if op_qid_list is None:
        qubits_list = model_config["qubits_list"]
        couplers_list = model_config["couplers_list"]
        op_qid_list = qubits_list + couplers_list

    latex_op_dict = {
        "b": {},
        "bdag": {},
        "O": {},
        "X": {},
        "Z01": {},
        "sp01": {},
        "sig01": {},
        "sig10": {},
        "sig12": {},
        "sig21": {},
    }
    for k in op_qid_list:
        latex_op_dict["b"].update({k: f"b_{k}"})
        latex_op_dict["bdag"].update({k: rf"b^{{\dagger}}_{k}"})
        latex_op_dict["O"].update({k: f"O_{k}"})
        latex_op_dict["X"].update({k: f"X_{k}"})
        latex_op_dict["Z01"].update({k: rf"\sigma^Z_{k}"})
        latex_op_dict["sp01"].update({k: rf"\sigma^+_{k}"})
        latex_op_dict["sig01"].update({k: rf"\sigma^+_{k}"})
        latex_op_dict["sig10"].update({k: rf"\sigma^-_{k}"})
        latex_op_dict["sig12"].update({k: rf"\Sigma^+_{k}"})
        latex_op_dict["sig21"].update({k: rf"\Sigma^-_{k}"})
        

    latex_op_connectors_dict = OrderedDict(
        [
            ("^", lambda a, b: rf"{a}{{\otimes}}{b}"),
            ("*", lambda a, b: f"{a}{b}"),
            ("+", lambda a, b: f"{a}+{b}"),
            ("-", lambda a, b: f"{a}-{b}"),
        ]
    )

    basic_dict = [
        rf"$O_i = b^{{\dagger}}_i b_i$",
        rf"$X_i = b^{{\dagger}}_i + b_i$",
        rf"$\sigma^+_i = \ket{1}\bra{0}, \quad \sigma^-_i = \ket{0}\bra{1}$",
        rf"$\Sigma^+_i = \ket{2}\bra{1}, \quad \Sigma^-_i = \ket{1}\bra{2}$",
    ]
    print("Dictionary")
    for i in basic_dict:
        display(Latex(i))
    print("\n")
    print("-" * 21)

    print("One-body drift terms:")
    print("-" * 21)
    for op_instruction in model_config["drift"]["one_body"]:
        op_instruction = op_from_instruction(
            op_instruction,
            latex_op_dict,
            latex_op_connectors_dict,
            multiply_coeff=False,
        )
        inst = format_instruction(op_instruction)
        display(Latex(inst[0]))
        display(Latex(inst[1]))
    print("-" * 21)

    print("Two-body drift terms:")
    print("-" * 21)
    if len(model_config["drift"]["two_body"]) == 0:
        print("None")
    for op_instruction in model_config["drift"]["two_body"]:
        op_instruction = op_from_instruction(
            op_instruction,
            latex_op_dict,
            latex_op_connectors_dict,
            multiply_coeff=False,
        )
        inst = format_instruction(op_instruction)
        display(Latex(inst[0]))
        display(Latex(inst[1]))
    print("-" * 21)

    print("One-body drive terms:")
    print("-" * 21)
    for drive_instructions in model_config["drive"].values():
        for op_instruction in drive_instructions:
            op_instruction = op_from_instruction(
                op_instruction,
                latex_op_dict,
                latex_op_connectors_dict,
                multiply_coeff=False,
            )
            inst = format_instruction(op_instruction)
            display(Latex(inst[0]))
            display(Latex(inst[1]))
    print("-" * 21)

    try:
        print("Two-body drive terms:")
        print("-" * 21)
        for drive_instructions in model_config["two_body_drive"].values():
            for op_instruction in drive_instructions:
                op_instruction = op_from_instruction(
                    op_instruction,
                    latex_op_dict,
                    latex_op_connectors_dict,
                    multiply_coeff=False,
                )
                inst = format_instruction(op_instruction)
                display(Latex(inst[0]))
                display(Latex(inst[1]))
        print("-" * 21)
    except:
        pass

    print("Dissipative terms:")
    print("-" * 21)
    for key in model_config["dissipation"].keys():
        print(">>", key, "Linblad operators:")
        if len(model_config["dissipation"][key]) == 0:
            print("None")
        for op_instruction in model_config["dissipation"][key]:
            op_instruction = op_from_instruction(
                op_instruction,
                latex_op_dict,
                latex_op_connectors_dict,
                multiply_coeff=False,
            )
            inst = format_instruction(op_instruction, is_dissipation=True)
            display(Latex(inst[0]))
            display(Latex(inst[1]))
    print("-" * 21)


def plot_overlaps(
    all_overlaps: dict,
    time_list: Optional[list] = None,
    time_label: Optional[str] = None,
    figsize: Optional[list] = None,
) -> Figure:
    """Plots the overlaps obtained.

    Args:
        all_overlaps (dict): Output of QutipSimulator.compute_overlaps.
        time_list (optional, list): List of simulation times (or x-axis parameter). Defaults to None.
        time_label (optional, str): Specifies the x-axis label in the plot. Defaults to None, in which case 'Time / simulation dt' will be used.
        figsize (optional, list): Width, height of figure in inches; used as optional input to `matplotlib.pyplot.figure` method.

    Returns:
        Figure: Figure of input overlap histories with labels.
    """     
    fig = plt.figure(figsize=figsize)
    for label, overlaps in all_overlaps.items():
        if time_list is None:
            plt.plot(overlaps, label=label)
        else:
            plt.plot(time_list, overlaps, label=label)

    plt.legend(loc="upper left")
    plt.ylabel("State overlap")
    if time_label is None:
        time_label = 'Time / simulation dt'
    plt.xlabel(time_label)
    print("Overlap of final state with basis states:")
    for label, overlaps in all_overlaps.items():
        print(
            label,
            overlaps[-1],
        )

    return fig
