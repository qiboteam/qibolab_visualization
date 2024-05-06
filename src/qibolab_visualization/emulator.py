from collections import OrderedDict

import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from IPython.display import Latex, display

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
                f"${op_instruction[0]/np.sqrt(2*np.pi)}~{ghz_units}$",
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
    }
    for k in op_qid_list:
        latex_op_dict["b"].update({k: f"b_{k}"})
        latex_op_dict["bdag"].update({k: rf"b^{{\dagger}}_{k}"})
        latex_op_dict["O"].update({k: f"O_{k}"})
        latex_op_dict["X"].update({k: f"X_{k}"})
        latex_op_dict["Z01"].update({k: rf"\sigma^Z_{k}"})
        latex_op_dict["sp01"].update({k: rf"\sigma^+_{k}"})

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


def plot_fidelity_history(
    fidelity_history: list
    labels: list = None,
    time_in_dt: bool = False,
) -> Figure:
    """Plots the fidelity history.

    Args:
        #time_list (list): Specifies the list of times used in simulation.
        #fid_list_all (list): List of fidelity histories.
        
        fidelity_history (list): Output of QutipSimulator.fidelity_history
        labels (list): List of labels. Displayed in the plot legend. Defaults to None, in which case the 
        time_in_dt (bool): Specify the units of the x-axis in the plots to be in dt (inverse sampling rate) if True and in ns if False. Defaults to False.

    Returns:
        Figure: Figure of input fidelity histories with labels.
    """
    #fidelity_history = self.fidelity_history(sim_index, reference_states)
    time_list, fid_list_all = fidelity_history[0], fidelity_history[1]
    if labels is None:
        labels = fidelity_history[-1]
            
    fig = plt.figure()
    for result_ind in range(len(labels)):
        plt.plot(
            time_list,
            fid_list_all[result_ind],
            label=labels[result_ind],
        )
    plt.legend(loc="upper left")
    plt.ylabel("Overlap with basis state")
    if time_in_dt:
        plt.xlabel("Time / dt")
    else:
        plt.xlabel("Time / ns")
    for result_ind in range(len(labels)):
        print(
            labels[result_ind],
            fid_list_all[result_ind][0],
            fid_list_all[result_ind][-1],
        )

    return fig
