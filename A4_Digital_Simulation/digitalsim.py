"""Tiny combinational logic simulator producing WaveDrom JSON.

Usage:
  python digitalsim.py path/to/circuit.net [--out out.json]

Input format sections (fixed order): INPUTS, OUTPUTS, GATES, STIMULUS.
Gates: OUT = AND(A, B) | OR(A, B) | XOR(A, B) | NOT(A)

Note: this template file uses the `argparse` module to get arguments
from the command line.  You are expected to retain this part of it
to make testing easier.  The function calls given in the `main` function
are only suggestions, and you can rename them or create others as long
as the interface to the outside world does not change.

This may make it a bit harder to run purely from an editor like VSCode. 
However, in practice you almost never run code directly from an editor,
so this is something you need to be able to handle anyway.
"""


import sys
import json
import re
import argparse
from pathlib import Path
from typing import List, Tuple


# ======================================================================================
# CLASSES
# ======================================================================================
class Circuit:
    def __init__(self, AST: dict, SYMTAB: dict):
        """Syntax tree and the Symbols associated with circuit"""
        self.AST = AST
        self.SYMTAB = SYMTAB


class Gate:
    def __init__(
        self, name: str, operation: str, parents: List[str], children: List[str]
    ):
        """Stores context about a gate as well as its topological position"""
        self.name = name
        self.parents = parents
        self.children = children
        self.operation = operation
        if parents == None:
            self.level = 0
        else:
            self.level = max(parent.level for parent in parents) + 1

    def eval(self, val: dict) -> int:
        """Returns gate output based on values of parent gates
        Returns None if called in incorrect order

        Input -> Dict{symbol : value}
        Output -> Value"""
        value = None

        x = int(val[self.parents[0].name])
        if self.operation == "NOT":
            value = int(x < 1)
            return value

        y = int(val[self.parents[1].name])
        if self.operation == "AND":
            value = x & y
        elif self.operation == "XOR":
            value = int((x + y == True))
        elif self.operation == "OR":
            value = int((x + y > 0))

        return value


# ======================================================================================
# FUNCTIONS
# ======================================================================================
def parse_netlist(text: str) -> Tuple[Circuit, dict, int]:
    """Takes a given netlist and extracts symbol definitions & context while running syntax checks

    Input-> Netlist,
    Output->  Circuit_object, Stimulus, validity
    """
    Inputs, Outputs = [], []
    circuit = Circuit(SYMTAB={}, AST=None)
    Stimulus = {}
    valid_netlist = 1

    # Parsing Inputs and Outputs

    reg_filter = r"INPUTS: ([a-zA-z0-9 ]+)"
    input_result = re.search(reg_filter, text)
    reg_filter = r"OUTPUTS: ([a-zA-Z0-9]+)"
    output_result = re.search(reg_filter, text)
    reg_filter = r"([A-Za-z0-9_ ]+)= (AND|NOT|XOR|OR)\(([A-Za-z0-9]+)(?:, ([A-Za-z0-9]+))?(?:, ([A-Za-z0-9]+))?\)"
    gate_result = re.findall(reg_filter, text)

    if input_result == None or output_result == None or gate_result == None:
        valid_netlist = 0
        return circuit, valid_netlist

    Inputs = input_result[1].split()
    Outputs = output_result[1].split()
    for item in Inputs:
        circuit.SYMTAB[item] = {"Type": "Input"}

    # Parsing Gates || capture groups --> 0 is name, 1 is gate, 2+ are operands
    for gate in gate_result:
        if gate[1] == "NOT":
            if gate[3] != "":  # Only 1 input for NOT
                print(f"Invalid Gate {gate[0]}")
                valid_netlist = 0
            parents = [gate[2]]

        elif gate[4] != "":
            print(f"Invalid Gate {gate[0]} ")
            valid_netlist = 0
            parents = gate[2:]
        else:
            parents = gate[2:4]

        if gate not in Outputs:
            circuit.SYMTAB[gate[0].strip()] = {
                "Type": "Gate",
                "Op": gate[1],
                "Dep": parents,
            }
        else:
            circuit.SYMTAB[gate[0].strip()] = {
                "Type": "Output",
                "Op": gate[1],
                "Dep": parents,
            }

    # Parsing Stimuli || capture groups --> 0 is time stamp, 1 is boolean series
    reg_filter = r"([0-9]+)  ([0-9 ]+)"
    result = re.findall(reg_filter, text)
    for entry in result:
        if entry[0] in Stimulus:
            continue
        Stimulus[entry[0]] = entry[1]

    if valid_netlist == 0:
        print("Netlist fault")
        exit(1)

    return circuit, Stimulus


def AST_generate(circuit: Circuit) -> Tuple[Circuit, int]:
    """Generates a syntax tree based on the given circuit. Performs topology checks for logic validity

    Input -> Circuit object
    Output -> depth, Circuit object, validity"""

    valid = 1
    SYMTAB, AST = circuit.SYMTAB, {}
    waiting, evaluated = [], {}

    for symbol in SYMTAB:
        details = SYMTAB[symbol]
        if details["Type"] == "Input":
            AST[symbol] = Gate(symbol, "Input", parents=None, children=[])
            evaluated[symbol] = 1
            continue

        skip = 0
        parents = details["Dep"]
        for parent in parents:
            # Logic Error catching
            if parent not in SYMTAB:
                print(f"Unevaluable gate {symbol}")
                print("Logic fault")
                exit(1)

            if parent not in AST:
                waiting.append([symbol, parents])
                evaluated[symbol] = 0
                print(symbol, evaluated[symbol])
                break
            else:
                evaluated[symbol] = 1

        if evaluated[symbol] == 0:
            continue

        # Adding to the tree
        AST[symbol] = Gate(
            symbol,
            details["Op"],
            parents=[AST[parent] for parent in parents],
            children=[],
        )
        for parent in parents:
            AST[parent].children.append(symbol)

    miss_count = 0  # To prevent infinite checking
    while len(waiting) > 0 and miss_count < 200:
        miss_count += 1

        for item in waiting:
            parents = item[1]
            for parent in parents:
                if evaluated[parent] == 0:
                    skip = 1
                    break
                else:
                    skip = 0

            if skip == 1:
                continue

            # Adding to the tree
            AST[symbol] = Gate(
                name=symbol, operation=details["Op"], parents=parents, children=[]
            )
            for parent in parents:
                AST[parent].children.append(symbol)

            waiting.remove(item)

    if miss_count == 200:
        print("Loop detected")

    circuit.AST = AST

    return circuit


def simulate(circuit: Circuit, Stimuli: dict) -> dict:
    """Generates time series values for each gate based on given Stimulus

    Input -> circuit, dict( time : values )
    Output -> dict( symbol : values )"""

    SYMTAB, AST = circuit.SYMTAB, circuit.AST
    Values = {gate: "" for gate in SYMTAB}
    eval_order = []

    # Generating evaluation order (can be improved later)
    depth = max(AST[symbol].level for symbol in AST)
    for level in range(1, depth + 1):
        for symbol in AST:
            if AST[symbol].level == level:
                eval_order.append(symbol)

    Inputs = [gate for gate in SYMTAB if SYMTAB[gate]["Type"] == "Input"]
    # Evaluation loop
    for time in Stimuli:
        time_values = Stimuli[time].split()
        current_values = dict(zip(Inputs, time_values))
        for gate in eval_order:
            gate_object = AST[gate]

            current_values[gate] = gate_object.eval(current_values)

        for gate in current_values:
            Values[gate] += str(current_values[gate])

    return Values


def to_wavedrom_json(Values, path) -> dict:
    """Converts time series to a WaveDROM format"""
    data = []
    for symbol in Values:
        symbol_data = {}
        symbol_data["name"], symbol_data["wave"] = symbol, Values[symbol]
        data.append(symbol_data)

    output = {"signal": data}
    with open(path, "w") as log:
        json.dump(output, log, indent=4)
    return output


# ======================================================================================
# EXECUTION
# ======================================================================================
def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("netlist", help=".net file path")
    ap.add_argument("--out", "-o", help="output JSON path")
    args = ap.parse_args(argv)

    text = Path(args.netlist).read_text()

    circuit, stimulus = parse_netlist(text)

    circuit = AST_generate(circuit)

    waves = simulate(circuit, stimulus)

    out_path = args.out
    if not out_path:
        p = Path(args.netlist)
        out_path = str(p.with_suffix(".json"))
    print(out_path)

    js = to_wavedrom_json(waves, path=out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


